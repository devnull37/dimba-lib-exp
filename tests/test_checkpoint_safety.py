"""Focused checks for resumable and atomic training checkpoints."""

from __future__ import annotations

import random
import sys
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dimba.distillation.trainer import DistillationTrainer
from dimba.training.grpo import GRPOConfig, GRPOTrainer
from dimba.utils.checkpointing import (
    ProgressiveCheckpointManager,
    atomic_torch_save,
    capture_rng_state,
    restore_rng_state,
)


class _TinyModel(nn.Module):
    def __init__(self, value: float = 1.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(value))
        self.denoiser = nn.Module()
        self.denoiser.blocks = nn.ModuleList()
        self.config = {}


class _PortableCheckpointModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        block = nn.Module()
        block.mamba_fwd = nn.Linear(1, 1, bias=False)
        self.denoiser = nn.Module()
        self.denoiser.blocks = nn.ModuleList([block])
        self.config = {"vocab_size": 7}


def _tiny_trainer(model: _TinyModel) -> DistillationTrainer:
    trainer = DistillationTrainer.__new__(DistillationTrainer)
    trainer.model = model
    trainer.teacher = None
    trainer.config = SimpleNamespace(
        teacher_type="causal", kd_weight=0.0, kd_temp=2.0, share_vocab=False
    )
    trainer.head_aligners = nn.ModuleList()
    trainer.projectors = nn.ModuleList()
    trainer._log_hook = None

    def step(self, input_ids, **_kwargs):
        return (self.model.weight * input_ids.float().mean() * torch.rand(())).square()

    trainer._stage3_step = MethodType(step, trainer)
    return trainer


def test_atomic_save_and_rng_roundtrip(tmp_path: Path) -> None:
    checkpoint = tmp_path / "state.pt"
    atomic_torch_save({"value": torch.tensor(1)}, checkpoint)
    atomic_torch_save({"value": torch.tensor(2)}, checkpoint)
    assert torch.load(checkpoint)["value"].item() == 2
    assert not list(tmp_path.glob("*.tmp"))

    torch.manual_seed(123)
    random.seed(123)
    state = capture_rng_state()
    expected = (torch.rand(3), random.random())
    torch.rand(10)
    random.random()
    restore_rng_state(state)
    actual = (torch.rand(3), random.random())
    assert torch.equal(actual[0], expected[0])
    assert actual[1] == expected[1]


def test_grpo_checkpoint_is_atomic_portable_and_strict(tmp_path: Path) -> None:
    model = _PortableCheckpointModel()
    trainer = GRPOTrainer.__new__(GRPOTrainer)
    trainer.model = model
    trainer.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda _: 1.0)
    trainer.step = 4
    trainer.cfg = GRPOConfig(save_dir=str(tmp_path))

    path = Path(trainer.save_checkpoint())
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert checkpoint["config"] == {"vocab_size": 7}
    assert "force_torch_mixer" not in checkpoint["config"]
    assert checkpoint["model_backend"] == "Linear"
    assert not list(tmp_path.glob("*.tmp"))

    expected = model.weight.detach().clone()
    with torch.no_grad():
        model.weight.zero_()
    trainer.load_checkpoint(str(path), weights_only=True)
    assert torch.equal(model.weight, expected)

    checkpoint["model_state_dict"]["unexpected.weight"] = torch.ones(1)
    broken = tmp_path / "broken.pt"
    atomic_torch_save(checkpoint, broken)
    with pytest.raises(RuntimeError, match="Unexpected key"):
        trainer.load_checkpoint(str(broken), weights_only=True)


def test_mps_rng_roundtrip_and_unavailable_restore_fails_closed(monkeypatch) -> None:
    saved_mps = torch.tensor([1, 2, 3], dtype=torch.uint8)
    restored = []
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.mps, "get_rng_state", lambda: saved_mps)
    monkeypatch.setattr(torch.mps, "set_rng_state", restored.append)

    state = capture_rng_state()
    assert torch.equal(state["torch_mps"], saved_mps)
    restore_rng_state(state)
    assert restored == [saved_mps]

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="MPS RNG state.*unavailable"):
        restore_rng_state(state)


def test_progressive_checkpoint_restores_model_and_optimizer(tmp_path: Path) -> None:
    model = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    manager = ProgressiveCheckpointManager([1], str(tmp_path))
    path = manager.save_checkpoint(model, optimizer, 3, 1)

    restored = nn.Linear(2, 2)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    metadata = manager.load_checkpoint(path, restored, restored_optimizer)
    assert metadata["global_step"] == 3
    for expected, actual in zip(model.parameters(), restored.parameters()):
        assert torch.equal(expected, actual)
    assert restored_optimizer.state_dict()["state"]


def test_stage_resume_restores_optimizer_rng_and_step() -> None:
    stage = {"name": "stage3", "id": "stage3a", "steps": 2, "lr": 1e-2}
    batches = [torch.tensor([[1]]), torch.tensor([[2]])]

    torch.manual_seed(7)
    full_model = _TinyModel()
    full = _tiny_trainer(full_model)
    full.run_stage(stage, batches)

    torch.manual_seed(7)
    first_model = _TinyModel()
    first = _tiny_trainer(first_model)
    first.run_stage({**stage, "steps": 1}, batches[:1])
    resume_state = {
        "stage_id": "stage3a",
        "stage_step": 1,
        "optimizer_name": "adamw",
        "optimizer_state_dict": first.optimizer.state_dict(),
        "rng_state": capture_rng_state(),
    }

    resumed_model = _TinyModel(first_model.weight.item())
    resumed = _tiny_trainer(resumed_model)
    resumed.run_stage(stage, batches[1:], resume_state=resume_state)

    assert resumed.stage_step == 2
    assert resumed.stage_completed
    assert torch.equal(resumed_model.weight, full_model.weight)


def test_nonfinite_loss_and_empty_data_stop_the_stage() -> None:
    stage = {"name": "stage3", "id": "stage3a", "steps": 1, "lr": 1e-2}
    trainer = _tiny_trainer(_TinyModel())

    def nonfinite(self, input_ids, **_kwargs):
        return self.model.weight * torch.tensor(float("nan"))

    trainer._stage3_step = MethodType(nonfinite, trainer)
    with pytest.raises(FloatingPointError, match="non-finite"):
        trainer.run_stage(stage, [torch.tensor([[1]])])

    with pytest.raises(RuntimeError, match="dataloader is empty"):
        _tiny_trainer(_TinyModel()).run_stage(stage, [])


def test_streaming_resume_replays_only_undelivered_batches(monkeypatch) -> None:
    import scripts.train_4090 as train

    class Rows(list):
        def shard(self, num_shards, index):
            return Rows(self[index::num_shards])

    rows = Rows({"text": str(i)} for i in range(20))
    fake_datasets = SimpleNamespace(load_dataset=lambda *_args, **_kwargs: rows)
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    tokenizer = SimpleNamespace(
        eos_token_id=None,
        encode=lambda text, add_special_tokens=False: [int(text)],
    )

    dataset = train._FineWebStreaming(
        tokenizer, 2, batch_size=2, resume_batches=3, revision="fixed"
    )
    monkeypatch.setattr(
        torch.utils.data,
        "get_worker_info",
        lambda: SimpleNamespace(num_workers=2, id=0),
    )
    iterator = iter(dataset)
    assert [next(iterator).tolist(), next(iterator).tolist()] == [[5, 0], [7, 0]]
    monkeypatch.setattr(
        torch.utils.data,
        "get_worker_info",
        lambda: SimpleNamespace(num_workers=2, id=1),
    )
    iterator = iter(dataset)
    assert [next(iterator).tolist(), next(iterator).tolist()] == [[8, 0], [10, 0]]


def test_h100_preflight_fails_without_quality_gate_or_cuda(monkeypatch) -> None:
    import scripts.train_h100 as h100

    monkeypatch.setattr(sys, "argv", ["train_h100.py", "--phase", "all"])
    with pytest.raises(RuntimeError, match="phase all"):
        h100.main()

    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(sys, "argv", ["train_h100.py", "--phase", "distill"])
    with pytest.raises(RuntimeError, match="requires CUDA"):
        h100.main()

    monkeypatch.setattr(
        sys,
        "argv",
        ["train_h100.py", "--phase", "sft", "--quality-gate-passed"],
    )
    with pytest.raises(RuntimeError, match="supports torchrun only"):
        h100.main()

    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(sys, "argv", ["train_h100.py", "--phase", "distill"])
    with pytest.raises(RuntimeError, match="requires CUDA"):
        h100.main()


def test_stage3_world_scaling_preserves_fixed_token_recipe() -> None:
    import scripts.train_4090 as train

    phases = [{"steps": 9}, {"steps": 1}]
    train._scale_stage3_steps_for_world(phases, 2)
    assert [phase["steps"] for phase in phases] == [5, 1]
    with pytest.raises(ValueError, match="world_size"):
        train._scale_stage3_steps_for_world(phases, 0)


@pytest.mark.parametrize("phase", ["sft", "grpo"])
def test_posttraining_rejects_torch_mamba2_when_fast_backend_required(
    monkeypatch, tmp_path: Path, phase: str
) -> None:
    import scripts.train_4090 as train

    class TorchMamba2:
        pass

    model = SimpleNamespace(
        denoiser=SimpleNamespace(
            blocks=[SimpleNamespace(mamba_fwd=TorchMamba2())],
            use_gradient_checkpointing=False,
        ),
        token_embed=SimpleNamespace(),
        output_head=SimpleNamespace(),
    )
    tokenizer = SimpleNamespace(
        get_vocab=lambda: {"<think>": 1, "</think>": 2},
        convert_tokens_to_ids=lambda token: {"<think>": 1, "</think>": 2}[token],
    )
    monkeypatch.setattr(train, "REQUIRE_FAST_MAMBA2", True)
    monkeypatch.setattr(train, "REQUIRE_QUALITY_GATE", False)
    monkeypatch.setattr(train, "_build_or_load_model", lambda *_args, **_kwargs: model)
    monkeypatch.setattr(train, "_load_tokenizer", lambda *_args, **_kwargs: tokenizer)

    run = train.run_sft if phase == "sft" else train.run_grpo
    with pytest.raises(RuntimeError, match=f"CUDA {phase.upper()}.*TorchMamba2 fallback"):
        run("checkpoint.pt", torch.device("cpu"), save_dir=str(tmp_path / phase))


def test_stage3_batch_downshift_preserves_token_budget(monkeypatch) -> None:
    import scripts.train_4090 as train

    observed = {}

    class Trainer:
        stage_step = 0

        def run_stage(self, stage, _loader, resume_state=None):
            observed.update(stage)
            self.stage_step = stage["steps"]
            self.stage_completed = True

    monkeypatch.setattr(train, "_STAGE3_BATCH", 8)
    monkeypatch.setattr(train, "_make_fineweb_streaming_loader", lambda *_a, **_k: [])
    result = train._run_stage3_resilient(
        Trainer(),
        {"name": "stage3", "steps": 10, "lr": 1e-3},
        tokenizer=None,
        seq_len=4,
        batch=4,
        device=torch.device("cpu"),
        revision="fixed",
    )
    assert result == 4
    assert observed["steps"] == 20


def test_stage3_next_phase_uses_cumulative_stream_cursor(monkeypatch) -> None:
    import scripts.train_4090 as train

    loader_cursors = []

    class Trainer:
        stage_step = 0

        def run_stage(self, stage, _loader, resume_state=None):
            self.stage_step = stage["steps"]
            self.stage_total_steps = stage["steps"]
            self.stage_completed = True

    monkeypatch.setattr(train, "_STAGE3_BATCH", 2)

    def loader(*_args, **kwargs):
        loader_cursors.append(kwargs["resume_batches"])
        return []

    monkeypatch.setattr(train, "_make_fineweb_streaming_loader", loader)
    trainer = Trainer()
    first_context = {}
    train._run_stage3_resilient(
        trainer,
        {"name": "stage3", "steps": 3, "lr": 1e-3},
        None,
        4,
        2,
        torch.device("cpu"),
        revision="fixed",
        stream_start_batch=0,
        checkpoint_context=first_context,
    )
    second_context = {}
    train._run_stage3_resilient(
        trainer,
        {"name": "stage3", "steps": 2, "lr": 1e-4},
        None,
        4,
        2,
        torch.device("cpu"),
        revision="fixed",
        stream_start_batch=3,
        checkpoint_context=second_context,
    )
    assert loader_cursors == [0, 3]
    assert second_context["stream_start_batch"] == 3


def test_stage3_resume_rejects_every_trajectory_change() -> None:
    import scripts.train_4090 as train

    saved = {
        "name": "stage3",
        "id": "stage3a",
        "steps": 10,
        "lr": 1e-4,
        "freeze_ffn": True,
        "kd_weight": 1.0,
        "kd_temp": 2.0,
        "ce_loss_weight": 1.0,
        "min_snr_gamma": 5.0,
        "snr_floor": 0.5,
        "ce_time_fade": True,
        "optimizer": "muon",
        "weight_decay": 0.0,
    }
    train._validate_stage3_resume_config(saved, saved, train._STAGE3_BATCH)
    train._validate_stage3_resume_config(
        {**saved, "steps": 20}, saved, train._STAGE3_BATCH // 2
    )
    for field in saved:
        replacement = saved[field] + 1 if field == "steps" else object()
        changed = {**saved, field: replacement}
        with pytest.raises(ValueError, match="trajectory fields"):
            train._validate_stage3_resume_config(saved, changed, train._STAGE3_BATCH)


def test_stage3_recipe_keeps_teacher_through_both_phases(capsys) -> None:
    import scripts.train_4090 as train
    import scripts.train_h100 as h100

    local_phases = train._coadapt_stages()
    assert [phase["kd_weight"] for phase in local_phases] == [1.0, 0.3]
    assert all(phase["kd_weight"] > 0.0 for phase in local_phases)
    assert all(phase["kd_time_fade"] is True for phase in local_phases)

    h100_phases = h100._build_stage3_phases(1, 1, 2e-4, 3e-5)
    assert [phase["kd_weight"] for phase in h100_phases] == [1.0, 0.3]
    assert all(phase["kd_time_fade"] is True for phase in h100_phases)
    h100._print_dry_run("validation", "adamw")
    assert "Stage-3 teacher KD    = 1.00 -> 0.30" in capsys.readouterr().out

    h100._print_dry_run("repair1b", "adamw")
    repair_output = capsys.readouterr().out
    assert "Stage-3 teacher KD    = 0.30" in repair_output
    repair_row = next(
        line for line in repair_output.splitlines() if line.strip().startswith("repair1b")
    )
    assert repair_row.split()[3] == "0"

    with pytest.raises(ValueError, match="must stay positive"):
        h100._build_stage3_phases(
            1,
            1,
            2e-4,
            3e-5,
            frozen_kd_weight=0.0,
        )


def test_grpo_resume_fails_closed_before_replaying_prompts(tmp_path: Path) -> None:
    import scripts.train_4090 as train

    with pytest.raises(NotImplementedError, match="Exact GRPO resume"):
        train.run_grpo(
            "weights.pt",
            torch.device("cpu"),
            save_dir=str(tmp_path),
            resume="grpo_step100.pt",
        )


def test_h100_save_dir_is_forwarded_and_isolates_ab_arms(monkeypatch) -> None:
    import dimba.models.denoiser as denoiser
    import scripts.train_h100 as h100

    calls = []
    current_optimizer = {"name": None}

    def apply_overrides(_preset, optimizer):
        current_optimizer["name"] = optimizer

    def distill(_device, **kwargs):
        calls.append(("distill", current_optimizer["name"], kwargs["save_dir"]))
        return str(Path(kwargs["save_dir"]) / "final.pt")

    def sft(_checkpoint, _device, **kwargs):
        calls.append(("sft", None, kwargs["save_dir"]))
        return str(Path(kwargs["save_dir"]) / "final.pt")

    def grpo(_checkpoint, _device, **kwargs):
        calls.append(("grpo", None, kwargs["save_dir"]))
        return str(Path(kwargs["save_dir"]) / "final.pt")

    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(
            name="H100", total_memory=80_000_000_000, major=9, minor=0
        ),
    )
    monkeypatch.setattr(denoiser, "HAS_MAMBA_SSM", True)
    monkeypatch.setitem(sys.modules, "causal_conv1d", SimpleNamespace())
    monkeypatch.setattr(h100, "apply_h100_overrides", apply_overrides)
    monkeypatch.setattr(h100, "run_distill", distill)
    monkeypatch.setattr(h100, "run_sft", sft)
    monkeypatch.setattr(h100, "run_grpo", grpo)

    commands = [
        ["--phase", "distill", "--stage3-optimizer", "adamw", "--save-dir", "arm-adamw"],
        ["--phase", "distill", "--stage3-optimizer", "muon", "--save-dir", "arm-muon"],
        [
            "--phase",
            "sft",
            "--quality-gate-passed",
            "--checkpoint",
            "base.pt",
            "--save-dir",
            "sft-arm",
        ],
        [
            "--phase",
            "grpo",
            "--quality-gate-passed",
            "--checkpoint",
            "sft.pt",
            "--save-dir",
            "grpo-arm",
        ],
    ]
    for command in commands:
        monkeypatch.setattr(sys, "argv", ["train_h100.py", *command])
        h100.main()

    assert calls == [
        ("distill", "adamw", "arm-adamw"),
        ("distill", "muon", "arm-muon"),
        ("sft", None, "sft-arm"),
        ("grpo", None, "grpo-arm"),
    ]


def test_distill_checkpoint_contains_full_resume_and_provenance(tmp_path: Path) -> None:
    import scripts.train_4090 as train

    model = _TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = SimpleNamespace(
        optimizer=optimizer,
        optimizer_name="adamw",
        current_stage_id="stage3a",
        stage_step=4,
        stage_total_steps=10,
    )
    path = tmp_path / "distill.pt"
    train._save_distill_checkpoint(
        str(path),
        model=model,
        trainer=trainer,
        stage_index=0,
        stage_config={"name": "stage3", "steps": 10, "lr": 1e-3},
        stage_complete=False,
        data_state={"batches_consumed": 4},
        device=torch.device("cpu"),
    )
    checkpoint = torch.load(path)
    assert {
        "model_state_dict",
        "optimizer_state_dict",
        "rng_state",
        "config",
        "stage_id",
        "stage_step",
        "stage_total_steps",
        "data_state",
        "optimizer_policy",
        "mixer_backend",
        "provenance",
    } <= checkpoint.keys()
    assert checkpoint["stage_step"] == 4
    assert checkpoint["provenance"]["git_sha"]
