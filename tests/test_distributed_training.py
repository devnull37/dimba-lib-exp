"""Focused checks for the native torchrun/DDP training path."""

from __future__ import annotations

import importlib.util
import inspect
import os
import socket
import sys
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset

from dimba import DIMBA
from dimba.distillation.teacher import TeacherOutputs
from dimba.distillation.trainer import DistillationConfig, DistillationTrainer
from dimba.training.distributed import (
    backward_sync_context,
    cycling_batches,
    init_distributed,
    make_sampler,
    rank_zero_torch_cache,
    restore_training_state,
    save_training_checkpoint,
    seed_process,
    wrap_ddp,
)
from dimba.training.masked import (
    MaskedDiffusionObjective,
    freeze_masked_only_unused_parameters,
)
from dimba.utils import backends


class _Rows(Dataset):
    def __init__(self) -> None:
        self.rows = (torch.arange(8)[:, None] + torch.arange(4)[None, :]) % 16

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.rows[index]


class _TinyMaskedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(17, 8)
        self.projection = nn.Linear(8, 17)

    def predict_token_features(self, input_ids, _t):
        return self.embedding(input_ids)

    def project_token_features(self, features):
        return self.projection(features)


class _TinyCausalTeacher(nn.Module):
    """Deterministic, network-free teacher with the interface Stage 3 needs."""

    num_layers = 1
    num_heads = 1
    d_model = 8
    vocab_size = 17
    is_causal = True

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor) -> TeacherOutputs:
        self.calls += 1
        target = ((input_ids + 1) % self.vocab_size).unsqueeze(-1)
        vocabulary = torch.arange(self.vocab_size, device=input_ids.device)
        logits = -(vocabulary - target).float().square()
        return TeacherOutputs((), (), logits)


def _tiny_stage3(
    context,
    *,
    stop_after_first_step: bool = False,
) -> tuple[DIMBA, _TinyCausalTeacher, DistillationTrainer]:
    model = DIMBA(
        vocab_size=17,
        d_model=8,
        d_prompt=8,
        num_diffusion_steps=8,
        num_denoiser_layers=1,
        d_state=2,
        d_conv=2,
        expand=1,
        dropout=0.0,
        use_weight_tying=True,
        use_simple_mamba=True,
    )
    teacher = _TinyCausalTeacher()
    config = DistillationConfig(
        teacher_model="stub",
        teacher_type="causal",
        share_vocab=True,
    )
    trainer = DistillationTrainer(
        model,
        teacher,
        config,
        distributed_context=context,
        log_hook=(lambda *_args: "stop") if stop_after_first_step else None,
    )
    return model, teacher, trainer


def _ddp_worker(rank: int, world_size: int, port: int, checkpoint_path: str) -> None:
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world_size),
    )
    context = init_distributed("cpu")
    try:
        seed_process(7, context)
        model = _TinyMaskedModel()
        objective = wrap_ddp(MaskedDiffusionObjective(model, 16, t_min=0.25), context)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        data = _Rows()
        sampler = make_sampler(data, context, seed=19)
        loader = DataLoader(data, batch_size=2, sampler=sampler, drop_last=True)
        batches = cycling_batches(loader, sampler)

        optimizer.zero_grad(set_to_none=True)
        seen = []
        epoch = batch_in_epoch = 0
        for microstep in range(2):
            rows, epoch, batch_in_epoch = next(batches)
            seen.extend(rows[:, 0].tolist())
            with backward_sync_context(objective, sync=microstep == 1):
                loss, _ = objective(rows)
                (loss / 2).backward()
        optimizer.step()
        scheduler.step()

        flat = torch.cat([parameter.detach().flatten() for parameter in model.parameters()])
        weights = [torch.empty_like(flat) for _ in range(world_size)]
        dist.all_gather(weights, flat)
        seen_by_rank = [None] * world_size
        dist.all_gather_object(seen_by_rank, seen)
        if context.is_main:
            assert all(torch.equal(weights[0], other) for other in weights[1:])
            assert set(seen_by_rank[0]).isdisjoint(seen_by_rank[1])
            assert set(seen_by_rank[0] + seen_by_rank[1]) == set(range(8))

        save_training_checkpoint(
            checkpoint_path,
            context=context,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=1,
            epoch=epoch,
            batch_in_epoch=batch_in_epoch,
            extra={"writer_rank": rank},
        )
        expected_random = torch.rand(4)
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        cursor = restore_training_state(
            state,
            context=context,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        assert cursor == (1, 0, 2)
        assert torch.equal(torch.rand(4), expected_random)
    finally:
        context.close()


def _real_dimba_ddp_worker(rank: int, world_size: int, port: int) -> None:
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world_size),
    )
    torch.set_num_threads(1)
    context = init_distributed("cpu")
    try:
        seed_process(31, context)
        model = DIMBA(
            vocab_size=17,
            d_model=8,
            d_prompt=8,
            num_diffusion_steps=8,
            num_denoiser_layers=1,
            d_state=2,
            d_conv=2,
            expand=1,
            dropout=0.0,
            use_weight_tying=True,
            use_simple_mamba=True,
        )
        frozen = freeze_masked_only_unused_parameters(model)
        assert frozen == (
            "prompt_encoder.mlp.0.weight",
            "prompt_encoder.mlp.0.bias",
            "prompt_encoder.mlp.3.weight",
            "prompt_encoder.mlp.3.bias",
        )
        assert not any(parameter.requires_grad for parameter in model.prompt_encoder.parameters())

        objective = wrap_ddp(MaskedDiffusionObjective(model, 16, t_min=0.25), context)
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=1e-2,
        )
        rows = (torch.arange(10).view(2, 5) + rank * 3) % 16
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss, _ = objective(rows)
            loss.backward()
            assert all(
                parameter.grad is not None
                for parameter in model.parameters()
                if parameter.requires_grad
            )
            optimizer.step()

        flat = torch.cat(
            [
                parameter.detach().flatten()
                for parameter in model.parameters()
                if parameter.requires_grad
            ]
        )
        weights = [torch.empty_like(flat) for _ in range(world_size)]
        dist.all_gather(weights, flat)
        assert all(torch.equal(weights[0], other) for other in weights[1:])
    finally:
        context.close()


def _stage3_ddp_worker(rank: int, world_size: int, port: int, checkpoint_path: str) -> None:
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world_size),
    )
    torch.set_num_threads(1)
    context = init_distributed("cpu")
    stage = {
        "name": "stage3",
        "id": "stage3a",
        "steps": 2,
        "lr": 1e-2,
        "weight_decay": 0.0,
        "kd_weight": 0.5,
        "kd_temp": 1.0,
        "ce_loss_weight": 0.25,
        "min_snr_gamma": 5.0,
        "snr_floor": 0.5,
        "ce_time_fade": True,
        "kd_time_fade": True,
    }
    batches = [
        (torch.arange(8).reshape(2, 4) + rank * 3 + offset) % 17
        for offset in (0, 5)
    ]

    try:
        seed_process(101, context)
        full_model, full_teacher, full = _tiny_stage3(context)
        full.run_stage(stage, batches)
        assert full_teacher.calls == 2
        assert isinstance(full._stage3_loss_module, DistributedDataParallel)
        full_weights = torch.cat(
            [parameter.detach().flatten() for parameter in full_model.parameters()]
        )

        seed_process(101, context)
        first_model, first_teacher, first = _tiny_stage3(
            context, stop_after_first_step=True
        )
        first.run_stage(stage, batches)
        assert first.stage_step == 1
        assert first_teacher.calls == 1
        scheduler = torch.optim.lr_scheduler.LambdaLR(first.optimizer, lambda _: 1.0)
        save_training_checkpoint(
            checkpoint_path,
            context=context,
            model=first_model,
            optimizer=first.optimizer,
            scheduler=scheduler,
            step=1,
            epoch=0,
            batch_in_epoch=1,
            extra={
                "stage_id": first.current_stage_id,
                "stage_step": first.stage_step,
                "optimizer_name": first.optimizer_name,
                "writer_rank": rank,
            },
        )

        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        resumed_model, resumed_teacher, resumed = _tiny_stage3(context)
        resumed_model.load_state_dict(state["model_state_dict"])
        resumed.run_stage(stage, batches[1:], resume_state=state)
        assert resumed.stage_completed
        assert resumed_teacher.calls == 1

        resumed_weights = torch.cat(
            [parameter.detach().flatten() for parameter in resumed_model.parameters()]
        )
        assert torch.equal(resumed_weights, full_weights)
        gathered = [torch.empty_like(resumed_weights) for _ in range(world_size)]
        dist.all_gather(gathered, resumed_weights)
        assert all(torch.equal(gathered[0], other) for other in gathered[1:])
    finally:
        context.close()


def _rank_zero_failure_worker(
    rank: int,
    world_size: int,
    port: int,
    cache_path: str,
    checkpoint_path: str,
) -> None:
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world_size),
    )
    dist.init_process_group(
        "gloo",
        init_method="env://",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=8),
    )
    context = init_distributed("cpu")
    try:
        started = time.perf_counter()
        try:
            rank_zero_torch_cache(
                cache_path,
                context,
                lambda: (_ for _ in ()).throw(OSError("injected cache failure")),
            )
        except RuntimeError as exc:
            cache_error = str(exc)
        else:
            raise AssertionError("cache failure was not propagated")
        assert time.perf_counter() - started < 4.0
        cache_errors = [None] * world_size
        dist.all_gather_object(cache_errors, cache_error)
        assert len(set(cache_errors)) == 1
        assert "injected cache failure" in cache_error

        model = _TinyMaskedModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        original_save = torch.save
        if context.is_main:
            def fail_after_write(value, path, *args, **kwargs):
                original_save(value, path, *args, **kwargs)
                if isinstance(path, (str, os.PathLike)) and Path(path) == Path(
                    checkpoint_path
                ).with_name(
                    f".{Path(checkpoint_path).name}.tmp"
                ):
                    raise OSError("injected checkpoint failure")

            torch.save = fail_after_write
        started = time.perf_counter()
        try:
            save_training_checkpoint(
                checkpoint_path,
                context=context,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=0,
                epoch=0,
                batch_in_epoch=0,
            )
        except RuntimeError as exc:
            checkpoint_error = str(exc)
        else:
            raise AssertionError("checkpoint failure was not propagated")
        finally:
            torch.save = original_save
        assert time.perf_counter() - started < 4.0
        checkpoint_errors = [None] * world_size
        dist.all_gather_object(checkpoint_errors, checkpoint_error)
        assert len(set(checkpoint_errors)) == 1
        assert "injected checkpoint failure" in checkpoint_error
        if context.is_main:
            destination = Path(checkpoint_path)
            assert not destination.exists()
            assert not destination.with_name(f".{destination.name}.tmp").exists()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_two_rank_gloo_syncs_weights_shards_data_and_saves_once(tmp_path):
    checkpoint = tmp_path / "ddp.pt"
    mp.spawn(_ddp_worker, args=(2, _free_port(), str(checkpoint)), nprocs=2, join=True)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert state["writer_rank"] == 0
    assert state["world_size"] == 2
    assert len(state["rng_states"]) == 2
    assert (state["step"], state["epoch"], state["batch_in_epoch"]) == (1, 0, 2)
    assert not list(tmp_path.glob(".*.tmp"))


def test_real_dimba_two_rank_two_step_masked_ddp_has_no_unused_parameters():
    mp.spawn(_real_dimba_ddp_worker, args=(2, _free_port()), nprocs=2, join=True)


def test_real_dimba_two_rank_stage3_kd_syncs_and_resumes_exactly(tmp_path):
    checkpoint = tmp_path / "stage3-ddp.pt"
    mp.spawn(
        _stage3_ddp_worker,
        args=(2, _free_port(), str(checkpoint)),
        nprocs=2,
        join=True,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert state["writer_rank"] == 0
    assert state["world_size"] == 2
    assert len(state["rng_states"]) == 2
    assert state["stage_step"] == 1


def test_rank_zero_io_failures_raise_promptly_on_every_rank_and_clean_temp_files(tmp_path):
    mp.spawn(
        _rank_zero_failure_worker,
        args=(
            2,
            _free_port(),
            str(tmp_path / "cache.pt"),
            str(tmp_path / "checkpoint.pt"),
        ),
        nprocs=2,
        join=True,
    )


def _backend_model(module_name: str, class_name: str):
    mixer_type = type(class_name, (), {})
    mixer_type.__module__ = module_name
    mixer = mixer_type()
    block = SimpleNamespace(mamba_fwd=mixer, mamba_bwd=mixer)
    return SimpleNamespace(denoiser=SimpleNamespace(blocks=[block]))


def test_cuda_backend_preflight_accepts_only_mamba2_and_causal_conv(monkeypatch):
    monkeypatch.setattr(backends.importlib, "import_module", lambda _name: SimpleNamespace())
    fast = _backend_model("mamba_ssm.modules.mamba2", "Mamba2")
    assert backends.require_fast_cuda_mamba2(fast, "cuda") == (
        "mamba_ssm.modules.mamba2.Mamba2",
    )

    for module_name, class_name in (
        ("mamba_ssm.modules.mamba_simple", "Mamba"),
        ("dimba.models.torch_mamba2", "TorchMamba2"),
    ):
        with pytest.raises(RuntimeError, match="requires live mamba_ssm.Mamba2"):
            backends.require_fast_cuda_mamba2(
                _backend_model(module_name, class_name),
                "cuda",
            )

    monkeypatch.setattr(
        backends.importlib,
        "import_module",
        lambda _name: (_ for _ in ()).throw(ImportError("missing")),
    )
    with pytest.raises(RuntimeError, match="working causal-conv1d"):
        backends.require_fast_cuda_mamba2(fast, "cuda")


def test_cuda_backend_setup_is_a_noop_off_cuda(monkeypatch):
    monkeypatch.setattr(
        backends.importlib,
        "import_module",
        lambda _name: (_ for _ in ()).throw(AssertionError("must not import")),
    )
    assert backends.require_fast_cuda_mamba2(object(), "cpu") == ()
    backends.configure_cuda_training("mps")


def test_cuda_backend_setup_enables_tf32(monkeypatch):
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    backends.configure_cuda_training("cuda")
    assert torch.backends.cuda.matmul.allow_tf32
    assert torch.backends.cudnn.allow_tf32


def test_sft_objective_keeps_the_complete_loss_inside_forward():
    model = _TinyMaskedModel()
    objective = MaskedDiffusionObjective(
        model,
        16,
        t_min=0.25,
        prompt_dropout=0.1,
        neighbor_unlikelihood_weight=0.5,
    )
    rows = _Rows().rows[:2]
    loss, parts = objective(rows, torch.tensor([1, 2]), torch.tensor([4, 4]))
    loss.backward()
    assert torch.isfinite(loss)
    assert parts.keys() == {"ce_masked", "t_mean", "ul"}


def test_single_process_tensor_cache_is_reused(tmp_path):
    context = init_distributed("cpu")
    path = tmp_path / "rows.pt"
    builds = 0

    def build():
        nonlocal builds
        builds += 1
        return torch.arange(8)

    try:
        assert torch.equal(rank_zero_torch_cache(str(path), context, build), torch.arange(8))
        assert torch.equal(rank_zero_torch_cache(str(path), context, build), torch.arange(8))
        assert builds == 1
    finally:
        context.close()


def test_fineweb_streaming_shards_every_rank_worker_and_resumes_locally(monkeypatch):
    import scripts.train_4090 as train

    shard_calls = []

    class Rows(list):
        def shard(self, num_shards, index):
            shard_calls.append((num_shards, index))
            return Rows(self[index::num_shards])

    rows = Rows({"text": str(index)} for index in range(64))
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_dataset=lambda *_args, **_kwargs: rows),
    )
    tokenizer = SimpleNamespace(
        eos_token_id=0,
        encode=lambda text, add_special_tokens=False: [int(text)],
    )

    observed = {}
    for rank in range(2):
        dataset = train._FineWebStreaming(
            tokenizer,
            2,
            batch_size=2,
            resume_batches=3,
            rank=rank,
            world_size=2,
        )
        for worker in range(2):
            monkeypatch.setattr(
                torch.utils.data,
                "get_worker_info",
                lambda worker=worker: SimpleNamespace(num_workers=2, id=worker),
            )
            iterator = iter(dataset)
            observed[(rank, worker)] = [next(iterator)[0].item() for _ in range(2)]

    assert shard_calls == [(4, 1), (4, 0), (4, 3), (4, 2)]
    assert observed == {
        (0, 0): [9, 13],
        (0, 1): [16, 20],
        (1, 0): [11, 15],
        (1, 1): [18, 22],
    }
    assert len({token for tokens in observed.values() for token in tokens}) == 8


@pytest.mark.parametrize(
    "script_name",
    ["masked_diffusion_finetune.py", "mdm_sft_cfg2.py"],
)
def test_masked_training_cli_uses_per_gpu_batch_and_exact_resume(script_name):
    path = Path(__file__).parents[1] / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(f"test_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    args = module.parse_args(
        [
            "--batch",
            "3",
            "--accumulate",
            "4",
            "--resume",
            "checkpoint.pt",
            "--output-dir",
            "checkpoints/arm-a",
            "--device",
            "cpu",
        ]
    )
    assert (args.batch, args.accumulate, args.resume, args.device) == (
        3,
        4,
        "checkpoint.pt",
        "cpu",
    )
    assert args.output_dir == "checkpoints/arm-a"
    assert 'fused=context.device.type == "cuda"' in inspect.getsource(module._run)
    assert "configure_cuda_training(context.device)" in inspect.getsource(module._run)
    assert "require_fast_cuda_mamba2(model, context.device)" in inspect.getsource(module._run)
