"""CPU-only checks for the H100 benchmark's CLI and promotion math."""

import subprocess

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import scripts.benchmark_h100 as benchmark
from dimba import DIMBA
from dimba.training import fused_ce
from scripts.benchmark_h100 import (
    CASES,
    _run_and_capture_single_output,
    compare_metrics,
    git_reproducibility_info,
    main,
    parse_args,
    percentile,
    planned_config,
    promotion_decision,
    token_output_agreement,
    validate_args,
)


def _metrics(p50, p95, memory):
    return {
        "p50_ms": p50,
        "p95_ms": p95,
        "incremental_peak_allocated_bytes": memory,
    }


def _result(name, baseline, optimized, correctness=True):
    return {
        "name": name,
        "baseline": baseline,
        "optimized": optimized,
        "comparison": compare_metrics(baseline, optimized),
        "correctness": {"pass": correctness},
    }


def test_percentile_uses_linear_interpolation() -> None:
    assert percentile([0, 10, 20], 0.5) == 10
    assert percentile([0, 10], 0.95) == pytest.approx(9.5)
    with pytest.raises(ValueError):
        percentile([], 0.5)


def test_promotion_requires_the_full_rule() -> None:
    micro = _result(
        "masked-training",
        _metrics(13.0, 14.0, 100),
        _metrics(10.0, 13.0, 80),
    )
    masked_inference = _result(
        "masked-inference",
        _metrics(106.0, 110.0, 100),
        _metrics(100.0, 105.0, 95),
    )
    ddim = _result(
        "continuous-ddim",
        _metrics(100.0, 100.0, 100),
        _metrics(100.0, 100.0, 100),
    )
    dpmpp = _result(
        "continuous-dpmpp",
        _metrics(100.0, 100.0, 100),
        _metrics(100.0, 100.0, 100),
    )
    fused_ce = _result(
        "continuous-fused-ce",
        _metrics(110.0, 110.0, 100),
        _metrics(100.0, 105.0, 70),
    )
    full_suite = [micro, masked_inference, ddim, dpmpp, fused_ce]
    assert promotion_decision(full_suite)["promote"]
    assert not promotion_decision(
        full_suite, quality_context_ok=False
    )["promote"]

    subset_gate = promotion_decision(
        [micro, masked_inference],
        requested_cases=["masked-training", "masked-inference"],
    )
    assert not subset_gate["promote"]
    assert subset_gate["non_default_case_selection_non_promotable"]
    assert not subset_gate["default_cases_present_exactly_once"]

    duplicate_gate = promotion_decision(full_suite + [ddim])
    assert not duplicate_gate["promote"]
    assert duplicate_gate["observed_case_counts"]["continuous-ddim"] == 2

    compile_gate = promotion_decision(full_suite, compile_requested=True)
    assert not compile_gate["promote"]
    assert compile_gate["compile_diagnostic_non_promotable"]

    missing_liger = [
        result if result["name"] != "continuous-fused-ce" else {
            "name": "continuous-fused-ce",
            "error": "RuntimeError: missing Liger",
        }
        for result in full_suite
    ]
    assert not promotion_decision(missing_liger)["promote"]

    regressed_tail = _result(
        "masked-inference",
        _metrics(106.0, 110.0, 100),
        _metrics(100.0, 111.0, 70),
    )
    gate = promotion_decision([micro, regressed_tail, ddim, dpmpp, fused_ce])
    assert not gate["promote"]
    assert not gate["no_p95_regression"]


def test_dry_run_plan_is_representative_and_cuda_free() -> None:
    args = parse_args(["--dry-run"])
    validate_args(args)
    plan = planned_config(args)
    assert plan["model"]["d_model"] == 576
    assert plan["masked_inference"]["total_seq_len"] == 512
    assert plan["masked_training_projection_ce"]["batch_size"] == 64
    assert plan["measurement"]["timing_order"] == "interleaved AB/BA"
    assert plan["promotion_case_selection"]["is_complete_default_suite"]
    assert set(plan["promotion_case_selection"]["required_exactly_once"]) == set(CASES)


def test_token_output_agreement_requires_exact_deterministic_output() -> None:
    reference = torch.tensor([[1, 2, 3]])
    assert token_output_agreement(reference, reference.clone())["exact_match"]
    mismatch = token_output_agreement(reference, torch.tensor([[1, 2, 4]]))
    assert not mismatch["exact_match"]
    assert mismatch["token_agreement"] == pytest.approx(2 / 3)


def test_final_logit_capture_records_exactly_one_projection() -> None:
    head = nn.Linear(3, 5)
    features = torch.randn(2, 4, 3)
    ids, logits = _run_and_capture_single_output(
        head,
        lambda: head(features).argmax(dim=-1),
    )
    assert torch.equal(ids, logits.argmax(dim=-1))
    with pytest.raises(RuntimeError, match="expected one final-logit projection"):
        _run_and_capture_single_output(head, lambda: torch.zeros(1))


def test_continuous_correctness_gates_seeded_final_logits(monkeypatch) -> None:
    from dimba.diffusion import sampling

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.d_latent = 3
            self.num_diffusion_steps = 8
            self.output_head = nn.Linear(3, 7, bias=False)

        def conditioning_from_prompt(self, _prompt, batch, device, drop_cond=False):
            value = -0.25 if drop_cond else 0.5
            return torch.full((batch, 1, self.d_latent), value, device=device)

        def denoise_to_x0_latent(self, x_t, _t, cond, x_self_cond=None):
            result = 0.5 * x_t + 0.1 * cond.expand(-1, x_t.shape[1], -1)
            return result if x_self_cond is None else result + 0.05 * x_self_cond

    def fake_sample(
        model,
        prompt_ids,
        seq_len,
        *,
        guidance_scale,
        device,
        **_kwargs,
    ):
        batch = prompt_ids.shape[0]
        x_t = torch.randn(batch, seq_len, model.d_latent, device=device)
        timestep = torch.zeros(batch, dtype=torch.long, device=device)
        cond = model.conditioning_from_prompt(prompt_ids, batch, device)
        uncond = model.conditioning_from_prompt(None, batch, device, drop_cond=True)
        both = torch.cat((cond, uncond), dim=0)
        conditional, unconditional = sampling._batched_cfg_denoise(
            model.denoise_to_x0_latent,
            x_t,
            timestep,
            cond,
            both,
            None,
        )
        guided = unconditional + guidance_scale * (conditional - unconditional)
        probabilities = model.output_head(guided).softmax(dim=-1)
        return torch.multinomial(probabilities.flatten(0, 1), 1).view(batch, seq_len)

    def fake_measure(*_args, **_kwargs):
        metrics = {
            "p50_ms": 1.0,
            "p95_ms": 1.0,
            "incremental_peak_allocated_bytes": 1,
        }
        return dict(metrics), dict(metrics), None

    monkeypatch.setattr(sampling, "sample_from_model", fake_sample)
    monkeypatch.setattr(benchmark, "measure_pair_cuda", fake_measure)
    args = parse_args(
        [
            "--vocab-size",
            "7",
            "--seq-len",
            "6",
            "--prompt-len",
            "2",
            "--inference-batch-size",
            "2",
            "--steps",
            "2",
            "--warmup",
            "0",
            "--repeats",
            "1",
        ]
    )
    result = benchmark.run_continuous(Model().eval(), args, torch.device("cpu"), "ddim")
    correctness = result["correctness"]
    assert correctness["pass"]
    assert correctness["seeded_final_response_logit_parity"]["pass"]
    assert correctness["shared_rng_scope"] == (
        "initial Gaussian noise and final multinomial draw"
    )


def test_masked_correctness_requires_exact_full_generation_output(monkeypatch) -> None:
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(9, 4)
            self.head = nn.Linear(4, 9, bias=False)

        def predict_token_features(self, ids, _t):
            return self.embedding(ids)

        def project_token_features(self, features, positions=None):
            if positions is not None:
                features = features.gather(
                    1,
                    positions.unsqueeze(-1).expand(-1, -1, features.shape[-1]),
                )
            return self.head(features)

        def predict_token_logits(self, ids, t):
            return self.project_token_features(self.predict_token_features(ids, t))

    def fake_measure(*_args, **_kwargs):
        metrics = {
            "p50_ms": 1.0,
            "p95_ms": 1.0,
            "incremental_peak_allocated_bytes": 1,
        }
        return dict(metrics), dict(metrics), None

    monkeypatch.setattr(benchmark, "measure_pair_cuda", fake_measure)
    args = parse_args(
        [
            "--vocab-size",
            "9",
            "--seq-len",
            "6",
            "--prompt-len",
            "2",
            "--inference-batch-size",
            "2",
            "--steps",
            "2",
            "--warmup",
            "0",
            "--repeats",
            "1",
        ]
    )
    result = benchmark.run_masked_inference(Model().eval(), args, torch.device("cpu"))
    agreement = result["correctness"]["deterministic_full_generation_output_agreement"]
    assert result["correctness"]["pass"]
    assert agreement["exact_match"]
    assert agreement["token_agreement"] == 1.0


def test_continuous_fused_ce_requires_liger(monkeypatch) -> None:
    monkeypatch.setattr(fused_ce, "liger_fused_ce_available", lambda: False)
    with pytest.raises(RuntimeError, match="requires an importable liger-kernel"):
        benchmark.run_continuous_fused_ce(
            object(),
            parse_args([]),
            torch.device("cpu"),
            torch.float32,
        )


def test_continuous_fused_ce_case_parity_with_fake_liger(monkeypatch) -> None:
    class FakeLiger:
        def __init__(self, reduction="mean"):
            self.reduction = reduction

        def __call__(self, weight, features, targets, bias=None):
            return F.cross_entropy(
                F.linear(features, weight, bias),
                targets,
                reduction=self.reduction,
            )

    def fake_measure(baseline, optimized, **_kwargs):
        baseline()
        optimized()
        baseline_metrics = {
            "p50_ms": 2.0,
            "p95_ms": 2.0,
            "incremental_peak_allocated_bytes": 100,
        }
        optimized_metrics = {
            "p50_ms": 1.0,
            "p95_ms": 1.0,
            "incremental_peak_allocated_bytes": 50,
        }
        return baseline_metrics, optimized_metrics, None

    fused_ce._liger_loss.cache_clear()
    monkeypatch.setattr(fused_ce, "_load_liger_loss", lambda: FakeLiger)
    monkeypatch.setattr(fused_ce, "_liger_runtime_ready", lambda _features: True)
    monkeypatch.setattr(benchmark, "measure_pair_cuda", fake_measure)
    model = DIMBA(
        vocab_size=17,
        d_model=8,
        d_prompt=8,
        num_diffusion_steps=8,
        num_denoiser_layers=1,
        d_state=2,
        d_conv=2,
        dropout=0.0,
        use_simple_mamba=True,
        use_weight_tying=True,
        use_head_norm=True,
    )
    args = parse_args(
        [
            "--vocab-size",
            "17",
            "--seq-len",
            "4",
            "--prompt-len",
            "1",
            "--training-batch-size",
            "2",
            "--repeats",
            "1",
        ]
    )
    try:
        result = benchmark.run_continuous_fused_ce(
            model,
            args,
            torch.device("cpu"),
            torch.float32,
        )
    finally:
        fused_ce._liger_loss.cache_clear()
    assert result["correctness"]["pass"]
    assert result["correctness"]["semantic_scope"].startswith("uniform mean CE only")
    assert result["comparison"]["incremental_hbm_reduction_fraction"] == 0.5


def test_git_reproducibility_hash_is_stable_and_includes_untracked(tmp_path, monkeypatch):
    def git(*args):
        return subprocess.run(
            ["git", *args],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        )

    git("init", "-q")
    git("config", "user.email", "benchmark@example.com")
    git("config", "user.name", "Benchmark Test")
    git("config", "commit.gpgsign", "false")
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("base\n", encoding="utf-8")
    git("add", "tracked.txt")
    git("commit", "-qm", "initial")
    monkeypatch.setattr(benchmark, "ROOT", tmp_path)

    clean = git_reproducibility_info()
    assert clean["git_dirty"] is False
    tracked.write_text("changed\n", encoding="utf-8")
    dirty = git_reproducibility_info()
    assert dirty["git_dirty"] is True
    assert dirty["git_worktree_diff_sha256"] != clean["git_worktree_diff_sha256"]
    assert dirty["git_worktree_diff_sha256"] == git_reproducibility_info()[
        "git_worktree_diff_sha256"
    ]

    (tmp_path / "untracked.txt").write_text("new\n", encoding="utf-8")
    with_untracked = git_reproducibility_info()
    assert with_untracked["git_untracked_files_hashed"] == 1
    assert with_untracked["git_worktree_diff_sha256"] != dirty["git_worktree_diff_sha256"]
    assert with_untracked["git_worktree_diff_hash_complete"]


def test_main_fails_clearly_without_cuda(monkeypatch, capsys) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    assert main(["--cases", "masked-training", "--repeats", "1"]) == 1
    assert "CUDA is required" in capsys.readouterr().err
