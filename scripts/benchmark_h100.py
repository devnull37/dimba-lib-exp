#!/usr/bin/env python3
"""Trustworthy CUDA benchmark for DIMBA's H100-critical performance seams.

Run one case per process when measuring ``torch.compile`` cold-start latency.
The default suite compares eager implementations and never extrapolates from CPU/MPS.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


CASES = {
    "masked-inference": (
        "full iterative masked generation: full/two-pass CFG vs selected/batched CFG"
    ),
    "continuous-ddim": "continuous DDIM: two sequential CFG dispatches vs one batched dispatch",
    "continuous-dpmpp": "continuous DPM-Solver++(2M): sequential vs batched CFG",
    "continuous-fused-ce": "uniform continuous CE: materialized logits vs Liger fused linear CE",
    "masked-training": "masked CE forward+backward: full [B,L,V] vs selected-token projection",
}
END_TO_END_CASES = frozenset({"masked-inference", "continuous-ddim", "continuous-dpmpp"})
MICRO_CASE = "masked-training"
FUSED_CE_CASE = "continuous-fused-ce"
MODEL_CASES = END_TO_END_CASES | {FUSED_CE_CASE}


def _is_complete_default_case_selection(cases: Sequence[str]) -> bool:
    return Counter(cases) == Counter(CASES.keys())


def percentile(values: Sequence[float], q: float) -> float:
    """Return a linearly interpolated percentile using only the standard library."""
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    rank = (len(ordered) - 1) * q
    low, high = math.floor(rank), math.ceil(rank)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark DIMBA on a real H100 with parity and promotion gates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cases", nargs="+", choices=tuple(CASES), default=list(CASES))
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path, help="Also write the JSON report here.")
    parser.add_argument("--fail-on-gate", action="store_true")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--allow-non-h100", action="store_true")
    parser.add_argument("--allow-torch-mamba2", action="store_true")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Native checkpoint with strict model_state_dict + config loading.",
    )

    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--vocab-size", type=int, default=49_152)
    parser.add_argument("--d-model", type=int, default=576)
    parser.add_argument("--layers", type=int, default=30)
    parser.add_argument("--d-state", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--prompt-len", type=int, default=64)
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--training-batch-size", type=int, default=64)
    parser.add_argument("--masked-fraction", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=2.0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-block-ffn", action="store_false", dest="block_ffn")
    parser.set_defaults(block_ffn=True)

    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="reduce-overhead",
    )
    parser.add_argument("--parity-rtol", type=float, default=2e-2)
    parser.add_argument("--parity-atol", type=float, default=2e-2)
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "vocab_size": args.vocab_size,
        "d_model": args.d_model,
        "layers": args.layers,
        "d_state": args.d_state,
        "seq_len": args.seq_len,
        "inference_batch_size": args.inference_batch_size,
        "training_batch_size": args.training_batch_size,
        "steps": args.steps,
        "repeats": args.repeats,
    }
    bad = [name for name, value in positive.items() if value < 1]
    if bad:
        raise ValueError(f"must be >= 1: {', '.join(bad)}")
    if args.warmup < 0:
        raise ValueError("warmup must be >= 0")
    if not 0 < args.prompt_len < args.seq_len:
        raise ValueError("prompt_len must be > 0 and < seq_len")
    if not 0 < args.masked_fraction <= 1:
        raise ValueError("masked_fraction must be in (0, 1]")
    if args.d_model % 64:
        raise ValueError("d_model must be divisible by Mamba2 headdim=64")


def planned_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Return exact workload geometry without touching CUDA."""
    generated = args.seq_len - args.prompt_len
    dtype_bytes = 2
    masked_train_tokens = max(1, round(args.seq_len * args.masked_fraction))
    return {
        "cases": list(args.cases),
        "promotion_case_selection": {
            "required_exactly_once": list(CASES),
            "is_complete_default_suite": _is_complete_default_case_selection(args.cases),
            "custom_selection_is_non_promotable": not _is_complete_default_case_selection(
                args.cases
            ),
        },
        "model": {
            "weights": str(args.checkpoint) if args.checkpoint else "random_seeded_synthetic",
            "vocab_size": args.vocab_size,
            "d_model": args.d_model,
            "num_denoiser_layers": args.layers,
            "d_state": args.d_state,
            "d_conv": 4,
            "expand": 2,
            "bidirectional": True,
            "block_ffn": args.block_ffn,
            "dtype": args.dtype,
        },
        "masked_inference": {
            "batch_size": args.inference_batch_size,
            "prompt_len": args.prompt_len,
            "generated_tokens_per_sample": generated,
            "total_seq_len": args.seq_len,
            "steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "baseline_dispatches": 2 * args.steps,
            "optimized_dispatches": args.steps,
            "effective_nfe_both": 2 * args.steps,
        },
        "continuous": {
            "batch_size": args.inference_batch_size,
            "prompt_len": args.prompt_len,
            "generated_tokens_per_sample": generated,
            "total_seq_len": args.seq_len,
            "steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "baseline_dispatches": 2 * args.steps,
            "optimized_dispatches": args.steps,
            "effective_nfe_both": 2 * args.steps,
        },
        "masked_training_projection_ce": {
            "batch_size": args.training_batch_size,
            "seq_len": args.seq_len,
            "masked_tokens_per_sample": masked_train_tokens,
            "selected_tokens_total": args.training_batch_size * masked_train_tokens,
            "full_logits_bytes": (
                args.training_batch_size * args.seq_len * args.vocab_size * dtype_bytes
            ),
            "includes_backward": True,
        },
        "continuous_fused_ce": {
            "batch_size": args.training_batch_size,
            "seq_len": args.seq_len,
            "tokens": args.training_batch_size * args.seq_len,
            "uniform_reduction": "mean",
            "full_logits_bytes": (
                args.training_batch_size * args.seq_len * args.vocab_size * dtype_bytes
            ),
            "includes_backward": True,
            "weighted_or_masked_semantics": "excluded; native selected CE remains required",
        },
        "measurement": {
            "warmup_pairs": args.warmup,
            "timed_repeats_per_variant": args.repeats,
            "timing_order": "interleaved AB/BA",
            "cuda_synchronization": "warmup, memory, and timed-region boundaries only",
            "compile_requested": args.compile,
            "compile_run_promotable": False if args.compile else None,
            "compile_mode": args.compile_mode if args.compile else None,
            "cold_compile": "first invocation per compiled callable, excluded from steady-state",
        },
    }


def _package_version(distribution: str) -> Optional[str]:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def detect_liger() -> Dict[str, Any]:
    installed = importlib.util.find_spec("liger_kernel") is not None
    result: Dict[str, Any] = {
        "installed": installed,
        "version": _package_version("liger-kernel"),
        "fused_linear_ce_available": False,
    }
    if not installed:
        return result
    errors = []
    for module_name in ("liger_kernel.transformers", "liger_kernel.ops.fused_linear_cross_entropy"):
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, "LigerFusedLinearCrossEntropyLoss"):
                result["fused_linear_ce_available"] = True
                result["import_path"] = (
                    f"{module_name}.LigerFusedLinearCrossEntropyLoss"
                )
                return result
        except Exception as exc:  # optional dependency; report, never hide it
            errors.append(f"{module_name}: {type(exc).__name__}: {exc}")
    if errors:
        result["import_errors"] = errors
    return result


def git_reproducibility_info() -> Dict[str, Any]:
    """Fingerprint tracked changes and untracked files without leaking their contents."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain=v1"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        tracked_diff = subprocess.run(
            ["git", "diff", "--binary", "--no-ext-diff", "--no-textconv", "HEAD", "--"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout.split(b"\0")
    except (OSError, subprocess.CalledProcessError) as exc:
        return {
            "git_commit": None,
            "git_dirty": None,
            "git_status_porcelain": [],
            "git_worktree_diff_sha256": None,
            "git_worktree_diff_hash_complete": False,
            "git_reproducibility_error": f"{type(exc).__name__}: {exc}",
        }

    digest = hashlib.sha256()
    digest.update(b"tracked-diff\0")
    digest.update(tracked_diff)
    complete = True
    untracked_files = [path for path in untracked if path]
    for relative_bytes in sorted(untracked_files):
        relative = os.fsdecode(relative_bytes)
        path = ROOT / relative
        digest.update(b"\0untracked\0")
        digest.update(relative_bytes)
        digest.update(b"\0")
        try:
            content = os.readlink(path).encode() if path.is_symlink() else path.read_bytes()
        except OSError as exc:
            complete = False
            content = f"UNREADABLE:{type(exc).__name__}:{exc}".encode()
        digest.update(content)
    return {
        "git_commit": commit,
        "git_dirty": bool(status),
        "git_status_porcelain": status,
        "git_worktree_diff_sha256": digest.hexdigest(),
        "git_worktree_diff_hash_complete": complete,
        "git_untracked_files_hashed": len(untracked_files),
        "git_worktree_diff_hash_scope": (
            "git diff --binary HEAD plus sorted untracked paths and file contents"
        ),
    }


def environment_info(device: torch.device) -> Dict[str, Any]:
    props = torch.cuda.get_device_properties(device)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        **git_reproducibility_info(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "mamba_ssm": _package_version("mamba-ssm"),
        "causal_conv1d": _package_version("causal-conv1d"),
        "causal_conv1d_importable": importlib.util.find_spec("causal_conv1d") is not None,
        "triton": _package_version("triton"),
        "liger": detect_liger(),
        "device": {
            "index": device.index,
            "name": props.name,
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "total_hbm_bytes": props.total_memory,
            "multiprocessors": props.multi_processor_count,
            "is_h100": "H100" in props.name.upper(),
        },
        "tf32": {
            "matmul": torch.backends.cuda.matmul.allow_tf32,
            "cudnn": torch.backends.cudnn.allow_tf32,
        },
    }


def build_model(
    args: argparse.Namespace, device: torch.device, dtype: torch.dtype
) -> Tuple[Any, Dict[str, Any]]:
    from dimba.models.diffusion import DIMBA

    torch.manual_seed(args.seed)
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError("checkpoint must be a dictionary")
        config = checkpoint.get("config")
        state = checkpoint.get("model_state_dict")
        if not isinstance(config, dict) or not isinstance(state, dict):
            raise ValueError("checkpoint must contain config and model_state_dict dictionaries")
        expected = {
            "vocab_size": args.vocab_size,
            "d_model": args.d_model,
            "num_denoiser_layers": args.layers,
            "d_state": args.d_state,
            "block_ffn": args.block_ffn,
        }
        mismatch = {
            key: {"checkpoint": config.get(key), "cli": value}
            for key, value in expected.items()
            if config.get(key) != value
        }
        if mismatch:
            raise ValueError(
                f"checkpoint/CLI shape mismatch: {mismatch}; pass the checkpoint's exact shape"
            )
        model = DIMBA(**config)
        model.load_state_dict(state, strict=True)
        source = {
            "kind": "checkpoint",
            "path": str(args.checkpoint.resolve()),
            "size_bytes": args.checkpoint.stat().st_size,
            "strict_load": True,
        }
    else:
        model = DIMBA(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            d_prompt=args.d_model,
            num_diffusion_steps=1_000,
            num_denoiser_layers=args.layers,
            d_state=args.d_state,
            d_conv=4,
            expand=2,
            conditioning_type="adaln",
            dropout=0.0,
            use_weight_tying=True,
            use_simple_mamba=False,
            latent_diffusion=False,
            bidirectional=True,
            self_conditioning=False,
            prediction_type="x0",
            block_ffn=args.block_ffn,
            use_flow_matching=False,
        )
        source = {"kind": "random_seeded_synthetic", "seed": args.seed}
    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model, source


def mamba_backend_info(model) -> Dict[str, Any]:
    mixers = []
    for block in model.denoiser.blocks:
        for mixer in (block.mamba_fwd, block.mamba_bwd):
            if mixer is not None:
                mixers.append(f"{type(mixer).__module__}.{type(mixer).__name__}")
    unique = sorted(set(mixers))
    fast = bool(unique) and all(
        name.startswith("mamba_ssm.") and name.endswith(".Mamba2") for name in unique
    )
    return {"implementations": unique, "all_fast_cuda_mamba2": fast}


def _seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _unique_graphs() -> Optional[int]:
    try:
        return int(torch._dynamo.utils.counters["stats"]["unique_graphs"])
    except (AttributeError, KeyError, TypeError):
        return None


def _clear_compile_counters() -> None:
    try:
        torch._dynamo.utils.counters.clear()
    except AttributeError:
        pass


def _memory_metrics(fn: Callable[[], torch.Tensor], device: torch.device) -> Dict[str, int]:
    gc.collect()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    base_allocated = torch.cuda.memory_allocated(device)
    base_reserved = torch.cuda.memory_reserved(device)
    output = fn()
    del output
    torch.cuda.synchronize(device)
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    return {
        "base_allocated_bytes": base_allocated,
        "base_reserved_bytes": base_reserved,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "incremental_peak_allocated_bytes": max(0, peak_allocated - base_allocated),
        "incremental_peak_reserved_bytes": max(0, peak_reserved - base_reserved),
    }


def measure_pair_cuda(
    baseline_fn: Callable[[], torch.Tensor],
    optimized_fn: Callable[[], torch.Tensor],
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
    monitor_compile: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[int]]:
    """Interleave AB/BA timing; synchronize only at measurement boundaries."""
    for index in range(warmup):
        pair = (baseline_fn, optimized_fn) if index % 2 == 0 else (optimized_fn, baseline_fn)
        for fn in pair:
            output = fn()
            del output
    torch.cuda.synchronize(device)
    baseline_memory = _memory_metrics(baseline_fn, device)
    optimized_memory = _memory_metrics(optimized_fn, device)
    graphs_before_timing = _unique_graphs() if monitor_compile else None

    timings: Dict[str, List[float]] = {"baseline": [], "optimized": []}
    recorded: List[Tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
    for index in range(repeats):
        order = ("baseline", "optimized") if index % 2 == 0 else ("optimized", "baseline")
        for name in order:
            fn = baseline_fn if name == "baseline" else optimized_fn
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = fn()
            end.record()
            del output
            recorded.append((name, start, end))
    torch.cuda.synchronize(device)
    for name, start, end in recorded:
        timings[name].append(start.elapsed_time(end))
    graphs_after_timing = _unique_graphs() if monitor_compile else None

    def summarize(name: str, memory: Dict[str, int]) -> Dict[str, Any]:
        values = timings[name]
        return {
            "p50_ms": statistics.median(values),
            "p95_ms": percentile(values, 0.95),
            "min_ms": min(values),
            "max_ms": max(values),
            "runs_ms": values,
            **memory,
        }

    recompile_delta = None
    if graphs_before_timing is not None and graphs_after_timing is not None:
        recompile_delta = max(0, graphs_after_timing - graphs_before_timing)
    return (
        summarize("baseline", baseline_memory),
        summarize("optimized", optimized_memory),
        recompile_delta,
    )


def tensor_parity(
    reference: torch.Tensor,
    actual: torch.Tensor,
    *,
    rtol: float,
    atol: float,
) -> Dict[str, Any]:
    same_shape = reference.shape == actual.shape
    finite = bool(torch.isfinite(reference).all() and torch.isfinite(actual).all())
    if not same_shape:
        return {
            "pass": False,
            "same_shape": False,
            "reference_shape": list(reference.shape),
            "actual_shape": list(actual.shape),
            "finite": finite,
        }
    max_abs = float((reference.float() - actual.float()).abs().max().item())
    allclose = bool(torch.allclose(reference.float(), actual.float(), rtol=rtol, atol=atol))
    result: Dict[str, Any] = {
        "pass": finite and allclose,
        "same_shape": True,
        "finite": finite,
        "allclose": allclose,
        "max_abs_error": max_abs,
        "rtol": rtol,
        "atol": atol,
    }
    if reference.ndim >= 2 and reference.shape[-1] > 1:
        result["argmax_agreement"] = float(
            (reference.argmax(dim=-1) == actual.argmax(dim=-1)).float().mean().item()
        )
    return result


def token_output_agreement(reference: torch.Tensor, actual: torch.Tensor) -> Dict[str, Any]:
    """Report exact and per-token agreement for generated token tensors."""
    same_shape = reference.shape == actual.shape
    if not same_shape:
        return {
            "exact_match": False,
            "token_agreement": 0.0,
            "reference_shape": list(reference.shape),
            "actual_shape": list(actual.shape),
        }
    exact = bool(torch.equal(reference, actual))
    agreement = float((reference == actual).float().mean().item()) if reference.numel() else 1.0
    return {
        "exact_match": exact,
        "token_agreement": agreement,
        "reference_shape": list(reference.shape),
        "actual_shape": list(actual.shape),
    }


def _run_and_capture_single_output(module, fn: Callable[[], torch.Tensor]):
    """Run ``fn`` and retain the sole output produced by ``module``."""
    captured = []

    def capture(_module, _inputs, output):
        if not torch.is_tensor(output):
            raise TypeError("captured module output must be a tensor")
        captured.append(output.detach())

    handle = module.register_forward_hook(capture)
    try:
        result = fn()
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError(f"expected one final-logit projection, captured {len(captured)}")
    return result, captured[0]


def compare_metrics(baseline: Dict[str, Any], optimized: Dict[str, Any]) -> Dict[str, Any]:
    baseline_memory = baseline["incremental_peak_allocated_bytes"]
    optimized_memory = optimized["incremental_peak_allocated_bytes"]
    hbm_reduction = (
        (baseline_memory - optimized_memory) / baseline_memory if baseline_memory else 0.0
    )
    return {
        "p50_speedup": baseline["p50_ms"] / optimized["p50_ms"],
        "p95_speedup": baseline["p95_ms"] / optimized["p95_ms"],
        "p95_no_regression": optimized["p95_ms"] <= baseline["p95_ms"],
        "incremental_hbm_reduction_fraction": hbm_reduction,
        "incremental_hbm_reduction_bytes": baseline_memory - optimized_memory,
    }


def promotion_decision(
    results: Sequence[Dict[str, Any]],
    *,
    environment_ok: bool = True,
    quality_context_ok: bool = True,
    requested_cases: Optional[Sequence[str]] = None,
    compile_requested: bool = False,
) -> Dict[str, Any]:
    """Enforce the agreed kernel/optimization merge rule across a complete suite."""
    successful = [result for result in results if "error" not in result]
    micro = next((result for result in successful if result["name"] == MICRO_CASE), None)
    fused_ce = next((result for result in successful if result["name"] == FUSED_CE_CASE), None)
    end_to_end = [result for result in successful if result["name"] in END_TO_END_CASES]
    micro_pass = bool(micro and micro["comparison"]["p50_speedup"] >= 1.3)
    end_to_end_pass = any(
        result["comparison"]["p50_speedup"] >= 1.05
        or result["comparison"]["incremental_hbm_reduction_fraction"] >= 0.20
        for result in end_to_end
    )
    fused_ce_pass = bool(
        fused_ce
        and (
            fused_ce["comparison"]["p50_speedup"] >= 1.05
            or fused_ce["comparison"]["incremental_hbm_reduction_fraction"] >= 0.20
        )
    )
    parity_pass = bool(successful) and all(
        result["correctness"]["pass"] for result in successful
    )
    p95_pass = bool(successful) and all(
        result["comparison"]["p95_no_regression"] for result in successful
    )
    result_names = [str(result.get("name")) for result in results]
    selected_cases = result_names if requested_cases is None else list(requested_cases)
    default_selection = _is_complete_default_case_selection(selected_cases)
    exact_results = _is_complete_default_case_selection(result_names)
    complete = default_selection and exact_results and len(successful) == len(results)
    promote = (
        environment_ok
        and quality_context_ok
        and not compile_requested
        and complete
        and micro_pass
        and fused_ce_pass
        and end_to_end_pass
        and parity_pass
        and p95_pass
    )
    return {
        "promote": promote,
        "environment_is_target_h100_fast_mamba2": environment_ok,
        "trained_checkpoint_quality_context": quality_context_ok,
        "compile_diagnostic_non_promotable": compile_requested,
        "suite_complete": complete,
        "default_cases_present_exactly_once": exact_results,
        "non_default_case_selection_non_promotable": not default_selection,
        "required_cases_exactly_once": list(CASES),
        "observed_case_counts": dict(sorted(Counter(result_names).items())),
        "microkernel_p50_speedup_at_least_1_3x": micro_pass,
        "fused_ce_p50_at_least_1_05x_or_hbm_reduction_at_least_20pct": fused_ce_pass,
        "end_to_end_p50_at_least_1_05x_or_hbm_reduction_at_least_20pct": end_to_end_pass,
        "correctness_parity_no_accuracy_regression": parity_pass,
        "no_p95_regression": p95_pass,
        "rule": (
            ">=1.3x selected-CE microkernel AND (>=1.05x fused CE OR >=20% fused-CE HBM "
            "reduction) AND (>=1.05x end-to-end OR >=20% incremental HBM reduction) AND "
            "parity AND no p95 regression"
        ),
    }


def _add_throughput(metrics: Dict[str, Any], units: int, key: str) -> None:
    metrics[key] = units * 1_000.0 / metrics["p50_ms"]


def _cold_compile(
    fn: Callable[..., torch.Tensor],
    sample_args: Tuple[Any, ...],
    *,
    device: torch.device,
    mode: str,
    dynamic: bool,
) -> Tuple[Callable[..., torch.Tensor], Dict[str, Any]]:
    _clear_compile_counters()
    compiled = torch.compile(fn, mode=mode, dynamic=dynamic, fullgraph=False)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        output = compiled(*sample_args)
    torch.cuda.synchronize(device)
    cold_ms = (time.perf_counter() - started) * 1_000.0
    del output
    return compiled, {
        "requested": True,
        "cold_compile_ms": cold_ms,
        "unique_graphs_after_cold_call": _unique_graphs(),
        "cold_compile_definition": (
            "first call of this callable in the current process; run one case per fresh process "
            "to exclude persistent Inductor cache hits"
        ),
    }


def _make_masked_predictors(model, mask_id: int, prompt_len: int, guidance: float):
    def baseline(ids: torch.Tensor, t):
        unconditional = ids.clone()
        unconditional[:, :prompt_len] = mask_id
        conditional_logits = model.predict_token_logits(ids, t).float()
        unconditional_logits = model.predict_token_logits(unconditional, t).float()
        return unconditional_logits + guidance * (conditional_logits - unconditional_logits)

    def optimized(ids: torch.Tensor, t, positions: Optional[torch.Tensor] = None):
        unconditional = ids.clone()
        unconditional[:, :prompt_len] = mask_id
        features = model.predict_token_features(torch.cat((ids, unconditional), dim=0), t)
        conditional_features, unconditional_features = features.chunk(2, dim=0)
        guided = unconditional_features + guidance * (
            conditional_features - unconditional_features
        )
        return model.project_token_features(guided, positions=positions).float()

    return baseline, optimized


def run_masked_inference(
    model,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    from dimba.diffusion.masked_sampling import masked_diffusion_sample

    batch = args.inference_batch_size
    generated = args.seq_len - args.prompt_len
    mask_id = args.vocab_size - 1
    prompt = torch.randint(0, mask_id, (batch, args.prompt_len), device=device)
    baseline_predict, optimized_predict = _make_masked_predictors(
        model, mask_id, args.prompt_len, args.guidance_scale
    )

    def sample(predict):
        return masked_diffusion_sample(
            predict,
            prompt,
            generated,
            mask_id,
            args.steps,
            temperature=1.0,
            remask=False,
            device=device,
        )

    active_positions = torch.arange(args.prompt_len, args.seq_len, device=device)[None, :]
    active_positions = active_positions.expand(batch, -1)
    initial_ids = torch.full((batch, args.seq_len), mask_id, dtype=torch.long, device=device)
    initial_ids[:, : args.prompt_len] = prompt
    with torch.inference_mode():
        full_reference = baseline_predict(initial_ids, 1.0)
        reference = full_reference[:, args.prompt_len :, :]
        optimized_logits = optimized_predict(initial_ids, 1.0, active_positions)
        logit_parity = tensor_parity(
            reference, optimized_logits, rtol=args.parity_rtol, atol=args.parity_atol
        )
        del full_reference, reference, optimized_logits
        baseline_ids = sample(baseline_predict)
        optimized_ids = sample(optimized_predict)
        output_agreement = token_output_agreement(baseline_ids, optimized_ids)
        valid_ids = bool(
            (optimized_ids >= 0).all()
            and (optimized_ids < args.vocab_size).all()
            and optimized_ids.shape == (batch, generated)
        )
        del baseline_ids, optimized_ids

    baseline_metrics, optimized_metrics, _ = measure_pair_cuda(
        lambda: sample(baseline_predict),
        lambda: sample(optimized_predict),
        device=device,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    units = batch * generated
    _add_throughput(baseline_metrics, units, "generated_tokens_per_second")
    _add_throughput(optimized_metrics, units, "generated_tokens_per_second")
    result: Dict[str, Any] = {
        "name": "masked-inference",
        "kind": "end_to_end",
        "baseline_variant": "full_sequence_vocab_projection_two_sequential_cfg_passes",
        "optimized_variant": "selected_unresolved_projection_one_batched_cfg_pass",
        "shape": {
            "batch": batch,
            "prompt": args.prompt_len,
            "generated": generated,
            "total_length": args.seq_len,
            "vocab": args.vocab_size,
            "steps": args.steps,
            "baseline_dispatches": 2 * args.steps,
            "optimized_dispatches": args.steps,
            "effective_nfe_both": 2 * args.steps,
        },
        "baseline": baseline_metrics,
        "optimized": optimized_metrics,
        "comparison": compare_metrics(baseline_metrics, optimized_metrics),
        "correctness": {
            "pass": logit_parity["pass"] and output_agreement["exact_match"] and valid_ids,
            "first_step_selected_logit_parity": logit_parity,
            "deterministic_full_generation_output_agreement": output_agreement,
            "full_generation_token_agreement": output_agreement["token_agreement"],
            "optimized_ids_valid": valid_ids,
            "accuracy_gate": (
                "selected logits match within tolerance and deterministic full outputs agree"
            ),
        },
        "compile": {"requested": False},
    }

    if args.compile:
        try:
            compiled_core, compile_report = _cold_compile(
                optimized_predict,
                (initial_ids, 1.0, active_positions),
                device=device,
                mode=args.compile_mode,
                dynamic=True,
            )

            def compiled_predict(ids: torch.Tensor, t, positions=None):
                return compiled_core(ids, t, positions)

            with torch.inference_mode():
                compiled_ids = sample(compiled_predict)
                eager_ids = sample(optimized_predict)
                compile_report["token_agreement_with_optimized_eager"] = float(
                    (compiled_ids == eager_ids).float().mean().item()
                )
                del compiled_ids, eager_ids
            compiled_baseline, compiled_metrics, recompiles = measure_pair_cuda(
                lambda: sample(baseline_predict),
                lambda: sample(compiled_predict),
                device=device,
                warmup=args.warmup,
                repeats=args.repeats,
                monitor_compile=True,
            )
            _add_throughput(compiled_metrics, units, "generated_tokens_per_second")
            compile_report.update(
                {
                    "baseline": compiled_baseline,
                    "optimized_compiled": compiled_metrics,
                    "comparison": compare_metrics(compiled_baseline, compiled_metrics),
                    "recompiles_during_timed_region": recompiles,
                    "no_timed_recompiles": recompiles == 0,
                }
            )
            result["compile"] = compile_report
        except Exception as exc:
            result["compile"] = {
                "requested": True,
                "error": f"{type(exc).__name__}: {exc}",
            }
    return result


def _two_pass_cfg(
    denoise_fn,
    x_t: torch.Tensor,
    t: torch.Tensor,
    cond: torch.Tensor,
    cfg_cond: Optional[torch.Tensor],
    x_self_cond: Optional[torch.Tensor],
):
    conditional = denoise_fn(x_t, t, cond, x_self_cond)
    if cfg_cond is None:
        return conditional, None
    batch = x_t.shape[0]
    unconditional = denoise_fn(x_t, t, cfg_cond[batch:], x_self_cond)
    return conditional, unconditional


@contextlib.contextmanager
def _cfg_implementation(helper):
    from dimba.diffusion import sampling

    original = sampling._batched_cfg_denoise
    sampling._batched_cfg_denoise = helper
    try:
        yield
    finally:
        sampling._batched_cfg_denoise = original


def run_continuous(
    model,
    args: argparse.Namespace,
    device: torch.device,
    sampler_name: str,
) -> Dict[str, Any]:
    from dimba.diffusion import sampling

    optimized_helper = sampling._batched_cfg_denoise
    batch = args.inference_batch_size
    generated = args.seq_len - args.prompt_len
    prompt = torch.randint(0, args.vocab_size, (batch, args.prompt_len), device=device)

    def sample_with(helper):
        with _cfg_implementation(helper):
            return sampling.sample_from_model(
                model,
                prompt,
                generated,
                num_steps=args.steps,
                guidance_scale=args.guidance_scale,
                temperature=1.0,
                sampler=sampler_name,
                device=device,
            )

    x_t = torch.randn(batch, args.seq_len, model.d_latent, device=device)
    timestep = torch.full(
        (batch,), model.num_diffusion_steps // 2, dtype=torch.long, device=device
    )
    cond = model.conditioning_from_prompt(prompt, batch, device)
    uncond = model.conditioning_from_prompt(None, batch, device, drop_cond=True)
    cfg_cond = torch.cat((cond, uncond), dim=0)
    x_self_cond = torch.randn_like(x_t)
    with torch.inference_mode():
        reference_cond, reference_uncond = _two_pass_cfg(
            model.denoise_to_x0_latent, x_t, timestep, cond, cfg_cond, x_self_cond
        )
        actual_cond, actual_uncond = optimized_helper(
            model.denoise_to_x0_latent, x_t, timestep, cond, cfg_cond, x_self_cond
        )
        assert reference_uncond is not None and actual_uncond is not None
        conditional_parity = tensor_parity(
            reference_cond, actual_cond, rtol=args.parity_rtol, atol=args.parity_atol
        )
        unconditional_parity = tensor_parity(
            reference_uncond, actual_uncond, rtol=args.parity_rtol, atol=args.parity_atol
        )
        shared_seed = args.seed + 17
        _seed(shared_seed)
        baseline_ids, baseline_final_logits = _run_and_capture_single_output(
            model.output_head,
            lambda: sample_with(_two_pass_cfg),
        )
        _seed(shared_seed)
        optimized_ids, optimized_final_logits = _run_and_capture_single_output(
            model.output_head,
            lambda: sample_with(optimized_helper),
        )
        final_logit_parity = tensor_parity(
            baseline_final_logits,
            optimized_final_logits,
            rtol=args.parity_rtol,
            atol=args.parity_atol,
        )
        output_agreement = token_output_agreement(baseline_ids, optimized_ids)
        valid_ids = bool(
            (optimized_ids >= 0).all()
            and (optimized_ids < args.vocab_size).all()
            and optimized_ids.shape == (batch, generated)
        )
        del baseline_ids, optimized_ids, baseline_final_logits, optimized_final_logits

    baseline_fn = lambda: sample_with(_two_pass_cfg)
    optimized_fn = lambda: sample_with(optimized_helper)
    baseline_metrics, optimized_metrics, _ = measure_pair_cuda(
        baseline_fn,
        optimized_fn,
        device=device,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    units = batch * generated
    _add_throughput(baseline_metrics, units, "generated_tokens_per_second")
    _add_throughput(optimized_metrics, units, "generated_tokens_per_second")
    case_name = f"continuous-{sampler_name}"
    result: Dict[str, Any] = {
        "name": case_name,
        "kind": "end_to_end",
        "baseline_variant": "two_sequential_cfg_denoiser_dispatches_per_step",
        "optimized_variant": "one_2B_batched_cfg_denoiser_dispatch_per_step",
        "shape": {
            "batch": batch,
            "prompt": args.prompt_len,
            "generated": generated,
            "total_length": args.seq_len,
            "latent_width": model.d_latent,
            "steps": args.steps,
            "sampler": sampler_name,
            "baseline_dispatches": 2 * args.steps,
            "optimized_dispatches": args.steps,
            "effective_nfe_both": 2 * args.steps,
        },
        "baseline": baseline_metrics,
        "optimized": optimized_metrics,
        "comparison": compare_metrics(baseline_metrics, optimized_metrics),
        "correctness": {
            "pass": (
                conditional_parity["pass"]
                and unconditional_parity["pass"]
                and final_logit_parity["pass"]
                and valid_ids
            ),
            "conditional_step_parity": conditional_parity,
            "unconditional_step_parity": unconditional_parity,
            "seeded_final_response_logit_parity": final_logit_parity,
            "seeded_full_generation_output_agreement": output_agreement,
            "seeded_full_generation_token_agreement": output_agreement["token_agreement"],
            "shared_rng_seed": shared_seed,
            "shared_rng_scope": "initial Gaussian noise and final multinomial draw",
            "optimized_ids_valid": valid_ids,
            "accuracy_gate": (
                "mid-trajectory CFG and seeded final response logits match within tolerance"
            ),
        },
        "compile": {"requested": False},
    }

    if args.compile:
        try:
            doubled_x = torch.cat((x_t, x_t), dim=0)
            doubled_t = torch.cat((timestep, timestep), dim=0)
            doubled_self_cond = torch.cat((x_self_cond, x_self_cond), dim=0)
            compiled_denoise, compile_report = _cold_compile(
                model.denoise_to_x0_latent,
                (doubled_x, doubled_t, cfg_cond, doubled_self_cond),
                device=device,
                mode=args.compile_mode,
                dynamic=False,
            )

            def compiled_helper(_denoise_fn, xt, ts, conditioning, both_conditioning, self_cond):
                return optimized_helper(
                    compiled_denoise, xt, ts, conditioning, both_conditioning, self_cond
                )

            _seed(args.seed + 19)
            compiled_ids = sample_with(compiled_helper)
            _seed(args.seed + 19)
            eager_ids = sample_with(optimized_helper)
            compile_report["token_agreement_with_optimized_eager"] = float(
                (compiled_ids == eager_ids).float().mean().item()
            )
            del compiled_ids, eager_ids
            compiled_baseline, compiled_metrics, recompiles = measure_pair_cuda(
                baseline_fn,
                lambda: sample_with(compiled_helper),
                device=device,
                warmup=args.warmup,
                repeats=args.repeats,
                monitor_compile=True,
            )
            _add_throughput(compiled_metrics, units, "generated_tokens_per_second")
            compile_report.update(
                {
                    "baseline": compiled_baseline,
                    "optimized_compiled": compiled_metrics,
                    "comparison": compare_metrics(compiled_baseline, compiled_metrics),
                    "recompiles_during_timed_region": recompiles,
                    "no_timed_recompiles": recompiles == 0,
                }
            )
            result["compile"] = compile_report
        except Exception as exc:
            result["compile"] = {
                "requested": True,
                "error": f"{type(exc).__name__}: {exc}",
            }
    return result


def run_continuous_fused_ce(
    model,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, Any]:
    """Benchmark exact uniform CE on a real DIMBA projection head."""
    from dimba.training.fused_ce import (
        liger_fused_ce_available,
        output_head_cross_entropy,
    )

    if not liger_fused_ce_available():
        raise RuntimeError(
            "continuous-fused-ce requires an importable liger-kernel CUDA build"
        )

    batch, length = args.training_batch_size, args.seq_len
    head = model.output_head
    embedding_weight = model.token_embed.get_weight()
    if head.use_weight_tying:
        weight, bias = embedding_weight, None
    elif isinstance(head.projection, torch.nn.Linear):
        weight, bias = head.projection.weight, head.projection.bias
    else:
        raise TypeError("continuous-fused-ce requires a tied or nn.Linear DIMBA head")
    width, vocab = weight.shape[1], weight.shape[0]
    features = (
        torch.randn(batch, length, width, device=device, dtype=dtype) / width**0.5
    ).requires_grad_(True)
    targets = torch.randint(0, vocab, (batch, length), device=device)

    def zero_grads() -> None:
        features.grad = None
        model.zero_grad(set_to_none=True)

    def baseline() -> torch.Tensor:
        zero_grads()
        logits = F.linear(features.flatten(0, 1), weight, bias)
        if head.use_norm:
            logits = logits * head.logit_scale.exp()
        loss = F.cross_entropy(logits, targets.flatten())
        loss.backward()
        return loss.detach()

    def optimized() -> torch.Tensor:
        zero_grads()
        loss = output_head_cross_entropy(
            model,
            features,
            targets,
            prepared=True,
            reduction="mean",
            mode="on",
            uniform_reduction=True,
        )
        loss.backward()
        return loss.detach()

    reference_loss = baseline()
    reference_feature_grad = features.grad.detach().clone()
    reference_weight_grad = weight.grad.detach().clone()
    reference_bias_grad = None if bias is None else bias.grad.detach().clone()
    actual_loss = optimized()
    loss_parity = tensor_parity(
        reference_loss[None], actual_loss[None], rtol=args.parity_rtol, atol=args.parity_atol
    )
    feature_grad_parity = tensor_parity(
        reference_feature_grad,
        features.grad,
        rtol=args.parity_rtol,
        atol=args.parity_atol,
    )
    weight_grad_parity = tensor_parity(
        reference_weight_grad,
        weight.grad,
        rtol=args.parity_rtol,
        atol=args.parity_atol,
    )
    bias_grad_parity = None
    if bias is not None:
        bias_grad_parity = tensor_parity(
            reference_bias_grad,
            bias.grad,
            rtol=args.parity_rtol,
            atol=args.parity_atol,
        )
    del reference_feature_grad, reference_weight_grad, reference_bias_grad
    del reference_loss, actual_loss

    baseline_metrics, optimized_metrics, _ = measure_pair_cuda(
        baseline,
        optimized,
        device=device,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    tokens = batch * length
    _add_throughput(baseline_metrics, tokens, "tokens_per_second")
    _add_throughput(optimized_metrics, tokens, "tokens_per_second")
    parity_checks = [loss_parity, feature_grad_parity, weight_grad_parity]
    if bias_grad_parity is not None:
        parity_checks.append(bias_grad_parity)
    return {
        "name": FUSED_CE_CASE,
        "kind": "microkernel",
        "baseline_variant": "materialized_F_linear_logits_plus_uniform_CE_backward",
        "optimized_variant": "Liger_fused_linear_uniform_CE_backward",
        "shape": {
            "batch": batch,
            "seq_len": length,
            "tokens": tokens,
            "hidden_width": width,
            "vocab": vocab,
            "tied_weight": head.use_weight_tying,
            "bias": bias is not None,
            "full_logits_bytes": tokens * vocab * torch.empty((), dtype=dtype).element_size(),
            "reduction": "mean",
        },
        "baseline": baseline_metrics,
        "optimized": optimized_metrics,
        "comparison": compare_metrics(baseline_metrics, optimized_metrics),
        "correctness": {
            "pass": all(check["pass"] for check in parity_checks),
            "loss_parity": loss_parity,
            "feature_gradient_parity": feature_grad_parity,
            "vocab_weight_gradient_parity": weight_grad_parity,
            "bias_gradient_parity": bias_grad_parity,
            "semantic_scope": "uniform mean CE only; weighted and masked CE excluded",
        },
        "liger": {
            "version": _package_version("liger-kernel"),
            "mode": "on",
        },
    }


def run_masked_training(
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, Any]:
    batch, length, width, vocab = (
        args.training_batch_size,
        args.seq_len,
        args.d_model,
        args.vocab_size,
    )
    masked_per_sample = max(1, round(length * args.masked_fraction))
    features = (torch.randn(batch, length, width, device=device, dtype=dtype) / width**0.5)
    features.requires_grad_(True)
    weight = (torch.randn(vocab, width, device=device, dtype=dtype) / width**0.5)
    weight.requires_grad_(True)
    targets = torch.randint(0, vocab, (batch, length), device=device)
    target_mask = torch.arange(length, device=device)[None, :] < masked_per_sample
    target_mask = target_mask.expand(batch, -1)
    selected_targets = targets[target_mask]

    def zero_grads() -> None:
        if features.grad is not None:
            features.grad.zero_()
        if weight.grad is not None:
            weight.grad.zero_()

    def baseline() -> torch.Tensor:
        zero_grads()
        full_logits = F.linear(features, weight)
        loss = F.cross_entropy(full_logits[target_mask], selected_targets)
        loss.backward()
        return loss.detach()

    def optimized() -> torch.Tensor:
        zero_grads()
        selected_logits = F.linear(features[target_mask], weight)
        loss = F.cross_entropy(selected_logits, selected_targets)
        loss.backward()
        return loss.detach()

    reference_loss = baseline()
    reference_feature_grad = features.grad.detach().clone()
    reference_weight_grad = weight.grad.detach().clone()
    actual_loss = optimized()
    loss_parity = tensor_parity(
        reference_loss[None], actual_loss[None], rtol=args.parity_rtol, atol=args.parity_atol
    )
    feature_grad_parity = tensor_parity(
        reference_feature_grad,
        features.grad,
        rtol=args.parity_rtol,
        atol=args.parity_atol,
    )
    weight_grad_parity = tensor_parity(
        reference_weight_grad,
        weight.grad,
        rtol=args.parity_rtol,
        atol=args.parity_atol,
    )
    del reference_feature_grad, reference_weight_grad, reference_loss, actual_loss

    baseline_metrics, optimized_metrics, _ = measure_pair_cuda(
        baseline,
        optimized,
        device=device,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    units = batch * masked_per_sample
    _add_throughput(baseline_metrics, units, "masked_tokens_per_second")
    _add_throughput(optimized_metrics, units, "masked_tokens_per_second")
    return {
        "name": "masked-training",
        "kind": "microkernel_seam",
        "baseline_variant": "full_B_L_V_projection_then_masked_CE_backward",
        "optimized_variant": "select_hidden_states_then_projection_CE_backward",
        "shape": {
            "batch": batch,
            "length": length,
            "hidden": width,
            "vocab": vocab,
            "masked_fraction": args.masked_fraction,
            "masked_tokens": units,
            "full_logits_bytes": batch * length * vocab * 2,
        },
        "baseline": baseline_metrics,
        "optimized": optimized_metrics,
        "comparison": compare_metrics(baseline_metrics, optimized_metrics),
        "correctness": {
            "pass": (
                loss_parity["pass"]
                and feature_grad_parity["pass"]
                and weight_grad_parity["pass"]
            ),
            "loss_parity": loss_parity,
            "hidden_gradient_parity": feature_grad_parity,
            "vocab_weight_gradient_parity": weight_grad_parity,
            "accuracy_gate": "loss and both gradients match within dtype tolerance",
        },
        "compile": {
            "requested": args.compile,
            "skipped": "training compile is not mixed into the projection/CE seam benchmark",
        },
    }


def _is_oom(exc: BaseException) -> bool:
    oom_type = getattr(torch.cuda, "OutOfMemoryError", ())
    return isinstance(exc, oom_type) or "out of memory" in str(exc).lower()


def _render(report: Dict[str, Any], output: Optional[Path]) -> None:
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    print(payload)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        validate_args(args)
    except ValueError as exc:
        print(f"invalid benchmark configuration: {exc}", file=sys.stderr)
        return 2

    if args.list_cases:
        print(json.dumps(CASES, indent=2))
        return 0
    if args.dry_run:
        _render({"dry_run": True, "config": planned_config(args)}, args.output)
        return 0
    if not torch.cuda.is_available():
        print(
            "CUDA is required. Use --dry-run or --list-cases on this machine; "
            "run measurements on the target H100 host.",
            file=sys.stderr,
        )
        return 1

    device = torch.device("cuda", args.device_index)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    env = environment_info(device)
    is_h100 = env["device"]["is_h100"]
    if not is_h100 and not args.allow_non_h100:
        print(
            f"Expected an H100, found {env['device']['name']!r}; pass --allow-non-h100 "
            "only for diagnostic, non-promotable measurements.",
            file=sys.stderr,
        )
        return 1

    _seed(args.seed)
    inference_cases = [case for case in args.cases if case in END_TO_END_CASES]
    model_cases = [case for case in args.cases if case in MODEL_CASES]
    model = None
    model_source = {"kind": "not_needed"}
    backend = {"implementations": [], "all_fast_cuda_mamba2": True}
    if model_cases:
        try:
            model, model_source = build_model(args, device, dtype)
        except Exception as exc:
            print(
                f"Failed to build/load benchmark model: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return 1
        backend = mamba_backend_info(model)
        if not backend["all_fast_cuda_mamba2"] and not args.allow_torch_mamba2:
            print(
                "The benchmark model is not using mamba_ssm.Mamba2 CUDA kernels: "
                f"{backend['implementations']}. Install compatible mamba-ssm and "
                "causal-conv1d builds, or pass --allow-torch-mamba2 for a "
                "non-promotable diagnostic.",
                file=sys.stderr,
            )
            return 1
        if not env["causal_conv1d_importable"] and not args.allow_torch_mamba2:
            print(
                "causal_conv1d is unavailable, so the production Mamba2 convolution fast "
                "path cannot be validated. Install causal-conv1d, or pass "
                "--allow-torch-mamba2 for a non-promotable diagnostic.",
                file=sys.stderr,
            )
            return 1
    env["model_mamba_backend"] = backend
    env["model_source"] = model_source
    if model is not None:
        env["model_config"] = model.config
        env["model_parameters"] = sum(parameter.numel() for parameter in model.parameters())

    results = []
    for case in args.cases:
        try:
            if case == "masked-inference":
                assert model is not None
                result = run_masked_inference(model, args, device)
            elif case == "continuous-ddim":
                assert model is not None
                result = run_continuous(model, args, device, "ddim")
            elif case == "continuous-dpmpp":
                assert model is not None
                result = run_continuous(model, args, device, "dpmpp")
            elif case == FUSED_CE_CASE:
                assert model is not None
                result = run_continuous_fused_ce(model, args, device, dtype)
            else:
                result = run_masked_training(args, device, dtype)
            results.append(result)
        except Exception as exc:
            if _is_oom(exc):
                error = f"CUDA OOM at the requested shape: {exc}"
            else:
                error = f"{type(exc).__name__}: {exc}"
            results.append({"name": case, "error": error})
            gc.collect()
            torch.cuda.empty_cache()

    environment_ok = bool(
        is_h100
        and (not inference_cases or backend["all_fast_cuda_mamba2"])
        and (not inference_cases or env["causal_conv1d_importable"])
    )
    gate = promotion_decision(
        results,
        environment_ok=environment_ok,
        quality_context_ok=args.checkpoint is not None,
        requested_cases=args.cases,
        compile_requested=args.compile,
    )
    report = {
        "schema_version": 1,
        "benchmark": "dimba-h100-performance-gates",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": env,
        "config": planned_config(args),
        "results": results,
        "promotion_gate": gate,
        "notes": [
            "CUDA-event p50/p95 excludes cold compile; compile first-call latency is separate.",
            "Runs requested with --compile are diagnostic-only and cannot be promoted.",
            "Peak HBM is measured separately from interleaved AB/BA timing.",
            "Uniform Liger fused-linear CE is timed and parity-gated; weighted/masked CE is not.",
            "No custom kernel is promoted unless the complete suite passes the recorded rule.",
            (
                "Random synthetic weights are useful for throughput only; promotion requires "
                "--checkpoint."
            ),
        ],
    }
    _render(report, args.output)
    if any("error" in result for result in results):
        return 2
    if args.fail_on_gate and not gate["promote"]:
        return 3
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
