#!/usr/bin/env python3
"""Benchmark script for DIMBA inference on CPU.

Builds a tiny DIMBA model and measures generation performance across a few
denoising-step settings. Reports parameter count, generation latency,
tokens/sec, NFE (number of network forward evaluations), and CPU wall-time
per denoising step.

The defaults are intentionally tiny so the benchmark completes in seconds on
CPU with no GPU, no compiled kernels (uses the pure-PyTorch ``SimpleMamba2``),
and no optional dependencies.

Usage:
    # Run with default tiny config
    python scripts/benchmark.py

    # Customize the model / sweep
    python scripts/benchmark.py --d-model 128 --seq-len 32 --num-steps 5 10 20

    # Increase the number of timed repeats for more stable numbers
    python scripts/benchmark.py --repeats 5 --warmup 1
"""

import argparse
import contextlib
import io
import sys
import time
from pathlib import Path
from typing import List, Optional

import torch

# Add src to path (src-layout) so ``import dimba`` works when run directly.
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = (SCRIPT_DIR / ".." / "src").resolve()
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def build_model(
    vocab_size: int,
    d_model: int,
    num_denoiser_layers: int,
    num_diffusion_steps: int,
) -> torch.nn.Module:
    """Build a tiny DIMBA model defensively.

    Args:
        vocab_size: Vocabulary size.
        d_model: Hidden dimension.
        num_denoiser_layers: Number of denoiser layers.
        num_diffusion_steps: Total diffusion steps (T).

    Returns:
        An initialized, eval-mode DIMBA model on CPU.

    Raises:
        SystemExit: If the model cannot be constructed, with a helpful message.
    """
    try:
        from dimba.models.diffusion import DIMBA
    except Exception as exc:  # noqa: BLE001 - want a friendly message for any failure
        raise SystemExit(
            "Failed to import DIMBA from 'dimba.models.diffusion'.\n"
            f"  Underlying error: {type(exc).__name__}: {exc}\n"
            "  Make sure you run this from the repo root and that the 'src/' "
            "layout is intact (the script adds 'src/' to sys.path automatically)."
        )

    try:
        model = DIMBA(
            vocab_size=vocab_size,
            d_model=d_model,
            d_prompt=d_model,
            num_diffusion_steps=num_diffusion_steps,
            num_denoiser_layers=num_denoiser_layers,
            use_simple_mamba=True,  # pure-PyTorch SSM: no CUDA / compilation needed
        )
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            "Failed to construct the DIMBA model.\n"
            f"  Underlying error: {type(exc).__name__}: {exc}\n"
            "  The model API may have changed during refactoring. Try adjusting "
            "the constructor arguments in scripts/benchmark.py:build_model()."
        )

    model.eval()
    return model


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """Return (total, trainable) parameter counts for ``model``."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def time_generation(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    seq_len: int,
    num_steps: int,
    repeats: int,
    warmup: int,
) -> float:
    """Time one full generation and return the best wall-time in seconds.

    Args:
        model: DIMBA model.
        prompt_ids: Prompt token IDs ``[batch, prompt_len]``.
        seq_len: Number of tokens to generate.
        num_steps: Number of denoising steps for this run.
        repeats: Number of timed repeats (the minimum is returned).
        warmup: Number of untimed warmup runs.

    Returns:
        Best (minimum) wall-clock time in seconds across ``repeats`` runs.
    """
    from dimba.diffusion.sampling import sample_from_model

    def _one() -> None:
        # The sampler prints per-step progress; silence it so the table stays clean.
        with contextlib.redirect_stdout(io.StringIO()):
            sample_from_model(
                model,
                prompt_ids,
                seq_len=seq_len,
                num_steps=num_steps,
                device=torch.device("cpu"),
            )

    with torch.no_grad():
        for _ in range(max(0, warmup)):
            _one()

        best = float("inf")
        for _ in range(max(1, repeats)):
            start = time.perf_counter()
            _one()
            best = min(best, time.perf_counter() - start)
    return best


def _fmt(value: float, width: int) -> str:
    """Right-align a formatted float in a fixed-width column."""
    return f"{value:>{width}.3f}"


def print_table(rows: List[dict], batch_size: int, seq_len: int) -> None:
    """Print a clean fixed-width results table.

    Args:
        rows: One dict per ``num_steps`` setting with measured metrics.
        batch_size: Batch size used for generation.
        seq_len: Sequence length generated per sample.
    """
    header = f"{'steps':>6} | {'NFE':>5} | {'latency(s)':>11} | {'ms/step':>9} | {'tokens/s':>10}"
    sep = "-" * len(header)
    print(sep)
    print(
        f"Batch size: {batch_size}   Seq len: {seq_len}   "
        f"Tokens/sample: {seq_len}   Total tokens/run: {batch_size * seq_len}"
    )
    print(sep)
    print(header)
    print(sep)
    for row in rows:
        print(
            f"{row['num_steps']:>6} | "
            f"{row['nfe']:>5} | "
            f"{_fmt(row['latency_s'], 11)} | "
            f"{_fmt(row['ms_per_step'], 9)} | "
            f"{_fmt(row['tokens_per_sec'], 10)}"
        )
    print(sep)
    print(
        "Notes: NFE = network forward evaluations (one denoiser call per step). "
        "ms/step = CPU wall-time per denoising step. Lower latency is better."
    )
    print(sep)


def run_benchmark(args: argparse.Namespace) -> List[dict]:
    """Build the model, run the sweep, and return the collected rows."""
    torch.manual_seed(args.seed)

    model = build_model(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_denoiser_layers=args.num_denoiser_layers,
        num_diffusion_steps=args.num_diffusion_steps,
    )

    total_params, trainable_params = count_parameters(model)

    print("=" * 64)
    print("DIMBA CPU Benchmark")
    print("=" * 64)
    print(f"torch version      : {torch.__version__}")
    print("device             : cpu")
    print(f"vocab_size         : {args.vocab_size}")
    print(f"d_model            : {args.d_model}")
    print(f"num_denoiser_layers: {args.num_denoiser_layers}")
    print(f"num_diffusion_steps: {args.num_diffusion_steps}  (model T)")
    print(f"seq_len            : {args.seq_len}")
    print(f"batch_size         : {args.batch_size}")
    print(f"total params       : {total_params:,}")
    print(f"trainable params   : {trainable_params:,}")
    print(f"timed repeats      : {args.repeats}  (warmup: {args.warmup})")
    print()

    # Build a small random prompt; generation pads/extends to seq_len internally.
    prompt_len = max(1, min(args.prompt_len, args.seq_len))
    prompt_ids = torch.randint(0, args.vocab_size, (args.batch_size, prompt_len))

    rows: List[dict] = []
    total_tokens = args.batch_size * args.seq_len
    for num_steps in args.num_steps:
        latency = time_generation(
            model,
            prompt_ids,
            seq_len=args.seq_len,
            num_steps=num_steps,
            repeats=args.repeats,
            warmup=args.warmup,
        )
        rows.append(
            {
                "num_steps": num_steps,
                # One denoiser forward eval per denoising step, per run.
                "nfe": num_steps,
                "latency_s": latency,
                "ms_per_step": (latency / num_steps) * 1000.0,
                "tokens_per_sec": total_tokens / latency if latency > 0 else float("inf"),
            }
        )

    print_table(rows, batch_size=args.batch_size, seq_len=args.seq_len)
    return rows


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark tiny DIMBA inference on CPU (finishes in seconds).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--vocab-size", type=int, default=256, help="Vocabulary size.")
    parser.add_argument("--d-model", type=int, default=64, help="Hidden dimension.")
    parser.add_argument(
        "--num-denoiser-layers", type=int, default=2, help="Number of denoiser layers."
    )
    parser.add_argument(
        "--num-diffusion-steps",
        type=int,
        default=10,
        help="Total diffusion steps T the model is built with.",
    )
    parser.add_argument(
        "--seq-len", type=int, default=16, help="Number of tokens to generate per sample."
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Generation batch size.")
    parser.add_argument("--prompt-len", type=int, default=4, help="Length of the random prompt.")
    parser.add_argument(
        "--num-steps",
        type=int,
        nargs="+",
        default=[2, 5, 10],
        help="Denoising-step counts to sweep over.",
    )
    parser.add_argument(
        "--repeats", type=int, default=3, help="Timed repeats per setting (min is reported)."
    )
    parser.add_argument("--warmup", type=int, default=1, help="Untimed warmup runs per setting.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    # Keep CPU thread count modest so the benchmark is reproducible and quick.
    try:
        torch.set_num_threads(max(1, torch.get_num_threads()))
    except Exception:  # noqa: BLE001 - non-fatal
        pass
    run_benchmark(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
