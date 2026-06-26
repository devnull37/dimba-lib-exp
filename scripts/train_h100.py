#!/usr/bin/env python3
"""DIMBA training entry point optimised for a single NVIDIA H100 80 GB.

This script is a thin wrapper around scripts/train_4090.py. It imports the
full distill → SFT → GRPO pipeline from that module and overrides the
module-level config globals for H100 80 GB before calling any run_* function.

What changes vs train_4090.py
------------------------------
  * PRETRAIN_BATCH 32 → 128  (Stage-3 co-adaptation)
  * _STAGE3_BATCH  32 → 128  (kept in sync so _steps_for_tokens is correct)
  * ALIGN_BATCH     4 → 16   (Stage-1/2 alignment; mixing-matrix O(B·nh·L²·nl))
  * SFT batch_size 32 → 64   (grad_accum 4 → 2; effective batch stays 128)
  * --preset flag maps to STAGE3_FROZEN/UNFROZEN_TOKENS per NEXT_RUN_PLAN.md

VRAM budget reasoning (H100 80 GB, 135M model)
------------------------------------------------
  Model (bf16):           ~270 MB   (135M × 2 bytes)
  AdamW fp32 states:     ~1.6 GB   (135M × 12 bytes; momentum + variance + master)
  Stage-3 vocab logits:  ~B×50 MB  (B × 512 × 49152 × 2 bytes)
    At B=128:  6.4 GB
  Activations (30 layers, Mamba CUDA kernel, rough):
    At B=128:  ~2–4 GB   (B × L × d_model × 2B × num_layers × fwd/bwd)
  Estimated total at B=128: ~10–12 GB → well within 80 GB with ~65 GB headroom.

  The SFT heavy case (vocab logits + fragmentation spikes):
    The 48 GB GPU OOM'd at B=64 for SFT at the epoch boundary (fragmentation).
    B=64 on 48 GB → ~49 GB peak.  On 80 GB the same B=64 has ~31 GB headroom.
    We keep effective batch = 128 via grad_accum=2.

  The Stage-3 OOM-retry logic (_run_stage3_resilient in train_4090.py) halves
  the batch and retries on any torch.cuda.OutOfMemoryError, so B=128 is safe to
  attempt even if the estimate is off.

Staged ladder — example commands
----------------------------------
  # S0 smoke: plumbing only (~135M tokens, <1 GPU-h)
  python scripts/train_h100.py --preset smoke

  # S1 validation: check recipe vs run #1 (~1B tokens, ~2–3 GPU-h on H100)
  python scripts/train_h100.py --preset validation

  # S2 scale-test / decision gate (~5B tokens, ~8–12 GPU-h on H100)
  python scripts/train_h100.py --preset scale

  # S3 full run — only after S2 GO gate (~30B tokens)
  python scripts/train_h100.py --preset full

  # Single phase:
  python scripts/train_h100.py --preset scale --phase distill
  python scripts/train_h100.py --preset scale --phase sft   --checkpoint checkpoints/distill/final.pt
  python scripts/train_h100.py --preset scale --phase grpo  --checkpoint checkpoints/sft/final.pt

  # Dry-run: print resolved config and exit without training:
  python scripts/train_h100.py --preset scale --dry-run

Preset → token budget mapping (from docs/NEXT_RUN_PLAN.md)
-------------------------------------------------------------
  smoke      : 100 M frozen + 35 M unfrozen ≈ 135 M total  (S0 plumbing gate)
  validation : 700 M frozen + 300 M unfrozen ≈ 1 B total   (S1 recipe check)
  scale      : 3 B frozen  + 2 B unfrozen  ≈ 5 B total     (S2 decision gate, default)
  full       : 20 B frozen + 10 B unfrozen ≈ 30 B total    (S3 post-gate, 67/33 split)
               (lower end of NEXT_RUN_PLAN.md's 20–50 B target; bump if gates pass cleanly)
"""
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import torch

# ── path setup ────────────────────────────────────────────────────────────────
# Both src/ (for dimba.*) and scripts/ (for train_4090) must be importable.
_SCRIPTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_SCRIPTS_DIR))

import train_4090 as _t4
from train_4090 import run_distill, run_sft, run_grpo

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_h100")

# ── H100 80 GB batch sizes ────────────────────────────────────────────────────
# See module docstring for the VRAM reasoning behind each choice.
#
# Stage-3 co-adaptation (distillation pretraining, CUDA Mamba2 kernel):
#   B=128 → estimated peak ~10–12 GB. OOM-retry halves automatically on OOM.
H100_PRETRAIN_BATCH   = 128   # vs 32 on 48 GB GPU
H100_PRETRAIN_SEQ_LEN = 512   # unchanged; L=512 is the correct context window
H100_STAGE3_BATCH     = 128   # must equal H100_PRETRAIN_BATCH (_steps_for_tokens uses this)
H100_STAGE3_SEQ       = 512   # must equal H100_PRETRAIN_SEQ_LEN

# Stage-1/2 alignment (mixing-matrix memory is O(B × nheads × L² × num_layers)):
#   4090 used B=4 for ~0.1–1 GB of matrix storage.  H100 has 1.67× more VRAM and
#   alignment is only 500 steps each → B=16 (4×) gives modest throughput gain with
#   very conservative matrix storage (~4 GB at B=16, L=256, 8 heads, 30 layers).
H100_ALIGN_BATCH   = 16   # vs 4 on 48 GB GPU
H100_ALIGN_SEQ_LEN = 256  # unchanged; limited by the L² mixing-matrix term

# SFT (heaviest phase: vocab logits B × L × 49152 dominate VRAM):
#   4090 OOM'd at B=64 SFT (~49 GB peak); B=32 was safe (~24 GB estimate).
#   H100: B=64 → same ~49 GB peak, now with 31 GB headroom.
#   Effective batch kept at 128 (same as 4090's 32 × grad_accum=4).
H100_SFT_BATCH_SIZE = 64  # vs 32 on 48 GB GPU
H100_SFT_GRAD_ACCUM = 2   # 64 × 2 = 128 effective (unchanged vs 4090)

# ── staged token-budget presets (docs/NEXT_RUN_PLAN.md §3) ───────────────────
# `subset` picks the FineWeb config so each preset draws GENUINELY UNIQUE tokens
# (no looping/repetition — the data-starvation failure mode of run #1). sample-10BT
# (~10B) covers everything up to the 5B scale gate; the 30B full run needs a bigger
# pool, so it uses sample-100BT (~100B unique). apply_h100_overrides sets
# train_4090.PRETRAIN_SUBSET from this.
_PRESETS: dict[str, dict] = {
    "smoke": {
        "frozen":   100_000_000,   # 100 M
        "unfrozen":  35_000_000,   #  35 M  → ~135 M total (S0 plumbing gate)
        "subset":   "sample-10BT",
        "desc": "~135M tokens — plumbing only; not a quality gate",
    },
    "validation": {
        "frozen":   700_000_000,   # 700 M
        "unfrozen": 300_000_000,   # 300 M  → ~1 B total (S1 recipe check)
        "subset":   "sample-10BT",
        "desc": "~1B tokens — validates recipe vs run #1",
    },
    "scale": {
        "frozen":   3_000_000_000,  # 3 B
        "unfrozen": 2_000_000_000,  # 2 B   → ~5 B total (S2 decision gate)
        "subset":   "sample-10BT",  # 5B < 10B → unique, no looping
        "desc": "~5B tokens — S2 scale-test / GO·NO-GO decision gate",
    },
    "full": {
        "frozen":   20_000_000_000,  # 20 B
        "unfrozen": 10_000_000_000,  # 10 B  → ~30 B total (S3, 67/33 split)
        "subset":   "sample-100BT",  # 30B unique needs >10B → 100BT pool
        "desc": "~30B tokens — full run after S2 GO gate (lower end of 20–50B target)",
    },
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _h100_steps_for_tokens(n_tokens: int) -> int:
    """Tokens → optimiser steps using the H100 Stage-3 batch geometry.

    Implemented independently of train_4090._steps_for_tokens so this file
    does not depend on that private helper (another agent may refactor it).
    """
    return max(1, round(n_tokens / (H100_STAGE3_BATCH * H100_STAGE3_SEQ)))


def _build_stage3_phases(
    frozen_tokens: int,
    unfrozen_tokens: int,
    frozen_lr: float,
    unfrozen_lr: float,
) -> list:
    """Build Stage-3 phase dicts in the schema expected by DistillationTrainer.

    Mirrors train_4090._coadapt_stages() but uses H100 batch sizes for step
    counts and takes explicit token budgets so the preset is the only knob.
    A phase is omitted when its token budget is zero.
    """
    phases = []
    if frozen_tokens > 0:
        phases.append({
            "name":           "stage3",
            "steps":          _h100_steps_for_tokens(frozen_tokens),
            "lr":             frozen_lr,
            "freeze_ffn":     True,
            "kd_weight":      0.0,
            "ce_loss_weight": 1.0,
            "min_snr_gamma":  5.0,
        })
    if unfrozen_tokens > 0:
        phases.append({
            "name":           "stage3",
            "steps":          _h100_steps_for_tokens(unfrozen_tokens),
            "lr":             unfrozen_lr,
            "freeze_ffn":     False,
            "kd_weight":      0.0,
            "ce_loss_weight": 1.0,
            "min_snr_gamma":  5.0,
        })
    return phases


def apply_h100_overrides(preset: str) -> None:
    """Mutate train_4090 module-level globals for H100 80 GB + the chosen preset.

    All run_* functions in train_4090 read their config from module globals at
    call time (not at definition time), so mutations here take effect for every
    subsequent run_distill / run_sft / run_grpo call.

    Special case: DISTILL_CFG['stages'] was materialised at import time via
    train_4090._coadapt_stages() using the 4090 token budgets.  We rebuild the
    stage-3 entries here using H100 batch sizes + the preset's token budgets and
    splice them back in, preserving the stage-1 and stage-2 alignment entries.
    """
    budget        = _PRESETS[preset]
    frozen_tokens = budget["frozen"]
    unfrozen_tok  = budget["unfrozen"]
    frozen_lr     = _t4.STAGE3_FROZEN_LR    # keep train_4090's LR defaults
    unfrozen_lr   = _t4.STAGE3_UNFROZEN_LR

    # ── Stage-3 / pretraining geometry ───────────────────────────────────────
    _t4.PRETRAIN_BATCH    = H100_PRETRAIN_BATCH
    _t4.PRETRAIN_SEQ_LEN  = H100_PRETRAIN_SEQ_LEN
    _t4._STAGE3_BATCH     = H100_STAGE3_BATCH
    _t4._STAGE3_SEQ       = H100_STAGE3_SEQ

    # ── Token budgets ─────────────────────────────────────────────────────────
    _t4.STAGE3_FROZEN_TOKENS   = frozen_tokens
    _t4.STAGE3_UNFROZEN_TOKENS = unfrozen_tok

    # ── FineWeb subset (unique-token pool sized to the budget) ────────────────
    # Stage-3 streams from this config; a budget larger than the pool would loop
    # the same docs (run #1's data-starvation failure). Pick a pool ≥ the budget.
    _t4.PRETRAIN_SUBSET = budget["subset"]

    # ── Alignment (Stage 1 + 2) ───────────────────────────────────────────────
    _t4.ALIGN_BATCH   = H100_ALIGN_BATCH
    _t4.ALIGN_SEQ_LEN = H100_ALIGN_SEQ_LEN

    # ── SFT (both stages) ─────────────────────────────────────────────────────
    # Preserve all other SFT knobs (lr, epochs, warmup, …) from train_4090.
    _t4.SFT_CFG = {
        **_t4.SFT_CFG,
        "batch_size": H100_SFT_BATCH_SIZE,
        "grad_accum": H100_SFT_GRAD_ACCUM,
    }
    _t4.SFT_STAGE2_CFG = {
        **_t4.SFT_STAGE2_CFG,
        "batch_size": H100_SFT_BATCH_SIZE,
        "grad_accum": H100_SFT_GRAD_ACCUM,
    }

    # ── Rebuild DISTILL_CFG['stages'] ────────────────────────────────────────
    # Keep the stage-1 and stage-2 alignment phases unchanged (their short seq /
    # small batch are driven by mixing-matrix memory, not VRAM capacity per se).
    # Replace the stage-3 entries with ones computed from H100 batch sizes.
    align_phases  = [s for s in _t4.DISTILL_CFG["stages"]
                     if s["name"] in ("stage1", "stage2")]
    stage3_phases = _build_stage3_phases(frozen_tokens, unfrozen_tok,
                                         frozen_lr, unfrozen_lr)
    _t4.DISTILL_CFG["stages"] = align_phases + stage3_phases

    logger.info(
        "H100 overrides applied | preset=%s | %s",
        preset, budget["desc"],
    )
    logger.info(
        "  Stage-3  batch=%d  seq=%d  subset=%s  |  "
        "frozen=%s tok → %s steps  |  unfrozen=%s tok → %s steps",
        H100_STAGE3_BATCH, H100_STAGE3_SEQ, budget["subset"],
        f"{frozen_tokens:,}", f"{_h100_steps_for_tokens(frozen_tokens):,}",
        f"{unfrozen_tok:,}", f"{_h100_steps_for_tokens(unfrozen_tok):,}",
    )
    logger.info(
        "  Align    batch=%d  seq=%d",
        H100_ALIGN_BATCH, H100_ALIGN_SEQ_LEN,
    )
    logger.info(
        "  SFT      batch=%d  grad_accum=%d  eff_batch=%d",
        H100_SFT_BATCH_SIZE, H100_SFT_GRAD_ACCUM,
        H100_SFT_BATCH_SIZE * H100_SFT_GRAD_ACCUM,
    )


def _print_dry_run(preset: str) -> None:
    """Print resolved config for all presets (or just the selected one) and exit."""
    print(f"\n{'─' * 70}")
    print(f"  train_h100.py — dry-run config check  (preset={preset})")
    print(f"{'─' * 70}")
    print(f"  H100_PRETRAIN_BATCH   = {H100_PRETRAIN_BATCH}")
    print(f"  H100_ALIGN_BATCH      = {H100_ALIGN_BATCH}")
    print(f"  H100_SFT_BATCH_SIZE   = {H100_SFT_BATCH_SIZE}")
    print(f"  H100_SFT_GRAD_ACCUM   = {H100_SFT_GRAD_ACCUM}")
    print(f"  SFT effective batch   = {H100_SFT_BATCH_SIZE * H100_SFT_GRAD_ACCUM}")
    print()
    print(f"  {'Preset':<12}  {'Frozen tokens':>18}  {'Unfrozen tokens':>18}  "
          f"{'Frozen steps':>14}  {'Unfrozen steps':>14}  {'FineWeb subset':>15}")
    print(f"  {'─' * 12}  {'─' * 18}  {'─' * 18}  {'─' * 14}  {'─' * 14}  {'─' * 15}")
    for name, b in _PRESETS.items():
        marker = " ← selected" if name == preset else ""
        print(
            f"  {name:<12}  {b['frozen']:>18,}  {b['unfrozen']:>18,}  "
            f"  {_h100_steps_for_tokens(b['frozen']):>12,}  "
            f"  {_h100_steps_for_tokens(b['unfrozen']):>12,}  "
            f"{b['subset']:>15}"
            f"{marker}"
        )
    print(f"{'─' * 70}\n")


# ── argument parsing ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--preset",
        choices=list(_PRESETS),
        default="validation",
        help=(
            "Token-budget preset (default: validation). "
            "smoke≈135M (plumbing), validation≈1B (recipe check), "
            "scale≈5B (S2 decision gate), full≈30B (post-gate run)."
        ),
    )
    p.add_argument(
        "--phase",
        choices=["distill", "sft", "grpo", "all"],
        default="all",
        help="Which phase to run (default: all)",
    )
    p.add_argument(
        "--checkpoint", default=None,
        help="Path to a .pt checkpoint to load (skips teacher build for sft/grpo)",
    )
    p.add_argument(
        "--resume", action="store_true",
        help="Resume training from the provided checkpoint (within the selected phase)",
    )
    p.add_argument(
        "--device", default="cuda",
        help="Torch device string (default: cuda)",
    )
    p.add_argument(
        "--grpo-steps", type=int, default=2000,
        help="Total GRPO optimiser steps (split evenly across domain passes for --grpo-data seq)",
    )
    p.add_argument(
        "--grpo-data",
        choices=["math", "orca", "code", "seq", "smoltalk", "both"],
        default="math",
        help=(
            "Data for GRPO rollouts: 'math'/'orca' = OrcaMath only; "
            "'code' = Bespoke-Stratos; 'seq' = sequential math→code; "
            "'both' = sequential math→general (SmolTalk)"
        ),
    )
    p.add_argument(
        "--teacher", default=None,
        help="HuggingFace teacher model (default: use train_4090.TEACHER_MODEL)",
    )
    p.add_argument(
        "--no-flow", action="store_true",
        help="Disable flow matching (fall back to DDPM cosine schedule)",
    )
    p.add_argument(
        "--no-frontier", action="store_true",
        help="Skip frontier CoT datasets in SFT (use only SmolTalk + Orca-Math)",
    )
    p.add_argument(
        "--hf-token", default=os.environ.get("HF_TOKEN"),
        help="HuggingFace API token for auto-upload (or set HF_TOKEN env var)",
    )
    p.add_argument(
        "--hf-repo", default=None,
        help="HuggingFace repo to upload checkpoints to, e.g. 'yourusername/dimba-135m'",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print the resolved config (batch sizes, step counts per preset) and exit",
    )
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Dry-run: just print config and exit (no imports of heavy ML deps needed).
    if args.dry_run:
        _print_dry_run(args.preset)
        return

    # Apply H100 overrides BEFORE any run_* function is called.
    # run_* functions read train_4090 module globals at call time, so mutations
    # done here are seen by run_distill / run_sft / run_grpo.
    apply_h100_overrides(args.preset)

    # Optional overrides that mirror train_4090.main() behaviour.
    if args.teacher:
        _t4.TEACHER_MODEL = args.teacher
        _t4.DISTILL_CFG["teacher_model"] = args.teacher

    if args.no_flow:
        _t4.STUDENT_CFG["use_flow_matching"] = False
        _t4.GRPO_CFG.num_diffusion_steps_inference = 30
        _t4.GRPO_CFG.sampler = "dpmpp"
        logger.info("flow matching disabled (--no-flow)")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        logger.info(
            "GPU: %s | VRAM: %.1f GB | SM: %d.%d",
            props.name, props.total_memory / 1e9,
            props.major, props.minor,
        )
        if props.total_memory / 1e9 < 70:
            logger.warning(
                "GPU has only %.1f GB VRAM (expected ≥80 GB for H100). "
                "The Stage-3 OOM-retry will reduce PRETRAIN_BATCH automatically, "
                "but SFT at batch=%d may OOM — consider reducing H100_SFT_BATCH_SIZE.",
                props.total_memory / 1e9, H100_SFT_BATCH_SIZE,
            )

    if args.hf_repo:
        logger.info("HuggingFace upload enabled → %s", args.hf_repo)

    # Normalise --grpo-data alias
    grpo_data = "math" if args.grpo_data == "orca" else args.grpo_data

    user_ckpt = args.checkpoint

    def _resume_for(phase: str) -> Optional[str]:
        # Pass --resume only for the explicitly selected single phase.
        return user_ckpt if (args.resume and args.phase == phase) else None

    ckpt = args.checkpoint
    hf_kw = dict(hf_token=args.hf_token, hf_repo=args.hf_repo)

    if args.phase in ("distill", "all"):
        ckpt = run_distill(device, resume=_resume_for("distill"), **hf_kw)

    if args.phase in ("sft", "all"):
        ckpt = run_sft(
            ckpt, device,
            resume=_resume_for("sft"),
            use_frontier=not args.no_frontier,
            **hf_kw,
        )

    if args.phase in ("grpo", "all"):
        ckpt = run_grpo(
            ckpt, device,
            resume=_resume_for("grpo"),
            num_steps=args.grpo_steps,
            data=grpo_data,
            **hf_kw,
        )

    logger.info("all done. final model: %s", ckpt)


if __name__ == "__main__":
    main()
