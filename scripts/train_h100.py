#!/usr/bin/env python3
"""DIMBA training entry point optimised for one or more NVIDIA H100 80 GB GPUs.

This script is a thin wrapper around scripts/train_4090.py. It imports the
full distill → SFT → GRPO pipeline from that module and overrides the
module-level config globals for H100 80 GB before calling any run_* function.

What changes vs train_4090.py
------------------------------
  * PRETRAIN_BATCH 32 → 64   (Stage-3 co-adaptation; 128 OOM'd at L=512)
  * _STAGE3_BATCH  32 → 64   (kept in sync so _steps_for_tokens is correct)
  * ALIGN_BATCH     4 → 16   (Stage-1/2 alignment; mixing-matrix O(B·nh·L²·nl))
  * SFT batch_size 32 → 64   (grad_accum 4 → 2; effective batch stays 128)
  * Stage-3 can opt into the Muon + AdamW hybrid for a gated pilot
  * --preset maps to the gated repair/validation budgets in NEXT_RUN_PLAN.md
  * every Stage-3 preset keeps the fixed high-noise objective and teacher KD active
  * Stage 3 supports one-process-per-GPU DDP; alignment runs once on rank 0

VRAM budget reasoning (H100 80 GB, 135M model)
------------------------------------------------
  Model (bf16):           ~270 MB   (135M × 2 bytes)
  Optimizer states:      ≤1.6 GB   (AdamW default; hybrid Muon uses less state)
  Teacher vocab logits:  ~B×50 MB  (B × 512 × 49152 × 2 bytes)
    At B=64:  3.2 GB. Student projection/KL is reduced in B=16 chunks
    (~0.8 GB bf16 logits per chunk), never a second full-B student tensor.
  Activations (30 layers, Mamba CUDA kernel, rough):
    At B=64:  ~1–2 GB   (B × L × d_model × 2B × num_layers × fwd/bwd)

  The SFT heavy case (vocab logits + fragmentation spikes):
    The 48 GB GPU OOM'd at B=64 for SFT at the epoch boundary (fragmentation).
    B=64 on 48 GB → ~49 GB peak.  On 80 GB the same B=64 has ~31 GB headroom.
    We keep effective batch = 128 via grad_accum=2.

  B=128 OOM'd at L=512 on a 96 GB H100 NVL, so production starts at B=64.

Next-run ladder — example commands
-----------------------------------
  # R1: repair the existing budget28 weights with the fixed loss and KD=0.3
  python scripts/train_h100.py --preset repair1b --phase distill \
    --checkpoint checkpoints/distill/final.pt --resume --weights-only-resume \
    --save-dir checkpoints/h100-budget28-repair1b-adamw

  # R2: fresh fixed-recipe validation; keep AdamW and Muon arms isolated
  python scripts/train_h100.py --preset validation --stage3-optimizer adamw \
    --save-dir checkpoints/h100-validation-adamw
  python scripts/train_h100.py --preset validation --stage3-optimizer muon \
    --save-dir checkpoints/h100-validation-muon

  # Same fixed global token budget on 8 GPUs. Batch is per GPU; optimizer-step
  # counts are divided by world size, and exact resume requires the same topology.
  torchrun --standalone --nproc-per-node=8 scripts/train_h100.py \
    --preset validation --phase distill \
    --save-dir checkpoints/h100-validation-8gpu

  # R3: scale only after the repair and fresh-validation coherence gates pass
  python scripts/train_h100.py --preset scale --save-dir checkpoints/h100-scale

  # The 50B option is post-gate only; more tokens alone do not fix budget28.
  python scripts/train_h100.py --preset full --save-dir checkpoints/h100-full

  # Single phase:
  python scripts/train_h100.py --preset scale --phase distill
  python scripts/train_h100.py --preset scale --phase sft  --quality-gate-passed --checkpoint checkpoints/distill/final.pt
  python scripts/train_h100.py --preset scale --phase grpo --quality-gate-passed --checkpoint checkpoints/sft/final.pt

  # Dry-run: print resolved config and exit without training:
  python scripts/train_h100.py --preset scale --dry-run

Preset → token budget mapping (from docs/NEXT_RUN_PLAN.md)
-------------------------------------------------------------
  repair1b   : 0 frozen + 1 B unfrozen                       (R1 budget28 repair)
  smoke      : 100 M frozen + 35 M unfrozen ≈ 135 M total   (plumbing only)
  validation : 700 M frozen + 300 M unfrozen ≈ 1 B total    (R2 fresh recipe)
  scale      : 3 B frozen + 2 B unfrozen ≈ 5 B total        (R3 decision gate)
  full       : 33 B frozen + 17 B unfrozen ≈ 50 B total     (post-gate option)

Stage 3 uses SNR floor 0.5, CE fade (1-t), 50/50 uniform/logit-normal timesteps,
and teacher KD 1.0→0.3 (or 0.3 for the unfrozen-only repair). The budget28 presets
are retained for historical continuation/inspection; they are not the recommended
first response to the budget28 failure. See docs/ROOT_CAUSE_BUDGET28.md.
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
from dimba.training.distributed import DistributedContext, init_distributed, seed_process
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
#   B=128 OOM'd at L=512 on a 96 GB H100 NVL (the auto-downshift then dropped it to
#   B=64, eff-batch 64, ~92/96 GB — healthy). We now set B=64 explicitly so the run
#   starts clean with no OOM-retry. Stage 3 has no grad_accum path, so global batch
#   under DDP is B/GPU × world size.
H100_PRETRAIN_BATCH   = 64    # per GPU; 64 fits at ~92/96 GB, no downshift
H100_PRETRAIN_SEQ_LEN = 512   # unchanged; L=512 is the correct context window
H100_STAGE3_BATCH     = 64    # must equal H100_PRETRAIN_BATCH (_steps_for_tokens uses this)
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
# `subset` picks the FineWeb config so each preset draws unique tokens without replay.
# The definitive budget28 failure was the high-noise objective, not token count, but
# replay would still invalidate a controlled token-budget comparison. sample-10BT
# (~10B) covers everything up to the 5B scale gate; the 50B full run needs a bigger
# pool, so it uses sample-100BT (~100B unique). apply_h100_overrides sets
# train_4090.PRETRAIN_SUBSET from this.
_PRESETS: dict[str, dict] = {
    "smoke": {
        "frozen":   100_000_000,   # 100 M
        "unfrozen":  35_000_000,   #  35 M  → ~135 M total (plumbing only)
        "subset":   "sample-10BT",
        "desc": "~135M tokens — plumbing only; not a quality gate",
    },
    "validation": {
        "frozen":   700_000_000,   # 700 M
        "unfrozen": 300_000_000,   # 300 M  → ~1 B total (R2 fresh recipe)
        "subset":   "sample-10BT",
        "desc": "~1B tokens — R2 fresh fixed-loss recipe gate",
    },
    "scale": {
        "frozen":   3_000_000_000,  # 3 B
        "unfrozen": 2_000_000_000,  # 2 B   → ~5 B total (R3 decision gate)
        "subset":   "sample-10BT",  # 5B < 10B → unique, no looping
        "desc": "~5B tokens — R3 scale-test / GO·NO-GO decision gate",
    },
    "full": {
        "frozen":   33_000_000_000,  # 33 B
        "unfrozen": 17_000_000_000,  # 17 B  → 50 B total (post-gate option)
        "subset":   "sample-100BT",  # 50B unique needs >10B → 100BT pool (~100B)
        "desc": "~50B tokens — post-R3 option (33B frozen + 17B unfrozen)",
    },
    # Historical cost-capped 28B shape. The post-mortem proved that simply rerunning
    # this budget is not a repair; use repair1b and its coherence gate first.
    "budget28": {
        "frozen":   18_000_000_000,  # 18 B
        "unfrozen": 10_000_000_000,  # 10 B  → 28 B total (cost-capped full run)
        "subset":   "sample-100BT",  # 28B unique needs >10B → 100BT pool (~100B)
        "desc": "legacy ~28B cost-capped shape; not a pre-gate repair recommendation",
    },
    "budget28_3b": {
        "frozen":            0,       # 0 → _build_stage3_phases skips the 3a frozen phase
        "unfrozen": 10_000_000_000,  # 10 B  → 3b-only unfrozen continuation
        "subset":   "sample-100BT",  # same pool as budget28
        "desc": "legacy 10B 3b-only continuation; run repair1b gate first",
    },
    "repair1b": {
        "frozen":            0,
        "unfrozen": 1_000_000_000,
        "subset":   "sample-10BT",
        "unfrozen_kd_weight": 0.3,
        "desc": "R1: 1B-token fixed-loss/KD=0.3 repair; inspect every quarter checkpoint",
    },
    # Short validation of the same teacher-guided recipe used by every production
    # preset. The smaller budget exposes a bad trajectory before a scale spend.
    "kdval2b": {
        "frozen":   1_500_000_000,   # 1.5 B — the plateau window from budget28
        "unfrozen":   500_000_000,   # 0.5 B low-LR FFN co-adaptation
        "subset":   "sample-10BT",   # 2B < 10B → unique tokens, no looping
        "align_steps": 2000,         # fix #4: 4× longer Stage-1/2 bridge (was 500)
        "desc": "~2B tokens, Stage-3 KD 1.0->0.3 — validates teacher-guided repair",
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
    frozen_kd_weight: float = 1.0,
    unfrozen_kd_weight: float = 0.3,
    optimizer: str = "adamw",
) -> list:
    """Build Stage-3 phase dicts in the schema expected by DistillationTrainer.

    Mirrors train_4090._coadapt_stages() but uses H100 batch sizes for step
    counts and takes explicit token budgets so the preset is the only knob.
    A phase is omitted when its token budget is zero.
    """
    if frozen_kd_weight <= 0.0 or unfrozen_kd_weight <= 0.0:
        raise ValueError("Stage-3 teacher KD weights must stay positive")
    phases = []
    weight_decay = (
        _t4.STAGE3_MUON_WEIGHT_DECAY
        if optimizer == "muon"
        else _t4.STAGE3_WEIGHT_DECAY
    )
    if frozen_tokens > 0:
        phases.append({
            "name":           "stage3",
            "steps":          _h100_steps_for_tokens(frozen_tokens),
            "lr":             frozen_lr,
            "freeze_ffn":     True,
            "kd_weight":      frozen_kd_weight,
            "ce_loss_weight": 1.0,
            "min_snr_gamma":  5.0,
            "snr_floor":      _t4.STAGE3_SNR_FLOOR,
            "ce_time_fade":   _t4.STAGE3_CE_TIME_FADE,
            "kd_time_fade":   True,
            "optimizer":      optimizer,
            "weight_decay":   weight_decay,
        })
    if unfrozen_tokens > 0:
        phases.append({
            "name":           "stage3",
            "steps":          _h100_steps_for_tokens(unfrozen_tokens),
            "lr":             unfrozen_lr,
            "freeze_ffn":     False,
            "kd_weight":      unfrozen_kd_weight,
            "ce_loss_weight": 1.0,
            "min_snr_gamma":  5.0,
            "snr_floor":      _t4.STAGE3_SNR_FLOOR,
            "ce_time_fade":   _t4.STAGE3_CE_TIME_FADE,
            "kd_time_fade":   True,
            "optimizer":      optimizer,
            "weight_decay":   weight_decay,
        })
    return phases


def apply_h100_overrides(preset: str, stage3_optimizer: str = "adamw") -> None:
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
    _t4.REQUIRE_FAST_MAMBA2 = True
    _t4.REQUIRE_ALIGNMENT_SUCCESS = True
    _t4.REQUIRE_QUALITY_GATE = True
    _t4.STAGE3_OPTIMIZER = stage3_optimizer
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
    # Stage-3 streams from this config; a budget larger than the pool would replay
    # documents and invalidate the token-budget comparison. Pick a pool ≥ the budget.
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
    # align_steps: preset override for the Stage-1/2 alignment length. budget28's
    # 500+500 steps were the only teacher bridge and stage2 ended at loss 0.556
    # (not converged) — see docs/ROOT_CAUSE_BUDGET28.md fix #4.
    if "align_steps" in budget:
        for s in align_phases:
            s["steps"] = budget["align_steps"]
    stage3_phases = _build_stage3_phases(
        frozen_tokens,
        unfrozen_tok,
        frozen_lr,
        unfrozen_lr,
        frozen_kd_weight=float(
            budget.get("frozen_kd_weight", _t4.STAGE3_FROZEN_KD_WEIGHT)
        ),
        unfrozen_kd_weight=float(
            budget.get("unfrozen_kd_weight", _t4.STAGE3_UNFROZEN_KD_WEIGHT)
        ),
        optimizer=stage3_optimizer,
    )
    _t4.DISTILL_CFG["stages"] = align_phases + stage3_phases

    logger.info(
        "H100 overrides applied | preset=%s | %s",
        preset, budget["desc"],
    )
    logger.info(
        "  Stage-3  optimizer=%s  batch=%d  seq=%d  subset=%s  |  "
        "frozen=%s tok → %s steps  |  unfrozen=%s tok → %s steps  |  KD=%.2f→%.2f",
        stage3_optimizer, H100_STAGE3_BATCH, H100_STAGE3_SEQ, budget["subset"],
        f"{frozen_tokens:,}", f"{_h100_steps_for_tokens(frozen_tokens):,}",
        f"{unfrozen_tok:,}", f"{_h100_steps_for_tokens(unfrozen_tok):,}",
        stage3_phases[0]["kd_weight"], stage3_phases[-1]["kd_weight"],
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


def _print_dry_run(preset: str, stage3_optimizer: str) -> None:
    """Print resolved config for all presets (or just the selected one) and exit."""
    print(f"\n{'─' * 70}")
    print(f"  train_h100.py — dry-run config check  (preset={preset})")
    print(f"{'─' * 70}")
    print(f"  H100_PRETRAIN_BATCH   = {H100_PRETRAIN_BATCH}")
    print(f"  H100_ALIGN_BATCH      = {H100_ALIGN_BATCH}")
    print(f"  H100_SFT_BATCH_SIZE   = {H100_SFT_BATCH_SIZE}")
    print(f"  H100_SFT_GRAD_ACCUM   = {H100_SFT_GRAD_ACCUM}")
    print(f"  SFT effective batch   = {H100_SFT_BATCH_SIZE * H100_SFT_GRAD_ACCUM}")
    print(f"  Stage-3 optimizer     = {stage3_optimizer}")
    selected = _PRESETS[preset]
    selected_phases = _build_stage3_phases(
        selected["frozen"],
        selected["unfrozen"],
        _t4.STAGE3_FROZEN_LR,
        _t4.STAGE3_UNFROZEN_LR,
        frozen_kd_weight=float(
            selected.get("frozen_kd_weight", _t4.STAGE3_FROZEN_KD_WEIGHT)
        ),
        unfrozen_kd_weight=float(
            selected.get("unfrozen_kd_weight", _t4.STAGE3_UNFROZEN_KD_WEIGHT)
        ),
        optimizer=stage3_optimizer,
    )
    kd_schedule = " -> ".join(f"{phase['kd_weight']:.2f}" for phase in selected_phases)
    print(
        f"  Stage-3 teacher KD    = {kd_schedule or 'none'}"
    )
    print()
    print(f"  {'Preset':<12}  {'Frozen tokens':>18}  {'Unfrozen tokens':>18}  "
          f"{'Frozen steps':>14}  {'Unfrozen steps':>14}  {'FineWeb subset':>15}")
    print(f"  {'─' * 12}  {'─' * 18}  {'─' * 18}  {'─' * 14}  {'─' * 14}  {'─' * 15}")
    for name, b in _PRESETS.items():
        marker = " ← selected" if name == preset else ""
        frozen_steps = _h100_steps_for_tokens(b["frozen"]) if b["frozen"] else 0
        unfrozen_steps = _h100_steps_for_tokens(b["unfrozen"]) if b["unfrozen"] else 0
        print(
            f"  {name:<12}  {b['frozen']:>18,}  {b['unfrozen']:>18,}  "
            f"  {frozen_steps:>12,}  "
            f"  {unfrozen_steps:>12,}  "
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
            "repair1b≈1B (R1 repair), validation≈1B (R2 fresh recipe), "
            "scale≈5B (R3 decision gate), full≈50B (post-gate only)."
        ),
    )
    p.add_argument(
        "--stage3-optimizer",
        choices=["adamw", "muon"],
        default="adamw",
        help=(
            "Stage-3 optimizer (default: adamw). Use muon only for the gated pilot; "
            "it keeps embeddings/heads/vectors on AdamW with weight_decay=0.0."
        ),
    )
    p.add_argument(
        "--phase",
        choices=["distill", "sft", "grpo", "all"],
        default="distill",
        help="Which phase to run (default: distill; 'all' is rejected by quality gates)",
    )
    p.add_argument(
        "--checkpoint", default=None,
        help="Path to a .pt checkpoint to load (skips teacher build for sft/grpo)",
    )
    p.add_argument(
        "--save-dir",
        default=None,
        help="Checkpoint directory (default: ./checkpoints/<phase>)",
    )
    p.add_argument(
        "--resume", action="store_true",
        help=(
            "Exact distillation resume from --checkpoint; SFT/GRPO exact resume "
            "is intentionally unsupported"
        ),
    )
    p.add_argument(
        "--weights-only-resume",
        action="store_true",
        help=(
            "Explicitly discard optimizer/RNG/data progress when resuming an old "
            "distillation checkpoint that contains weights only"
        ),
    )
    p.add_argument(
        "--quality-gate-passed",
        action="store_true",
        help=(
            "Confirm the input checkpoint passed the documented coherence/quality gate; "
            "required before H100 SFT or GRPO can start"
        ),
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
        _print_dry_run(args.preset, args.stage3_optimizer)
        return

    if args.weights_only_resume and not (args.resume and args.checkpoint):
        raise ValueError("--weights-only-resume requires --resume and --checkpoint")
    if args.phase == "all":
        raise RuntimeError(
            "H100 stages are gated and must run phase-by-phase; --phase all would "
            "auto-advance before the required quality evaluation."
        )
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1 and args.phase != "distill":
        raise RuntimeError(
            "train_h100.py supports torchrun only for continuous Stage-3 distillation; "
            "SFT/GRPO remain single-process and phase-gated."
        )
    if args.phase in ("sft", "grpo") and not args.quality_gate_passed:
        raise RuntimeError(
            "H100 SFT/GRPO requires explicit --quality-gate-passed; refusing to "
            "auto-advance from a warning. Run and record the documented gate first."
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "train_h100.py requires CUDA and the fused Mamba2 stack. Use --dry-run "
            "for config validation on CPU/MPS."
        )
    if not str(args.device).startswith("cuda"):
        raise ValueError("train_h100.py requires a CUDA --device")
    context = (
        init_distributed("cuda")
        if world_size > 1
        else DistributedContext(0, 0, 1, torch.device(args.device))
    )
    device = context.device
    seed_process(42, context)

    from dimba.models.denoiser import HAS_MAMBA_SSM

    if not HAS_MAMBA_SSM:
        raise RuntimeError(
            "mamba_ssm.Mamba2 is unavailable; refusing the H100 run instead of "
            "silently using TorchMamba2. Install the pinned CUDA training extra."
        )
    try:
        import causal_conv1d  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "causal-conv1d is unavailable; install the pinned CUDA training extra."
        ) from exc

    # Apply H100 overrides BEFORE any run_* function is called.
    # run_* functions read train_4090 module globals at call time, so mutations
    # done here are seen by run_distill / run_sft / run_grpo.
    apply_h100_overrides(args.preset, args.stage3_optimizer)
    _t4.QUALITY_GATE_PASSED = args.quality_gate_passed

    # Optional overrides that mirror train_4090.main() behaviour.
    if args.teacher:
        _t4.TEACHER_MODEL = args.teacher
        _t4.DISTILL_CFG["teacher_model"] = args.teacher

    if args.no_flow:
        _t4.STUDENT_CFG["use_flow_matching"] = False
        _t4.GRPO_CFG.num_diffusion_steps_inference = 30
        _t4.GRPO_CFG.sampler = "dpmpp"
        logger.info("flow matching disabled (--no-flow)")

    if device.type == "cuda" and context.is_main:
        props = torch.cuda.get_device_properties(device)
        logger.info(
            "GPU: %s | VRAM: %.1f GB | SM: %d.%d",
            props.name, props.total_memory / 1e9,
            props.major, props.minor,
        )
        if props.total_memory / 1e9 < 70:
            stage3_oom_guidance = (
                "Distributed Stage 3 aborts coherently on OOM; choose a lower batch for a "
                "new run (a changed batch cannot exact-resume)."
                if context.enabled
                else "Single-process Stage 3 may downshift before its first completed step."
            )
            logger.warning(
                "GPU has only %.1f GB VRAM (expected ≥80 GB for H100). "
                "%s SFT at batch=%d may OOM — consider reducing H100_SFT_BATCH_SIZE.",
                props.total_memory / 1e9,
                stage3_oom_guidance,
                H100_SFT_BATCH_SIZE,
            )

    if args.hf_repo:
        logger.info("HuggingFace upload enabled → %s", args.hf_repo)

    # Normalise --grpo-data alias
    grpo_data = "math" if args.grpo_data == "orca" else args.grpo_data

    user_ckpt = args.checkpoint

    def _resume_for(phase: str) -> Optional[str]:
        # H100 rejects --phase all. Distillation owns the only exact-resume path;
        # run_sft/run_grpo reject resume instead of silently resetting optimizer state.
        if phase == "distill" and args.resume and args.phase == "distill":
            return user_ckpt
        return user_ckpt if (args.resume and args.phase == phase) else None

    ckpt = args.checkpoint
    hf_kw = dict(hf_token=args.hf_token, hf_repo=args.hf_repo)
    save_kw = {"save_dir": args.save_dir} if args.save_dir else {}

    if args.phase in ("distill", "all"):
        ckpt = run_distill(
            device,
            resume=_resume_for("distill"),
            weights_only_resume=args.weights_only_resume,
            context=context,
            **save_kw,
            **hf_kw,
        )

    if args.phase in ("sft", "all"):
        ckpt = run_sft(
            ckpt, device,
            resume=_resume_for("sft"),
            use_frontier=not args.no_frontier,
            **save_kw,
            **hf_kw,
        )

    if args.phase in ("grpo", "all"):
        ckpt = run_grpo(
            ckpt, device,
            resume=_resume_for("grpo"),
            num_steps=args.grpo_steps,
            data=grpo_data,
            **save_kw,
            **hf_kw,
        )

    if context.is_main:
        logger.info("all done. final model: %s", ckpt)
    context.close()


if __name__ == "__main__":
    main()
