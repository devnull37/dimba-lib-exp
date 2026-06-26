#!/usr/bin/env python3
"""Full DIMBA training pipeline optimised for a 48 GB GPU (RTX 4090 modded / L40S / A6000 Ada).

Phases
------
  1. distill   Cross-architecture distillation from a HF teacher (SmolLM-135M default).
               Runs Stage 1 (matrix align) → Stage 2 (hidden align) → Stage 3 (diffusion ft).
  2. sft       Two-stage supervised fine-tune on block-CoT reasoning data:
               Stage 1 — full mix: SmolTalk + Orca-Math + all Frontier CoT (~284 K rows)
               Stage 2 — hard subset: Frontier CoT + Orca-Math only (no direct-answer data),
                         forcing the model to refine on examples that require actual reasoning.
  3. grpo      GRPO/MGPO policy optimisation with verifiable rewards.
               Sequential domain training: math RL → code RL.
               MGPO (VibeThinker): upweights capability-boundary prompts.
               Long2Short: normalized brevity bonus for correct-but-shorter completions.

Quick start (runs all three phases end-to-end):
    python scripts/train_4090.py

Run a single phase:
    python scripts/train_4090.py --phase distill
    python scripts/train_4090.py --phase sft   --checkpoint checkpoints/distill/final.pt
    python scripts/train_4090.py --phase grpo  --checkpoint checkpoints/sft/final.pt

Resume from a checkpoint:
    python scripts/train_4090.py --phase grpo --checkpoint checkpoints/grpo/grpo_step200.pt --resume

GRPO domain options:
    --grpo-data math   OrcaMath only (default)
    --grpo-data code   Bespoke-Stratos code split only
    --grpo-data seq    Sequential: math RL (1 K steps) → code RL (1 K steps)
    --grpo-data both   Sequential: math RL (num_steps//2) → SmolTalk/general RL (num_steps//2)

Flow matching is ON by default (15 Euler steps ≈ 2× faster inference than 30 DPM++ steps).

HuggingFace auto-upload (private repo, created automatically):
    python scripts/train_4090.py --hf-token hf_xxx --hf-repo yourusername/dimba-135m
    Uploads after each phase: distill/final.pt → sft/final.pt → grpo/final.pt

The script writes ./training_state.json after every log_interval steps so that a
Claude /loop monitor (scripts/monitor.py) can read progress and adjust config.
"""
import argparse
import gc
import json
import logging
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset

# ── add project root to path ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dimba.models.diffusion import DIMBA
from dimba.distillation.trainer import DistillationTrainer, DistillationConfig
from dimba.distillation.surgery import build_student_from_teacher
from dimba.distillation.teacher import TeacherWrapper
from dimba.training.grpo import (
    GRPOConfig, build_math_grpo_trainer, build_code_grpo_trainer,
    build_general_grpo_trainer,
)
from dimba.data.cot_dataset import SmolTalkDataset, OrcaMathDataset, BlockCoTDataset
from dimba.data.frontier_cot import (
    FrontierCoTMix, BespokeStratosDataset,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_4090")

# ── hardware flags (set before any CUDA calls) ────────────────────────────────
# expandable_segments cuts CUDA fragmentation — important for the memory-heavy
# pure-PyTorch SSD scan (TorchMamba2). TOKENIZERS_PARALLELISM=false silences the
# fork-mode warning from fast tokenizers running inside DataLoader workers.
# setdefault so an explicitly-set environment still wins.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# TF32: free ~10% matmul speedup on Ampere/Ada; does NOT affect bf16 forward passes.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ── defaults (tuned for 48 GB VRAM + SmolLM-135M teacher) ────────────────────

TEACHER_MODEL = "HuggingFaceTB/SmolLM-135M"
TEACHER_TYPE = "causal"

# DIMBA student config (matches teacher width after Mode-A surgery)
STUDENT_CFG = dict(
    vocab_size=49152,       # SmolLM tokenizer vocab size
    d_model=576,            # SmolLM-135M hidden dim
    d_prompt=576,
    num_diffusion_steps=1000,
    num_denoiser_layers=30, # SmolLM-135M depth
    d_state=64,
    d_conv=4,
    expand=2,
    conditioning_type="adaln",   # AdaLN-Zero (DiT-style, better than FiLM)
    use_flow_matching=True,      # rectified flow — ~2-3× faster inference
    flow_logit_normal=True,      # logit-normal timestep sampling (SD3/FLUX schedule)
    use_weight_tying=True,
    use_simple_mamba=False,
    block_ffn=True,              # inherit teacher FFN (Mode A)
    ffn_type="swiglu",           # SmolLM uses SwiGLU
)

# ── Stage-3 co-adaptation token budget ────────────────────────────────────────
# The token budget is the knob; optimiser *steps* are derived from it so the recipe
# scales by editing one or two numbers.  One Stage-3 step on the fast CUDA kernel
# processes PRETRAIN_BATCH × PRETRAIN_SEQ_LEN = 32 × 512 = 16,384 tokens.  (On the
# slow TorchMamba2 fallback the realised batch is smaller, so realised tokens scale
# down proportionally — the run logs the true count.)  See docs/NEXT_RUN_PLAN.md for
# the per-stage budget and the citations behind these numbers.
_STAGE3_BATCH = 32   # keep in sync with PRETRAIN_BATCH (defined below)
_STAGE3_SEQ   = 512  # keep in sync with PRETRAIN_SEQ_LEN (defined below)


def _steps_for_tokens(n_tokens: int) -> int:
    """Convert a Stage-3 token budget into optimiser steps (CUDA-kernel batch)."""
    return max(1, round(n_tokens / (_STAGE3_BATCH * _STAGE3_SEQ)))


# Co-adaptation budget, split FROZEN-FFN → UNFROZEN-FFN.  Run #1 used ~1B
# frozen-only tokens and was incoherent.  The attention→Mamba conversion literature
# lands at 3-20B tokens WITH the FFN co-adapting end-to-end (MOHAWK/Phi-Mamba 3-5B;
# Llamba-1B 8B; Mamba-in-Llama 20B).  DIMBA additionally swaps the objective
# (autoregressive → diffusion/flow-matching) and direction (causal → bidirectional),
# which dilutes per-token signal, so we budget toward the upper end of that band AND
# always run the previously-dropped FFN-unfrozen "Finetune #2".
#
# Presets (set the two knobs below; either may be 0 to skip that phase):
#   SMOKE / plumbing gate    : 100_000_000  frozen +  35_000_000  unfrozen (~135M)
#   VALIDATION gate          : 700_000_000  frozen + 300_000_000  unfrozen (~1B)
#   SCALE-TEST gate (default): 3_000_000_000 frozen + 2_000_000_000 unfrozen (~5B)
#   FULL run (post-gate)     : 20_000_000_000+ frozen + 10_000_000_000+ unfrozen
#   VALIDATION continuation  : 0 frozen + 5_000_000_000..10_000_000_000 unfrozen,
#                              run with --resume on the existing base checkpoint to
#                              cheaply test the undertraining fix (FFN unfrozen only).
STAGE3_FROZEN_TOKENS   = 3_000_000_000   # 3B: re-bed the new Mamba mixer into the inherited FFN
STAGE3_UNFROZEN_TOKENS = 2_000_000_000   # 2B: low-LR FFN co-adaptation (the dropped "Finetune #2")

# LRs for the two co-adaptation phases.  The unfrozen phase uses a much lower LR so
# the inherited ~600B-token SmolLM FFN co-adapts to the matured mixer WITHOUT being
# washed out (standard low-LR continued-pretraining / unfreeze practice).
STAGE3_FROZEN_LR   = 2e-4
STAGE3_UNFROZEN_LR = 3e-5


def _coadapt_stages() -> list:
    """Build the Stage-3 co-adaptation phases (frozen FFN → unfrozen FFN).

    Both phases run the same DIMBA diffusion objective on raw web text; they differ
    only in whether the inherited FFN is trainable and at what LR.  A phase with a
    zero token budget is omitted (e.g. set STAGE3_FROZEN_TOKENS=0 for a pure
    unfrozen continuation of an already-co-adapted base).
    """
    phases = []
    if STAGE3_FROZEN_TOKENS > 0:
        # FFN FROZEN: only the Mamba mixer trains, learning to produce activations
        # the inherited FFN already knows how to consume (MOHAWK-faithful).
        phases.append({"name": "stage3", "steps": _steps_for_tokens(STAGE3_FROZEN_TOKENS),
                       "lr": STAGE3_FROZEN_LR, "freeze_ffn": True, "kd_weight": 0.0,
                       "ce_loss_weight": 1.0, "min_snr_gamma": 5.0})
    if STAGE3_UNFROZEN_TOKENS > 0:
        # FFN UNFROZEN ("Finetune #2"): let the FFN co-adapt to the matured mixer at
        # low LR.  This is the step dropped in run #1 that left the FFN never adapted.
        phases.append({"name": "stage3", "steps": _steps_for_tokens(STAGE3_UNFROZEN_TOKENS),
                       "lr": STAGE3_UNFROZEN_LR, "freeze_ffn": False, "kd_weight": 0.0,
                       "ce_loss_weight": 1.0, "min_snr_gamma": 5.0})
    return phases


DISTILL_CFG = dict(
    teacher_model=TEACHER_MODEL,
    teacher_type=TEACHER_TYPE,
    mode="convert",
    block_ffn=True,
    inherit_embeddings=True,
    inherit_ffn=True,
    inherit_head=True,
    principled_init=False,
    layer_map_mode="uniform",
    num_student_layers=None,
    share_vocab=True,
    kd_weight=1.0,
    kd_temp=2.0,
    device="cuda",
    stages=[
        # Stage 1: align Mamba mixing matrices → teacher attention maps (clean text)
        {"name": "stage1", "steps": 500,  "lr": 1e-3},
        # Stage 2: align hidden states → teacher residual stream (clean text)
        {"name": "stage2", "steps": 500,  "lr": 5e-4},
        # Stage 3: co-adaptation pretraining, FFN FROZEN then UNFROZEN (see knobs above).
        # kd_weight=0 → no teacher forward pass needed during Stage 3 (fast on the CUDA kernel).
        *_coadapt_stages(),
    ],
)

# Pretraining dataset for distillation Stage 3 — raw web text, no labels.
# FineWeb "sample-10BT" is 10B tokens of high-quality filtered CommonCrawl.
PRETRAIN_DATASET = "HuggingFaceFW/fineweb"
PRETRAIN_SUBSET  = "sample-10BT"
PRETRAIN_SEQ_LEN = 512
PRETRAIN_BATCH   = 32   # Stage-3 batch on the fast CUDA Mamba2 kernel
# The pure-PyTorch SSD scan (TorchMamba2) materialises a [B, L, L, nheads] decay matrix
# per layer, so it needs a much smaller batch than the fused CUDA kernel. When Stage 3
# runs on TorchMamba2 (CUDA kernel unavailable / transfer failed) we use this batch and
# enable gradient checkpointing; the graceful-OOM handler halves it further if needed.
PRETRAIN_BATCH_TORCH = 8

# Stage 1+2 alignment use SHORT sequences. Stage-1 materializes the mixing matrix
# [B, nheads, L, L] for every layer, which is O(B*nheads*L^2*num_layers): at L=512
# that is ~2.3 G fp32 elements and trips the denoiser OOM guard. L=256, B=4 keeps it
# ~0.1-1 GB and is plenty of context for attention→Mamba matrix alignment. Stage 3
# (no matrices) uses the full PRETRAIN_* size for the heavy co-adaptation run.
ALIGN_SEQ_LEN = 256
ALIGN_BATCH   = 4

# SFT stage 1 — full data mix
SFT_CFG = dict(
    batch_size=32,          # 32 (was 64): batch 64 ran right at the 49 GB ceiling and OOM'd
                            # at the epoch-1→2 boundary. The vocab-49k output logits at
                            # B×512 dominate memory for this otherwise-tiny model, so halving
                            # the micro-batch buys real headroom. expandable_segments helps
                            # fragmentation but doesn't shrink the footprint — this does.
    grad_accum=4,           # effective batch = 128 (unchanged: 32×4 == old 64×2)
    lr=2e-5,
    warmup_steps=100,
    num_epochs=2,
    max_seq_len=512,
    block_size=64,
    num_think_blocks=2,
    response_len=128,
    weight_decay=0.01,
    max_grad_norm=1.0,
    save_interval=500,
    log_interval=20,
    num_workers=8,
    prefetch_factor=2,
)

# SFT stage 2 — hard subset (reasoning-only data, lower LR)
SFT_STAGE2_CFG = {**SFT_CFG, 'lr': 5e-6, 'num_epochs': 1, 'warmup_steps': 50, 'save_interval': 200}

# GRPO hyperparams — MGPO + Long2Short (VibeThinker) + flow matching
GRPO_CFG = GRPOConfig(
    block_size=64,
    max_think_blocks=2,           # hard cap — prevents overthinking
    response_len=128,
    adaptive_stop=True,
    # MGPO: upweight capability-boundary prompts (VibeThinker)
    use_mgpo=True,
    mgpo_gamma=2.0,               # peaks at group accuracy = 0.5
    # Long2Short: normalized brevity bonus (replaces fixed thinking_length_weight)
    long2short_lambda=0.2,        # 0 = disable, use thinking_length_weight instead
    min_response_reward_bonus=0.05,
    group_size=8,
    kl_coeff=0.04,
    mc_samples=2,
    antithetic=True,
    lr=5e-6,
    weight_decay=0.01,
    max_grad_norm=1.0,
    warmup_steps=50,
    total_steps=2000,             # for cosine schedule
    bf16=True,
    log_interval=10,
    save_interval=200,
    save_dir="./checkpoints/grpo",
    state_file="./training_state.json",
    # Flow matching: 15 Euler steps ≈ same quality as 30 DPM++ steps, ~2× faster
    num_diffusion_steps_inference=15,
    sampler="euler",
)


# ── utilities ─────────────────────────────────────────────────────────────────

def _write_state(state: dict, path: str = "./training_state.json") -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump({**state, "timestamp": time.time()}, f, indent=2)


def _upload_checkpoint(
    local_path: str,
    hf_token: Optional[str],
    hf_repo: Optional[str],
    repo_filename: Optional[str] = None,
) -> None:
    """Upload a checkpoint file to a private HuggingFace repo.

    Creates the repo if it doesn't exist (always private).
    Silently skips if hf_token or hf_repo are not set.
    """
    if not hf_token or not hf_repo:
        return
    try:
        from huggingface_hub import HfApi
    except ImportError:
        logger.warning("huggingface_hub not installed — skipping upload. pip install huggingface_hub")
        return
    try:
        api = HfApi(token=hf_token)
        api.create_repo(hf_repo, private=True, exist_ok=True, repo_type="model")
        dest = repo_filename or os.path.basename(local_path)
        logger.info("uploading %s → hf://%s/%s …", local_path, hf_repo, dest)
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=dest,
            repo_id=hf_repo,
            repo_type="model",
        )
        logger.info("upload done → https://huggingface.co/%s", hf_repo)
    except Exception as exc:
        logger.warning("HuggingFace upload failed (non-fatal): %s", exc)


def _load_tokenizer(teacher_model: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(teacher_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _build_or_load_model(
    checkpoint: Optional[str],
    device: torch.device,
) -> DIMBA:
    if checkpoint and os.path.isfile(checkpoint):
        logger.info("loading model from checkpoint: %s", checkpoint)
        import inspect
        ckpt = torch.load(checkpoint, map_location="cpu")
        sd = ckpt["model_state_dict"]
        cfg = ckpt.get("config")
        dimba_params = set(inspect.signature(DIMBA.__init__).parameters) - {"self"}
        if not (isinstance(cfg, dict) and "vocab_size" in cfg and set(cfg) <= dimba_params):
            raise ValueError(
                f"checkpoint {checkpoint} has no usable DIMBA model config "
                f"(found keys: {sorted(cfg) if isinstance(cfg, dict) else type(cfg)}). "
                "Re-save it with model.config, or rebuild from teacher.")
        # Trust the tensors over the recorded config for vocab_size (resize may have grown it).
        emb_key = next(k for k in sd if k.endswith("token_embed.embedding.weight"))
        true_vocab = sd[emb_key].shape[0]
        cfg = {**cfg, "vocab_size": true_vocab}
        model = DIMBA(**cfg)
        # SFT/GRPO checkpoints saved after the <think> resize carry a redundant tied-weight
        # key (output_head.embedding_weight == token_embed weight). A freshly-built model
        # registers that as a NON-persistent buffer (absent from its state_dict), so a strict
        # load rejects it as an "unexpected key" — which is why SFT intermediates weren't
        # directly resumable. Drop it before loading: weight tying is re-established at build
        # (and again after the <think> resize downstream), so the tensor is recreated, not
        # lost. This makes every SFT/GRPO intermediate checkpoint resumable without surgery.
        sd.pop("output_head.embedding_weight", None)
        model.load_state_dict(sd, strict=True)
    else:
        logger.info("building fresh student model (Mode A: inherit teacher weights)")
        _build_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        teacher = TeacherWrapper(TEACHER_MODEL, device=str(device), dtype=_build_dtype)
        model, _layer_map = build_student_from_teacher(
            teacher,
            num_diffusion_steps=1000,
            d_state=64,
            d_conv=4,
            expand=2,
            conditioning_type=STUDENT_CFG["conditioning_type"],
            use_flow_matching=STUDENT_CFG["use_flow_matching"],
            flow_logit_normal=STUDENT_CFG["flow_logit_normal"],
            use_weight_tying=True,
            use_simple_mamba=False,
            block_ffn=DISTILL_CFG["block_ffn"],
            inherit_embeddings=DISTILL_CFG["inherit_embeddings"],
            inherit_ffn=DISTILL_CFG["inherit_ffn"],
            inherit_head=DISTILL_CFG["inherit_head"],
        )
        teacher.unload()

    model = model.to(device)
    if device.type == "cuda":
        model = model.to(torch.bfloat16)  # bf16 storage; no GradScaler needed
    return model


def _log_vram(device: torch.device, tag: str = "") -> None:
    if device.type != "cuda":
        return
    used = torch.cuda.memory_allocated(device) / 1e9
    reserved = torch.cuda.memory_reserved(device) / 1e9
    logger.info("VRAM %s: %.1f GB allocated / %.1f GB reserved", tag, used, reserved)


def _make_block_cot_dataset(
    base: Dataset,
    tokenizer,
    cfg: dict,
    think_start_id: Optional[int],
    think_end_id: Optional[int],
) -> BlockCoTDataset:
    def _tok_fn(text: str):
        return tokenizer.encode(text, add_special_tokens=False)

    return BlockCoTDataset(
        base,
        tokenizer=_tok_fn,
        max_prompt_len=128,
        block_size=cfg["block_size"],
        num_think_blocks=cfg["num_think_blocks"],
        response_len=cfg["response_len"],
        think_start_id=think_start_id,
        think_end_id=think_end_id,
        pad_id=tokenizer.pad_token_id or 0,
    )


def _make_loader(ds: Dataset, cfg: dict, shuffle: bool = True) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=cfg["batch_size"],
        shuffle=shuffle,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        prefetch_factor=cfg["prefetch_factor"],
        persistent_workers=True,
    )


def _sft_loop(
    model: nn.Module,
    loader: DataLoader,
    cfg: dict,
    device: torch.device,
    save_dir: str,
    label: str,
    start_step: int = 0,
) -> int:
    """Run one SFT stage; returns the global step count at end."""
    accum = cfg["grad_accum"]
    warmup = cfg["warmup_steps"]
    # LR schedule operates in optimizer steps, not micro-batch steps.
    # Use ceil to account for partial-group flushes at epoch end.
    total_optimizer_steps = math.ceil(len(loader) / accum) * cfg["num_epochs"]

    _fused = device.type == "cuda"
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"],
        fused=_fused,
    )

    def _lr_lambda(opt_step: int) -> float:
        if opt_step < warmup:
            return opt_step / max(1, warmup)
        progress = (opt_step - warmup) / max(1, total_optimizer_steps - warmup)
        return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
    global_step = start_step
    opt_step = 0
    running_loss, running_n = 0.0, 0
    model.train()

    for epoch in range(cfg["num_epochs"]):
        logger.info("%s epoch %d/%d", label, epoch + 1, cfg["num_epochs"])
        micro_step = 0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            response_mask = batch["response_mask"].to(device)
            prompt_mask = batch["prompt_mask"].to(device)

            # Sample random diffusion timesteps — DIMBA.forward requires t.
            t = torch.randint(
                0, model.num_diffusion_steps, (input_ids.shape[0],), device=device
            )
            # Model is bf16 natively — no autocast or GradScaler needed.
            # Pass prompt_mask so the prompt stays clean and pooled conditioning
            # is built from it, matching the conditioning path used at inference.
            x_pred, _, _ = model(input_ids, t, prompt_mask=prompt_mask)
            # Pass live embedding weight so weight-tying stays correct after resize.
            logits = model.output_head(x_pred, embedding_weight=model.token_embed.get_weight())
            loss_per_token = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                input_ids.reshape(-1),
                reduction="none",
            )
            loss = (loss_per_token * response_mask.reshape(-1)).sum() \
                   / response_mask.sum().clamp(min=1)
            (loss / accum).backward()

            running_loss += loss.item()
            running_n += 1

            micro_step += 1
            if micro_step % accum == 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
                optimizer.step()
                optimizer.zero_grad()
                opt_step += 1
                scheduler.step()

                # Log and save keyed on optimizer steps (not micro-batch steps)
                # so cadence is independent of grad_accum.
                if opt_step % cfg["log_interval"] == 0:
                    lr_now = scheduler.get_last_lr()[0]
                    avg_loss = running_loss / max(1, running_n)
                    logger.info("%s step %d | loss=%.4f | lr=%.2e",
                                label, opt_step, avg_loss, lr_now)
                    _write_state({"stage": label, "step": opt_step,
                                  "loss": avg_loss, "lr": lr_now})
                    running_loss, running_n = 0.0, 0

                if opt_step % cfg["save_interval"] == 0:
                    ckpt_path = os.path.join(save_dir, f"sft_step{opt_step}.pt")
                    torch.save({
                        "model_state_dict": getattr(model, '_orig_mod', model).state_dict(),
                        "config": getattr(model, "config", STUDENT_CFG),
                        "step": opt_step,
                        "phase": label,
                    }, ckpt_path)
                    logger.info("saved → %s", ckpt_path)

            global_step += 1

        # Flush any remaining accumulated gradients at epoch end.
        if micro_step % accum != 0:
            nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
            optimizer.step()
            optimizer.zero_grad()
            opt_step += 1
            scheduler.step()

    return opt_step


# ── phase 1: distillation ─────────────────────────────────────────────────────

class _PackedChunks(Dataset):
    """Fixed-length chunks of a packed token stream — every position is real content.

    Packing (concatenate docs, split into seq_len blocks) avoids padding entirely, so
    Stage-3 co-adaptation never trains the diffusion loss on pad/eos filler. Holds a
    pre-tokenized int64 tensor, so it pickles cleanly for fork/spawn DataLoader workers.
    One token stream is sliced at two seq_lens (align L=256, Stage-3 L=512).
    """

    def __init__(self, stream: torch.Tensor, seq_len: int) -> None:
        n = (stream.numel() // seq_len) * seq_len
        if n == 0:
            raise ValueError(f"token stream ({stream.numel()} toks) shorter than seq_len={seq_len}")
        self._data = stream[:n].view(-1, seq_len).contiguous()

    def __len__(self) -> int:
        return self._data.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self._data[idx]


def _build_fineweb_stream(tokenizer, n_cache: int = 50_000) -> torch.Tensor:
    """Stream FineWeb 'sample-10BT', tokenize the first *n_cache* docs, pack into one stream.

    Used ONLY for Stage 1+2 alignment (short-seq, small step count). Streaming means
    the dataset is never fully materialised to disk. Documents are separated by the eos
    token. Returns a flat ``[T]`` int64 tensor for the alignment DataLoader.

    Stage 3 co-adaptation uses _FineWebStreaming (true on-the-fly streaming, no RAM cache)
    so the multi-billion-token budget can scale without repeating data.
    """
    from datasets import load_dataset

    logger.info("building FineWeb pretrain cache (%d docs) …", n_cache)
    ds = load_dataset(PRETRAIN_DATASET, name=PRETRAIN_SUBSET,
                      split="train", streaming=True)
    eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    # Accumulate one small tensor per doc and cat once at the end. A flat Python int
    # list would balloon to ~8 GB at n_cache=400 K (each CPython int is ~28 B); per-doc
    # tensors keep peak RAM ~2x the final stream (~5 GB at ~290 M tokens).
    chunks: list = []
    i = -1
    for i, row in enumerate(ds):
        if i >= n_cache:
            break
        enc = tokenizer.encode(row["text"], add_special_tokens=False)
        enc.append(eos)
        chunks.append(torch.tensor(enc, dtype=torch.long))
    stream = torch.cat(chunks) if chunks else torch.empty(0, dtype=torch.long)
    logger.info("pretrain cache ready: %d docs → %.1f M packed tokens",
                min(n_cache, i + 1), stream.numel() / 1e6)
    return stream


def _make_fineweb_loader(stream: torch.Tensor, seq_len: int,
                         batch_size: int, num_workers: int = 4) -> DataLoader:
    """DataLoader of packed seq_len chunks over the shared FineWeb token stream.

    The loader cycles; the trainer stops at the configured step count.
    """
    ds = _PackedChunks(stream, seq_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, pin_memory=True,
                      persistent_workers=(num_workers > 0))


# ── Stage-3 true streaming dataset ────────────────────────────────────────────

# sample-10BT covers ~10B tokens. For runs that need more unique tokens, switch
# PRETRAIN_SUBSET to a larger FineWeb config (e.g. "CC-MAIN-2024-10" or the full
# "default" dump, see https://huggingface.co/datasets/HuggingFaceFW/fineweb).
# This constant is referenced by the budget-check warning below.
_FINEWEB_SUBSET_APPROX_TOKENS = 10_000_000_000  # sample-10BT ≈ 10 B tokens


class _FineWebStreaming(IterableDataset):
    """True streaming IterableDataset over FineWeb — no RAM cache.

    Each DataLoader worker shards the HuggingFace IterableDataset via
    ``ds.shard(num_shards=num_workers, index=worker_id)`` so that workers
    produce DISJOINT documents (no duplicate data). Tokenization and seq_len
    packing are done on the fly.

    Streams continuously; the trainer stops at its configured step count.
    sample-10BT is ~10 B tokens. If the total Stage-3 budget exceeds
    _FINEWEB_SUBSET_APPROX_TOKENS, data will loop. See the comment above
    that constant for how to switch to a larger FineWeb subset.
    """

    def __init__(self, tokenizer, seq_len: int) -> None:
        super().__init__()
        self._tokenizer = tokenizer
        self._seq_len = seq_len

    def __iter__(self):
        import itertools
        from datasets import load_dataset

        worker_info = torch.utils.data.get_worker_info()

        ds = load_dataset(PRETRAIN_DATASET, name=PRETRAIN_SUBSET,
                          split="train", streaming=True)

        if worker_info is not None and worker_info.num_workers > 1:
            # Each worker gets a disjoint slice of documents.
            ds = ds.shard(num_shards=worker_info.num_workers,
                          index=worker_info.id)

        eos = (self._tokenizer.eos_token_id
               if self._tokenizer.eos_token_id is not None else 0)
        seq_len = self._seq_len

        buf: list[int] = []
        # Repeat the stream so training never exhausts it.
        for row in itertools.cycle(ds):
            enc = self._tokenizer.encode(row["text"], add_special_tokens=False)
            enc.append(eos)
            buf.extend(enc)
            while len(buf) >= seq_len:
                yield torch.tensor(buf[:seq_len], dtype=torch.long)
                buf = buf[seq_len:]


def _make_fineweb_streaming_loader(tokenizer, seq_len: int,
                                   batch_size: int,
                                   num_workers: int = 4) -> DataLoader:
    """DataLoader backed by _FineWebStreaming — no shuffle, no RAM tensor.

    Each worker tokenizes and packs its own disjoint shard of FineWeb on the
    fly, so the Stage-3 co-adaptation budget can be arbitrarily large without
    cycling the same documents. The trainer stops at its step count.

    Notes:
        * No shuffle: streaming order is effectively random (CommonCrawl shards
          are shuffled at build time), so per-worker sequential reads are fine.
        * pin_memory=True: async DMA to GPU; harmless on CPU-only boxes.
        * persistent_workers=True (when num_workers>0): avoids re-launching the
          dataset iterator (and the HuggingFace streaming connection) on every
          DataLoader exhaustion cycle.
    """
    total_stage3_tokens = STAGE3_FROZEN_TOKENS + STAGE3_UNFROZEN_TOKENS
    if total_stage3_tokens > _FINEWEB_SUBSET_APPROX_TOKENS:
        logger.warning(
            "Stage-3 budget (%.1f B tokens) exceeds sample-10BT size (~10 B tokens). "
            "Data will loop. For >10 B unique-token runs set PRETRAIN_SUBSET to a "
            "larger FineWeb config (e.g. 'CC-MAIN-2024-10' or 'default').",
            total_stage3_tokens / 1e9,
        )
    ds = _FineWebStreaming(tokenizer, seq_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True,
                      persistent_workers=(num_workers > 0))


def _live_mixer_is_torch(model) -> bool:
    """True if the model's mixers are the pure-PyTorch TorchMamba2 (vs CUDA Mamba2).

    Detects the ACTUAL backend in use — covers both an explicit force_torch_mixer build
    and the case where mamba_ssm.Mamba2 can't be constructed and ``_make_mixer`` silently
    fell back to TorchMamba2.
    """
    base = getattr(model, "_orig_mod", model)
    try:
        return type(base.denoiser.blocks[0].mamba_fwd).__name__ == "TorchMamba2"
    except Exception:  # noqa: BLE001 — never let a probe crash training
        return False


def _save_config_with_backend(model) -> dict:
    """``model.config`` plus ``force_torch_mixer=True`` iff the live backend is TorchMamba2.

    ``DIMBA._config`` deliberately omits force_torch_mixer so a CUDA checkpoint reloads on
    the fast kernel. But if this model is actually running on TorchMamba2 (CUDA kernel
    unavailable), the next phase MUST rebuild on the same backend to load the weights —
    so we pin the flag based on the real backend, not on which build path was requested.
    """
    base = getattr(model, "_orig_mod", model)
    cfg = dict(base.config)
    if _live_mixer_is_torch(model):
        cfg["force_torch_mixer"] = True
    return cfg


def _enable_torch_memory_savings(model) -> bool:
    """Turn on gradient checkpointing iff the model runs on TorchMamba2.

    The pure-PyTorch SSD scan is memory-heavy, so checkpointing (recompute blocks in the
    backward pass) is needed to fit. Returns True iff the torch backend is in use, so the
    caller can also shrink the batch. No-op on the fast CUDA kernel.
    """
    if not _live_mixer_is_torch(model):
        return False
    getattr(model, "_orig_mod", model).denoiser.use_gradient_checkpointing = True
    return True


def _shrink_batch_for_torch(cfg: dict, max_bs: int = 16) -> dict:
    """Cap batch_size at max_bs, scaling grad_accum up to preserve the effective batch.

    Used for the memory-heavy TorchMamba2 path. Returns a copy; the original is unchanged.
    """
    bs = cfg.get("batch_size", max_bs)
    if bs <= max_bs:
        return dict(cfg)
    out = dict(cfg)
    factor = (bs + max_bs - 1) // max_bs
    out["batch_size"] = max_bs
    out["grad_accum"] = cfg.get("grad_accum", 1) * factor
    return out


def _run_stage3_resilient(trainer, stage_cfg, tokenizer, seq_len, batch, device,
                          min_batch: int = 1) -> int:
    """Run a distillation stage, halving the batch and retrying on CUDA OOM.

    This is the graceful "reallocate instead of crash" path: an OOM frees the loader,
    empties the cache, and re-runs the stage at half the batch (down to ``min_batch``).
    Stage 3 builds a fresh optimiser each attempt, so a retry is a clean restart — and
    OOM almost always fires on the first step, so little work is lost. Returns the batch
    size that ultimately succeeded.

    Uses the TRUE STREAMING loader (_make_fineweb_streaming_loader) so that the
    Stage-3 co-adaptation budget can scale to 5 B+ tokens without cycling the same
    ~290 M cached tokens. The streaming loader is rebuilt on each OOM retry so the
    worker processes are cleanly replaced after memory is freed.
    """
    while True:
        loader = _make_fineweb_streaming_loader(tokenizer, seq_len, batch, num_workers=2)
        try:
            logger.info("Stage 3: co-adaptation at batch=%d, L=%d", batch, seq_len)
            trainer.run_stage(stage_cfg, loader)
            return batch
        except torch.cuda.OutOfMemoryError:
            del loader
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if batch <= min_batch:
                logger.error("CUDA OOM even at batch=%d — cannot proceed.", batch)
                raise
            new_batch = max(min_batch, batch // 2)
            logger.warning("CUDA OOM at batch=%d → freed memory, retrying Stage 3 at batch=%d.",
                           batch, new_batch)
            batch = new_batch


def run_distill(
    device: torch.device,
    save_dir: str = "./checkpoints/distill",
    resume: Optional[str] = None,
    hf_token: Optional[str] = None,
    hf_repo: Optional[str] = None,
) -> str:
    os.makedirs(save_dir, exist_ok=True)
    logger.info("=== PHASE 1: DISTILLATION (MOHAWK-style) ===")
    logger.info("Stage 1+2: matrix/hidden alignment (short seq, teacher active)")
    logger.info("Stage 3:   co-adaptation pretraining (~FFN frozen, teacher unused)")

    cfg = DistillationConfig.from_dict({**DISTILL_CFG, "device": str(device)})
    # Split the configured stages into alignment (stage1/stage2, teacher active, short
    # seq) and co-adaptation (one or more stage3 phases, teacher unused, full seq).
    # Ordered lists (not a name→stage dict) let us run MULTIPLE stage3 phases — the
    # FFN-frozen phase followed by the FFN-unfrozen "Finetune #2".
    align_stages = [s for s in DISTILL_CFG["stages"] if s["name"] in ("stage1", "stage2")]
    coadapt_stages = [s for s in DISTILL_CFG["stages"] if s["name"] == "stage3"]

    compute_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    teacher = TeacherWrapper(TEACHER_MODEL, device=str(device), dtype=compute_dtype)

    # ── Build the student on the BEST available backend ─────────────────────────
    # No force_torch_mixer: with mamba_ssm installed this uses the fast CUDA Mamba2
    # kernel; only a CPU/no-mamba box falls back to pure-PyTorch TorchMamba2. The
    # Stage-1 mixing matrices are computed from whichever mixer's live parameters
    # (denoiser._ssd_mixing_matrix_from_params handles the CUDA kernel, which has no
    # materialize method), so no cross-backend weight transfer is needed.
    model, _layer_map = build_student_from_teacher(
        teacher,
        num_diffusion_steps=1000,
        d_state=64,
        d_conv=4,
        expand=2,
        conditioning_type=STUDENT_CFG["conditioning_type"],
        use_flow_matching=STUDENT_CFG["use_flow_matching"],
        flow_logit_normal=STUDENT_CFG["flow_logit_normal"],
        use_weight_tying=True,
        use_simple_mamba=False,
        block_ffn=DISTILL_CFG["block_ffn"],
        inherit_embeddings=DISTILL_CFG["inherit_embeddings"],
        inherit_ffn=DISTILL_CFG["inherit_ffn"],
        inherit_head=DISTILL_CFG["inherit_head"],
    )

    if resume and os.path.isfile(resume):
        ckpt = torch.load(resume, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        logger.info("resumed distillation from %s", resume)

    model = model.to(device)
    if device.type == "cuda":
        model = model.to(torch.bfloat16)

    on_torch = _live_mixer_is_torch(model)
    logger.info("distill backend: %s", "TorchMamba2 (pure PyTorch — CPU/fallback)"
                if on_torch else "CUDA Mamba2 kernel (fast binary)")
    # Only the memory-heavy pure-PyTorch fallback needs checkpointing + a small batch;
    # the CUDA binary kernel handles the full batch efficiently.
    if on_torch:
        model.denoiser.use_gradient_checkpointing = True
    _log_vram(device, "post-model-load")

    tokenizer = _load_tokenizer(TEACHER_MODEL)
    # Stage 1+2 alignment: cache a small slice (400 K docs ≈ 290 M tokens) for the
    # short-sequence align passes. The cache is fine here — these stages are tiny
    # (~1 K steps each) and the cached loader avoids re-downloading on every call.
    stream = _build_fineweb_stream(tokenizer, n_cache=400_000)
    align_loader = _make_fineweb_loader(stream, ALIGN_SEQ_LEN, ALIGN_BATCH, num_workers=2)

    # ── Stages 1 + 2: alignment (Stage-1 matrices from the live mixer params) ───
    trainer = DistillationTrainer(model=model, teacher=teacher, config=cfg)
    for stage_cfg in align_stages:
        logger.info("alignment %s on short-seq loader (L=%d, B=%d)",
                    stage_cfg["name"], ALIGN_SEQ_LEN, ALIGN_BATCH)
        trainer.run_stage(stage_cfg, align_loader)
    if getattr(trainer, "_stage1_warned", False):
        logger.warning("Stage 1 matrix alignment fell back to a NO-OP — matrices could "
                       "not be materialized (see warning above).")
    else:
        logger.info("Stage 1+2 alignment complete (matrices materialized).")
    del align_loader
    # Free the RAM cache used by alignment stages — Stage 3 uses the true streaming
    # loader and doesn't need the cached tensor (which would hold ~2-5 GB of RAM).
    del stream
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # The teacher is unused in Stage 3 (kd_weight=0) — free its VRAM. The wrapper stays
    # alive so its metadata (num_layers, …) still answers any trainer query.
    teacher.unload()
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    _log_vram(device, "pre-stage3")

    # ── Stage 3: co-adaptation pretraining (FFN frozen, teacher unused) ─────────
    # Graceful OOM handling: halve the batch and retry instead of crashing.
    # Run the co-adaptation phases in order (FFN-frozen → FFN-unfrozen). Each
    # run_stage call re-applies its own trainability mask and builds a fresh AdamW,
    # so the unfrozen phase correctly unfreezes the FFN at its own (lower) LR.
    # Stage 3 uses _make_fineweb_streaming_loader (true streaming, no RAM tensor)
    # so the 5B+ token budget never cycles the same data.
    stage3_batch = PRETRAIN_BATCH_TORCH if on_torch else PRETRAIN_BATCH
    total_toks = 0
    for i, stage_cfg in enumerate(coadapt_stages):
        frozen = bool(stage_cfg.get("freeze_ffn", False))
        phase = chr(ord("a") + i)  # 3a, 3b, ...
        logger.info("Stage 3%s: co-adaptation — FFN %s, lr=%g, %d steps",
                    phase, "FROZEN" if frozen else "UNFROZEN",
                    stage_cfg["lr"], int(stage_cfg["steps"]))
        eff_batch = _run_stage3_resilient(
            trainer, stage_cfg, tokenizer, PRETRAIN_SEQ_LEN, stage3_batch, device,
        )
        stage3_batch = eff_batch  # carry the OOM-survivable batch into the next phase
        total_toks += int(stage_cfg["steps"]) * eff_batch * PRETRAIN_SEQ_LEN
        # Checkpoint after each phase except the last so a crash in a later phase
        # (e.g. the new FFN-unfrozen Finetune #2) doesn't lose the frozen-FFN base.
        if i < len(coadapt_stages) - 1:
            inter_path = os.path.join(save_dir, f"distill_stage3{phase}.pt")
            torch.save({
                "model_state_dict": getattr(model, "_orig_mod", model).state_dict(),
                "config": _save_config_with_backend(model),
                "phase": "distill",
            }, inter_path)
            logger.info("saved intermediate co-adaptation checkpoint → %s", inter_path)
    if coadapt_stages:
        logger.info("Stage 3 (all phases) done at batch=%d → ~%.2f B tokens co-adaptation.",
                    stage3_batch, total_toks / 1e9)

    # ── Save — config pins the backend the checkpoint must be rebuilt on ────────
    final_path = os.path.join(save_dir, "final.pt")
    torch.save({
        "model_state_dict": getattr(model, "_orig_mod", model).state_dict(),
        "config": _save_config_with_backend(model),
        "phase": "distill",
    }, final_path)
    logger.info("distillation done → %s", final_path)
    _upload_checkpoint(final_path, hf_token, hf_repo, "distill/final.pt")
    return final_path


# ── phase 2: SFT (two-stage) ─────────────────────────────────────────────────

def run_sft(
    checkpoint: str,
    device: torch.device,
    save_dir: str = "./checkpoints/sft",
    resume: Optional[str] = None,
    use_frontier: bool = True,
    hf_token: Optional[str] = None,
    hf_repo: Optional[str] = None,
) -> str:
    """Two-stage SFT.

    Stage 1 — full mix (SmolTalk + OrcaMath + all Frontier CoT, ~284 K rows).
              2 epochs at lr=2e-5.  Teaches the model the block-CoT format and
              exposes it to the broad distribution of instruction + reasoning data.

    Stage 2 — hard subset (OrcaMath + Frontier CoT only, no SmolTalk).
              1 epoch at lr=5e-6.  Refines on examples that require real chain-of-
              thought reasoning; replicates VibeThinker's two-stage curriculum
              without needing an oracle model to label difficulty.
    """
    os.makedirs(save_dir, exist_ok=True)
    logger.info("=== PHASE 2: SFT (two-stage block-CoT) ===")
    # COHERENCE GATE: SFT (and GRPO) only pay off once the distilled BASE produces
    # coherent continuations. Run #1 was incoherent because the base was token-starved
    # and the FFN never co-adapted; SFT on limited instruction data could not rescue it
    # and GRPO accuracy stayed ~0 (nothing to optimise). Before running this phase,
    # confirm the base checkpoint passes the coherence gate in docs/NEXT_RUN_PLAN.md
    # (coherent ≥30-token continuations / perplexity below threshold).
    logger.warning(
        "SFT precondition: the base checkpoint must pass the coherence gate "
        "(see docs/NEXT_RUN_PLAN.md). SFT cannot fix an incoherent / undertrained base."
    )

    if resume:
        logger.warning(
            "SFT resume is not implemented; --resume is ignored for the SFT phase, "
            "optimizer/step state will restart from 0. Only model weights are loaded "
            "via the positional checkpoint argument."
        )

    model = _build_or_load_model(checkpoint, device)
    tokenizer = _load_tokenizer(TEACHER_MODEL)

    # Add <think>/</think> special tokens before compile so the embedding resize
    # happens as a structural mutation. Mirrors the run_grpo path so SFT trains
    # on the same block-CoT delimiters GRPO uses.
    if "<think>" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": ["<think>", "</think>"]})
        model.token_embed.resize(len(tokenizer))
        if hasattr(model.output_head, "embedding_weight"):
            model.output_head.embedding_weight = model.token_embed.get_weight()
    think_start_id = tokenizer.convert_tokens_to_ids("<think>")
    think_end_id = tokenizer.convert_tokens_to_ids("</think>")

    # If the model is on the memory-heavy TorchMamba2 backend, enable gradient
    # checkpointing and shrink the batch (grad_accum compensates to keep the effective
    # batch). On the fast CUDA kernel this is a no-op.
    on_torch = _enable_torch_memory_savings(model)
    sft_cfg = _shrink_batch_for_torch(SFT_CFG) if on_torch else SFT_CFG
    sft_stage2_cfg = _shrink_batch_for_torch(SFT_STAGE2_CFG) if on_torch else SFT_STAGE2_CFG
    if on_torch:
        logger.info("TorchMamba2 backend: gradient checkpointing ON; SFT batch %d→%d, "
                    "grad_accum→%d (effective batch unchanged).",
                    SFT_CFG["batch_size"], sft_cfg["batch_size"], sft_cfg["grad_accum"])

    # NOTE: torch.compile is intentionally NOT used here. mode="reduce-overhead" wraps
    # the model in CUDA Graphs, whose reused buffers alias across the SFT loop's
    # multi-call structure (forward → separate output_head → backward, with grad
    # accumulation), producing "accessing tensor output of CUDAGraphs that has been
    # overwritten by a subsequent run" on the first backward. The model already runs on
    # the fast mamba_ssm kernel, so eager mode is plenty and rock-solid.

    from torch.utils.data import ConcatDataset

    # ── Stage 1: full mix ─────────────────────────────────────────────────────
    logger.info("SFT Stage 1: loading full data mix …")
    logger.info("  loading SmolTalk …")
    smol = SmolTalkDataset(split="train", subset="smol-magpie-ultra")
    logger.info("  loading Orca-Math …")
    orca = OrcaMathDataset(split="train")
    parts = [smol, orca]
    if use_frontier:
        logger.info("  loading frontier CoT mix (~284 K rows across 7 sources) …")
        frontier = FrontierCoTMix()
        logger.info("  frontier CoT mix: %d examples", len(frontier))
        parts.append(frontier)

    stage1_data = ConcatDataset(parts)
    logger.info("Stage 1 dataset: %d total examples", len(stage1_data))

    _log_vram(device, "SFT-stage1 start")
    ds1 = _make_block_cot_dataset(stage1_data, tokenizer, sft_cfg, think_start_id, think_end_id)
    loader1 = _make_loader(ds1, sft_cfg)
    global_step = _sft_loop(model, loader1, sft_cfg, device, save_dir, "sft_s1")

    # ── Stage 2: hard subset (reasoning-only data) ────────────────────────────
    logger.info("SFT Stage 2: hard subset (Orca-Math + Frontier CoT) …")
    hard_parts = [orca]
    if use_frontier:
        hard_parts.append(frontier)
    stage2_data = ConcatDataset(hard_parts)
    logger.info("Stage 2 dataset: %d examples (dropped SmolTalk direct-answer examples)", len(stage2_data))

    ds2 = _make_block_cot_dataset(stage2_data, tokenizer, sft_stage2_cfg, think_start_id, think_end_id)
    loader2 = _make_loader(ds2, sft_stage2_cfg)
    _log_vram(device, "SFT-stage2 start")
    global_step = _sft_loop(model, loader2, sft_stage2_cfg, device, save_dir, "sft_s2",
                             start_step=global_step)

    final_path = os.path.join(save_dir, "final.pt")
    torch.save({
        "model_state_dict": getattr(model, '_orig_mod', model).state_dict(),
        "config": _save_config_with_backend(model),
        "phase": "sft",
        "total_steps": global_step,
    }, final_path)
    logger.info("SFT done → %s  (total steps: %d)", final_path, global_step)
    _upload_checkpoint(final_path, hf_token, hf_repo, "sft/final.pt")
    return final_path


# ── phase 3: GRPO ─────────────────────────────────────────────────────────────

def _load_grpo_data(data: str):
    """Load prompt/reference lists for the given GRPO domain.

    Returns a list of (prompts, references, label) tuples, one per domain pass.
    For data='seq', returns two passes: math then code.
    """
    passes = []

    if data in ("math", "orca", "both", "seq"):
        prompts, refs = [], []
        orca_ds = OrcaMathDataset(split="train", max_examples=20000)
        for i in range(len(orca_ds)):
            item = orca_ds[i]
            prompts.append(item["prompt"])
            refs.append(item["response"])
        passes.append((prompts, refs, "math"))

    if data in ("smoltalk", "both"):
        prompts, refs = [], []
        smol_ds = SmolTalkDataset(split="train", max_examples=10000)
        for i in range(len(smol_ds)):
            item = smol_ds[i]
            prompts.append(item["prompt"])
            refs.append(item["response"])
        passes.append((prompts, refs, "general"))

    if data in ("code", "seq"):
        # Bespoke-Stratos has ~5 K code + 10 K math; use the full set (code-heavy
        # problems are identified by the model's group accuracy hitting the boundary,
        # so MGPO will naturally upweight those even in the mixed dataset).
        prompts, refs = [], []
        stratos = BespokeStratosDataset(split="train")
        for i in range(len(stratos)):
            item = stratos[i]
            if item["reasoning"] or item["response"]:
                prompts.append(item["prompt"])
                refs.append(item["response"])
        passes.append((prompts, refs, "code"))

    if not passes:
        raise ValueError(f"Unknown --grpo-data option: {data!r}")

    return passes


def _run_grpo_pass(
    model: nn.Module,
    tokenizer,
    think_start_id: int,
    think_end_id: int,
    prompts: list,
    references: list,
    label: str,
    num_steps: int,
    device: torch.device,
    save_dir: str,
    resume: Optional[str],
    resume_weights_only: bool = False,
) -> str:
    """One GRPO domain pass.  Returns path to final checkpoint.

    resume_weights_only=True (the sequential cross-domain handoff, e.g. math→code)
    restores only the shared model's weights and gives this pass a FRESH LR schedule +
    step counter — otherwise the new domain inherits the previous pass's exhausted
    cosine scheduler and trains the whole pass pinned at the LR floor.
    """
    if not prompts:
        raise ValueError(
            f"GRPO domain {label!r} produced 0 prompts — check the dataset schema "
            f"(e.g. the Bespoke-Stratos 'conversations' column may have drifted). "
            f"Failing fast instead of dividing by zero mid-pass."
        )
    from dataclasses import replace as _dc_replace
    grpo_cfg = _dc_replace(
        GRPO_CFG,
        think_start_id=think_start_id,
        think_end_id=think_end_id,
        eos_id=tokenizer.eos_token_id,
        total_steps=num_steps,
        save_dir=save_dir,
        state_file="./training_state.json",
    )

    if label == "code":
        trainer = build_code_grpo_trainer(model, tokenizer, grpo_cfg)
    elif label == "general":
        trainer = build_general_grpo_trainer(model, tokenizer, grpo_cfg)
    else:
        trainer = build_math_grpo_trainer(model, tokenizer, grpo_cfg)

    if resume and os.path.isfile(resume):
        trainer.load_checkpoint(resume, weights_only=resume_weights_only)

    logger.info("GRPO pass: %s | %d prompts | %d steps", label, len(prompts), num_steps)
    _log_vram(device, f"GRPO-{label} start")

    for step_idx in range(num_steps):
        idx = step_idx % len(prompts)
        metrics = trainer.train_step([prompts[idx]], [references[idx]])

        if step_idx % 50 == 0:
            logger.info(
                "GRPO[%s] step %d/%d | reward=%.3f | think=%.2f | "
                "kl=%.4f | acc=%.2f | lr=%.2e",
                label, step_idx, num_steps,
                metrics["mean_reward"], metrics["mean_think_blocks"],
                metrics["kl"], metrics["group_accuracy"], metrics["lr"],
            )

    pass_path = os.path.join(save_dir, f"grpo_{label}_final.pt")
    return trainer.save_checkpoint(pass_path)


def run_grpo(
    checkpoint: str,
    device: torch.device,
    save_dir: str = "./checkpoints/grpo",
    resume: Optional[str] = None,
    num_steps: int = 2000,
    data: str = "math",
    hf_token: Optional[str] = None,
    hf_repo: Optional[str] = None,
) -> str:
    """GRPO/MGPO policy optimisation.

    data='seq': runs math RL (num_steps//2) then code RL (num_steps//2) sequentially.
    All other values: single-domain pass.
    """
    os.makedirs(save_dir, exist_ok=True)
    logger.info("=== PHASE 3: GRPO (MGPO + Long2Short + flow sampler) ===")
    logger.info("MGPO gamma=%.1f | Long2Short λ=%.2f | group_size=%d",
                GRPO_CFG.mgpo_gamma, GRPO_CFG.long2short_lambda, GRPO_CFG.group_size)
    # COHERENCE GATE: GRPO optimises verifiable rewards over sampled completions. If the
    # SFT model cannot produce coherent, occasionally-correct completions, every group is
    # uniformly wrong, advantages are ~0, and GRPO makes no progress (run #1: accuracy ~0).
    logger.warning(
        "GRPO precondition: the SFT model must already produce coherent, sometimes-correct "
        "completions (see docs/NEXT_RUN_PLAN.md). GRPO cannot bootstrap signal from an "
        "incoherent policy."
    )

    model = _build_or_load_model(checkpoint, device)
    tokenizer = _load_tokenizer(TEACHER_MODEL)

    special_tokens = {}
    if "<think>" not in tokenizer.get_vocab():
        special_tokens["additional_special_tokens"] = ["<think>", "</think>"]
        tokenizer.add_special_tokens(special_tokens)
        model.token_embed.resize(len(tokenizer))
        # Sync the output head's weight-tying buffer with the new embedding table.
        if hasattr(model.output_head, "embedding_weight"):
            model.output_head.embedding_weight = model.token_embed.get_weight()

    think_start_id = tokenizer.convert_tokens_to_ids("<think>")
    think_end_id = tokenizer.convert_tokens_to_ids("</think>")

    # On the memory-heavy TorchMamba2 backend, enable gradient checkpointing for the
    # policy/ref log-prob forward passes (generation runs under no_grad already).
    on_torch = _enable_torch_memory_savings(model)
    if on_torch:
        logger.info("TorchMamba2 backend: gradient checkpointing ON for GRPO.")

    # NOTE: torch.compile is intentionally NOT used here. reduce-overhead's CUDA Graphs
    # alias buffers across GRPO's many model calls (generation under no_grad, ELBO
    # log-prob forwards, backward), causing the same CUDAGraphs-overwrite crash SFT hit;
    # and GRPO's variable generation lengths would thrash recompilation anyway. The
    # mamba_ssm kernel is already fast, so eager mode is the reliable choice.

    passes = _load_grpo_data(data)

    # For sequential (math → code): split steps evenly across passes.
    steps_per_pass = num_steps // len(passes)
    last_ckpt = None

    for i, (prompts, refs, label) in enumerate(passes):
        pass_dir = os.path.join(save_dir, label)
        os.makedirs(pass_dir, exist_ok=True)
        # Only pass the original --resume into the first domain pass. For later passes
        # the shared model already carries the trained weights, so we restore weights
        # only and give each new domain a FRESH LR schedule (resume_weights_only=True);
        # restoring the prior pass's exhausted cosine scheduler would pin it at the LR floor.
        pass_resume = resume if i == 0 else last_ckpt
        last_ckpt = _run_grpo_pass(
            model, tokenizer, think_start_id, think_end_id,
            prompts, refs, label, steps_per_pass,
            device, pass_dir, pass_resume,
            resume_weights_only=(i > 0),
        )
        logger.info("GRPO domain pass '%s' done → %s", label, last_ckpt)

    # Copy the very last checkpoint to a canonical final.pt
    final_path = os.path.join(save_dir, "final.pt")
    shutil.copy2(last_ckpt, final_path)
    logger.info("GRPO done → %s", final_path)
    _upload_checkpoint(final_path, hf_token, hf_repo, "grpo/final.pt")
    return final_path


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase", choices=["distill", "sft", "grpo", "all"], default="all",
                   help="Which phase to run (default: all)")
    p.add_argument("--checkpoint", default=None,
                   help="Path to a .pt checkpoint to load (skips building from teacher)")
    p.add_argument("--resume", action="store_true",
                   help="Resume from the checkpoint instead of starting fresh")
    p.add_argument("--device", default="cuda",
                   help="Device string (default: cuda)")
    p.add_argument("--grpo-steps", type=int, default=2000,
                   help="Total GRPO update steps (split evenly across domain passes for --grpo-data seq)")
    p.add_argument("--grpo-data",
                   choices=["math", "orca", "code", "seq", "smoltalk", "both"],
                   default="math",
                   help=(
                       "Data for GRPO rollouts: "
                       "'math'/'orca' = OrcaMath only; "
                       "'code' = Bespoke-Stratos; "
                       "'seq' = sequential math→code; "
                       "'both' = sequential math→general (SmolTalk)"
                   ))
    p.add_argument("--teacher", default=TEACHER_MODEL,
                   help=f"HuggingFace teacher model (default: {TEACHER_MODEL})")
    p.add_argument("--no-flow", action="store_true",
                   help="Disable flow matching (fall back to DDPM cosine schedule)")
    p.add_argument("--no-frontier", action="store_true",
                   help="Skip frontier CoT datasets in SFT (use only SmolTalk + Orca-Math)")
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"),
                   help="HuggingFace API token for auto-upload (or set HF_TOKEN env var)")
    p.add_argument("--hf-repo", default=None,
                   help="HuggingFace repo to upload checkpoints to, e.g. 'yourusername/dimba-135m'")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        logger.info("GPU: %s  |  VRAM: %.1f GB  |  SM: %d.%d",
                    props.name, props.total_memory / 1e9,
                    props.major, props.minor)

    if args.no_flow:
        STUDENT_CFG["use_flow_matching"] = False
        GRPO_CFG.num_diffusion_steps_inference = 30
        GRPO_CFG.sampler = "dpmpp"
        logger.info("flow matching disabled (--no-flow)")

    global TEACHER_MODEL
    TEACHER_MODEL = args.teacher
    DISTILL_CFG["teacher_model"] = args.teacher

    # Normalise alias: --grpo-data orca is the same as math
    grpo_data = args.grpo_data if args.grpo_data != "orca" else "math"

    # user_ckpt is the checkpoint the user provided via --checkpoint.
    # --resume is only meaningful for a single explicitly-selected phase: it
    # restores optimizer/scheduler state from that same checkpoint. In --phase all
    # mode the cross-phase weight handoff (distill->sft->grpo) uses the positional
    # checkpoint argument; passing auto-generated previous-phase final.pt as
    # resume= would inject an SFT-only file into GRPOTrainer.load_checkpoint,
    # which only makes sense for genuine GRPO checkpoints.
    user_ckpt = args.checkpoint

    def _resume_for(phase: str) -> Optional[str]:
        # Within-phase resume is only meaningful when the user explicitly selects
        # that single phase AND provides a checkpoint that was saved by that phase.
        return user_ckpt if (args.resume and args.phase == phase) else None

    ckpt = args.checkpoint

    hf_kw = dict(hf_token=args.hf_token, hf_repo=args.hf_repo)
    if args.hf_repo:
        logger.info("HuggingFace upload enabled → %s", args.hf_repo)

    if args.phase in ("distill", "all"):
        ckpt = run_distill(device, resume=_resume_for("distill"), **hf_kw)

    if args.phase in ("sft", "all"):
        ckpt = run_sft(ckpt, device, resume=_resume_for("sft"),
                       use_frontier=not args.no_frontier, **hf_kw)

    if args.phase in ("grpo", "all"):
        ckpt = run_grpo(ckpt, device,
                        resume=_resume_for("grpo"),
                        num_steps=args.grpo_steps,
                        data=grpo_data,
                        **hf_kw)

    logger.info("all done. final model: %s", ckpt)


if __name__ == "__main__":
    main()
