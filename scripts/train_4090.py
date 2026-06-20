#!/usr/bin/env python3
"""Full DIMBA training pipeline optimised for a 48 GB GPU (RTX 4090 modded / L40S / A6000 Ada).

Phases
------
  1. distill   Cross-architecture distillation from a HF teacher (SmolLM-135M default).
               Runs Stage 1 (matrix align) → Stage 2 (hidden align) → Stage 3 (diffusion ft).
  2. sft        Supervised fine-tune on block-CoT reasoning data (SmolTalk + Orca-Math).
  3. grpo       GRPO policy optimisation with verifiable rewards + anti-overthinking.

Quick start (runs all three phases end-to-end):
    python scripts/train_4090.py

Run a single phase:
    python scripts/train_4090.py --phase distill
    python scripts/train_4090.py --phase sft   --checkpoint checkpoints/distill/final.pt
    python scripts/train_4090.py --phase grpo  --checkpoint checkpoints/sft/final.pt

Resume from a checkpoint:
    python scripts/train_4090.py --phase grpo --checkpoint checkpoints/grpo/grpo_step200.pt --resume

Flow matching is ON by default (15 Euler steps ≈ 2× faster inference than 30 DPM++ steps).

The script writes ./training_state.json after every log_interval steps so that a
Claude /loop monitor (scripts/monitor.py) can read progress and adjust config.
"""
import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ── add project root to path ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dimba.models.diffusion import DIMBA
from dimba.distillation.trainer import DistillationTrainer, DistillationConfig
from dimba.distillation.surgery import build_student_from_teacher
from dimba.distillation.teacher import TeacherWrapper
from dimba.training.grpo import GRPOConfig, GRPOTrainer, build_math_grpo_trainer
from dimba.training.rewards import CompositeReward, NumericAnswerReward, LengthPenaltyReward
from dimba.data.cot_dataset import SmolTalkDataset, OrcaMathDataset, BlockCoTDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_4090")

# ── hardware flags (set before any CUDA calls) ────────────────────────────────
# tf32 gives free ~10% matmul speedup on Ampere/Ada with no quality loss.
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
    conditioning_type="adaln",  # AdaLN-Zero (DiT-style, better than FiLM)
    use_flow_matching=True,     # rectified flow — ~2-3× faster inference
    flow_logit_normal=True,     # logit-normal timestep sampling (SD3/FLUX schedule)
    use_weight_tying=True,
    use_simple_mamba=False,
    block_ffn=True,             # inherit teacher FFN (Mode A)
    ffn_type="swiglu",          # SmolLM uses SwiGLU
)

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
        # Stage 1: align Mamba mixing matrices → teacher attention maps (clean pass)
        {"name": "stage1", "steps": 500,  "lr": 1e-3},
        # Stage 2: align student hidden states → teacher residual stream (clean pass)
        {"name": "stage2", "steps": 500,  "lr": 5e-4},
        # Stage 3: diffusion objective (flow matching) + optional logit-KD
        # More steps than before: flow matching needs a short warmup before loss
        # drops because the velocity target is a different scale than DDPM MSE.
        {"name": "stage3", "steps": 3000, "lr": 2e-4,
         "ce_loss_weight": 1.0, "min_snr_gamma": 5.0},
    ],
)

# SFT hyperparams — 48 GB lets us double the batch size vs 24 GB
SFT_CFG = dict(
    batch_size=64,          # 48 GB → 64 (was 32 on 24 GB)
    grad_accum=2,           # effective batch = 128 (same as before)
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
    num_workers=8,          # more CPU workers for fast NVMe on Vast
    prefetch_factor=2,
)

# GRPO hyperparams (anti-overthinking tuned for 135M + flow matching)
GRPO_CFG = GRPOConfig(
    block_size=64,
    max_think_blocks=2,          # hard cap — prevents overthinking
    response_len=128,
    adaptive_stop=True,
    thinking_length_weight=0.02, # 0.02 × 64 × n_blocks subtracted from reward
    min_response_reward_bonus=0.05,
    group_size=8,
    kl_coeff=0.04,
    mc_samples=2,
    antithetic=True,
    lr=5e-6,
    weight_decay=0.01,
    max_grad_norm=1.0,
    warmup_steps=50,
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
        ckpt = torch.load(checkpoint, map_location="cpu")
        cfg = ckpt.get("config", STUDENT_CFG)
        model = DIMBA(**cfg)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        logger.info("building fresh student model (Mode A: inherit teacher weights)")
        teacher = TeacherWrapper(TEACHER_MODEL, device=str(device))
        model = build_student_from_teacher(
            teacher,
            extra_student_kwargs=dict(
                num_diffusion_steps=1000,
                d_state=64,
                d_conv=4,
                expand=2,
                conditioning_type="adaln",
                use_flow_matching=True,
                flow_logit_normal=True,
                use_simple_mamba=False,
            ),
        )
        teacher.unload()

    model = model.to(device)
    if device.type == "cuda":
        model = model.to(torch.bfloat16)  # bf16 storage; no GradScaler needed
        model = torch.compile(model, mode="reduce-overhead")
    return model


def _log_vram(device: torch.device, tag: str = "") -> None:
    if device.type != "cuda":
        return
    used = torch.cuda.memory_allocated(device) / 1e9
    reserved = torch.cuda.memory_reserved(device) / 1e9
    logger.info("VRAM %s: %.1f GB allocated / %.1f GB reserved", tag, used, reserved)


# ── phase 1: distillation ─────────────────────────────────────────────────────

def run_distill(
    device: torch.device,
    save_dir: str = "./checkpoints/distill",
    resume: Optional[str] = None,
) -> str:
    os.makedirs(save_dir, exist_ok=True)
    logger.info("=== PHASE 1: DISTILLATION ===")

    cfg = DistillationConfig(**DISTILL_CFG)
    cfg.device = str(device)

    teacher = TeacherWrapper(TEACHER_MODEL, device=str(device))
    model = build_student_from_teacher(
        teacher,
        extra_student_kwargs=dict(
            use_flow_matching=True,
            flow_logit_normal=True,
            conditioning_type="adaln",
        ),
    )

    if resume and os.path.isfile(resume):
        ckpt = torch.load(resume, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        logger.info("resumed distillation from %s", resume)

    model = model.to(device)
    if device.type == "cuda":
        model = model.to(torch.bfloat16)
    _log_vram(device, "post-model-load")

    trainer = DistillationTrainer(
        student=model,
        teacher=teacher,
        config=cfg,
        state_file="./training_state.json",
    )
    trainer.run()

    final_path = os.path.join(save_dir, "final.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": model.config,
        "phase": "distill",
    }, final_path)
    logger.info("distillation done → %s", final_path)
    return final_path


# ── phase 2: SFT ─────────────────────────────────────────────────────────────

def run_sft(
    checkpoint: str,
    device: torch.device,
    save_dir: str = "./checkpoints/sft",
    resume: Optional[str] = None,
) -> str:
    os.makedirs(save_dir, exist_ok=True)
    logger.info("=== PHASE 2: SFT (block-CoT) ===")

    cfg = SFT_CFG
    model = _build_or_load_model(checkpoint, device)
    tokenizer = _load_tokenizer(TEACHER_MODEL)

    think_start_id = tokenizer.convert_tokens_to_ids("<think>") if "<think>" in tokenizer.get_vocab() else None
    think_end_id = tokenizer.convert_tokens_to_ids("</think>") if "</think>" in tokenizer.get_vocab() else None

    logger.info("loading SmolTalk …")
    smol = SmolTalkDataset(split="train", subset="smol-magpie-ultra")
    logger.info("loading Orca-Math …")
    orca = OrcaMathDataset(split="train")

    from torch.utils.data import ConcatDataset
    combined = ConcatDataset([smol, orca])

    def _tok_fn(text: str):
        return tokenizer.encode(text, add_special_tokens=False)

    ds = BlockCoTDataset(
        combined,
        tokenizer=_tok_fn,
        max_prompt_len=128,
        block_size=cfg["block_size"],
        num_think_blocks=cfg["num_think_blocks"],
        response_len=cfg["response_len"],
        think_start_id=think_start_id,
        think_end_id=think_end_id,
        pad_id=tokenizer.pad_token_id or 0,
    )
    loader = DataLoader(
        ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        prefetch_factor=cfg["prefetch_factor"],
        persistent_workers=True,
    )

    _fused = device.type == "cuda"
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"],
        fused=_fused,
    )
    total_steps = cfg["num_epochs"] * len(loader)
    warmup = cfg["warmup_steps"]

    def _lr_lambda(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    start_step = 0
    if resume and os.path.isfile(resume):
        ckpt = torch.load(resume, map_location="cpu")
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt.get("step", 0)

    global_step = start_step
    accum = cfg["grad_accum"]
    model.train()
    _log_vram(device, "SFT start")

    for epoch in range(cfg["num_epochs"]):
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            response_mask = batch["response_mask"].to(device)

            # Model is bf16; no autocast or GradScaler needed
            x_pred, _, _ = model(input_ids)
            logits = model.output_head(x_pred)             # [B, L, vocab]
            loss_per_token = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                input_ids.reshape(-1),
                reduction="none",
            )
            loss = (loss_per_token * response_mask.reshape(-1)).sum() \
                   / response_mask.sum().clamp(min=1)
            (loss / accum).backward()

            if (global_step + 1) % accum == 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            global_step += 1

            if global_step % cfg["log_interval"] == 0:
                lr_now = scheduler.get_last_lr()[0]
                logger.info("SFT step %d | loss=%.4f | lr=%.2e",
                            global_step, loss.item() * accum, lr_now)
                _write_state({"stage": "sft", "step": global_step,
                              "loss": loss.item() * accum, "lr": lr_now})

            if global_step % cfg["save_interval"] == 0:
                ckpt_path = os.path.join(save_dir, f"sft_step{global_step}.pt")
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": getattr(model, "config", STUDENT_CFG),
                    "step": global_step,
                    "phase": "sft",
                }, ckpt_path)
                logger.info("saved → %s", ckpt_path)

    final_path = os.path.join(save_dir, "final.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": getattr(model, "config", STUDENT_CFG),
        "phase": "sft",
    }, final_path)
    logger.info("SFT done → %s", final_path)
    return final_path


# ── phase 3: GRPO ─────────────────────────────────────────────────────────────

def run_grpo(
    checkpoint: str,
    device: torch.device,
    save_dir: str = "./checkpoints/grpo",
    resume: Optional[str] = None,
    num_steps: int = 2000,
    data: str = "orca",
) -> str:
    os.makedirs(save_dir, exist_ok=True)
    logger.info("=== PHASE 3: GRPO (anti-overthinking, flow sampler) ===")

    model = _build_or_load_model(checkpoint, device)
    tokenizer = _load_tokenizer(TEACHER_MODEL)

    special_tokens = {}
    if "<think>" not in tokenizer.get_vocab():
        special_tokens["additional_special_tokens"] = ["<think>", "</think>"]
        tokenizer.add_special_tokens(special_tokens)
        model.token_embed.resize(tokenizer.vocab_size)

    think_start_id = tokenizer.convert_tokens_to_ids("<think>")
    think_end_id = tokenizer.convert_tokens_to_ids("</think>")

    grpo_cfg = GRPO_CFG
    grpo_cfg.think_start_id = think_start_id
    grpo_cfg.think_end_id = think_end_id
    grpo_cfg.eos_id = tokenizer.eos_token_id
    grpo_cfg.save_dir = save_dir
    grpo_cfg.state_file = "./training_state.json"

    trainer = build_math_grpo_trainer(model, tokenizer, grpo_cfg)

    if resume and os.path.isfile(resume):
        trainer.load_checkpoint(resume)

    logger.info("loading GRPO prompts (%s) …", data)
    prompts, references = [], []
    if data in ("orca", "both"):
        orca_ds = OrcaMathDataset(split="train", max_examples=20000)
        for i in range(len(orca_ds)):
            item = orca_ds[i]
            prompts.append(item["prompt"])
            references.append(item["response"])
    if data in ("smoltalk", "both"):
        smol_ds = SmolTalkDataset(split="train", max_examples=10000)
        for i in range(len(smol_ds)):
            item = smol_ds[i]
            prompts.append(item["prompt"])
            references.append(item["response"])

    logger.info("%d GRPO prompts loaded", len(prompts))

    _log_vram(device, "GRPO start")
    for step_idx in range(num_steps):
        idx = step_idx % len(prompts)
        metrics = trainer.train_step([prompts[idx]], [references[idx]])

        if step_idx % 50 == 0:
            logger.info(
                "GRPO step %d/%d | reward=%.3f | think_blks=%.2f | loss=%.4f",
                step_idx, num_steps,
                metrics["mean_reward"], metrics["mean_think_blocks"], metrics["loss"],
            )

    final_path = trainer.save_checkpoint(os.path.join(save_dir, "final.pt"))
    logger.info("GRPO done → %s", final_path)
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
                   help="Number of GRPO update steps")
    p.add_argument("--grpo-data", choices=["orca", "smoltalk", "both"], default="orca",
                   help="Dataset for GRPO rollouts")
    p.add_argument("--teacher", default=TEACHER_MODEL,
                   help=f"HuggingFace teacher model (default: {TEACHER_MODEL})")
    p.add_argument("--no-flow", action="store_true",
                   help="Disable flow matching (fall back to DDPM cosine schedule)")
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
        DISTILL_CFG.setdefault("stages", [])  # stages unchanged
        GRPO_CFG.num_diffusion_steps_inference = 30
        GRPO_CFG.sampler = "dpmpp"
        logger.info("flow matching disabled (--no-flow)")

    global TEACHER_MODEL
    TEACHER_MODEL = args.teacher
    DISTILL_CFG["teacher_model"] = args.teacher

    ckpt = args.checkpoint

    if args.phase in ("distill", "all"):
        ckpt = run_distill(device, resume=ckpt if args.resume else None)

    if args.phase in ("sft", "all"):
        ckpt = run_sft(ckpt, device, resume=ckpt if args.resume else None)

    if args.phase in ("grpo", "all"):
        ckpt = run_grpo(ckpt, device,
                        resume=ckpt if args.resume else None,
                        num_steps=args.grpo_steps,
                        data=args.grpo_data)

    logger.info("all done. final model: %s", ckpt)


if __name__ == "__main__":
    main()
