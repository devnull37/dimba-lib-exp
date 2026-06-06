#!/usr/bin/env python3
"""Train a small DIMBA model on Shakespeare text using Apple Silicon MPS.

Quick start:
    PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/train_shakespeare.py

This loads ``data/shakespeare.txt``, chunks it into character sequences, trains a
small character-level diffusion model, and saves checkpoints + tokenizer to
``checkpoints/shakespeare/``.

Generated text samples are printed at the end of training for a quick sanity check.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

# Allow running from scripts/ or repo root
_SRC_DIR = (Path(__file__).resolve().parent / ".." / "src").resolve()
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from dimba.training import DIMBALightningModule
from dimba.training.trainer import GenerationSampleCallback
from dimba.tokenizers import SimpleCharacterTokenizer
from dimba.data import TextDataset, collate_fn


def _repo_path(rel: str) -> Path:
    """Resolve a path relative to the repo root."""
    return (Path(__file__).resolve().parent / ".." / rel).resolve()


def load_chunks(path: str, seq_len: int = 256, min_len: int = 64) -> list[str]:
    """Load text file and chunk it into fixed-length sequences."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    stride = max(1, seq_len // 2)
    chunks = [text[i : i + seq_len] for i in range(0, len(text) - seq_len + 1, stride)]
    return [c for c in chunks if len(c) >= min_len]


def generate_sample(model, tokenizer, max_len: int = 256, device: str = "cpu") -> str:
    """Unconditionally sample text from the (unconditionally pretrained) model.

    We pretrain without prompts, so we sample from the null conditioning
    (``prompt_ids=None``). The model is moved to ``device`` first because the
    mamba_ssm CUDA kernels require GPU tensors and Lightning can leave the model
    on CPU after DDP teardown.
    """
    from dimba.diffusion.sampling import sample_from_model

    model = model.to(device).eval()
    with torch.no_grad():
        generated = sample_from_model(
            model, prompt_ids=None, seq_len=max_len,
            num_steps=50, temperature=0.8, top_p=0.95, device=device,
        )
    return tokenizer.decode(generated[0])


def main():
    parser = argparse.ArgumentParser(
        description="Train DIMBA on Shakespeare",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/train_shakespeare.py --epochs 10
  PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/train_shakespeare.py --accelerator cpu --epochs 5
  python scripts/train_shakespeare.py --dry-run""",
    )
    parser.add_argument("--text-path", type=str, default=None, help="Shakespeare text (default: data/shakespeare.txt)")
    parser.add_argument("--seq-len", type=int, default=64, help="Sequence length")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", type=int, default=2, help="Training epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--d-model", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--num-layers", type=int, default=2, help="Denoiser layers")
    parser.add_argument("--num-steps", type=int, default=64, help="Diffusion timesteps")
    parser.add_argument("--d-state", type=int, default=16, help="SSM state size (use 64 for the mamba-ssm kernel)")
    parser.add_argument("--use-mamba-ssm", action="store_true",
                        help="Use the precompiled mamba_ssm CUDA kernels (fast, low memory) instead of the pure-PyTorch scan")
    parser.add_argument("--grad-checkpoint", action="store_true",
                        help="Gradient-checkpoint denoiser blocks (much lower memory, ~25%% slower)")
    parser.add_argument("--accelerator", type=str, default="mps", choices=["mps", "cpu", "gpu"])
    parser.add_argument("--devices", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--strategy", type=str, default="auto", help="auto, ddp, ddp_spawn")
    parser.add_argument("--precision", type=str, default="32-true", help="32-true (MPS) or 16-mixed (CUDA)")
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Quick dry run (1 epoch, 2 batches)")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers per process")
    parser.add_argument("--patience", type=int, default=10, help="EarlyStopping patience (epochs)")
    parser.add_argument("--warmup-steps", type=int, default=200, help="LR warmup steps")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping norm (0 disables)")
    parser.add_argument("--tokenizer", choices=["char", "bpe"], default="char",
                        help="Tokenizer type: char (character-level) or bpe (subword BPE)")
    parser.add_argument("--bpe-vocab-size", type=int, default=4096,
                        help="BPE vocabulary size (only used when --tokenizer bpe)")
    args = parser.parse_args()

    text_path = args.text_path or str(_repo_path("data/shakespeare.txt"))
    ckpt_dir = args.checkpoint_dir or str(_repo_path("checkpoints/shakespeare"))
    log_dir = args.log_dir or str(_repo_path("logs"))

    # ── Config ──
    # The mamba_ssm CUDA kernels only run on GPU; anywhere else use the
    # pure-PyTorch scan regardless of the flag.
    use_simple_mamba = not (args.use_mamba_ssm and args.accelerator == "gpu")
    model_cfg = {
        "d_model": args.d_model,
        "d_prompt": args.d_model // 2,
        "num_diffusion_steps": args.num_steps,
        "num_denoiser_layers": args.num_layers,
        "d_state": args.d_state,
        "d_conv": 4,
        "expand": 2,
        "conditioning_type": "adaln",
        "latent_norm": True,
        "self_conditioning": True,
        "dropout": 0.1,
        "use_weight_tying": True,
        "use_simple_mamba": use_simple_mamba,
        "use_gradient_checkpointing": args.grad_checkpoint,
        "latent_diffusion": True,
        "d_latent": args.d_model // 2,
        "embed_init_std": 0.02,
        # Generation-quality fixes for new runs (see the repo audit):
        #  - v-prediction balances the loss across noise levels; x0-prediction +
        #    min-SNR structurally under-trains the high-noise (generative) regime,
        #    which made the prior 30M a strong denoiser but a weak generator.
        #  - head LayerNorm + learnable logit-scale fixes near-uniform tied-head
        #    logits (the tiny-embedding rounding bottleneck behind garbled words).
        "prediction_type": "v",
        "use_head_norm": True,
        # Context-aware (non-autoregressive) rounding head: each position picks its
        # token with full-sequence self-attention -- the cited fix for the rounding
        # bottleneck behind garbled words. Latent diffusion + Mamba backbone unchanged.
        "head_type": "attn",
    }

    accel = args.accelerator.upper() if args.accelerator != "gpu" else f"CUDA ({torch.cuda.get_device_name(0)})"
    print("=" * 54)
    print(f"  DIMBA → Shakespeare  |  {accel}")
    print("=" * 54)
    print(f"  data   : {text_path}")
    print(f"  config : d_model={args.d_model}  layers={args.num_layers}  T={args.num_steps}")
    print(f"  train  : epochs={args.epochs}  batch={args.batch_size}  lr={args.lr}")

    # ── 1. Data ──
    print("\n── Loading data")
    chunk_chars = args.seq_len * 4 if args.tokenizer == "bpe" else args.seq_len
    texts = load_chunks(text_path, chunk_chars)
    print(f"   {len(texts)} chunks of {chunk_chars} chars")

    if args.tokenizer == "bpe":
        try:
            from dimba.tokenizers.bpe import BPETokenizer
        except ImportError as exc:
            raise ImportError(
                "BPETokenizer requires the 'tokenizers' library. "
                "Install it with: pip install tokenizers"
            ) from exc
        print(f"   training BPE tokenizer (vocab_size={args.bpe_vocab_size}) ...")
        tokenizer = BPETokenizer.train(texts, vocab_size=args.bpe_vocab_size)
        tokenizer.vocab_size = tokenizer.tokenizer.get_vocab_size()
        print(f"   BPE vocab size: {tokenizer.vocab_size}")
    else:
        tokenizer = SimpleCharacterTokenizer(vocab_size=128)

    split = int(len(texts) * 0.9)
    nw = args.num_workers
    loader_kw = dict(
        batch_size=args.batch_size, collate_fn=collate_fn,
        num_workers=nw, pin_memory=(args.accelerator == "gpu"),
        persistent_workers=(nw > 0),
    )
    train_loader = DataLoader(
        TextDataset(texts[:split], tokenizer, max_length=args.seq_len),
        shuffle=True, **loader_kw,
    )
    val_loader = DataLoader(
        TextDataset(texts[split:], tokenizer, max_length=args.seq_len),
        shuffle=False, **loader_kw,
    )

    # ── 2. Model ──
    print(f"\n── Building model  (vocab={tokenizer.vocab_size})")
    module = DIMBALightningModule(
        vocab_size=tokenizer.vocab_size,
        model_config=model_cfg,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        ema_decay=0.9999,
        use_ema=False,
        ce_loss_weight=1.0,
        min_snr_gamma=5.0,
        # logit-normal timestep sampling concentrates training on the mid-noise regime
        # where token content is decided (SD3/FLUX); antithetic pairing cuts gradient
        # variance; skip the degenerate t=0 (x_t == x_0 under zero-terminal-SNR).
        timestep_sampling="logit_normal",
        antithetic_t=True,
        exclude_zero_t=True,
    )
    n = sum(p.numel() for p in module.parameters())
    print(f"   {n:,} params  ({n * 4 / 1024**2:.1f} MB fp32)")

    # Verify the requested kernel is actually active. The denoiser silently
    # falls back to the pure-PyTorch scan if mamba_ssm is missing or rejects the
    # dims — that path is slow and OOM-prone, so fail loudly rather than waste a
    # GPU session discovering it the hard way.
    mixer = type(module.model.denoiser.blocks[0].mamba_fwd).__name__
    print(f"   ssm kernel : {mixer}")
    if args.use_mamba_ssm and args.accelerator == "gpu" and mixer == "SimpleMamba2":
        raise RuntimeError(
            "Requested --use-mamba-ssm but the model fell back to SimpleMamba2 "
            "(pure-PyTorch scan). Install the kernels with:\n"
            "    pip install --no-build-isolation causal-conv1d mamba-ssm\n"
            "and make sure the dims are kernel-compatible (try --d-state 64)."
        )

    # Calibrate latent scale (required for latent_diffusion)
    print("   calibrating latent scale ...")
    calib_batch = next(iter(train_loader))
    module.model.calibrate_latent_scale(calib_batch["input_ids"])
    print(f"   latent_scale = {module.model.latent_scale:.4f}")

    # ── 3. Train ──
    print(f"\n── Training on {args.accelerator}")
    limit = 2 if args.dry_run else None
    trainer = pl.Trainer(
        max_epochs=1 if args.dry_run else args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=args.strategy,
        precision=args.precision,
        gradient_clip_val=(args.grad_clip or None),
        callbacks=[
            ModelCheckpoint(dirpath=ckpt_dir, filename="shakespeare1",
                            monitor="val/loss", mode="min", save_top_k=1),
            EarlyStopping(monitor="val/loss", patience=args.patience, mode="min"),
            GenerationSampleCallback(
                tokenizer=tokenizer, prompt="ROMEO:\n",
                seq_len=120, every_n_epochs=1,
            ),
        ],
        logger=TensorBoardLogger(log_dir, name="shakespeare"),
        log_every_n_steps=10,
        val_check_interval=1.0,
        enable_progress_bar=True,
        limit_train_batches=limit,
        limit_val_batches=limit,
    )

    trainer.fit(module, train_loader, val_loader)

    # ── 4. Save ──
    tokenizer_path = str(Path(ckpt_dir) / "tokenizer.json")
    tokenizer.save(tokenizer_path)

    best = getattr(trainer.checkpoint_callback, "best_model_path", "")
    print(f"\n  ✓ checkpoints → {ckpt_dir}")
    print(f"  ✓ tokenizer   → {tokenizer_path}")
    if best:
        print(f"  ✓ best model  → {best}")

    # ── 5. Sample (rank 0 only; unconditional, matching how we pretrain) ──
    if trainer.is_global_zero:
        print("\n── Sample generation (unconditional)")
        sys.stdout.flush()
        sample_device = "cuda" if args.accelerator == "gpu" else (
            "mps" if args.accelerator == "mps" else "cpu")
        for k in range(2):
            try:
                sample = generate_sample(module.model, tokenizer, max_len=args.seq_len, device=sample_device)
                print(f"  [sample {k + 1}]\n{sample[:400]}\n")
            except Exception as e:
                print(f"  (skipped: {e})")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
