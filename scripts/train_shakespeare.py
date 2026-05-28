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


def generate_sample(model, tokenizer, prompt: str, max_len: int = 128) -> str:
    """Generate a text sample from the trained model."""
    from dimba.diffusion.sampling import sample_from_model

    model.eval()
    dev = next(model.parameters()).device
    prompt_ids = torch.tensor([tokenizer.encode(prompt)], device=dev)

    with torch.no_grad():
        generated = sample_from_model(
            model, prompt_ids=prompt_ids, seq_len=max_len,
            num_steps=50, temperature=0.8, top_p=0.95,
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
    parser.add_argument("--accelerator", type=str, default="mps", choices=["mps", "cpu", "gpu"])
    parser.add_argument("--precision", type=str, default="32-true", help="32-true (MPS) or 16-mixed (CUDA)")
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Quick dry run (1 epoch, 2 batches)")
    args = parser.parse_args()

    text_path = args.text_path or str(_repo_path("data/shakespeare.txt"))
    ckpt_dir = args.checkpoint_dir or str(_repo_path("checkpoints/shakespeare"))
    log_dir = args.log_dir or str(_repo_path("logs"))

    # ── Config ──
    model_cfg = {
        "d_model": args.d_model,
        "d_prompt": args.d_model // 2,
        "num_diffusion_steps": args.num_steps,
        "num_denoiser_layers": args.num_layers,
        "d_state": 16,
        "d_conv": 4,
        "expand": 2,
        "conditioning_type": "film",
        "dropout": 0.1,
        "use_weight_tying": True,
        "use_simple_mamba": True,
        "latent_diffusion": True,
        "d_latent": args.d_model // 2,
        "embed_init_std": 0.02,
    }

    print("=" * 54)
    print("  DIMBA → Shakespeare  |  Apple Silicon MPS")
    print("=" * 54)
    print(f"  data   : {text_path}")
    print(f"  config : d_model={args.d_model}  layers={args.num_layers}  T={args.num_steps}")
    print(f"  train  : epochs={args.epochs}  batch={args.batch_size}  lr={args.lr}")

    # ── 1. Data ──
    print("\n── Loading data")
    texts = load_chunks(text_path, args.seq_len)
    print(f"   {len(texts)} chunks of {args.seq_len} chars")

    tokenizer = SimpleCharacterTokenizer(vocab_size=128)

    split = int(len(texts) * 0.9)
    train_loader = DataLoader(
        TextDataset(texts[:split], tokenizer, max_length=args.seq_len),
        batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        TextDataset(texts[split:], tokenizer, max_length=args.seq_len),
        batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
    )

    # ── 2. Model ──
    print(f"\n── Building model  (vocab={tokenizer.vocab_size})")
    module = DIMBALightningModule(
        vocab_size=tokenizer.vocab_size,
        model_config=model_cfg,
        learning_rate=args.lr,
        warmup_steps=100,
        ema_decay=0.9999,
        use_ema=False,
        ce_loss_weight=1.0,
        min_snr_gamma=5.0,
    )
    n = sum(p.numel() for p in module.parameters())
    print(f"   {n:,} params  ({n * 4 / 1024**2:.1f} MB fp32)")

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
        devices=1,
        precision=args.precision,
        callbacks=[
            ModelCheckpoint(dirpath=ckpt_dir, filename="shakespeare1",
                            monitor="val/loss", mode="min", save_top_k=1),
            EarlyStopping(monitor="val/loss", patience=3, mode="min"),
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

    # ── 5. Sample ──
    print("\n── Sample generation")
    sys.stdout.flush()
    try:
        sample = generate_sample(module.model, tokenizer, prompt="Juliet:", max_len=args.seq_len // 2)
        print(f'  prompt    : "Juliet:"')
        print(f"  generated : {sample[:200]}")
    except Exception as e:
        print(f"  (skipped: {e})")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
