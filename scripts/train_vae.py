#!/usr/bin/env python3
"""Standalone script to pre-train the VAE on a dataset.

This script pre-trains a TokenVAE on token sequences before full diffusion training.
The trained VAE can then be used for latent diffusion.

Usage:
    # Basic usage with default settings
    python scripts/train_vae.py --dataset wikitext --dataset-config wikitext-2-raw-v1

    # With custom hyperparameters
    python scripts/train_vae.py \
        --dataset wikitext \
        --dataset-config wikitext-2-raw-v1 \
        --latent-dim 256 \
        --kl-weight 0.1 \
        --learning-rate 1e-4 \
        --batch-size 64 \
        --epochs 10

    # With PyTorch Lightning
    python scripts/train_vae.py \
        --use-lightning \
        --dataset wikitext \
        --gpus 1 \
        --max-steps 100000
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add src to path
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = (SCRIPT_DIR / ".." / "src").resolve()
sys.path.insert(0, str(SRC_DIR))

from dimba.data.dataset import DummyDataset, HuggingFaceDataset, collate_fn
from dimba.models.embeddings import TokenEmbedding
from dimba.models.vae import TokenVAE
from dimba.training.trainer import VAELightningModule, VAETrainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train TokenVAE for latent diffusion')
    parser.add_argument('--dataset', type=str, default='wikitext',
                        help='Dataset name (e.g., wikitext, openwebtext)')
    parser.add_argument('--dataset-config', type=str, default='wikitext-2-raw-v1',
                        help='Dataset configuration name')
    parser.add_argument('--text-column', type=str, default='text',
                        help='Column name containing text data')
    parser.add_argument('--max-length', type=int, default=256,
                        help='Maximum sequence length')
    parser.add_argument('--use-dummy', action='store_true',
                        help='Use dummy dataset for testing')
    parser.add_argument('--vocab-size', type=int, default=50000,
                        help='Vocabulary size')
    parser.add_argument('--d-model', type=int, default=512,
                        help='Token embedding dimension')
    parser.add_argument('--latent-dim', type=int, default=256,
                        help='VAE latent dimension')
    parser.add_argument('--hidden-dim', type=int, default=None,
                        help='Hidden dimension (default: max(d_model, latent_dim))')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='Number of layers in encoder/decoder')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--kl-weight', type=float, default=1.0,
                        help='Weight for KL divergence loss (beta-VAE)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--warmup-steps', type=int, default=1000,
                        help='Number of warmup steps')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1,
                        help='Gradient accumulation steps')
    parser.add_argument('--use-amp', action='store_true',
                        help='Use automatic mixed precision')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/vae',
                        help='Directory to save checkpoints')
    parser.add_argument('--save-every', type=int, default=1,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Resume from checkpoint path')
    parser.add_argument('--use-lightning', action='store_true',
                        help='Use PyTorch Lightning for training')
    parser.add_argument('--gpus', type=int, default=0,
                        help='Number of GPUs to use (0 for CPU)')
    parser.add_argument('--max-steps', type=int, default=None,
                        help='Maximum number of training steps (Lightning only)')
    parser.add_argument('--val-check-interval', type=float, default=1.0,
                        help='Validation check interval (Lightning only)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_dataloaders(args):
    """Create training and validation dataloaders."""
    if args.use_dummy:
        print("Using dummy dataset for testing")
        train_dataset = DummyDataset(size=1000, vocab_size=args.vocab_size,
                                     seq_length=args.max_length)
        val_dataset = DummyDataset(size=100, vocab_size=args.vocab_size,
                                   seq_length=args.max_length)
    else:
        print(f"Loading dataset: {args.dataset}")
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained('gpt2')
            tokenizer.pad_token = tokenizer.eos_token
        except ImportError:
            print("Warning: transformers not installed, using dummy tokenization")
            tokenizer = None
        train_dataset = HuggingFaceDataset(
            dataset_name=args.dataset, dataset_config=args.dataset_config,
            split='train', tokenizer=tokenizer, max_length=args.max_length,
            text_column=args.text_column)
        val_dataset = HuggingFaceDataset(
            dataset_name=args.dataset, dataset_config=args.dataset_config,
            split='validation', tokenizer=tokenizer, max_length=args.max_length,
            text_column=args.text_column)

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    return train_dataloader, val_dataloader


def train_with_lightning(args):
    """Train VAE using PyTorch Lightning."""
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
    from pytorch_lightning.loggers import TensorBoardLogger

    train_dataloader, val_dataloader = create_dataloaders(args)
    model_config = {'hidden_dim': args.hidden_dim, 'num_layers': args.num_layers,
                    'dropout': args.dropout}

    if args.resume_from:
        print(f"Resuming from checkpoint: {args.resume_from}")
        module = VAELightningModule.load_checkpoint(
            args.resume_from, learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps, weight_decay=args.weight_decay,
            kl_weight=args.kl_weight)
    else:
        module = VAELightningModule(
            vocab_size=args.vocab_size, d_model=args.d_model,
            latent_dim=args.latent_dim, model_config=model_config,
            learning_rate=args.learning_rate, warmup_steps=args.warmup_steps,
            weight_decay=args.weight_decay, kl_weight=args.kl_weight)

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir, filename='vae-{epoch:02d}-{val/loss:.4f}',
        save_top_k=3, monitor='val/loss', mode='min', save_last=True)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    logger = TensorBoardLogger(save_dir=os.path.dirname(args.checkpoint_dir),
                               name='vae_logs')

    trainer_kwargs = {
        'max_epochs': args.epochs, 'callbacks': [checkpoint_callback, lr_monitor],
        'logger': logger, 'gradient_clip_val': 1.0,
        'accumulate_grad_batches': args.gradient_accumulation_steps,
        'precision': 16 if args.use_amp else 32}
    if args.gpus > 0:
        trainer_kwargs['accelerator'] = 'gpu'
        trainer_kwargs['devices'] = args.gpus
    else:
        trainer_kwargs['accelerator'] = 'cpu'
    if args.max_steps is not None:
        trainer_kwargs['max_steps'] = args.max_steps
        trainer_kwargs['max_epochs'] = -1

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(module, train_dataloader, val_dataloader)
    final_path = os.path.join(args.checkpoint_dir, 'final.ckpt')
    module.save_checkpoint(final_path)
    print(f"Saved final checkpoint to {final_path}")
    return module


def train_simple(args):
    """Train VAE using simple training loop."""
    train_dataloader, val_dataloader = create_dataloaders(args)
    token_embed = TokenEmbedding(vocab_size=args.vocab_size, embed_dim=args.d_model)
    vae = TokenVAE(input_dim=args.d_model, latent_dim=args.latent_dim,
                   hidden_dim=args.hidden_dim, num_layers=args.num_layers,
                   dropout=args.dropout, kl_weight=args.kl_weight)

    if args.resume_from:
        print(f"Resuming from checkpoint: {args.resume_from}")
        vae, config, step = VAETrainer.load_checkpoint(args.resume_from, args.device)
        vae = vae.to(args.device)

    trainer = VAETrainer(
        vae=vae, token_embed=token_embed, train_dataloader=train_dataloader,
        val_dataloader=val_dataloader, device=args.device, num_epochs=args.epochs,
        learning_rate=args.learning_rate, warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay, kl_weight=args.kl_weight,
        gradient_accumulation_steps=args.gradient_accumulation_steps, use_amp=args.use_amp)
    trainer.train()
    final_path = os.path.join(args.checkpoint_dir, 'final.pt')
    trainer.save_checkpoint(final_path)
    print(f"Saved final checkpoint to {final_path}")
    return vae, token_embed


def main():
    """Main entry point."""
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print("=" * 60)
    print("VAE Training Configuration")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Vocabulary size: {args.vocab_size}")
    print(f"Model dimension (d_model): {args.d_model}")
    print(f"Latent dimension: {args.latent_dim}")
    print(f"KL weight (beta): {args.kl_weight}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print(f"Use Lightning: {args.use_lightning}")
    print("=" * 60)

    if args.use_lightning:
        train_with_lightning(args)
    else:
        train_simple(args)
    print("Training complete!")


if __name__ == '__main__':
    main()
