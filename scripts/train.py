#!/usr/bin/env python3
"""Training script for DIMBA model.

Usage:
    # Train with default config
    python scripts/train.py

    # Train with custom config
    python scripts/train.py --config configs/my_config.yaml --max-epochs 20

    # Train on CPU
    python scripts/train.py --gpus 0

    # Train with mixed precision
    python scripts/train.py --mixed-precision 16-mixed
"""

import argparse
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

# Add src to path
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = (SCRIPT_DIR / ".." / "src").resolve()
sys.path.insert(0, str(SRC_DIR))

from dimba import DIMBA
from dimba.data import DummyDataset, HuggingFaceDataset, TextDataset, collate_fn
from dimba.training import DIMBALightningModule
from dimba.tokenizers import BPETokenizer, SimpleCharacterTokenizer


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_dataloaders(config: dict, vocab_size: int):
    """Create training and validation dataloaders, handling tokenizers."""
    from dimba.tokenizers import BPETokenizer, SimpleCharacterTokenizer

    data_config = config.get('data', {})
    dataset_type = data_config.get('type', 'dummy')
    batch_size = data_config.get('batch_size', 32)
    num_workers = data_config.get('num_workers', 0)
    max_length = data_config.get('max_length', 256)

    # ----- Tokenizer creation -----
    tokenizer_cfg = config.get('tokenizer', {})
    tokenizer_type = tokenizer_cfg.get('type', 'simple')
    vocab_sz = tokenizer_cfg.get('vocab_size', vocab_size)
    if tokenizer_type == 'bpe':
        # Initialize BPE tokenizer (untrained; users can pre‑train separately)
        tokenizer = BPETokenizer(vocab_size=vocab_sz)
    else:
        tokenizer = SimpleCharacterTokenizer(vocab_size=vocab_sz)
    # --------------------------------

    if dataset_type == 'dummy':
        num_examples = data_config.get('num_examples', 1000)
        train_dataset = DummyDataset(
            size=num_examples,
            vocab_size=vocab_size,
            seq_length=max_length,
        )
        val_dataset = DummyDataset(
            size=num_examples // 10,
            vocab_size=vocab_size,
            seq_length=max_length,
        )
    elif dataset_type == 'huggingface':
        dataset_name = data_config.get('dataset_name', 'wikitext')
        dataset_config = data_config.get('dataset_config', 'wikitext-2-raw-v1')
        num_examples = data_config.get('num_examples', None)
        streaming = data_config.get('streaming', False)

        try:
            train_dataset = HuggingFaceDataset(
                dataset_name=dataset_name,
                split='train',
                tokenizer=tokenizer,
                max_length=max_length,
                num_examples=num_examples,
                streaming=streaming,
            )
            val_dataset = HuggingFaceDataset(
                dataset_name=dataset_name,
                split='validation',
                tokenizer=tokenizer,
                max_length=max_length,
                num_examples=num_examples // 10 if num_examples else None,
                streaming=streaming,
            )
        except Exception as e:
            print(f"Failed to load HuggingFace dataset: {e}")
            print("Falling back to dummy dataset")
            train_dataset = DummyDataset(size=1000, vocab_size=vocab_size, seq_length=max_length)
            val_dataset = DummyDataset(size=100, vocab_size=vocab_size, seq_length=max_length)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    # Return both loader and tokenizer for later saving
    return train_loader, val_loader, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Train DIMBA model")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--vocab-size', type=int, default=10000, help='Vocabulary size')
    parser.add_argument('--max-epochs', type=int, default=10, help='Maximum number of epochs')
    parser.add_argument('--gpus', type=int, default=1 if torch.cuda.is_available() else 0, help='Number of GPUs')
    parser.add_argument('--mixed-precision', type=str, default=None, help='Mixed precision (16-mixed, 16-true, 32-true)')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--log-dir', type=str, default='logs', help='Logging directory')

    args = parser.parse_args()

    # Load config
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Config file {args.config} not found. Using default config.")
        config = {
            'model': {},
            'data': {'type': 'dummy', 'batch_size': 32},
            'training': {'learning_rate': 2e-5, 'warmup_steps': 500},
        }

    # Model config
    model_config = config.get('model', {})
    training_config = config.get('training', {})

    print("=" * 50)
    print("DIMBA Training")
    print("=" * 50)
    print(f"Model config: {model_config}")
    print(f"Training config: {training_config}")

    # Create dataloaders
    print("\nPreparing data...")
    train_loader, val_loader, tokenizer = create_dataloaders(config, args.vocab_size)
    print(f"Train set: {len(train_loader.dataset)} examples")
    print(f"Val set: {len(val_loader.dataset)} examples")
    print(f"Tokenizer: {type(tokenizer).__name__} (vocab_size={tokenizer.vocab_size})")

    # Create Lightning module
    print("\nCreating model...")
    lightning_module = DIMBALightningModule(
        vocab_size=args.vocab_size,
        model_config=model_config,
        learning_rate=float(training_config.get('learning_rate', 2e-5)),
        warmup_steps=int(training_config.get('warmup_steps', 500)),
        ema_decay=float(training_config.get('ema_decay', 0.9999)),
        use_ema=training_config.get('use_ema', True),
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename='dimba-{epoch:02d}-{val/loss:.4f}',
        monitor='val/loss',
        mode='min',
        save_top_k=3,
    )

    early_stop_callback = EarlyStopping(
        monitor='val/loss',
        mode='min',
        patience=3,
    )

    # Logger
    logger = TensorBoardLogger(args.log_dir, name='dimba')

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        devices=args.gpus if args.gpus > 0 else 1,  # CPU trainer needs devices=1 (not None or -1)
        accelerator='gpu' if args.gpus > 0 else 'cpu',
        strategy='auto',
        precision=args.mixed_precision or '32-true',
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        log_every_n_steps=10,
        val_check_interval=0.5,
        enable_progress_bar=True,
    )

    print("\nStarting training...")
    print("=" * 50)

    # Train
    trainer.fit(
        lightning_module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    print("=" * 50)
    print("Training complete!")
    print(f"Best model saved to: {args.checkpoint_dir}")
    print(f"Logs saved to: {args.log_dir}")

    # Save tokenizer
    tokenizer_path = f"{args.checkpoint_dir}/tokenizer.json"
    tokenizer.save(tokenizer_path)
    print(f"Tokenizer saved to: {tokenizer_path}")


if __name__ == '__main__':
    main()
