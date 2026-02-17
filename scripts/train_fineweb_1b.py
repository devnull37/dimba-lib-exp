#!/usr/bin/env python3
"""Training script for DIMBA 1.5B model on FineWeb dataset (L40S 48GB).

Usage:
    python scripts/train_fineweb_1b.py

This script trains a 1.5B parameter DIMBA model on the FineWeb dataset,
optimized for an L40S GPU with 48GB VRAM.
"""

import os
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# Add src to path
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = (SCRIPT_DIR / ".." / "src").resolve()
sys.path.insert(0, str(SRC_DIR))

from dimba.data import HuggingFaceDataset, collate_fn
from dimba.tokenizers import BPETokenizer
from dimba.training import DIMBALightningModule


def main():
    print("=" * 60)
    print("DIMBA 1.5B Model Training on FineWeb (L40S 48GB)")
    print("=" * 60)
    
    # Configuration for L40S 48GB
    config = {
        'model': {
            'd_model': 2560,
            'd_prompt': 2560,
            'num_diffusion_steps': 1000,
            'num_denoiser_layers': 28,
            'd_state': 64,
            'd_conv': 4,
            'expand': 2,
            'conditioning_type': 'film',
            'dropout': 0.1,
            'use_weight_tying': True,
            'use_simple_mamba': False,
        },
        'data': {
            'type': 'huggingface',
            'dataset_name': 'HuggingFaceFW/fineweb',
            'dataset_config': 'sample-10BT',
            'batch_size': 32,
            'max_length': 1024,
            'num_workers': 8,
            'streaming': False,
        },
        'training': {
            'learning_rate': 1e-4,
            'warmup_steps': 2000,
            'weight_decay': 0.01,
            'ema_decay': 0.9999,
            'use_ema': True,
            'gradient_clip': 1.0,
            'max_steps': 100000,
            'log_interval': 100,
            'val_interval': 1000,
        },
        'tokenizer': {
            'type': 'bpe',
            'vocab_size': 32000,
        }
    }
    
    # Check GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create tokenizer
    print("\nCreating BPE tokenizer...")
    tokenizer = BPETokenizer(vocab_size=config['tokenizer']['vocab_size'])
    print(f"Tokenizer created: vocab_size={tokenizer.vocab_size}")
    
    # Create datasets
    print("\nLoading FineWeb dataset...")
    try:
        train_dataset = HuggingFaceDataset(
            dataset_name=config['data']['dataset_name'],
            split='train',
            tokenizer=tokenizer,
            max_length=config['data']['max_length'],
            streaming=config['data']['streaming'],
        )
        val_dataset = HuggingFaceDataset(
            dataset_name=config['data']['dataset_name'],
            split='validation',
            tokenizer=tokenizer,
            max_length=config['data']['max_length'],
            streaming=config['data']['streaming'],
        )
        print(f"✓ Dataset loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        collate_fn=collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        collate_fn=collate_fn,
    )
    print(f"✓ Dataloaders created")
    
    # Create model
    print("\nCreating DIMBA model...")
    lightning_module = DIMBALightningModule(
        vocab_size=config['tokenizer']['vocab_size'],
        model_config=config['model'],
        learning_rate=config['training']['learning_rate'],
        warmup_steps=config['training']['warmup_steps'],
        ema_decay=config['training']['ema_decay'],
        use_ema=config['training']['use_ema'],
    )
    
    # Calculate model parameters
    total_params = sum(p.numel() for p in lightning_module.model.parameters())
    trainable_params = sum(p.numel() for p in lightning_module.model.parameters() if p.requires_grad)
    print(f"✓ Model created")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: ~{total_params * 4 / 1e9:.1f} GB (FP32)")
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='./checkpoints/fineweb_1b',
        filename='dimba-1b-{epoch:02d}-{val/loss:.4f}',
        monitor='val/loss',
        mode='min',
        save_top_k=5,
        save_last=True,
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val/loss',
        mode='min',
        patience=5,
    )
    
    # Logger
    logger = TensorBoardLogger('./logs', name='dimba_fineweb_1b')
    
    # Trainer
    trainer = pl.Trainer(
        max_steps=config['training']['max_steps'],
        devices=1,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        precision='16-mixed' if torch.cuda.is_available() else '32-true',
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        log_every_n_steps=config['training']['log_interval'],
        val_check_interval=config['training']['val_interval'],
        enable_progress_bar=True,
        benchmark=True,
    )
    
    # Save tokenizer
    tokenizer_path = './checkpoints/fineweb_1b/tokenizer.json'
    os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)
    tokenizer.save(tokenizer_path)
    print(f"✓ Tokenizer saved to: {tokenizer_path}")
    
    # Start training
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    
    try:
        trainer.fit(
            lightning_module,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE!")
        print("=" * 60)
        print(f"Best model saved to: {checkpoint_callback.best_model_path}")
        print(f"Logs saved to: {logger.log_dir}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed: {e}")
        raise


if __name__ == '__main__':
    main()