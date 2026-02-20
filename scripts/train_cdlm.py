#!/usr/bin/env python3
"""Training script for DIMBA with CDLM (Consistency Diffusion Language Model) support.

CDLM enables up to 14x faster inference by training the model to produce consistent
predictions across different timesteps. This script extends the standard training
with consistency loss as described in the Together AI paper.

Reference: "Consistency diffusion language models: Up to 14x faster inference without
sacrificing quality" - Together AI

Usage:
    python train_cdlm.py --config config.yaml --consistency-weight 0.5
"""

import os
import sys
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import yaml

# Add src to path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, '..', 'src')
sys.path.insert(0, os.path.abspath(src_dir))

from dimba import DIMBA
from dimba.data import HuggingFaceDataset, DummyDataset, collate_fn
from dimba.tokenizers import BPETokenizer, SimpleTokenizer
from dimba.training import DIMBALightningModule


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_tokenizer(config: dict):
    """Create tokenizer based on config."""
    tokenizer_config = config.get('tokenizer', {'type': 'bpe', 'vocab_size': 10000})
    
    if tokenizer_config['type'] == 'bpe':
        return BPETokenizer(vocab_size=tokenizer_config.get('vocab_size', 10000))
    else:
        return SimpleTokenizer(vocab_size=tokenizer_config.get('vocab_size', 256))


def create_datasets(config: dict, tokenizer):
    """Create training and validation datasets."""
    data_config = config['data']
    
    if data_config['type'] == 'huggingface':
        train_dataset = HuggingFaceDataset(
            dataset_name=data_config.get('dataset_name', 'wikitext'),
            dataset_config=data_config.get('dataset_config', 'wikitext-2-raw-v1'),
            split='train',
            tokenizer=tokenizer,
            max_length=data_config.get('max_length', 256),
            streaming=data_config.get('streaming', False),
        )
        val_dataset = HuggingFaceDataset(
            dataset_name=data_config.get('dataset_name', 'wikitext'),
            dataset_config=data_config.get('dataset_config', 'wikitext-2-raw-v1'),
            split='validation',
            tokenizer=tokenizer,
            max_length=data_config.get('max_length', 256),
            streaming=data_config.get('streaming', False),
        )
    else:
        # Dummy dataset for testing
        train_dataset = DummyDataset(
            num_examples=data_config.get('num_examples', 1000),
            max_length=data_config.get('max_length', 256),
            vocab_size=tokenizer.vocab_size,
        )
        val_dataset = DummyDataset(
            num_examples=data_config.get('num_examples', 100) // 10,
            max_length=data_config.get('max_length', 256),
            vocab_size=tokenizer.vocab_size,
        )
    
    return train_dataset, val_dataset


def main():
    parser = argparse.ArgumentParser(description='Train DIMBA with CDLM consistency training')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--consistency-weight', type=float, default=None,
                        help='Override consistency loss weight (default: from config)')
    parser.add_argument('--enable-consistency', action='store_true',
                        help='Enable consistency training')
    parser.add_argument('--disable-consistency', action='store_true',
                        help='Disable consistency training')
    parser.add_argument('--output-dir', type=str, default='./checkpoints/cdlm',
                        help='Output directory for checkpoints')
    parser.add_argument('--name', type=str, default='dimba_cdlm',
                        help='Experiment name for logging')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("DIMBA with CDLM (Consistency Diffusion Language Model) Training")
    print("=" * 70)
    
    # Load configuration
    print(f"\nLoading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Override consistency settings from command line
    training_config = config.get('training', {})
    
    if args.enable_consistency:
        training_config['use_consistency_training'] = True
        print("✓ Consistency training ENABLED via command line")
    elif args.disable_consistency:
        training_config['use_consistency_training'] = False
        print("✓ Consistency training DISABLED via command line")
    
    if args.consistency_weight is not None:
        training_config['consistency_loss_weight'] = args.consistency_weight
        print(f"✓ Consistency loss weight set to: {args.consistency_weight}")
    
    # Update config
    config['training'] = training_config
    
    # Check device
    device_config = config.get('device', {})
    use_gpu = device_config.get('use_gpu', True) and torch.cuda.is_available()
    
    print(f"\nDevice Configuration:")
    print(f"  Device: {'CUDA' if use_gpu else 'CPU'}")
    if use_gpu:
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create tokenizer
    print("\nCreating tokenizer...")
    tokenizer = create_tokenizer(config)
    print(f"  Vocab size: {tokenizer.vocab_size}")
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset, val_dataset = create_datasets(config, tokenizer)
    print(f"  Train examples: {len(train_dataset)}")
    print(f"  Val examples: {len(val_dataset)}")
    
    # Create dataloaders
    data_config = config['data']
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=data_config.get('batch_size', 32),
        shuffle=True,
        num_workers=data_config.get('num_workers', 0),
        collate_fn=collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=data_config.get('batch_size', 32),
        shuffle=False,
        num_workers=data_config.get('num_workers', 0),
        collate_fn=collate_fn,
    )
    
    # Create model
    print("\nCreating DIMBA model...")
    model_config = config['model']
    
    lightning_module = DIMBALightningModule(
        vocab_size=tokenizer.vocab_size,
        model_config=model_config,
        learning_rate=training_config.get('learning_rate', 2e-5),
        warmup_steps=training_config.get('warmup_steps', 500),
        ema_decay=training_config.get('ema_decay', 0.9999),
        use_ema=training_config.get('use_ema', True),
        use_consistency_training=training_config.get('use_consistency_training', False),
        consistency_loss_weight=training_config.get('consistency_loss_weight', 0.5),
        consistency_delta_min=training_config.get('consistency_delta_min', 50),
        consistency_delta_max=training_config.get('consistency_delta_max', 200),
    )
    
    # Calculate model parameters
    total_params = sum(p.numel() for p in lightning_module.model.parameters())
    trainable_params = sum(p.numel() for p in lightning_module.model.parameters() if p.requires_grad)
    
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: ~{total_params * 4 / 1e9:.2f} GB (FP32)")
    
    # Print CDLM status
    if lightning_module.use_consistency_training:
        print(f"\n  CDLM Consistency Training:")
        print(f"    ✓ ENABLED")
        print(f"    Loss weight: {lightning_module.consistency_loss_weight}")
        print(f"    Timestep delta: [{lightning_module.consistency_delta_min}, {lightning_module.consistency_delta_max}]")
    else:
        print(f"\n  CDLM Consistency Training: DISABLED")
        print(f"    (Enable in config with use_consistency_training: true)")
    
    # Callbacks
    os.makedirs(args.output_dir, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename=f'{args.name}-{{epoch:02d}}-{{val/loss:.4f}}',
        monitor='val/loss',
        mode='min',
        save_top_k=5,
        save_last=True,
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val/loss',
        mode='min',
        patience=training_config.get('early_stop_patience', 5),
    )
    
    # Logger
    logger = TensorBoardLogger(
        './logs',
        name=args.name,
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_steps=training_config.get('max_steps', -1),
        max_epochs=training_config.get('num_epochs', 10),
        devices=1 if use_gpu else 0,
        accelerator='gpu' if use_gpu else 'cpu',
        precision='16-mixed' if use_gpu and device_config.get('mixed_precision', False) else '32-true',
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        log_every_n_steps=training_config.get('log_interval', 100),
        val_check_interval=training_config.get('val_interval', 500),
        enable_progress_bar=True,
        benchmark=device_config.get('benchmark', True),
    )
    
    # Save tokenizer
    tokenizer_path = os.path.join(args.output_dir, 'tokenizer.json')
    tokenizer.save(tokenizer_path)
    print(f"\n  Tokenizer saved to: {tokenizer_path}")
    
    # Save config
    config_path = os.path.join(args.output_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    print(f"  Config saved to: {config_path}")
    
    # Start training
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    
    try:
        trainer.fit(
            lightning_module,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )
        
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE!")
        print("=" * 70)
        print(f"Best model: {checkpoint_callback.best_model_path}")
        print(f"Best val loss: {checkpoint_callback.best_model_score:.4f}")
        print(f"Logs: {logger.log_dir}")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print(f"\n\nTraining failed: {e}")
        raise


if __name__ == '__main__':
    main()
