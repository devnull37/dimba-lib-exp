#!/usr/bin/env python3
"""Interactive training script for DIMBA with auto GPU detection and config wizard.

Usage:
    # Launch interactive mode
    python scripts/train_interactive.py

    # Resume from checkpoint
    python scripts/train_interactive.py --resume

    # Resume specific checkpoint
    python scripts/train_interactive.py --resume-from ./checkpoints/interactive

    # Skip wizard and use specific preset
    python scripts/train_interactive.py --preset a4000-500m

    # Use custom config with auto GPU detection
    python scripts/train_interactive.py --config my_config.yaml --auto-gpu

Available presets:
    - cpu-small: CPU training, tiny model for testing
    - a4000-500m: RTX A4000 16GB, ~500M params
    - l40s-1b: L40S 48GB, ~1.5B params
    - a100-3b: A100 80GB, ~3B params
    - multi-gpu: Multi-GPU training (auto-detect count)
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

import torch
import yaml

# Add src to path
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = (SCRIPT_DIR / ".." / "src").resolve()
sys.path.insert(0, str(SRC_DIR))

# Try to import training modules (may fail if dependencies not installed)
try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger

    from dimba.data import DummyDataset, HuggingFaceDataset, collate_fn
    from dimba.tokenizers import BPETokenizer, SimpleCharacterTokenizer
    from dimba.training import DIMBALightningModule

    HAS_DEPS = True
except ImportError as e:
    HAS_DEPS = False
    DEPS_ERROR = str(e)


# =============================================================================
# GPU Detection & Profiling
# =============================================================================

def detect_gpus() -> list[dict]:
    """Detect available GPUs and return their specs."""
    gpus = []

    if not torch.cuda.is_available():
        return gpus

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        name = torch.cuda.get_device_name(i)
        memory_gb = props.total_memory / 1e9

        # Classify GPU tier
        tier = classify_gpu(name, memory_gb)

        gpus.append({
            'index': i,
            'name': name,
            'memory_gb': memory_gb,
            'tier': tier,
            'compute_capability': f"{props.major}.{props.minor}",
        })

    return gpus


def classify_gpu(name: str, memory_gb: float) -> str:
    """Classify GPU into performance tier."""
    name_lower = name.lower()

    # High-end data center GPUs
    if any(x in name_lower for x in ['a100', 'h100', 'h200']):
        return 'high'
    # Mid-range data center / high-end consumer
    elif any(x in name_lower for x in ['l40', 'l40s', 'rtx 4090', 'rtx 3090']):
        return 'mid-high'
    # Mid-range
    elif any(x in name_lower for x in ['a4000', 'rtx 4080', 'rtx 3080', 'a5000']):
        return 'mid'
    # Entry-level
    elif any(x in name_lower for x in ['a2000', 'rtx 4070', 'rtx 3070', 'rtx 3060']):
        return 'low-mid'
    else:
        return 'unknown'


def get_recommended_batch_size(memory_gb: float, model_size: str) -> int:
    """Get recommended batch size based on GPU memory and model size."""
    # Rough estimates for FP16 training
    size_multipliers = {
        'tiny': 0.1,      # < 100M params
        'small': 0.25,    # ~100-300M
        'medium': 0.5,    # ~300M-1B
        'large': 1.0,     # ~1-2B
        'xlarge': 2.0,    # > 2B
    }

    multiplier = size_multipliers.get(model_size, 0.5)
    base_batch = int((memory_gb / 16) * 32 / multiplier)
    return max(1, min(base_batch, 128))


# =============================================================================
# Configuration Presets
# =============================================================================

PRESETS = {
    'cpu-small': {
        'name': 'CPU Training (Tiny Model)',
        'description': 'Tiny model for testing on CPU',
        'model': {
            'd_model': 256,
            'd_prompt': 256,
            'num_diffusion_steps': 100,
            'num_denoiser_layers': 4,
            'd_state': 16,
            'd_conv': 4,
            'expand': 2,
            'conditioning_type': 'film',
            'dropout': 0.1,
            'use_weight_tying': True,
        },
        'data': {
            'type': 'dummy',
            'batch_size': 4,
            'max_length': 128,
            'num_workers': 0,
        },
        'training': {
            'learning_rate': 1e-4,
            'warmup_steps': 100,
            'max_steps': 1000,
            'ema_decay': 0.9999,
            'use_ema': False,
        },
        'tokenizer': {
            'type': 'simple',
            'vocab_size': 1000,
        },
    },

    'a4000-500m': {
        'name': 'RTX A4000 16GB (~500M params)',
        'description': 'Optimized for RTX A4000 with 16GB VRAM',
        'model': {
            'd_model': 1536,
            'd_prompt': 1536,
            'num_diffusion_steps': 1000,
            'num_denoiser_layers': 20,
            'd_state': 64,
            'd_conv': 4,
            'expand': 2,
            'conditioning_type': 'film',
            'dropout': 0.1,
            'use_weight_tying': True,
        },
        'data': {
            'type': 'huggingface',
            'dataset_name': 'HuggingFaceFW/fineweb',
            'dataset_config': 'sample-10BT',
            'batch_size': 16,
            'max_length': 512,
            'num_workers': 4,
            'streaming': False,
        },
        'training': {
            'learning_rate': 1e-4,
            'warmup_steps': 1000,
            'weight_decay': 0.01,
            'ema_decay': 0.9999,
            'use_ema': True,
            'ema_device': 'cpu',
            'max_steps': 50000,
            'gradient_clip': 1.0,
            'accumulate_grad_batches': 1,
        },
        'tokenizer': {
            'type': 'bpe',
            'vocab_size': 32000,
        },
    },

    'l40s-1b': {
        'name': 'L40S 48GB (~1.5B params)',
        'description': 'Optimized for L40S with 48GB VRAM',
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
            'max_steps': 100000,
            'gradient_clip': 1.0,
        },
        'tokenizer': {
            'type': 'bpe',
            'vocab_size': 32000,
        },
    },

    'a100-3b': {
        'name': 'A100 80GB (~3B params)',
        'description': 'Large model for A100 with 80GB VRAM',
        'model': {
            'd_model': 3584,
            'd_prompt': 3584,
            'num_diffusion_steps': 1000,
            'num_denoiser_layers': 36,
            'd_state': 128,
            'd_conv': 4,
            'expand': 2,
            'conditioning_type': 'film',
            'dropout': 0.1,
            'use_weight_tying': True,
        },
        'data': {
            'type': 'huggingface',
            'dataset_name': 'HuggingFaceFW/fineweb',
            'dataset_config': 'sample-10BT',
            'batch_size': 48,
            'max_length': 2048,
            'num_workers': 8,
            'streaming': False,
        },
        'training': {
            'learning_rate': 8e-5,
            'warmup_steps': 3000,
            'weight_decay': 0.01,
            'ema_decay': 0.9999,
            'use_ema': True,
            'max_steps': 200000,
            'gradient_clip': 1.0,
        },
        'tokenizer': {
            'type': 'bpe',
            'vocab_size': 50000,
        },
    },
}


# =============================================================================
# Interactive UI
# =============================================================================

def print_header(text: str):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def find_interactive_checkpoints(base_dir: str = "./checkpoints/interactive") -> list[dict]:
    """Find existing checkpoints in the interactive folder.
    
    Returns list of checkpoint info dicts with path, step, config, etc.
    """
    checkpoints = []
    base_path = Path(base_dir)
    
    if not base_path.exists():
        return checkpoints
    
    # Look for checkpoint files
    for ckpt_file in base_path.glob("*.ckpt"):
        # Parse checkpoint info
        name = ckpt_file.stem
        step = None
        val_loss = None
        
        # Try to extract step from filename (dimba-{step:07d}-{val/loss:.4f})
        if "step=" in name or "-" in name:
            parts = name.split("-")
            for i, part in enumerate(parts):
                if part.isdigit():
                    step = int(part)
                elif ".ckpt" not in part and i > 0:
                    try:
                        val_loss = float(part)
                    except:
                        pass
        
        # Check for config
        config_path = ckpt_file.parent / "train_config.yaml"
        has_config = config_path.exists()
        
        # Check for tokenizer
        tokenizer_path = ckpt_file.parent / "tokenizer.json"
        has_tokenizer = tokenizer_path.exists()
        
        # Get file stats
        stat = ckpt_file.stat()
        size_mb = stat.st_size / (1024 * 1024)
        mtime = stat.st_mtime
        
        checkpoints.append({
            'path': str(ckpt_file),
            'name': name,
            'step': step,
            'val_loss': val_loss,
            'size_mb': size_mb,
            'modified': mtime,
            'has_config': has_config,
            'has_tokenizer': has_tokenizer,
            'config_path': str(config_path) if has_config else None,
            'dir': str(ckpt_file.parent),
        })
    
    # Sort by modified time (newest first)
    checkpoints.sort(key=lambda x: x['modified'], reverse=True)
    return checkpoints


def interactive_resume_selection(checkpoints: list[dict]) -> tuple[Optional[str], Optional[str], bool]:
    """Let user select whether to resume and which checkpoint to use.
    
    Returns: (checkpoint_path, config_path, use_new_config)
    """
    if not checkpoints:
        return None, None, False
    
    print_header("Existing Checkpoints Found")
    print(f"\nFound {len(checkpoints)} checkpoint(s) in interactive folder:\n")
    
    for i, ckpt in enumerate(checkpoints[:5], 1):  # Show top 5
        step_str = f"step {ckpt['step']}" if ckpt['step'] else "unknown step"
        loss_str = f"val_loss={ckpt['val_loss']:.4f}" if ckpt['val_loss'] else ""
        size_str = f"{ckpt['size_mb']:.1f} MB"
        config_marker = "✓" if ckpt['has_config'] else "✗"
        
        print(f"  [{i}] {ckpt['name']}")
        print(f"      {step_str} {loss_str} | {size_str} | config: {config_marker}")
    
    print("\nOptions:")
    print("  [r] Resume from latest checkpoint (use saved config)")
    print("  [n] Resume but use NEW config (keep weights, change settings)")
    print("  [1-5] Resume from specific checkpoint above")
    print("  [f] Start FRESH (ignore existing checkpoints)")
    
    choice = input("\nSelect option [r]: ").strip().lower() or 'r'
    
    if choice == 'f':
        return None, None, False
    elif choice == 'n':
        # Resume latest but use new config
        return checkpoints[0]['path'], None, True
    elif choice == 'r':
        # Resume latest with saved config
        ckpt = checkpoints[0]
        return ckpt['path'], ckpt['config_path'] if ckpt['has_config'] else None, False
    elif choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(checkpoints):
            ckpt = checkpoints[idx]
            return ckpt['path'], ckpt['config_path'] if ckpt['has_config'] else None, False
    
    # Default: resume latest
    ckpt = checkpoints[0]
    return ckpt['path'], ckpt['config_path'] if ckpt['has_config'] else None, False


def print_gpu_info(gpus: list[dict]):
    """Print GPU information."""
    if not gpus:
        print("\n⚠️  No GPUs detected. Will use CPU.")
        return

    print(f"\n🖥️  Detected {len(gpus)} GPU(s):")
    print("-" * 60)
    for gpu in gpus:
        print(f"  GPU {gpu['index']}: {gpu['name']}")
        print(f"    Memory: {gpu['memory_gb']:.1f} GB")
        print(f"    Tier: {gpu['tier']}")
        print(f"    Compute: {gpu['compute_capability']}")
    print("-" * 60)


def interactive_gpu_selection(gpus: list[dict]) -> tuple[int, list[int]]:
    """Let user select which GPUs to use."""
    if not gpus:
        return 0, []

    print_gpu_info(gpus)

    if len(gpus) == 1:
        use = input(f"\nUse GPU 0? [Y/n]: ").strip().lower()
        if use in ('n', 'no'):
            return 0, []
        return 1, [0]

    print("\nOptions:")
    print("  a) Use all GPUs")
    print("  s) Use specific GPU(s)")
    print("  c) Use CPU only")

    choice = input("\nSelect option [a/s/c]: ").strip().lower()

    if choice == 'c':
        return 0, []
    elif choice == 'a':
        return len(gpus), list(range(len(gpus)))
    else:
        indices = input("Enter GPU indices to use (comma-separated, e.g., 0,1): ").strip()
        try:
            selected = [int(x.strip()) for x in indices.split(',')]
            valid = [i for i in selected if 0 <= i < len(gpus)]
            if not valid:
                print("No valid GPUs selected. Using CPU.")
                return 0, []
            return len(valid), valid
        except ValueError:
            print("Invalid input. Using first GPU.")
            return 1, [0]


def interactive_preset_selection(gpus: list[dict]) -> dict:
    """Let user select or customize a preset."""
    print_header("Select Training Configuration")

    # Auto-suggest based on GPU
    suggested = None
    if gpus:
        total_memory = sum(g['memory_gb'] for g in gpus)
        max_tier = max(g['tier'] for g in gpus) if gpus else 'unknown'

        if max_tier == 'high' and total_memory > 70:
            suggested = 'a100-3b'
        elif max_tier in ('high', 'mid-high') or total_memory > 40:
            suggested = 'l40s-1b'
        elif max_tier == 'mid' or total_memory > 12:
            suggested = 'a4000-500m'
        else:
            suggested = 'cpu-small'

    print("\nAvailable presets:")
    for key, preset in PRESETS.items():
        marker = " 👈 SUGGESTED" if key == suggested else ""
        print(f"  [{key:12}] {preset['name']}{marker}")
        print(f"               {preset['description']}")

    print("\n  [custom]     Create custom configuration")
    print("  [file]       Load from YAML file")

    while True:
        choice = input(f"\nSelect preset [{suggested or 'custom'}]: ").strip()
        if not choice and suggested:
            choice = suggested

        if choice in PRESETS:
            return PRESETS[choice].copy()
        elif choice == 'custom':
            return interactive_custom_config(gpus)
        elif choice == 'file':
            path = input("Enter config file path: ").strip()
            try:
                with open(path) as f:
                    return yaml.safe_load(f)
            except Exception as e:
                print(f"Error loading file: {e}")
                continue
        else:
            print(f"Unknown preset: {choice}")


def interactive_custom_config(gpus: list[dict]) -> dict:
    """Guide user through creating a custom config."""
    print_header("Custom Configuration Wizard")

    config = {
        'model': {},
        'data': {},
        'training': {},
        'tokenizer': {},
    }

    # Model size
    print("\n1. Model Size:")
    print("   [tiny]   < 100M params (fast testing)")
    print("   [small]  ~100-300M params")
    print("   [medium] ~300M-1B params")
    print("   [large]  ~1-2B params")
    print("   [xlarge] > 2B params")

    size = input("Select size [medium]: ").strip() or 'medium'

    # Suggest dims based on size
    size_configs = {
        'tiny':   {'d_model': 256,  'layers': 4,  'd_state': 16},
        'small':  {'d_model': 768,  'layers': 12, 'd_state': 32},
        'medium': {'d_model': 1536, 'layers': 20, 'd_state': 64},
        'large':  {'d_model': 2560, 'layers': 28, 'd_state': 64},
        'xlarge': {'d_model': 3584, 'layers': 36, 'd_state': 128},
    }

    sc = size_configs.get(size, size_configs['medium'])

    config['model']['d_model'] = int(input(f"d_model [{sc['d_model']}]: ") or sc['d_model'])
    config['model']['d_prompt'] = config['model']['d_model']
    config['model']['num_denoiser_layers'] = int(input(f"Layers [{sc['layers']}]: ") or sc['layers'])
    config['model']['d_state'] = int(input(f"d_state [{sc['d_state']}]: ") or sc['d_state'])
    config['model']['d_conv'] = 4
    config['model']['expand'] = 2
    config['model']['conditioning_type'] = 'film'
    config['model']['dropout'] = 0.1
    config['model']['use_weight_tying'] = True
    config['model']['num_diffusion_steps'] = int(input("Diffusion steps [1000]: ") or 1000)

    # Data config
    print("\n2. Data Configuration:")
    dataset_type = input("Dataset type [huggingface/dummy] [huggingface]: ").strip() or 'huggingface'
    config['data']['type'] = dataset_type

    if dataset_type == 'huggingface':
        config['data']['dataset_name'] = input("Dataset name [HuggingFaceFW/fineweb]: ").strip() or 'HuggingFaceFW/fineweb'
        config['data']['dataset_config'] = input("Dataset config [sample-10BT]: ").strip() or 'sample-10BT'
        config['data']['streaming'] = input("Use streaming? [y/N]: ").strip().lower() == 'y'
    else:
        config['data']['num_examples'] = int(input("Number of examples [1000]: ") or 1000)

    # Batch size based on GPU
    default_batch = 8
    if gpus:
        avg_memory = sum(g['memory_gb'] for g in gpus) / len(gpus)
        default_batch = get_recommended_batch_size(avg_memory, size)

    config['data']['batch_size'] = int(input(f"Batch size [{default_batch}]: ") or default_batch)
    config['data']['max_length'] = int(input("Max sequence length [512]: ") or 512)
    config['data']['num_workers'] = int(input("Num workers [4]: ") or 4)

    # Training config
    print("\n3. Training Configuration:")
    config['training']['learning_rate'] = float(input("Learning rate [1e-4]: ") or 1e-4)
    config['training']['warmup_steps'] = int(input("Warmup steps [1000]: ") or 1000)
    config['training']['max_steps'] = int(input("Max steps [50000]: ") or 50000)
    config['training']['weight_decay'] = 0.01
    config['training']['ema_decay'] = 0.9999
    config['training']['use_ema'] = input("Use EMA? [Y/n]: ").strip().lower() != 'n'
    config['training']['gradient_clip'] = 1.0

    # Tokenizer
    print("\n4. Tokenizer:")
    tok_type = input("Tokenizer type [bpe/simple] [bpe]: ").strip() or 'bpe'
    config['tokenizer']['type'] = tok_type
    config['tokenizer']['vocab_size'] = int(input("Vocab size [32000]: ") or 32000)

    return config


def interactive_config_review(config: dict, gpus: list[dict]) -> dict:
    """Let user review and edit the config before training."""
    print_header("Configuration Review")

    print("\nCurrent configuration:")
    print(json.dumps(config, indent=2))

    if input("\nEdit configuration? [y/N]: ").strip().lower() == 'y':
        # Simple key-value editing
        print("\nEnter path=value to edit (e.g., model.d_model=1024)")
        print("Enter 'done' when finished")

        while True:
            edit = input("Edit: ").strip()
            if edit.lower() == 'done':
                break

            try:
                path, value = edit.split('=', 1)
                keys = path.split('.')

                # Navigate to the right dict
                target = config
                for key in keys[:-1]:
                    target = target[key]

                # Try to parse value
                try:
                    value = eval(value)
                except:
                    pass  # Keep as string

                target[keys[-1]] = value
                print(f"  Set {path} = {value}")
            except Exception as e:
                print(f"  Error: {e}")

    return config


def estimate_memory_usage(config: dict, num_gpus: int) -> dict:
    """Estimate memory usage for the configuration."""
    d_model = config['model']['d_model']
    num_layers = config['model']['num_denoiser_layers']
    vocab_size = config['tokenizer']['vocab_size']
    batch_size = config['data']['batch_size']
    max_length = config['data']['max_length']
    expand = config['model'].get('expand', 2)

    # Estimate parameters
    embedding_params = vocab_size * d_model
    mamba_params_per_layer = d_model * d_model * expand + d_model * 64 + d_model * 4
    mamba_params = num_layers * mamba_params_per_layer
    total_params = embedding_params + mamba_params

    # Memory estimates (FP16)
    model_mem = total_params * 2 / 1e9
    gradient_mem = model_mem
    optimizer_mem = model_mem * 2  # AdamW states
    activation_mem = batch_size * max_length * d_model * 2 / 1e9

    total_per_gpu = model_mem + gradient_mem + optimizer_mem + activation_mem

    return {
        'total_params': total_params,
        'model_gb': model_mem,
        'gradients_gb': gradient_mem,
        'optimizer_gb': optimizer_mem,
        'activations_gb': activation_mem,
        'total_per_gpu_gb': total_per_gpu,
        'model_size': f"{total_params / 1e6:.0f}M",
    }


def print_memory_estimate(est: dict, gpus: list[dict], selected_gpus: list[int]):
    """Print memory usage estimate."""
    print_header("Memory Usage Estimate")

    print(f"\nModel size: {est['model_size']} parameters")
    print(f"\nPer-GPU memory breakdown:")
    print(f"  Model parameters:  {est['model_gb']:.1f} GB")
    print(f"  Gradients:         {est['gradients_gb']:.1f} GB")
    print(f"  Optimizer states:  {est['optimizer_gb']:.1f} GB")
    print(f"  Activations:       {est['activations_gb']:.1f} GB")
    print(f"  ─────────────────────────")
    print(f"  Total per GPU:     {est['total_per_gpu_gb']:.1f} GB")

    if gpus and selected_gpus:
        for idx in selected_gpus:
            gpu = gpus[idx]
            headroom = gpu['memory_gb'] - est['total_per_gpu_gb']
            status = "✅ OK" if headroom > 2 else "⚠️  TIGHT" if headroom > 0 else "❌ EXCEEDS"
            print(f"\n  GPU {idx} ({gpu['name']}): {gpu['memory_gb']:.1f} GB")
            print(f"    Headroom: {headroom:.1f} GB {status}")


# =============================================================================
# Training
# =============================================================================

def create_dataloaders(config: dict, tokenizer):
    """Create training and validation dataloaders."""
    from torch.utils.data import DataLoader

    data_cfg = config['data']
    dataset_type = data_cfg.get('type', 'dummy')

    if dataset_type == 'dummy':
        train_dataset = DummyDataset(
            size=data_cfg.get('num_examples', 1000),
            vocab_size=tokenizer.vocab_size,
            seq_length=data_cfg['max_length'],
        )
        val_dataset = DummyDataset(
            size=data_cfg.get('num_examples', 1000) // 10,
            vocab_size=tokenizer.vocab_size,
            seq_length=data_cfg['max_length'],
        )
    elif dataset_type == 'huggingface':
        train_dataset = HuggingFaceDataset(
            dataset_name=data_cfg['dataset_name'],
            split='train',
            tokenizer=tokenizer,
            max_length=data_cfg['max_length'],
            streaming=data_cfg.get('streaming', False),
        )
        try:
            val_dataset = HuggingFaceDataset(
                dataset_name=data_cfg['dataset_name'],
                split='validation',
                tokenizer=tokenizer,
                max_length=data_cfg['max_length'],
                streaming=data_cfg.get('streaming', False),
            )
        except:
            print("Warning: No validation split found, using dummy validation set")
            val_dataset = DummyDataset(size=100, vocab_size=tokenizer.vocab_size, seq_length=data_cfg['max_length'])
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=data_cfg['batch_size'],
        shuffle=not data_cfg.get('streaming', False),
        num_workers=data_cfg.get('num_workers', 0),
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=data_cfg['batch_size'],
        shuffle=False,
        num_workers=data_cfg.get('num_workers', 0),
        collate_fn=collate_fn,
    )

    return train_loader, val_loader


def run_training(config: dict, num_gpus: int, gpu_indices: list[int], 
                 resume_ckpt: Optional[str] = None, force_new_config: bool = False,
                 hf_repo_id: Optional[str] = None, hf_token: Optional[str] = None,
                 hf_private: bool = False, skip_hf_prompt: bool = False):
    """Run the training loop.
    
    Args:
        config: Training configuration
        num_gpus: Number of GPUs to use
        gpu_indices: List of GPU indices
        resume_ckpt: Path to checkpoint to resume from (optional)
        force_new_config: If True, use the provided config instead of loading from checkpoint
        hf_repo_id: HuggingFace repo ID for auto-upload
        hf_token: HuggingFace API token
        hf_private: Whether to create private HF repo
        skip_hf_prompt: If True, don't prompt for HF upload (useful for --yes mode)
    """
    if not HAS_DEPS:
        print(f"\n❌ Error: Missing dependencies: {DEPS_ERROR}")
        print("Please install required packages:")
        print("  pip install pytorch-lightning datasets tokenizers")
        return

    print_header("Starting Training")

    # Determine checkpoint directory
    if resume_ckpt and not force_new_config:
        # Use the directory of the checkpoint
        checkpoint_dir = str(Path(resume_ckpt).parent)
        print(f"Resuming from: {resume_ckpt}")
        print(f"Checkpoint directory: {checkpoint_dir}")
    else:
        # Get checkpoint directory from user or use default
        checkpoint_dir = input("\nCheckpoint directory [./checkpoints/interactive]: ").strip() or "./checkpoints/interactive"
    
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load or save config
    config_path = os.path.join(checkpoint_dir, "train_config.yaml")
    
    if resume_ckpt and not force_new_config:
        # Load config from checkpoint directory
        if os.path.exists(config_path):
            print(f"Loading config from: {config_path}")
            with open(config_path) as f:
                config = yaml.safe_load(f)
        else:
            print("Warning: No config file found in checkpoint directory, using current config")
            # Save the config we're using
            with open(config_path, 'w') as f:
                yaml.dump(config, f, sort_keys=False)
    else:
        # Save new config
        with open(config_path, 'w') as f:
            yaml.dump(config, f, sort_keys=False)
        print(f"Config saved to: {config_path}")

    # Create tokenizer
    tok_cfg = config['tokenizer']
    if tok_cfg['type'] == 'bpe':
        tokenizer = BPETokenizer(vocab_size=tok_cfg['vocab_size'])
    else:
        tokenizer = SimpleCharacterTokenizer(vocab_size=tok_cfg['vocab_size'])

    print(f"Tokenizer: {type(tokenizer).__name__} (vocab_size={tokenizer.vocab_size})")

    # Save tokenizer
    tokenizer.save(os.path.join(checkpoint_dir, "tokenizer.json"))

    # Create dataloaders
    print("Loading datasets...")
    train_loader, val_loader = create_dataloaders(config, tokenizer)

    # Create or load model
    print("Creating model...")
    train_cfg = config['training']
    model_config = config['model']

    lightning_module = DIMBALightningModule(
        vocab_size=tokenizer.vocab_size,
        model_config=model_config,
        learning_rate=float(train_cfg['learning_rate']),
        warmup_steps=int(train_cfg['warmup_steps']),
        ema_decay=float(train_cfg.get('ema_decay', 0.9999)),
        use_ema=train_cfg.get('use_ema', True),
        ema_device=train_cfg.get('ema_device', 'cpu'),
    )

    total_params = sum(p.numel() for p in lightning_module.model.parameters())
    print(f"Model parameters: {total_params:,} (~{total_params/1e6:.1f}M)")

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="dimba-{step:07d}-{val/loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )

    callbacks = [checkpoint_callback]
    if input("Enable early stopping? [y/N]: ").strip().lower() == 'y':
        early_stop = EarlyStopping(monitor="val/loss", patience=5, mode="min")
        callbacks.append(early_stop)

    # Trainer setup
    trainer_kwargs = {
        'max_steps': int(train_cfg['max_steps']),
        'callbacks': callbacks,
        'logger': TensorBoardLogger("./logs", name="dimba_interactive"),
        'log_every_n_steps': int(train_cfg.get('log_interval', 50)),
        'val_check_interval': int(train_cfg.get('val_interval', 500)),
        'gradient_clip_val': float(train_cfg.get('gradient_clip', 1.0)),
        'accumulate_grad_batches': int(train_cfg.get('accumulate_grad_batches', 1)),
    }
    
    # Add resume_from_checkpoint if specified
    if resume_ckpt:
        trainer_kwargs['resume_from_checkpoint'] = resume_ckpt
        print(f"Resuming training from checkpoint: {resume_ckpt}")

    if num_gpus > 0:
        trainer_kwargs['accelerator'] = 'gpu'
        trainer_kwargs['devices'] = len(gpu_indices) if len(gpu_indices) > 1 else 1
        if len(gpu_indices) > 1:
            trainer_kwargs['strategy'] = 'ddp'
        trainer_kwargs['precision'] = '16-mixed'
    else:
        trainer_kwargs['accelerator'] = 'cpu'
        trainer_kwargs['devices'] = 1

    print(f"\nTraining on: {trainer_kwargs.get('accelerator', 'cpu').upper()}")
    if num_gpus > 0:
        print(f"  GPUs: {gpu_indices}")
        print(f"  Precision: {trainer_kwargs['precision']}")

    if input("\nStart training? [Y/n]: ").strip().lower() == 'n':
        print("Training cancelled.")
        return

    trainer = pl.Trainer(**trainer_kwargs)

    try:
        trainer.fit(lightning_module, train_dataloaders=train_loader, val_dataloaders=val_loader)
        print(f"\n✅ Training complete!")
        print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
        
        # HuggingFace upload
        if hf_repo_id:
            # Auto-upload if repo specified via CLI
            upload_to_huggingface(checkpoint_dir, repo_id=hf_repo_id, token=hf_token, private=hf_private)
        elif not skip_hf_prompt:
            # Interactive prompt
            interactive_upload_prompt(checkpoint_dir)
            
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        print(f"Last checkpoint: {checkpoint_callback.last_model_path}")
        print("\nTo resume training, run:")
        print(f"  python scripts/train_interactive.py --resume")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        raise


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Interactive DIMBA training with auto GPU detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch interactive wizard
  python scripts/train_interactive.py

  # Resume from latest checkpoint
  python scripts/train_interactive.py --resume

  # Resume from specific checkpoint directory
  python scripts/train_interactive.py --resume-from ./checkpoints/interactive

  # Resume with new config (keep weights, change settings)
  python scripts/train_interactive.py --resume --new-config

  # Use specific preset with auto HF upload
  python scripts/train_interactive.py --preset a4000-500m --hf-repo-id username/dimba-500m

  # Load config with auto GPU detection
  python scripts/train_interactive.py --config my_config.yaml --auto-gpu

  # Non-interactive mode with specific GPUs and HF upload
  python scripts/train_interactive.py --preset l40s-1b --gpus 0,1 --yes --hf-repo-id username/dimba-1b
        """
    )
    parser.add_argument('--preset', type=str, choices=list(PRESETS.keys()),
                        help='Use a predefined configuration preset')
    parser.add_argument('--config', type=str, help='Load configuration from YAML file')
    parser.add_argument('--auto-gpu', action='store_true',
                        help='Auto-detect and use best GPU configuration')
    parser.add_argument('--gpus', type=str, help='Specific GPU indices to use (comma-separated, e.g., 0,1)')
    parser.add_argument('--yes', '-y', action='store_true',
                        help='Skip confirmation prompts (non-interactive mode)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from latest checkpoint in interactive folder')
    parser.add_argument('--resume-from', type=str,
                        help='Resume from specific checkpoint path or directory')
    parser.add_argument('--new-config', action='store_true',
                        help='When resuming, use new config instead of saved config')
    
    # HuggingFace upload arguments
    parser.add_argument('--hf-repo-id', type=str,
                        help='HuggingFace repo ID for auto-upload (username/model-name)')
    parser.add_argument('--hf-token', type=str,
                        help='HuggingFace API token (or set HF_TOKEN env var)')
    parser.add_argument('--hf-private', action='store_true',
                        help='Create private HuggingFace repo')
    parser.add_argument('--skip-hf-upload', action='store_true',
                        help='Skip HuggingFace upload prompt after training')

    args = parser.parse_args()

    print("=" * 60)
    print("  DIMBA Interactive Training")
    print("=" * 60)

    # Check dependencies early
    if not HAS_DEPS:
        print(f"\n⚠️  Warning: Some dependencies not available: {DEPS_ERROR}")
        print("You can still configure, but training will fail.")
        if not args.yes:
            input("Press Enter to continue...")

    # Handle resume logic
    resume_ckpt = None
    loaded_config_path = None
    force_new_config = args.new_config
    
    if args.resume or args.resume_from:
        if args.resume_from:
            # User specified a path
            resume_path = Path(args.resume_from)
            if resume_path.is_file() and resume_path.suffix == '.ckpt':
                resume_ckpt = str(resume_path)
                loaded_config_path = str(resume_path.parent / "train_config.yaml")
            elif resume_path.is_dir():
                # Find latest checkpoint in directory
                checkpoints = find_interactive_checkpoints(str(resume_path))
                if checkpoints:
                    resume_ckpt = checkpoints[0]['path']
                    loaded_config_path = checkpoints[0]['config_path']
                else:
                    print(f"No checkpoints found in {resume_path}")
                    return
            else:
                print(f"Invalid resume path: {args.resume_from}")
                return
        else:
            # Auto-find latest checkpoint
            checkpoints = find_interactive_checkpoints()
            if checkpoints:
                if args.yes:
                    # Non-interactive: just use latest
                    resume_ckpt = checkpoints[0]['path']
                    loaded_config_path = checkpoints[0]['config_path']
                else:
                    # Interactive selection
                    resume_ckpt, loaded_config_path, force_new_config = interactive_resume_selection(checkpoints)
            else:
                print("\nNo existing checkpoints found to resume from.")
                if input("Start fresh training? [Y/n]: ").strip().lower() == 'n':
                    return
    
    # Detect GPUs
    gpus = detect_gpus()

    # Determine GPU configuration
    if args.gpus:
        # User specified GPUs
        gpu_indices = [int(x.strip()) for x in args.gpus.split(',')]
        gpu_indices = [i for i in gpu_indices if i < len(gpus)]
        num_gpus = len(gpu_indices)
    elif args.auto_gpu:
        # Auto-detect: use all available
        num_gpus = len(gpus)
        gpu_indices = list(range(len(gpus)))
    else:
        # Interactive selection (skip if resuming non-interactively)
        if resume_ckpt and args.yes:
            num_gpus = len(gpus)
            gpu_indices = list(range(len(gpus)))
        else:
            num_gpus, gpu_indices = interactive_gpu_selection(gpus)

    # Get configuration
    if force_new_config or (not resume_ckpt):
        # Use new config (either fresh start or --new-config flag)
        if args.config:
            with open(args.config) as f:
                config = yaml.safe_load(f)
        elif args.preset:
            config = PRESETS[args.preset].copy()
        else:
            config = interactive_preset_selection(gpus)
        
        # Review and edit config
        if not args.yes:
            config = interactive_config_review(config, gpus)
    else:
        # Load config from checkpoint
        if loaded_config_path and os.path.exists(loaded_config_path):
            print(f"\nLoading configuration from checkpoint: {loaded_config_path}")
            with open(loaded_config_path) as f:
                config = yaml.safe_load(f)
            print("✅ Config loaded successfully")
            
            if not args.yes:
                # Still allow editing
                config = interactive_config_review(config, gpus)
        else:
            print("\n⚠️  No config file found in checkpoint directory.")
            print("You'll need to specify configuration.")
            if args.preset:
                config = PRESETS[args.preset].copy()
            else:
                config = interactive_preset_selection(gpus)

    # Show memory estimate
    mem_estimate = estimate_memory_usage(config, num_gpus if num_gpus > 0 else 1)
    print_memory_estimate(mem_estimate, gpus, gpu_indices)

    # Final confirmation
    if not args.yes:
        if input("\nProceed with training? [Y/n]: ").strip().lower() == 'n':
            print("Training cancelled.")
            return

    # Run training with HF upload params
    run_training(
        config, num_gpus, gpu_indices, 
        resume_ckpt, force_new_config,
        hf_repo_id=args.hf_repo_id,
        hf_token=args.hf_token,
        hf_private=args.hf_private,
        skip_hf_prompt=args.skip_hf_upload or args.yes
    )


def upload_to_huggingface(checkpoint_dir: str, repo_id: Optional[str] = None, 
                          token: Optional[str] = None, private: bool = False) -> bool:
    """Upload training artifacts to HuggingFace Hub.
    
    Args:
        checkpoint_dir: Directory containing checkpoints and config
        repo_id: HuggingFace repo ID (username/model-name). If None, will prompt.
        token: HuggingFace API token. If None, will check env var or prompt.
        private: Whether to create private repo
    
    Returns:
        True if upload successful, False otherwise
    """
    try:
        from huggingface_hub import HfApi, upload_folder
    except ImportError:
        print("\n⚠️  huggingface_hub not installed. Install with:")
        print("  pip install huggingface_hub")
        return False
    
    # Get repo ID
    if repo_id is None:
        repo_id = input("\nHuggingFace repo ID (username/model-name): ").strip()
        if not repo_id:
            print("No repo ID provided, skipping upload.")
            return False
    
    # Get token
    if token is None:
        token = os.getenv("HF_TOKEN")
        if token is None:
            token = input("HuggingFace token (or set HF_TOKEN env var): ").strip()
            if not token:
                print("No token provided, skipping upload.")
                return False
    
    try:
        print(f"\n📤 Uploading to HuggingFace: https://huggingface.co/{repo_id}")
        print(f"   Directory: {checkpoint_dir}")
        print(f"   Private: {private}")
        
        api = HfApi(token=token)
        
        # Create repo if doesn't exist
        api.create_repo(
            repo_id=repo_id, 
            repo_type="model", 
            private=private, 
            exist_ok=True
        )
        
        # Upload all files in checkpoint directory
        upload_folder(
            repo_id=repo_id,
            folder_path=checkpoint_dir,
            repo_type="model",
            token=token,
            commit_message="Upload DIMBA training checkpoint",
        )
        
        print(f"\n✅ Upload complete!")
        print(f"   URL: https://huggingface.co/{repo_id}")
        return True
        
    except Exception as e:
        print(f"\n❌ Upload failed: {e}")
        return False


def interactive_upload_prompt(checkpoint_dir: str) -> bool:
    """Prompt user for HF upload after training."""
    if input("\nUpload to HuggingFace Hub? [y/N]: ").strip().lower() != 'y':
        return False
    
    # Ask for repo details
    repo_id = input("Repo ID (username/model-name): ").strip()
    if not repo_id:
        print("No repo ID provided, skipping upload.")
        return False
    
    private = input("Make repo private? [y/N]: ").strip().lower() == 'y'
    
    return upload_to_huggingface(checkpoint_dir, repo_id=repo_id, private=private)
    main()
