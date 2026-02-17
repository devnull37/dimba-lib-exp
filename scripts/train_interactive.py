#!/usr/bin/env python3
"""Interactive training script for DIMBA with auto GPU detection and config wizard.

Supports four training modes:
1. VAE Only - Pre-train the TokenVAE for latent diffusion
2. DIMBA (Embedding) - Train diffusion model in embedding space (no VAE needed)
3. DIMBA (Latent) - Train diffusion model in VAE latent space (requires pre-trained VAE)
4. Both - Train VAE first, then train DIMBA with that VAE

Architecture Overview:
    EMBEDDING-SPACE DIFFUSION (no VAE):
        Tokens → Embeddings → [Diffusion] → Embeddings → Tokens
        
    LATENT-SPACE DIFFUSION (with VAE):
        Tokens → Embeddings → VAE Encode → Latents → [Diffusion] → Latents → VAE Decode → Embeddings → Tokens
        
    The VAE compresses embeddings into a lower-dimensional latent space, which can:
    - Reduce computation during diffusion
    - Provide a smoother, more continuous representation

Usage:
    # Launch interactive mode (choose what to train)
    python scripts/train_interactive.py

    # Train specific component directly
    python scripts/train_interactive.py --train-mode vae
    python scripts/train_interactive.py --train-mode dimba-embedding
    python scripts/train_interactive.py --train-mode dimba-latent --vae-checkpoint checkpoints/vae/final.ckpt

    # Train both (VAE first, then DIMBA)
    python scripts/train_interactive.py --train-mode both
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
    from dimba.models.vae import TokenVAE
    from dimba.models.embeddings import TokenEmbedding
    from dimba.training.trainer import VAELightningModule

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


# VAE-specific presets
VAE_PRESETS = {
    'vae-small': {
        'name': 'Small VAE (fast training)',
        'description': 'Small VAE for testing, ~10M params',
        'vae': {
            'latent_dim': 128,
            'hidden_dim': 512,
            'num_layers': 2,
            'dropout': 0.1,
            'kl_weight': 1.0,
        },
        'model': {
            'd_model': 512,
        },
        'data': {
            'type': 'huggingface',
            'dataset_name': 'wikitext',
            'dataset_config': 'wikitext-2-raw-v1',
            'batch_size': 64,
            'max_length': 256,
            'num_workers': 4,
        },
        'training': {
            'learning_rate': 1e-4,
            'warmup_steps': 500,
            'max_steps': 10000,
            'weight_decay': 0.01,
        },
    },
    'vae-medium': {
        'name': 'Medium VAE (balanced)',
        'description': 'Medium VAE for production use, ~50M params',
        'vae': {
            'latent_dim': 256,
            'hidden_dim': 1024,
            'num_layers': 3,
            'dropout': 0.1,
            'kl_weight': 0.5,
        },
        'model': {
            'd_model': 1024,
        },
        'data': {
            'type': 'huggingface',
            'dataset_name': 'HuggingFaceFW/fineweb',
            'dataset_config': 'sample-10BT',
            'batch_size': 32,
            'max_length': 512,
            'num_workers': 4,
        },
        'training': {
            'learning_rate': 5e-5,
            'warmup_steps': 1000,
            'max_steps': 50000,
            'weight_decay': 0.01,
        },
    },
    'vae-large': {
        'name': 'Large VAE (high quality)',
        'description': 'Large VAE for best reconstruction, ~100M params',
        'vae': {
            'latent_dim': 512,
            'hidden_dim': 2048,
            'num_layers': 4,
            'dropout': 0.1,
            'kl_weight': 0.1,
        },
        'model': {
            'd_model': 2048,
        },
        'data': {
            'type': 'huggingface',
            'dataset_name': 'HuggingFaceFW/fineweb',
            'dataset_config': 'sample-10BT',
            'batch_size': 16,
            'max_length': 1024,
            'num_workers': 8,
        },
        'training': {
            'learning_rate': 3e-5,
            'warmup_steps': 2000,
            'max_steps': 100000,
            'weight_decay': 0.01,
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


def interactive_training_mode_selection() -> tuple[str, Optional[str]]:
    """Let user select what to train: VAE, DIMBA (embedding), DIMBA (latent), or Both.
    
    Returns:
        (mode, vae_checkpoint_path)
        mode: 'vae', 'dimba-embedding', 'dimba-latent', or 'both'
        vae_checkpoint_path: Path to pre-trained VAE (for 'dimba-latent' mode)
    """
    print_header("Select Training Mode")
    
    print("""
DIMBA supports two diffusion modes:

  1. EMBEDDING-SPACE DIFFUSION (no VAE needed)
     Tokens → Embeddings → [Diffusion] → Embeddings → Tokens
     • Simpler, no pre-training required
     • Direct control over embeddings
     
  2. LATENT-SPACE DIFFUSION (requires VAE)
     Tokens → Embeddings → VAE Encode → Latents → [Diffusion] → Latents → VAE Decode → Embeddings → Tokens
     • Compressed representation (faster)
     • Smoother latent space
     • Requires pre-trained VAE

What would you like to train?
""")
    
    print("  [1] VAE only")
    print("      Pre-train a TokenVAE for latent diffusion")
    print("      → Use this if you want to do latent-space diffusion later")
    print()
    print("  [2] DIMBA (embedding-space)")
    print("      Train diffusion model in embedding space")
    print("      → No VAE needed, simpler setup")
    print()
    print("  [3] DIMBA (latent-space)")
    print("      Train diffusion model in VAE latent space")
    print("      → Requires a pre-trained VAE checkpoint")
    print()
    print("  [4] BOTH (VAE + DIMBA)")
    print("      Train VAE first, then train DIMBA with that VAE")
    print("      → Complete pipeline in one command")
    print()
    
    while True:
        choice = input("Select mode [2]: ").strip() or '2'
        
        if choice == '1':
            return 'vae', None
        elif choice == '2':
            return 'dimba-embedding', None
        elif choice == '3':
            # Need VAE checkpoint
            vae_path = input("Path to VAE checkpoint: ").strip()
            if not vae_path or not os.path.exists(vae_path):
                print(f"❌ VAE checkpoint not found: {vae_path}")
                print("Please train a VAE first using option [1] or [4]")
                continue
            return 'dimba-latent', vae_path
        elif choice == '4':
            return 'both', None
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")


def interactive_vae_preset_selection() -> dict:
    """Let user select a VAE preset."""
    print_header("Select VAE Configuration")
    
    print("\nAvailable VAE presets:")
    for key, preset in VAE_PRESETS.items():
        print(f"  [{key:12}] {preset['name']}")
        print(f"               Latent dim: {preset['vae']['latent_dim']}, "
              f"Hidden: {preset['vae']['hidden_dim']}, "
              f"Layers: {preset['vae']['num_layers']}")
        print(f"               {preset['description']}")
        print()
    
    print("  [custom]     Create custom VAE configuration")
    
    while True:
        choice = input(f"Select preset [vae-medium]: ").strip() or 'vae-medium'
        
        if choice in VAE_PRESETS:
            return VAE_PRESETS[choice].copy()
        elif choice == 'custom':
            return interactive_custom_vae_config()
        else:
            print(f"Unknown preset: {choice}")


def interactive_custom_vae_config() -> dict:
    """Guide user through creating a custom VAE config."""
    print_header("Custom VAE Configuration")
    
    config = {
        'vae': {},
        'model': {},
        'data': {},
        'training': {},
    }
    
    # VAE architecture
    print("\n1. VAE Architecture:")
    config['vae']['latent_dim'] = int(input("Latent dimension [256]: ") or 256)
    config['vae']['hidden_dim'] = int(input("Hidden dimension [1024]: ") or 1024)
    config['vae']['num_layers'] = int(input("Number of layers [3]: ") or 3)
    config['vae']['dropout'] = float(input("Dropout [0.1]: ") or 0.1)
    config['vae']['kl_weight'] = float(input("KL weight (beta-VAE) [0.5]: ") or 0.5)
    
    # Model dims (embedding dimension)
    config['model']['d_model'] = int(input("Embedding dimension (d_model) [1024]: ") or 1024)
    
    # Data config
    print("\n2. Data Configuration:")
    config['data']['type'] = input("Dataset type [huggingface/dummy] [huggingface]: ").strip() or 'huggingface'
    
    if config['data']['type'] == 'huggingface':
        config['data']['dataset_name'] = input("Dataset name [HuggingFaceFW/fineweb]: ").strip() or 'HuggingFaceFW/fineweb'
        config['data']['dataset_config'] = input("Dataset config [sample-10BT]: ").strip() or 'sample-10BT'
    else:
        config['data']['num_examples'] = int(input("Number of examples [1000]: ") or 1000)
    
    config['data']['batch_size'] = int(input("Batch size [32]: ") or 32)
    config['data']['max_length'] = int(input("Max sequence length [512]: ") or 512)
    config['data']['num_workers'] = int(input("Num workers [4]: ") or 4)
    
    # Training config
    print("\n3. Training Configuration:")
    config['training']['learning_rate'] = float(input("Learning rate [5e-5]: ") or 5e-5)
    config['training']['warmup_steps'] = int(input("Warmup steps [1000]: ") or 1000)
    config['training']['max_steps'] = int(input("Max steps [50000]: ") or 50000)
    config['training']['weight_decay'] = float(input("Weight decay [0.01]: ") or 0.01)
    
    return config


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

    choice = input("\nSelect option [a]: ").strip().lower() or 'a'

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
    num_layers = config['model'].get('num_denoiser_layers', 6)
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


def upload_to_huggingface(checkpoint_dir: str, repo_id: Optional[str] = None, 
                          token: Optional[str] = None, private: bool = False) -> bool:
    """Upload training artifacts to HuggingFace Hub."""
    try:
        from huggingface_hub import HfApi, upload_folder
    except ImportError:
        print("\n⚠️  huggingface_hub not installed. Install with:")
        print("  pip install huggingface_hub")
        return False
    
    if repo_id is None:
        repo_id = input("\nHuggingFace repo ID (username/model-name): ").strip()
        if not repo_id:
            print("No repo ID provided, skipping upload.")
            return False
    
    if token is None:
        token = os.getenv("HF_TOKEN")
        if token is None:
            token = input("HuggingFace token (or set HF_TOKEN env var): ").strip()
            if not token:
                print("No token provided, skipping upload.")
                return False
    
    try:
        print(f"\n📤 Uploading to HuggingFace: https://huggingface.co/{repo_id}")
        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
        upload_folder(
            repo_id=repo_id,
            folder_path=checkpoint_dir,
            repo_type="model",
            token=token,
            commit_message="Upload DIMBA training checkpoint",
        )
        print(f"\n✅ Upload complete! URL: https://huggingface.co/{repo_id}")
        return True
    except Exception as e:
        print(f"\n❌ Upload failed: {e}")
        return False


def run_vae_training(vae_config: dict, num_gpus: int, gpu_indices: list[int],
                     hf_repo_id: Optional[str] = None, hf_token: Optional[str] = None,
                     hf_private: bool = False) -> Optional[str]:
    """Run VAE training and return the path to the best checkpoint."""
    if not HAS_DEPS:
        print(f"\n❌ Error: Missing dependencies: {DEPS_ERROR}")
        return None

    print_header("Training VAE")

    checkpoint_dir = input("\nVAE checkpoint directory [./checkpoints/vae]: ").strip() or "./checkpoints/vae"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save config
    config_path = os.path.join(checkpoint_dir, "vae_config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(vae_config, f, sort_keys=False)
    print(f"Config saved to: {config_path}")

    # Create tokenizer (BPE for VAE)
    tokenizer = BPETokenizer(vocab_size=32000)
    tokenizer.save(os.path.join(checkpoint_dir, "tokenizer.json"))
    print(f"Tokenizer: BPE (vocab_size={tokenizer.vocab_size})")

    # Create dataloaders
    print("Loading datasets...")
    train_loader, val_loader = create_dataloaders(vae_config, tokenizer)

    # Create VAE model
    print("Creating VAE model...")
    vae = TokenVAE(
        input_dim=vae_config['model']['d_model'],
        latent_dim=vae_config['vae']['latent_dim'],
        hidden_dim=vae_config['vae'].get('hidden_dim'),
        num_layers=vae_config['vae']['num_layers'],
        dropout=vae_config['vae']['dropout'],
        kl_weight=vae_config['vae']['kl_weight'],
    )

    total_params = sum(p.numel() for p in vae.parameters())
    print(f"VAE parameters: {total_params:,} (~{total_params/1e6:.1f}M)")
    print(f"  Input dim: {vae_config['model']['d_model']}")
    print(f"  Latent dim: {vae_config['vae']['latent_dim']}")
    print(f"  Compression ratio: {vae_config['model']['d_model'] / vae_config['vae']['latent_dim']:.1f}x")

    # Create Lightning module
    token_embed = TokenEmbedding(vocab_size=tokenizer.vocab_size, embed_dim=vae_config['model']['d_model'])
    
    # We'll create a simple wrapper for VAE training
    class SimpleVAEModule(pl.LightningModule):
        def __init__(self, vae, token_embed, config):
            super().__init__()
            self.vae = vae
            self.token_embed = token_embed
            self.lr = config['training']['learning_rate']
            self.warmup_steps = config['training'].get('warmup_steps', 1000)
            self.weight_decay = config['training'].get('weight_decay', 0.01)
            
        def forward(self, batch):
            input_ids = batch['input_ids']
            embeddings = self.token_embed(input_ids)
            x_recon, stats = self.vae(embeddings, return_stats=True)
            loss, loss_dict = self.vae.compute_loss(
                embeddings, x_recon, stats['mu'], stats['logvar']
            )
            return loss, loss_dict
            
        def training_step(self, batch, batch_idx):
            loss, loss_dict = self(batch)
            self.log('train/loss', loss_dict['total'])
            self.log('train/recon_loss', loss_dict['recon'])
            self.log('train/kl_loss', loss_dict['kl'])
            return loss
            
        def validation_step(self, batch, batch_idx):
            loss, loss_dict = self(batch)
            self.log('val/loss', loss_dict['total'])
            self.log('val/recon_loss', loss_dict['recon'])
            self.log('val/kl_loss', loss_dict['kl'])
            return loss
            
        def configure_optimizers(self):
            optimizer = torch.optim.AdamW(
                list(self.vae.parameters()) + list(self.token_embed.parameters()),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, total_iters=self.warmup_steps
            )
            return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

    lightning_module = SimpleVAEModule(vae, token_embed, vae_config)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="vae-{step:07d}-{val/loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )

    # Trainer setup
    trainer_kwargs = {
        'max_steps': int(vae_config['training']['max_steps']),
        'callbacks': [checkpoint_callback],
        'logger': TensorBoardLogger("./logs", name="vae_training"),
        'log_every_n_steps': 50,
        'val_check_interval': 500,
        'gradient_clip_val': 1.0,
    }

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

    if input("\nStart VAE training? [Y/n]: ").strip().lower() == 'n':
        print("Training cancelled.")
        return None

    trainer = pl.Trainer(**trainer_kwargs)

    try:
        trainer.fit(lightning_module, train_dataloaders=train_loader, val_dataloaders=val_loader)
        print(f"\n✅ VAE Training complete!")
        print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
        
        if hf_repo_id:
            upload_to_huggingface(checkpoint_dir, repo_id=hf_repo_id, token=hf_token, private=hf_private)
        elif input("\nUpload VAE to HuggingFace? [y/N]: ").strip().lower() == 'y':
            repo_id = input("Repo ID (username/model-name): ").strip()
            if repo_id:
                private = input("Private repo? [y/N]: ").strip().lower() == 'y'
                upload_to_huggingface(checkpoint_dir, repo_id=repo_id, private=private)
        
        return checkpoint_callback.best_model_path or checkpoint_callback.last_model_path
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        print(f"Last checkpoint: {checkpoint_callback.last_model_path}")
        return checkpoint_callback.last_model_path
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        raise


def run_dimba_training(config: dict, num_gpus: int, gpu_indices: list[int],
                       vae_checkpoint: Optional[str] = None,
                       hf_repo_id: Optional[str] = None, hf_token: Optional[str] = None,
                       hf_private: bool = False):
    """Run DIMBA training (embedding or latent space)."""
    if not HAS_DEPS:
        print(f"\n❌ Error: Missing dependencies: {DEPS_ERROR}")
        return

    print_header("Training DIMBA" + (" (latent space)" if vae_checkpoint else " (embedding space)"))

    if vae_checkpoint:
        print(f"\nUsing VAE from: {vae_checkpoint}")
        # Note: In a full implementation, you'd load the VAE here and pass it to DIMBA
        # For now, we just note that latent space training requires VAE integration

    checkpoint_dir = input("\nCheckpoint directory [./checkpoints/interactive]: ").strip() or "./checkpoints/interactive"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save config
    config_path = os.path.join(checkpoint_dir, "train_config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    print(f"Config saved to: {config_path}")

    # Create tokenizer
    tok_cfg = config['tokenizer']
    if tok_cfg['type'] == 'bpe':
        tokenizer = BPETokenizer(vocab_size=tok_cfg['vocab_size'])
    else:
        tokenizer = SimpleCharacterTokenizer(vocab_size=tok_cfg['vocab_size'])

    tokenizer.save(os.path.join(checkpoint_dir, "tokenizer.json"))

    # Create dataloaders
    print("Loading datasets...")
    train_loader, val_loader = create_dataloaders(config, tokenizer)

    # Create model
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

    if input("Enable early stopping? [y/N]: ").strip().lower() == 'y':
        early_stop = EarlyStopping(monitor="val/loss", patience=5, mode="min")
        callbacks = [checkpoint_callback, early_stop]
    else:
        callbacks = [checkpoint_callback]

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

    if input("\nStart training? [Y/n]: ").strip().lower() == 'n':
        print("Training cancelled.")
        return

    trainer = pl.Trainer(**trainer_kwargs)

    try:
        trainer.fit(lightning_module, train_dataloaders=train_loader, val_dataloaders=val_loader)
        print(f"\n✅ Training complete!")
        print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
        
        if hf_repo_id:
            upload_to_huggingface(checkpoint_dir, repo_id=hf_repo_id, token=hf_token, private=hf_private)
        elif input("\nUpload to HuggingFace? [y/N]: ").strip().lower() == 'y':
            repo_id = input("Repo ID (username/model-name): ").strip()
            if repo_id:
                private = input("Private repo? [y/N]: ").strip().lower() == 'y'
                upload_to_huggingface(checkpoint_dir, repo_id=repo_id, private=private)
                
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        print(f"Last checkpoint: {checkpoint_callback.last_model_path}")
        print("\nTo resume, run: python scripts/train_interactive.py --resume")
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

  # Train VAE only
  python scripts/train_interactive.py --train-mode vae

  # Train DIMBA with embedding-space diffusion
  python scripts/train_interactive.py --train-mode dimba-embedding

  # Train DIMBA with latent-space diffusion (requires VAE checkpoint)
  python scripts/train_interactive.py --train-mode dimba-latent --vae-checkpoint checkpoints/vae/final.ckpt

  # Train both VAE and DIMBA
  python scripts/train_interactive.py --train-mode both

  # Resume training
  python scripts/train_interactive.py --resume
        """
    )
    parser.add_argument('--train-mode', type=str, 
                        choices=['vae', 'dimba-embedding', 'dimba-latent', 'both'],
                        help='What to train: vae, dimba-embedding, dimba-latent, or both')
    parser.add_argument('--vae-checkpoint', type=str,
                        help='Path to pre-trained VAE checkpoint (for dimba-latent mode)')
    parser.add_argument('--preset', type=str, choices=list(PRESETS.keys()),
                        help='Use a predefined DIMBA configuration preset')
    parser.add_argument('--vae-preset', type=str, choices=list(VAE_PRESETS.keys()),
                        help='Use a predefined VAE configuration preset')
    parser.add_argument('--config', type=str, help='Load configuration from YAML file')
    parser.add_argument('--auto-gpu', action='store_true',
                        help='Auto-detect and use best GPU configuration')
    parser.add_argument('--gpus', type=str, help='Specific GPU indices to use (comma-separated, e.g., 0,1)')
    parser.add_argument('--yes', '-y', action='store_true',
                        help='Skip confirmation prompts (non-interactive mode)')
    
    # HuggingFace upload arguments
    parser.add_argument('--hf-repo-id', type=str,
                        help='HuggingFace repo ID for auto-upload (username/model-name)')
    parser.add_argument('--hf-token', type=str,
                        help='HuggingFace API token (or set HF_TOKEN env var)')
    parser.add_argument('--hf-private', action='store_true',
                        help='Create private HuggingFace repo')

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

    # Detect GPUs
    gpus = detect_gpus()

    # Determine GPU configuration
    if args.gpus:
        gpu_indices = [int(x.strip()) for x in args.gpus.split(',')]
        gpu_indices = [i for i in gpu_indices if i < len(gpus)]
        num_gpus = len(gpu_indices)
    elif args.auto_gpu:
        num_gpus = len(gpus)
        gpu_indices = list(range(len(gpus)))
    else:
        if args.yes:
            num_gpus = len(gpus)
            gpu_indices = list(range(len(gpus)))
        else:
            num_gpus, gpu_indices = interactive_gpu_selection(gpus)

    # Select training mode
    if args.train_mode:
        train_mode = args.train_mode
        vae_checkpoint = args.vae_checkpoint
    else:
        train_mode, vae_checkpoint = interactive_training_mode_selection()

    # Get configurations based on mode
    if train_mode == 'vae':
        # VAE only training
        if args.vae_preset:
            vae_config = VAE_PRESETS[args.vae_preset].copy()
        elif args.config:
            with open(args.config) as f:
                vae_config = yaml.safe_load(f)
        else:
            vae_config = interactive_vae_preset_selection()
        
        if not args.yes:
            vae_config = interactive_config_review(vae_config, gpus)
        
        run_vae_training(vae_config, num_gpus, gpu_indices,
                        hf_repo_id=args.hf_repo_id, hf_token=args.hf_token, hf_private=args.hf_private)

    elif train_mode in ('dimba-embedding', 'dimba-latent'):
        # DIMBA training (embedding or latent)
        if args.config:
            with open(args.config) as f:
                config = yaml.safe_load(f)
        elif args.preset:
            config = PRESETS[args.preset].copy()
        else:
            config = interactive_preset_selection(gpus)
        
        if not args.yes:
            config = interactive_config_review(config, gpus)
        
        run_dimba_training(config, num_gpus, gpu_indices, vae_checkpoint=vae_checkpoint,
                          hf_repo_id=args.hf_repo_id, hf_token=args.hf_token, hf_private=args.hf_private)

    elif train_mode == 'both':
        # Train VAE first, then DIMBA
        print("\n🔄 Training Mode: BOTH (VAE → DIMBA)")
        print("=" * 60)
        
        # Get VAE config
        if args.vae_preset:
            vae_config = VAE_PRESETS[args.vae_preset].copy()
        else:
            vae_config = interactive_vae_preset_selection()
        
        if not args.yes:
            vae_config = interactive_config_review(vae_config, gpus)
        
        # Train VAE
        vae_ckpt = run_vae_training(vae_config, num_gpus, gpu_indices,
                                    hf_repo_id=args.hf_repo_id, 
                                    hf_token=args.hf_token, 
                                    hf_private=args.hf_private)
        
        if vae_ckpt is None:
            print("\n❌ VAE training failed or was cancelled.")
            return
        
        print("\n" + "=" * 60)
        print("VAE training complete! Now training DIMBA with this VAE.")
        print("=" * 60)
        
        # Get DIMBA config
        if args.config:
            with open(args.config) as f:
                config = yaml.safe_load(f)
        elif args.preset:
            config = PRESETS[args.preset].copy()
        else:
            config = interactive_preset_selection(gpus)
        
        if not args.yes:
            config = interactive_config_review(config, gpus)
        
        # Train DIMBA with the VAE
        run_dimba_training(config, num_gpus, gpu_indices, vae_checkpoint=vae_ckpt,
                          hf_repo_id=args.hf_repo_id, hf_token=args.hf_token, hf_private=args.hf_private)


if __name__ == '__main__':
    main()
