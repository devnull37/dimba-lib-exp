#!/usr/bin/env python3
"""Interactive setup script for DIMBA training configuration.

Provides a menu-driven interface for:
- Setting up progressive checkpointing milestones
- Creating and modifying config files
- Managing training configurations
"""

import os
import sys
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add src to path for imports
script_dir = Path(__file__).parent
src_dir = script_dir.parent / "src"
sys.path.insert(0, str(src_dir))

try:
    from dimba.utils.checkpointing import parse_milestone_input
except ImportError:
    # Fallback if dimba not installed
    def parse_milestone_input(input_str: str) -> List[int]:
        """Parse milestone input string into parameter counts."""
        milestones = []
        for part in input_str.split(","):
            part = part.strip().upper()
            if part.endswith("B"):
                value = float(part[:-1]) * 1e9
            elif part.endswith("M"):
                value = float(part[:-1]) * 1e6
            elif part.endswith("K"):
                value = float(part[:-1]) * 1e3
            else:
                value = float(part)
                if value < 1000:
                    value *= 1e9
            milestones.append(int(value))
        return sorted(milestones)


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_menu(options: Dict[str, str]):
    """Print a menu with numbered options."""
    for key, value in options.items():
        print(f"  [{key}] {value}")


def get_input(prompt: str, default: Optional[str] = None) -> str:
    """Get user input with optional default value."""
    if default:
        prompt = f"{prompt} [{default}]: "
    else:
        prompt = f"{prompt}: "
    
    response = input(prompt).strip()
    return response if response else (default or "")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        return {}
    except yaml.YAMLError as e:
        print(f"Error parsing YAML: {e}")
        return {}


def save_config(config: Dict[str, Any], config_path: str):
    """Save configuration to YAML file."""
    try:
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"\n✓ Configuration saved to: {config_path}")
    except Exception as e:
        print(f"\n✗ Error saving config: {e}")


def format_param_count(count: int) -> str:
    """Format parameter count in human-readable format."""
    if count >= 1e9:
        return f"{count / 1e9:.1f}B"
    elif count >= 1e6:
        return f"{count / 1e6:.1f}M"
    else:
        return f"{count / 1e3:.1f}K"


def setup_progressive_checkpoints():
    """Interactive setup for progressive checkpointing."""
    print_header("Progressive Checkpointing Setup")
    
    print("\nProgressive checkpointing saves models as they grow through")
    print("parameter milestones (e.g., 1B, 5B, 10B, 30B parameters).\n")
    
    # Ask whether to load existing config or create new
    print("Options:")
    print_menu({
        "1": "Continue from existing config (load config.yaml)",
        "2": "Create new config from scratch",
    })
    
    choice = get_input("\nSelect option", "1")
    
    config = {}
    config_path = "config.yaml"
    
    if choice == "1":
        config_path = get_input("Enter config file path", "config.yaml")
        config = load_config(config_path)
        if config:
            print(f"\n✓ Loaded existing config from: {config_path}")
            # Show current progressive checkpoint settings if they exist
            if 'progressive_checkpoints' in config:
                pc = config['progressive_checkpoints']
                print(f"\nCurrent settings:")
                print(f"  Enabled: {pc.get('enabled', False)}")
                milestones = pc.get('milestones', [])
                print(f"  Milestones: {[format_param_count(m) for m in milestones]}")
                print(f"  Save dir: {pc.get('save_dir', './progressive_checkpoints')}")
        else:
            print("\n⚠ Could not load config, creating new one")
            config = get_default_config()
    else:
        config = get_default_config()
        config_path = get_input("Enter config file path to create", "config.yaml")
    
    # Progressive checkpointing settings
    print("\n" + "-" * 60)
    print("Configure Progressive Checkpointing")
    print("-" * 60)
    
    # Enable/disable
    enable_str = get_input("Enable progressive checkpointing? (yes/no)", "yes")
    enabled = enable_str.lower() in ['yes', 'y', 'true', '1']
    
    # Milestones
    print("\nEnter milestone sizes (comma-separated in billions)")
    print("Examples: '1,5,10,30' for 1B, 5B, 10B, 30B")
    print("          '100M,500M,1B,3B' for smaller models")
    
    default_milestones = "1,5,10,30"
    if 'progressive_checkpoints' in config and config['progressive_checkpoints'].get('milestones'):
        existing = config['progressive_checkpoints']['milestones']
        default_milestones = ",".join([str(int(m / 1e9)) for m in existing if m >= 1e9])
    
    milestone_input = get_input("Milestone sizes (in billions)", default_milestones)
    
    try:
        milestones = parse_milestone_input(milestone_input)
        print(f"\n✓ Parsed milestones: {[format_param_count(m) for m in milestones]}")
    except Exception as e:
        print(f"\n✗ Error parsing milestones: {e}")
        print("  Using default: 1B, 5B, 10B, 30B")
        milestones = [1_000_000_000, 5_000_000_000, 10_000_000_000, 30_000_000_000]
    
    # Save directory
    default_dir = config.get('progressive_checkpoints', {}).get('save_dir', './progressive_checkpoints')
    save_dir = get_input("Save directory for checkpoints", default_dir)
    
    # Update config
    config['progressive_checkpoints'] = {
        'enabled': enabled,
        'milestones': milestones,
        'save_dir': save_dir,
    }
    
    # Display summary
    print("\n" + "=" * 60)
    print("Configuration Summary")
    print("=" * 60)
    print(f"  Enabled: {enabled}")
    print(f"  Milestones: {[format_param_count(m) for m in milestones]}")
    print(f"  Save directory: {save_dir}")
    
    # Save
    confirm = get_input("\nSave this configuration? (yes/no)", "yes")
    if confirm.lower() in ['yes', 'y']:
        save_config(config, config_path)
        
        # Also create a sample training command
        print("\n" + "=" * 60)
        print("Sample Training Command")
        print("=" * 60)
        print(f"\npython scripts/train_cdlm.py --config {config_path}")
        if enabled:
            print("\nOr programmatically:")
            print("""
from dimba.training import DIMBALightningModule

module = DIMBALightningModule(
    vocab_size=32000,
    model_config=model_config,
    enable_progressive_checkpoints=True,
    progressive_milestones=""" + str(milestones) + """,
    progressive_save_dir=""" + f'"{save_dir}"' + """,
)
""")
    else:
        print("\n✗ Configuration not saved")


def get_default_config() -> Dict[str, Any]:
    """Get default configuration template."""
    return {
        'model': {
            'd_model': 256,
            'd_prompt': 128,
            'num_diffusion_steps': 1000,
            'num_denoiser_layers': 6,
            'd_state': 16,
            'd_conv': 4,
            'expand': 2,
            'conditioning_type': 'film',
            'dropout': 0.1,
            'use_weight_tying': False,
            'use_simple_mamba': False,
            'latent_diffusion': False,
            'd_latent': None,
            'latent_projector_depth': 2,
            'latent_loss_weight': 1.0,
            'recon_loss_weight': 1.0,
        },
        'tokenizer': {
            'type': 'bpe',
            'vocab_size': 10000,
            'bpe_vocab_size': 10000,
        },
        'data': {
            'type': 'huggingface',
            'batch_size': 32,
            'max_length': 256,
            'num_workers': 0,
            'num_examples': 1000,
            'dataset_name': 'wikitext',
            'dataset_config': 'wikitext-2-raw-v1',
        },
        'training': {
            'learning_rate': 2e-5,
            'warmup_steps': 500,
            'weight_decay': 0.01,
            'ema_decay': 0.9999,
            'use_ema': True,
            'gradient_clip': 1.0,
            'num_epochs': 10,
            'max_steps': None,
            'log_interval': 100,
            'val_interval': 500,
            'use_consistency_training': False,
            'consistency_loss_weight': 0.5,
            'consistency_delta_min': 50,
            'consistency_delta_max': 200,
        },
        'device': {
            'use_gpu': True,
            'multi_gpu': False,
            'mixed_precision': False,
            'benchmark': True,
        },
        'inference': {
            'num_steps': 50,
            'temperature': 1.0,
            'top_k': None,
            'top_p': 0.95,
            'use_ddim': False,
            'ddim_eta': 0.0,
        },
        'checkpoint': {
            'save_dir': './checkpoints',
            'save_interval': 500,
            'keep_last_k': 3,
        },
        'progressive_checkpoints': {
            'enabled': False,
            'milestones': [1_000_000_000, 5_000_000_000, 10_000_000_000, 30_000_000_000],
            'save_dir': './progressive_checkpoints',
        },
    }


def view_progressive_checkpoints():
    """View existing progressive checkpoints."""
    print_header("View Progressive Checkpoints")
    
    save_dir = get_input("Enter checkpoint directory", "./progressive_checkpoints")
    
    if not os.path.exists(save_dir):
        print(f"\n✗ Directory not found: {save_dir}")
        return
    
    checkpoint_files = sorted(Path(save_dir).glob("checkpoint_*.json"))
    
    if not checkpoint_files:
        print(f"\n⚠ No checkpoint metadata found in: {save_dir}")
        return
    
    print(f"\nFound {len(checkpoint_files)} checkpoint(s):\n")
    print(f"{'Milestone':<15} {'Step':<10} {'Parameters':<15} {'Filename'}")
    print("-" * 70)
    
    for meta_file in checkpoint_files:
        try:
            import json
            with open(meta_file) as f:
                meta = json.load(f)
            milestone_str = meta.get('milestone_str', 'Unknown')
            step = meta.get('global_step', 0)
            params = meta.get('param_count_formatted', 'Unknown')
            filename = meta.get('filename', meta_file.stem + '.pt')
            print(f"{milestone_str:<15} {step:<10} {params:<15} {filename}")
        except Exception as e:
            print(f"{meta_file.name:<30} Error reading: {e}")
    
    print("\n" + "-" * 70)
    print("\nTo resume from a checkpoint:")
    print("  1. Load the checkpoint file")
    print("  2. Continue training - milestones will be tracked from the loaded state")


def main():
    """Main menu loop."""
    while True:
        print_header("DIMBA Training Setup")
        
        print_menu({
            "1": "Setup Progressive Checkpointing",
            "2": "View Existing Progressive Checkpoints",
            "3": "Exit",
        })
        
        choice = get_input("\nSelect option")
        
        if choice == "1":
            setup_progressive_checkpoints()
        elif choice == "2":
            view_progressive_checkpoints()
        elif choice == "3":
            print("\nGoodbye!")
            break
        else:
            print("\n✗ Invalid option. Please try again.")
        
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()
