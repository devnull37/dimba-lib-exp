#!/usr/bin/env python3
"""Training scripts for DIMBA.

This directory contains standalone scripts for training, evaluating, and
using DIMBA models.

## Structure

### Interactive Training
- `train_interactive.py` - **RECOMMENDED** Interactive wizard with auto GPU detection, 
  config presets, memory estimation, resume support, and HuggingFace auto-upload

### Main Training Scripts
- `train.py` - Generic training script with config file support
- `train_vae.py` - Pre-train the TokenVAE for latent diffusion
- `train_fineweb_1b.py` - Train 1.5B model on FineWeb (L40S 48GB profile)
- `train_fineweb_500m_a4000.py` - Train 500M model on FineWeb (A4000 16GB profile)

### Inference & Evaluation
- `generate.py` - Generate text from a trained checkpoint
- `evaluate.py` - Evaluate model perplexity and inference speed
- `upload_to_hf.py` - Upload checkpoints to HuggingFace Hub

### Utilities (`utils/`)
- `calculate_memory.py` - Calculate memory usage for different configs
- `test_config.py` - Test dependencies, GPU, and dataset access
- `test_dataset_loading.py` - Test FineWeb dataset loading

## Quick Start

### Interactive training (easiest)
```bash
python scripts/train_interactive.py
```

This launches a wizard that:
- Auto-detects your GPU(s)
- Suggests optimal presets based on VRAM
- Estimates memory usage before training
- Lets you customize everything interactively

### Auto-upload to HuggingFace

After training completes, you can automatically upload to HuggingFace:

```bash
# Interactive mode - will prompt for upload
python scripts/train_interactive.py

# Auto-upload with repo specified
python scripts/train_interactive.py --preset a4000-500m --hf-repo-id username/dimba-500m

# Auto-upload with token (or set HF_TOKEN env var)
python scripts/train_interactive.py --preset a4000-500m \
    --hf-repo-id username/dimba-500m \
    --hf-token $HF_TOKEN \
    --hf-private  # for private repo
```

### Resume training

If training is interrupted, you can easily resume:

```bash
# Resume from latest checkpoint (interactive)
python scripts/train_interactive.py --resume

# Resume from specific checkpoint directory
python scripts/train_interactive.py --resume-from ./checkpoints/interactive

# Resume with new config (keep weights, change hyperparameters)
python scripts/train_interactive.py --resume --new-config

# Non-interactive resume
python scripts/train_interactive.py --resume --yes
```

When resuming:
- The saved `train_config.yaml` is automatically loaded
- Tokenizer is loaded from the checkpoint folder
- Training continues from the last step
- All checkpoints are saved to the same folder

### Train from config
```bash
python scripts/train.py --config config.yaml --max-epochs 10
```

### Train VAE for latent diffusion
```bash
python scripts/train_vae.py --dataset wikitext --latent-dim 256 --epochs 10
```

### Generate text
```bash
python scripts/generate.py \
    --checkpoint checkpoints/dimba.ckpt \
    --config config.yaml \
    --prompt "The future of AI is" \
    --length 100 \
    --num-steps 50
```

### Evaluate model
```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/dimba.ckpt \
    --vocab-size 32000 \
    --eval-speed
```

## Common Arguments

Most scripts support these common arguments:
- `--config` - Path to YAML config file
- `--checkpoint` - Path to model checkpoint
- `--device` - Device to use (cuda/cpu)
- `--vocab-size` - Vocabulary size
- `--batch-size` - Batch size for training/inference

See individual script help for full options:
```bash
python scripts/<script>.py --help
```

## Presets (train_interactive.py)

Available configuration presets:

| Preset | GPU | Params | VRAM |
|--------|-----|--------|------|
| `cpu-small` | CPU | <100M | N/A |
| `a4000-500m` | RTX A4000 16GB | ~500M | ~12GB |
| `l40s-1b` | L40S 48GB | ~1.5B | ~35GB |
| `a100-3b` | A100 80GB | ~3B | ~65GB |

Use a preset directly:
```bash
python scripts/train_interactive.py --preset a4000-500m
```

## Interactive Training Output

Interactive mode saves everything to `checkpoints/interactive/`:
- `train_config.yaml` - The configuration used
- `tokenizer.json` - Tokenizer state
- `dimba-*.ckpt` - Model checkpoints
- `last.ckpt` - Most recent checkpoint (for resuming)

This folder is automatically gitignored to avoid committing large files.

To use a different checkpoint directory:
```bash
# The script will ask, or specify in your config
```

## HuggingFace Integration

### Auto-upload after training

The interactive script can automatically upload to HuggingFace Hub:

```bash
# Specify repo for auto-upload
python scripts/train_interactive.py --preset a4000-500m --hf-repo-id username/dimba-500m

# With private repo
python scripts/train_interactive.py --preset a4000-500m \
    --hf-repo-id username/dimba-500m \
    --hf-private

# Non-interactive with auto-upload
python scripts/train_interactive.py --preset l40s-1b \
    --yes \
    --hf-repo-id username/dimba-1b \
    --hf-token $HF_TOKEN
```

**Upload includes:**
- All checkpoints (`*.ckpt`)
- `train_config.yaml` - training configuration
- `tokenizer.json` - tokenizer state

**Requirements:**
```bash
pip install huggingface_hub
```

**Environment variable:**
```bash
export HF_TOKEN=your_token_here
# Then you can skip --hf-token
python scripts/train_interactive.py --hf-repo-id username/model
```

"""
