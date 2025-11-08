# GPU Training Guide for DIMBA

This guide explains how to train DIMBA on GPU using Colab, Kaggle, or your local machine.

## Quick Start

### Option 1: Google Colab (Recommended for beginners)

1. **Upload to Google Drive:**
   ```bash
   # From your local machine, upload dimba-lib-exp folder to Google Drive
   ```

2. **Open Colab Notebook:**
   - Go to https://colab.research.google.com
   - Upload `notebooks/train_colab.ipynb`
   - Or create a new notebook and copy the cells

3. **Run the notebook:**
   - The notebook will:
     - Install all dependencies (mamba-ssm, tokenizers, etc.)
     - Mount your Google Drive
     - Create and train the model
     - Save checkpoints back to Drive

### Option 2: Kaggle

1. **Create Kaggle dataset** with your dimba-lib-exp folder
2. **Create new notebook** on Kaggle
3. **Copy code from** `notebooks/train_colab.ipynb`
4. **Modify paths** for Kaggle (`/kaggle/input/...`)

### Option 3: Local GPU Machine (Linux recommended)

```bash
# Install GPU dependencies
pip install mamba-ssm causal-conv1d  # For mamba-ssm optimization

# Or install with GPU extras
pip install -e ".[gpu]"

# Run training
python scripts/train.py --config config.yaml
```

## Configuration for GPU Training

Edit `config.yaml` to optimize for GPU:

```yaml
device:
  use_gpu: true              # Enable GPU
  multi_gpu: false           # Set to true for multi-GPU (requires modification)
  mixed_precision: false     # Set to true for faster training (experimental)
  benchmark: true            # Enable cudnn benchmark

model:
  use_simple_mamba: false    # Use mamba-ssm (GPU optimized)
  d_model: 512               # Can be larger on GPU
  num_denoiser_layers: 6     # Can be more layers on GPU

data:
  type: "huggingface"        # Use real dataset instead of dummy
  dataset_name: "wikitext"   # WikiText, OpenWebText, etc.
  batch_size: 64             # Larger batch sizes on GPU
  num_workers: 4             # Parallel data loading

training:
  learning_rate: 2e-4        # Typical LR for diffusion models
  num_epochs: 10
  warmup_steps: 1000
```

## Tokenizer Options

### 1. Simple Character Tokenizer (Default, good for testing)

```yaml
tokenizer:
  type: "simple"
  vocab_size: 256
```

- CPU-friendly
- Fast training
- Limited vocabulary

### 2. BPE Tokenizer (Recommended for training)

```yaml
tokenizer:
  type: "bpe"
  vocab_size: 10000
```

- Better compression
- Subword-level understanding
- Requires training data
- GPU-friendly

**To train BPE tokenizer:**

```python
from dimba.tokenizers import BPETokenizer

# Your training texts
texts = ["sample text 1", "sample text 2", ...]

# Train tokenizer
tokenizer = BPETokenizer.train(
    texts=texts,
    vocab_size=10000,
)

# Save for later
tokenizer.save("tokenizer.json")
```

## Hardware Requirements

### Minimum for training:
- **GPU RAM:** 8GB (T4 on Colab/Kaggle)
- **CPU RAM:** 16GB
- **Storage:** 20GB (for checkpoints + data)

### Recommended:
- **GPU RAM:** 24GB+ (A100, RTX 3090)
- **CPU RAM:** 32GB+
- **Storage:** 100GB+

## Performance Tips

1. **Increase batch size** on GPU (32 → 64 → 128)
2. **Use mixed precision** for faster training (experimental)
3. **Enable multi-GPU** if available
4. **Use causal-conv1d** optimized library with mamba-ssm
5. **Pin memory** in DataLoader for faster GPU transfer

## Training from Checkpoint

To resume from a checkpoint:

```python
# Load checkpoint
checkpoint = torch.load('path/to/checkpoint.ckpt')
model.load_state_dict(checkpoint, strict=False)

# Continue training
# ... training code ...
```

## Monitoring Training

### TensorBoard (Local)
```bash
tensorboard --logdir=./logs
```

### Weights & Biases (Colab/Kaggle)
```python
import wandb
wandb.init(project="dimba-training")
```

## Troubleshooting

### Out of Memory (OOM)
- Reduce `batch_size` in config.yaml
- Reduce `d_model` or `num_denoiser_layers`
- Set `mixed_precision: true`

### GPU not detected
```python
import torch
print(torch.cuda.is_available())  # Should print True
print(torch.cuda.get_device_name(0))  # Should print GPU name
```

### Slow training
- Increase batch size
- Use `pin_memory: true` in DataLoader
- Enable `benchmark: true` in device config
- Use larger learning rate (with caution)

## Next Steps

1. **Train a model** on your dataset
2. **Save checkpoints** regularly
3. **Evaluate** on validation set
4. **Generate text** using `scripts/generate.py`
5. **Fine-tune** on specific domains

## References

- mamba-ssm: https://github.com/state-spaces/mamba
- HuggingFace Tokenizers: https://github.com/huggingface/tokenizers
- PyTorch Lightning: https://www.pytorchlightning.ai/
- DIMBA Paper: (See paper/ folder)
