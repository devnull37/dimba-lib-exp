# Quick Start: GPU Training

## 30-Second Setup

### Google Colab (Easiest)
1. Upload `dimba-lib-exp/` folder to Google Drive
2. Go to https://colab.research.google.com
3. Open `notebooks/train_colab.ipynb`
4. Run all cells ▶️
5. Done! Checkpoints saved to Drive

### Local GPU (Linux)
```bash
pip install -e ".[gpu]"              # Install GPU deps
python scripts/train.py               # Start training
```

### Kaggle
1. Create dataset with `dimba-lib-exp/` folder
2. Create notebook
3. Copy `notebooks/train_colab.ipynb` code
4. Adjust paths for Kaggle
5. Run

---

## Config Quick Changes

### To use GPU:
```yaml
device:
  use_gpu: true              # ✅ Enabled (auto-detects)
```

### To use CPU only:
```yaml
model:
  use_simple_mamba: true     # Force CPU Mamba
device:
  use_gpu: false             # Disable GPU
```

### To use better tokenizer:
```yaml
tokenizer:
  type: "bpe"                # Better than "simple"
  vocab_size: 10000
```

### To train faster (GPU):
```yaml
data:
  batch_size: 64             # Increase batch size
training:
  learning_rate: 2e-4        # Increase LR slightly
```

---

## Tokenizer Usage

```python
# Simple (CPU, character-level)
from dimba.tokenizers import SimpleCharacterTokenizer
tok = SimpleCharacterTokenizer(vocab_size=256)

# BPE (GPU, subword-level)
from dimba.tokenizers import BPETokenizer
tok = BPETokenizer(vocab_size=10000)

# Encode
tokens = tok.encode("hello world")  # [104, 101, 108, 108, 111, ...]

# Decode
text = tok.decode(tokens)           # "hello world"

# Save/Load
tok.save("tokenizer.json")
tok.load("tokenizer.json")
```

---

## Training Commands

```bash
# Colab: Just run the notebook

# Local (GPU):
python scripts/train.py --config config.yaml

# Local (CPU only):
# Edit config.yaml first, set use_gpu: false
python scripts/train.py --config config.yaml

# Generate from checkpoint:
python scripts/generate.py \
  --checkpoint path/to/checkpoint.ckpt \
  --prompt "hello world" \
  --num-steps 50
```

---

## Check GPU Setup

```python
import torch
print(torch.cuda.is_available())        # Should be True
print(torch.cuda.get_device_name(0))    # Should show GPU name
print(torch.cuda.memory_allocated())    # GPU memory used

# Check mamba-ssm
try:
    from mamba_ssm import Mamba
    print("✓ mamba-ssm available (GPU optimized)")
except ImportError:
    print("✗ mamba-ssm not installed (will use CPU version)")
```

---

## File Locations

| File | Purpose | GPU? |
|------|---------|------|
| `src/dimba/tokenizers/simple.py` | Character tokenizer | ❌ CPU |
| `src/dimba/tokenizers/bpe.py` | Subword tokenizer | ✅ GPU |
| `src/dimba/models/denoiser.py` | Auto-selects Mamba | Both |
| `notebooks/train_colab.ipynb` | Full training notebook | ✅ GPU |
| `config.yaml` | Training config | Both |

---

## Performance

| Setup | Speed | Memory |
|-------|-------|--------|
| SimpleMamba2 (CPU) | ~1 token/sec | Low |
| mamba-ssm (T4 GPU) | ~20 token/sec | 8GB |
| mamba-ssm (A100 GPU) | ~50+ token/sec | 24GB |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| CUDA not found | Install PyTorch with CUDA support |
| OOM error | Reduce `batch_size` in config |
| mamba-ssm not found | `pip install mamba-ssm causal-conv1d` |
| Slow on GPU | Check `torch.cuda.is_available()` |
| Colab timeout | Run notebook in sections, save checkpoints |

---

## Key Features

✅ **Backward Compatible** - CPU training still works
✅ **GPU Optimized** - 10-25x faster with mamba-ssm
✅ **Better Tokenizer** - BPE for real text data
✅ **Easy Setup** - One Colab notebook has everything
✅ **Flexible Config** - Control all aspects in YAML

---

## Next Steps

1. **Run Colab notebook** or **local training**
2. **Monitor loss** - Should decrease over time
3. **Save checkpoints** - Auto-saved during training
4. **Generate text** - Use `generate.py` with checkpoint
5. **Iterate** - Tweak config and retrain

---

**Questions?** See `GPU_TRAINING_GUIDE.md` for details.
