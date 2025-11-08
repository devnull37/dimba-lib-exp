# DIMBA: Diffusion-based Mamba for Non-Autoregressive Text Generation

A complete PyTorch implementation of **DIMBA** (Diffusion + Mamba-based Architecture) from the paper "DIMBA: Revolutionizing Theoretical Ultra-Fast Inference and Advanced Reasoning with Mamba-Based Diffusion".

DIMBA is a non-autoregressive language model that fuses a cosine-scheduled diffusion process with Mamba-2 state-space models to generate entire token sequences in parallel. This offers potential for faster inference compared to autoregressive models while maintaining semantic coherence.

## Features

- ✅ **Complete DIMBA architecture**: Token embeddings, prompt encoder, Mamba-2 denoiser with FiLM/additive conditioning
- ✅ **Cosine noise schedule**: With zero terminal SNR fix for stable training/inference
- ✅ **PyTorch Lightning training**: Easy multi-GPU training with EMA support
- ✅ **Multiple sampling strategies**: Standard denoising + DDIM-style acceleration
- ✅ **Evaluation metrics**: Perplexity, BLEU, ROUGE, METEOR
- ✅ **Flexible data pipeline**: Support for dummy data, HuggingFace datasets, and custom text
- ✅ **Configuration-driven**: YAML-based hyperparameter management

## Installation

### Requirements
- Python 3.9+
- CUDA 11.6+ (for GPU acceleration, optional)

### Quick Install

```bash
# Clone repository
git clone https://github.com/devnull37/dimba-lib-exp.git
cd dimba-lib-exp

# Install in development mode
pip install -e .

# For GPU support (Linux with CUDA)
pip install -e ".[gpu]"

# For evaluation metrics
pip install -e ".[eval]"

# For experiment tracking
pip install -e ".[tracking]"

# For development (includes testing + linting tools)
pip install -e ".[dev]"
```

## Quick Start

### 1. Basic Training

```bash
# Train with default config (uses dummy dataset on CPU)
python scripts/train.py --config config.yaml

# Train with GPU (if available)
python scripts/train.py --config config.yaml --gpus 1 --max-epochs 10

# Train with custom settings
python scripts/train.py \
    --config config.yaml \
    --vocab-size 10000 \
    --max-epochs 5 \
    --gpus 2 \
    --mixed-precision 16-mixed
```

### 2. Model Training Example (Python)

```python
import torch
from torch.utils.data import DataLoader
from dimba import DIMBA
from dimba.data import DummyDataset, collate_fn
from dimba.training import DIMBALightningModule
import pytorch_lightning as pl

# Create model
model = DIMBA(
    vocab_size=50000,
    d_model=512,
    num_diffusion_steps=1000,
    num_denoiser_layers=6,
)

# Create dataset
dataset = DummyDataset(size=1000, vocab_size=50000, seq_length=256)
dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)

# Create Lightning module
lightning_module = DIMBALightningModule(
    vocab_size=50000,
    model_config={'d_model': 512, 'num_denoiser_layers': 6},
)

# Train
trainer = pl.Trainer(max_epochs=10, accelerator='auto', devices='auto')
trainer.fit(lightning_module, dataloader)
```

### 3. Generation

```python
from dimba import DIMBA, sample_from_model
import torch

# Load trained model
model = DIMBA(vocab_size=50000)
model.load_state_dict(torch.load('checkpoint.pt'))

# Generate text
prompt_ids = torch.tensor([[10, 20, 30]])  # Example token IDs
generated = sample_from_model(
    model,
    prompt_ids,
    seq_len=100,
    num_steps=50,          # Use 50 denoising steps
    temperature=1.0,
    top_p=0.95,
)
print(f"Generated: {generated.shape}")
```

## Configuration

Edit `config.yaml` to customize model and training parameters:

```yaml
model:
  d_model: 512              # Hidden dimension
  num_diffusion_steps: 1000 # Diffusion steps T
  num_denoiser_layers: 6    # Mamba-2 blocks
  conditioning_type: "film" # FiLM or additive conditioning

training:
  learning_rate: 2e-5
  warmup_steps: 500
  ema_decay: 0.9999         # EMA for smoother samples

data:
  type: "dummy"             # dummy, huggingface, or text
  batch_size: 32
  max_length: 256
```

## Architecture Overview

### Core Components

1. **Token Embedding**: Maps discrete tokens to continuous embeddings
2. **Prompt Encoder**: Encodes prompt context as conditioning information
3. **Diffusion Schedule**: Cosine-scheduled noise injection (per Nichol & Dhariwal 2021)
4. **Timestep Embedding**: Sinusoidal embeddings to condition denoiser on noise level
5. **Mamba-2 Denoiser**: SSM-based denoiser with FiLM/additive conditioning
6. **Output Head**: Projects denoised embeddings back to token logits

### Training Procedure

```
For each batch:
  1. Sample random timestep t
  2. Add noise to clean embeddings: x_t = √ᾱ(t) * x_0 + √(1 - ᾱ(t)) * ε
  3. Encode prompt to conditioning vector
  4. Predict clean embeddings using Mamba-2 denoiser
  5. Compute MSE loss: L = ||x_pred - x_0||²
  6. Update model + EMA model
```

### Inference Procedure

```
1. Encode prompt to conditioning
2. Initialize with random noise
3. Iteratively denoise from t=T down to t=0:
   - For each timestep, predict cleaner version
   - Add controlled noise for next step
4. Project final embeddings to token logits
5. Sample tokens or take argmax
```

## Evaluation

### Metrics

Compute various evaluation metrics:

```python
from dimba.evaluation import compute_perplexity, evaluate_generation

# Perplexity
ppl = compute_perplexity(logits, targets)

# Multiple metrics
results = evaluate_generation(
    predictions=generated_texts,
    references=reference_texts,
    compute_bleu=True,
    compute_rouge=True,
    compute_meteor=False,  # Requires METEOR
)
```

### Evaluation Script

```bash
# Evaluate checkpoint
python scripts/evaluate.py \
    --checkpoint checkpoints/dimba-best.pt \
    --vocab-size 50000 \
    --eval-speed  # Measure inference speed at different step counts
```

## Performance Characteristics

### Speed/Quality Trade-off

DIMBA allows tuning inference speed via the number of diffusion steps `T`:

- **T=10-25**: Fastest, may have lower quality
- **T=50-100**: Balanced (recommended)
- **T=200-500**: Higher quality, slower
- **T=1000**: Highest quality, slowest (same as training)

### Supported Architectures

- **Model dimension**: 256-1024 (tested)
- **Sequence length**: 64-1024 tokens
- **Number of layers**: 4-12 Mamba-2 blocks
- **State size**: 8-32 SSM state size

## Known Limitations (From Paper)

1. **Training cost**: Diffusion training is computationally expensive
2. **Discrete-continuous gap**: Token embedding mapping can affect rare words
3. **Hyperparameter sensitivity**: Performance varies with step count T and other parameters
4. **Conditioning robustness**: Need empirical validation across diverse prompts

## Project Structure

```
dimba-lib-exp/
├── src/dimba/                    # Main package
│   ├── models/                   # Model components
│   │   ├── diffusion.py          # DIMBA wrapper
│   │   ├── denoiser.py           # Mamba-2 denoiser
│   │   └── embeddings.py         # Embedding layers
│   ├── diffusion/                # Diffusion process
│   │   ├── schedules.py          # Noise schedules
│   │   └── sampling.py           # Inference sampling
│   ├── data/                     # Data pipeline
│   │   └── dataset.py            # Dataset classes
│   ├── training/                 # Training loop
│   │   └── trainer.py            # PyTorch Lightning module
│   └── evaluation/               # Metrics
│       └── metrics.py            # Evaluation functions
├── scripts/
│   ├── train.py                  # Training entry point
│   ├── generate.py               # Generation script
│   └── evaluate.py               # Evaluation script
├── tests/                        # Unit tests
├── notebooks/                    # Jupyter notebooks
├── config.yaml                   # Configuration template
├── pyproject.toml               # Project metadata
└── README.md                     # This file
```

## Development

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v --cov=src/dimba

# Run linting
black src/ scripts/
isort src/ scripts/
flake8 src/ scripts/
```

### Code Style

- **Formatter**: Black (line length: 100)
- **Import sorting**: isort
- **Linter**: flake8

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{allafi2025dimba,
  title={DIMBA: Revolutionizing Theoretical Ultra-Fast Inference and Advanced Reasoning with Mamba-Based Diffusion},
  author={Allafi, Faris},
  year={2025}
}
```

## Acknowledgments

- **Mamba-2**: https://github.com/state-spaces/mamba
- **Diffusion Models**: Nichol & Dhariwal (2021) "Improved Denoising Diffusion Probabilistic Models"
- **PyTorch Lightning**: Falcon et al. (2019)

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome! This is a research implementation, so all suggestions for improvements are appreciated.

For questions or issues:
- Open an issue on GitHub
- Check the paper for architectural details (see `paper/main.txt`)
- Review `CLAUDE.md` for development guidelines 