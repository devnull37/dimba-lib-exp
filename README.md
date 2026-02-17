# DIMBA: Diffusion-based Mamba for Non-Autoregressive Text Generation

A PyTorch implementation of **DIMBA** (Diffusion + Mamba-based Architecture) - a non-autoregressive language model combining cosine-scheduled diffusion with Mamba-2 state-space models for parallel text generation.

## Overview

DIMBA generates entire token sequences in parallel using iterative denoising, enabling controllable speed-quality trade-offs by adjusting diffusion steps T. It leverages Mamba-2's efficient long-range dependency modeling for linear-time sequence processing.

## Installation

### Requirements
- Python 3.9+
- CUDA 11.6+ (optional, for GPU acceleration)

### Install

```bash
git clone https://github.com/devnull37/dimba-lib-exp.git
cd dimba-lib-exp

# Basic installation (CPU + SimpleMamba fallback)
pip install -e .

# With GPU support (full Mamba-2)
pip install -e ".[gpu]"

# Full development setup
pip install -e ".[all]"
```

## Quick Start

### Training

```bash
# Train on GPU
python scripts/train.py --config config.yaml --gpus 1 --max-epochs 10

# Train on CPU (uses SimpleMamba)
python scripts/train.py --config config.yaml
```

### Generation

```bash
python scripts/generate.py --checkpoint checkpoints/best.pt --prompt "Hello world"
```

### Evaluation

```bash
python scripts/evaluate.py --checkpoint checkpoints/best.pt --eval-speed
```

### Python API

```python
import torch
from dimba import DIMBA, sample_from_model

# Create model
model = DIMBA(vocab_size=50000, d_model=512, num_diffusion_steps=1000)

# Generate text
prompt_ids = torch.tensor([[10, 20, 30]])
generated = sample_from_model(model, prompt_ids, seq_len=100, num_steps=50)
```

## VAE Pre-training for Latent Diffusion

DIMBA supports Variational Autoencoder (VAE) based latent diffusion, which compresses token embeddings into a probabilistic latent space. This can improve diffusion efficiency and model capacity.

### VAE Architecture

The `TokenVAE` class implements:
- **Encoder**: Maps token embeddings to latent distribution (μ, log σ²)
- **Reparameterization**: Stochastic sampling z = μ + σ·ε where ε ~ N(0,I)
- **Decoder**: Maps latent z back to embedding space
- **ELBO Loss**: Reconstruction loss + β·KL divergence

### Pre-training the VAE

Pre-train the VAE before full diffusion training:

```bash
# Basic VAE training
python scripts/train_vae.py \
    --dataset wikitext \
    --dataset-config wikitext-2-raw-v1 \
    --latent-dim 256 \
    --kl-weight 1.0 \
    --learning-rate 1e-4 \
    --batch-size 64 \
    --epochs 10

# With PyTorch Lightning (multi-GPU support)
python scripts/train_vae.py \
    --use-lightning \
    --dataset wikitext \
    --gpus 1 \
    --latent-dim 256 \
    --kl-weight 0.1 \
    --max-steps 100000

# Resume from checkpoint
python scripts/train_vae.py \
    --resume-from checkpoints/vae/last.ckpt \
    --epochs 20
```

### Using VAE in DIMBA

After pre-training, use the VAE checkpoint for latent diffusion:

```python
from dimba import DIMBA

model = DIMBA(
    vocab_size=50000,
    d_model=512,
    latent_diffusion=True,
    d_latent=256,
    use_vae_latent=True,           # Use VAE instead of deterministic projector
    vae_kl_weight=1.0,             # KL weight for VAE
    vae_checkpoint_path='checkpoints/vae/final.ckpt',  # Load pre-trained VAE
)
```

### VAE Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--latent-dim` | Dimension of latent space | 256 |
| `--kl-weight` | Weight for KL divergence (β-VAE) | 1.0 |
| `--hidden-dim` | Hidden layer dimension | max(d_model, latent_dim) |
| `--num-layers` | Number of encoder/decoder layers | 2 |
| `--dropout` | Dropout rate | 0.1 |

### Key Features

- **Deterministic inference**: Use μ for encoding during diffusion, sample z only during training
- **Pre-training**: Train VAE independently to learn a good latent space
- **Checkpoint loading**: Load pre-trained VAE weights into DIMBA
- **KL regularization**: β-VAE formulation allows controlling latent space structure

## Architecture

### Core Components

1. **Token Embeddings**: Learnable embedding matrix mapping tokens to continuous space
2. **Prompt Encoder**: Lightweight MLP encoding prompt to conditioning vectors
3. **Cosine Noise Schedule**: Following Nichol & Dhariwal (2021) with formula `ᾱ(t) = cos²((t/T + s)/(1+s)·π/2)`
4. **Timestep Embeddings**: Sinusoidal encodings with MLP to condition on noise level
5. **Mamba-2 Denoiser**: Stack of Mamba-2 SSM blocks with FiLM/additive conditioning
6. **Output Projection**: Linear layer (optionally weight-tied) projecting to token logits

### Training Procedure

```
For each batch:
  1. Sample random timestep t ~ Uniform(1, T)
  2. Add noise: x_t = √ᾱ(t)·x₀ + √(1-ᾱ(t))·ε  where ε ~ N(0,I)
  3. Encode prompt: C = PromptEncoder(x₀)
  4. Get timestep embedding: τ = MLP(t)
  5. Predict: x_pred = Denoiser(x_t, C, τ)
  6. Compute loss: L = ||x_pred - x₀||²
  7. Update parameters with AdamW + warmup
```

### Inference Procedure

```
1. Compute prompt conditioning: C = PromptEncoder(prompt_ids)
2. Initialize with noise: x_T ~ N(0, I)
3. Iterative denoising (t = T down to 1):
   - τ = MLP(t)
   - x_{t-1} = Denoiser(x_t, C, τ)
4. Project to logits: logits = Linear(x_0)
5. Sample tokens with top-k/top-p
```

## Configuration

Edit `config.yaml`:

```yaml
model:
  d_model: 512
  num_diffusion_steps: 1000  # Controls speed/quality: T=50 for fast, T=1000 for best
  num_denoiser_layers: 6
  conditioning_type: "film"  # or "additive"
  use_simple_mamba: false    # Set true for CPU or if mamba-ssm not installed

training:
  learning_rate: 2e-5
  warmup_steps: 500
  ema_decay: 0.9999
  num_epochs: 10

data:
  type: "huggingface"  # "dummy", "huggingface", or "text"
  dataset_name: "wikitext"
  batch_size: 32
  max_length: 256
```

## GPU Training

### Setup on Cloud GPU (e.g., TensorDock)

```bash
# On your local machine
git push origin main

# On cloud GPU
git clone https://github.com/your-username/dimba-lib-exp.git
cd dimba-lib-exp
pip install -e ".[gpu]"
python scripts/train.py --config config.yaml --gpus 1
```

### Verify GPU Works

```python
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.get_device_name(0))  # GPU name
```

### Performance Tips

- Increase batch size on GPU (32 → 64 → 128)
- Use larger learning rates with GPU
- Enable mixed precision with `--precision 16-mixed`
- Use T=50 for fast inference, T=1000 for best quality

## Project Structure

```
dimba-lib-exp/
├── src/dimba/
│   ├── models/
│   │   ├── diffusion.py       # DIMBA model wrapper
│   │   ├── denoiser.py        # Mamba-2 denoiser
│   │   ├── embeddings.py      # Token, timestep, prompt embeddings
│   │   └── simple_mamba.py    # CPU fallback implementation
│   ├── diffusion/
│   │   ├── schedules.py       # Cosine noise schedule
│   │   └── sampling.py        # Inference and sampling
│   ├── data/
│   │   └── dataset.py         # Dataset loaders
│   ├── training/
│   │   └── trainer.py         # PyTorch Lightning training
│   └── evaluation/
│       └── metrics.py         # BLEU, ROUGE, METEOR, perplexity
├── scripts/
│   ├── train.py
│   ├── generate.py
│   └── evaluate.py
├── tests/
├── config.yaml
├── README.md
└── AGENTS.md
```

## Known Limitations

1. **Training cost**: Diffusion requires substantial compute
2. **Discrete-continuous gap**: Embedding mapping affects rare tokens
3. **Hyperparameter sensitivity**: Performance varies with T, architecture
4. **Conditioning robustness**: Needs empirical validation across prompts

## Next Steps

1. **Train on real data**: Swap DummyDataset for HuggingFace datasets
2. **Optimize hyperparameters**: Tune d_model, num_layers, T for your task
3. **Benchmark**: Compare against autoregressive baselines
4. **Evaluate**: Test on diverse prompts for conditioning robustness

## Citation

```bibtex
@article{allafi2025dimba,
  title={DIMBA: Revolutionizing Theoretical Ultra-Fast Inference and Advanced Reasoning with Mamba-Based Diffusion},
  author={Allafi, Faris},
  year={2025}
}
```

## License

MIT License - See LICENSE file for details.
## RTX A4000 (16GB) recipe: FineWeb + FiLM + embedding diffusion (~500M)

This profile matches your requested setup: **single RTX A4000 16GB**, **embedding diffusion**, **FiLM conditioning**, and a **~500M parameter target**.

### 1) Instance setup (A4000 specific)

```bash
bash install_a4000.sh
```

### 2) Train the 500M profile

```bash
python scripts/train_fineweb_500m_a4000.py --config configs/fineweb_500m_a4000.yaml
```

Optional: auto-upload to Hugging Face when training finishes:

```bash
export HF_TOKEN=hf_xxx
python scripts/train_fineweb_500m_a4000.py \
  --config configs/fineweb_500m_a4000.yaml \
  --repo-id your-username/dimba-500m-fineweb-a4000
```

Notes for 16GB VRAM:
- Uses `batch_size=2` with `accumulate_grad_batches=16` (effective batch size 32)
- Uses mixed precision (`16-mixed`) on CUDA
- Uses sequence length 512 to fit A4000 memory more reliably

### 3) Upload to Hugging Face when finished

```bash
export HF_TOKEN=hf_xxx
python scripts/upload_to_hf.py \
  --repo-id your-username/dimba-500m-fineweb-a4000 \
  --artifacts-dir ./checkpoints/fineweb_500m_a4000
```

Optional private repo:

```bash
python scripts/upload_to_hf.py \
  --repo-id your-username/dimba-500m-fineweb-a4000 \
  --artifacts-dir ./checkpoints/fineweb_500m_a4000 \
  --private
```

Expected artifacts in `./checkpoints/fineweb_500m_a4000`:
- `last.ckpt` and top-k validation checkpoints
- `tokenizer.json`
- `train_config.yaml`
