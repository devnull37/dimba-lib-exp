# DIMBA 🐍✨

[![PyPI version](https://badge.fury.io/py/dimba-lib.svg)](https://badge.fury.io/py/dimba-lib)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

> **Diffusion-based Mamba Architecture for Non-Autoregressive Text Generation**

DIMBA is a research-grade language model that combines the power of diffusion models with Mamba-2 State Space Models (SSM) to enable **fast, parallel text generation**. Unlike traditional autoregressive models that generate tokens one-by-one, DIMBA generates entire sequences simultaneously through iterative denoising.

🔬 **Research Paper**: *"DIMBA: Revolutionizing Theoretical Ultra-Fast Inference and Advanced Reasoning with Mamba-Based Diffusion"* — Faris Allafi (2025)

🌐 **Website**: [dimbalabs.xyz](https://dimbalabs.xyz)  
👤 **Author**: [farisallafi.xyz](https://farisallafi.xyz)

---

## 🚀 Key Features

### ⚡ Pure PyTorch Mamba-2 Implementation
- **No CUDA dependencies required** — runs on CPU, GPU, and Apple Silicon
- Custom `SimpleMamba2` fallback implementation when `mamba-ssm` is unavailable
- Seamlessly switches between high-performance CUDA kernels and pure PyTorch

### 🎯 Latent Space Diffusion with VAE
- Optional Variational Autoencoder for compressing token embeddings
- Trainable latent spaces with KL-regularization (β-VAE)
- Improves diffusion efficiency and model capacity

### 🍎 Native Apple Silicon (MPS) Support
- First-class Metal Performance Shaders support
- Optimized for M1/M2/M3 Macs without CUDA

### 🎮 Interactive Training Scripts
- `train_interactive.py` — guided wizard for easy configuration
- Automatic hardware detection and optimization recommendations
- One-command training for various GPU tiers (A4000, L40S, etc.)

### 🔧 Multiple Decoding Strategies
- **Standard diffusion sampling** — flexible step counts
- **DDIM sampling** — faster inference with fewer steps
- **Consistency training** (CDLM) — up to 14× faster inference
- Top-k, top-p, and temperature-based sampling

---

## 📐 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     DIMBA Architecture                       │
├─────────────────────────────────────────────────────────────┤
│  Input Tokens                                               │
│       ↓                                                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐ │
│  │   Token     │───→│   Prompt    │───→│  Conditioning   │ │
│  │ Embeddings  │    │  Encoder    │    │      (C)        │ │
│  └─────────────┘    └─────────────┘    └─────────────────┘ │
│       ↓                                      ↓              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           Latent Projection (Optional VAE)          │   │
│  │     z = μ + σ·ε  (reparameterization trick)         │   │
│  └─────────────────────────────────────────────────────┘   │
│       ↓                                                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Cosine Noise Schedule                   │   │
│  │     ᾱ(t) = cos²((t/T + s)/(1+s)·π/2)               │   │
│  │     x_t = √ᾱ(t)·x₀ + √(1-ᾱ(t))·ε                  │   │
│  └─────────────────────────────────────────────────────┘   │
│       ↓                                                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         Mamba-2 Denoiser (T iterations)             │   │
│  │  ┌─────────────────────────────────────────────┐   │   │
│  │  │  Mamba-2 SSM Block × N layers              │   │   │
│  │  │  - Linear-time sequence processing         │   │   │
│  │  │  - Selective state spaces (S6)             │   │   │
│  │  │  - FiLM/Additive conditioning              │   │   │
│  │  └─────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────┘   │
│       ↓                                                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐ │
│  │   Output    │───→│   Latent    │───→│  Token Logits   │ │
│  │ Projection  │    │    Decode   │    │   (Softmax)     │ │
│  └─────────────┘    └─────────────┘    └─────────────────┘ │
│                                                  ↓          │
│                                          Generated Text     │
└─────────────────────────────────────────────────────────────┘
```

### Core Components

| Component | Description |
|-----------|-------------|
| **Token Embeddings** | Learnable embeddings mapping discrete tokens to continuous space |
| **Prompt Encoder** | Lightweight MLP for conditioning on prefix tokens |
| **Noise Schedule** | Cosine schedule following Nichol & Dhariwal (2021) |
| **Timestep Embeddings** | Sinusoidal encodings with MLP projection |
| **Mamba-2 Denoiser** | Stack of SSM blocks with FiLM/additive conditioning |
| **VAE (Optional)** | Token-level variational autoencoder for latent diffusion |

---

## 🚀 Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/devnull37/dimba-lib-exp.git
cd dimba-lib-exp

# Basic installation (CPU + SimpleMamba fallback)
pip install -e .

# With GPU support (full Mamba-2 with CUDA)
pip install -e ".[gpu]"

# Full development setup (includes all extras)
pip install -e ".[all]"
```

### Quick Start

#### Option 1: Interactive Setup (Recommended)

```bash
# Launch the interactive training wizard
python scripts/train_interactive.py
```

The wizard will guide you through:
- Hardware detection (CUDA, MPS, or CPU)
- Model size selection
- Dataset configuration
- Training hyperparameters

#### Option 2: Command-Line Training

```bash
# Train on GPU
python scripts/train.py --config config.yaml --gpus 1 --max-epochs 10

# Train on CPU (uses SimpleMamba)
python scripts/train.py --config config.yaml

# Train on Apple Silicon
python scripts/train.py --config config.yaml --mps
```

#### Option 3: Python API

```python
import torch
from dimba import DIMBA, sample_from_model

# Create a DIMBA model
model = DIMBA(
    vocab_size=50000,
    d_model=512,
    num_diffusion_steps=1000,
    num_denoiser_layers=8,
)

# Generate text
prompt_ids = torch.tensor([[10, 20, 30]])  # Tokenized prompt
generated = sample_from_model(
    model, 
    prompt_ids, 
    seq_len=100, 
    num_steps=50,  # Fewer steps = faster, more steps = better quality
    temperature=1.0,
    top_p=0.95
)

print(generated)
```

---

## 🖥️ Hardware Support

| Platform | Status | Notes |
|----------|--------|-------|
| **NVIDIA CUDA** | ✅ Full support | Best performance with `mamba-ssm>=2.2.0` |
| **Apple Silicon (MPS)** | ✅ Full support | Native Metal backend for M1/M2/M3 |
| **CPU** | ✅ Supported | Uses pure PyTorch `SimpleMamba2` fallback |
| **AMD ROCm** | ⚠️ Experimental | Via PyTorch ROCm builds |

### Hardware-Specific Training Scripts

```bash
# RTX A4000 (16GB VRAM) - 500M parameter model
python scripts/train_fineweb_500m_a4000.py

# L40S / A100 - 1.5B parameter model  
python scripts/train_fineweb_1b.py

# CDLM (Consistency Training) - up to 14× faster inference
python scripts/train_cdlm.py
```

---

## 🧪 Advanced Features

### VAE Pre-training for Latent Diffusion

Pre-train a Variational Autoencoder to compress token embeddings:

```bash
# Basic VAE training
python scripts/train_vae.py \
    --dataset wikitext \
    --dataset-config wikitext-2-raw-v1 \
    --latent-dim 256 \
    --kl-weight 1.0 \
    --epochs 10
```

Use the pre-trained VAE in DIMBA:

```python
model = DIMBA(
    vocab_size=50000,
    d_model=512,
    latent_diffusion=True,
    d_latent=256,
    use_vae_latent=True,
    vae_checkpoint_path='checkpoints/vae/final.ckpt',
)
```

### Consistency Training (CDLM)

Train with Consistency Models for ultra-fast inference:

```bash
python scripts/train_cdlm.py \
    --config config.yaml \
    --consistency-weight 0.5 \
    --delta-min 50 \
    --delta-max 200
```

---

## 📊 Project Status

### ✅ What's Working

- [x] Core diffusion training pipeline
- [x] Mamba-2 denoiser with FiLM conditioning
- [x] Pure PyTorch SimpleMamba2 fallback
- [x] VAE-based latent diffusion
- [x] DDIM sampling for faster inference
- [x] Interactive training wizard
- [x] Multi-GPU training (PyTorch Lightning)
- [x] Apple Silicon (MPS) support
- [x] HuggingFace datasets integration
- [x] BPE tokenization
- [x] EMA (Exponential Moving Average) training
- [x] Checkpointing and resumption

### 🚧 Experimental / In Progress

- [ ] Consistency model training (CDLM)
- [ ] Multi-modal extensions
- [ ] Quantization support (INT8, INT4)
- [ ] ONNX export
- [ ] Flash Attention integration
- [ ] Rotary Position Embeddings (RoPE)

### ⚠️ Known Limitations

1. **Training cost**: Diffusion models require substantial compute for pre-training
2. **Discrete-continuous gap**: Mapping between discrete tokens and continuous embeddings affects rare token handling
3. **Hyperparameter sensitivity**: Performance varies significantly with diffusion steps (T), architecture depth
4. **Conditioning robustness**: Long-context conditioning requires careful tuning

---

## 📁 Project Structure

```
dimba-lib-exp/
├── src/dimba/                 # Core library
│   ├── models/               # Model implementations
│   │   ├── diffusion.py      # Main DIMBA model
│   │   ├── denoiser.py       # Mamba-2 denoiser
│   │   ├── vae.py            # Token VAE
│   │   ├── embeddings.py     # Embedding layers
│   │   └── simple_mamba.py   # Pure PyTorch Mamba
│   ├── diffusion/            # Diffusion utilities
│   │   ├── schedules.py      # Noise schedules
│   │   └── sampling.py       # Sampling algorithms
│   ├── data/                 # Dataset loaders
│   ├── training/             # Training utilities
│   ├── evaluation/           # Metrics (BLEU, ROUGE, etc.)
│   └── tokenizers/           # Tokenization
├── scripts/                  # Training & utility scripts
│   ├── train_interactive.py  # Interactive wizard ⭐
│   ├── train.py              # Generic training
│   ├── train_vae.py          # VAE pre-training
│   ├── train_cdlm.py         # Consistency training
│   ├── generate.py           # Text generation
│   ├── evaluate.py           # Evaluation
│   └── setup/                # Installation scripts
├── configs/                  # Configuration files
├── tests/                    # Unit tests
├── notebooks/                # Jupyter notebooks
├── paper/                    # Research paper
└── docs/                     # Documentation
```

---

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Install** development dependencies: `pip install -e ".[dev]"`
4. **Make** your changes
5. **Run** tests: `pytest`
6. **Format** code: `black src/ && isort src/`
7. **Submit** a Pull Request

### Development Setup

```bash
pip install -e ".[all]"
pre-commit install  # Optional: for automated formatting
```

---

## 📖 Citation

If you use DIMBA in your research, please cite:

```bibtex
@article{allafi2025dimba,
  title={DIMBA: Revolutionizing Theoretical Ultra-Fast Inference and Advanced Reasoning with Mamba-Based Diffusion},
  author={Allafi, Faris},
  year={2025}
}
```

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🔗 Links

- 🌐 **Website**: [dimbalabs.xyz](https://dimbalabs.xyz)
- 👤 **Author**: [farisallafi.xyz](https://farisallafi.xyz)
- 📄 **Paper**: Available in the `paper/` directory
- 💻 **Repository**: [github.com/devnull37/dimba-lib-exp](https://github.com/devnull37/dimba-lib-exp)
- 🐛 **Issues**: [GitHub Issues](https://github.com/devnull37/dimba-lib-exp/issues)

---

## 💡 Acknowledgments

- **Mamba** — [State Space Models](https://github.com/state-spaces/mamba) by Tri Dao and Albert Gu
- **Diffusion Models** — Inspired by works from OpenAI, Google Research, and the broader diffusion community
- **PyTorch Lightning** — For the excellent training framework
- **HuggingFace** — For datasets and transformers infrastructure

---

<p align="center">
  <i>Built with ❤️ by Faris Allafi</i>
</p>
