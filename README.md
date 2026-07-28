# DIMBA 🐍✨

[![PyPI version](https://badge.fury.io/py/dimba-lib.svg)](https://badge.fury.io/py/dimba-lib)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

> **Diffusion-based Mamba Architecture for Non-Autoregressive Text Generation**

DIMBA combines diffusion language modeling with a bidirectional Mamba-2 state-space backbone.
Instead of generating strictly left to right, it iteratively denoises whole token sequences.

This public repository contains the model architecture, PyTorch/MLX inference, evaluation tools,
tests, and release evidence. Training, fine-tuning, distillation, weight transmutation, and
scale-run orchestration are not included.

- **Paper:** [DIMBA: Revolutionizing Theoretical Ultra-Fast Inference and Advanced Reasoning with Mamba-Based Diffusion](https://doi.org/10.55277/researchhub.lu30m581.2)
- **Website:** [dimbalabs.xyz](https://dimbalabs.xyz)
- **Author:** [farisallafi.xyz](https://farisallafi.xyz)

## Released model

[`devnull37/hr-diffuse-1-nano`](https://huggingface.co/devnull37/hr-diffuse-1-nano) is a
masked discrete-diffusion model on a bidirectional Mamba-2 backbone:

- **287.9M measured parameters**, described as 135M-class capacity because it was transferred
  from SmolLM-135M and stores largely redundant directional stacks.
- SmolLM-135M tokenizer plus one `[MASK]` row: vocab 49,153, mask id 49,152.
- Required production recipe: 128 steps, temperature 0.7, top-k 20, CFG 2.0, and
  exempt-first frequency penalty 0.7.
- Published 40-item results: 15.0% factual QA, 14.0% native infill recovery, and 7.5% loop rate.

Read the [system card](docs/hr-diffuse-1-nano_system_card.md), [benchmark table](docs/benchmarks.md),
and [quality-slider study](docs/slider_curve.md) for methodology and limitations.

## What the public runtime supports

- Bidirectional Mamba-2 denoising with prompt conditioning, self-conditioning, and CFG.
- Masked diffusion plus continuous DDIM, DPM-Solver++, and flow sampling.
- Weight-compatible pure-PyTorch `TorchMamba2` fallback for CPU and Apple MPS.
- Native MLX inference on Apple Silicon.
- Device-resident masked sampling, selected-position vocabulary projection, and batched CFG.
- Optional CUDA-graph replay of the fixed-shape raw denoiser.
- BPE, Hugging Face, and simple character tokenizers.

The CUDA-graph path has one directional RTX 4090 production-shape measurement:
**17.92 s eager -> 1.16 s graphed (15.49x)** with exact 40/40-token parity. It is not a repeated
H100 promotion result. See [benchmarks](docs/benchmarks.md) for the claim boundary.

## Installation

```bash
git clone https://github.com/devnull37/dimba-lib-exp.git
cd dimba-lib-exp

# CPU / PyTorch inference
pip install -e .

# NVIDIA fused Mamba-2 inference
pip install -e ".[gpu]"

# Apple-Silicon MLX inference
pip install -e ".[mlx]"

# Development and optional backends
pip install -e ".[all]"
```

## Generate with the released model

The production CLI downloads the release checkpoint from Hugging Face when a local checkpoint is
not present:

```bash
python scripts/generate.py "What is the capital of France?" --quality 0.5
```

Useful variants:

```bash
# Force the PyTorch backend
python scripts/generate.py "Complete this sentence:" --backend torch

# MLX is auto-selected on a supported Mac; fp16 weights are opt-in
python scripts/generate.py "Complete this sentence:" --backend mlx --mlx-dtype fp16

# Inspect every supported option
python scripts/generate.py --help
```

The quality knob changes denoising steps and best-of-N candidates. Its compute cost scales
predictably, but accuracy was noisy at this model size; do not assume higher quality always scores
better. See the [measured curve](docs/slider_curve.md).

## Python API

```python
import torch
from dimba import DIMBA, sample_from_model

model = DIMBA(
    vocab_size=50000,
    d_model=512,
    num_diffusion_steps=1000,
    num_denoiser_layers=8,
).eval()

prompt_ids = torch.tensor([[10, 20, 30]])
generated = sample_from_model(
    model,
    prompt_ids,
    seq_len=100,
    num_steps=50,
    temperature=1.0,
    top_p=0.95,
)
```

For a trained checkpoint, rebuild the model from its stored configuration and load its state dict
strictly. The release CLI demonstrates both local and Hugging Face checkpoint loading.

## Backends

| Platform | Public inference path | Claim boundary |
|---|---|---|
| NVIDIA CUDA | Native `mamba_ssm.Mamba2`; optional denoiser CUDA graphs | One directional RTX 4090 graph result; repeated H100 gate pending |
| Apple Silicon MLX | Full MLX sampler | Fastest supported Mac path; see measured tables |
| Apple Silicon MPS | PyTorch `TorchMamba2` | Supported fallback |
| CPU | PyTorch `TorchMamba2` or tiny `SimpleMamba2` | Supported; slow for release-size sampling |

See [BACKENDS.md](docs/BACKENDS.md) for parity, precision, and Apple-Silicon measurements.

## Evaluation and utility scripts

```bash
# Tiny dependency-light CPU benchmark
python scripts/benchmark.py

# Compare PyTorch and MLX implementations
python scripts/verify_mlx_model.py

# Inspect checkpoint evaluation/comparison interfaces
python scripts/evaluate.py --help
python scripts/eval_vs_smollm.py --help
python scripts/perplexity_eval.py --help
```

`scripts/upload_to_hf.py` is an explicit release-artifact uploader. It does not run during normal
inference or tests.

## Repository structure

```text
src/dimba/
  models/         DIMBA, denoiser, TorchMamba2/SimpleMamba2, VAE
  diffusion/      schedules, corruption, masked/continuous sampling, reranking
  backends/mlx/   Apple-GPU inference
  inference/      block-sequential inference
  evaluation/     metrics
  tokenizers/     tokenizer implementations
  utils/          compile and CUDA-graph helpers

scripts/          generation, sampling, evaluation, parity, benchmark, release upload
tests/            architecture, sampler, backend, parity, and regression tests
docs/             backend guidance and public release evidence
paper/            historical v1 paper
```

There are no public training or distillation entry points. Read [AGENTS.md](AGENTS.md) before
changing the repository.

## Development

```bash
pip install -e ".[dev]"
python3 -m compileall src/dimba scripts tests
python3 -m pytest tests/ -q -p no:cacheprovider --override-ini addopts=
git diff --check
```

CUDA/H100 performance must be verified on the named NVIDIA hardware. This Apple-Silicon
development machine can validate CPU/MPS/MLX behavior but cannot substantiate CUDA throughput.

## Citation

```bibtex
@article{allafi2025dimba,
  title={DIMBA: Revolutionizing Theoretical Ultra-Fast Inference and Advanced Reasoning with Mamba-Based Diffusion},
  author={Allafi, Faris},
  year={2025}
}
```

## License

This project is licensed under the
[Creative Commons Attribution-NonCommercial 4.0 International License](LICENSE).

## Links

- [Website](https://dimbalabs.xyz)
- [Author](https://farisallafi.xyz)
- [Paper](https://doi.org/10.55277/researchhub.lu30m581.2)
- [Repository](https://github.com/devnull37/dimba-lib-exp)
- [Issues](https://github.com/devnull37/dimba-lib-exp/issues)
