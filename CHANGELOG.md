# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

The v2 overhaul focuses on correctness, modern diffusion-LM capabilities, and
CPU/GPU performance. Items below are in progress on the `feature/dimba-v2-overhaul`
branch and describe the direction at a high level.

### Added

- **Self-conditioning** for the denoiser: the model can condition each denoising
  step on its own previous clean-sample estimate, improving sample quality at a
  small compute cost.
- **Classifier-free guidance (CFG)**: joint conditional/unconditional training via
  prompt dropout, with a guidance scale applied at sampling time.
- **Discrete / masked diffusion mode**: an alternative to continuous Gaussian
  diffusion over embeddings, operating directly on token states with a masked /
  absorbing-state corruption process.
- **Preference optimization (DPO)**: direct preference optimization for aligning
  generations to preferred outputs, building on the existing preference-data
  pipeline.
- **Performance backends**: pluggable denoiser backends so the optimized
  `mamba-ssm` kernels are used when available (GPU) while the pure-PyTorch
  `SimpleMamba2` remains the default CPU-friendly fallback.
- **Infrastructure**: a CPU inference benchmark (`scripts/benchmark.py`),
  smoke/import test suites, GitHub Actions CI (Python 3.10 and 3.12, CPU only),
  pre-commit hooks (black, isort, trailing-whitespace, end-of-file-fixer), and
  this changelog.

### Changed

- Sampling is being consolidated around correct, schedule-consistent update rules
  for both ancestral and DDIM-style accelerated inference, with configurable
  step counts and guidance.
- Conditioning, latent-projection, and timestep-embedding interfaces are being
  unified so continuous, latent, and discrete modes share a single denoiser path.

### Fixed

- Correctness fixes to the noise schedule and the train/inference sampling math so
  the reverse process is consistent with the forward (training) process, including
  terminal-SNR handling and per-step variance computation.
- More robust logit post-processing during sampling (temperature, top-k / top-p)
  to avoid NaNs from fully-masked distributions.

## [0.1.0] - 2025-01-24

### Added

- Initial DIMBA library: continuous Gaussian diffusion over token embeddings with
  a Mamba-2 denoiser for non-autoregressive text generation.
- Pure-PyTorch `SimpleMamba2` denoiser for CPU usage without compiled kernels.
- Cosine noise schedule, `sample_from_model`, and a DDIM sampler.
- Character and BPE tokenizers, dataset utilities, evaluation metrics, and
  PyTorch Lightning training utilities.
- LoRA / Q-LoRA adapters and a finetuning data pipeline (SFT and preference data).

[Unreleased]: https://github.com/devnull37/dimba-lib/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/devnull37/dimba-lib/releases/tag/v0.1.0
