# Changelog

## Unreleased

- Kept the public package focused on model architecture, diffusion, inference,
  evaluation, tokenization, and CPU/MPS/MLX/CUDA inference backends.
- Added device-resident sampling, batched classifier-free guidance, selected-token
  projection, guarded compilation, and fixed-shape CUDA graph replay.
- Added public inference benchmarks, backend parity checks, and release tooling.

## 0.1.0 - 2025-01-24

- Released DIMBA's bidirectional Mamba diffusion architecture, samplers,
  tokenizers, evaluation metrics, and portable pure-PyTorch fallback.

[Unreleased]: https://github.com/devnull37/dimba-lib-exp/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/devnull37/dimba-lib-exp/releases/tag/v0.1.0
