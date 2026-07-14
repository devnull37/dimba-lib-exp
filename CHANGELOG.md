# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

The merged v2 overhaul and the 2026-07-13 next-run pass focus on correctness,
modern diffusion-LM capabilities, exact run recovery, and measured CPU/GPU efficiency.

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
  `TorchMamba2` remains the weight-compatible CPU/MPS fallback. `SimpleMamba2`
  is retained only as an explicit lightweight/reference implementation.
- **Infrastructure**: a CPU inference benchmark (`scripts/benchmark.py`),
  smoke/import test suites, GitHub Actions CI (Python 3.10 and 3.12, CPU only),
  pre-commit hooks (black, isort, trailing-whitespace, end-of-file-fixer), and
  this changelog.
- **Hybrid Muon/AdamW optimizer**: native PyTorch Muon when available and a compatible
  Moonlight-style fallback, with semantic parameter routing and checkpoint/scheduler support.
- **Native masked DDP**: one process per GPU, no-sync gradient accumulation, non-duplicating
  sharding, rank-zero shared caches/checkpoints, reduced log metrics, and exact same-topology
  resume for `masked_diffusion_finetune.py` and `mdm_sft_cfg2.py`.
- **Masked training objective**: vectorized corruption and exact selected-response-token
  vocabulary projection rather than full `[batch, sequence, vocab]` materialization.
- **CUDA benchmark gate**: full masked inference, DDIM, DPM-Solver++(2M), continuous fused CE,
  and masked CE forward/backward cases with p50/p95, HBM, throughput, parity, environment, and
  source-tree provenance in `scripts/benchmark_h100.py`.
- **Optional Liger fused CE** for exact uniform continuous reductions; weighted and masked
  objectives retain native selected-token math.

### Changed

- Sampling is consolidated around schedule-consistent DDIM, DPM-Solver++(2M), flow,
  and masked updates with device-resident schedules/trajectories, batched CFG, selected
  vocabulary projection, batched verification, and no redundant final forward.
- CUDA production launchers are Mamba-2 only and fail closed unless every live mixer is
  `mamba_ssm.Mamba2` and `causal-conv1d` is available. CPU/MPS keep the weight-compatible
  TorchMamba2 fallback; Mamba-1 is never selected.
- AdamW remains the production default; Muon is an explicit A/B pilot.
- Checkpoints use atomic replacement. Continuous Stage 3 records exact optimizer/RNG/phase/data
  cursor/backend state; masked DDP records per-rank RNG/data state and world size.
- MLX masked generation keeps the trajectory on Apple GPU, batches CFG, selects unresolved
  positions before vocabulary projection, and converts to host once.

### Fixed

- Correctness fixes to the noise schedule and the train/inference sampling math so
  the reverse process is consistent with the forward (training) process, including
  terminal-SNR handling and per-step variance computation.
- More robust logit post-processing during sampling (temperature, top-k / top-p)
  to avoid NaNs from fully-masked distributions.
- Prompt/response scope in continuous GRPO generation now uses the shared clean-prefix sampler
  rather than a private reverse loop.
- Rank-zero cache/checkpoint failures are broadcast so DDP peers raise instead of deadlocking.
- Stage-3 3b continues from the cumulative 3a data cursor; partial accumulation groups use their
  true divisor; training metric logging no longer forces a CUDA synchronization every microbatch.

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
