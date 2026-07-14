# v2 Overhaul — historical record and current status

The original v2 overhaul was developed on `feature/dimba-v2-overhaul` and is now in `main`.
The first section below records that architectural work. The 2026-07-13 section is the source
for the current optimization and next-run preparation state.

## 2026-07-13 optimization and next-run preparation

- **Mamba-2 only:** `denoiser.py` imports `mamba_ssm.Mamba2`, never the Mamba-1 API. CUDA
  production launchers inspect every live mixer and require both native Mamba-2 and
  `causal-conv1d`; CPU/MPS retain the weight-compatible TorchMamba2 fallback.
- **Muon pilot:** a shared hybrid optimizer routes eligible hidden matrices to Muon and leaves
  embeddings, heads, norms, biases, vectors, and SSM state parameters on AdamW. AdamW remains
  the default until a time-to-quality A/B passes.
- **Masked mode is first-class:** vectorized corruption, exact selected-token CE, canonical base
  and SFT launchers, clean-prefix generation, strict checkpoints, and a real `[MASK]` row are
  implemented and tested.
- **Multi-GPU:** the canonical masked launchers use native one-process-per-GPU DDP, `no_sync`
  accumulation, rank-zero cache/checkpoint I/O with failure broadcast, reduced log metrics, and
  exact same-world-size resume. Continuous H100 Stage 3 now uses the same DDP process model with
  rank-zero alignment, rank+worker-sharded FineWeb streams, complete-loss/KD wrapping, per-rank
  RNG checkpoints, and fixed-global-token step math.
- **Inference:** masked and continuous trajectories stay on accelerator; CFG is batched into one
  dispatch; selected positions alone reach the vocabulary head; schedules, confidence ranking,
  self-conditioning, and finalization no longer force per-step host synchronization. Best-of-N
  verification is batched.
- **CUDA efficiency:** BF16/TF32, fused AdamW, official Mamba-2/causal-conv kernels, and optional
  semantics-gated Liger fused CE are wired. Bespoke kernels are gated on the target-H100 benchmark
  instead of being added speculatively.
- **Run safety:** atomic checkpoints, exact Stage-3 optimizer/RNG/data-cursor resume, backend and
  trajectory signatures, cumulative 3a→3b data, isolated save directories, one-phase quality
  gates, and safe OOM/stop behavior replace the run-1 shortcuts.

See [PERFORMANCE_AND_SCALING.md](PERFORMANCE_AND_SCALING.md) for commands, measured results,
forecasts, and kernel-promotion gates; [NEXT_RUN_PLAN.md](NEXT_RUN_PLAN.md) for the training ladder;
and [BABYSIT_LOOP.md](BABYSIT_LOOP.md) for operational recovery.

## Original v2 architectural overhaul

**Correctness fixes (core)**
- Conditioning leak removed — prompt is encoded as *clean context* (pooled prompt + clean in-sequence prefix), never the target; response-only loss when a `prompt_mask` is given. (`models/diffusion.py`)
- Real **zero-terminal-SNR** cosine schedule (Lin et al. 2023), replacing the docstring-only claim. (`diffusion/schedules.py`)
- **Bidirectional** Mamba denoiser + genuine **Mamba-2** preference (was importing the Mamba-1 API). (`models/denoiser.py`)
- `SimpleMamba2` rewritten: stable negative-`A` state matrix, per-channel input (was collapsing the inner dim), no double norm/residual; uses the vectorized scan. (`models/simple_mamba.py`)
- Correct x0-parameterized **DDIM** sampler (+ optional v-prediction); removed library `print()`s. (`diffusion/sampling.py`)
- `forward()` now always returns the 3-tuple the trainer expects; `get_model_config` reads a stored config (was reading non-existent attrs). (`models/diffusion.py`, `training/trainer.py`)
- FiLM identity-init bug fixed (γ was `sum(cond)`, now `1`). (`models/embeddings.py`)
- `denoise_step` referenced a renamed helper (`_run_denoiser`) → fixed to delegate to `denoise_to_x0_latent`. (`models/diffusion.py`)
- `SimpleMamba2`'s vectorized scan can underflow to NaN for large state-decay over long sequences → now falls back to the stable sequential scan when the parallel result is non-finite. (`models/simple_mamba.py`)
- `pyproject.toml` isort key `multi_line_mode` → `multi_line_output` (the typo crashed the isort/pre-commit hook).
- **Latent/embedding scale calibration** — the diffused signal is now scaled to ~unit variance (`latent_scale`, à la Stable Diffusion's `0.18215`) so the schedule's SNR is meaningful; `DIMBA.calibrate_latent_scale(batch)` measures it for the VAE/latent path. Embeddings initialized at std 0.02 against unit-variance noise were crushing the effective SNR at every timestep. (`models/diffusion.py`, `models/embeddings.py`)

**Research upgrades**
- **Self-conditioning**, **classifier-free guidance**, **min-SNR-γ** weighting, **cross-entropy / rounding** anchor + latent-AE consistency, **v-prediction** option. (`models/diffusion.py`, `training/trainer.py`)

**New capabilities**
- **Discrete / masked + hybrid diffusion**: `diffusion/corruption.py` (`GaussianEmbeddingCorruption`, `AbsorbingMaskCorruption`, novel `HybridCorruption`), `diffusion/masked_sampling.py` (LLaDA-style confidence remasking), and a `DIMBA.predict_token_logits` hook.
- **Post-training**: `training/preference.py` (DPO/IPO/SimPO + diffusion ELBO surrogate + VRPO antithetic sampling), `training/rewards.py` (verifiable/pluggable rewards; token-overlap demoted to a warned legacy option), `scripts/finetuning/finetune_dpo.py`; GRPO reward made pluggable (`--reward`, default `numeric`).
- **Performance**: `models/parallel_scan.py` (chunked, numerically-stable, length-parallel selective scan; bidirectional), `utils/compile.py` (`maybe_compile`), and the initial `backends/mlx/` port.
- **Inference**: `diffusion/rerank.py` (best-of-K via ELBO self-scoring).
- **Infra**: `scripts/benchmark.py`, GitHub Actions CI, pre-commit, `CHANGELOG.md`, and new tests.

## Original validation record

- **`python -m compileall src/dimba scripts tests` → exit 0** (every file, all 5 work packages).
- **Historical end-to-end runtime smoke → 12/12 OK**: forward/backward/loss across all 6 model modes, prompt-mask path, sampling + CFG, masked hook, corruption, masked sampling, and scan parity (1.4e-6). The former `.sisyphus/smoke_full.py` helper is not in the current checkout; use the maintained pytest smoke below.
- Parallel-scan parity vs the sequential reference: **7e-15** (float64, per the perf work package).

**Environment note:** on this Windows box the bare `python` alias hangs and `import torch` segfaults at interpreter *teardown*. The working interpreter is **`venv\Scripts\python.exe`**; scripts that import torch should finish with `os._exit(0)` after flushing, or just run under pytest in CI. The GitHub Actions workflow runs the suite on a clean Linux runner with working torch.

## Current validation commands

```bash
python3 -m compileall -q src/dimba scripts tests
PYTHONPATH=src python3 -m pytest -q -o addopts=''
PYTHONPATH=src python3 -m pytest -q -o addopts='' tests/test_smoke.py
python3 scripts/benchmark_h100.py --dry-run           # CUDA-free plan validation
```

## Remaining work that genuinely needs compute or research

- Run the complete trained-checkpoint H100 benchmark and record the JSON before promoting compile
  or any new CUDA kernel.
- Run the AdamW/Muon next-run pilot and compare wall-clock time to the same held-out quality, not
  only loss at a fixed step.
- Extend exact DDP resume beyond the implemented continuous Stage-3 path only if another
  stateful continuous pipeline becomes a scale bottleneck.
- Cross-attention prompt conditioning (stronger than pooled-global) — see `docs/RESEARCH_DIRECTIONS.md`.
- Train and calibrate a real latent VAE before making latent-mode quality claims.
