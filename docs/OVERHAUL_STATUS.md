# v2 Overhaul — Status & Validation

Branch: `feature/dimba-v2-overhaul`. This summarizes the autonomous overhaul pass:
what changed, how it was validated, and what's left.

## What changed

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
- **Performance**: `models/parallel_scan.py` (chunked, numerically-stable, length-parallel selective scan; bidirectional), `utils/compile.py` (`maybe_compile`), `backends/mlx/` (MLX denoiser skeleton + safetensors bridge).
- **Inference**: `diffusion/rerank.py` (best-of-K via ELBO self-scoring).
- **Infra**: `scripts/benchmark.py`, GitHub Actions CI, pre-commit, `CHANGELOG.md`, and new tests.

## Validation

- **`python -m compileall src/dimba scripts tests` → exit 0** (every file, all 5 work packages).
- **End-to-end runtime smoke → 12/12 OK** (`.sisyphus/smoke_full.py`): forward/backward/loss across all 6 model modes, prompt-mask path, sampling + CFG, masked hook, corruption, masked sampling, and scan parity (1.4e-6).
- Parallel-scan parity vs the sequential reference: **7e-15** (float64, per the perf work package).

**Environment note:** on this Windows box the bare `python` alias hangs and `import torch` segfaults at interpreter *teardown*. The working interpreter is **`venv\Scripts\python.exe`**; scripts that import torch should finish with `os._exit(0)` after flushing, or just run under pytest in CI. The GitHub Actions workflow runs the suite on a clean Linux runner with working torch.

## How to validate yourself

```bash
venv\Scripts\python.exe -m pytest tests -q          # full suite (needs pytest installed)
venv\Scripts\python.exe .sisyphus\smoke_full.py     # quick end-to-end smoke
venv\Scripts\python.exe scripts\benchmark.py         # latency / NFE / tokens-per-sec
```

## Not committed

All changes are left **staged-but-uncommitted** on the branch for your review (your in-progress `scripts/train_interactive.py` is intentionally untouched and excluded). To commit just the overhaul:

```bash
git add src tests docs .github .pre-commit-config.yaml CHANGELOG.md README.md pyproject.toml \
        scripts/benchmark.py scripts/finetuning/finetune_dpo.py scripts/finetuning/finetune_grpo.py
git commit -m "feat: v2 overhaul — correctness fixes, self-cond/CFG, discrete mode, DPO, perf"
```

## Suggested follow-ups (need compute / runtime iteration)

- A first-class masked-mode training script + a `[MASK]` token in the tokenizer (building blocks are in `corruption.py` / `masked_sampling.py`).
- Make `finetune_sft.py` use the new clean-prefix conditional forward (it currently uses its own leak-free per-position path — fine, but could share the new API).
- Cross-attention prompt conditioning (stronger than pooled-global) — see `docs/RESEARCH_DIRECTIONS.md`.
- Real quality/speed benchmarks once compute is available, to replace the paper's projected "ultra-fast" claim with measured numbers.
