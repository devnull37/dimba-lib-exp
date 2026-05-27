# AGENTS.md

Guidance for any agentic coding model working in this repo. Reflects the **v2 overhaul**
(branch `feature/dimba-v2-overhaul`, PR #18). For deeper detail see
`docs/OVERHAUL_STATUS.md`, `docs/IMPROVEMENT_PLAN.md`, and `docs/RESEARCH_DIRECTIONS.md`.

## Project overview

**DIMBA** is a non-autoregressive **latent-diffusion** language model: continuous Gaussian
diffusion runs in a learned latent space (VAE/projector over token embeddings; raw-embedding
diffusion is the degenerate `latent_diffusion=False` case), denoised by a **bidirectional
Mamba** backbone, generating whole sequences in parallel by iterative denoising.

- **v1 = `paper/main.pdf`** — an *architectural concept* (explicitly untested). Do not treat
  it as ground truth: it contains a prompt-conditioning leak (`C = PromptEncoder(X₀)`) and an
  MSE-only objective that the v2 code deliberately fixes.
- **v2 = this repo** — the implementation + the overhaul below. **This file describes v2.**

## ⚠️ Environment gotchas (read first)

- **`import torch` segfaults at interpreter *teardown*** on the original dev box (Windows),
  and the bare `python` on PATH is a WindowsApps shim that hangs. **Use the project venv**:
  `venv\Scripts\python.exe` (Windows) / `venv/bin/python` (mac). For scripts that import
  torch, **end with `os._exit(0)`** after flushing to dodge the teardown crash.
- **Validate without running torch**: `python -m compileall src/dimba scripts tests` (syntax).
- **Runtime smoke**: `venv/bin/python .sisyphus/smoke_full.py` (end-to-end, uses `os._exit`).
- **CI** (`.github/workflows/ci.yml`) runs the real `pytest` suite on clean Linux runners
  (py3.10/3.12) with working torch — that's the source of truth for runtime tests.
- **Apple Silicon (M1/M2/M3) training**: use the **PyTorch MPS** path (the `backends/mlx/`
  port is a skeleton, not training-ready). Set `PYTORCH_ENABLE_MPS_FALLBACK=1`, use **fp32**,
  and `latent_diffusion=True`. `SimpleMamba2` uses the vectorized scan on MPS; keep `seq_len`
  ≤ ~256. See the small-model recipe in `docs/OVERHAUL_STATUS.md`.

## Architecture (current data flow)

```
input_ids ─► token_embed ─► encode_latent (×latent_scale → ~unit variance)
                                   │
        prompt (pooled, response-free) ─┐         add_noise (cosine, zero-terminal-SNR)
        timestep_embed τ(t) ────────────┤              │  (only response positions if prompt_mask)
                                         ▼              ▼
                       Mamba denoiser (N bidirectional blocks, FiLM/additive cond)
                                         │   [+ self-conditioning: prev x̂₀ fused in]
                                         ▼
                         raw pred → x̂₀ latent (x0 or v param)
                                         ▼
                       decode_latent (÷latent_scale → embedding) ─► output_head ─► logits
```

Key points that differ from v1 and **must not be reintroduced as bugs**:
- **No conditioning leak.** Conditioning is the *prompt only* — a pooled prompt summary, and
  (when `prompt_mask` is given) the prompt tokens kept *clean in-sequence* while only the
  response is noised, with loss on the response. Never condition on the clean target.
- **`forward()` always returns the 3-tuple** `(x_pred, noise, latent_info)`.
- **`encode_latent`/`decode_latent` carry `latent_scale`** and round-trip exactly. Anything
  that diffuses or samples must go through them (signal must be ~unit variance for a
  calibrated SNR). Call `model.calibrate_latent_scale(batch)` before training in latent mode.
- The model stores its full constructor config in `model.config` (used to build EMA/replicas).

## Model API (`src/dimba/models/diffusion.py`)

`DIMBA(...)` notable kwargs: `latent_diffusion`, `d_latent`, `use_vae_latent`,
`bidirectional=True`, `self_conditioning=False`, `prediction_type="x0"|"v"`,
`zero_terminal_snr=True`, `embed_init_std=0.02`, `latent_scale=None` (auto = `1/embed_init_std`
for embedding mode, `1.0` for latent mode → calibrate).

- `forward(input_ids, t, noise=None, prompt_mask=None, x_self_cond=None, drop_cond=False)`
- `predict_token_logits(input_ids, t)` → `[B,L,vocab]` (the **discrete/masked** track)
- `denoise_to_x0_latent(x_t, t, cond, x_self_cond=None)` / `denoise_step(...)` (inference)
- `conditioning_from_prompt(prompt_ids=None, batch_size, device, drop_cond=False)` → `[B,1,cond_dim]`
- `encode_latent` / `decode_latent` / `calibrate_latent_scale(batch, target_std=1.0)`

## Diffusion modes

1. **Continuous latent (default)** — `GaussianEmbeddingCorruption`; the `forward()` path above.
2. **Discrete / masked (LLaDA/MDLM)** — `diffusion/corruption.py:AbsorbingMaskCorruption` +
   `diffusion/masked_sampling.py:masked_diffusion_sample(predict_logits, ...)`; model side is
   `predict_token_logits`. Needs a `[MASK]` token id (not in the tokenizer yet — pass explicitly).
3. **Hybrid (novel, experimental)** — `HybridCorruption` interpolates masked ↔ Gaussian per token.

## Training (`src/dimba/training/trainer.py`)

Use **`compute_dimba_losses(model, input_ids, t, *, ce_loss_weight=1.0, min_snr_gamma=5.0,
prompt_mask=None)`** → `(loss, parts)`. It combines:
- **min-SNR-γ-weighted** diffusion regression in latent space (x0 or v target),
- a **cross-entropy / rounding anchor** (trains the head/decoder, ties to real tokens),
- **latent autoencoder consistency** + optional **VAE KL** (latent mode).

`DIMBALightningModule` and `SimpleTrainer` both call it; both accept `ce_loss_weight` /
`min_snr_gamma`. The CDLM consistency loss (`compute_consistency_loss`) is de-leaked (null cond).
Schedule helpers: `CosineNoiseSchedule(num_steps, zero_terminal_snr=True)` with `.add_noise`,
`.velocity`, `.predict_x0_from_v`, `.snr`, plus `enforce_zero_terminal_snr`.

## Inference (`src/dimba/diffusion/sampling.py`)

- `sample_from_model(model, prompt_ids, seq_len, num_steps, temperature, top_k, top_p,
  guidance_scale=1.0, eta=0.0, clamp_to_tokens=False)` — correct x0-DDIM, CFG, self-cond carry,
  clean-prefix conditioning. `DDIMSampler` wraps it.
- `diffusion/rerank.py:best_of_k(generate_fn, score_fn, k)` + `diffusion_elbo_score(...)` — best-of-K.

## Post-training (`scripts/finetuning/`)

- `finetune_sft.py` — SFT (leak-free per-position prompt cond; response-only CE via labels).
- `finetune_dpo.py` + `training/preference.py` — **DPO/IPO/SimPO** with a diffusion-ELBO/VRPO
  surrogate (diffusion log-likelihoods are intractable). Preferred for preference pairs.
- `finetune_grpo.py` + `training/rewards.py` — GRPO with a **pluggable `--reward`** (default
  `numeric`; `token_overlap` kept but deprecated — it just teaches copying).

## Performance (`src/dimba/models/parallel_scan.py`, `utils/compile.py`, `backends/`)

- `selective_scan(dt, A, Bmat, C, x, *, stable=True, chunk_size=64)` — vectorized, numerically
  stable (chunked); `selective_scan_sequential` is the parity reference; `bidirectional_*` too.
  `SimpleMamba2` uses it and falls back to the sequential scan if the result is non-finite.
- `maybe_compile(module)` — `torch.compile` on CUDA only. `backends/mlx/` — MLX skeleton (WIP).

## Repo layout

```
src/dimba/{models,diffusion,training,data,tokenizers,evaluation,utils,backends}/
scripts/{train*,generate,evaluate,benchmark}.py  scripts/finetuning/finetune_{sft,dpo,grpo,interactive}.py
configs/  tests/  notebooks/  docs/  paper/
```

## Conventions

- Python ≥3.9, **black line-length 100**, type hints, Google-style docstrings.
- Don't reintroduce the conditioning leak, the 2-tuple `forward`, positive-`A` SSM, or
  un-scaled latents. Run `compileall` + the smoke before claiming a change works.

## Current status (2026-05-27)

- **PR #18** open into `main`: `c2352ba` (overhaul) + `60f30eb` (latent scale-factor). **Not
  merged.** All known bugs fixed; `compileall` clean; runtime smoke 14/14 (venv python).
- **Open follow-ups**: first-class masked-mode training script + `[MASK]` token; an M1
  quickstart config; train a real VAE to calibrate the latent against; cross-attention
  conditioning (stronger than pooled-global); real speed/quality benchmarks once compute lands.
