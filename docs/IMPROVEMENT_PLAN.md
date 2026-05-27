# DIMBA Improvement Plan

> Status: proposed roadmap, 2026-05-27. Author of plan: research synthesis for DimbaLabs.
> Scope: turn DIMBA from a faithfully-implemented *concept* into an empirically validated, competitive diffusion LM.

## Context — why this plan exists

DIMBA has two layers:

- **v1 = the paper** (`paper/main.pdf`). It is explicitly an *architectural proposal*: "This work is architectural; implementation and empirical evaluation are future work due to current compute constraints." It defines the diffusion-over-embeddings + Mamba-2 denoiser design and a conceptual training/inference procedure.
- **v2 = this repo**. A faithful implementation of v1 **plus** extensions: latent diffusion (VAE), LoRA/Q-LoRA, an SFT/GRPO finetuning suite, a homegrown "CDLM" consistency loss, interactive wizards, MPS support.

The repo is therefore at the exact stage the paper named as future work: **validation, benchmarking, and ablations**. Recent research (2022–2026) tells us, fairly decisively, which of the original design choices will and won't hold up. This plan fixes the fragile choices, adds the high-impact techniques, and adds the paradigm (discrete/masked diffusion) that has actually scaled — while keeping DIMBA's identity (Mamba backbone, non-autoregressive, fast inference).

The guiding principle: **you cannot improve what you cannot measure.** Several "features" are currently claimed but unverified (the speed claims, the "zero-terminal-SNR fix", "Mamba-2"). Phase 0 makes the project honest and measurable; everything after is gated on benchmarks.

---

## Verified findings (read the code, not just the docs)

| # | Finding | Where | Severity |
|---|---------|-------|----------|
| 1 | **Conditioning leak / train-inference mismatch.** Training conditions on `encode_prompt(input_ids)` — the *clean target itself* (paper: `C = PromptEncoder(X₀)`). At inference the model is conditioned on a *different* prompt. The denoiser can "cheat" during training and faces a distribution shift at inference. | `src/dimba/models/diffusion.py:239-240` | High |
| 2 | **"Zero terminal SNR fix" is claimed but not implemented.** Docstring promises it; code just clamps `alphas_cumprod` to a *minimum* of 1e-4 (nonzero terminal SNR). The model never trains on the pure-noise state it starts sampling from. | `src/dimba/diffusion/schedules.py:9-12,50` | High |
| 3 | **MSE-on-embeddings only.** No cross-entropy / rounding term anchoring embeddings to tokens. This is the documented "embedding collapse" failure mode for continuous text diffusion. | `src/dimba/training/trainer.py` (loss) | High |
| 4 | **Backbone is causal (unidirectional).** Mamba is left-to-right by default, but non-autoregressive denoising needs each position to see the *whole* noisy sequence → use bidirectional scans. Likely a real quality ceiling. | `src/dimba/models/denoiser.py`, `simple_mamba.py` | High |
| 5 | **It's Mamba-1, not Mamba-2.** `from mamba_ssm import Mamba` is the v1 API despite "Mamba2" naming everywhere. | `src/dimba/models/denoiser.py:65` | Medium |
| 6 | **`SimpleMamba2` is a Python for-loop scan.** O(L) sequential; the dominant CPU/MPS bottleneck. | `src/dimba/models/simple_mamba.py` | Medium (perf) |
| 7 | **DDIM sampler math is non-standard; "CDLM" is a homegrown loss**, not the real Consistency-Models recipe — won't yield true few-step generation. | `src/dimba/diffusion/sampling.py`, `training/trainer.py:18` | Medium |
| 8 | **No self-conditioning, no classifier-free guidance.** Both are near-mandatory for competitive conditional text diffusion. | model + sampling | High (missed upside) |
| 9 | **GRPO reward is a token-overlap heuristic** (`0.7·F1 + 0.3·bigram`) → teaches copying, not quality. DPO is the right tool for the preference-pair data. | `scripts/finetuning/finetune_grpo.py` | High |
| 10 | **No CI, sparse tests (~1.1k LOC), no real benchmarks**, no `mlx` code (MPS ≠ MLX). | repo-wide | Medium |

---

## Strategy: one backbone, two diffusion paradigms

Keep the Mamba backbone, conditioning, training loop, and finetuning suite. Abstract the **corruption process** and **loss head** so DIMBA supports two modes behind one API:

- **Track A — Continuous (current), but *fixed*.** Apply the known tricks (self-conditioning, CFG, CE/rounding term, min-SNR, zero-terminal-SNR, bidirectional). Differentiator: gradient/classifier guidance and fine-grained controllability. Research shows this *can* rival discrete (LangFlow 2026) but only with all the tricks.
- **Track B — Discrete / masked (the scaling bet).** Swap Gaussian noise → absorbing `[MASK]` corruption and MSE → masked cross-entropy (MDLM/LLaDA recipe). Every diffusion LM that has scaled or shipped (LLaDA-8B, Mercury, Gemini Diffusion) is discrete/masked; the compute-gap-to-autoregressive is ~16× (masked) vs ~64× (continuous).

A key convergence: the **fix for the conditioning leak is the same in both tracks** — keep the prompt as *clean, unmasked context* and only noise/mask the *response*, computing loss on the response only. This is the standard conditional recipe in SSD-LM (continuous) and LLaDA (masked).

---

## Phase 0 — Make it measurable & honest (do first; ~days)

Goal: a benchmark harness and CI so every later change is judged on numbers, not vibes.

- **Eval harness** (`scripts/benchmark.py`): wall-clock tokens/sec at fixed quality, NFE (network evals), and quality (validation loss, generative perplexity via a held-out scorer, plus a tiny task like GSM8K-subset once instruction-tuned). Reuse `src/dimba/evaluation/metrics.py`.
- **Tiny end-to-end smoke train** on `tinyshakespeare`/`wikitext-2` with `SimpleMamba2` (CPU/MPS) that asserts loss decreases — a regression guardrail.
- **CI** (`.github/workflows/ci.yml`): run `pytest`, `black --check`, `mypy` on PRs (CPU only).
- **Fix the docstrings that lie** (schedule, "Mamba-2") so the repo states what it actually does.
- **Decide the metric of record** for "ultra-fast inference" so the paper's headline claim becomes testable.

## Phase 1 — Correctness fixes (high impact, mostly cheap)

These make the existing continuous model *sound*. Order matters: 1.1 unblocks CFG.

- **1.1 Fix conditioning (the leak).** Change `DIMBA.forward` to take `prompt_ids` and `target_ids` separately; condition on the prompt only; noise + compute loss on the response span only (prompt span kept clean). Update `encode_prompt`, the trainer, and the SFT/GRPO data path (response masking already exists via `ignore_index=-100`). Files: `models/diffusion.py`, `training/trainer.py`, `scripts/finetuning/*`. *This is the single most important fix.*
- **1.2 Zero-terminal-SNR.** Implement the Lin et al. (2023) rescale of `alphas_cumprod` so terminal SNR = 0, and make the sampler start at the true terminal step. File: `diffusion/schedules.py`, `diffusion/sampling.py`.
- **1.3 Bidirectional Mamba.** Add forward+backward scans with separate SSM params (Vim/Vision-Mamba recipe), summed/concatenated. File: `models/denoiser.py`, `models/simple_mamba.py`. Gate behind a `bidirectional=True` config so checkpoints stay comparable.
- **1.4 Mamba-1 → Mamba-2.** Switch to the `Mamba2` API where kernels are available; keep the simple fallback. File: `models/denoiser.py`.
- **1.5 Fix DDIM.** Replace the non-standard update with the canonical DDIM step; verify 1000→~50 steps holds quality on the Phase-0 harness. File: `diffusion/sampling.py`.

## Phase 2 — High-impact research upgrades (cheap, large quality wins)

- **2.1 Self-conditioning** (Analog Bits / SED — *SED is literally DIMBA's setup*). 50%-of-steps double-forward; widen denoiser input proj to take the previous x̂₀; carry x̂₀ across sampling steps. Highest single ROI. Files: `models/denoiser.py`, `models/diffusion.py`, `diffusion/sampling.py`.
- **2.2 Classifier-free guidance** (needs 1.1 first). Drop conditioning 10–20% of the time in training (learned null embedding); at sampling combine `pred_uncond + w·(pred_cond − pred_uncond)`. Files: training + sampling.
- **2.3 CE / rounding term** (Diffusion-LM). Add `−log p(token | x̂₀)` over the embedding table to the MSE loss; anchors embeddings, gives a real likelihood, curbs collapse. Also add the **clamping trick** at sampling (snap x̂₀ to nearest real embedding). Files: `training/trainer.py`, `diffusion/sampling.py`.
- **2.4 Min-SNR-γ loss weighting** (γ=5; for x0-pred the weight is `min(SNR,γ)`). ~3.4× faster convergence, ~5 lines. File: `training/trainer.py`.
- **2.5 v-prediction** (optional, pairs with 1.2; prerequisite for good distillation). File: `training/trainer.py`, model + sampler.

## Phase 3 — Discrete / masked diffusion mode (the scaling bet)

- **3.1 Corruption + loss abstraction.** Introduce a `CorruptionProcess` interface; implement `GaussianEmbeddingDiffusion` (current) and `AbsorbingMaskDiffusion` (new). Loss head becomes pluggable (MSE+CE vs masked-CE). Files: new `src/dimba/diffusion/corruption.py`, refactor `models/diffusion.py`, `training/trainer.py`.
- **3.2 MDLM/LLaDA training**: mask schedule, masked cross-entropy (MDLM's Rao-Blackwellized NELBO), prompt-unmasked conditioning. Reference: MDLM, LLaDA.
- **3.3 Confidence-based remasking sampler** (LLaDA-style low-confidence remasking; optionally ReMDM). File: `diffusion/sampling.py`.
- **3.4 Benchmark A vs B** on the Phase-0 harness; let data pick the default track.

## Phase 4 — Post-training done right

- **4.1 DPO** for the existing preference pairs (replaces the token-overlap reward as the default). New `scripts/finetuning/finetune_dpo.py`; reuse the LoRA/Q-LoRA plumbing. Use the diffusion-correct surrogate: DPO on the **ELBO** difference (Diffusion-DPO style), with **VRPO** variance reduction (antithetic sampling, MC-budget allocation) per LLaDA 1.5.
- **4.2 Keep GRPO, fix the reward.** Make the reward pluggable; default to **verifiable rewards** (exact-match for math, unit-tests for code) or a small **reward model**, not token overlap. For the diffusion log-prob, use the diffu-GRPO one-step surrogate (fast, biased) or GDPO/SPG (lower bias) — selectable. Files: `scripts/finetuning/finetune_grpo.py`.
- **4.3 PEFT upgrade**: move custom LoRA to (or align with) `peft`; add **DoRA**; document correct Mamba target modules (`in_proj`, `x_proj`, `dt_proj`, `out_proj`).

## Phase 5 — Performance (back the "ultra-fast" claim)

- **5.1 CUDA quick wins (hours):** wire the official `mamba-ssm` + `causal-conv1d` kernels (already optional), and `torch.compile(denoiser)`. Expect ~10–25× over the Python-loop scan on real models. Files: `models/denoiser.py`.
- **5.2 Kill the Python-loop scan** on CPU/MPS with a vectorized associative (chunked) scan. File: `models/simple_mamba.py`.
- **5.3 Few-step generation:** proper multistep (2–8 step) consistency distillation from a solid teacher (after 1.2/2.5), in VAE-latent if used (LCM-style). Replaces the homegrown "CDLM". File: `training/trainer.py`, `diffusion/sampling.py`.
- **5.4 MLX backend (Mac):** a separate MLX port of the denoiser + sampling for Apple Silicon (no MLX parallel-scan primitive yet → sequential scan acceptable initially), with a `safetensors` checkpoint bridge to/from the PyTorch reference. Keep PyTorch as the source of truth. New `src/dimba/backends/mlx/`.
- **5.5 (later) Block / semi-autoregressive decoding** (BD3-LM) for KV-cache-style reuse + arbitrary-length output.

## Phase 6 — Engineering & polish

- Expand tests (sampling correctness, schedule properties, conditioning shapes, DPO loss); add property tests for the corruption processes.
- `console_scripts` entry points; `CHANGELOG.md`; docs build; pin a lockfile.
- README: replace aspirational claims with measured numbers from Phase 0.

---

## On `shard` (krish1905/shard)

KV-cache compression for **autoregressive Transformer** inference (PyTorch + Triton, CUDA; ~10–11× KV memory at long context, decode ~0.5× speed). **Not applicable to DIMBA's core** (non-autoregressive, Mamba has no attention KV-cache, CUDA/Triton not MLX). Revisit only if a Transformer/hybrid or AR-scoring path is added (see Phase 5.5 / cross-attention conditioning).

## The other "Dimba"

There is a real 2024 paper named **"Dimba: Transformer-Mamba Diffusion Models"** (Fei et al., text-to-image) that alternates Mamba and attention blocks with cross-attention conditioning. Two takeaways: (a) name collision to be aware of for branding/SEO; (b) their cross-attention-to-prompt conditioning is a strong alternative to the current FiLM-on-summed-vectors — a candidate experiment in Phase 1/2.

---

## Sequencing & risk

```
Phase 0 ──▶ Phase 1 ──▶ Phase 2 ──▶ benchmark ─┬─▶ Phase 3 (discrete) ─┐
                                                └─▶ Phase 5 (perf)      ├─▶ Phase 4 (post-train) ─▶ Phase 6
```

- **Independent / parallelizable now:** Phase 0 (CI+bench), Phase 4.1 DPO (new file), Phase 5.1 CUDA wins, Phase 6 tests.
- **Must be sequential (all touch `diffusion.py`/`denoiser.py`):** 1.1 → 1.3/1.4 → 2.1 → 2.2. Do these on one branch, in order, with the Phase-0 harness as the gate.
- **Biggest risk:** changing `forward()` (1.1) ripples into the trainer, finetuning scripts, and checkpoint format — land it behind tests first, keep old checkpoints loadable via a shim.
- **Compute reality:** validate everything at small scale (char-level / wikitext-2, <100M params) on CPU/MPS/Mac before spending real GPU credits.

## Key references

- Diffusion-LM (Li 2022) · SED self-conditioning (Strudel 2022, arXiv:2211.04236) · Analog Bits (Chen 2022) · CDCD (Dieleman 2022)
- Classifier-free guidance (Ho & Salimans 2022) · zero-terminal-SNR (Lin 2023, arXiv:2305.08891) · Min-SNR-γ (Hang 2023, arXiv:2303.09556) · v-prediction (Salimans & Ho 2022) · EDM (Karras 2022)
- SEDD (Lou 2024) · MDLM (Sahoo 2024, arXiv:2406.07524) · LLaDA (2025, arXiv:2502.09992) · LLaDA 1.5 / VRPO (arXiv:2505.19223) · Block Diffusion BD3-LM (arXiv:2503.09573) · Mercury (arXiv:2506.17298)
- Vision Mamba / Vim (arXiv:2401.09417) · Dimba: Transformer-Mamba (Fei 2024, arXiv:2406.01159) · DiffuSSM (arXiv:2311.18257)
- DPO (Rafailov 2023, arXiv:2305.18290) · Diffusion-DPO (Wallace 2023, arXiv:2311.12908) · d1/diffu-GRPO (arXiv:2504.12216) · GDPO (arXiv:2510.08554) · SPG (arXiv:2510.09541) · GRPO/DeepSeekMath (arXiv:2402.03300)
- Consistency Models (Song 2023) · Multistep CM (arXiv:2403.06807) · LCM (arXiv:2310.04378)
- PyTorch Mamba2 kernel fusion (pytorch.org/blog) · MLX (ml-explore) · mamba.py MLX port (alxndrTL/mamba.py)
