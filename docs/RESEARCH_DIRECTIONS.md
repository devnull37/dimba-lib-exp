# DIMBA Research Directions

> Status: **research agenda — everything below is experimental and unvalidated.**
> Author: SA-5 (innovation), 2026-05-27. Audience: DimbaLabs research.
> Companion to `docs/IMPROVEMENT_PLAN.md` (which fixes known correctness issues).
> This document proposes *new* DIMBA-specific research, not bug fixes.

## Framing: DIMBA is a latent diffusion language model

DIMBA runs **continuous Gaussian diffusion in a learned latent space** and denoises
with a **bidirectional Mamba** backbone. Concretely (see `src/dimba/models/diffusion.py`):

- A token sequence is embedded (`TokenEmbedding`, `d_model`), then **encoded into a
  latent** by a learned projector — either a deterministic `LatentProjector` or a
  `TokenVAE` (`src/dimba/models/vae.py`). Raw-embedding diffusion (`latent_diffusion=False`)
  is the **degenerate case** where the latent equals the embedding.
- Forward diffusion adds Gaussian noise to the latent `z_0` per the
  `CosineNoiseSchedule` (`src/dimba/diffusion/schedules.py`), now with **zero-terminal-SNR**.
- `Mamba2Denoiser` (`src/dimba/models/denoiser.py`) predicts the **clean latent**
  (predict-`x0`) conditioned on the prompt (FiLM/additive) and a timestep embedding.
  Blocks are **bidirectional** (forward + backward scans, separate SSM params).
- `decode_latent` maps the denoised latent back to embedding space; `DenoisingHead`
  projects to vocab logits. Sampling lives in `src/dimba/diffusion/sampling.py`
  (`sample_from_model`, `DDIMSampler`).

A second forward process — **discrete absorbing-`[MASK]`** diffusion — is being built in
`src/dimba/diffusion/corruption.py` (`GaussianEmbeddingCorruption`, `AbsorbingMaskCorruption`,
`HybridCorruption`) with a model-agnostic iterative decoder in
`src/dimba/diffusion/masked_sampling.py`. Several directions below sit at the
**latent-continuous ↔ discrete-masked** boundary, which is exactly where DIMBA is
architecturally distinctive (a *latent* diffusion text model with an *SSM* denoiser —
not a Transformer, not pixel/embedding-space).

Two structural facts drive most of the novelty here:

1. **The denoiser is an SSM, not attention.** This changes the cost model: there is a
   *recurrent state* to exploit (Directions 3, 7) and no KV-cache to compress.
2. **Diffusion is in a learned latent.** The VAE/projector is a first-class object we can
   quantize (Direction 5), regularize for consistency (Direction 6), or shape so that the
   *continuum* between noise and mask is well-defined (Direction 1).

Each direction lists: **(a)** the idea, **(b)** why it could win, **(c)** an implementation
sketch against real files/classes, **(d)** a cheap CPU-validatable experiment, **(e)**
risks/unknowns, **(f)** references.

---

## Direction 1 — Hybrid noisy-masked latent diffusion (a learned continuum)

**(a) Idea.** Treat "continuous Gaussian latent diffusion" and "discrete absorbing-`[MASK]`
diffusion" not as two separate tracks but as the **endpoints of one corruption family**,
and train a *single* DIMBA denoiser to span them via a mixing coefficient `λ ∈ [0, 1]`.
`HybridCorruption` in `src/dimba/diffusion/corruption.py` already implements the per-token
Bernoulli mixture (`mask_weight`); the research question is whether `λ` should be a **learned,
per-token, SNR-dependent gate** rather than a fixed hyperparameter, and whether annealing `λ`
over a sampling trajectory (mask-first → denoise-latent-last) beats either pure mode.

**(b) Why it could win.** Discrete masked diffusion has *scaled* (LLaDA, Mercury) because the
absorbing state gives a clean categorical likelihood and avoids embedding collapse; continuous
latent diffusion offers *fine-grained, gradient-based control* and smooth interpolation but is
finicky to train. A hybrid lets early reverse steps **commit easy tokens discretely** (cheap,
high-confidence, like MaskGIT) while **hard/ambiguous positions stay in the continuous latent
channel** where the model can move them gradually before committing. The SNR-dependent gate is
the novel part: at high noise, mask (discrete) is the better corruption; at low noise, small
Gaussian latent perturbations refine. This is a *DIMBA-native* unification because the latent
space is exactly where a soft "partially-masked" representation can live.

**(c) Implementation sketch.**
- Reuse `HybridCorruption(mask_token_id, alphas_cumprod, embed_fn, mask_weight, schedule)`.
  Make `mask_weight` a callable `mask_weight_fn(t) -> float` so it can anneal with the shared
  timestep `t`; the class already exposes `_t_to_index` and `_absorbing.mask_prob`.
- Add a tiny **gate head** on top of `Mamba2Denoiser` output: a `nn.Linear(d_latent, 1)` →
  sigmoid per position predicting "is this token ready to commit discretely?". Train it with
  the confidence signal from `masked_diffusion_sample` (the softmax max-prob).
- Sampling: alternate one `masked_diffusion_sample`-style commit step (reuse the
  `_unmask_count_schedule` + top-k logic from `src/dimba/diffusion/masked_sampling.py`) with
  one continuous DDIM latent step (`DDIMSampler.sample` body in `sampling.py`) on the
  still-uncommitted positions. The denoiser is called once per step and feeds both heads
  (`DenoisingHead` for logits, `decode_latent` for the latent residual).

**(d) Cheap CPU experiment.** No training. (i) Construct `HybridCorruption` with `mask_weight`
∈ {0, 0.25, 0.5, 0.75, 1.0} on tiny tensors (`B=8, L=32, d=8`) and verify the loss is finite and
the discrete/continuous channel masks partition positions (already covered by
`tests/test_corruption.py::TestHybridCorruption`). (ii) **Interpolation sanity check**: with a
*randomly initialized* DIMBA (`use_simple_mamba=True`, tiny config) confirm that as `λ→1` the
per-token error distribution shifts from MSE-dominated to CE-dominated, and that the combined
`loss` is continuous in `λ` (monotone-ish). This validates the *continuum* claim mechanically
before any training. Run in `python -c` with `torch` CPU.

**(e) Risks/unknowns.** (1) Two heads sharing one backbone may **interfere** (the categorical
head wants logits, the regression head wants smooth embeddings); needs a representation-sharing
ablation. (2) The "right" `λ(t)` schedule is unknown and may be task-dependent. (3) The
hybrid's marginal forward process is not a clean known diffusion → the ELBO is only a *bound on
a bound*; report it as a training objective, not a likelihood. (4) Decoding order interacts with
Mamba's bidirectionality (committing tokens changes the state both scans see).

**(f) References.** MDLM (Sahoo et al., 2024, arXiv:2406.07524); LLaDA (Nie et al., 2025,
arXiv:2502.09992); MaskGIT (Chang et al., 2022, arXiv:2202.04200); CDCD continuous-discrete
(Dieleman et al., 2022, arXiv:2211.15089); the repo's own `corruption.py` `HybridCorruption`.

---

## Direction 2 — ELBO / score-based self-reranking of K parallel samples (**implemented**)

**(a) Idea.** Non-autoregressive diffusion generates all tokens in parallel, so any *single*
sample is often locally inconsistent. Draw **K** independent samples and keep the one the model
scores best under its **own** training objective — a negative Monte-Carlo estimate of the
diffusion denoising error (an ELBO proxy). This is implemented in
`src/dimba/diffusion/rerank.py` (`rerank_candidates`, `diffusion_elbo_score`, `best_of_k`).

**(b) Why it could win.** Best-of-K is the single cheapest quality lever for parallel decoders:
it needs **no training**, parallelizes trivially (K independent generations), and the scorer is
*free* because DIMBA already computes the denoising MSE during training. For diffusion LMs the
sample-to-sample quality variance is high (the reverse SDE is stochastic), so even K=4–8 should
move quality measurably. Because DIMBA is *latent* diffusion, the ELBO proxy is naturally a
**latent-space reconstruction error**, which is exactly the quantity the denoiser is optimized
for — the scorer is perfectly aligned with the model.

**(c) Implementation sketch.** Already done. The contract:
- `diffusion_elbo_score(model_forward, input_ids, schedule_alphas_cumprod, num_mc=8, weighting=...)`
  samples `num_mc` timesteps, has the callable noise+denoise, and returns `−mean MSE`
  (higher = better). The `model_forward(input_ids, t)` callable returns either
  `(x0_pred, x0_target)` *or* a scalar MSE, so it is decoupled from the refactored core model.
- To wire DIMBA in, define:
  ```python
  def model_forward(input_ids, t):
      x0 = model.token_embed(input_ids)
      z0 = model.encode_latent(x0)
      z_t, _ = model.noise_schedule.add_noise(z0, t)
      cond = model.project_conditioning(model.encode_prompt(prompt_ids))
      z_pred = model.denoise_step(z_t, t, cond)   # predict-x0 in latent space
      return z_pred, z0
  ```
- `best_of_k(generate_fn, score_fn, k)` runs the existing `sample_from_model` `k` times with
  different seeds and returns the best. For the **masked** track, score with a log-likelihood
  via `sequence_logprob_score` instead of the MSE proxy.

**(d) Cheap CPU experiment.** Covered by `tests/test_rerank.py`: a toy `score_fn` where one
candidate is unambiguously best, `best_of_k` returns the max, and `diffusion_elbo_score` returns
finite scalars and ranks a near-perfect denoiser above a random one. Next step (still CPU, tiny
model): generate K=8 from a randomly-initialized tiny DIMBA, confirm scores have non-trivial
*spread* (std > 0) and that the argmax is stable across `num_mc` seeds when `shared_timesteps=True`.

**(e) Risks/unknowns.** (1) The score is a **proxy, not the true NELBO** — biased by the chosen
weighting (`uniform` vs `snr`) and Monte-Carlo variance; only *relative* scores matter for
ranking. (2) **Latent-vs-token gap**: the score lives in latent space and ignores the discrete
rounding term, so it can prefer a sequence that denoises cleanly but argmax-decodes differently —
mitigate by adding a CE/rounding term to `model_forward`. (3) On an *untrained* model the score is
near-random; the real payoff needs a trained checkpoint. (4) K× inference cost (embarrassingly
parallel, but real).

**(f) References.** Best-of-N / reranking is folklore; diffusion ELBO weighting from VDM (Kingma
et al., 2021, arXiv:2107.00630) and Min-SNR (Hang et al., 2023, arXiv:2303.09556); MBR-style
self-consistency for NAR decoding (Kumar & Byrne, 2004; recent diffusion-LM use). Module docstring
in `rerank.py` documents the approximation and its bias in full.

---

## Direction 3 — SSM recurrent-state caching across adjacent diffusion steps

**(a) Idea.** Across two adjacent reverse-diffusion steps the noisy latent `z_t` changes only
slightly (especially at low noise / with few-step samplers). A Mamba block's output is a function
of its **recurrent SSM state**; if the input barely changed, the state barely changed. **Cache the
per-block SSM state** (and the short-conv ring buffer) from step `t` and **reuse/refresh** it at
step `t−Δ` instead of recomputing the full scan — i.e. *feature caching* for an SSM diffusion
denoiser, analogous to DeepCache/∆-DiT for Transformer diffusion, but exploiting Mamba's state
rather than attention activations.

**(b) Why it could win.** Diffusion's dominant cost is the **number of full denoiser evaluations
(NFE)**. Transformer diffusion accelerators cache attention/feature maps; DIMBA has *no attention*
but *does* have a compact recurrent state — a structurally different and potentially cheaper thing
to cache (state is `O(d_state · d_model)`, far smaller than full activations). If `z_t` is nearly
unchanged on a subset of positions (the ones already "resolved"), recomputing their contribution to
the scan is wasted work. This is a **uniquely-SSM** acceleration that a Transformer DIMBA could not do.

**(c) Implementation sketch.**
- The pure-PyTorch fallback `SimpleMamba2` (`src/dimba/models/simple_mamba.py`) is a sequential scan
  — the natural place to prototype, since it explicitly materializes a state. Add an optional
  `(state_in, conv_buffer_in) -> (y, state_out, conv_buffer_out)` interface and a per-step cache
  keyed by block index, owned by the sampler (do **not** edit the model's forward signature; wrap it).
- In `DDIMSampler.sample` (`src/dimba/diffusion/sampling.py`), maintain a `cache` dict and a
  staleness criterion: refresh the cache (full recompute) every `R` steps or when
  `‖z_t − z_{cached}‖ / ‖z_t‖ > τ`; otherwise reuse the cached state, optionally applying a
  cheap linear correction. The real Mamba/Mamba-2 kernels already expose stepping state
  (`InferenceParams`) — the same wrapper applies when `HAS_MAMBA_SSM`.
- A weaker but trivially-safe variant: **block-skip caching** — skip recomputation of the *last*
  `k` denoiser blocks on `1−p` of steps and reuse their previous residual (since deep blocks change
  slowest). No state surgery; just cache `Mamba2Block` outputs.

**(d) Cheap CPU experiment.** No training. (i) Run a tiny randomly-initialized `Mamba2Denoiser`
(`use_simple_mamba=True`, `d_model=8, num_layers=2, L=16`) on a sequence of slightly-perturbed inputs
`z, z+εδ` and measure **how slowly the per-block output changes vs `ε`** (Lipschitz-in-input curve).
This quantifies the cacheability headroom: if outputs change <1% for the `ε` typical between adjacent
DDIM steps, caching is promising. (ii) Implement block-skip caching in a sampler *wrapper* and verify
that with skip-probability 0 it is bit-identical to the baseline (correctness), and measure NFE saved
vs latent drift at skip 0.3/0.5 on tiny tensors.

**(e) Risks/unknowns.** (1) **Bidirectional scans** complicate state reuse: the backward scan's
state for position `i` depends on positions `>i`, so committing/changing a later token invalidates
earlier backward states — caching may only be valid for the forward scan or for *suffix-stable*
regions. (2) Error accumulates across reused steps → needs a refresh schedule; quality/NFE is a
Pareto curve, not free. (3) The real CUDA Mamba kernels don't expose intermediate per-token states
cheaply; the win may be CPU/MPS-specific or require a custom kernel. (4) Interaction with
self-conditioning (Direction-2 of the IMPROVEMENT_PLAN) and CFG (which doubles NFE).

**(f) References.** DeepCache (Ma et al., 2023, arXiv:2312.00858); ∆-DiT / feature caching for DiT
(arXiv:2406.01125); Faster Diffusion / cache-me-if-you-can (arXiv:2312.09608); Mamba inference-state
stepping (Gu & Dao, 2023, arXiv:2312.00752); applies to `SimpleMamba2` and `Mamba2Block` here.

---

## Direction 4 — Guidance distillation: "free" classifier-free guidance in one pass

**(a) Idea.** Classifier-free guidance (CFG) doubles inference cost: every step runs the denoiser
**twice** (conditional + unconditional) and combines `pred_cond + w·(pred_cond − pred_uncond)`.
**Distill** that two-pass, fixed-`w` behavior into a **single forward pass** of a student DIMBA that
takes `w` as an extra conditioning input — so guided sampling costs 1 NFE instead of 2.

**(b) Why it could win.** CFG is near-mandatory for competitive *conditional* text diffusion
(IMPROVEMENT_PLAN Phase 2.2), but it halves throughput — directly undercutting DIMBA's "fast
inference" identity. Guidance distillation is a proven win in image diffusion (Meng et al.) and maps
cleanly onto DIMBA's existing conditioning machinery (`TimestepEmbedding` already injects a scalar via
sinusoidal embedding — a `w`-embedding is the same trick). Halving NFE on the *conditional* path is a
2× inference speedup with (empirically, in vision) negligible quality loss.

**(c) Implementation sketch.**
- Prereq: CFG training (drop conditioning `p≈0.15`, learned null embedding). The null embedding is a
  single `nn.Parameter(d_prompt)` substituted for `cond` in `DIMBA.encode_prompt`/`project_conditioning`.
- **Guidance embedding.** Add a `GuidanceEmbedding(nn.Module)` mirroring `TimestepEmbedding`
  (`src/dimba/models/embeddings.py`): sinusoidal-encode `w`, MLP to `cond_dim`, **add** to the
  `combined_cond` inside `Mamba2Denoiser.forward` (same place `time_proj` output is added). This is a
  new module, not an edit to the denoiser's math contract.
- **Distillation loss** (trainer-side): teacher = frozen CFG two-pass DIMBA at sampled `w`; student =
  one-pass DIMBA conditioned on `(t, w)`. Minimize `‖student(z_t, t, w, cond) − teacher_cfg(z_t, t, w, cond)‖²`
  in **latent space** (predict-`x0`), `w ~ U[w_min, w_max]`. Reuse `add_noise` for `z_t`.

**(d) Cheap CPU experiment.** No training of the student, but validate the *mechanism*: (i) instantiate
a tiny DIMBA, implement the two-pass CFG combine on tiny tensors, and confirm the guided `x0`-prediction
is finite and reduces to the conditional prediction at `w=0` and amplifies the cond−uncond delta linearly
in `w`. (ii) Build `GuidanceEmbedding`, confirm it produces a `[B, cond_dim]` vector that, when added to
`combined_cond`, changes the denoiser output monotonically with `w` (a controllability sanity check). Both
are `python -c` smoke checks on random weights.

**(e) Risks/unknowns.** (1) Distillation quality depends on a **good teacher** → gated on CFG training
landing first. (2) Range of `w` to distill is a hyperparameter; too wide hurts fidelity. (3) Text CFG is
less studied than image CFG; the cond/uncond gap in *latent* space may behave differently than in pixel
space. (4) Adding `w`-conditioning slightly grows the model and could interact with self-conditioning.

**(f) References.** CFG (Ho & Salimans, 2022, arXiv:2207.12598); guidance distillation (Meng et al., 2023,
"On Distillation of Guided Diffusion Models", arXiv:2210.03142); maps onto `embeddings.TimestepEmbedding`
and `denoiser.Mamba2Denoiser.forward` conditioning sum.

---

## Direction 5 — VQ discrete-latent masked diffusion reusing TokenVAE (MaskGIT-for-text in DIMBA's latent)

**(a) Idea.** Add a **vector-quantization (VQ)** bottleneck to the existing `TokenVAE`
(`src/dimba/models/vae.py`) so each token-position latent maps to a **discrete codebook index**, then
run **MaskGIT-style absorbing-`[MASK]` diffusion over the *code* indices** (not over the raw vocabulary).
This is a *latent* discrete diffusion: the model denoises a grid of codebook IDs, and the VQ decoder maps
the final code grid back to embeddings → tokens.

**(b) Why it could win.** It marries the two things that have actually worked: (i) discrete/absorbing
diffusion (scales, clean likelihood) and (ii) a *learned, compressed latent* (DIMBA's VAE). A VQ latent
gives a **smaller, denoised-friendly discrete space** than the full vocab (codebook of, say, 1–4k vs vocab
of 32k+), shorter effective sequences, and decouples "semantic planning" (over codes) from "surface
realization" (VQ decoder). DIMBA is one of the few text models already carrying a latent autoencoder, so a
VQ variant is a small delta with a potentially large payoff — a genuinely novel "latent MaskGIT for text on
an SSM backbone".

**(c) Implementation sketch.**
- Subclass `TokenVAE` → `VQTokenVAE` adding a codebook `nn.Embedding(num_codes, latent_dim)`, nearest-code
  lookup with straight-through gradients, and a commitment loss (VQ-VAE). Keep `encode`/`decode` signatures
  so it drops into `TokenVAEWithDeterministicFallback` and `DIMBA`'s `latent_projector` slot unchanged.
- The diffusion then operates on **code indices**, which is exactly the discrete-masked setting:
  reuse `AbsorbingMaskCorruption` (treating `num_codes` as the "vocab", with a dedicated `[MASK]` code) and
  the `masked_diffusion_sample` decoder from `src/dimba/diffusion/masked_sampling.py` — both are already
  model-agnostic via the `predict_logits(ids, t)` callable. The denoiser `Mamba2Denoiser` predicts a
  categorical over codes; a new tiny head maps `d_latent → num_codes`.
- Final decode: committed code grid → `VQTokenVAE.decode` → embeddings → `DenoisingHead` (or directly to
  tokens if codes are token-aligned).

**(d) Cheap CPU experiment.** No training. (i) Build `VQTokenVAE` (tiny: `num_codes=16, latent_dim=8`),
confirm encode→quantize→decode runs, gradients flow through the straight-through estimator (grad is finite
on the encoder), and the commitment loss is finite/positive. (ii) Feed code indices through
`AbsorbingMaskCorruption(mask_token_id=num_codes)` and `masked_diffusion_sample` with a toy
`predict_logits` over `num_codes+1` and assert the decoder ends fully unmasked (mirrors
`tests/test_corruption.py` patterns). This proves the *plumbing* end-to-end on CPU.

**(e) Risks/unknowns.** (1) **Codebook collapse** is the classic VQ failure (few codes used); needs EMA
codebook / commitment tuning / k-means init. (2) Two-stage training (VAE then diffusion) is heavier than
one-stage continuous. (3) Token→code alignment: if codes are *per token* the sequence length is unchanged
(no compression win); if *grouped*, you need a length model (see Direction 8). (4) Quantization caps the
achievable reconstruction → an upper bound on quality set by the VAE, not the diffuser.

**(f) References.** VQ-VAE (van den Oord et al., 2017, arXiv:1711.00937); MaskGIT (Chang et al., 2022,
arXiv:2202.04200); latent diffusion (Rombach et al., 2022, arXiv:2112.10752); discrete latent text diffusion
(e.g. DiffusionBERT lineage); reuses `vae.TokenVAE`, `corruption.AbsorbingMaskCorruption`,
`masked_sampling.masked_diffusion_sample`.

---

## Direction 6 — Self-conditioned latent consistency distillation for few-step generation

**(a) Idea.** Distill the multi-step DIMBA latent-diffusion sampler into a **2–8 step** sampler via a
proper **consistency / latent-consistency-model (LCM)** objective in the **VAE latent space**, coupled with
**self-conditioning** (feed the previous `x̂0` back in). The consistency property — *the model maps any point
on a trajectory to the same clean latent* — is enforced directly, replacing the repo's homegrown "CDLM" loss
(flagged as non-standard in IMPROVEMENT_PLAN finding #7).

**(b) Why it could win.** Few-step generation is the most credible path to DIMBA's "ultra-fast" claim: going
from ~50 NFE to ~4 NFE is a >10× inference speedup. Doing consistency distillation **in the learned latent**
(rather than embedding space) is the LCM insight — the latent is lower-dimensional and smoother, so the
consistency map is easier to learn. **Self-conditioning is nearly free** for DIMBA (SED is literally DIMBA's
setup — continuous diffusion over embeddings/latents) and is known to be the single highest-ROI quality add
for this regime; combining it with consistency distillation is the natural DIMBA-specific recipe.

**(c) Implementation sketch.**
- **Self-conditioning** first: widen the denoiser input projection to optionally concatenate a previous
  `x̂0` estimate (`Mamba2Denoiser` consumes `[B,L,d_model]`; add a `prev_x0` input projected and summed at
  the embedding). Carry `x̂0` across steps in `sampling.py` (50%-of-steps double-forward at train time).
- **Consistency loss** (trainer-side, new — not an edit to core math): teacher = the EMA of the model (or a
  pretrained multi-step DIMBA); for adjacent timesteps `t, t'` on a trajectory, minimize
  `d(f_θ(z_t, t), f_{θ⁻}(z_{t'}, t'))` in latent space, with `f` predicting `x̂0` via
  `schedule.predict_x0_from_*` (already in `schedules.py`). Use the existing `CosineNoiseSchedule` and
  `add_noise` to build trajectory points.
- **Few-step sampler**: a new function in `sampling.py` that takes 2–8 `timesteps`, calls `denoise_step`,
  decodes via `decode_latent`. Compose with **best-of-K** (Direction 2) since few-step samples are higher
  variance.

**(d) Cheap CPU experiment.** No training. (i) Verify the schedule's inversion identities hold on tiny
tensors: `predict_x0_from_v(velocity(x0,noise,t), ...) ≈ x0` and `predict_x0_from_noise(add_noise(x0,t))`
recovers `x0` — these are the math primitives consistency distillation relies on (`schedules.py`). (ii)
Implement the self-conditioning concat path on a tiny denoiser and confirm a forward pass with `prev_x0=0`
matches the no-self-cond baseline (backward-compatible), and `prev_x0=x̂0` changes the output. (iii) Write the
consistency loss on two trajectory points from a random model and assert it is finite and **zero when
`t==t'`** (the trivial consistency check). All `python -c` on CPU.

**(e) Risks/unknowns.** (1) Consistency distillation needs a **decent teacher**; on a random model it is
meaningless → gated on a trained checkpoint. (2) Distilling in latent space couples quality to VAE fidelity
(shared risk with Direction 5). (3) Self-conditioning adds a (50%-of-steps) training-time double-forward.
(4) The exact metric `d(·,·)` (LPIPS-analogue for text doesn't exist) — likely latent MSE + a CE/rounding
anchor; needs ablation.

**(f) References.** Consistency Models (Song et al., 2023, arXiv:2303.01469); LCM (Luo et al., 2023,
arXiv:2310.04378); Multistep Consistency (Heek et al., 2024, arXiv:2403.06807); self-conditioning / SED
(Strudel et al., 2022, arXiv:2211.04236) and Analog Bits (Chen et al., 2022, arXiv:2208.04202); replaces the
"CDLM" loss; uses `schedules.predict_x0_from_v/noise`, `sampling.denoise_step`, `decode_latent`.

---

## Direction 7 — Block / semi-autoregressive Mamba decoding (BD3-LM-style) with SSM-state reuse

**(a) Idea.** Decode the sequence in **blocks**: run parallel diffusion *within* a block while being
**autoregressive across blocks** (BD3-LM). Crucially, because the backbone is an **SSM**, the prefix's
Mamba **recurrent state can be carried forward** as the "context" for the next block — an SSM analogue of a
KV-cache — giving arbitrary-length generation with bounded per-block cost.

**(b) Why it could win.** Pure NAR diffusion fixes the sequence length up front and pays full cost for the
whole sequence each step; pure AR is slow and sequential. Block diffusion interpolates: it supports
**arbitrary-length** output and **reuses computation across blocks**. On an SSM this reuse is natural and
*cheap* — the forward scan's state at the block boundary **is** a sufficient statistic of the prefix
(unlike a Transformer, which must store/attend a growing KV-cache). This makes block-DIMBA a strong fit for
long generation and is a concrete way to deliver "arbitrary-length non-autoregressive output" while keeping
Mamba's O(1)-state decoding advantage. Listed as Phase 5.5 in IMPROVEMENT_PLAN; here it becomes SSM-specific.

**(c) Implementation sketch.**
- **Causal-across-blocks conditioning.** Generate block `b` conditioned on the *clean* committed blocks
  `<b`. The clean prefix is embedded and **its forward SSM state is computed once**; the current block's
  forward scan is *initialized* from that state (the backward scan stays block-local, or spans only the
  current block to preserve bidirectionality within the block).
- Reuse `masked_diffusion_sample` (or the continuous sampler) **per block** with `prompt_ids` = prefix; it
  already keeps the prompt fixed and only denoises the response — set `gen_len = block_size`, loop over blocks,
  append each committed block to the prompt. No core-model edit; a sampler-level loop in a new function.
- The SSM-state carry uses the same stepping interface introduced in Direction 3 (`SimpleMamba2` /
  `Mamba2Block` state passthrough), so Directions 3 and 7 share machinery.

**(d) Cheap CPU experiment.** No training. (i) Implement the block loop around `masked_diffusion_sample`
on a tiny `predict_logits` and verify: output length is `num_blocks · block_size`, each block ends fully
unmasked, and **earlier committed blocks are never modified** when later blocks decode (the AR-across-blocks
invariant). (ii) On a tiny `SimpleMamba2`, verify that running the forward scan on `[prefix ‖ block]` in one
shot vs. running `prefix` then continuing the scan on `block` **from the carried state** gives (near-)identical
outputs on the block positions — this validates that the SSM-state carry is a correct prefix substitute.

**(e) Risks/unknowns.** (1) **Bidirectionality vs causality** tension: full bidirectional scans over the
whole sequence break the across-block causal factorization; you likely keep the backward scan block-local,
which may reduce within-block quality. (2) Block-boundary artifacts / discontinuities. (3) Choosing block size
trades latency vs. global coherence. (4) Training must match inference (train with block-causal masking), so
this is *not* purely an inference change despite the cheap experiment.

**(f) References.** BD3-LM block diffusion (Arriola et al., 2025, arXiv:2503.09573); SSD-LM
semi-autoregressive (Han et al., 2022, arXiv:2210.17432); Mamba state stepping (Gu & Dao, 2023,
arXiv:2312.00752); uses `masked_sampling.masked_diffusion_sample`, `simple_mamba.SimpleMamba2`.

---

## Direction 8 — Learned length-prediction head for variable-length NAR output

**(a) Idea.** Non-autoregressive diffusion must **fix the output length before denoising**
(`sample_from_model` takes `seq_len` as an argument). Add a small **length-prediction head** conditioned on
the prompt that predicts a distribution over response lengths, so DIMBA generates **variable-length** output
without an external oracle — and can sample/rank multiple length hypotheses (pairs naturally with best-of-K).

**(b) Why it could win.** Length mis-specification is a known, concrete failure mode of NAR generation:
too short truncates, too long pads with filler/`[EOS]` spam. Every shipped masked-diffusion LM needs a length
story; making it **learned and prompt-conditioned** (rather than a fixed hyperparameter) is a small module
with outsized practical impact on output quality and a prerequisite for honest open-ended generation. It also
unlocks **length-conditioned best-of-K**: generate at the top-`m` predicted lengths and rerank with Direction 2.

**(c) Implementation sketch.**
- New `LengthPredictor(nn.Module)`: pool the prompt conditioning from `DIMBA.encode_prompt`
  (mean/attention-pool over `[B, prompt_len, d_prompt]`) → `nn.Linear(d_prompt, max_len)` → categorical over
  lengths (or a Poisson/`(μ, σ)` regression head). Trained with the gold response length (cross-entropy or NLL).
  Lives alongside the model; consumes only `encode_prompt` output, so it does not touch diffusion math.
- **Sampling integration** (`sampling.py` / `masked_sampling.py`): replace the hard `seq_len` arg with
  `seq_len = length_predictor(prompt_cond).sample()` (or top-`m` lengths). `masked_diffusion_sample` already
  takes `gen_len`; just feed the predicted length. For the continuous sampler, allocate the noise tensor at the
  predicted length.
- Combine with **best-of-K** (Direction 2): score each length hypothesis with `diffusion_elbo_score` /
  log-likelihood and keep the best — a principled, training-free length selector at inference.

**(d) Cheap CPU experiment.** No training. (i) Build `LengthPredictor`, run it on a tiny prompt-conditioning
tensor, confirm it emits a valid distribution over `[1, max_len]` (sums to 1, finite) and that argmax/sample
return integers in range. (ii) Wire its sampled length into `masked_diffusion_sample`'s `gen_len` with a toy
`predict_logits` and confirm the output length matches the prediction and is fully unmasked. (iii) Length-
conditioned best-of-K smoke: generate at 3 candidate lengths, score with a dummy `score_fn`, assert
`best_of_k` returns the highest-scoring length hypothesis. All CPU `python -c`.

**(e) Risks/unknowns.** (1) **Length-quality entanglement**: the ELBO proxy may systematically prefer
shorter sequences (fewer terms to get wrong) → needs a length-normalized score (the `rerank` module's
per-element-mean MSE partially addresses this, but verify). (2) Multi-modal length distributions (many valid
lengths) are hard for a single categorical. (3) Requires response-length labels in the data pipeline
(`src/dimba/data/`). (4) Interacts with block decoding (Direction 7), which sidesteps fixed length differently.

**(f) References.** NAT length prediction (Gu et al., 2018, "Non-Autoregressive NMT", arXiv:1711.02281);
length beam / `[LENGTH]` token in CMLM (Ghazvininejad et al., 2019, arXiv:1904.09324); SUNDAE/Diffusion-LM
length handling; pairs with `rerank.best_of_k` and `sampling.sample_from_model` / `masked_sampling`.

---

## Cross-cutting notes

- **Shared machinery.** Directions 3 and 7 both need an SSM-state passthrough on `SimpleMamba2` /
  `Mamba2Block`; build it once. Directions 2 and 8 share `rerank.best_of_k`. Directions 5 and 6 share the
  "quality is capped by VAE fidelity" risk and both motivate a stronger `TokenVAE`.
- **Validation discipline.** Every direction has a *mechanism* check that runs on CPU with random weights
  (shapes, finiteness, invariants, monotonicity) and a separate *payoff* claim that is honestly gated on a
  trained checkpoint. Do the mechanism checks now; do not claim quality wins without the Phase-0 benchmark
  harness (`docs/IMPROVEMENT_PLAN.md`).
- **Priority (highest ROI / lowest risk first).** (2) reranking — *done, free, parallel* → (6)
  self-conditioning half of the recipe — *near-free, highest single quality ROI* → (3)/(7) SSM-state reuse —
  *DIMBA-unique speed* → (4) guidance distillation — *2× speed, gated on CFG* → (1)/(5) hybrid & VQ latent —
  *higher-risk research bets* → (8) length head — *small, practical, do alongside any track*.

## Consolidated references

VDM ELBO weighting (Kingma 2021, 2107.00630) · Min-SNR-γ (Hang 2023, 2303.09556) · CFG (Ho & Salimans 2022,
2207.12598) · Guidance distillation (Meng 2023, 2210.03142) · Consistency Models (Song 2023, 2303.01469) ·
LCM (Luo 2023, 2310.04378) · Multistep CM (Heek 2024, 2403.06807) · Self-conditioning/SED (Strudel 2022,
2211.04236) · Analog Bits (Chen 2022, 2208.04202) · MDLM (Sahoo 2024, 2406.07524) · LLaDA (Nie 2025,
2502.09992) · MaskGIT (Chang 2022, 2202.04200) · VQ-VAE (van den Oord 2017, 1711.00937) · Latent Diffusion
(Rombach 2022, 2112.10752) · CDCD (Dieleman 2022, 2211.15089) · DeepCache (Ma 2023, 2312.00858) · ∆-DiT
(2406.01125) · Mamba (Gu & Dao 2023, 2312.00752) · Vision Mamba/Vim (2401.09417) · BD3-LM (Arriola 2025,
2503.09573) · SSD-LM (Han 2022, 2210.17432) · NAT length (Gu 2018, 1711.02281) · CMLM (Ghazvininejad 2019,
1904.09324).
