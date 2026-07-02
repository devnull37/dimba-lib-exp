# Root cause: budget28 run (28B tokens) produced an incoherent model

Date: 2 Jul 2026 (UAE). Run: distill→SFT→GRPO on 1×H100, preset `budget28`
(~28B tokens, ~$358). Pipeline completed mechanically; all checkpoints valid and
loadable. The final model does not produce coherent text.

> **UPDATE (2 Jul 2026, evening): the primary root cause was found and is NOT
> what the first analysis below concluded.** See "Definitive root cause (found
> by experiment)" at the end. In particular, "the model learned unigram
> statistics and stopped" is **refuted**: the budget28 model reconstructs
> *unseen* English text with 98–100% accuracy from up to 90% noise. The model
> learned strong conditional structure; it is blind only at exactly t=1.0 —
> the pure-noise starting point of every generation. The KD-off finding below
> remains true and still matters, but it is secondary.

## Symptom

All three checkpoints (`distill/final.pt`, `sft/final.pt`, `grpo/final.pt`)
emit individually-plausible English tokens with **no grammatical structure**
("word salad"). SFT/GRPO look slightly *more* degenerate than the distill base
(collapse toward "is/the/0"; GRPO adds digit-spam from math RL). No prompt is
answered; e.g. "What is 2 + 3?" gets unrelated tokens.

## Ruled out

1. **Checkpoint corruption** — strict load 0 missing / 0 unexpected vs embedded
   config (after the 49152→49154 block-CoT vocab override). No NaN/Inf.
2. **Wrong/undersized architecture at load** — verified d_model=576, 30 layers,
   weight-tied, matches config.
3. **Sampler/schedule mismatch** — the model was trained with
   `use_flow_matching=True`; tested BOTH `sample_from_model` (DDIM/DPM++, wrong
   family) AND the correct `sample_from_model_flow` (Euler n=20; Heun n=50;
   T∈{0.3, 0.8}; top_k=50). Word salad under every schedule. Not a decoding bug.
4. **SFT/GRPO malfunction** — SFT ran its full two-stage schedule (~13,960
   stage-1 + ~3,780 stage-2 steps, ~4h13m, stable loss ≈0.29–0.33). GRPO ran all
   2000 math steps, stable, kl<0.13 throughout. Both executed correctly — on a
   base they could not fix. The pipeline *logged its own preconditions*:
   - "SFT cannot fix an incoherent / undertrained base."
   - "GRPO cannot bootstrap signal from an incoherent policy."
   Both were violated; the warnings do not gate, so the run proceeded anyway.

## Evidence: the Stage-3 loss plateaued at ~1.6B of the 28B tokens

Stage-3 diffusion CE (batch 64 × seq 512 on H100):

| step (frozen /549,316) | loss |
|---|---|
| 50 | 13.50 |
| 250 | 1.57 |
| 50,000 (~1.6B tok) | **0.360** |
| 100,000 | 0.357 |
| 300,000 | 0.374 |
| 549,316 (end, 18B tok) | 0.377 |
| unfrozen 50,000 | 0.360 |
| unfrozen 305,176 (end, +10B tok) | 0.349 |

Loss reached ~0.35 within ~1.6B tokens and did not improve for the remaining
~26B. **Token budget was not the bottleneck; the model converged to a bad
optimum almost immediately.** (Loss floor ~0.09–0.10 was touched only in early
transients.) The output signature — correct marginal token distribution, no
conditional structure — matches a model that learned unigram statistics and
stopped.

## Root cause (two compounding design gaps)

### 1. The teacher was only connected for 1,000 steps out of ~855k

```
stage1 (matrix align → teacher attn):      500 steps
stage2 (hidden align → teacher residual):  500 steps
stage3 (ALL 28B tokens):                   kd_weight = 0.0   ← teacher unused
```

Stage 3 — 99.9% of compute — was **from-scratch diffusion pretraining** of a
135M model with no KD signal, from a warm init. That is not distillation; the
28B tokens carried none of SmolLM's competence. Diffusion LMs are substantially
more sample-hungry than AR models at equal size; 28B tokens from-scratch at
135M is far below what coherence requires under this objective.

### 2. Double distribution gap: AR→diffusion AND causal→bidirectional

The teacher (SmolLM-135M) is causal autoregressive; the student is a
bidirectional flow-matching denoiser. The config's own comment acknowledges the
signal dilution. Stage-1/2 alignment (1,000 steps total, stage2 ended at loss
0.556 — not obviously converged) is far too little bridge for both gaps at
once, and then the bridge was removed entirely for Stage 3.

## Why the babysitting metrics looked "healthy"

- Distill loss ~0.35 stable → read as converged-and-training; the *absolute*
  level of a min-SNR-weighted x0-prediction loss is not interpretable like AR
  CE, so "flat and low-ish" hid "flat because collapsed".
- GRPO flat reward 0.00 was attributed to 135M math weakness; it was actually
  the policy having no coherent behaviour to reward at all.
- **Lesson:** loss curves cannot substitute for a generation-based coherence
  gate between phases.

## Fixes for the next run (in order of importance)

1. **Keep KD on through Stage 3** (`kd_weight > 0`, e.g. anneal 1.0→0.3).
   Teacher forward on the same batch; distill logits (and optionally hidden
   states) into the denoiser's x0 prediction. This is the single change most
   likely to matter. Costs a teacher forward pass — at 135M this is cheap
   relative to the student's diffusion step.
2. **Make the coherence gate BLOCKING.** After Stage 3 (and again after SFT),
   run N fixed prompts through `sample_from_model_flow`; require a minimal
   grammaticality/perplexity bar (e.g. teacher-scored PPL of samples under a
   threshold) before the pipeline advances. The warnings already exist —
   convert them to hard asserts.
3. **Stop on plateau, don't burn the budget.** Early-stop Stage 3 when loss
   improvement < ε over a window (this run: would have saved ~26B tokens ≈ 93%
   of Stage-3 compute for reallocation or refund).
4. **Longer Stage-1/2 alignment** (≥5–10k steps each, until stage2 hidden-align
   loss actually flattens), since these 1,000 steps are the only bridge across
   the AR→diffusion gap.
5. **Consider staging the conversion:** distill SmolLM → *causal AR* Mamba-2
   first (pure MOHAWK, known-good regime, 3–5B tokens), verify coherence, THEN
   convert AR-Mamba → diffusion with the AR model as teacher. One gap at a
   time, each with a checkable intermediate.
6. **Cheap validation before any big spend:** the plateau shows up by ~1.6B
   tokens (~3–4 H100-hours). Any candidate fix can be validated with a ~1–2B
   token run + the blocking coherence gate before committing to a full budget.

## Definitive root cause (found by experiment, 2 Jul 2026 evening)

A chain of cheap controlled experiments (~$3 total, scratchpad scripts) located
the true failure. All on the exact production architecture (d_model=576, 30
layers, flow matching, logit-normal t, bf16).

### Experiment chain

1. **Overfit contract test** (`overfit_test.py`): train from scratch on 8 fixed
   sentences with the *identical* stage-3 loss path until memorized.
   Result: t=0 clean-pass reconstruction accuracy **1.000**, yet
   `sample_from_model_flow` produced unigram soup ("in in the the, and…").
   → the train→sample contract itself is broken; no token budget or KD signal
   could ever have produced coherent generations.
2. **Diffusion-LM clamp trick** (`clamp_sample_test.py`): snapping x0̂ to the
   nearest token embedding each ODE step (all-steps and late-steps variants,
   Euler/Heun). **Refuted** — still soup. Not a sampler-level fix.
3. **Partial-noise recovery ladder** (`partial_noise_test.py`) — the decisive
   experiment. Noise a known row to level t_start, integrate the flow ODE to 0:

   | t_start | 0.1 | 0.3 | 0.5 | 0.7 | 0.8 | 0.9 | **1.0** |
   |---|---|---|---|---|---|---|---|
   | recovery (overfit model) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **0.02** |
   | recovery (budget28, UNSEEN text) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.98–1.00 | **0.00–0.01** |

   A razor-sharp cliff at exactly t=1.0. The model is an excellent denoiser
   whenever *any* true signal exists and blind only at pure noise — which is
   where every real generation must start.

### Mechanism

Two compounding terms in `compute_dimba_losses`:

- **min-SNR weighting kills the high-t gradient.** `snr_flow = ((1−t)/t)²`
  clamped to [1e-3, 5]: at t=0.9 the diffusion loss weight is ~0.012, at t≈1 it
  is 1e-3. The model gets essentially no gradient pressure to make its x0̂
  *depend on the input noise* at high t, so it converges to a constant
  posterior-mean blend there.
- **The unweighted CE anchor actively teaches unigram output at high t.** CE is
  applied at full strength at every t; from pure noise its optimum is the
  marginal token distribution. The model is literally trained to emit
  "in/the/and/," when shown noise.

At sampling time the ODE starts at t=1.0 in exactly that regime: the first
steps write a noise-independent unigram-blend seed, and the (excellent)
denoiser then faithfully amplifies that garbage seed into polished soup. This
also explains why the stage-3 loss plateau was misread: the min-SNR-weighted
loss was dominated by well-trained mid/low-t regions and could not show that
the high-t region was dead.

### Fix (validated at overfit scale, `overfit_v2_fixedloss.py`)

Two lines in the objective, plus t-coverage:

1. `weight = clamp(snr_flow, min=0.5, max=5.0)` (was min=1e-3) — forces the
   model to learn noise-*dependent* prediction at high t.
2. `ce_loss = (ce_per_sample * (1 − t)).mean()` — the CE anchor fades at high
   noise instead of teaching unigram there.
3. 50/50 uniform/logit-normal timestep sampling (logit-normal alone
   undersamples t≈1).

Result: free generation from pure noise went from pure unigram soup to
locally-coherent memorized phrases. Remaining known gap: **global mode
commitment** — the sample stitches fragments of different memorized rows
(topic drift). Clamp trick and high-t-dense ODE schedules do not fix the
drift; it appears to need training-scale exposure and/or architectural work
(bidirectional Mamba's effective range through noise). For natural-language
(non-memorization) generation, "locally coherent with drift" is the normal
operating point of weak continuous text diffusion — categorically better than
budget28's output.

### Consequence for the budget28 checkpoint

The dead zone lives in the **weights** (trained under the broken weighting), so
no sampler patch can rescue `distill/final.pt` as-is. However, the ladder shows
the 28B tokens bought a genuinely strong denoiser for t ≤ 0.9 on unseen text.
A **repair finetune** — resume from `final.pt` with the fixed objective,
oversampling high t — only needs to teach the thin t∈(0.9, 1.0] slice and is
the cheapest plausible path to recovering the $358 investment (est. 1–4 H100
hours to first signal, vs 100+ hours to retrain).

## Artifacts

- Checkpoints: `checkpoints/{distill,sft,grpo}/final.pt` (+ `grpo/math/`).
- Full log: `~/dimba_train.log` (filter `grep -vE "httpx|HTTP Request|cas-bridge|examples/s"`).
- Inference test scripts (correct loaders, incl. flow sampler):
  scratchpad `infer_gpu_cmp.py`, `infer_flow.py`, `infer_cpu_multi.py`.
- Root-cause experiment scripts (scratchpad, 2 Jul): `overfit_test.py`,
  `clamp_sample_test.py`, `partial_noise_test.py`, `overfit_v2_fixedloss.py`,
  `dense_schedule_test.py`; logs `~/dimba_overfit*.log`; trained probes
  `overfit_model.pt`, `overfit_v2_model.pt` (scratchpad — ephemeral).
- HF backup repo: `devnull37/d1-135m-50b` (private).
