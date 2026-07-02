# Root cause: budget28 run (28B tokens) produced an incoherent model

Date: 2 Jul 2026 (UAE). Run: distill→SFT→GRPO on 1×H100, preset `budget28`
(~28B tokens, ~$358). Pipeline completed mechanically; all checkpoints valid and
loadable. The final model does not produce coherent text.

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

## Artifacts

- Checkpoints: `checkpoints/{distill,sft,grpo}/final.pt` (+ `grpo/math/`).
- Full log: `~/dimba_train.log` (filter `grep -vE "httpx|HTTP Request|cas-bridge|examples/s"`).
- Inference test scripts (correct loaders, incl. flow sampler):
  scratchpad `infer_gpu_cmp.py`, `infer_flow.py`, `infer_cpu_multi.py`.
- HF backup repo: `devnull37/d1-135m-50b` (private).
