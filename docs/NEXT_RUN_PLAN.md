# DIMBA next-run plan

**Status (2026-07-13):** the definitive post-mortem is
[ROOT_CAUSE_BUDGET28.md](ROOT_CAUSE_BUDGET28.md). The failed run was not a 1B-token,
FFN-frozen run: it completed roughly **28B Stage-3 tokens** and both frozen and unfrozen
phases. More tokens alone are not the repair.

## What actually failed

The budget28 model reconstructs unseen English with 98-100% accuracy from noise levels through
approximately `t=0.9`, yet collapses at exactly `t=1.0`, the pure-noise starting point of every
free generation. The model learned a strong conditional denoiser; the objective left a high-noise
dead zone and the sampler faithfully amplified its first unigram-like estimate.

The primary mechanism was:

1. min-SNR weighting reduced the diffusion gradient near `t=1` to roughly zero;
2. full-strength CE at the same timesteps taught the marginal token distribution;
3. logit-normal timestep sampling alone undersampled the endpoint.

A secondary problem was that Stage 3 set `kd_weight=0`, disconnecting the causal teacher for almost
the entire run. The original 500+500 alignment steps were also short. These issues matter, but they
do not overturn the controlled experiment that isolated the high-noise cliff.

## Corrected Stage-3 contract

Every fresh or repair Stage-3 run now has all of the following:

| Control | Current value | Reason |
|---|---:|---|
| high-noise SNR floor | `0.5` | preserves regression pressure near pure noise |
| CE time fade | `CE * (1 - t)` | stops teaching unigram output at high noise |
| KD time fade | `KD * (1 - t)` | keeps per-sample clean teacher targets away from pure noise |
| timestep distribution | 50% uniform + 50% logit-normal | covers both `t≈1` and the useful middle |
| Stage 3a teacher KD | `1.0` | keeps inherited causal knowledge connected while the mixer beds in |
| Stage 3b teacher KD | `0.3` | retains teacher structure during low-LR whole-model adaptation |
| Stage 3a FFN | frozen, LR `2e-4` | adapt the new mixer to inherited features |
| Stage 3b FFN | unfrozen, LR `3e-5` | co-adapt without washing out inherited knowledge |
| optimizer | AdamW default | Muon remains a named A/B pilot |

The causal teacher and bidirectional student are aligned on the same target token: teacher logits
at position `i` are compared with student logits at `i+1`. The teacher stays resident whenever a
Stage-3 KD weight is positive. Tests assert the nonzero 1.0→0.3 recipe, causal shift, SNR/CE controls,
and mixed timestep draws. Exact checkpoints record the phase dictionary, so resuming under a zero-KD
or different objective is rejected.

Muon routes eligible hidden 2-D matrices through match-RMS Muon and leaves embeddings, vocabulary
heads, norms, biases, vectors, and SSM state parameters on AdamW. A controlled 2,000-step masked
continuation ended at CE 5.453 vs 5.470 for AdamW, but that is not a continuous Stage-3
time-to-quality result. Keep separate AdamW and Muon pilot directories; promote Muon only if it wins
the same held-out gate in less wall time.

## Validation ladder

Do not jump from a green loss curve to SFT. Free generation and the high-noise recovery ladder are
blocking gates.

### R0 — contract smoke

Before renting a long run, rerun the tiny overfit/partial-noise checks on the exact commit and CUDA
stack. Require:

- finite forward/backward under the fixed loss;
- nonzero samples from both timestep-mixture components;
- near-perfect recovery below `t=1` after overfit;
- free generation from `t=1` that is no longer unigram soup.

This verifies the mechanism, not general language quality.

### R1 — recover the 28B checkpoint first

The existing checkpoint contains a strong denoiser below the dead zone. The cheapest informative
run is a **1B-token unfrozen repair** with the fixed objective and KD 0.3:

```bash
python3 scripts/train_h100.py --preset repair1b --phase distill \
  --checkpoint checkpoints/distill/final.pt --resume --weights-only-resume \
  --stage3-optimizer adamw \
  --save-dir checkpoints/h100-budget28-repair1b-adamw
```

`--weights-only-resume` is intentionally explicit because the old final checkpoint cannot restore
optimizer/RNG/data progress. It skips Stage 1/2, restarts the one unfrozen repair phase from the old
weights, and writes new exact checkpoints. Short runs save at quarter points (long runs remain
capped at 25,000-step cadence), so inspect the first checkpoint before spending the whole budget.

At each quarter point, record on fixed seeds/prompts:

- the partial-noise recovery ladder at `t={0.8,0.9,0.95,0.99,1.0}`;
- free-generation grammar, repetition, topic stability, and prompt relevance;
- teacher-scored sample perplexity and held-out fixed-loss components;
- exact sampler settings and checkpoint hash.

**GO:** the `t=1` cliff closes and free samples become locally coherent without degrading the
already-strong `t≤0.9` recovery. **NO-GO:** `t=1` remains near random or lower-noise recovery
regresses. Stop; do not spend another 5-50B tokens and do not start SFT.

### R2 — fresh recipe validation

Only after R1 shows the objective can repair pure-noise generation, run a fresh conversion:

```bash
python3 scripts/train_h100.py --preset validation --phase distill \
  --stage3-optimizer adamw \
  --save-dir checkpoints/h100-validation-adamw
```

The `validation` preset is roughly 1B tokens (700M frozen + 300M unfrozen). It tests whether the
full teacher-guided recipe learns the endpoint from a fresh initialization. It is not evidence that
1B is a final training budget.

Run the Muon arm separately only after AdamW establishes the baseline:

```bash
python3 scripts/train_h100.py --preset validation --phase distill \
  --stage3-optimizer muon \
  --save-dir checkpoints/h100-validation-muon
```

Compare wall-clock time to the same recovery/coherence gate, not loss at an arbitrary step.

### R3 — scale decision

If R2 passes, `--preset scale` runs roughly 5B tokens (3B frozen + 2B unfrozen). Require coherent
30+ token continuations, low repetition, no `t=1` cliff, and a non-trivial teacher-relative held-out
score before considering 20-50B. A 50B `full` run is a post-gate option, not the default answer to
the budget28 failure.

SFT and GRPO cannot repair an incoherent base. The H100 launcher requires a human-recorded
`--quality-gate-passed` before either phase and rejects `--phase all`.

## Production launch and resume rules

CUDA training fails closed unless every live mixer is `mamba_ssm.Mamba2` and
`causal-conv1d` imports. The continuous launcher remains one-phase-at-a-time, but Stage 3 supports
single-node one-process-per-GPU DDP. Alignment runs once on rank 0, teachers are replicated and
frozen, and the complete diffusion + KD objective is DDP-wrapped.

Resume an exact Stage-3 checkpoint with the same preset, optimizer, topology, backend, batch, data
revision, and save directory:

```bash
python3 scripts/train_h100.py --preset validation --phase distill \
  --stage3-optimizer adamw \
  --checkpoint checkpoints/h100-validation-adamw/distill_latest.pt --resume \
  --save-dir checkpoints/h100-validation-adamw
```

For eight H100s, use the same command under `torchrun`:

```bash
torchrun --standalone --nproc-per-node=8 scripts/train_h100.py \
  --preset validation --phase distill --stage3-optimizer adamw \
  --checkpoint checkpoints/h100-validation-8gpu/distill_latest.pt --resume \
  --save-dir checkpoints/h100-validation-8gpu
```

Omit `--checkpoint --resume` for a fresh run. Batch 64 is per GPU; the launcher divides resolved
optimizer steps by world size to preserve the preset's global token budget. Exact resume requires
the same world size. SFT and GRPO remain single-process.

This restores model, optimizer, Python/CPU/CUDA RNG, phase/step, pinned revisions, and the cumulative
3a→3b stream cursor. A trajectory mismatch, including KD/loss/timestep settings, fails closed.

After a recorded base gate:

```bash
python3 scripts/train_h100.py --preset validation --phase sft \
  --checkpoint checkpoints/h100-validation-adamw/final.pt \
  --quality-gate-passed --save-dir checkpoints/h100-validation-adamw-sft

python3 scripts/train_h100.py --preset validation --phase grpo \
  --checkpoint checkpoints/h100-validation-adamw-sft/final.pt \
  --quality-gate-passed --save-dir checkpoints/h100-validation-adamw-grpo
```

SFT/GRPO do not implement exact continuation. Restart an interrupted post-training phase from its
input weights; do not label that a resume.

To stop Stage 3 safely, write `{"stop": true}` to `training_state_override.json`, wait for a fresh
`distill_latest.pt` and process exit, then stop the host. See [BABYSIT_LOOP.md](BABYSIT_LOOP.md).

## Multi-GPU tracks

Continuous Stage 3 uses the H100 command above. Canonical **masked** base and SFT training also use
one DDP process per CUDA GPU:

```bash
PYTHONPATH=src torchrun --standalone --nproc-per-node=8 \
  scripts/masked_diffusion_finetune.py \
  --device cuda --batch 8 --accumulate 1 --optimizer adamw \
  --output-dir checkpoints/masked-base-adamw

PYTHONPATH=src torchrun --standalone --nproc-per-node=8 \
  scripts/mdm_sft_cfg2.py \
  --device cuda --batch 8 --accumulate 1 --optimizer adamw \
  --output-dir checkpoints/masked-sft-adamw
```

`--batch` is per GPU; global batch is `batch × world_size × accumulate`. Rank 0 builds the shared
cache and writes atomic checkpoints; failures are broadcast so peers do not deadlock. Exact resume
requires the same world size and run signature. Use a distinct `--output-dir` for each optimizer
arm. Single-node DDP is tested. Standard multi-node
`torchrun` is plausible but not yet validated for this repo and additionally requires a filesystem
visible at the same cache/checkpoint path on every node.

## Kernel and performance gate

Use the official Mamba-2 and causal-convolution kernels. Liger fused linear CE is enabled only for
uniform reductions where its backward is exact; weighted masked/response losses use selected native
projection. No bespoke CUDA kernel should be added or promoted without the trained-checkpoint H100
p50/p95/HBM/parity gate in [PERFORMANCE_AND_SCALING.md](PERFORMANCE_AND_SCALING.md).

## Evidence hierarchy

1. [ROOT_CAUSE_BUDGET28.md](ROOT_CAUSE_BUDGET28.md) — definitive controlled post-mortem.
2. This file — current launch/gate decision.
3. [PERFORMANCE_AND_SCALING.md](PERFORMANCE_AND_SCALING.md) — optimization and benchmark truth.
4. Historical model cards/write-ups — preserve old results, but do not override the current recipe.
