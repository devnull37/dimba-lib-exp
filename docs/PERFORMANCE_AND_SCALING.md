# Performance and scaling

This is the source of truth for the optimization pass completed on 2026-07-13. It separates
measurements from forecasts. Historical model cards and experiment write-ups keep their original
numbers and should not be read as benchmarks of the current tree.

## Production choices

- **Backbone:** Mamba-2 only. CPU/MPS can use the weight-compatible `TorchMamba2` fallback.
  CUDA training fails closed unless every live mixer is `mamba_ssm.Mamba2` and
  `causal-conv1d` imports. There is no Mamba-1 fallback.
- **Optimizer:** AdamW remains the default. `--optimizer muon` (or
  `--stage3-optimizer muon`) is an opt-in pilot that sends eligible hidden 2-D matrices to
  Muon and keeps embeddings, vocabulary heads, norms, biases, SSM state parameters, and vectors
  on AdamW. PyTorch's native Muon is used when present; otherwise the repo's compatible
  Moonlight-style implementation is used.
- **Multi-GPU:** one process per GPU with `DistributedDataParallel` (DDP). This is the best fit
  for the current 350M/1.5B class because a full replica fits an H100 and Muon needs intact 2-D
  matrices. `DataParallel` is not used; FSDP2/ZeRO/tensor-sharded Muon is deliberately unsupported.
  First-class DDP applies to the canonical masked base/SFT launchers and continuous H100 Stage 3.
  Continuous alignment runs once on rank 0; DDP then broadcasts that student. Each rank keeps a
  frozen teacher replica, and the complete diffusion + KD loss runs inside one DDP forward.
- **CUDA math:** BF16, TF32 matmul/convolution, fused AdamW, official fused Mamba-2 and
  causal-conv kernels. These are enabled only on CUDA; the Mac path remains MPS/MLX compatible.

## What changed

### Inference

- Masked sampling keeps the schedule, mask state, confidence ranking, and token selection on the
  accelerator. It transfers only final output, projects the vocabulary only for unresolved
  positions, computes top-k confidence without a full probability tensor, and no longer performs
  a redundant final model call.
- Conditional and unconditional classifier-free-guidance rows are concatenated into one model
  dispatch. For masked inference, features are combined before a single selected vocabulary
  projection. This removes dispatch and projection overhead; it does not pretend that CFG has only
  one row of model FLOPs.
- Continuous DDIM, DPM-Solver++(2M), and flow sampling use device-resident schedules and one batched
  CFG dispatch, with fast paths for guidance 0 and 1 and self-conditioning retained on device.
- Best-of-N verification is batched instead of calling the model once per candidate.
- MLX masked sampling now keeps the whole trajectory on the Apple GPU, uses selected projection,
  batches CFG, and converts to host memory once at the end. MLX 0.29's batch-3+ NaN workaround uses
  stable two-row chunks without changing token results.

### Training

- Masked training corrupts and selects tokens on device and materializes `[active_tokens, vocab]`
  rather than `[batch, sequence, vocab]`. The weighted masked objective remains the exact native
  PyTorch loss.
- Optional Liger fused linear cross-entropy is used only for mathematically compatible uniform
  continuous `mean`/`sum` reductions. Weighted, masked, response-only, time-faded, and
  unlikelihood losses stay on the exact selected-token path because Liger's unreduced backward
  does not preserve those gradients.
- Host synchronization was removed from hot logging paths. Metrics stay as detached device tensors
  until the configured log boundary. Pinned data loaders and non-blocking CUDA copies are used where
  applicable.
- Gradient accumulation uses DDP `no_sync()` for non-update microbatches. Partial accumulation
  groups use their true divisor instead of under-scaling the final group.
- Checkpoints are atomic. Masked and continuous DDP checkpoints include per-rank RNG/data state and
  require the same world size on exact resume. Continuous Stage 3 additionally records optimizer,
  phase, cumulative per-rank stream cursor, pinned data revisions, backend, and run signature.

## Multi-GPU launch

`--batch` is the microbatch **per GPU**. The global batch is:

```text
global_batch = batch_per_gpu * world_size * accumulate
```

Continuous Stage 3 on eight H100s (the preset keeps a fixed global token budget, so resolved
optimizer steps are divided by world size):

```bash
torchrun --standalone --nproc-per-node=8 scripts/train_h100.py \
  --preset validation --phase distill --stage3-optimizer adamw \
  --save-dir checkpoints/h100-validation-8gpu
```

The H100 launcher uses batch 64 per GPU and accumulation 1. A distributed OOM fails the torchrun
coherently. Exact resume requires the checkpoint's original batch, so a smaller per-GPU batch is a
new run (optionally initialized with an explicitly weight-only checkpoint), not an exact resume.

Single-node masked base training on eight GPUs:

```bash
torchrun --standalone --nproc-per-node=8 scripts/masked_diffusion_finetune.py \
  --batch 64 --accumulate 1 --optimizer adamw --workers 4 \
  --output-dir checkpoints/masked-base-adamw
```

Run the controlled Muon arm with a distinct `--output-dir`:

```bash
torchrun --standalone --nproc-per-node=8 scripts/masked_diffusion_finetune.py \
  --batch 64 --accumulate 1 --optimizer muon --workers 4 \
  --output-dir checkpoints/masked-base-muon
```

Masked SFT after the base-quality gate:

```bash
torchrun --standalone --nproc-per-node=8 scripts/mdm_sft_cfg2.py \
  --batch 64 --accumulate 1 --optimizer adamw --workers 4 \
  --output-dir checkpoints/masked-sft-adamw
```

The continuous complete-loss/KD path and exact resume are exercised by a two-rank CPU/Gloo test;
the canonical masked path also has two-rank coverage. Target-H100/NCCL measurement is still a
hardware gate, not a local claim. Standard
multi-node `torchrun` is a plausible next step, not a validated capability: it also requires the
rank-zero-built cache and checkpoints to live on a filesystem visible at the same path from every
node. Run a multi-node smoke and failure/restart test before treating it as production-ready; the
current exact-resume claim covers same-world-size single-node runs only. See
[NEXT_RUN_PLAN.md](NEXT_RUN_PLAN.md) for gate order and [BABYSIT_LOOP.md](BABYSIT_LOOP.md) for
recovery rules.

## Kernel policy

No bespoke Mamba or causal-convolution kernel is justified: the production path already uses the
specialized upstream kernels, and maintaining a duplicate would add correctness and compiler risk.
The only extra fused kernel wired here is Liger's fused linear cross-entropy at the exact semantic
seam described above.

Any future Triton/CUDA kernel must pass the real-H100 suite before it is promoted:

1. Run every default case exactly once with a trained checkpoint and the native fast Mamba-2 path.
2. Preserve deterministic masked outputs and continuous final-logit parity.
3. Achieve at least **1.30x** p50 on selected masked CE, and at least **1.05x** p50 or **20%**
   incremental HBM reduction on both the fused-CE seam and at least one end-to-end inference case.
4. Regress no p95 latency. A `--compile` run is diagnostic and intentionally non-promotable.

```bash
# Plan/config check on any machine
python3 scripts/benchmark_h100.py --dry-run

# Promotion run on the target H100
python3 scripts/benchmark_h100.py \
  --checkpoint /path/to/trained.pt \
  --output artifacts/h100-performance.json \
  --fail-on-gate

# Separate compile diagnostic (never a promotion result)
python3 scripts/benchmark_h100.py \
  --checkpoint /path/to/trained.pt --compile \
  --output artifacts/h100-compile-diagnostic.json
```

The JSON records device/software versions, native mixer types, checkpoint context, source-tree
fingerprints, CUDA-event p50/p95, cold compile time, timed recompiles, throughput, NFE, incremental
peak HBM, and parity.

## Measurements available now

### Apple M1 Pro, masked MLX inference

A paired before/after run on this 16-GB M1 Pro used `hr-diffuse-1-nano`, one candidate, 128
unmasking steps, CFG 2.0, and a 40-token answer:

| Implementation | Wall time | Relative |
|---|---:|---:|
| Before this pass | 7.8311 s | 1.000x |
| Current device-resident/selected-projection path | 6.8176 s | **1.149x** |

That is a measured **12.9% latency reduction**, with identical final argmax tokens. It is one
paired local measurement, not a multi-run H100 result. Older CPU/MPS/MLX tables in
[BACKENDS.md](BACKENDS.md) use different models and recipes and are retained as historical context.

### Muon pilot

The controlled 2,000-step masked-continuation pilot ended at CE **5.453 for Muon vs 5.470 for
AdamW** under the same data/seed recipe. This is a small quality/convergence signal, not evidence
that each Muon step is faster; Newton-Schulz work can make a step slightly slower. Keep AdamW as
the control arm until the next-run pilot confirms time-to-quality.

### CUDA

No CUDA or H100 timing was measured on the development Mac. The H100 harness is implemented, but
all CUDA estimates below remain forecasts until its JSON report passes the gate.

The corrected continuous Stage-3 recipe is a correctness repair, not a throughput optimization.
Each KD-enabled step adds a logits-only teacher forward plus the student vocabulary head and KL;
it reuses the base objective's continuous student pass, projects/reduces in exact batch chunks, and
fades KD as `(1-t)` so it does not fight the pure-noise repair. It can still take longer per optimizer
step than the invalid budget28 KD-off run.
No H100 Stage-3 speedup is claimed here: compare wall time to the fixed recovery/coherence gate,
and record the first 100-step tokens/s window before projecting cost.

## Planning estimates, not measurements

Use these ranges for capacity planning only:

| Scope | Expected improvement | Basis |
|---|---:|---|
| Masked inference on one H100 | **1.2-1.6x** | selected vocabulary projection, resident trajectory, batched CFG |
| Continuous CFG inference on one H100 | **1.3-1.8x** | one 2B dispatch, resident schedules, fewer launches/syncs |
| Masked training step | **1.05-1.30x** | active-token projection; benefit grows with vocabulary and mask sparsity |
| Batched best-of-N verifier seam | **3-7x** | one batched feature dispatch instead of N serial forwards |
| Single-node eight-H100 DDP vs one H100 | **6.0-7.2x** | 75-90% scaling efficiency when input/cache and network are healthy |

The optimization and DDP factors apply to different bottlenecks and must not be multiplied blindly.
For the next masked training run, a reasonable end-to-end target is roughly **6-9x aggregate
throughput on eight H100s versus the old single-H100 path**, with the benchmark and first 100-step
throughput window deciding the real number. Accuracy-preservation comes from exact objective math,
strict checkpoint loading, and parity gates—not from assuming every fast path is equivalent.
