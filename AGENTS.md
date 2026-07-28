# AGENTS.md — DIMBA public repository guide

Read this first. This file is the durable onboarding guide for the public DIMBA repository:
scope, verified status, invariants, maintained entry points, and live TODOs. `CLAUDE.md` only
points here.

**Standing maintenance rule:** any agent that changes code, defaults, repository layout,
measurements, decisions, or unfinished work must update the relevant status/TODO text and the
“Last full update” line here in the same commit, without asking first. Record exact verification
commands and hardware for measurements, distinguish measurements from estimates, and never leave
transient working-tree notes here.

Last full update: **2026-07-28** (public repository reduced to inference, evaluation, and release
evidence).

---

## 1. Scope and privacy boundary

DIMBA (`dimba-lib`) is a non-autoregressive diffusion language model on a bidirectional Mamba-2
backbone. This public repository contains:

- the reusable DIMBA model architecture;
- masked and continuous diffusion samplers;
- PyTorch CPU/MPS/CUDA and MLX inference paths;
- evaluation and release utilities;
- tests, the v1 paper, and public release evidence.

It intentionally does **not** contain training, fine-tuning, distillation, weight-transmutation,
teacher-logit caching, scale-run configuration, H100 orchestration, cloud checkpoint automation,
private checkpoints, teacher outputs, or internal run logs. Do not add those here and do not link
public files to a private repository.

Existing public Git history remains public. Removing training material from the current tree does
not make old revisions private, so never claim otherwise.

The v1 paper under `paper/` is a historical, untested concept. The code in this repository is v2.

---

## 2. Current verified status

- The released model, `devnull37/hr-diffuse-1-nano`, is a masked-diffusion,
  bidirectional-Mamba-2 research model with **287.9M measured parameters**.
- Its published 40-item evaluation measured 15.0% factual QA, 14.0% native infill recovery,
  and a 7.5% repetition-loop rate. See `docs/benchmarks.md` for the full comparison and caveats.
- CUDA-graph replay captures the fixed-shape raw denoiser while latent decode and shape-varying
  token projection stay eager. One RTX 4090 production-shape run measured
  **17.92 s eager -> 1.16 s graphed (15.49x)** with exact 40/40-token parity. This is a
  directional one-run measurement, not a repeated H100 promotion result.
- Apple Silicon behavior is documented in `docs/BACKENDS.md`. The current masked sampler measured
  6.82 s versus 7.83 s before its paired optimization on an M1 Pro, with identical final argmax
  tokens. That result is a different workload from the older continuous-model MLX comparison.
- The 2026-07-28 cleanup removed public training/distillation code, run configs, scratch scripts,
  notebooks, internal run documents, and tests that only exercised removed functionality.

**Current cleanup verification baseline:** on 2026-07-28, the Apple-Silicon Mac ran
`python3 -m pytest tests/ -q -p no:cacheprovider --override-ini addopts=` with
**288 passed, 1 warning in 12.34 s**. The warning is the known tensor-to-scalar warning in the
latent-scale calibration assertion; no test failed.

---

## 3. Live public TODOs

- [x] Record the full post-cleanup test result in §2: 288 passed on the Apple-Silicon Mac.
- [ ] Repeat the production CUDA-graph shape with one warmup and five measured runs, archive
      publishable p50/p95 evidence, and retain exact-token/capture-state gates. The current 4090
      result is directional only.
- [ ] Re-run the quality probes before changing `--commit-threshold` or any quality-slider
      default. Token parity alone is not a quality gate.
- [ ] On MLX >=0.30, re-run the batch-4 parity repro before removing the version-gated
      two-row workaround.
- [ ] Keep release docs and package metadata synchronized with the inference-only tree.

Training, distillation, transmutation, teacher-logit streaming, private scale runs, and cloud
operations are intentionally not public TODOs.

---

## 4. Architecture and inference invariants

```text
input ids -> token embedding -> latent encoding
prompt-only conditioning + timestep embedding
    -> bidirectional Mamba-2 denoiser
    -> raw prediction -> latent decode -> output head -> token logits
```

- **No conditioning leak.** Conditioning is prompt-only; never condition on a clean target.
- `DIMBA.forward()` returns `(x_pred, noise, latent_info)`.
- `encode_latent` and `decode_latent` carry `latent_scale`; latent-mode callers calibrate it.
- The masked inference fast path is intentionally split:
  `predict_token_features(ids, t)` is fixed-shape and graph-capturable;
  `project_token_features(features, positions)` is shape-varying and eager.
- fp32 precision islands are load-bearing: SSD scan/log-decay handling, confidence softmax,
  schedule tables, and MLX scan behavior must not be casually downcast.
- Avoid `.item()`/`.any()` host synchronization in hot loops. Documented quality-affecting
  threshold logic is the exception.
- `model.config` stores the complete constructor configuration used for checkpoint rebuilds and
  MLX conversion.
- CUDA uses native `mamba_ssm.Mamba2` when the fused stack is available. CPU/MPS use the
  weight-compatible `TorchMamba2` fallback.
- A CUDA-graph measurement is valid only when capture counters show at least one graph, eager
  fallback is false, and output parity passes.

Do not reintroduce the historical two-value forward return, positive-`A` SSM, unscaled latent
path, prompt-conditioning leak, or unsupported full-sampler graph boundary.

---

## 5. Maintained repository map

```text
src/dimba/
  models/         DIMBA, denoiser, TorchMamba2/SimpleMamba2, VAE, embeddings
  diffusion/      schedules, corruption, masked/continuous sampling, reranking
  backends/mlx/   Apple-GPU inference implementation
  inference/      block-sequential inference
  evaluation/     metrics
  tokenizers/     tokenizer implementations
  utils/          compile and CUDA-graph helpers

scripts/
  generate.py             released-model CLI and quality slider
  sample_mlx.py           MLX checkpoint sampler
  evaluate.py             checkpoint evaluation
  eval_vs_smollm.py       comparison harness
  perplexity_eval.py      denoising-reconstruction perplexity
  benchmark.py            tiny CPU inference benchmark
  verify_mlx_model.py     PyTorch/MLX parity check
  upload_to_hf.py         explicit release-artifact uploader

tests/                    architecture, sampler, parity, backend, and regression tests
docs/                     backend guidance and public release evidence
paper/                    historical v1 paper
```

There is no public training entry point. Inspect a retained script and its tests before changing
it. Prefer an existing library helper over a new one-off script.

---

## 6. Documentation authority

| Document | Role |
|---|---|
| `AGENTS.md` | Current public scope, invariants, status, and TODOs |
| `README.md` | Public installation, inference usage, and project overview |
| `docs/hr-diffuse-1-nano_system_card.md` | Released-model facts and required sampling recipe |
| `docs/benchmarks.md` | Public evaluation evidence and measurement caveats |
| `docs/slider_curve.md` | Published accuracy/cost sweep |
| `docs/BACKENDS.md` | CPU, MPS, MLX, and CUDA-checkpoint inference behavior |
| `docs/writeup.md` | Historical research narrative; not current operational guidance |

Never point public documentation at deleted training files, internal run docs, or private URLs.

---

## 7. Verification and claim boundaries

Run from the repository root:

```bash
python3 -m compileall src/dimba scripts tests
python3 -m pytest tests/ -q -p no:cacheprovider --override-ini addopts=
git diff --check
```

Focused checks are useful while iterating, but source changes need the relevant full/focused gate.
Record fresh counts instead of copying an old baseline.

This development machine is Apple Silicon without NVIDIA CUDA. It can verify CPU/MPS/MLX behavior,
shapes, parity, and unit tests. It cannot verify CUDA graphs, fused CUDA Mamba-2, NCCL, H100
memory, or H100 throughput.

CUDA performance claims require a real NVIDIA run with exact-output parity, capture-state
evidence, warmups, p50/p95, and the hardware/software stack. Quality-affecting sampler changes
require the published quality probes, not just token parity.

---

## 8. Development and handoff conventions

- Python >=3.9; Black line length 100; match surrounding type/docstring style.
- Use the standard library or an existing helper before adding a dependency, abstraction, or
  script.
- Preserve unrelated working-tree changes.
- Keep credentials, weights, caches, generated outputs, and local logs out of Git.
- Treat all upload commands as explicit release operations; never upload by implication.
- Before handoff, update this file for every changed fact or TODO, run `git diff --check`, and
  report exactly what was and was not verified.
