# AGENTS.md — project guide for any agent

**Read this first.** It is the single onboarding document for any agentic coding model (or
human) landing in this repo: what the project is, its full history, exactly where it stands
today, what is verified vs. estimated, and the live TODO list. `CLAUDE.md` points here.

**Maintenance rule (standing instruction from the repo owner):** any agent that changes the
state of the project — lands code, measures a number, completes/adds a TODO, changes a
default — must update this file to match, **autonomously, without asking**, and commit the
update so every other agent sees the current truth. Keep §2 (status), §4 (TODO), and the
"Last full update" line below accurate.

Last full update: **2026-07-17** (RTX 4090 CUDA-graph verification and capture-boundary fix).

---

## 1. What this project is

**DIMBA** (`dimba-lib`) is a non-autoregressive **diffusion language model on a bidirectional
Mamba-2 backbone**. Instead of generating left-to-right, it generates whole sequences in
parallel by iterative denoising. Two diffusion tracks coexist:

1. **Masked / absorbing-state (LLaDA/MDLM-style)** — the **production track**. Tokens are
   corrupted to `[MASK]` and iteratively unmasked by confidence. This produced the released
   model **hr-diffuse-1-nano**.
2. **Continuous Gaussian latent diffusion** — the original track (DDIM/DPM++/flow samplers).
   Its 28B-token run failed (see §3, Budget28) and it is now the distillation/repair track.

- **v1 = `paper/main.pdf`** — an *untested architectural concept*. It contains a
  prompt-conditioning leak and an MSE-only objective that v2 deliberately fixes. Never treat
  it as ground truth.
- **v2 = this repo.** Everything below describes v2.

### The released model: hr-diffuse-1-nano

- Masked discrete diffusion, bidirectional Mamba-2, **287.9M measured params**
  ("135M-class" capacity: denoiser 258.1M + embeddings 28.3M). Tokenizer SmolLM-135M + one
  `[MASK]` row (id **49152**; vocab 49153). HF: `devnull37/hr-diffuse-1-nano`.
- Built for **~$450** on rented H100s: backbone distilled from SmolLM-135M over 28B tokens →
  converted to masked diffusion in 40k steps → SFT on 422k instruction pairs with CFG dropout.
- Evals (40-item QA, 2026-07-04): QA **15.0%** vs teacher SmolLM-135M's 82.5% (capacity-bound,
  expected); **native infill 14.0%** where every AR baseline is ~0 (the structural win); loop
  rate 7.5% vs SmolLM's 37.5%; latency **13.33 s**/40-token answer @ 128 steps (the weak axis —
  being attacked right now, see §4).
- Required sampling recipe: 128 steps, temp 0.7, top-k 20, CFG 2.0, exempt-first frequency
  penalty 0.7. Quality slider trades 2.2–27.7 s/answer for QA 7.5%→20%.
- Central research finding (whole project, see `docs/writeup.md`): **at 135M scale,
  self-judgment fails; external constraints work.** The 300k-param critic head detects errors
  well (precision@k 52.5% vs 10% chance) but cannot improve generation (Goodhart).

---

## 2. Current status — what is happening RIGHT NOW

**The CUDA-graph sampler is verified on an RTX 4090.** Full-feature capture initially failed at
the post-denoiser latent decode boundary. Capturing the expensive raw denoiser only fixed it:
one production-shape run measured **17.92 s eager -> 1.16 s graphed (15.49x)** with exact final
tokens (40/40, seed 0). A separate harness timing against the historical two-pass/full-vocab
baseline measured 8.05 s -> 0.223 s (36.12x), but that is not the apples-to-apples headline.

What the speedup pass shipped (all parity-tested on CPU/MPS/MLX; CUDA numbers are ESTIMATES
until the GPU run):

| Change | Where | Default | Expected (est.) | Gate |
|---|---|---|---|---|
| CUDA-graph replay of the sampler denoiser | `src/dimba/utils/cuda_graphs.py` + `scripts/generate.py` | ON for CUDA (`--no-graph` off-switch) | **MEASURED 15.49x on RTX 4090** (17.92 s -> 1.16 s, one run) | exact-token parity (40/40) |
| Confidence-threshold commits + real early exit | `generate.py --commit-threshold TAU`, MLX `sample_masked` | OFF (quality-affecting) | 1.5–4× fewer steps | quality probes (benchmarks.md A–C), NOT parity |
| Adaptive CFG truncation | `generate.py --cfg-drop FRAC` | OFF | 1.1–1.4×, only post-graphs | quality probes |
| Chunked selected-token CE (no `[N,49k]` logits retained) | `training/fused_ce.py:_ChunkedLinearCE`, wired in `masked_token_loss` | ON (2048-token chunks) | ~1.6 GB activation memory back → bigger `--batch` | grad parity (tests + benchmark) |
| `--compile` of the denoiser forward | both masked training launchers | OFF (opt-in) | 1.1–1.35× tok/s | loss-curve overlay vs eager |
| MLX fp16/bf16 weights (fp32 islands) | `MLXDIMBA.from_torch(dtype=...)`, `--mlx-dtype` | OFF | **MEASURED 1.01× (M1 Pro)** — dispatch-bound, kept opt-in | argmax agreement ≥95% |
| MLX one-graph-per-step + version-gated NaN chunking | `backends/mlx/model.py` | ON | small | existing 38 MLX parity tests |

Why this direction and not custom kernels: the CUDA path was diagnosed as
**launch-overhead-bound** (~100 ms/step of Python dispatch + ~1000 kernel launches for
single-digit-ms of GPU compute at L≈50). The scan already runs fused `mamba-ssm`, the vocab
GEMM is cuBLAS, masked CE already projects selected positions only. A prior profiling session
(`yc-hamiltonian-research-claude-code.txt`, repo root) estimated a bespoke fused sampler at
only ~3–7%. **Do not write custom Mamba/conv kernels** without passing the promotion gate in
`scripts/benchmark_h100.py` (§ Kernel policy in `docs/PERFORMANCE_AND_SCALING.md`).

**Also uncommitted in the working tree** (untracked, tests pass 18/18): the **YC application
demo app** — `scripts/yc_app_demo.py` (one-command DIMBA-vs-SmolLM timed demo → HTML+JSON in
`demos/yc_app/`), `scripts/yc_app_tui.py` + `src/dimba/demo/` (Textual TUI with live
denoising), `tests/test_yc_{demo,tui}.py`. The README already references it. Decide with the
user whether/when to commit it.

---

## 3. Project history in five acts (why the docs say what they say)

1. **v1 concept → v2 overhaul.** `docs/IMPROVEMENT_PLAN.md` (2026-05-27, historical) lists the
   10 verified v1 findings (conditioning leak, fake zero-terminal-SNR, MSE-only loss, causal
   backbone, Mamba-1, Python-loop scan…). The v2 overhaul fixed all of them;
   `docs/OVERHAUL_STATUS.md` is the ledger of what's done.
2. **Budget28 failure ($358, 28B tokens).** The continuous-diffusion run produced word salad.
   `docs/ROOT_CAUSE_BUDGET28.md` is the definitive post-mortem: the model denoises 90%-noised
   text at 98–100% accuracy but is blind exactly at t=1.0 (min-SNR zeroed the gradient there,
   CE taught the unigram marginal, logit-normal undersampled the endpoint; KD was
   accidentally off). "Just train longer" is REFUTED — do not resurrect it.
3. **Masked-diffusion pivot → hr-diffuse-1-nano.** The masked track worked; SFT'd release +
   system card (`docs/hr-diffuse-1-nano_system_card.md`), benchmarks (`docs/benchmarks.md`),
   quality-slider study (`docs/slider_curve.md`), narrative (`docs/writeup.md`).
4. **Production hardening (→2026-07-14, `a1a8e89`).** Mamba-2-only enforcement (CUDA fails
   closed without fused `mamba_ssm` + `causal-conv1d`), one-process-per-GPU DDP for masked and
   continuous tracks, device-resident samplers, batched CFG, atomic checkpoints + exact
   resume, Liger CE where mathematically safe, Muon opt-in pilot, H100 benchmark harness with
   a promotion gate.
5. **Speedup pass (2026-07-17, `e243b98`).** §2 above. Verification pending.

---

## 4. Live TODO list (ordered)

**Now / blocking:**
- [x] Verify lossless CUDA-graph replay on the rented 4090: **15.49x**, exact 40/40 tokens.
- [ ] Run the production shape with 1 warmup + 5 repeats and archive the JSON; the current
      15.49x result is a one-run directional 4090 measurement, not the formal H100 promotion.
- [ ] Run the complete remote CUDA suite after making CPU-only unit models explicitly select
      `TorchMamba2`; the focused local graph/benchmark suite is 22/22, while the initial remote
      full suite had 15 environment-selection failures when fused Mamba was installed.
- [ ] If graphs verify: re-run the quality probes with `--commit-threshold 0.9` and pick a
      default τ for the quality slider (currently OFF by default, plumbed through
      `slider_generate` overrides).
- [ ] Decide whether to commit the untracked YC demo app (tests pass; README references it).

**Next training run (the big one — `docs/NEXT_RUN_PLAN.md` is authoritative):**
- [ ] Validation ladder for the repaired continuous Stage-3 contract: R0 contract smoke →
      R1 1B-token repair of the 28B checkpoint (`--preset repair1b`, KD 0.3,
      `--weights-only-resume`) → R2 fresh ~1B `validation` preset → R3 ~5B `scale`; the 50B
      `full` run only after gates pass. Contract: SNR floor 0.5, CE fade `(1-t)`, 50/50
      uniform/logit-normal timesteps, KD 1.0 (3a, frozen FFN) → 0.3 (3b, unfrozen).
- [ ] Roadmap model: **1.5B params distilled from SmolLM2-1.7B** (~$1,500–4,000), using the
      shared-base + per-direction LoRA bidirectionality that won the A/B
      (`docs/bidir_ab.md`: shared+LoRA rank16 = best params-per-nat).
- [ ] AdamW vs Muon time-to-quality pilot on the next run (2k-step CE 5.453 vs 5.470 is
      suggestive, not decisive).

**Research backlog (gated behind the above):**
- [ ] Train/calibrate a real latent VAE (continuous track currently uses projector latents).
- [ ] Cross-attention prompt conditioning (currently pooled prompt summary).
- [ ] Step-count distillation of the masked sampler to 8–16 NFE (4–8× est.; infrastructure
      exists in `src/dimba/distillation/`).
- [ ] Consistency loss (`docs/CDLM.md`): implemented, **no validated DIMBA speedup** — needs
      an experiment before any claim.
- [ ] MLX ≥0.30 (needs py≥3.10): the 2-row NaN chunking auto-lifts (version-gated); re-run
      the batch-4 parity test as the repro, then delete the workaround comment.

---

## 5. Verification protocol for the GPU box (4090/H100)

```bash
# 0) setup — needs the fused stack, or CUDA runs fail closed / fall back:
pip install -e ".[gpu]"          # mamba-ssm>=2.2.0, causal-conv1d>=1.4.0, liger-kernel
# 1) the headline claim: graphed sampler at the production shape, parity-gated:
python scripts/benchmark_h100.py --cases masked-inference --preset production \
    --checkpoint <ckpt> --allow-non-h100 --output graph_report.json
#    -> check report: promotion gate, p50/p95, cuda_graph.captured_graphs >= 1,
#       cuda_graph.eager_fallback == false, token_output_agreement.exact_match == true.
#       captured_graphs == 0 means capture silently failed -> the numbers are eager, not graphs.
# 2) A/B the CLI end to end (13.33s baseline shape):
python scripts/generate.py "What is the capital of France?" --backend torch --steps 128 --seed 0
python scripts/generate.py "..." --backend torch --steps 128 --seed 0 --no-graph   # compare
# 3) step-reduction lever (quality-affecting; record outputs for the quality probes):
python scripts/generate.py "..." --steps 128 --commit-threshold 0.9 --seed 0
# 4) training levers (short run, watch the printed tok/s line @ every 100 steps):
torchrun --standalone --nproc-per-node=1 scripts/masked_diffusion_finetune.py \
    --steps 300 --batch <max_that_fits> --compile
#    chunked CE is already default-on; raise --batch vs the pre-e243b98 ceiling to bank the win.
# 5) write results into docs/PERFORMANCE_AND_SCALING.md ("Measurements available now")
#    and docs/benchmarks.md Test D; mark superseded estimates as measured.
```

Notes: `--preset production` = batch 1, prompt 10, gen 40, 128 steps (the shape the 13.33s
came from). `--compile` benchmark runs are diagnostic-only and never promotable. On a 4090 use
`--allow-non-h100`; treat results as directional, the promotion gate formally wants H100.
Known risk: CUDA-graph capture through `mamba_ssm`'s Triton kernels is the untested step — on
failure `GraphedFn` warns once and runs eager (check `cuda_graph.eager_fallback`).

---

## 6. Environment gotchas (read before running anything)

- **This dev box is an Apple-Silicon Mac, Python 3.9** (system LibreSSL warning is harmless).
  MLX is capped at **0.29.3** by py3.9 — hence the 2-row NaN chunk workaround.
- Run tests as: `python3 -m pytest tests/ -q -p no:cacheprovider --override-ini addopts=`
  (the pyproject addopts want pytest-cov which may be absent). Full suite ≈ 90 s, **543 pass**
  as of `e243b98`.
- **Original Windows dev box**: `import torch` segfaults at interpreter teardown; use the
  project venv and end torch scripts with `os._exit(0)`. Syntax-check without torch via
  `python -m compileall src/dimba scripts tests`.
- **CI** (`.github/workflows/ci.yml`, py3.10/3.12 Linux) is the source of truth for runtime
  tests.
- **Apple Silicon training** = PyTorch **MPS** path, fp32, `PYTORCH_ENABLE_MPS_FALLBACK=1`,
  `latent_diffusion=True`, seq ≤ ~256 (`scripts/train_interactive.py`, preset `mps-small`).
  `backends/mlx/` is **inference-only**, never a training backend.
- **CUDA training fails closed** unless every mixer is fused `mamba_ssm.Mamba2` and
  `causal_conv1d` imports (`utils/backends.py:require_fast_cuda_mamba2`). This is deliberate.

## 7. Architecture and invariants (do not break these)

```
input_ids ─► token_embed ─► encode_latent (×latent_scale → ~unit variance)
        prompt (pooled, response-free) ─┐          add_noise / [MASK]-corrupt
        timestep_embed τ(t) ────────────┤              │
                                         ▼             ▼
                Mamba-2 denoiser (N bidirectional blocks, AdaLN-Zero/FiLM cond,
                optional per-block FFN)   [+ self-conditioning]
                                         ▼
                 raw pred → x̂₀ latent → decode_latent → output_head → logits
```

- **No conditioning leak.** Conditioning is the prompt only (pooled summary; prompt tokens
  kept clean in-sequence for masked/prompt_mask runs). NEVER condition on the clean target.
- **`forward()` returns the 3-tuple** `(x_pred, noise, latent_info)`.
- **`encode_latent`/`decode_latent` carry `latent_scale`** and round-trip exactly; everything
  that diffuses goes through them. Call `calibrate_latent_scale(batch)` in latent mode.
- Masked inference fast path: `predict_token_features(ids, t)` (fixed shape, graph-capturable)
  + `project_token_features(features, positions)` (shape-varying, stays eager). The CUDA-graph
  and CFG logic in `scripts/generate.py` depend on this split — keep it.
- fp32 precision islands are load-bearing everywhere: SSD scan log-decay masked to `-inf`
  BEFORE `exp` (no `inf*0=NaN`), fp32 CE accumulation under bf16, fp32 confidence softmax
  (bf16 reorders rankings), fp32 scan even under MLX fp16.
- Sync-hygiene: no `.item()`/`.any()` in hot loops (the corruption sampler and the sampler
  loop are deliberately sync-free; `--commit-threshold` is the one documented exception).
- `model.config` stores the full constructor config (used to rebuild EMA/replicas/MLX).
- The 2-tuple forward, positive-`A` SSM, un-scaled latents, and the conditioning leak are
  historical bugs — do not reintroduce.

## 8. Map of the code

```
src/dimba/
  models/       DIMBA (diffusion.py), Mamba2Denoiser+DenoisingHead (denoiser.py),
                TorchMamba2 (pure-PyTorch SSD scan, CPU/MPS fallback), vae.py, lora.py
  diffusion/    schedules, sampling.py (DDIM/DPM++/flow), masked_sampling.py (LLaDA loop),
                corruption.py (gaussian/absorbing/hybrid), rerank.py
  training/     masked.py (production loss), fused_ce.py (Liger + chunked CE),
                distributed.py (DDP), optimizers.py (AdamW/HybridMuon), trainer.py,
                preference.py (DPO/IPO/SimPO), grpo.py, rewards.py
  distillation/ MOHAWK-style cross-arch teacher→DIMBA (trainer.py)
  backends/mlx/ full Apple-GPU inference replica (model.py MLXDIMBA, mamba2.py scan)
  utils/        backends.py (CUDA enforcement), compile.py, cuda_graphs.py (GraphedFn)
  demo/         [UNTRACKED] YC demo TUI/HTML modules
scripts/
  generate.py               production CLI sampler (quality slider, CUDA graphs, verifiers)
  masked_diffusion_finetune.py / mdm_sft_cfg2.py   masked base / SFT launchers (DDP)
  train_h100.py             continuous distillation (phase-gated, Stage-3 contract)
  benchmark_h100.py         THE benchmark+promotion harness (--preset production)
  finetuning/  experiments/  diagnostics/  setup/  utils/
configs/      config.yaml + fineweb_{1.5b_l40s,500m_a4000}.yaml
tests/        ~55 files, 543 tests; test_speedups*.py cover the e243b98 pass
docs/         see §9;  paper/ = the v1 concept PDF (historical)
```

## 9. Map of the docs (authority order)

| Doc | Role |
|---|---|
| `docs/NEXT_RUN_PLAN.md` | **Authoritative** plan + contract for the next training run |
| `docs/ROOT_CAUSE_BUDGET28.md` | **Authoritative** Budget28 post-mortem (refutes "train longer") |
| `docs/PERFORMANCE_AND_SCALING.md` | Perf truth: measured vs estimate, kernel policy, all new flags |
| `docs/OVERHAUL_STATUS.md` | v2 ledger: done vs remaining |
| `docs/hr-diffuse-1-nano_system_card.md` | Released-model facts, recipe, evals |
| `docs/benchmarks.md` | QA/infill/latency tables (Test D latency predates the graph sampler) |
| `docs/BACKENDS.md` | CPU/MPS/MLX guidance, MLX version gates, fp16 measurement |
| `docs/BABYSIT_LOOP.md` | Operational brief for babysitting long GPU runs (/loop) |
| `docs/NEXT_RUN_PLAN` companions | `bidir_ab.md` (shared+LoRA wins), `slider_curve.md`, `writeup.md` |
| `docs/IMPROVEMENT_PLAN.md` | Historical v1 findings (2026-05-27) — rationale, not current truth |
| `docs/{CDLM,RESEARCH_DIRECTIONS,PROGRESSIVE_CHECKPOINTING}.md` | Research/aux, see their headers |
| `yc-hamiltonian-research-claude-code.txt` | Raw profiling session that killed the custom-kernel idea |

## 10. Training launch truth (condensed — full detail in the docs above)

- Continuous H100: `torchrun ... scripts/train_h100.py --phase distill` — phase-by-phase
  (`--phase all` rejected), Stage-3 Budget28 repair contract enforced, exact resume pins
  preset/optimizer/batch/topology/data. SFT/GRPO have NO exact resume and require a
  human-recorded `--quality-gate-passed`.
- Masked base/SFT: `torchrun --standalone --nproc-per-node=N scripts/masked_diffusion_finetune.py`
  (or `mdm_sft_cfg2.py`). `--batch` is per-GPU; global = `batch × N × accumulate`. Rank 0
  builds the shared data cache. New flags: `--compile` (opt-in); chunked CE is default-on.
- Stop Stage 3 safely by writing `{"stop": true}` to `training_state_override.json` and
  waiting for `distill_latest.pt`.
- Every optimizer arm gets its own `--save-dir`/`--output-dir`.

## 11. Conventions and definition-of-done

- Python ≥3.9, black line-length 100, type hints, Google-style docstrings. Match surrounding
  style; comments state constraints, not narration.
- Before claiming a change works: `compileall` + full pytest suite (543 green baseline) +
  the relevant parity/quality gate. CUDA perf claims additionally require the
  `benchmark_h100.py` gate; quality-affecting sampler changes require the quality probes,
  not token parity.
- Estimates vs measurements are kept honest in the docs — when you measure something, move it
  from the estimate table to the measured section and say which machine.
- Commit style: imperative summary + detailed body (see `git log`); end with the Claude
  co-author trailer when applicable.
