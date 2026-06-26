# DIMBA — Next-Run Plan (token budget + FFN schedule + go/no-go gates)

**Status:** post-mortem of run #1 complete. Run #1 (Stage-3 co-adaptation on ~1B
tokens with the FFN **frozen the whole time** → SFT → GRPO) produced an incoherent
model. Two independent root causes:

1. **Token-starved base** — ~1B co-adaptation tokens. The inherited SmolLM FFN
   encodes ~600B tokens of knowledge, but the *new* Mamba-mixer ↔ FFN interface
   never matured on only 1B tokens.
2. **FFN never adapted in the base** — Stage 3 ran `freeze_ffn=True` for its entire
   duration; the original plan's later low-LR FFN-unfreeze stage (“Finetune #2”) was
   dropped. (SFT later trained the FFN, but on limited instruction data, after the
   base was already weak.)

GRPO accuracy stayed ~0 because there was nothing coherent to optimize.

This plan fixes the recipe and gates the scale-up on measured coherence.

> **Hardware:** see the main agent's recommendation. (Token budgets below are
> hardware-agnostic; wall-clock estimates assume the run-#1 throughput of ~1B
> co-adaptation tokens in ~5–6 h on the fast CUDA Mamba-2 kernel.)

---

## 1. What the literature actually used (evidence for the budget)

DIMBA is an **attention→Mamba conversion that inherits the teacher's FFN /
embeddings / head**. The right reference class is *cross-architecture distillation /
conversion*, **not** from-scratch pretraining. Token counts actually used by that
reference class:

| Work | Conversion | Distillation tokens | Notes |
|---|---|---|---|
| **MOHAWK / Phi-Mamba** (Bick et al., NeurIPS 2024, [arXiv:2408.10189](https://arxiv.org/abs/2408.10189)) | Phi-1.5 (1.5B) attn → Mamba-2 | **3B total** (80M stage-1 + 160M stage-2 + **2.76B stage-3**), C4, seq 2048 | <1% of the 315B tokens used for from-scratch Mamba/Mamba-2. Hybrid-Phi-Mamba-1.5B: **5B**. |
| **The Mamba in the Llama** (Wang et al., NeurIPS 2024, [arXiv:2408.15237](https://arxiv.org/abs/2408.15237)) | Zephyr/Llama-3-8B attn → Mamba (hybrid) | **~20B total** | A variant trained on only **3B tokens** already beats Mamba-7B trained from scratch on 1.2T. 50–175× fewer tokens than from-scratch (1.2T–3.7T). |
| **Llamba** (Bick et al., 2025, [arXiv:2502.14458](https://arxiv.org/abs/2502.14458)) | Llama-3 → pure Mamba-2, MOHAWK | **Llamba-1B: 8B · Llamba-3B: 10B · Llamba-8B: 12B** | fineweb-edu (matrix+hidden) + OpenHermes-2.5 (KD, 4×200M). <0.1% of from-scratch data. **Smaller model → more tokens/param** (1B got 8 tok/param; 8B got 1.5). |
| **Data-Efficient Transformer-to-Mamba** ([arXiv:2510.19266](https://arxiv.org/abs/2510.19266)) | any Transformer → Mamba, attention bridge | **2B–4B**, two-stage | data-efficiency-focused. |

**Teacher / from-scratch anchors:**

- **SmolLM-135M** (our teacher) trained on **600B tokens** (600k steps, 64×H100) over
  Cosmo-Corpus = Cosmopedia-v2 (28B) + Python-Edu (4B) + FineWeb-Edu (220B)
  ([SmolLM blog](https://huggingface.co/blog/smollm)). At 600B/135M ≈ **~4,400
  tokens/param** — i.e. ~220× past Chinchilla.
- **Chinchilla** compute-optimal ≈ **20 tokens/param** (Hoffmann et al. 2022,
  [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)). For 135M that's only ~2.7B
  tokens **from scratch**.
- Modern small-model practice trains **far past Chinchilla** for inference efficiency
  (e.g. Llama-3-70B ≈ ~200 tok/param ≈ 10× Chinchilla; SmolLM-135M ≈ 220×).

### Reasoning: how much does *DIMBA* need?

- **Chinchilla is the wrong frame** here. It's a from-scratch compute-optimal law; we
  *inherit* a 600B-token FFN/embeddings/head. Its ~2.7B-for-135M number is a floor for
  the wrong question. The conversion reference class is the relevant anchor.
- **Conversion floor ≈ 3–20B tokens, WITH the FFN co-adapting end-to-end.** Every
  cited conversion (MOHAWK 3–5B, Llamba 8–12B, Mamba-in-Llama 20B) trains the whole
  model (mixer **and** FFN/output) in its final stage and recovers most teacher
  quality. None freezes the FFN for the entire run.
- **DIMBA is strictly harder than those.** On top of attn→Mamba it *also* changes:
  (a) the **objective** — autoregressive next-token → **diffusion / flow-matching**;
  (b) the **direction** — causal → **bidirectional**. Diffusion dilutes the per-token
  learning signal (each token is trained at many noise levels; the effective gradient
  per *unique* token is a fraction of AR's). So DIMBA should need **more** tokens than
  a pure AR attn→Mamba swap, not fewer.
- **Run #1 was undertrained for two compounding reasons:** 1B is below even the
  MOHAWK 3B floor, *and* the FFN was frozen the whole time (the cited conversions
  never do this). The incoherence is fully explained — we don't need to suspect the
  architecture yet.

**Budget conclusion (with ranges, because this is a novel diffusion-Mamba
conversion — see §6):**

- **Floor to re-bed the new mixer into inherited knowledge:** ~**5–10B** tokens
  (conversion band 3–20B, padded upward for diffusion signal-dilution).
- **Practical sweet spot to mature the new architecture:** ~**20–50B** tokens.
- **Full-exploit upper bracket (the user's 50–100B guess):** plausible as a
  **ceiling**, but likely diminishing returns. **Verdict: refine the 50–100B guess
  down.** The evidence says the *target* is ~20–50B, with 50–100B justified only if
  the 5B and 20B gates keep improving monotonically. The minimum needed merely to
  *test the hypothesis* is ~5–10B.

> **Data-uniqueness caveat (important):** the current Stage-3 stream caches
> `n_cache=400_000` docs ≈ ~290M **unique** tokens. At 1B that's ~3.4 epochs (fine);
> at 5B it's ~17 epochs and at 20B+ it's pathological repetition. **For any budget
> >~1B, raise `n_cache` (or stream a larger FineWeb slice — `sample-10BT` has 10B
> unique tokens)** so token *count* is backed by token *diversity*. Repeating 290M
> tokens 17× is not 5B of learning.

---

## 2. The fix: FFN frozen → unfrozen co-adaptation schedule

Stage 3 is now **two phases** (parameterized in `scripts/train_4090.py`):

| Phase | FFN | LR | Share of co-adapt tokens | Purpose |
|---|---|---|---|---|
| **3a** | **FROZEN** | `STAGE3_FROZEN_LR = 2e-4` | ~60–70% | MOHAWK-faithful: only the Mamba mixer trains, learning to emit activations the inherited FFN already consumes. |
| **3b** | **UNFROZEN** (“Finetune #2”) | `STAGE3_UNFROZEN_LR = 3e-5` | ~30–40% | The dropped step. FFN co-adapts to the now-matured mixer at **low** LR (~1/7 of 3a) so the inherited 600B-token knowledge is **not washed out**. |

Knobs (one or two numbers to change per scale):

```python
STAGE3_FROZEN_TOKENS   = 3_000_000_000   # 3B  (phase 3a)
STAGE3_UNFROZEN_TOKENS = 2_000_000_000   # 2B  (phase 3b)   → default total ≈ 5B
STAGE3_FROZEN_LR   = 2e-4
STAGE3_UNFROZEN_LR = 3e-5
```

- Either knob set to `0` **skips** that phase (e.g. `STAGE3_FROZEN_TOKENS=0` →
  pure unfrozen continuation of an already-co-adapted base — see §4).
- Steps are derived from tokens via `_steps_for_tokens()` (16,384 tok/step on the
  CUDA kernel = `PRETRAIN_BATCH × PRETRAIN_SEQ_LEN`).
- `run_distill` runs phases in order, **checkpoints after 3a** so a crash in 3b
  doesn't lose the frozen base, and logs the realized token total.

**Verified in tests** (`tests/test_distillation.py`):
`test_stage3_unfrozen_ffn_receives_gradients` — with `freeze_ffn=False` the FFN params
are trainable **and actually change** after optimisation. `test_stage3_frozen_ffn_stays_frozen`
— with `freeze_ffn=True` the FFN is left **exactly unchanged** while the mixer still
trains. (Full suite: 366 passed.)

### Secondary base-quality fix: timestep sampling

Stage-3 previously sampled timesteps with a plain uniform `randint`, **bypassing** the
model's configured `flow_logit_normal=True` (SD3/FLUX logit-normal) schedule — a
footgun explicitly flagged in `FlowMatchingSchedule.sample_timesteps`'s own docstring.
Uniform sampling starves the **mid-noise** band where token content is actually
decided. Fixed in `src/dimba/distillation/trainer.py::_stage3_step`: when the model is
flow-matching + logit-normal, Stage 3 now draws logit-normal **integer** timesteps via
the existing, tested `noise_schedule.sample_timesteps(..., mode="logit_normal")`
(falls back to the old uniform behaviour otherwise, so nothing else changes).

> Known lower-priority follow-up (not changed, to stay conservative): SFT and GRPO
> still sample timesteps uniformly (`sample_timesteps` in `diffusion/sampling.py`).
> The base is where coherence is won, so the Stage-3 fix is the high-value one.

---

## 3. Per-stage token budget + go/no-go gates (the validation ladder)

The model is always 135M params. The “stages” below are **escalating token budgets**
with a **gate** between each — cheap → expensive, stop early if a gate fails. Run them
in order; do not skip a gate.

| Stage | Co-adapt tokens (3a frozen / 3b unfrozen) | ~Steps (CUDA batch) | ~Cost | **GO/NO-GO gate** |
|---|---|---|---|---|
| **S0 — Smoke / plumbing** | **135M** (100M / 35M) | ~8.2k | ~1–2 GPU-h | Loss decreases monotonically; no NaN/Inf; FFN receives gradients in 3b; output is not pure-repeat garbage. *Plumbing only, not a quality gate.* |
| **S1 — Validation (= run-#1 budget)** | **1B** (700M / 300M) | ~61k | ~5–6 GPU-h | **Clearly better than run #1 at equal budget** — less degenerate repetition, lower held-out diffusion loss / perplexity. Isolates “did the recipe fix help?” |
| **S2 — Scale-test (DECISION gate)** | **5B** (3B / 2B) — *current default* | ~305k | ~25–30 GPU-h | **Coherent ≥30-token continuations** on held-out prompts (grammatical, on-topic, low repetition) **AND** held-out perplexity within ~1.3–1.5× the SmolLM-135M teacher on the same set (relative signal — DIMBA is bidirectional-diffusion, exact PPL parity is *not* expected) **AND** `eval_vs_smollm.py` shows DIMBA non-trivially close to teacher. |
| **S3 — Full run (post-gate)** | **20–50B** (~60–70% frozen / ~30–40% unfrozen); up to ~100B only if gates keep improving | ~1.2–3.0M | days | Proceed to SFT → GRPO. |

**Gate logic:**
- **S0 fail** → pipeline bug; fix before spending tokens.
- **S1 fail** (no better than run #1) → the recipe change isn't the lever; re-examine
  surgery / init / loss before scaling.
- **S2 GO** → the undertraining hypothesis is confirmed; commit to S3.
- **S2 NO-GO** (5B still incoherent) → the problem is **not just tokens**. Do **not**
  burn the big run. Investigate: bidirectional-diffusion objective scale, latent/VAE
  calibration, principled init, or `use_flow_matching` schedule. (See `RESEARCH_DIRECTIONS.md`.)

Don't bother enumerating SFT/GRPO budgets until S2 is GO — the coherence-gate guards
in `run_sft`/`run_grpo` now warn that those phases can't rescue an incoherent base.

---

## 4. Validation-FIRST step (do this before the big run) — CONFIRMED

The recommendation on the table — **continue the EXISTING base for ~5–10B more tokens
with the FFN unfrozen** — is the right cheap test of the undertraining hypothesis.
**Confirmed, with a refinement to make it even cheaper to read:**

1. **Resume the existing run-#1 base checkpoint** (it already had its frozen phase, so
   skip 3a):
   ```python
   STAGE3_FROZEN_TOKENS   = 0
   STAGE3_UNFROZEN_TOKENS  = 8_000_000_000   # 5–10B, FFN unfrozen, low LR
   ```
   ```bash
   python scripts/train_4090.py --phase distill --resume checkpoints/distill/final.pt
   ```
   (Also raise `n_cache` so 8B isn't ~27 epochs of 290M tokens — see §1 caveat.)
2. **Read coherence early, at the ~2–3B mark.** If continuations are *clearly
   improving* (less repetition, more on-topic), continue to the full 5–10B. If they're
   **flat**, stop — the issue isn't tokens, and you've spent ~3B instead of 10B
   discovering it.
3. If the continuation reaches coherence → you've de-risked the big run and can go
   straight to S3 with confidence. If it improves but plateaus short of the S2 gate →
   the budget needs to go higher (20–50B) and/or data diversity needs raising.

This is the single most informative ~$ you can spend: it discriminates
**“undertrained” vs “architecturally broken”** before committing compute.

---

## 5. Concrete recommended numbers (summary)

- **Immediate:** validation continuation, **5–10B unfrozen tokens** on the existing
  base (§4), early coherence read at ~2–3B.
- **If/when doing a fresh run:** default config = **S2 scale-test, 5B** (3B frozen +
  2B unfrozen). This is the gate that decides the big run.
- **Full run (after S2 GO):** **20–50B** (target), 50–100B only if gates keep
  improving. Frozen:unfrozen ≈ 60–70 : 30–40.
- **LRs:** frozen 2e-4, unfrozen 3e-5.
- **Data:** raise `n_cache` / use a larger FineWeb slice for any budget >~1B.

---

## 6. Confidence & what I'm unsure about (honest)

- **Medium-high confidence** that run #1's incoherence is *undertraining + frozen FFN*,
  not a fundamental architecture flaw: the budget was below the conversion floor on
  two axes, and the literature is consistent.
- **Medium confidence** on the absolute numbers. The 5–10B floor and 20–50B target are
  **extrapolations** from AR attn→Mamba conversions (3–20B), adjusted upward by
  judgement for the diffusion + bidirectional objective change. **No published work
  converts a Transformer into a *diffusion* Mamba LM**, so there is no direct anchor —
  the diffusion signal-dilution factor (how much more than AR) is a genuine unknown. It
  could be ~1.5× (→ target ~15–30B) or ~3× (→ target ~40–60B). The gated ladder exists
  precisely to measure this instead of guessing.
- **Lower confidence** on the exact frozen:unfrozen split (60–70/30–40) and the
  unfrozen LR (3e-5). These are reasoned defaults (low-LR continued-pretraining
  practice), not measured optima; worth a small sweep at S2 if budget allows.
- **Caveat I can't rule out:** if S2 (5B) is still incoherent, the cause may be on the
  objective/latent side (bidirectional diffusion, VAE/latent scale, init) rather than
  tokens. The S2 NO-GO branch covers this.

---

### Change log (code backing this plan)
- `scripts/train_4090.py` — token-budget knobs + `_steps_for_tokens`/`_coadapt_stages`;
  two-phase Stage 3 (frozen→unfrozen) with intermediate checkpointing; coherence-gate
  warnings in `run_sft`/`run_grpo`.
- `src/dimba/distillation/trainer.py` — logit-normal Stage-3 timestep sampling for
  flow-matching models.
- `tests/test_distillation.py` — FFN-trains-when-unfrozen / FFN-frozen-stays-frozen tests.
