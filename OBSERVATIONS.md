# Observations — non-critical issues

Notes collected while babysitting the budget28 DIMBA run (stage-3 distillation).
None of these block training; the live run is healthy. Each entry says what it is,
why it's non-critical, and a concrete quick fix for later. **Not applied yet unless
marked APPLIED.**

Last updated: 30 Jun 2026 (UAE), during 3b co-adaptation.

---

## 1. `scripts/generate.py` silently runs a near-random model  ⚠️ highest value
**What:** `load_checkpoint()` builds the model from an external `config.yaml`
(default `config.yaml`) instead of the config embedded in the checkpoint
(`ck["config"]`), then calls `model.load_state_dict(model_state, strict=False)`.
With the stale config (see #2) the model is built at `d_model=256, vocab=10000`,
so almost none of the real 576-dim / 49,152-vocab weights load — `strict=False`
hides the mismatch. Result: it "generates" from an essentially random model and
exits 0. Also defaults to a fake `ord(c)%vocab` tokenizer.

**Why non-critical:** only affects the standalone generate/eval script, not
training or the saved checkpoints. The checkpoints themselves are fine (verified:
strict load 0 missing / 0 unexpected against the embedded config).

**Quick fix:**
- In `load_checkpoint`, prefer the embedded config:
  ```python
  ck = torch.load(path, map_location=device)
  if isinstance(ck, dict) and "config" in ck and "model_state_dict" in ck:
      import inspect
      sig = set(inspect.signature(DIMBA.__init__).parameters) - {"self"}
      kwargs = {k: v for k, v in ck["config"].items() if k in sig}
      model_state = ck["model_state_dict"]
  ```
- Use `strict=True` (or at least log/raise on non-empty missing/unexpected) so a
  mismatch is loud, not silent.
- Default the tokenizer to the teacher (`HuggingFaceTB/SmolLM-135M`) via
  `AutoTokenizer.from_pretrained(...)` when `--tokenizer` is omitted, instead of
  the `ord(c)` hack.
- Reference implementations that already do this correctly:
  `scratchpad/infer_real.py` (GPU) and `scratchpad/infer_cpu.py` (CPU).

## 2. `config.yaml` ships a CPU-test-sized model as the default
**What:** `config.yaml` has `model.d_model: 256  # reduced for CPU testing`,
`vocab_size: 10000`, etc. — a tiny smoke-test config, not the real run config
(`d_model=576`, `vocab=49152`, `num_denoiser_layers=30`). Anything that reads this
file as "the model" (e.g. #1) gets the wrong architecture.

**Why non-critical:** the real training path passes config via the preset/CLI, not
this file; the checkpoint carries its own config.

**Quick fix:** either (a) update `config.yaml` to the real budget28 architecture,
or (b) add a clear header comment that it is a CPU smoke-test config and tools
should prefer the checkpoint's embedded config. Lowest-risk: option (b) plus #1.

## 3. No CPU-inference flag exposed, even though a pure-torch path exists
**What:** `DIMBA(..., force_torch_mixer=True)` (also `use_simple_mamba=True`) swaps
the CUDA-only `causal_conv1d`/SSD kernel for the pure-PyTorch `TorchMamba2`, which
runs on CPU/MPS. `generate.py` never exposes this, so a CPU run dies with
`RuntimeError: Expected x.is_cuda() to be true` and looks like a hard limitation
when it isn't.

**Why non-critical:** GPU inference works; this is a usability gap, not a failure.

**Quick fix:** add a `--device cpu` path (or `--force-torch-mixer` flag) to
`generate.py` that sets `force_torch_mixer=True` when device is not CUDA. Verified
working in `scratchpad/infer_cpu.py` (strict 0/0, real English on CPU).

## 4. `boundary_relaunch.sh` verify used a CPU forward smoke-test on a CUDA-only arch  ✅ APPLIED (local)
**What:** the original `verify` step ran a CPU `generate.py` forward pass as the
"is the checkpoint broken?" gate. On this arch a CPU forward can't run (CUDA-only
kernel; see #3), so the gate failed on a perfectly good checkpoint.

**Status:** patched locally (commit `168d527`, **not pushed**) to detect the
`is_cuda()/causal_conv1d` signature and fall back to a CPU integrity check
(deserialize + key/shape + NaN/Inf scan).

**Better long-term fix:** make the smoke-test use `force_torch_mixer=True` so it
can do a real (if slow) CPU forward pass instead of only an integrity scan — or run
the smoke test on GPU when one is free.

## 5. `nvidia-smi` PID ≠ container PID (monitoring foot-gun)
**What:** inside this container `nvidia-smi --query-compute-apps=pid` reports the
**host-namespace** PID (e.g. 7672) while the same process is a different PID inside
the container (e.g. 294841). Any monitor that matches the nvidia-smi PID against a
container-side PID/pidfile will think "the trainer isn't the one on the GPU."

**Why non-critical:** cosmetic/observability; doesn't affect training. Tripped me
up for a minute during a health check.

**Quick fix:** in monitoring, don't cross-match nvidia-smi PIDs to container PIDs.
Identify the trainer by `pgrep -f train_h100.py` and confirm GPU use via
`utilization.gpu` + `memory.used`, not by PID equality. Optionally map via
`/proc/<container_pid>/status` `NSpid` if an exact mapping is ever needed.

## 6. `generate.py` detokenize fallback drops non-ASCII
**What:** the no-tokenizer fallback only keeps `32 <= id < 127`, so with a real
vocab everything prints as `<non-printable tokens>`. Subsumed by fixing #1 (use the
real tokenizer), noted for completeness.

**Quick fix:** resolved automatically once #1 defaults to the SmolLM tokenizer.

---

### Suggested order when we get to it
1. #1 + #2 + #6 together (one PR on `generate.py` + a `config.yaml` comment) — biggest payoff.
2. #3 (`--device cpu` / `--force-torch-mixer`) — small, enables CPU eval anywhere.
3. #4 long-term (torch-mixer smoke test) — optional; local patch already unblocks the run.
4. #5 — doc note in the monitor tooling.

Reference scripts (correct, working): `scratchpad/infer_real.py`, `scratchpad/infer_cpu.py`.
