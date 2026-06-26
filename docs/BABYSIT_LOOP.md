# DIMBA training babysitter — complete agent brief

This is a **self-contained `/loop` prompt**. A fresh Claude agent, given only this
text, will **launch** the DIMBA 50B training run, **monitor** every phase, **apply
live adjustments**, **self-heal simple errors**, and **escalate** anything risky.

## How the human starts it (one time, on the GPU box)

1. Open a Claude Code session **on the GPU box** (where the GPU, the repo, and
   `nvidia-smi` live). The repo is at `~/dimba-lib-exp` and the box `git pull`s
   from `main`.
2. Fill in the **CONFIG** block below (at minimum `HF_REPO`).
3. Run `/loop 45m` and paste **this entire file**. That's it — the agent does the
   rest, including launching training on its first tick. Cadence 30–60 min is
   right (`30m` early in a phase, `60m` once stable).

```
# ===== CONFIG (the human fills these in before pasting) =====
REPO_DIR   = ~/dimba-lib-exp
PRESET     = full              # full = 50B target. SEE "Which preset" below — running
                               #   `validation` (~1B, a few $/hours) FIRST is strongly
                               #   recommended to sanity-check the recipe before 50B.
PHASE      = all               # all = distill -> SFT -> GRPO (the full pipeline)
HF_REPO    = <you>/dimba-135m  # HuggingFace repo for checkpoint backups (REQUIRED for
                               #   off-box durability; leave blank to skip uploads)
HF_TOKEN   = (env HF_TOKEN)    # set `export HF_TOKEN=hf_...` in the shell before /loop
LOG        = ~/dimba_train.log
PIDFILE    = ~/dimba_train.pid
LOOPSTATE  = ~/dimba_loop_state.json   # the agent's own memory between ticks
# ============================================================
```

---

## THE PROMPT (everything below is for the babysitter agent)

You are the **babysitter** for a long, expensive (multi-day, single-H100) training
run. Read this whole brief once, then on each tick do: **figure out the situation →
act → report → persist your notes → stop**. The `/loop` harness re-invokes you every
30–60 min, so each tick is short and you rely on `LOOPSTATE` to remember context.

### 0. What you are training (so you understand what you're watching)

- **DIMBA** is a ~135M-parameter **bidirectional Mamba-2 *diffusion* language model**.
  It is NOT a normal autoregressive transformer: it denoises tokens (diffusion /
  flow-matching) rather than predicting strictly left-to-right, and its token-mixing
  is a Mamba-2 state-space layer, not attention.
- We **distilled** it from **SmolLM-135M** (a strong small transformer) using a
  MOHAWK-style conversion: keep SmolLM's embeddings + FFN/MLP + LM head, replace
  attention with a bidirectional Mamba mixer, then align and co-adapt.
- **Why this run exists:** run #1 came out incoherent. Root causes — now fixed —
  were (a) too few tokens (~1B), (b) the inherited FFN stayed **frozen** the whole
  time so it never adapted to consuming Mamba (not attention) outputs, and (c) a
  timestep-sampling bug. This run fixes all three and scales tokens up.
- **Success looks like:** the distilled base produces *coherent* text continuations;
  SFT then teaches instruction/chain-of-thought format; GRPO sharpens reasoning.

### 0.1 The three phases (what each is, in order)

The pipeline is `distill → SFT → GRPO`. With `PHASE=all` it runs them back-to-back.

1. **DISTILL** (the long one, where the 50B tokens go). Internally three stages:
   - **Stage 1 — matrix alignment**: the Mamba mixer learns to mimic SmolLM's
     attention mixing matrices. Short, teacher active. ~hundreds of steps.
   - **Stage 2 — hidden alignment**: each block's hidden state is matched to
     SmolLM's. Short, teacher active.
   - **Stage 3 — co-adaptation pretraining** (THE 50B): plain language modelling on
     FineWeb, teacher unloaded. Two sub-phases:
       * **3a, FFN-frozen** — 33B tokens, lr 2e-4, ~503,540 steps. The Mamba mixer
         learns to feed the (frozen) inherited FFN in-distribution.
       * **3b, FFN-unfrozen** — 17B tokens, lr 3e-5, ~259,399 steps. The FFN now
         co-adapts at a low LR (the fix for run #1). A checkpoint
         `distill_stage3a.pt` is saved at the 3a→3b boundary.
   - Output: `checkpoints/distill/final.pt`.
2. **SFT** — supervised fine-tuning, 2 stages (full instruction+math mix, then a
   hard reasoning-only subset), teaching the `<think>…</think>` block-CoT format.
   Output: `checkpoints/sft/final.pt`. **SFT cannot fix an incoherent base** — see
   the COHERENCE GATE.
3. **GRPO** — reinforcement learning on verifiable rewards (default: math). Watches
   reward, KL divergence, and "think-block" counts. Output: `checkpoints/grpo/…`.

### 0.2 Files, signals, and terms you will use

- `LOG` (`~/dimba_train.log`): training stdout. Your primary raw signal. Key lines:
  - `distill backend: CUDA Mamba2 kernel (fast binary)` ← MUST see this (see 3.B).
  - `DistillationTrainer [stage3] step N/M — loss=X` ← distill progress (~every 50 steps).
  - `GRPO[label] step N/M | reward=… | think=… | kl=… | acc=… | lr=…` ← GRPO progress.
  - `distillation done → …/distill/final.pt`, `SFT done → …`, `all done. final model: …`.
- `scripts/monitor.py`: run `python3 scripts/monitor.py`. Reads `training_state.json`
  (now written in **every** phase, distill included) and prints phase, step, loss,
  trends, and concrete recommendations. This is your formatted dashboard.
- `training_state.json`: `{stage, step, loss, lr, …}`, refreshed each log interval.
- `training_state_override.json`: **you** write this to adjust the live run (see §5).
  The trainer reads + DELETES it (applies once).
- `LOOPSTATE` (`~/dimba_loop_state.json`): **your** memory between ticks. You create
  and update it. Suggested shape:
  `{"situation","last_phase","last_step","last_loss","last_ts","fix_attempts":{},"launched":true}`.
- `checkpoints/{distill,sft,grpo}/`: saved weights. NEVER delete a `final.pt` or
  `distill_stage3a.pt`. Intermediate `*_step*.pt` are deletable to free disk.

### 1. FIGURE OUT THE SITUATION (do this first, every tick)

Run these, then pick exactly one branch:
```bash
cd <REPO_DIR>
ALIVE=no; [ -f <PIDFILE> ] && ps -p "$(cat <PIDFILE>)" >/dev/null 2>&1 && ALIVE=yes
tail -n 5 <LOG> 2>/dev/null
```
- **DONE** — `LOG` contains `all done. final model:` → the run finished. Post a final
  summary, confirm `grpo`/final checkpoints exist + uploaded, and **end the loop**.
- **RUNNING** — `ALIVE=yes` → go to **§2 (monitor)**.
- **CRASHED/STOPPED** — `ALIVE=no` and `LOG` exists and does NOT say `all done`
  (look for a traceback) → go to **§4 (error recovery)**.
- **KICKOFF** — no `PIDFILE`/`LOG` yet, or `LOOPSTATE.launched` is not set → this is
  the very first tick → go to **§1.5 (launch it)**.

### 1.5 KICKOFF — launch training (first tick only)

Do these in order; if any check fails, treat it as an error (§4) or escalate.
```bash
cd <REPO_DIR>
git pull                                   # get the latest code (config, fixes)
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader   # expect an H100, ~80GB
python3 -c "import mamba_ssm, causal_conv1d" \
  || MAX_JOBS=4 pip install mamba-ssm causal-conv1d --no-build-isolation
python3 scripts/train_h100.py --preset <PRESET> --dry-run        # sanity: 50B for full
```
- If VRAM < ~70 GB → ALERT the human (this isn't a full H100; SFT batch may OOM).
- Launch detached + logged. Include `--hf-repo` only if `HF_REPO` is set; if it's
  blank, WARN that checkpoints won't be backed up off-box.
```bash
export HF_TOKEN=<HF_TOKEN>     # if set
nohup python3 scripts/train_h100.py --preset <PRESET> --phase <PHASE> \
    ${HF_REPO:+--hf-repo <HF_REPO>} > <LOG> 2>&1 &
echo $! > <PIDFILE>
sleep 30
```
- Confirm it started: process alive (`ps -p $(cat <PIDFILE>)`) AND `LOG` shows the
  banner (`=== PHASE 1: DISTILLATION`). If it died in 30s → §4 with the traceback.
- **Verify the fast kernel** from the log: it must say `CUDA Mamba2 kernel (fast
  binary)`, NOT `TorchMamba2 (...fallback)`. Fallback = ~10× slower + OOM-prone →
  §4 (fix = the `pip install` above, then relaunch).
- Set `LOOPSTATE.launched = true`, record phase/step/ts, and post a "🚀 STARTED"
  report. Done for this tick.

### 2. MONITOR (when RUNNING) — observe progress
- `python3 scripts/monitor.py` — note phase, step, loss, its trends, and any
  recommendations it prints.
- `tail -n 30 <LOG>` — read the latest raw trainer lines (cross-check the monitor).
- `nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader`.
- Compute **steps/min** from `(step − last_step)/(now − last_ts)` and an **ETA** for
  the remaining steps of the current stage (frozen ≈503,540; unfrozen ≈259,399;
  use `--dry-run` numbers; SFT/GRPO are far shorter).

### 3. HEALTH CHECKS (flag ANY as an issue → §4 or §5 as noted)
- **A. Loss**: must be finite and trending **down** or flat-low. There's no hard
  target number — judge by the trend. `NaN`/`Inf`, or a sustained 2×+ rise over
  several readings → **divergence**: first try a live LR halving (§5); if it keeps
  diverging next tick → escalate.
- **B. Backend** (verify once): `CUDA Mamba2 kernel (fast binary)` present. Fallback
  → §4.
- **C. Stalled**: step unchanged vs `last_step` AND GPU util low → hang → §4.
- **D. Stage-3 OOM**: `grep -c "CUDA OOM" <LOG>`. One early downshift is BY DESIGN
  (batch halves, steps auto-scale to keep the token budget). Continuous downshifting
  → escalate.
- **E. SFT OOM skips**: `grep -c "CUDA OOM on a micro-batch" <LOG>`. A few = fine;
  approaching 20 consecutive = the SFT batch is too big and the run will abort →
  escalate ("lower H100_SFT_BATCH_SIZE, restart SFT").
- **F. Disk**: `df -h .`. >90% on the checkpoints filesystem → the next save crashes.
  You MAY autonomously delete old intermediates: keep the 2 newest `*_step*.pt` per
  phase, NEVER touch `final.pt` / `distill_stage3a.pt`. Re-check and report.
- **G. Checkpoints landing**: `ls -la checkpoints/*/`. A phase's `final.pt` must
  appear (and upload to HF if `HF_REPO` set); `distill_stage3a.pt` at the 3a→3b
  boundary.

### 4. ERROR-RECOVERY SUBROUTINE (crash / traceback / hang / red test / failed command)
Do EXACTLY this, in order — this is the heart of "self-heal then escalate":
1. **Capture** the error: `grep -nE "Traceback|Error|Exception|CUDA|Killed" <LOG> | tail`
   then read around it. Form a short `error_sig` = exception type + the file:line.
2. **Pull new code first** (a fix may already be on `main`):
   `git stash -u 2>/dev/null; git pull --rebase; git stash pop 2>/dev/null || true`.
3. **Test** the code is healthy: `python3 -m compileall -q scripts/ src/` plus the
   most relevant suite (`pytest tests/test_distillation.py -q -o addopts=""` for
   trainer errors; `tests/test_override.py` for monitor/override; etc.).
4. **If healthy after the pull** → relaunch from the latest checkpoint (§4a),
   report "recovered via git pull", done.
5. **If still broken**, judge: is this a **SIMPLE FIX**? (see Definitions.)
   - **NOT simple** → **ALERT THE HUMAN**: paste the traceback, your diagnosis, and
     why it's beyond a simple fix. Do NOT edit code. Record it, stop.
   - **Simple** → fix it, **max 5 distinct attempts total** for this `error_sig`
     (track the count in `LOOPSTATE.fix_attempts[error_sig]` across ticks):
       a. Make the **minimal** edit (one clear hypothesis).
       b. Verify: `compileall` + the relevant test.
       c. **Fixed** → `git add -A && git commit -m "loop: fix <error_sig>"`, relaunch
          (§4a), report what you changed, then **try** `git push` (if blocked, say so
          — the fix is already on the box's working tree so the run gets it; the
          human pushes later). Done.
       d. **Not fixed** → increment the counter, try a *different* hypothesis (never
          repeat a failed edit). At 5 attempts → **ALERT THE HUMAN** with everything
          you tried. Stop fixing; keep monitoring on later ticks.

**4a. Relaunch after a fix** — resume from the crashed phase's latest checkpoint:
```bash
export HF_TOKEN=<HF_TOKEN>
nohup python3 scripts/train_h100.py --preset <PRESET> --phase <crashed_phase> \
    --checkpoint "$(ls -t checkpoints/<crashed_phase>/*.pt | head -1)" --resume \
    ${HF_REPO:+--hf-repo <HF_REPO>} > <LOG> 2>&1 &
echo $! > <PIDFILE>
```
⚠️ Resume is **coarse** (re-runs the current stage from its start, not the exact
step). If the crash was **deep into an expensive stage** (e.g. >30% through the 33B
frozen distill), the re-run wastes a lot of compute → **prefer to ALERT with the
fix committed and ready**, and let the human decide whether to eat the re-run.

### 5. LIVE ADJUSTMENTS (override file — apply a monitor recommendation)
The trainers consume `training_state_override.json` once (then delete it). Keys:
- `"lr"` (float) — any phase. e.g. halve a diverging Stage-3 lr.
- `"kl_coeff"` (float) — GRPO, if monitor flags `kl > 0.5`.
- `"thinking_length_weight"` (float) — GRPO, for overthinking / negative reward.
- `"stop"` (bool) — end the current phase gracefully (saves, moves on).
```bash
echo '{"lr": 1.0e-4}' > training_state_override.json
```
Apply ONE change only when clearly warranted, and **report it**. Observe its effect
next tick before changing anything else. These are reversible nudges; a code fix or
a restart is NOT — those follow §4.

### 6. GATES (decision points — surface to the human)
- **COHERENCE GATE** — when `LOG` shows `distillation done → …/distill/final.pt`.
  SFT/GRPO **cannot rescue an incoherent base**. With `PHASE=all`, SFT starts
  automatically, so **ALERT NOW**: tell the human to sample a few continuations from
  `distill/final.pt` (e.g. `scripts/generate.py` or `scripts/eval_vs_smollm.py`; the
  checkpoint embeds its config, tokenizer = SmolLM) and decide: let SFT continue, or
  write `{"stop": true}` to halt. Coherence is the human's call — you flag the moment.
- **GRPO** — reward should trend up, KL stay < 0.5, mean think-blocks ≤ ~1.5
  (monitor.py uses think>1.8, kl>0.5, reward<−0.2 as alarms). Relay its
  recommendations and apply the safe ones via §5.

### 7. REPORT (one concise block each tick)
```
[14:20] phase=stage3(frozen) step=120480/503540 (24%) loss=4.21 (↓ from 4.30)
        gpu=92% mem=63/80GB ~45 steps/min ETA≈2.4h(stage) disk=41% ckpts=OK
        ACTIONS: none | HEALTHY
```
Then any ALERTS with the exact remediation. If healthy + unchanged: one line.

### 8. PERSIST + STOP
- Write `LOOPSTATE` with the current `{situation, last_phase, last_step, last_loss,
  last_ts, fix_attempts, launched}` for the next tick.
- End the loop on **DONE** (`all done`) or when you've escalated a blocker and are
  waiting on the human.

### Definitions & autonomy boundaries (read carefully)
- **SIMPLE FIX** (you MAY do, ≤5 tries): localized, obvious-cause, contained — a
  traceback at one line (typo, wrong dict key, missing None-check, bad import / path
  / CLI arg), a missing dependency (`pip install …`), a disk/permissions issue, a
  stale checkpoint path. Reversible and fully understood.
- **NOT SIMPLE** (ALWAYS escalate, never edit): the model architecture, the
  loss/recipe/math, numerical divergence a single LR cut didn't fix, multi-file
  refactors, CUDA/driver/hardware faults, or anything you don't fully understand.
- **You MAY autonomously**: read logs/state, `nvidia-smi`/`df`/`ps`/`ls`/`grep`,
  `git pull`, run tests, launch/relaunch training per §1.5/§4a, delete safe
  intermediate checkpoints (§3F), write ONE override (§5), apply a simple verified
  fix + commit locally (§4) — **except** a costly deep-stage re-run (escalate).
- **You MUST escalate (alert, then wait)**: not-simple errors, 5 exhausted attempts,
  persistent divergence, expensive deep-stage restarts, pushing to `main` (commit
  locally and say so), VRAM < 70GB, and anything ambiguous on this expensive run.
- Default disposition: **when unsure, alert — don't act.** A wrong autonomous action
  can waste GPU-days; a missed alert just waits for the human.

### Which preset (a note for the human, surfaced once)
`full` = 50B is the real target but multi-day and costly. The recommended path is to
run `validation` (~1B, a few hours / a few dollars) first as a recipe check vs run
#1, then `scale` (~5B) as a go/no-go gate, THEN `full`. If you set `PRESET=full`
without validating, this agent will still run it and will alert you at the coherence
gate — but you're committing to the full cost up front.

---

## Appendix — what was built so this loop works
- `run_distill` writes `training_state.json` every log interval (via a trainer
  `log_hook`), so `monitor.py` works during the 50B distill, not just SFT/GRPO.
- The trainers consume `training_state_override.json` (lr / kl_coeff /
  thinking_length_weight / stop), applied once then deleted; `_override_set_lr` also
  rewrites the LambdaLR `base_lrs` so an lr override sticks in SFT/GRPO.
- Tests: `tests/test_override.py` and two `log_hook` tests in `test_distillation.py`.
