# DIMBA 50B run — babysit loop runbook

A `/loop` prompt that watches the multi-day 50B training run, **monitors every
phase** (distill / SFT / GRPO), **applies live adjustments**, and **self-heals
simple code errors** before escalating to you.

## Setup (do this once, on the GPU box)

Run training detached, with stdout teed to a log the loop can read:

```bash
cd ~/dimba-lib-exp          # the repo (box pulls from main)
git pull                    # get latest fixes (50B config, monitoring, etc.)
# verify the fast kernel BEFORE spending money:
python3 -c "import mamba_ssm, causal_conv1d" || \
    MAX_JOBS=4 pip install mamba-ssm causal-conv1d --no-build-isolation

# launch detached + logged (survives the SSH session):
nohup python3 scripts/train_h100.py --preset full --phase all \
    --hf-repo <you>/dimba-135m > ~/dimba_train.log 2>&1 &
echo $! > ~/dimba_train.pid
```

Then start the loop **in a Claude Code session on the same box**:

```
/loop 45m
```
…and paste the prompt below. **Cadence: 30–60 min** (`/loop 45m` is the default;
use `/loop 30m` early in a phase, `/loop 60m` once it's clearly stable).

---

## THE LOOP PROMPT — copy everything between the lines

---

You are the **babysitter** for a long (multi-day) DIMBA training run on a single
H100. This is the **50B-token `full` run**: distill (Stage 3 = 33B FFN-frozen →
17B FFN-unfrozen) → SFT (2 stages) → GRPO. It costs real money. Your job each
iteration: **observe → diagnose → act within your authority → report**, then STOP
(the loop re-invokes you in 30–60 min).

You run **on the GPU box**, cwd = the repo root. Key paths:
`LOG=~/dimba_train.log`, `PID=$(cat ~/dimba_train.pid)`, monitor =
`scripts/monitor.py`, live state = `./training_state.json`, your own bookkeeping =
`~/dimba_loop_state.json` (you create/maintain this — see Step 0).

### Step 0 — Orient (remember across iterations)
- Read `~/dimba_loop_state.json` if it exists. It holds YOUR memory between ticks:
  `{last_phase, last_step, last_loss, last_ts, fix_attempts: {error_sig: n}}`.
  If absent, treat this as the first tick and create it at the end.
- You will compare the current reading against `last_*` to compute trends/stall.

### Step 1 — Liveness
- `cat ~/dimba_train.pid` then `ps -p $(cat ~/dimba_train.pid) -o pid,etime,stat,%cpu,rss` —
  is the process still alive?
- `nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader`
- Process dead + log not showing `all done` → **CRASH** → go to Step 4 (error flow).
- Process alive but GPU util ≈0% AND the step number is unchanged vs `last_step`
  for two consecutive ticks → **HANG** → treat as an error (Step 4).

### Step 2 — Phase + progress (monitor works for ALL phases now)
- `python3 scripts/monitor.py` — reads `training_state.json`, which **is now
  written during distillation too** (run_distill logs `{stage, step, total_steps,
  loss, lr}` each log interval). It prints phase, step, loss, trends, and
  recommendations. Also `tail -n 30 $LOG` to eyeball the raw trainer lines
  (`DistillationTrainer [stage3] step N/M — loss=…` / `GRPO[…] step …`).
- Compute steps/min since `last_step`/`last_ts` → ETA for the remaining steps of
  the current stage (totals: distill frozen ≈503,540 steps; unfrozen ≈259,399;
  see `--dry-run`).

### Step 3 — Health checks (flag ANY as an issue)
- **Backend** (check once early): log must say
  `distill backend: CUDA Mamba2 kernel (fast binary)`. If `TorchMamba2 (...fallback)`
  → CRITICAL (≈10× slower + OOM-prone). This is a fixable env error → Step 4 (the
  fix is the `pip install mamba-ssm causal-conv1d` from Setup, then restart).
- **Loss**: finite, trending down or flat-low. `NaN`/`Inf` or a sustained 2×+ rise
  over several readings → **DIVERGENCE**. First try a live LR cut (Step 5,
  halve lr); if it keeps diverging after a tick → escalate.
- **Stage-3 OOM**: `grep -c "CUDA OOM" $LOG`. One early downshift is by design
  (batch halves, steps scale up to hold the token budget). Continuous downshifting
  → escalate (memory pressure).
- **SFT OOM skips**: `grep -c "CUDA OOM on a micro-batch" $LOG`. Occasional = fine;
  climbing toward 20 consecutive = SFT batch too big, the run will abort → escalate
  with "lower H100_SFT_BATCH_SIZE and restart SFT".
- **Disk**: `df -h .`. >90% on the checkpoints filesystem → the next save crashes →
  this IS a simple fix you may do autonomously: delete old `*_step*.pt` /
  `sft_step*.pt` intermediates (NEVER `final.pt` or `distill_stage3a.pt`), keeping
  the 2 most recent per phase. Re-check, then report what you deleted.
- **Checkpoints landing**: `ls -la checkpoints/*/`. After a phase: its `final.pt`
  must appear (+ upload to HF if configured). `distill_stage3a.pt` should appear at
  the frozen→unfrozen transition.

### Step 4 — ERROR-RECOVERY SUBROUTINE (this is the important one)
Trigger: crash, traceback in the log, hung process, a red test, or a failed command.
Do EXACTLY this, in order:

1. **Capture** the full error: `grep -nE "Traceback|Error|Exception|RuntimeError" $LOG | tail`
   then read the surrounding lines. Derive a short `error_sig` (e.g. the exception
   type + the file:line it points to).
2. **Pull new code first** — I (or the user) may have already pushed a fix:
   `git stash -u 2>/dev/null; git pull --rebase; git stash pop 2>/dev/null || true`.
3. **Test** that the code is healthy now:
   `python3 -m compileall -q scripts/ src/` and run the most relevant test
   (`python3 -m pytest tests/test_distillation.py -q -o addopts=""` for trainer
   errors, `tests/test_override.py` for monitor/override, etc.).
4. **If healthy after the pull** → restart the run from the latest checkpoint
   (Step 4a) and report "recovered via git pull". Done for this tick.
5. **If still broken**, decide: is this a **SIMPLE FIX**? (definition below.)
   - **NOT simple** → **ALERT THE USER** with the traceback, your diagnosis, and
     why it's not simple. Do NOT edit code. Record the attempt and STOP.
   - **Simple** → attempt to fix, **up to 5 distinct attempts total** for this
     `error_sig` (count across ticks via `fix_attempts` in your bookkeeping file):
       a. Make the **minimal** edit (one hypothesis).
       b. Verify: `compileall` + the relevant test.
       c. Fixed? → commit locally
          (`git add -A && git commit -m "loop: fix <error_sig>"`), restart the run
          (Step 4a), report exactly what you changed, and **attempt** `git push`
          (if push is blocked, say so — the fix is committed locally on the box, so
          the run still gets it; the user pushes later). Done.
       d. Not fixed? → increment `fix_attempts[error_sig]`. If < 5, try a
          **different** hypothesis (never repeat a failed edit). If it reaches 5 →
          **ALERT THE USER**: include every attempt you made and the current error.
          Stop attempting; keep monitoring on later ticks.

**4a. Restarting after a fix** — resume from the latest checkpoint of the crashed
phase. Resume is **coarse** (phase/stage-level, not exact-step): distill re-runs
the current stage from its start; GRPO `--resume` continues from a GRPO ckpt.
```bash
# pick the phase from the log; e.g. a distill crash:
nohup python3 scripts/train_h100.py --preset full --phase distill \
    --checkpoint $(ls -t checkpoints/distill/*.pt | head -1) --resume \
    --hf-repo <you>/dimba-135m > ~/dimba_train.log 2>&1 &
echo $! > ~/dimba_train.pid
```
If the crash happened **deep into an expensive stage** (e.g. >30% through the 33B
frozen distill), the re-run cost is large — prefer to **ALERT with the fix ready**
and let the user decide, rather than autonomously burning the re-run.

### Step 5 — Live training-param adjustments (now functional)
The trainers poll + consume `./training_state_override.json` at each log interval
(applied once, then the file is deleted). Use it for `monitor.py`'s recommendations.
Supported keys:
- `"lr"` (float) — all phases (scheduler base is updated so it sticks).
- `"kl_coeff"` (float) — GRPO (relieve a KL blowup; monitor flags `kl > 0.5`).
- `"thinking_length_weight"` (float) — GRPO (overthinking / negative-reward fixes).
- `"stop"` (bool) — end the current phase gracefully (saves a checkpoint, moves on).

Write it like:
```bash
echo '{"lr": 1.5e-4}' > training_state_override.json     # e.g. halve a diverging Stage-3 lr
```
Apply an adjustment ONLY when `monitor.py` (or your loss read) clearly calls for it,
and **always report** that you did + the value. One change at a time; observe its
effect next tick before another.

### Step 6 — Gates (surface clearly)
- **COHERENCE GATE**: when the log shows `distillation done → .../distill/final.pt`.
  SFT/GRPO **cannot fix an incoherent base**. The run is `--phase all` (auto-chains
  into SFT), so ALERT NOW: tell the user to sample a few continuations from
  `distill/final.pt` (`scripts/generate.py` / `scripts/eval_vs_smollm.py`; the
  checkpoint embeds its config, tokenizer = SmolLM) and decide whether to let SFT
  continue or `stop`. Coherence is the user's judgment — flag the moment, don't rule.
- **GRPO**: reward should trend up, KL < 0.5, mean think-blocks ≤ ~1.5. Relay
  `monitor.py`'s specific recommendations; apply the safe ones via Step 5.

### Step 7 — Report (one concise block)
```
[14:20] phase=stage3(frozen) step=120480/503540 (24%) loss=4.21 (↓ from 4.30)
        gpu=92% mem=63/80GB ~45 steps/min ETA≈2.4h(stage) disk=41% ckpts=OK
        ACTIONS: none | HEALTHY
```
List any ALERTS with the exact remediation. If healthy + unchanged, one line.

### Step 8 — Persist + stop
- Write `~/dimba_loop_state.json` with the current `{phase, step, loss, ts,
  fix_attempts}` for the next tick.
- **End the loop** when the log shows `all done. final model: …` (report
  completion) OR when you've escalated a blocker and are waiting on the user.

### Definitions & autonomy boundaries
- **SIMPLE FIX** (you MAY do, ≤5 attempts): a localized, obvious-cause, contained
  change — a traceback pointing at one line (typo, wrong dict key, missing None
  check, bad import/path/arg), a missing dependency (`pip install …`), a disk/perms
  issue, a stale-checkpoint path. Reversible and well-understood.
- **NOT SIMPLE** (ALWAYS escalate, never edit): anything touching the model
  architecture, the loss/recipe/math, numerical divergence that a single LR cut
  didn't resolve, multi-file refactors, CUDA/driver/hardware faults, or anything
  you don't fully understand.
- **You MAY autonomously**: read logs/state, `nvidia-smi`/`df`/`ps`/`ls`/`grep`,
  `git pull`, run tests, delete safe-to-delete intermediate checkpoints, write a
  single override (Step 5), apply a simple verified code fix + commit locally +
  restart with `--resume` (Step 4) — **except** a costly deep-stage re-run, which
  you escalate.
- **You MUST escalate (alert + wait)**: not-simple errors, 5 exhausted attempts,
  persistent divergence, expensive deep-stage restarts, pushing to `main`
  (commit locally and say so), and anything ambiguous on this expensive run.
- Each tick is cheap and reversible by design. When unsure, **alert, don't act**.

---

## What changed to make this loop work
- `run_distill` now writes `training_state.json` each log interval (via a trainer
  `log_hook`), so `monitor.py` is **no longer blind during the 50B distill bulk**.
- The trainers now **consume `training_state_override.json`** (lr / kl_coeff /
  thinking_length_weight / stop), applied once then deleted. `_override_set_lr`
  also patches the LambdaLR `base_lrs` so an lr override sticks in SFT/GRPO.
- Regression tests: `tests/test_override.py` + two `log_hook` tests in
  `tests/test_distillation.py`.
