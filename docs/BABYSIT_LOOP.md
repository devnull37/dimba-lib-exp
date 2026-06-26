# DIMBA 50B run — babysit loop prompt

Run this in a Claude Code session **on the GPU box** (where the training log,
`nvidia-smi`, and this repo all live). Start the run with stdout teed to a log:

```bash
python3 scripts/train_h100.py --preset full --phase all \
    --hf-repo <you>/dimba-135m 2>&1 | tee ~/dimba_train.log
# (or run detached: ... > ~/dimba_train.log 2>&1 &)
```

Then launch the loop and paste the prompt below:

```
/loop 20m
```

Recommended cadence: **20 min** (`/loop 20m`). A 50B run is multi-day; 20-min
checks catch divergence/stalls early without spamming. Adjust as you like.

---

## THE LOOP PROMPT (copy everything below)

You are babysitting a long (multi-day) DIMBA training run on a single H100. This
is the **50B-token `full` run**: distill (Stage 3 = 33B FFN-frozen → 17B
FFN-unfrozen) → SFT (2 stages) → GRPO. It is expensive. Your job each iteration
is to **catch problems early and surface them clearly** — NOT to silently change
anything. Then STOP; the loop re-invokes you on the next interval.

Assume cwd is the repo root and the training log is at `~/dimba_train.log` (set
`LOG=~/dimba_train.log`). Remember your previous reading (phase, step, loss,
timestamp) across iterations so you can compute trends.

### Each iteration, do exactly this:

**1. Liveness + GPU**
- `nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader`
- During training, util should be high (>70%). Util ≈0% with no log progress for
  two checks in a row → likely **HANG / dead process** → ALERT.

**2. Phase + progress** (the live signal differs by phase)
- `tail -n 40 $LOG` to find the current phase + latest step.
- **DISTILL phase** (the 50B bulk — `monitor.py` is BLIND here, it writes no
  state during distill): parse the trainer line
  `DistillationTrainer [stage3] step N/M — loss=X` (logged every ~50 steps).
  Record N, M, loss; compare to last check.
- **SFT / GRPO phase**: run `python3 scripts/monitor.py` and relay its report +
  recommendations (it reads `training_state.json`, which IS written in these
  phases).
- Compute steps/min since last check → rough ETA for `M − N` remaining steps.

**3. Health checks — flag ANY of these as an ALERT (with the fix command):**
- **BACKEND** (check once, early): the log must say
  `distill backend: CUDA Mamba2 kernel (fast binary)`. If it says
  `TorchMamba2 (...fallback)` → **CRITICAL**: ~10× slower + OOM-prone. Fix:
  stop, `MAX_JOBS=4 pip install mamba-ssm causal-conv1d --no-build-isolation`,
  restart from the last checkpoint.
- **LOSS**: must be finite and trending **down** (or flat-low) in Stage 3.
  `NaN`/`Inf`, or a sustained 2×+ rise over several readings → **DIVERGENCE** →
  ALERT (recommend halving the Stage-3 LR and restarting `--resume` from the
  last checkpoint).
- **Stage-3 OOM**: `grep -c "CUDA OOM" $LOG`. One downshift early is fine (by
  design — batch halves, steps scale up to hold the token budget). Repeated /
  ongoing downshifting → ALERT (memory pressure).
- **SFT OOM skips**: `grep "CUDA OOM on a micro-batch" $LOG | tail`. Occasional
  skips are fine; many *consecutive* (approaching 20) means the SFT batch is too
  big and the loop will abort → ALERT before that happens.
- **Stall**: latest step unchanged vs last check AND GPU util low → process
  died/hung → ALERT.
- **Disk**: `df -h .`. If the checkpoints filesystem is >90% full → ALERT (the
  next checkpoint save will crash). A 50B run + frequent checkpoints fills disk.
- **Checkpoints landing**: `ls -la checkpoints/*/`. After a phase ends its
  `final.pt` should appear (and upload to HF if `--hf-repo` set). The inter-phase
  `distill_stage3a.pt` should appear at the frozen→unfrozen transition.

**4. Gates (decision points — surface clearly):**
- **COHERENCE GATE**: when the log shows `distillation done → .../distill/final.pt`,
  the base is built. SFT/GRPO **cannot fix an incoherent base**. If the run is
  `--phase all` (auto-chaining into SFT), ALERT the user NOW: pause and eyeball a
  few continuations from `distill/final.pt` (e.g. via `scripts/generate.py` /
  `scripts/eval_vs_smollm.py`; the checkpoint embeds its config, tokenizer =
  SmolLM) before letting SFT spend more. Coherence is a human judgment — your job
  is to flag the moment, not to rule on it.
- **GRPO**: reward should trend up, KL stay < 0.5, mean think-blocks ≤ ~1.5.
  `monitor.py` emits specific recommendations (overthinking, KL blowup, plateau,
  negative reward, divergence) — relay them verbatim.

**5. Report — output ONE concise status block, e.g.:**
```
[14:20] phase=stage3 step=120480/503540 (24%) loss=4.21 (↓ from 4.30)
        gpu=92% mem=63/80GB | ~45 steps/min ETA≈2.4h to end of frozen phase
        disk 41% | checkpoints OK | HEALTHY
```
Then list any ALERTS with the exact recommended remediation command. If
everything is healthy and unchanged, say so in one line — don't pad.

### Intervention policy (important):
- Live overrides are **NOT** auto-applied. `monitor.py` can write
  `training_state_override.json`, but the trainer does not read it, so editing
  LR/config via that file does **nothing**. Do not rely on it.
- For any problem, ALERT the user with the precise issue + the exact command to
  fix it (kill + `--resume` from the last checkpoint, reinstall the kernel, lower
  the batch, free disk, …). **Do NOT kill or restart the run yourself** — this run
  is expensive; confirm with the user first.
- Safe autonomous actions only: reading logs / state, `nvidia-smi`, `df`, `ls`,
  `grep`. Anything that mutates the run goes through the user.

### Stop the loop when:
- The log shows `all done. final model: …` (or the selected `--phase` finished
  and its `final.pt` uploaded) → report completion and end the loop, OR
- A blocker is surfaced and you're waiting on the user's decision.

---

## Known monitoring gaps (so you trust the right signals)

1. **`monitor.py` is blind during distillation.** `run_distill` does not write
   `training_state.json` — only SFT/GRPO do. During the 50B distill phase (the
   bulk), use the trainer's stdout log line as the signal, not `monitor.py`.
2. **Overrides are not consumed.** `training_state_override.json` is written by
   `monitor.py` but read by nothing. Interventions must be manual (alert + the
   user acts), not auto-applied.

> Both gaps are fixable (have `run_distill` write state each log interval; add an
> override poll to the trainers). Ask Claude to implement them if you want the
> loop to monitor distill via `monitor.py` and/or auto-apply safe adjustments.
