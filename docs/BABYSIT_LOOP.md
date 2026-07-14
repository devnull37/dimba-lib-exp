# DIMBA training babysitter

This is the `/loop` brief for a long GPU run. It handles one phase at a time, reports,
persists state, and stops. It never advances through a quality gate automatically.

## Human setup

Run the loop on the GPU box from a clean checkout. Fill in:

```bash
export REPO_DIR="$HOME/dimba-lib-exp"
export PHASE="distill"              # distill, sft, or grpo; never all
export PRESET="validation"          # smoke, validation, scale, full, ...
export OPTIMIZER="adamw"            # adamw default; muon is a pilot
export SAVE_DIR="$HOME/checkpoints/dimba-validation-adamw"
export CHECKPOINT=""                 # required for sft/grpo or resume
export HF_REPO="you/dimba-135m"      # empty disables upload
export LOG="$HOME/dimba_train.log"
export PIDFILE="$HOME/dimba_train.pid"
export LOOPSTATE="$HOME/dimba_loop_state.json"
```

Set `HF_TOKEN` in the shell before starting the loop; never write it to `LOOPSTATE`.
Use a different `SAVE_DIR` for every run and every AdamW/Muon arm. Start with
`validation`; do not spend the `full` budget until its gates pass.

## Non-negotiable launch rules

- `scripts/train_h100.py` is continuous and phase-by-phase. It rejects `--phase all`.
  Stage 3 may run under single-node `torchrun`; SFT/GRPO remain single-process.
- CUDA training must resolve every mixer to `mamba_ssm.Mamba2` and must import
  `causal_conv1d`. The code fails closed instead of running TorchMamba2. Mamba-1 is
  never a fallback.
- SFT and GRPO require `--quality-gate-passed`. That flag records a human decision;
  the babysitter must not infer it from a falling loss.
- AdamW is the default. Use Muon only for a named pilot/A-B arm.
- Exact resume exists for continuous Stage 3. SFT and GRPO do not have exact resume;
  never pass `--resume` to them.

## What is being watched

The definitive failed run consumed about 28B Stage-3 tokens. It recovered unseen text
through `t≈0.9` but failed at the pure-noise `t=1` starting point. The current recipe
uses SNR floor 0.5, CE fade `(1-t)`, a 50/50 uniform/logit-normal timestep mixture,
and `(1-t)`-faded teacher KD 1.0→0.3 across frozen 3a / low-LR unfrozen 3b. Loss alone cannot prove
the endpoint is fixed. SFT cannot rescue an incoherent base; GRPO cannot bootstrap
from uniformly wrong samples.

Important files:

- `<LOG>`: process output.
- `training_state.json`: current stage, step, loss, and LR.
- `training_state_override.json`: one-shot live override, consumed and deleted.
- `<SAVE_DIR>/distill_latest.pt`: exact rolling Stage-3 checkpoint.
- `<SAVE_DIR>/distill_stage3a.pt`: exact 3a boundary checkpoint.
- `<SAVE_DIR>/final.pt`: selected phase output.
- `<LOOPSTATE>`: babysitter state across ticks.

## First tick: preflight and launch

Run:

```bash
cd "$REPO_DIR"
git status --short
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -c "from mamba_ssm import Mamba2; import causal_conv1d"
python3 scripts/train_h100.py --preset "$PRESET" \
  --stage3-optimizer "$OPTIMIZER" --dry-run
```

The dry run must print `Stage-3 teacher KD = 1.00 -> 0.30` (or `0.30 -> 0.30` for
the unfrozen-only repair preset). Stop if either Stage-3 phase resolves to zero KD.

Stop and alert if the worktree is unexpectedly dirty, CUDA is unavailable, the fused
imports fail, or the GPU is not the intended box. Do not install or change a CUDA stack
mid-run without recording it.

Launch exactly one phase.

Fresh distillation:

```bash
nohup python3 scripts/train_h100.py --preset "$PRESET" --phase distill \
  --stage3-optimizer "$OPTIMIZER" --save-dir "$SAVE_DIR" \
  --hf-repo "$HF_REPO" > "$LOG" 2>&1 &
echo $! > "$PIDFILE"
```

SFT after a human-approved distillation gate:

```bash
nohup python3 scripts/train_h100.py --preset "$PRESET" --phase sft \
  --checkpoint "$CHECKPOINT" --quality-gate-passed --save-dir "$SAVE_DIR" \
  --hf-repo "$HF_REPO" > "$LOG" 2>&1 &
echo $! > "$PIDFILE"
```

GRPO after a human-approved SFT gate:

```bash
nohup python3 scripts/train_h100.py --preset "$PRESET" --phase grpo \
  --checkpoint "$CHECKPOINT" --quality-gate-passed --save-dir "$SAVE_DIR" \
  --hf-repo "$HF_REPO" > "$LOG" 2>&1 &
echo $! > "$PIDFILE"
```

Confirm the process is alive and the log shows the selected phase. During distillation,
the log must report `CUDA Mamba2 kernel (fast binary)`.

## Every tick

```bash
cd "$REPO_DIR"
ALIVE=no; [ -f "$PIDFILE" ] && ps -p "$(cat "$PIDFILE")" >/dev/null 2>&1 && ALIVE=yes
tail -n 30 "$LOG" 2>/dev/null
python3 scripts/monitor.py
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu \
  --format=csv,noheader
df -h .
```

Choose one state:

- **Running:** process alive. Check health, report, persist state, stop this tick.
- **Selected phase complete:** log contains `all done. final model:` and the expected
  checkpoint exists. Report the gate, persist state, and end the loop. Do not launch
  the next phase.
- **Stopped/crashed:** process dead without the completion line. Capture the traceback
  and use the recovery rules below.
- **Not launched:** run the preflight once. Never launch a second process if the PID is
  merely stale or ambiguous.

Health checks:

- Loss is finite and not sustaining a sharp rise.
- Step advances and GPU utilization is non-trivial.
- Distillation remains on the fused Mamba2 backend.
- At repair/validation checkpoints, the fixed-seed partial-noise ladder retains
  `t≤0.9` recovery and improves `t≈1`; a lower scalar loss is not a substitute.
- Checkpoints land atomically and disk remains below 90%.
- In a single-process Stage 3, one OOM before progress may downshift batch and scale steps.
  DDP aborts coherently on any OOM. Exact resume keeps the saved batch, so lowering batch means
  a new run (or an explicitly lossy weight-only restart), not continuation of the exact state.
- A stopped Stage 3 must have a fresh `distill_latest.pt` before the process exits.

Suggested report:

```text
[14:20] phase=stage3a step=120480/503540 loss=4.21 trend=down
gpu=92% mem=63/80GB disk=41% checkpoint=OK action=none
```

## Exact Stage-3 resume

Resume only from a full distillation checkpoint, normally `distill_latest.pt`, using
the same preset, optimizer, batch/backend, save directory, and data configuration:

```bash
nohup python3 scripts/train_h100.py --preset "$PRESET" --phase distill \
  --stage3-optimizer "$OPTIMIZER" --checkpoint "$CHECKPOINT" --resume \
  --save-dir "$SAVE_DIR" --hf-repo "$HF_REPO" > "$LOG" 2>&1 &
echo $! > "$PIDFILE"
```

Exact resume restores model, optimizer, RNG, step, pinned data revisions, and the
cumulative FineWeb cursor. Stage 3b continues after all Stage-3a batches; it does not
restart the stream.

If the checkpoint is legacy weights only, the code refuses exact resume. Do not add
`--weights-only-resume` automatically. A human may explicitly choose this lossy path:

```bash
python3 scripts/train_h100.py --preset "$PRESET" --phase distill \
  --stage3-optimizer "$OPTIMIZER" --checkpoint "$CHECKPOINT" --resume \
  --weights-only-resume --save-dir "$SAVE_DIR"
```

That discards optimizer, RNG, step, and cursor and restarts the configured Stage-3 plan.

SFT/GRPO interruption is different: there is no exact resume. Alert the human with the
last input and output checkpoints. A restart begins that phase again from its input
weights; do not claim continuity.

## Recovery boundaries

1. Capture the traceback and the last good checkpoint path.
2. Run `python3 -m compileall -q src/dimba scripts` and the smallest relevant test.
3. If the worktree is clean, check whether the remote already contains a fix. Do not
   stash or overwrite unknown local work.
4. Apply only a localized, understood fix; verify it before relaunching.
5. Escalate architecture/loss changes, persistent divergence, CUDA/driver faults,
   ambiguous state, or any costly restart.

Never delete `final.pt`, `distill_latest.pt`, or `distill_stage3a.pt`. If disk exceeds
90%, keep those files and only prune older, confirmed intermediate checkpoints.

## Live overrides and stopping

The trainer consumes one JSON override at a logging point:

```bash
echo '{"lr": 1.0e-4}' > training_state_override.json
```

Use one adjustment, report it, and wait for the next tick before another.

For a safe Stage-3 stop:

```bash
echo '{"stop": true}' > training_state_override.json
```

Wait for a new `distill_latest.pt`, the early-stop log line, and process exit. Resume it
with the exact command above. The stop override only ends the current SFT substage or
GRPO pass and does not create exact post-training state, so do not use it as a resumable
phase-wide stop for SFT/GRPO.

## Multi-GPU training

Continuous Stage 3 supports one process per H100. Rank 0 alone runs alignment; every GPU keeps a
frozen teacher replica, and preset step counts scale down by world size to retain the global token
budget:

```bash
torchrun --standalone --nproc-per-node=8 scripts/train_h100.py \
  --preset validation --phase distill \
  --save-dir checkpoints/h100-validation-8gpu
```

The masked objective also supports DDP:

```bash
PYTHONPATH=src torchrun --standalone --nproc-per-node=8 \
  scripts/masked_diffusion_finetune.py \
  --device cuda --batch 8 --accumulate 1 --optimizer adamw \
  --output-dir checkpoints/masked-base-adamw
```

`--batch` is per GPU. Global batch is
`batch × nproc-per-node × accumulate` (8 × 8 × 1 = 64 above). Rank 0 builds the shared
cache once; all ranks must see and load that file (memory-mapped where supported). AdamW
is the default; `--optimizer muon` is an opt-in pilot and must use a distinct `--output-dir`.

Exact masked resume uses the same world size and trajectory flags:

```bash
PYTHONPATH=src torchrun --standalone --nproc-per-node=8 \
  scripts/masked_diffusion_finetune.py \
  --device cuda --batch 8 --accumulate 1 --optimizer adamw \
  --output-dir checkpoints/masked-base-adamw \
  --resume checkpoints/masked-base-adamw/mdm_latest.pt
```

The checkpoint validates world size, per-GPU batch, accumulation, optimizer, seed,
planned steps, and data signature, and restores each rank's RNG plus epoch/batch cursor.
`scripts/mdm_sft_cfg2.py` follows the same DDP and exact-resume contract.
Single-node DDP is tested. Multi-node is not yet validated and additionally requires
the cache/checkpoint path to be shared identically across nodes.

## Persist and stop

Write `<LOOPSTATE>` with situation, phase, step, loss, timestamp, checkpoint, and any
action. Then stop the tick. End the loop when the selected phase completes, when a
human decision is required, or when a blocker is escalated. If there is no action,
report healthy state and stop; do not invent work.
