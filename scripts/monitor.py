#!/usr/bin/env python3
"""Training monitor for Claude /loop usage on Vast.ai.

Reads training_state.json written by train_4090.py and reports:
  - Current phase + step
  - Loss trend (last 5 readings)
  - Reward trend (GRPO phase)
  - Mean thinking blocks generated
  - Recommendations (LR adjustment, early stop, stage transition)

Usage in Claude /loop:
    /loop check training progress and adjust if needed

Claude will run this script, read the output, and decide whether to:
  - Continue as-is
  - Write an adjustment to training_state_override.json (train_4090.py polls this)
  - Alert you that something is wrong

Run manually:
    python scripts/monitor.py
    python scripts/monitor.py --state ./training_state.json --history ./training_history.jsonl
"""
import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional


STATE_FILE = "./training_state.json"
HISTORY_FILE = "./training_history.jsonl"
OVERRIDE_FILE = "./training_state_override.json"


def _load_state(path: str) -> Optional[Dict]:
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _append_history(state: Dict, history_path: str) -> None:
    with open(history_path, "a") as f:
        f.write(json.dumps(state) + "\n")


def _read_history(history_path: str, last_n: int = 20) -> List[Dict]:
    if not os.path.isfile(history_path):
        return []
    lines = deque(maxlen=last_n)
    with open(history_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    lines.append(json.loads(line))
                except Exception:
                    pass
    return list(lines)


def _trend(values: List[float]) -> str:
    if len(values) < 2:
        return "n/a"
    delta = values[-1] - values[0]
    pct = abs(delta) / max(abs(values[0]), 1e-9) * 100
    arrow = "↓" if delta < 0 else "↑"
    return f"{arrow} {pct:.1f}%"


def _age_str(ts: float) -> str:
    age = time.time() - ts
    if age < 60:
        return f"{age:.0f}s ago"
    elif age < 3600:
        return f"{age/60:.1f}m ago"
    return f"{age/3600:.1f}h ago"


def _recommend(state: Dict, history: List[Dict]) -> List[str]:
    """Generate recommendations based on recent training state."""
    recs = []
    stage = state.get("stage", "unknown")
    step = state.get("step", 0)

    if stage == "grpo":
        reward = state.get("mean_reward", 0)
        think_blks = state.get("mean_think_blocks", 0)
        kl = state.get("kl", 0)
        loss = state.get("loss", 0)

        # Overthinking check
        if think_blks > 1.8:
            recs.append(
                f"OVERTHINKING: mean think blocks = {think_blks:.2f} (target ≤ 1.5). "
                "Increase thinking_length_weight from 0.02 → 0.04 in GRPO_CFG."
            )

        # KL blowup
        if kl > 0.5:
            recs.append(
                f"KL BLOWUP: kl = {kl:.3f} > 0.5. Reduce kl_coeff from 0.04 → 0.02."
            )

        # Reward plateau
        if len(history) >= 10:
            recent_rewards = [h.get("mean_reward", 0) for h in history[-10:] if h.get("stage") == "grpo"]
            if len(recent_rewards) >= 5 and _trend(recent_rewards).startswith("↑ 0"):
                recs.append(
                    f"PLATEAU: reward flat over last 10 steps ({reward:.3f}). "
                    "Try increasing lr from 5e-6 → 1e-5."
                )

        # Reward going negative
        if reward < -0.2:
            recs.append(
                f"NEGATIVE REWARD: {reward:.3f}. "
                "Length penalty may be too strong — reduce thinking_length_weight to 0.01."
            )

        # Loss diverging
        if len(history) >= 5:
            recent_losses = [h.get("loss", 0) for h in history[-5:] if h.get("stage") == "grpo"]
            if len(recent_losses) >= 3 and recent_losses[-1] > recent_losses[0] * 2:
                recs.append(
                    f"LOSS DIVERGING: {recent_losses[0]:.4f} → {recent_losses[-1]:.4f}. "
                    "Halve lr immediately."
                )

    elif stage == "sft":
        loss = state.get("loss", 0)
        if len(history) >= 5:
            recent_losses = [h.get("loss", 0) for h in history[-5:] if h.get("stage") == "sft"]
            if len(recent_losses) >= 3 and recent_losses[-1] > recent_losses[0] * 1.5:
                recs.append("SFT loss increasing — check for data issues or reduce lr.")

    elif stage in ("stage1", "stage2", "stage3", "distill"):
        loss = state.get("loss", 0)
        if not any(x > 0 for x in [loss]):
            recs.append("Distillation loss is zero — check that student is receiving gradients.")

    if not recs:
        recs.append("Training looks healthy. No adjustments needed.")

    return recs


def _write_override(adjustments: Dict, path: str = OVERRIDE_FILE) -> None:
    """Write adjustments for train_4090.py to pick up on its next poll."""
    with open(path, "w") as f:
        json.dump({**adjustments, "written_at": time.time()}, f, indent=2)
    print(f"\n[monitor] wrote override → {path}")
    print("  train_4090.py will apply these on the next log_interval step.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--state", default=STATE_FILE)
    p.add_argument("--history", default=HISTORY_FILE)
    p.add_argument("--override", default=OVERRIDE_FILE)
    args = p.parse_args()

    state = _load_state(args.state)
    if state is None:
        print(f"[monitor] no state file found at {args.state} — training not started yet?")
        sys.exit(0)

    # Append to history for trend analysis
    _append_history(state, args.history)
    history = _read_history(args.history, last_n=30)

    # ── report ────────────────────────────────────────────────────────────────
    age = _age_str(state.get("timestamp", 0))
    stage = state.get("stage", "unknown")
    step = state.get("step", 0)

    print("=" * 60)
    print(f"  DIMBA Training Monitor  |  {age}")
    print("=" * 60)
    print(f"  Stage:  {stage}")
    print(f"  Step:   {step}")

    if stage == "grpo":
        r = state.get("mean_reward", "?")
        t = state.get("mean_think_blocks", "?")
        kl = state.get("kl", "?")
        loss = state.get("loss", "?")
        print(f"  Loss:   {loss:.4f}" if isinstance(loss, float) else f"  Loss:   {loss}")
        print(f"  Reward: {r:.3f}" if isinstance(r, float) else f"  Reward: {r}")
        print(f"  Thinks: {t:.2f} blocks avg" if isinstance(t, float) else f"  Thinks: {t}")
        print(f"  KL:     {kl:.4f}" if isinstance(kl, float) else f"  KL:     {kl}")

        # Trend from history
        grpo_hist = [h for h in history if h.get("stage") == "grpo"]
        if len(grpo_hist) >= 2:
            reward_vals = [h.get("mean_reward", 0) for h in grpo_hist]
            loss_vals = [h.get("loss", 0) for h in grpo_hist]
            think_vals = [h.get("mean_think_blocks", 0) for h in grpo_hist]
            print(f"\n  Trends (last {len(grpo_hist)} readings):")
            print(f"    reward: {_trend(reward_vals)}")
            print(f"    loss:   {_trend(loss_vals)}")
            print(f"    thinks: {_trend(think_vals)}")

    elif stage == "sft":
        loss = state.get("loss", "?")
        lr = state.get("lr", "?")
        print(f"  Loss:   {loss:.4f}" if isinstance(loss, float) else f"  Loss:   {loss}")
        print(f"  LR:     {lr:.2e}" if isinstance(lr, float) else f"  LR:     {lr}")

    else:
        for k, v in state.items():
            if k not in ("timestamp", "stage", "step") and isinstance(v, (int, float)):
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # ── recommendations ───────────────────────────────────────────────────────
    recs = _recommend(state, history)
    print("\n  Recommendations:")
    for rec in recs:
        print(f"    • {rec}")

    print("=" * 60)

    # ── Claude can call _write_override to adjust training ────────────────────
    # Example (for Claude /loop):
    #   from scripts.monitor import _write_override
    #   _write_override({"lr": 1e-5, "thinking_length_weight": 0.04})
    # train_4090.py polls OVERRIDE_FILE every log_interval steps.


if __name__ == "__main__":
    main()
