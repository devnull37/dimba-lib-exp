# Progressive model snapshots

`ProgressiveCheckpointManager` saves a model/optimizer snapshot when the number of trainable
parameters crosses configured milestones. It is useful only for workflows that actually grow or
unfreeze the model. A normal fixed-shape DIMBA does **not** grow from 1B to 5B parameters during
training, so token/step checkpoints are usually the correct mechanism.

## Important: this is not exact resume

A progressive snapshot contains model weights, optimizer state, global step, parameter counts, and
milestone metadata. It does **not** capture the full scheduler, Python/CPU/CUDA/MPS RNG state, data
cursor, distributed topology, pinned revisions, backend, or run signature. Loading one is a
weight/optimizer continuation and may replay data or take a different trajectory.

Use launcher-owned exact checkpoints instead for:

- continuous Stage 3: `distill_latest.pt` from `scripts/train_h100.py`;
- masked base/SFT DDP: the canonical launchers' exact same-world-size checkpoint;
- any run where reproducibility or billing-safe continuation matters.

See [NEXT_RUN_PLAN.md](NEXT_RUN_PLAN.md) and [BABYSIT_LOOP.md](BABYSIT_LOOP.md).

## Configuration

```yaml
progressive_checkpoints:
  enabled: true
  milestones: [100000000, 500000000, 1000000000]
  save_dir: ./progressive_checkpoints
```

Milestones are **trainable parameter counts**, not tokens or steps. For a fixed 300M-parameter
model, a 100M milestone is already satisfied at the first check and a 500M milestone is never
reached.

## Programmatic use

```python
from dimba.utils.checkpointing import ProgressiveCheckpointManager

manager = ProgressiveCheckpointManager(
    milestones=[100_000_000, 500_000_000],
    save_dir="./progressive_checkpoints",
)

should_save, milestone = manager.should_save_checkpoint(model)
if should_save and milestone is not None:
    path = manager.save_checkpoint(
        model=model,
        optimizer=optimizer,
        global_step=global_step,
        milestone=milestone,
    )
```

`should_save_checkpoint()` returns `(should_save, milestone)`, and that returned milestone is a
required argument to `save_checkpoint()`.

## Snapshot contents

The `.pt` file stores:

- `model_state_dict`;
- `optimizer_state_dict`;
- `global_step`;
- current/target parameter counts;
- configured and reached milestones;
- optional metadata.

The adjacent JSON file is human-readable metadata. Saves use atomic replacement, so an interrupted
write does not replace the last complete file.

## Loading for non-exact continuation

```python
info = manager.load_checkpoint(
    filepath="./progressive_checkpoints/checkpoint_0.1B_step_10000.pt",
    model=model,
    optimizer=optimizer,
)
print(info["global_step"])
```

After loading, explicitly rebuild/restore any scheduler and data state your custom workflow owns.
Do not describe the result as an exact resume.
