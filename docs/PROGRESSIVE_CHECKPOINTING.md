# Progressive Checkpointing

DIMBA supports milestone-based progressive checkpointing that automatically saves models as they grow through parameter count targets (e.g., 1B, 5B, 10B, 30B parameters).

## Overview

Progressive checkpointing helps you:
- Track model growth through parameter milestones
- Save checkpoints at significant model sizes
- Resume training from any milestone
- Keep a history of model development

## Configuration

### Config File

Add to your `config.yaml`:

```yaml
progressive_checkpoints:
  enabled: true
  milestones: [1000000000, 5000000000, 10000000000, 30000000000]  # 1B, 5B, 10B, 30B
  save_dir: "./progressive_checkpoints"
```

### Interactive Setup

Use the interactive setup script:

```bash
python scripts/interactive_setup.py
```

Select "1. Setup Progressive Checkpointing" and follow the prompts:
- Choose to continue from existing config or create new
- Enter milestone sizes (e.g., `1,5,10,30` for billions)
- Specify save directory

### Programmatic Usage

```python
from dimba.training import DIMBALightningModule

module = DIMBALightningModule(
    vocab_size=32000,
    model_config=model_config,
    # Progressive checkpointing settings
    enable_progressive_checkpoints=True,
    progressive_milestones=[1_000_000_000, 5_000_000_000, 10_000_000_000],
    progressive_save_dir="./progressive_checkpoints",
)
```

## Checkpoint Format

When a milestone is reached, two files are created:

### Model Checkpoint
`checkpoint_1.0B_step_10000.pt` contains:
- `model_state_dict`: Model weights
- `optimizer_state_dict`: Optimizer state
- `global_step`: Training step
- `current_params`: Actual parameter count
- `target_milestone`: The milestone reached
- `milestones`: List of all configured milestones
- `reached_milestones`: Milestones already reached
- `metadata`: Additional training info

### Metadata JSON
`checkpoint_1.0B_step_10000.json` contains:
- Checkpoint filename
- Global step
- Parameter counts (raw and formatted)
- Milestone information
- Easy to read without loading PyTorch

## Resuming from Checkpoints

```python
from dimba.utils.checkpointing import ProgressiveCheckpointManager

# Create manager with same milestones
manager = ProgressiveCheckpointManager(
    milestones=[1_000_000_000, 5_000_000_000, 10_000_000_000],
    save_dir="./progressive_checkpoints",
)

# Load checkpoint
info = manager.load_checkpoint(
    filepath="./progressive_checkpoints/checkpoint_1.0B_step_10000.pt",
    model=model,
    optimizer=optimizer,
)

print(f"Resumed from step {info['global_step']}")
print(f"Next milestone: {manager.get_next_milestone()}")
```

## Viewing Checkpoints

### Using Interactive Script

```bash
python scripts/interactive_setup.py
# Select "2. View Existing Progressive Checkpoints"
```

### Using Python

```python
from dimba.utils.checkpointing import ProgressiveCheckpointManager

manager = ProgressiveCheckpointManager(
    milestones=[1_000_000_000],  # Dummy milestones for viewing
    save_dir="./progressive_checkpoints",
)

checkpoints = manager.list_checkpoints()
for cp in checkpoints:
    print(f"{cp['milestone_str']}: step {cp['global_step']}")
```

## Milestone Input Formats

The interactive setup and `parse_milestone_input()` support various formats:

| Input | Result | Description |
|-------|--------|-------------|
| `1,5,10,30` | 1B, 5B, 10B, 30B | Billions (default) |
| `1B,5B,10B` | 1B, 5B, 10B | Explicit billions |
| `100M,500M,1B` | 100M, 500M, 1B | Mixed units |
| `500M,1B,3B` | 500M, 1B, 3B | Millions + billions |

## How It Works

1. **Parameter Counting**: During training, the model counts actual trainable parameters
2. **Milestone Checking**: After each training step, checks if current_params >= next_milestone
3. **Checkpoint Saving**: When a milestone is reached:
   - Saves model and optimizer state
   - Creates JSON metadata
   - Logs to console with 🎯 emoji
   - Marks milestone as reached (no duplicate saves)
4. **Resume Support**: When loading, restores milestone tracking state

## Example Output

```
Epoch 1/10 | Step 500/1000 | Loss: 0.4523

🎯 Progressive checkpoint saved: 1.0B (current: 1.2B) at step 5000
   Path: ./progressive_checkpoints/checkpoint_1.0B_step_5000.pt

Epoch 1/10 | Step 501/1000 | Loss: 0.4481
```

## Best Practices

1. **Set realistic milestones**: Base milestones on your expected final model size
2. **Use with CDLM**: Combine with consistency training for best results
3. **Regular backups**: Progressive checkpoints complement regular checkpoints
4. **Monitor growth**: Use the JSON metadata to track model development

## Integration with Training

Progressive checkpointing works with both trainers:

### PyTorch Lightning
```python
# Automatically checked in training_step
module = DIMBALightningModule(..., enable_progressive_checkpoints=True)
trainer.fit(module, train_loader)
```

### SimpleTrainer
```python
# Automatically checked in train loop
trainer = SimpleTrainer(..., enable_progressive_checkpoints=True)
trainer.train()
```

## API Reference

### ProgressiveCheckpointManager

```python
class ProgressiveCheckpointManager:
    def __init__(
        self,
        milestones: List[int],
        save_dir: str = "./progressive_checkpoints",
        enabled: bool = True,
    )
    
    def count_parameters(self, model: nn.Module) -> int
    def should_save_checkpoint(self, model: nn.Module) -> Tuple[bool, Optional[int]]
    def save_checkpoint(...) -> str
    def load_checkpoint(...) -> Dict[str, Any]
    def list_checkpoints() -> List[Dict[str, Any]]
    def format_param_count(count: int) -> str
```
