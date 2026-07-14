# CDLM consistency training (experimental legacy path)

> **Status:** mechanism implemented, DIMBA speed/quality payoff unvalidated. The “up to 14×”
> figure belongs to the upstream consistency-diffusion target; it is not a DIMBA measurement.
> The next production run is gated on the corrected base objective in
> [NEXT_RUN_PLAN.md](NEXT_RUN_PLAN.md), not on this optional loss.

This document describes DIMBA's legacy consistency-loss experiment. It may reduce the number of
network evaluations after successful training, but no current DIMBA checkpoint has demonstrated
an accuracy-preserving end-to-end speedup from it.

## Overview

CDLM is a post-training recipe from Together AI that accelerates diffusion language models by adding a consistency loss during training. The key insight is that predictions at different timesteps should be consistent - the prediction at timestep `t` should match the prediction at `t-δ` for positions that are still "noisy" at the later timestep.

Reference: [Together AI Blog Post](https://www.together.ai/blog/consistency-diffusion-language-models)

## How It Works

### Standard Diffusion Training
- Sample a timestep `t`
- Add noise to clean embeddings: `x_t = sqrt(α̅(t)) * x_0 + sqrt(1 - α̅(t)) * ε`
- Train model to predict `x_0` from `x_t`

### CDLM Consistency Training
In addition to standard training:
1. Sample two timesteps: `t_early` (more noise) and `t_late = t_early - δ` (less noise)
2. Get predictions at both timesteps
3. Use prediction at `t_late` as a **stop-gradient target** for `t_early`
4. Loss: `L = L_denoise + λ * MSE(pred_early, pred_late.detach())`

### Why This Works for DIMBA

The intended fit with DIMBA is:
- ✅ No KV caching issues (Mamba-2 is O(L), not O(L²))
- ✅ No need for block-wise masking
- ✅ Can do full-sequence consistency across all timesteps
- no KV cache is involved, but efficiency and quality still require measurement

## Configuration

Enable CDLM in your config file:

```yaml
training:
  # Standard training settings
  learning_rate: 2e-5
  warmup_steps: 500
  
  # CDLM Consistency Training
  use_consistency_training: true      # Enable CDLM
  consistency_loss_weight: 0.5        # Lambda weight (0.5-1.0 recommended)
  consistency_delta_min: 50           # Minimum timestep gap
  consistency_delta_max: 200          # Maximum timestep gap
```

## Usage

### Using the CDLM Training Script

```bash
# Enable consistency training with default settings
python scripts/train_cdlm.py --config config.yaml --enable-consistency

# Override consistency loss weight
python scripts/train_cdlm.py --config config.yaml --enable-consistency --consistency-weight 0.7

# Use with existing config
python scripts/train_cdlm.py --config config.yaml  # uses config settings
```

### Using the Lightning Module Directly

```python
from dimba.training import DIMBALightningModule

module = DIMBALightningModule(
    vocab_size=32000,
    model_config=model_config,
    use_consistency_training=True,
    consistency_loss_weight=0.5,
    consistency_delta_min=50,
    consistency_delta_max=200,
)
```

### Using SimpleTrainer

```python
from dimba.training import SimpleTrainer

trainer = SimpleTrainer(
    model=model,
    train_dataloader=train_loader,
    use_consistency_training=True,
    consistency_loss_weight=0.5,
)
trainer.train()
```

## Historical experiment settings

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| `consistency_loss_weight` | 0.5 | 0.3-1.0 | Higher = stronger consistency, may need LR adjustment |
| `consistency_delta_min` | 50 | 20-100 | Smaller = harder consistency targets |
| `consistency_delta_max` | 200 | 100-400 | Larger = more diverse timestep pairs |

## Monitoring Training

When consistency training is enabled, you'll see additional metrics:

```
train/loss: 0.4523
train/consistency_loss: 0.0891
```

The consistency loss should decrease over time as the model learns to make consistent predictions across timesteps.

## Inference claim boundary

The original target was to reduce 1,000 denoising steps to roughly 70-100:

| Illustrative target | Standard steps | Consistency steps | Arithmetic NFE ratio |
|-------|---------------|------------|---------|
| Small | 1000 | ~100 | ~10x |
| Large | 1000 | ~70 | ~14x |

These are arithmetic NFE ratios, not measured DIMBA latency or quality. A valid result must compare
the same trained checkpoint family at a fixed quality threshold and report wall time, p50/p95, and
the exact sampler. Until then, do not advertise 10-14× for DIMBA.

## Implementation Details

### Key Differences from Original CDLM Paper

1. **No Distillation Loss**: We skip the teacher distillation since we don't have a larger teacher model
2. **Continuous Diffusion**: DIMBA operates on continuous embeddings, not discrete tokens
3. **Noise-Weighted Loss**: We weight the consistency loss by the remaining noise level at `t_late`
4. **Full-Sequence Consistency**: Mamba-2 allows us to do full-sequence consistency without block-wise masking

### Code Changes

The implementation adds:
- `compute_consistency_loss()` in `dimba.training.trainer`
- Consistency loss calculation in `training_step()`
- Configuration parameters in `config.yaml`
- New `train_cdlm.py` script with CLI support

## References

1. Together AI. "Consistency diffusion language models: Up to 14x faster inference without sacrificing quality." 2024.
2. Song et al. "Consistency Models." ICML 2023.
3. Nichol & Dhariwal. "Improved Denoising Diffusion Probabilistic Models." ICML 2021.
