"""Training module for DIMBA."""

from .trainer import (
    DIMBALightningModule,
    SimpleTrainer,
    VAELightningModule,
    VAETrainer,
    compute_dimba_losses,
    compute_consistency_loss,
)
from .preference import (
    sequence_logprob,
    elbo_sequence_logprob,
    antithetic_timesteps,
    dpo_loss,
    ipo_loss,
    simpo_loss,
)

__all__ = [
    "DIMBALightningModule",
    "SimpleTrainer",
    "VAELightningModule",
    "VAETrainer",
    "compute_dimba_losses",
    "compute_consistency_loss",
    "sequence_logprob",
    "elbo_sequence_logprob",
    "antithetic_timesteps",
    "dpo_loss",
    "ipo_loss",
    "simpo_loss",
]
