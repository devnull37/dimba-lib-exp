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
    sample_elbo_trajectories,
    dpo_loss,
    ipo_loss,
    simpo_loss,
)
from .optimizers import HybridMuon, build_optimizer, split_muon_parameters
from .fused_ce import liger_fused_ce_available, output_head_cross_entropy
from .masked import freeze_masked_only_unused_parameters

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
    "sample_elbo_trajectories",
    "dpo_loss",
    "ipo_loss",
    "simpo_loss",
    "HybridMuon",
    "build_optimizer",
    "split_muon_parameters",
    "liger_fused_ce_available",
    "output_head_cross_entropy",
    "freeze_masked_only_unused_parameters",
]
