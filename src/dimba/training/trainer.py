"""PyTorch Lightning training module for DIMBA."""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Optional, Dict, Any, Tuple, List
import math
import os

from ..models.diffusion import DIMBA
from ..models.vae import TokenVAE
from ..diffusion.sampling import sample_timesteps
from ..utils.checkpointing import ProgressiveCheckpointManager


def compute_consistency_loss(
    model: DIMBA,
    input_ids: torch.Tensor,
    x_0: torch.Tensor,
    t_early: torch.Tensor,
    delta_min: int,
    delta_max: int,
) -> torch.Tensor:
    """Compute CDLM consistency loss.

    Shared utility function for computing consistency loss between predictions
    at different timesteps. Aligns predictions at timestep t with predictions
    at t-delta (later, less noisy state).

    Args:
        model: DIMBA model instance
        input_ids: Target token IDs [batch_size, seq_len]
        x_0: Clean embeddings [batch_size, seq_len, d_model]
        t_early: Timesteps for early (noisier) state [batch_size]
        delta_min: Minimum timestep delta for consistency pairs
        delta_max: Maximum timestep delta for consistency pairs

    Returns:
        consistency_loss: MSE between predictions at t and t-delta
    """
    device = t_early.device
    batch_size = input_ids.shape[0]

    # Adaptive delta sampling: ensure delta doesn't exceed t_early - delta_min
    # This prevents t_late from becoming negative
    max_possible_delta = torch.clamp(t_early - delta_min, min=0)
    effective_max_delta = torch.min(
        torch.full_like(t_early, delta_max),
        max_possible_delta
    )

    # Only process items where we can have a valid delta
    valid_mask = effective_max_delta >= delta_min
    if not valid_mask.any():
        return torch.tensor(0.0, device=device)

    # Filter to valid items
    t_early = t_early[valid_mask]
    effective_max_delta = effective_max_delta[valid_mask]
    input_ids = input_ids[valid_mask]
    x_0 = x_0[valid_mask]

    # Sample delta timesteps within valid range for each item
    delta = torch.stack([
        torch.randint(
            delta_min,
            min(int(eff.item()) + 1, delta_max + 1),
            (1,),
            device=device
        ).squeeze(0)
        for eff in effective_max_delta
    ])
    delta = torch.min(delta, effective_max_delta.long())

    # Compute t_late
    t_late = t_early - delta

    # Encode to latent space (same as main loss)
    z_0 = model.encode_latent(x_0)

    # Add noise at both timesteps
    x_t_early, _ = model.noise_schedule.add_noise(z_0, t_early)
    x_t_late, _ = model.noise_schedule.add_noise(z_0, t_late)

    # Encode prompt
    cond = model.encode_prompt(input_ids)
    cond = model.project_conditioning(cond)

    # Get timestep embeddings
    time_emb_early = model.timestep_embed(t_early)
    time_emb_late = model.timestep_embed(t_late)

    # Predict at t_early (trainable)
    z_pred_early = model.denoiser(x_t_early, cond, time_emb_early)

    # Predict at t_late (stop-gradient target)
    with torch.no_grad():
        z_pred_late = model.denoiser(x_t_late, cond, time_emb_late)

    # Weight by remaining noise level at t_late
    # Positions with more remaining noise get higher weight
    noise_level_late = model.noise_schedule.sqrt_one_minus_alphas_cumprod[t_late]
    noise_level_late = noise_level_late.view(-1, 1, 1)
    weights = noise_level_late.clamp(min=0.01)

    # Compute weighted MSE
    diff = (z_pred_early - z_pred_late.detach()) * weights
    consistency_loss = (diff ** 2).mean()

    return consistency_loss


class DIMBALightningModule(pl.LightningModule):
    """PyTorch Lightning module for training DIMBA.

    Handles training loop, optimization, EMA, and logging.
    Supports CDLM (Consistency Diffusion Language Model) training for faster inference.
    Supports progressive checkpointing based on parameter milestones.

    Args:
        vocab_size: Size of vocabulary
        model_config: Dictionary of model configuration parameters
        learning_rate: Learning rate for optimizer (default: 2e-5)
        warmup_steps: Number of warmup steps (default: 500)
        weight_decay: Weight decay for optimizer (default: 0.01)
        ema_decay: Exponential moving average decay (default: 0.9999)
        use_ema: Whether to use EMA (default: True)
        ema_device: Optional device to store EMA weights; None keeps default Lightning placement
        ema_update_interval: Update EMA every N steps (default: 1)
        use_consistency_training: Enable CDLM consistency loss (default: False)
        consistency_loss_weight: Weight for consistency loss (default: 0.5)
        consistency_delta_min: Minimum timestep delta for consistency pairs (default: 50)
        consistency_delta_max: Maximum timestep delta for consistency pairs (default: 200)
        progressive_milestones: List of parameter count milestones for progressive checkpointing
        progressive_save_dir: Directory for progressive checkpoints
        enable_progressive_checkpoints: Whether to enable progressive checkpointing
    """

    def __init__(
        self,
        vocab_size: int,
        model_config: Dict[str, Any],
        learning_rate: float = 2e-5,
        warmup_steps: int = 500,
        weight_decay: float = 0.01,
        ema_decay: float = 0.9999,
        use_ema: bool = True,
        ema_device: Optional[str] = None,
        ema_update_interval: int = 1,
        use_consistency_training: bool = False,
        consistency_loss_weight: float = 0.5,
        consistency_delta_min: int = 50,
        consistency_delta_max: int = 200,
        progressive_milestones: Optional[List[int]] = None,
        progressive_save_dir: str = "./progressive_checkpoints",
        enable_progressive_checkpoints: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.vocab_size = vocab_size
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.ema_update_interval = max(1, int(ema_update_interval))
        self.ema_device = torch.device(ema_device) if ema_device is not None else None

        # CDLM consistency training parameters
        self.use_consistency_training = use_consistency_training
        self.consistency_loss_weight = consistency_loss_weight
        self.consistency_delta_min = consistency_delta_min
        self.consistency_delta_max = consistency_delta_max

        # Progressive checkpointing
        self.progressive_checkpoint_manager = None
        if enable_progressive_checkpoints and progressive_milestones:
            self.progressive_checkpoint_manager = ProgressiveCheckpointManager(
                milestones=progressive_milestones,
                save_dir=progressive_save_dir,
                enabled=True,
            )

        # Build model
        self.model = DIMBA(vocab_size=vocab_size, **model_config)

        # EMA model
        if use_ema:
            self.ema_model = DIMBA(vocab_size=vocab_size, **model_config)
            # Keep EMA weights off-GPU only when explicitly configured via ema_device.
            if self.ema_device is not None:
                self.ema_model.to(self.ema_device)
            self.ema_model.requires_grad_(False)
            self.ema_model.eval()
            self._update_ema_model_once()

        # Loss function
        self.loss_fn = nn.MSELoss()

        # Metrics
        self.train_loss = 0.0
        self.num_training_steps = 0

    def _update_ema_model_once(self):
        """Initialize EMA model with current model weights."""
        for ema_param, param in zip(
            self.ema_model.parameters(),
            self.model.parameters()
        ):
            ema_param.data.copy_(
                param.detach().to(device=(self.ema_device or ema_param.device), dtype=ema_param.dtype)
            )

    def _update_ema_model(self):
        """Update EMA model parameters."""
        with torch.no_grad():
            for ema_param, param in zip(
                self.ema_model.parameters(),
                self.model.parameters()
            ):
                param_on_ema_device = param.detach().to(
                    device=(self.ema_device or ema_param.device),
                    dtype=ema_param.dtype,
                )
                ema_param.data.mul_(self.ema_decay).add_(param_on_ema_device, alpha=(1 - self.ema_decay))

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
        )

        # Warmup scheduler
        def lr_lambda(step):
            if step < self.warmup_steps:
                return float(step) / float(max(1, self.warmup_steps))
            return max(0.0, float(self.trainer.max_steps - step) / float(max(1, self.trainer.max_steps - self.warmup_steps)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def forward(self, input_ids: torch.Tensor, t: torch.Tensor):
        """Forward pass."""
        return self.model(input_ids, t)

    def training_step(self, batch, batch_idx):
        """Training step with optional CDLM consistency loss and progressive checkpointing."""
        input_ids = batch["input_ids"]
        batch_size = input_ids.shape[0]

        # Sample random timesteps for main denoising loss
        t = sample_timesteps(batch_size, self.model.num_diffusion_steps, self.device)

        # Forward pass for denoising loss
        x_pred, noise, latent_info = self.model(input_ids, t)

        # Get clean embeddings
        x_0 = self.model.token_embed(input_ids)

        # Compute denoising loss (predict clean embeddings or latent targets)
        loss = self.loss_fn(x_pred, x_0) * self.model.recon_loss_weight
        if self.model.latent_diffusion and latent_info is not None:
            latent_loss = self.loss_fn(latent_info["z_pred"], latent_info["z_0"])
            loss = loss + latent_loss * self.model.latent_loss_weight

        # CDLM Consistency loss: align predictions at t with predictions at t-delta
        consistency_loss = torch.tensor(0.0, device=self.device)
        if self.use_consistency_training and self.consistency_loss_weight > 0:
            consistency_loss = compute_consistency_loss(
                model=self.model,
                input_ids=input_ids,
                x_0=x_0,
                t_early=t,
                delta_min=self.consistency_delta_min,
                delta_max=self.consistency_delta_max,
            )
            loss = loss + self.consistency_loss_weight * consistency_loss

        # Update EMA periodically to reduce transfer overhead for CPU-offloaded EMA.
        if self.use_ema and (self.global_step + 1) % self.ema_update_interval == 0:
            self._update_ema_model()

        # Check for progressive checkpoint milestones
        if self.progressive_checkpoint_manager is not None:
            should_save, milestone = self.progressive_checkpoint_manager.should_save_checkpoint(self.model)
            if should_save and milestone is not None:
                optimizer = self.optimizers()
                checkpoint_path = self.progressive_checkpoint_manager.save_checkpoint(
                    model=self.model,
                    optimizer=optimizer,
                    global_step=self.global_step,
                    milestone=milestone,
                    metadata={
                        "epoch": self.current_epoch,
                        "train_loss": loss.item(),
                        "use_consistency_training": self.use_consistency_training,
                    },
                )
                milestone_str = self.progressive_checkpoint_manager.format_param_count(milestone)
                current_str = self.progressive_checkpoint_manager.format_param_count(
                    self.progressive_checkpoint_manager.count_parameters(self.model)
                )
                self.log("train/progressive_checkpoint_saved", float(milestone), prog_bar=True)
                print(f"\n🎯 Progressive checkpoint saved: {milestone_str} (current: {current_str}) at step {self.global_step}")
                print(f"   Path: {checkpoint_path}")

        # Logging
        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        self.log("train/learning_rate", self.optimizers().param_groups[0]["lr"], sync_dist=True)
        if self.use_consistency_training:
            self.log("train/consistency_loss", consistency_loss, prog_bar=False, sync_dist=True)

        self.train_loss = loss.item()
        self.num_training_steps += 1

        return loss

    def on_fit_start(self):
        """Ensure EMA stays on the configured device after Lightning device placement."""
        if self.use_ema and self.ema_device is not None:
            self.ema_model.to(self.ema_device)

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        input_ids = batch["input_ids"]
        batch_size = input_ids.shape[0]

        # Use middle timesteps for validation
        t = torch.full((batch_size,), self.model.num_diffusion_steps // 2, device=self.device)

        # Keep validation on the active training model to avoid moving EMA to GPU.
        model = self.model

        # Forward pass
        x_pred, _, _ = model(input_ids, t)
        x_0 = model.token_embed(input_ids)

        # Compute loss
        loss = self.loss_fn(x_pred, x_0)

        self.log("val/loss", loss, prog_bar=True, sync_dist=True)

        return loss

    def test_step(self, batch, batch_idx):
        """Test step."""
        input_ids = batch["input_ids"]
        batch_size = input_ids.shape[0]

        # Use model at various timesteps
        losses = []
        for t_val in [100, 500, 900]:
            t = torch.full((batch_size,), min(t_val, self.model.num_diffusion_steps - 1), device=self.device)
            x_pred, _, _ = self.model(input_ids, t)
            x_0 = self.model.token_embed(input_ids)
            loss = self.loss_fn(x_pred, x_0)
            losses.append(loss)

        avg_loss = torch.mean(torch.stack(losses))
        self.log("test/loss", avg_loss, sync_dist=True)

        return avg_loss

    def get_model_for_inference(self):
        """Get model for inference (EMA if available)."""
        if self.use_ema:
            return self.ema_model
        return self.model

    def on_train_end(self):
        """Called at end of training."""
        if self.use_ema:
            # Optionally copy EMA weights back to main model
            pass


def get_model_config(model: DIMBA) -> Dict[str, Any]:
    """Extract configuration from a DIMBA model for creating replicas.
    
    Preserves all model configuration parameters including architecture-specific
    settings like d_state, d_conv, expand, etc.
    
    Args:
        model: DIMBA model instance
        
    Returns:
        Dictionary with all model configuration parameters
    """
    config = {
        'vocab_size': model.vocab_size,
        'd_model': model.d_model,
        'd_prompt': model.d_prompt,
        'num_diffusion_steps': model.num_diffusion_steps,
        # Extract from denoiser if available
        'num_denoiser_layers': len(model.denoiser.layers) if hasattr(model.denoiser, 'layers') else 6,
        'd_state': getattr(model.denoiser, 'd_state', 16) if hasattr(model, 'denoiser') else 16,
        'd_conv': getattr(model.denoiser, 'd_conv', 4) if hasattr(model, 'denoiser') else 4,
        'expand': getattr(model.denoiser, 'expand', 2) if hasattr(model, 'denoiser') else 2,
        # Latent diffusion settings
        'latent_diffusion': model.latent_diffusion,
        'd_latent': getattr(model, 'd_latent', None),
        'latent_loss_weight': getattr(model, 'latent_loss_weight', 1.0),
        'recon_loss_weight': getattr(model, 'recon_loss_weight', 1.0),
    }
    return config


class SimpleTrainer:
    """Simple training loop without PyTorch Lightning (for debugging/custom needs).

    Supports CDLM consistency training and progressive checkpointing.

    Args:
        model: DIMBA model
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        device: Device to train on
        num_epochs: Number of epochs
        learning_rate: Learning rate
        warmup_steps: Number of warmup steps
        ema_decay: EMA decay rate
        use_consistency_training: Enable CDLM consistency loss (default: False)
        consistency_loss_weight: Weight for consistency loss (default: 0.5)
        consistency_delta_min: Minimum timestep delta (default: 50)
        consistency_delta_max: Maximum timestep delta (default: 200)
        progressive_milestones: List of parameter count milestones (default: None)
        progressive_save_dir: Directory for progressive checkpoints (default: "./progressive_checkpoints")
        enable_progressive_checkpoints: Enable progressive checkpointing (default: False)
    """

    def __init__(
        self,
        model: DIMBA,
        train_dataloader,
        val_dataloader=None,
        device: str = "cuda",
        num_epochs: int = 10,
        learning_rate: float = 2e-5,
        warmup_steps: int = 500,
        ema_decay: float = 0.9999,
        use_consistency_training: bool = False,
        consistency_loss_weight: float = 0.5,
        consistency_delta_min: int = 50,
        consistency_delta_max: int = 200,
        progressive_milestones: Optional[List[int]] = None,
        progressive_save_dir: str = "./progressive_checkpoints",
        enable_progressive_checkpoints: bool = False,
    ):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.ema_decay = ema_decay

        # CDLM parameters
        self.use_consistency_training = use_consistency_training
        self.consistency_loss_weight = consistency_loss_weight
        self.consistency_delta_min = consistency_delta_min
        self.consistency_delta_max = consistency_delta_max

        # Progressive checkpointing
        self.progressive_checkpoint_manager = None
        if enable_progressive_checkpoints and progressive_milestones:
            self.progressive_checkpoint_manager = ProgressiveCheckpointManager(
                milestones=progressive_milestones,
                save_dir=progressive_save_dir,
                enabled=True,
            )

        # EMA model - preserve all config
        ema_config = get_model_config(model)
        self.ema_model = DIMBA(**ema_config).to(device)
        self._copy_model_weights()

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
        )

        # Loss
        self.loss_fn = nn.MSELoss()

        # Tracking
        self.global_step = 0

    def _copy_model_weights(self):
        """Copy model weights to EMA model."""
        for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_param.data.copy_(param.data)

    def _update_ema(self):
        """Update EMA model."""
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data = ema_param.data * self.ema_decay + param.data * (1 - self.ema_decay)

    def train(self):
        """Run training loop with optional CDLM consistency training and progressive checkpointing."""
        total_steps = len(self.train_dataloader) * self.num_epochs

        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_denoise_loss = 0.0
            epoch_consistency_loss = 0.0

            for batch_idx, batch in enumerate(self.train_dataloader):
                # Learning rate warmup
                if self.global_step < self.warmup_steps:
                    lr = self.learning_rate * (self.global_step / self.warmup_steps)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = lr

                # Forward pass
                input_ids = batch["input_ids"].to(self.device)
                batch_size = input_ids.shape[0]
                t = sample_timesteps(batch_size, self.model.num_diffusion_steps, torch.device(self.device))

                x_pred, _, _ = self.model(input_ids, t)
                x_0 = self.model.token_embed(input_ids)

                denoise_loss = self.loss_fn(x_pred, x_0)
                loss = denoise_loss

                # CDLM Consistency loss
                consistency_loss = torch.tensor(0.0, device=self.device)
                if self.use_consistency_training and self.consistency_loss_weight > 0:
                    consistency_loss = compute_consistency_loss(
                        model=self.model,
                        input_ids=input_ids,
                        x_0=x_0,
                        t_early=t,
                        delta_min=self.consistency_delta_min,
                        delta_max=self.consistency_delta_max,
                    )
                    loss = loss + self.consistency_loss_weight * consistency_loss
                    epoch_consistency_loss += consistency_loss.item()

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                # Update EMA
                self._update_ema()

                epoch_loss += loss.item()
                epoch_denoise_loss += denoise_loss.item()
                self.global_step += 1

                # Check for progressive checkpoint milestones
                if self.progressive_checkpoint_manager is not None:
                    should_save, milestone = self.progressive_checkpoint_manager.should_save_checkpoint(self.model)
                    if should_save and milestone is not None:
                        checkpoint_path = self.progressive_checkpoint_manager.save_checkpoint(
                            model=self.model,
                            optimizer=self.optimizer,
                            global_step=self.global_step,
                            milestone=milestone,
                            metadata={
                                "epoch": epoch,
                                "train_loss": loss.item(),
                                "use_consistency_training": self.use_consistency_training,
                            },
                        )
                        milestone_str = self.progressive_checkpoint_manager.format_param_count(milestone)
                        current_str = self.progressive_checkpoint_manager.format_param_count(
                            self.progressive_checkpoint_manager.count_parameters(self.model)
                        )
                        print(f"\n🎯 Progressive checkpoint saved: {milestone_str} (current: {current_str}) at step {self.global_step}")
                        print(f"   Path: {checkpoint_path}")

                if batch_idx % 100 == 0:
                    log_msg = (
                        f"Epoch {epoch + 1}/{self.num_epochs} | "
                        f"Step {batch_idx}/{len(self.train_dataloader)} | "
                        f"Loss: {loss.item():.4f}"
                    )
                    if self.use_consistency_training:
                        log_msg += f" (Denoise: {denoise_loss.item():.4f}, Consistency: {consistency_loss.item():.4f})"
                    print(log_msg)

            avg_epoch_loss = epoch_loss / len(self.train_dataloader)
            avg_denoise_loss = epoch_denoise_loss / len(self.train_dataloader)
            print(f"Epoch {epoch + 1} | Avg Loss: {avg_epoch_loss:.4f} | Denoise: {avg_denoise_loss:.4f}", end="")
            if self.use_consistency_training:
                avg_consistency_loss = epoch_consistency_loss / len(self.train_dataloader)
                print(f" | Consistency: {avg_consistency_loss:.4f}")
            else:
                print()

            # Validation
            if self.val_dataloader is not None:
                val_loss = self.validate()
                print(f"Validation Loss: {val_loss:.4f}")

    def validate(self):
        """Run validation."""
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in self.val_dataloader:
                input_ids = batch["input_ids"].to(self.device)
                batch_size = input_ids.shape[0]
                t = torch.full((batch_size,), self.model.num_diffusion_steps // 2, device=self.device)

                x_pred, _, _ = self.model(input_ids, t)
                x_0 = self.model.token_embed(input_ids)

                loss = self.loss_fn(x_pred, x_0)
                val_loss += loss.item()

        return val_loss / len(self.val_dataloader)


class VAELightningModule(pl.LightningModule):
    """PyTorch Lightning module for training TokenVAE.

    Handles VAE pre-training on token sequences.

    Args:
        vocab_size: Size of vocabulary
        d_model: Token embedding dimension
        latent_dim: VAE latent dimension
        model_config: Dictionary of VAE configuration parameters
        learning_rate: Learning rate for optimizer (default: 1e-4)
        warmup_steps: Number of warmup steps (default: 1000)
        weight_decay: Weight decay for optimizer (default: 0.01)
        kl_weight: Weight for KL divergence loss (default: 1.0)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        latent_dim: int = 256,
        model_config: Optional[Dict[str, Any]] = None,
        learning_rate: float = 1e-4,
        warmup_steps: int = 1000,
        weight_decay: float = 0.01,
        kl_weight: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.kl_weight = kl_weight

        model_config = model_config or {}

        # Token embeddings (shared with diffusion model)
        from ..models.embeddings import TokenEmbedding
        self.token_embed = TokenEmbedding(vocab_size, d_model)

        # VAE model
        self.vae = TokenVAE(
            input_dim=d_model,
            latent_dim=latent_dim,
            kl_weight=kl_weight,
            **model_config,
        )

        # Metrics
        self.train_loss = 0.0
        self.num_training_steps = 0

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = AdamW(
            self.vae.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
        )

        # Warmup scheduler
        def lr_lambda(step):
            if step < self.warmup_steps:
                return float(step) / float(max(1, self.warmup_steps))
            return max(0.0, float(self.trainer.max_steps - step) / float(max(1, self.trainer.max_steps - self.warmup_steps)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def forward(self, input_ids: torch.Tensor):
        """Forward pass through VAE.

        Args:
            input_ids: Token IDs [batch_size, seq_len]

        Returns:
            x_recon: Reconstructed embeddings
            stats: VAE statistics (mu, logvar, z)
        """
        x_0 = self.token_embed(input_ids)
        x_recon, stats = self.vae(x_0, return_stats=True)
        return x_recon, stats

    def training_step(self, batch, batch_idx):
        """Training step."""
        input_ids = batch["input_ids"]

        # Get embeddings
        x_0 = self.token_embed(input_ids)

        # Forward through VAE
        x_recon, stats = self.vae(x_0, return_stats=True)
        mu = stats["mu"]
        logvar = stats["logvar"]

        # Compute loss
        loss, loss_dict = self.vae.compute_loss(x_0, x_recon, mu, logvar)

        # Logging
        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        self.log("train/recon_loss", loss_dict["recon"], sync_dist=True)
        self.log("train/kl_loss", loss_dict["kl"], sync_dist=True)
        self.log("train/learning_rate", self.optimizers().param_groups[0]["lr"], sync_dist=True)

        self.train_loss = loss.item()
        self.num_training_steps += 1

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        input_ids = batch["input_ids"]

        # Get embeddings
        x_0 = self.token_embed(input_ids)

        # Forward through VAE
        x_recon, stats = self.vae(x_0, return_stats=True)
        mu = stats["mu"]
        logvar = stats["logvar"]

        # Compute loss
        loss, loss_dict = self.vae.compute_loss(x_0, x_recon, mu, logvar)

        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        self.log("val/recon_loss", loss_dict["recon"], sync_dist=True)
        self.log("val/kl_loss", loss_dict["kl"], sync_dist=True)

        return loss

    def get_vae(self) -> TokenVAE:
        """Get the trained VAE model."""
        return self.vae

    def save_checkpoint(self, path: str):
        """Save VAE checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            "vae_state_dict": self.vae.state_dict(),
            "token_embed_state_dict": self.token_embed.state_dict(),
            "hparams": self.hparams,
        }
        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(cls, path: str, **override_params):
        """Load VAE checkpoint."""
        checkpoint = torch.load(path, map_location="cpu")
        hparams = checkpoint.get("hparams", {})
        hparams.update(override_params)

        module = cls(**hparams)
        module.vae.load_state_dict(checkpoint["vae_state_dict"])
        module.token_embed.load_state_dict(checkpoint["token_embed_state_dict"])

        return module


class VAETrainer:
    """Simple training loop for VAE without PyTorch Lightning.

    Args:
        vae: TokenVAE model
        token_embed: TokenEmbedding layer
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        device: Device to train on
        num_epochs: Number of epochs
        learning_rate: Learning rate
        warmup_steps: Number of warmup steps
        kl_weight: Weight for KL divergence
        gradient_accumulation_steps: Gradient accumulation
        use_amp: Use mixed precision
    """

    def __init__(
        self,
        vae: TokenVAE,
        token_embed,
        train_dataloader,
        val_dataloader=None,
        device: str = "cuda",
        num_epochs: int = 10,
        learning_rate: float = 1e-4,
        warmup_steps: int = 1000,
        weight_decay: float = 0.01,
        kl_weight: float = 1.0,
        gradient_accumulation_steps: int = 1,
        use_amp: bool = False,
    ):
        self.vae = vae.to(device)
        self.token_embed = token_embed.to(device)
        self.token_embed.requires_grad_(False)  # Freeze token embeddings during VAE training
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.kl_weight = kl_weight
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.use_amp = use_amp
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        # Optimizer - only optimize VAE parameters
        self.optimizer = AdamW(
            self.vae.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Tracking
        self.global_step = 0
        self.best_val_loss = float("inf")

    def train(self):
        """Run training loop."""
        total_steps = len(self.train_dataloader) * self.num_epochs

        for epoch in range(self.num_epochs):
            self.vae.train()
            epoch_loss = 0.0
            epoch_recon_loss = 0.0
            epoch_kl_loss = 0.0
            self.optimizer.zero_grad()

            for batch_idx, batch in enumerate(self.train_dataloader):
                # Learning rate warmup
                if self.global_step < self.warmup_steps:
                    lr = self.learning_rate * (self.global_step / self.warmup_steps)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = lr

                # Forward pass
                input_ids = batch["input_ids"].to(self.device)

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    # Get embeddings (no grad for token_embed)
                    with torch.no_grad():
                        x_0 = self.token_embed(input_ids)

                    # Forward through VAE
                    x_recon, stats = self.vae(x_0, return_stats=True)
                    mu = stats["mu"]
                    logvar = stats["logvar"]

                    # Compute loss
                    loss, loss_dict = self.vae.compute_loss(x_0, x_recon, mu, logvar)
                    loss = loss / self.gradient_accumulation_steps

                # Backward
                self.scaler.scale(loss).backward()

                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.vae.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                epoch_loss += loss.item() * self.gradient_accumulation_steps
                epoch_recon_loss += loss_dict["recon"].item()
                epoch_kl_loss += loss_dict["kl"].item()

                if batch_idx % 100 == 0:
                    print(
                        f"Epoch {epoch + 1}/{self.num_epochs} | "
                        f"Step {batch_idx}/{len(self.train_dataloader)} | "
                        f"Loss: {loss.item() * self.gradient_accumulation_steps:.4f} | "
                        f"Recon: {loss_dict['recon'].item():.4f} | "
                        f"KL: {loss_dict['kl'].item():.4f}"
                    )

            avg_epoch_loss = epoch_loss / len(self.train_dataloader)
            avg_recon_loss = epoch_recon_loss / len(self.train_dataloader)
            avg_kl_loss = epoch_kl_loss / len(self.train_dataloader)
            print(
                f"Epoch {epoch + 1} | "
                f"Avg Loss: {avg_epoch_loss:.4f} | "
                f"Avg Recon: {avg_recon_loss:.4f} | "
                f"Avg KL: {avg_kl_loss:.4f}"
            )

            # Validation
            if self.val_dataloader is not None:
                val_loss = self.validate()
                print(f"Validation Loss: {val_loss:.4f}")

                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    print(f"New best validation loss: {val_loss:.4f}")

    def validate(self):
        """Run validation."""
        self.vae.eval()
        val_loss = 0.0
        val_recon = 0.0
        val_kl = 0.0

        with torch.no_grad():
            for batch in self.val_dataloader:
                input_ids = batch["input_ids"].to(self.device)

                # Get embeddings
                x_0 = self.token_embed(input_ids)

                # Forward through VAE
                x_recon, stats = self.vae(x_0, return_stats=True)
                mu = stats["mu"]
                logvar = stats["logvar"]

                # Compute loss
                loss, loss_dict = self.vae.compute_loss(x_0, x_recon, mu, logvar)
                val_loss += loss.item()
                val_recon += loss_dict["recon"].item()
                val_kl += loss_dict["kl"].item()

        n = len(self.val_dataloader)
        return val_loss / n, val_recon / n, val_kl / n

    def save_checkpoint(self, path: str):
        """Save VAE checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            "vae_state_dict": self.vae.state_dict(),
            "config": {
                "input_dim": self.vae.input_dim,
                "latent_dim": self.vae.latent_dim,
                "hidden_dim": self.vae.hidden_dim,
                "num_layers": self.vae.num_layers,
                "kl_weight": self.vae.kl_weight,
            },
            "training_step": self.global_step,
        }
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")

    @staticmethod
    def load_checkpoint(path: str, map_location="cpu"):
        """Load VAE checkpoint.

        Returns:
            vae: TokenVAE model
            config: Configuration dictionary
            step: Training step
        """
        checkpoint = torch.load(path, map_location=map_location)
        config = checkpoint["config"]

        vae = TokenVAE(**config)
        vae.load_state_dict(checkpoint["vae_state_dict"])

        return vae, config, checkpoint.get("training_step", 0)
