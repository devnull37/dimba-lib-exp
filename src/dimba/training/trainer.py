"""PyTorch Lightning training module for DIMBA."""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Optional, Dict, Any
import math

from ..models.diffusion import DIMBA
from ..diffusion.sampling import sample_timesteps


class DIMBALightningModule(pl.LightningModule):
    """PyTorch Lightning module for training DIMBA.

    Handles training loop, optimization, EMA, and logging.

    Args:
        vocab_size: Size of vocabulary
        model_config: Dictionary of model configuration parameters
        learning_rate: Learning rate for optimizer (default: 2e-5)
        warmup_steps: Number of warmup steps (default: 500)
        weight_decay: Weight decay for optimizer (default: 0.01)
        ema_decay: Exponential moving average decay (default: 0.9999)
        use_ema: Whether to use EMA (default: True)
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
    ):
        super().__init__()
        self.save_hyperparameters()

        self.vocab_size = vocab_size
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.ema_device = torch.device("cpu")

        # Build model
        self.model = DIMBA(vocab_size=vocab_size, **model_config)

        # EMA model
        if use_ema:
            self.ema_model = DIMBA(vocab_size=vocab_size, **model_config)
            # Keep EMA weights on CPU to avoid doubling GPU memory for large models.
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
                param.detach().to(device=self.ema_device, dtype=ema_param.dtype)
            )

    def _update_ema_model(self):
        """Update EMA model parameters."""
        with torch.no_grad():
            for ema_param, param in zip(
                self.ema_model.parameters(),
                self.model.parameters()
            ):
                param_cpu = param.detach().to(device=self.ema_device, dtype=ema_param.dtype)
                ema_param.data.mul_(self.ema_decay).add_(param_cpu, alpha=(1 - self.ema_decay))

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
        """Training step."""
        input_ids = batch["input_ids"]
        batch_size = input_ids.shape[0]

        # Sample random timesteps
        t = sample_timesteps(batch_size, self.model.num_diffusion_steps, self.device)

        # Forward pass
        x_pred, noise, latent_info = self.model(input_ids, t)

        # Get clean embeddings
        x_0 = self.model.token_embed(input_ids)

        # Compute loss (predict clean embeddings or latent targets)
        loss = self.loss_fn(x_pred, x_0) * self.model.recon_loss_weight
        if self.model.latent_diffusion and latent_info is not None:
            latent_loss = self.loss_fn(latent_info["z_pred"], latent_info["z_0"])
            loss = loss + latent_loss * self.model.latent_loss_weight

        # Update EMA
        if self.use_ema:
            self._update_ema_model()

        # Logging
        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        self.log("train/learning_rate", self.optimizers().param_groups[0]["lr"], sync_dist=True)

        self.train_loss = loss.item()
        self.num_training_steps += 1

        return loss

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


class SimpleTrainer:
    """Simple training loop without PyTorch Lightning (for debugging/custom needs).

    Args:
        model: DIMBA model
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        device: Device to train on
        num_epochs: Number of epochs
        learning_rate: Learning rate
        warmup_steps: Number of warmup steps
        ema_decay: EMA decay rate
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
    ):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.ema_decay = ema_decay

        # EMA model
        self.ema_model = DIMBA(
            vocab_size=model.vocab_size,
            d_model=model.d_model,
            d_prompt=model.d_prompt,
            num_diffusion_steps=model.num_diffusion_steps,
        ).to(device)
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
        """Run training loop."""
        total_steps = len(self.train_dataloader) * self.num_epochs

        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_loss = 0.0

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

                loss = self.loss_fn(x_pred, x_0)

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                # Update EMA
                self._update_ema()

                epoch_loss += loss.item()
                self.global_step += 1

                if batch_idx % 100 == 0:
                    print(
                        f"Epoch {epoch + 1}/{self.num_epochs} | "
                        f"Step {batch_idx}/{len(self.train_dataloader)} | "
                        f"Loss: {loss.item():.4f}"
                    )

            avg_epoch_loss = epoch_loss / len(self.train_dataloader)
            print(f"Epoch {epoch + 1} | Avg Loss: {avg_epoch_loss:.4f}")

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
