"""PyTorch Lightning training module for DIMBA."""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import AdamW
from typing import Optional, Dict, Any
import os

from ..models.diffusion import DIMBA
from ..models.vae import TokenVAE
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
        ema_device: Optional device to store EMA weights; None keeps default Lightning placement
        ema_update_interval: Update EMA every N steps (default: 1)
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

        # Update EMA periodically to reduce transfer overhead for CPU-offloaded EMA.
        if self.use_ema and (self.global_step + 1) % self.ema_update_interval == 0:
            self._update_ema_model()

        # Logging
        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        self.log("train/learning_rate", self.optimizers().param_groups[0]["lr"], sync_dist=True)

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
