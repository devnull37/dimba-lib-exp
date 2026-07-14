"""PyTorch Lightning training module for DIMBA."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Optional, Dict, Any, Tuple, List
import math
import os

from ..models.diffusion import DIMBA
from ..models.vae import TokenVAE
from ..diffusion.sampling import sample_timesteps, sample_from_model
from ..utils.checkpointing import ProgressiveCheckpointManager, atomic_torch_save
from .fused_ce import output_head_cross_entropy
from .optimizers import build_optimizer


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
    if delta_min < 0 or delta_max < delta_min:
        raise ValueError("expected 0 <= delta_min <= delta_max")
    device = t_early.device
    batch_size = input_ids.shape[0]

    # Adaptive delta sampling: ensure delta doesn't exceed t_early - delta_min
    # This prevents t_late from becoming negative
    max_possible_delta = torch.clamp(t_early - delta_min, min=0)
    effective_max_delta = torch.min(
        torch.full_like(t_early, delta_max),
        max_possible_delta
    )

    # Vectorized per-row integer sampling. The former list comprehension called
    # ``eff.item()`` once per sample, synchronizing CUDA for every row.
    valid_mask = effective_max_delta >= delta_min
    span = (effective_max_delta - delta_min + 1).clamp(min=1)
    delta = delta_min + torch.floor(torch.rand(batch_size, device=device) * span).long()
    delta = torch.where(valid_mask, delta, torch.zeros_like(delta))

    # Compute t_late
    t_late = t_early - delta

    # Encode to latent space (same as main loss)
    z_0 = model.encode_latent(x_0)

    # Add noise at both timesteps
    x_t_early, _ = model.noise_schedule.add_noise(z_0, t_early)
    x_t_late, _ = model.noise_schedule.add_noise(z_0, t_late)

    # Unconditional (null) conditioning -- avoids the prompt leak; CDLM aligns the
    # model's own clean-latent predictions across noise levels.
    cond = model.conditioning_from_prompt(None, batch_size, device, drop_cond=True)

    # Predict clean latents at both timesteps (the later one is a stop-grad target).
    z_pred_early = model.denoise_to_x0_latent(x_t_early, t_early, cond)
    with torch.no_grad():
        z_pred_late = model.denoise_to_x0_latent(x_t_late, t_late, cond)

    # Weight by remaining noise level at t_late
    # Positions with more remaining noise get higher weight
    noise_level_late = model.noise_schedule.sqrt_one_minus_alphas_cumprod[t_late]
    noise_level_late = noise_level_late.view(-1, 1, 1)
    weights = noise_level_late.clamp(min=0.01)

    # Compute weighted MSE
    diff = (z_pred_early - z_pred_late.detach()) * weights
    diff = diff * valid_mask[:, None, None]
    values_per_sample = diff.shape[1] * diff.shape[2]
    consistency_loss = diff.square().sum() / (
        valid_mask.sum().clamp(min=1) * values_per_sample
    )

    return consistency_loss


def _run_dimba_loss_forward(
    model: DIMBA,
    input_ids: torch.Tensor,
    t: torch.Tensor,
    prompt_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Run the possibly self-conditioned student pass used by the loss."""
    if getattr(model, "self_conditioning", False) and bool(torch.rand(()) < 0.5):
        shared_noise = torch.randn(
            input_ids.shape[0], input_ids.shape[1], model.d_latent, device=input_ids.device
        )
        with torch.no_grad():
            _, _, info_sc = model(input_ids, t, noise=shared_noise, prompt_mask=prompt_mask)
        x_self_cond = info_sc["z0_hat"].detach()
        x_pred, _noise, info = model(
            input_ids, t, noise=shared_noise, prompt_mask=prompt_mask, x_self_cond=x_self_cond
        )
    else:
        x_pred, _noise, info = model(input_ids, t, prompt_mask=prompt_mask)
    return x_pred, info


def compute_dimba_losses(
    model: DIMBA,
    input_ids: torch.Tensor,
    t: torch.Tensor,
    *,
    ce_loss_weight: float = 1.0,
    min_snr_gamma: float = 5.0,
    snr_floor: float = 1e-3,
    ce_time_fade: bool = False,
    prompt_mask: Optional[torch.Tensor] = None,
    loss_mask: Optional[torch.Tensor] = None,
    fused_ce_mode: str = "auto",
    _forward_output: Optional[Tuple[torch.Tensor, Dict[str, torch.Tensor]]] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute the DIMBA training loss.

    Combines three signals the original MSE-only objective lacked:

    * **Min-SNR-weighted diffusion regression** in the latent space (Hang et al.,
      2023): per-timestep weight ``min(SNR, gamma)`` for x0-prediction
      (``/(SNR+1)`` for v-prediction). Target is ``z_0`` (x0) or the velocity ``v``.
    * **Cross-entropy / rounding anchor** (Diffusion-LM, Li et al. 2022): trains the
      output head + decoder and ties the continuous prediction back to real tokens.
    * **Latent autoencoder consistency** + optional VAE KL when diffusing in a
      learned latent space.

    When ``prompt_mask`` is given (True = clean prompt context), the diffusion and
    cross-entropy terms use response positions only.

    Returns:
        ``(loss, parts)`` where ``parts`` holds detached scalar components.
    """
    # Stage-3 teacher KD consumes the same predicted x0. It can supply the already
    # computed forward here to avoid running the entire Mamba stack twice.
    if _forward_output is None:
        x_pred, info = _run_dimba_loss_forward(model, input_ids, t, prompt_mask)
    else:
        x_pred, info = _forward_output
    diffuse_mask = info.get("diffuse_mask")

    # --- diffusion regression (min-SNR weighted), in latent space ---
    if getattr(model, "use_flow_matching", False) and model.prediction_type == "v":
        raise ValueError(
            "prediction_type='v' is not supported with use_flow_matching=True; "
            "flow matching uses raw x0 prediction (set prediction_type='x0')."
        )
    if model.prediction_type == "v":
        target = model.noise_schedule.velocity(info["z_0"], info["noise"], t)
        pred = info["pred_raw"]
    else:
        target = info["z_0"]
        pred = info["z0_hat"]
    # Effective per-position loss mask: response positions (diffuse_mask) intersected
    # with non-padding (loss_mask). None -> mean over all positions (unchanged default).
    eff = None
    if diffuse_mask is not None:
        eff = diffuse_mask.to(torch.float32)
    if loss_mask is not None:
        lm = loss_mask.to(torch.float32)
        eff = lm if eff is None else eff * lm

    per_pos = ((pred - target) ** 2).mean(dim=-1)  # [B, L]
    if eff is not None:
        per_sample = (per_pos * eff).sum(dim=1) / eff.sum(dim=1).clamp(min=1.0)
    else:
        per_sample = per_pos.mean(dim=1)

    if getattr(model, "use_flow_matching", False):
        # Match forward(): t is a discrete index -> continuous noise level.
        t_cont = t.float() / (model.num_diffusion_steps - 1)
        t_cont = t_cont.clamp(1e-5, 1.0 - 1e-5)
        snr_flow = ((1.0 - t_cont) / t_cont) ** 2  # SNR of x_t=(1-t)z0+t*eps
        weight = torch.clamp(snr_flow, max=min_snr_gamma)
        if model.prediction_type == "v":
            # flow velocity target has unit-ish scale; SNR+1 normalization keeps min-SNR-v consistent
            weight = weight / (snr_flow + 1.0)
        # snr_floor: the default 1e-3 leaves the high-noise region effectively
        # untrained, so x0_hat degenerates to a noise-independent blend at t~1
        # and generation from pure noise fails (see docs/ROOT_CAUSE_BUDGET28.md).
        # A floor >=0.5 keeps gradient pressure on the mode-selection regime.
        weight = weight.clamp(min=snr_floor)
    else:
        snr = model.noise_schedule.snr(t)
        weight = torch.clamp(snr, max=min_snr_gamma)
        if model.prediction_type == "v":
            weight = weight / (snr + 1.0)
        weight = weight.clamp(min=snr_floor)
    diff_loss = (per_sample * weight).mean()

    # --- cross-entropy / rounding anchor ---
    if eff is None and not ce_time_fade:
        # Uniform reduction: Liger can fuse projection + CE without [B,L,V].
        ce_loss = output_head_cross_entropy(
            model,
            x_pred,
            input_ids,
            reduction="mean",
            mode=fused_ce_mode,
            uniform_reduction=True,
        )
    else:
        # Non-uniform sample/mask weights require exact unreduced gradients. Select
        # active tokens before native projection; upstream Liger cannot do this yet.
        batch, length = input_ids.shape
        active = None if eff is None else eff != 0
        ce_per = output_head_cross_entropy(
            model,
            x_pred,
            input_ids,
            active_mask=active,
            reduction="none",
            mode=fused_ce_mode,
            uniform_reduction=False,
        )
        ce_per_fp32 = ce_per.float()
        if eff is not None:
            weights = eff.float()
            sample_ids = torch.arange(batch, device=input_ids.device)[:, None].expand(
                batch, length
            )
            selected_samples = sample_ids[active]
            selected_weights = weights[active]
            ce_sums = torch.zeros(batch, device=ce_per.device, dtype=torch.float32)
            ce_sums.scatter_add_(0, selected_samples, ce_per_fp32 * selected_weights)
            ce_sample = ce_sums / weights.sum(dim=1).clamp(min=1.0)
        else:
            ce_sample = ce_per_fp32.view(batch, length).mean(dim=1)
        if ce_time_fade:
            # At high noise the CE-optimal output is the unigram distribution, so a
            # full-strength anchor there actively teaches token-frequency spam.
            t_fade = (t.float() / (model.num_diffusion_steps - 1)).clamp(0.0, 1.0)
            ce_sample = ce_sample * (1.0 - t_fade)
        ce_loss = ce_sample.mean()

    loss = model.recon_loss_weight * diff_loss + ce_loss_weight * ce_loss
    parts = {"diff_loss": diff_loss.detach(), "ce_loss": ce_loss.detach()}

    # --- latent autoencoder consistency + optional VAE KL ---
    if model.latent_diffusion:
        x_0 = model.token_embed(input_ids)
        ae_loss = F.mse_loss(model.decode_latent(info["z_0"]), x_0)
        loss = loss + model.latent_loss_weight * ae_loss
        parts["ae_loss"] = ae_loss.detach()
    if info.get("vae_kl_loss") is not None:
        kl = info["vae_kl_loss"] / max(1, info["z_0"].numel())
        loss = loss + getattr(model, "vae_kl_weight", 1.0) * kl
        parts["vae_kl"] = kl.detach()

    return loss, parts


class GenerationSampleCallback(pl.Callback):
    """Periodically generate + decode text during training so generation quality is
    actually visible. (The prior model's failure went unnoticed because validation
    only measured a single mid-noise reconstruction loss.)

    Args:
        tokenizer: object with ``.encode(str)`` / ``.decode(ids)``.
        prompt: optional prompt string (None -> unconditional generation).
        seq_len: number of response tokens to generate.
        num_steps: diffusion steps (None -> model default).
        every_n_epochs: run every N validation epochs.
        num_samples: how many samples to print.
        temperature: sampling temperature.
    """

    def __init__(self, tokenizer, prompt=None, seq_len=128, num_steps=None,
                 every_n_epochs=1, num_samples=2, temperature=0.8):
        super().__init__()
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.seq_len = seq_len
        self.num_steps = num_steps
        self.every_n_epochs = max(1, int(every_n_epochs))
        self.num_samples = num_samples
        self.temperature = temperature

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return
        model = pl_module.model
        was_training = model.training
        model.eval()
        try:
            prompt_ids = None
            if self.prompt is not None:
                ids = self.tokenizer.encode(self.prompt)
                prompt_ids = torch.tensor([ids], dtype=torch.long, device=pl_module.device)
            print(f"\n[gen @ epoch {trainer.current_epoch + 1}]")
            for i in range(self.num_samples):
                out = sample_from_model(
                    model, prompt_ids, self.seq_len, num_steps=self.num_steps,
                    temperature=self.temperature, device=pl_module.device,
                )
                print(f"  sample {i + 1}: {self.tokenizer.decode(out[0])!r}")
        finally:
            if was_training:
                model.train()


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
        optimizer: ``"adamw"`` (default) or the Muon + AdamW hybrid
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
        ce_loss_weight: float = 1.0,
        min_snr_gamma: float = 5.0,
        fused_ce_mode: str = "auto",
        timestep_sampling: str = "uniform",
        antithetic_t: bool = False,
        exclude_zero_t: bool = False,
        progressive_milestones: Optional[List[int]] = None,
        progressive_save_dir: str = "./progressive_checkpoints",
        enable_progressive_checkpoints: bool = False,
        optimizer: str = "adamw",
    ):
        super().__init__()
        self.save_hyperparameters()

        self.vocab_size = vocab_size
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer.lower()
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.ema_update_interval = max(1, int(ema_update_interval))
        self.ema_device = torch.device(ema_device) if ema_device is not None else None

        # CDLM consistency training parameters
        self.use_consistency_training = use_consistency_training
        self.consistency_loss_weight = consistency_loss_weight
        self.consistency_delta_min = consistency_delta_min
        self.consistency_delta_max = consistency_delta_max

        # Loss weights for the cross-entropy anchor and min-SNR weighting.
        self.ce_loss_weight = ce_loss_weight
        self.min_snr_gamma = min_snr_gamma
        self.fused_ce_mode = fused_ce_mode

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
        optimizer = build_optimizer(
            self.model,
            name=self.optimizer_name,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
            fused=self.device.type == "cuda",
        )

        # Linear warmup -> cosine decay to zero over the whole run.
        # NOTE: the previous schedule divided by ``self.trainer.max_steps``, which
        # is -1 when training by ``max_epochs`` — that broke the decay entirely.
        # ``estimated_stepping_batches`` is the correct total (accounts for epochs,
        # devices, and gradient accumulation).
        total_steps = max(2, int(self.trainer.estimated_stepping_batches))
        warmup = max(1, min(self.warmup_steps, total_steps - 1))

        def lr_lambda(step):
            if step < warmup:
                return float(step) / float(warmup)
            progress = float(step - warmup) / float(max(1, total_steps - warmup))
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

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

        # Sample timesteps for the main denoising loss (distribution is configurable:
        # uniform / logit_normal / logsnr_uniform, with optional antithetic pairing).
        t = self.model.noise_schedule.sample_timesteps(
            batch_size, self.device,
            mode=self.hparams.timestep_sampling,
            antithetic=self.hparams.antithetic_t,
            exclude_zero=self.hparams.exclude_zero_t,
        )

        prompt_mask = batch.get("prompt_mask")
        loss, loss_parts = compute_dimba_losses(
            self.model,
            input_ids,
            t,
            ce_loss_weight=self.ce_loss_weight,
            min_snr_gamma=self.min_snr_gamma,
            prompt_mask=prompt_mask,
            loss_mask=batch.get("attention_mask"),
            fused_ce_mode=self.fused_ce_mode,
        )

        # CDLM consistency loss: align the model's clean-latent predictions across timesteps.
        consistency_loss = torch.tensor(0.0, device=self.device)
        if self.use_consistency_training and self.consistency_loss_weight > 0:
            x_0 = self.model.token_embed(input_ids)
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
        # Per-step distributed reductions serialize every rank. Rank-zero's local
        # minibatch is an unbiased progress signal; validation remains globally reduced.
        self.log("train/loss", loss, prog_bar=True, sync_dist=False)
        for _name, _val in loss_parts.items():
            self.log(f"train/{_name}", _val, sync_dist=False)
        self.log(
            "train/learning_rate",
            self.optimizers().param_groups[0]["lr"],
            sync_dist=False,
        )
        if self.use_consistency_training:
            self.log(
                "train/consistency_loss",
                consistency_loss,
                prog_bar=False,
                sync_dist=False,
            )

        self.train_loss = loss.detach()
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

        # Sample timesteps across the whole schedule. A fixed midpoint (the previous
        # behavior) only probes easy mid-noise denoising and hides high-noise /
        # generative failure -- exactly the regime where this model was found to break.
        t = sample_timesteps(batch_size, self.model.num_diffusion_steps, self.device)

        # Keep validation on the active training model to avoid moving EMA to GPU.
        model = self.model

        # Compute the same combined loss as training (response-aware if masked).
        loss, _ = compute_dimba_losses(
            model,
            input_ids,
            t,
            ce_loss_weight=self.ce_loss_weight,
            min_snr_gamma=self.min_snr_gamma,
            prompt_mask=batch.get("prompt_mask"),
            loss_mask=batch.get("attention_mask"),
            fused_ce_mode=self.fused_ce_mode,
        )

        self.log("val/loss", loss, prog_bar=True, sync_dist=True)

        return loss

    def test_step(self, batch, batch_idx):
        """Test step."""
        input_ids = batch["input_ids"]
        batch_size = input_ids.shape[0]

        # Evaluate the combined loss at a few timesteps.
        losses = []
        for t_val in [100, 500, 900]:
            t = torch.full((batch_size,), min(t_val, self.model.num_diffusion_steps - 1), device=self.device)
            loss, _ = compute_dimba_losses(
                self.model,
                input_ids,
                t,
                ce_loss_weight=self.ce_loss_weight,
                min_snr_gamma=self.min_snr_gamma,
                fused_ce_mode=self.fused_ce_mode,
                prompt_mask=batch.get("prompt_mask"),
                loss_mask=batch.get("attention_mask"),
            )
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
        """Copy EMA weights into the main model so the final saved weights are the EMA."""
        if self.use_ema:
            for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
                param.data.copy_(ema_param.detach().to(param.device, param.dtype))


def get_model_config(model: DIMBA) -> Dict[str, Any]:
    """Extract configuration from a DIMBA model for creating replicas.
    
    Preserves all model configuration parameters including architecture-specific
    settings like d_state, d_conv, expand, etc.
    
    Args:
        model: DIMBA model instance
        
    Returns:
        Dictionary with all model configuration parameters
    """
    if hasattr(model, "config"):
        return model.config
    # Best-effort fallback for models that predate the stored config.
    return {
        "vocab_size": model.vocab_size,
        "d_model": model.d_model,
        "d_prompt": model.d_prompt,
        "num_diffusion_steps": model.num_diffusion_steps,
        "latent_diffusion": model.latent_diffusion,
        "d_latent": getattr(model, "d_latent", None),
        "latent_loss_weight": getattr(model, "latent_loss_weight", 1.0),
        "recon_loss_weight": getattr(model, "recon_loss_weight", 1.0),
    }


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
        weight_decay: Optimizer weight decay (default: 0.01)
        optimizer: ``"adamw"`` (default) or the Muon + AdamW hybrid
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
        ce_loss_weight: float = 1.0,
        min_snr_gamma: float = 5.0,
        fused_ce_mode: str = "auto",
        progressive_milestones: Optional[List[int]] = None,
        progressive_save_dir: str = "./progressive_checkpoints",
        enable_progressive_checkpoints: bool = False,
        weight_decay: float = 0.01,
        optimizer: str = "adamw",
    ):
        device = torch.device(device)
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer.lower()
        self.ema_decay = ema_decay

        # CDLM parameters
        self.use_consistency_training = use_consistency_training
        self.consistency_loss_weight = consistency_loss_weight
        self.consistency_delta_min = consistency_delta_min
        self.consistency_delta_max = consistency_delta_max

        # Loss weights for the cross-entropy anchor and min-SNR weighting.
        self.ce_loss_weight = ce_loss_weight
        self.min_snr_gamma = min_snr_gamma
        self.fused_ce_mode = fused_ce_mode

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
        self.optimizer = build_optimizer(
            model,
            name=self.optimizer_name,
            lr=learning_rate,
            weight_decay=weight_decay,
            fused=device.type == "cuda",
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
            epoch_metrics = torch.zeros(3, device=self.device)

            for batch_idx, batch in enumerate(self.train_dataloader):
                # Learning rate warmup
                if self.global_step < self.warmup_steps:
                    lr = self.learning_rate * (self.global_step / self.warmup_steps)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = lr

                # Forward pass
                input_ids = batch["input_ids"].to(
                    self.device, non_blocking=self.device.type == "cuda"
                )
                batch_size = input_ids.shape[0]
                t = sample_timesteps(batch_size, self.model.num_diffusion_steps, torch.device(self.device))

                _attn_mask = batch.get("attention_mask")
                if _attn_mask is not None:
                    _attn_mask = _attn_mask.to(
                        self.device, non_blocking=self.device.type == "cuda"
                    )
                _prompt_mask = batch.get("prompt_mask")
                if _prompt_mask is not None:
                    _prompt_mask = _prompt_mask.to(
                        self.device, non_blocking=self.device.type == "cuda"
                    )
                loss, _parts = compute_dimba_losses(
                    self.model,
                    input_ids,
                    t,
                    ce_loss_weight=self.ce_loss_weight,
                    min_snr_gamma=self.min_snr_gamma,
                    prompt_mask=_prompt_mask,
                    loss_mask=_attn_mask,
                    fused_ce_mode=self.fused_ce_mode,
                )
                denoise_loss = _parts["diff_loss"]

                # CDLM Consistency loss
                consistency_loss = torch.tensor(0.0, device=self.device)
                if self.use_consistency_training and self.consistency_loss_weight > 0:
                    x_0 = self.model.token_embed(input_ids)
                    consistency_loss = compute_consistency_loss(
                        model=self.model,
                        input_ids=input_ids,
                        x_0=x_0,
                        t_early=t,
                        delta_min=self.consistency_delta_min,
                        delta_max=self.consistency_delta_max,
                    )
                    loss = loss + self.consistency_loss_weight * consistency_loss

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                # Update EMA
                self._update_ema()

                batch_metrics = torch.stack(
                    (loss.detach(), denoise_loss.detach(), consistency_loss.detach())
                ).float()
                epoch_metrics += batch_metrics
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
                                "train_loss": batch_metrics[0].item(),
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
                    current_loss, current_denoise, current_consistency = (
                        batch_metrics.cpu().tolist()
                    )
                    log_msg = (
                        f"Epoch {epoch + 1}/{self.num_epochs} | "
                        f"Step {batch_idx}/{len(self.train_dataloader)} | "
                        f"Loss: {current_loss:.4f}"
                    )
                    if self.use_consistency_training:
                        log_msg += (
                            f" (Denoise: {current_denoise:.4f}, "
                            f"Consistency: {current_consistency:.4f})"
                        )
                    print(log_msg)

            avg_epoch_loss, avg_denoise_loss, avg_consistency_loss = (
                epoch_metrics.div(len(self.train_dataloader)).cpu().tolist()
            )
            print(f"Epoch {epoch + 1} | Avg Loss: {avg_epoch_loss:.4f} | Denoise: {avg_denoise_loss:.4f}", end="")
            if self.use_consistency_training:
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
        val_loss = torch.zeros((), device=self.device)

        with torch.no_grad():
            for batch in self.val_dataloader:
                input_ids = batch["input_ids"].to(
                    self.device, non_blocking=self.device.type == "cuda"
                )
                batch_size = input_ids.shape[0]
                # Sample across the full schedule (a fixed midpoint hides high-noise /
                # generative failure) -- matches DIMBALightningModule.validation_step.
                t = sample_timesteps(batch_size, self.model.num_diffusion_steps, torch.device(self.device))

                _attn_mask = batch.get("attention_mask")
                if _attn_mask is not None:
                    _attn_mask = _attn_mask.to(
                        self.device, non_blocking=self.device.type == "cuda"
                    )
                _prompt_mask = batch.get("prompt_mask")
                if _prompt_mask is not None:
                    _prompt_mask = _prompt_mask.to(
                        self.device, non_blocking=self.device.type == "cuda"
                    )
                loss, _ = compute_dimba_losses(
                    self.model,
                    input_ids,
                    t,
                    ce_loss_weight=self.ce_loss_weight,
                    min_snr_gamma=self.min_snr_gamma,
                    prompt_mask=_prompt_mask,
                    loss_mask=_attn_mask,
                    fused_ce_mode=self.fused_ce_mode,
                )
                val_loss += loss.detach()

        return (val_loss / len(self.val_dataloader)).item()


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
        self.log("train/loss", loss, prog_bar=True, sync_dist=False)
        self.log("train/recon_loss", loss_dict["recon"], sync_dist=False)
        self.log("train/kl_loss", loss_dict["kl"], sync_dist=False)
        self.log(
            "train/learning_rate",
            self.optimizers().param_groups[0]["lr"],
            sync_dist=False,
        )

        self.train_loss = loss.detach()
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
        checkpoint = {
            "vae_state_dict": self.vae.state_dict(),
            "token_embed_state_dict": self.token_embed.state_dict(),
            "hparams": self.hparams,
        }
        atomic_torch_save(checkpoint, path)

    @classmethod
    def load_checkpoint(cls, path: str, **override_params):
        """Load VAE checkpoint."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
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
        device = torch.device(device)
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
        amp_enabled = use_amp and device.type == "cuda"
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        else:  # PyTorch 2.0 compatibility
            self.scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

        # Optimizer - only optimize VAE parameters
        self.optimizer = AdamW(
            self.vae.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            fused=device.type == "cuda",
        )

        # Tracking
        self.global_step = 0
        self.best_val_loss = float("inf")

    def train(self):
        """Run training loop."""
        total_steps = len(self.train_dataloader) * self.num_epochs

        for epoch in range(self.num_epochs):
            self.vae.train()
            epoch_metrics = torch.zeros(3, device=self.device)
            self.optimizer.zero_grad()

            for batch_idx, batch in enumerate(self.train_dataloader):
                # Learning rate warmup
                if self.global_step < self.warmup_steps:
                    lr = self.learning_rate * (self.global_step / self.warmup_steps)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = lr

                # Forward pass
                input_ids = batch["input_ids"].to(
                    self.device, non_blocking=self.device.type == "cuda"
                )

                with torch.autocast(
                    device_type="cuda",
                    enabled=self.use_amp and self.device.type == "cuda",
                ):
                    # Get embeddings (no grad for token_embed)
                    with torch.no_grad():
                        x_0 = self.token_embed(input_ids)

                    # Forward through VAE
                    x_recon, stats = self.vae(x_0, return_stats=True)
                    mu = stats["mu"]
                    logvar = stats["logvar"]

                    # Compute loss
                    loss, loss_dict = self.vae.compute_loss(x_0, x_recon, mu, logvar)
                    group_start = (
                        batch_idx // self.gradient_accumulation_steps
                    ) * self.gradient_accumulation_steps
                    group_size = min(
                        self.gradient_accumulation_steps,
                        len(self.train_dataloader) - group_start,
                    )
                    scaled_loss = loss / group_size

                # Backward
                self.scaler.scale(scaled_loss).backward()

                if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or (
                    batch_idx + 1 == len(self.train_dataloader)
                ):
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.vae.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                batch_metrics = torch.stack(
                    (
                        loss.detach(),
                        loss_dict["recon"].detach(),
                        loss_dict["kl"].detach(),
                    )
                ).float()
                epoch_metrics += batch_metrics

                if batch_idx % 100 == 0:
                    current_loss, current_recon, current_kl = batch_metrics.cpu().tolist()
                    print(
                        f"Epoch {epoch + 1}/{self.num_epochs} | "
                        f"Step {batch_idx}/{len(self.train_dataloader)} | "
                        f"Loss: {current_loss:.4f} | "
                        f"Recon: {current_recon:.4f} | "
                        f"KL: {current_kl:.4f}"
                    )

            avg_epoch_loss, avg_recon_loss, avg_kl_loss = (
                epoch_metrics.div(len(self.train_dataloader)).cpu().tolist()
            )
            print(
                f"Epoch {epoch + 1} | "
                f"Avg Loss: {avg_epoch_loss:.4f} | "
                f"Avg Recon: {avg_recon_loss:.4f} | "
                f"Avg KL: {avg_kl_loss:.4f}"
            )

            # Validation
            if self.val_dataloader is not None:
                val_loss, val_recon, val_kl = self.validate()
                print(
                    f"Validation Loss: {val_loss:.4f} | "
                    f"Recon: {val_recon:.4f} | KL: {val_kl:.4f}"
                )

                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    print(f"New best validation loss: {val_loss:.4f}")

    def validate(self):
        """Run validation."""
        self.vae.eval()
        metrics = torch.zeros(3, device=self.device)

        with torch.no_grad():
            for batch in self.val_dataloader:
                input_ids = batch["input_ids"].to(
                    self.device, non_blocking=self.device.type == "cuda"
                )

                # Get embeddings
                x_0 = self.token_embed(input_ids)

                # Forward through VAE
                x_recon, stats = self.vae(x_0, return_stats=True)
                mu = stats["mu"]
                logvar = stats["logvar"]

                # Compute loss
                loss, loss_dict = self.vae.compute_loss(x_0, x_recon, mu, logvar)
                metrics += torch.stack(
                    (loss.detach(), loss_dict["recon"].detach(), loss_dict["kl"].detach())
                ).float()

        n = len(self.val_dataloader)
        return tuple(metrics.div(n).cpu().tolist())

    def save_checkpoint(self, path: str):
        """Save VAE checkpoint."""
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
        atomic_torch_save(checkpoint, path)
        print(f"Saved checkpoint to {path}")

    @staticmethod
    def load_checkpoint(path: str, map_location="cpu"):
        """Load VAE checkpoint.

        Returns:
            vae: TokenVAE model
            config: Configuration dictionary
            step: Training step
        """
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        config = checkpoint["config"]

        vae = TokenVAE(**config)
        vae.load_state_dict(checkpoint["vae_state_dict"])

        return vae, config, checkpoint.get("training_step", 0)
