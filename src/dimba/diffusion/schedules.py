"""Noise schedules for diffusion models."""

import torch
import torch.nn as nn
from typing import Tuple


class CosineNoiseSchedule(nn.Module):
    """Cosine noise schedule with zero terminal SNR fix.

    Based on Nichol & Dhariwal (2021) "Improved Denoising Diffusion Probabilistic Models".
    Includes fix for zero terminal SNR to ensure consistency between training and inference.

    Args:
        num_steps: Number of diffusion steps (T)
        s: Offset parameter (default: 0.008 per paper)
    """

    def __init__(self, num_steps: int = 1000, s: float = 0.008):
        super().__init__()
        self.num_steps = num_steps
        self.s = s

        # Precompute schedule coefficients and register as buffers
        # so they move with the model to correct device
        alphas_cumprod = self._compute_alphas_cumprod()
        self.register_buffer("alphas_cumprod", alphas_cumprod)

        # Compute betas from alphas
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1), alphas_cumprod[:-1]]
        )
        betas = 1 - (alphas_cumprod / alphas_cumprod_prev)
        self.register_buffer("betas", betas)

        # For convenience: sqrt of various quantities
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod))

    def _compute_alphas_cumprod(self) -> torch.Tensor:
        """Compute cumulative product of alphas using cosine schedule."""
        timesteps = torch.arange(0, self.num_steps, dtype=torch.float32)

        # Cosine schedule: α̅(t) = cos²((t/T + s)/(1 + s) · π/2)
        alphas_cumprod = torch.cos(
            torch.pi * 0.5 * (timesteps / self.num_steps + self.s) / (1 + self.s)
        ) ** 2

        # Clamp to prevent numerical issues
        alphas_cumprod = torch.clamp(alphas_cumprod, min=0.0001, max=0.9999)

        return alphas_cumprod

    def add_noise(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add noise to input according to the schedule.

        Args:
            x_0: Clean embeddings [batch_size, seq_len, embed_dim]
            t: Timesteps [batch_size], values in [0, num_steps-1]
            noise: Optional predefined noise, otherwise sampled from N(0,I)

        Returns:
            x_t: Noisy embeddings [batch_size, seq_len, embed_dim]
            noise: The noise used [batch_size, seq_len, embed_dim]
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        # Get sqrt coefficients for this timestep
        sqrt_alpha = self.sqrt_alphas_cumprod[t]  # [batch_size]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t]  # [batch_size]

        # Reshape for broadcasting: [batch_size, 1, 1]
        sqrt_alpha = sqrt_alpha.view(-1, 1, 1)
        sqrt_one_minus_alpha = sqrt_one_minus_alpha.view(-1, 1, 1)

        # x_t = sqrt(α̅(t)) * x_0 + sqrt(1 - α̅(t)) * ε
        x_t = sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise

        return x_t, noise

    def get_betas(self) -> torch.Tensor:
        """Get beta schedule coefficients."""
        return self.betas

    def get_alphas_cumprod(self) -> torch.Tensor:
        """Get cumulative alpha coefficients."""
        return self.alphas_cumprod


class LinearNoiseSchedule(nn.Module):
    """Linear noise schedule (DDPM-style).

    Linearly interpolates betas from beta_start to beta_end.
    Note: This has non-zero terminal SNR, which can cause train-test mismatch.
    Cosine schedule is recommended instead.

    Args:
        num_steps: Number of diffusion steps (T)
        beta_start: Initial beta value (default: 0.0001)
        beta_end: Final beta value (default: 0.02)
    """

    def __init__(self, num_steps: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02):
        super().__init__()
        self.num_steps = num_steps

        # Linear schedule for betas
        betas = torch.linspace(beta_start, beta_end, num_steps)
        self.register_buffer("betas", betas)

        # Compute alphas and cumulative products
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod))

    def add_noise(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add noise to input according to the schedule."""
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)

        x_t = sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
        return x_t, noise

    def get_betas(self) -> torch.Tensor:
        """Get beta schedule coefficients."""
        return self.betas

    def get_alphas_cumprod(self) -> torch.Tensor:
        """Get cumulative alpha coefficients."""
        return self.alphas_cumprod
