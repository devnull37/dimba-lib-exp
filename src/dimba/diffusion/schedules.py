"""Noise schedules for diffusion (DIMBA).

The default cosine schedule follows Nichol & Dhariwal (2021), "Improved Denoising
Diffusion Probabilistic Models". Unlike the previous implementation, it now
actually enforces a **zero terminal SNR** per Lin et al. (2023), "Common
Diffusion Noise Schedules and Sample Steps are Flawed" (arXiv:2305.08891).

Why this matters: at inference we begin sampling from pure Gaussian noise, which
corresponds to ``alpha_cumprod == 0`` (zero signal-to-noise ratio) at the final
timestep. A vanilla cosine schedule leaves a small but *nonzero* terminal
``alpha_cumprod``, so the model is never trained on the pure-noise state it must
start denoising from -> a train/inference mismatch. Rescaling to zero terminal
SNR removes that mismatch. The previous code merely clamped ``alpha_cumprod`` to
a minimum of 1e-4 (which guarantees a *nonzero* terminal SNR) while its docstring
claimed to fix it; that is corrected here.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


def enforce_zero_terminal_snr(alphas_cumprod: torch.Tensor) -> torch.Tensor:
    """Rescale a monotonically-decreasing ``alphas_cumprod`` to zero terminal SNR.

    Keeps ``alphas_cumprod[0]`` unchanged and forces ``alphas_cumprod[-1] == 0``,
    linearly rescaling ``sqrt(alphas_cumprod)`` in between (Lin et al., 2023, Algo 1).

    Args:
        alphas_cumprod: 1D tensor of cumulative alpha products, decreasing in t.

    Returns:
        Rescaled ``alphas_cumprod`` with the same shape and a true zero terminal SNR.
    """
    sqrt_acp = alphas_cumprod.clamp(min=0.0).sqrt()
    sqrt_acp_0 = sqrt_acp[0].clone()
    sqrt_acp_T = sqrt_acp[-1].clone()

    # Shift so the final value is exactly 0, then scale so the first is unchanged.
    sqrt_acp = sqrt_acp - sqrt_acp_T
    sqrt_acp = sqrt_acp * (sqrt_acp_0 / (sqrt_acp_0 - sqrt_acp_T).clamp(min=1e-8))
    return sqrt_acp**2


def _reshape_to(coef: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Reshape a per-sample coefficient ``[B]`` to broadcast against ``x`` ``[B, ...]``."""
    return coef.view(-1, *([1] * (x.dim() - 1)))


class CosineNoiseSchedule(nn.Module):
    """Cosine noise schedule with optional zero-terminal-SNR rescaling.

    Args:
        num_steps: Number of diffusion steps ``T``.
        s: Offset parameter (default 0.008, per Nichol & Dhariwal).
        zero_terminal_snr: If True (default), rescale so ``alpha_cumprod_{T-1} == 0``.
    """

    def __init__(
        self,
        num_steps: int = 1000,
        s: float = 0.008,
        zero_terminal_snr: bool = True,
    ):
        super().__init__()
        self.num_steps = num_steps
        self.s = s
        self.zero_terminal_snr = zero_terminal_snr

        alphas_cumprod = self._compute_alphas_cumprod()
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
        betas = 1.0 - (alphas_cumprod / alphas_cumprod_prev.clamp(min=1e-8))

        # Registered as buffers so they follow the model across devices.
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("betas", betas)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt((1.0 - alphas_cumprod).clamp(min=0.0)),
        )

    def _compute_alphas_cumprod(self) -> torch.Tensor:
        """Compute cumulative alpha products using the (normalized) cosine schedule."""
        steps = torch.arange(self.num_steps, dtype=torch.float32)
        # f(t) = cos^2(((t/T + s) / (1 + s)) * pi/2)
        f = torch.cos(((steps / self.num_steps + self.s) / (1 + self.s)) * torch.pi * 0.5) ** 2
        alphas_cumprod = f / f[0]  # normalize so alphas_cumprod[0] == 1
        if self.zero_terminal_snr:
            alphas_cumprod = enforce_zero_terminal_snr(alphas_cumprod)
        return alphas_cumprod.clamp(min=0.0, max=1.0)

    def add_noise(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion: ``x_t = sqrt(acp_t) * x_0 + sqrt(1 - acp_t) * noise``.

        Args:
            x_0: Clean signal ``[B, L, D]`` (latent or embedding).
            t: Timesteps ``[B]`` in ``[0, num_steps - 1]``.
            noise: Optional pre-sampled noise, otherwise drawn from ``N(0, I)``.

        Returns:
            ``(x_t, noise)``.
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alpha = _reshape_to(self.sqrt_alphas_cumprod[t], x_0)
        sqrt_one_minus = _reshape_to(self.sqrt_one_minus_alphas_cumprod[t], x_0)
        x_t = sqrt_alpha * x_0 + sqrt_one_minus * noise
        return x_t, noise

    def velocity(self, x_0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """v-prediction target ``v = sqrt(acp) * noise - sqrt(1 - acp) * x_0`` (Salimans & Ho, 2022)."""
        sqrt_alpha = _reshape_to(self.sqrt_alphas_cumprod[t], x_0)
        sqrt_one_minus = _reshape_to(self.sqrt_one_minus_alphas_cumprod[t], x_0)
        return sqrt_alpha * noise - sqrt_one_minus * x_0

    def predict_x0_from_v(self, x_t: torch.Tensor, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Recover ``x_0`` from a v-prediction: ``x_0 = sqrt(acp) * x_t - sqrt(1 - acp) * v``."""
        sqrt_alpha = _reshape_to(self.sqrt_alphas_cumprod[t], x_t)
        sqrt_one_minus = _reshape_to(self.sqrt_one_minus_alphas_cumprod[t], x_t)
        return sqrt_alpha * x_t - sqrt_one_minus * v

    def predict_x0_from_noise(self, x_t: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Recover ``x_0`` from an eps-prediction: ``x_0 = (x_t - sqrt(1-acp)*eps) / sqrt(acp)``."""
        sqrt_alpha = _reshape_to(self.sqrt_alphas_cumprod[t], x_t).clamp(min=1e-8)
        sqrt_one_minus = _reshape_to(self.sqrt_one_minus_alphas_cumprod[t], x_t)
        return (x_t - sqrt_one_minus * noise) / sqrt_alpha

    def snr(self, t: torch.Tensor) -> torch.Tensor:
        """Signal-to-noise ratio ``acp / (1 - acp)`` at timestep ``t`` (for min-SNR weighting)."""
        acp = self.alphas_cumprod[t]
        return acp / (1.0 - acp).clamp(min=1e-8)

    def get_betas(self) -> torch.Tensor:
        """Get beta schedule coefficients."""
        return self.betas

    def get_alphas_cumprod(self) -> torch.Tensor:
        """Get cumulative alpha coefficients."""
        return self.alphas_cumprod
