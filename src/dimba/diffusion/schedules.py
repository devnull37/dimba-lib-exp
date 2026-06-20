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

import math
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
        noise_dist: str = "gaussian",
        noise_df: float = 5.0,
    ):
        super().__init__()
        self.num_steps = num_steps
        self.s = s
        self.zero_terminal_snr = zero_terminal_snr
        # Forward-process noise distribution. "gaussian" is the standard DDPM choice;
        # "student_t" injects heavy-tailed (Levy-like) corruption, rescaled to unit
        # variance so the SNR schedule stays calibrated (Heavy-Tailed Diffusion,
        # arXiv:2410.14171). Large noise_df -> Gaussian; small -> heavier tails.
        self.noise_dist = noise_dist
        self.noise_df = float(noise_df)

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
        # log-SNR per timestep (monotonically decreasing in t), used by logSNR-uniform
        # timestep sampling. Clamp acp away from {0,1} so the endpoints stay finite --
        # 1e-5 (not 1e-8) because (1 - 1e-8) rounds to exactly 1.0 in float32, which
        # would make log(1-acp) = -inf at t=0.
        acp_c = alphas_cumprod.clamp(1e-5, 1.0 - 1e-5)
        self.register_buffer("logsnr", torch.log(acp_c) - torch.log(1.0 - acp_c))

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
            noise = self.sample_noise(x_0)
        sqrt_alpha = _reshape_to(self.sqrt_alphas_cumprod[t], x_0)
        sqrt_one_minus = _reshape_to(self.sqrt_one_minus_alphas_cumprod[t], x_0)
        x_t = sqrt_alpha * x_0 + sqrt_one_minus * noise
        return x_t, noise

    def sample_noise(self, x_0: torch.Tensor) -> torch.Tensor:
        """Draw forward-process noise shaped like ``x_0`` from the configured distribution.

        ``gaussian`` -> standard normal. ``student_t`` -> unit-variance Student-t
        (heavy-tailed); rescaled by ``sqrt((df-2)/df)`` so the marginal variance matches
        N(0, I) and the SNR schedule stays valid.
        """
        if self.noise_dist == "student_t":
            df = self.noise_df
            n = torch.distributions.StudentT(df).sample(x_0.shape).to(x_0.device, x_0.dtype)
            if df > 2.0:
                n = n * math.sqrt((df - 2.0) / df)
            return n
        return torch.randn_like(x_0)

    def sample_timesteps(
        self,
        batch: int,
        device: torch.device,
        mode: str = "uniform",
        exclude_zero: bool = False,
        antithetic: bool = False,
    ) -> torch.Tensor:
        """Sample training timesteps ``[batch]`` with the chosen distribution.

        ``mode``: "uniform" (clock-uniform, the classic default); "logit_normal"
        (t = round((T-1)*sigmoid(N(0,1))), SD3/FLUX -- concentrates on mid-noise where
        token content is decided); "logsnr_uniform" (uniform over log-SNR, PLAID /
        Improved-Noise-Schedule -- equal effort per SNR decade).
        ``exclude_zero``: skip t=0 (degenerate under zero-terminal-SNR: x_t == x_0).
        ``antithetic``: pair each draw with its mirror ``T-1-t`` for variance reduction
        (MDLM / DiffuMamba low-discrepancy sampling).
        """
        if antithetic:
            half = (batch + 1) // 2
            base = self._draw_t(half, device, mode, exclude_zero)
            mirror = (self.num_steps - 1) - base
            return torch.cat([base, mirror], dim=0)[:batch]
        return self._draw_t(batch, device, mode, exclude_zero)

    def _draw_t(self, n: int, device, mode: str, exclude_zero: bool) -> torch.Tensor:
        T = self.num_steps
        lo = 1 if exclude_zero else 0
        if mode == "uniform":
            return torch.randint(lo, T, (n,), device=device)
        if mode == "logit_normal":
            s = torch.sigmoid(torch.randn(n, device=device))
            return (s * (T - 1)).round().long().clamp(lo, T - 1)
        if mode == "logsnr_uniform":
            ls = self.logsnr.to(device)                      # decreasing in t
            target = torch.empty(n, device=device).uniform_(ls[-1].item(), ls[0].item())
            idx_inc = torch.searchsorted(torch.flip(ls, dims=[0]), target).clamp(0, T - 1)
            return ((T - 1) - idx_inc).clamp(lo, T - 1).long()
        raise ValueError(f"unknown timestep sampling mode: {mode}")

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


class FlowMatchingSchedule(nn.Module):
    """Rectified flow matching schedule (Liu et al. 2022; Lipman et al. 2022).

    Forward process: ``x_t = (1-t)*x_0 + t*eps``  — linear interpolation between
    clean data (t=0) and noise (t=1).  The model predicts the *velocity*
    ``v = eps - x_0`` (the constant direction from data to noise).

    Inference is a simple ODE: integrate ``dx/dt = v_theta(x_t, t)`` from t=1
    back to t=0 using Euler or Heun steps — 8–20 steps is sufficient vs
    50–1000 for DDPM (trajectories are straight lines, not curved).

    Timestep sampling uses a logit-normal distribution (SD3 / FLUX default):
    ``t = sigmoid(N(mean, std))``, which concentrates training effort on
    t ≈ 0.5 where the most content decisions are made.

    References
        Rectified Flow:               arXiv:2209.03003  (Liu et al.)
        Flow Matching:                arXiv:2210.02747  (Lipman et al.)
        Stable Diffusion 3 schedule:  arXiv:2403.03206  (Esser et al.)
    """

    def __init__(
        self,
        logit_normal_sampling: bool = True,
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
    ) -> None:
        super().__init__()
        self.logit_normal_sampling = logit_normal_sampling
        self.logit_mean = logit_mean
        self.logit_std = logit_std

    def sample_timesteps(self, batch: int, device: torch.device) -> torch.Tensor:
        """Sample continuous ``t ∈ (0, 1)`` per example. Returns ``[B]`` float tensor."""
        if self.logit_normal_sampling:
            u = torch.sigmoid(
                torch.randn(batch, device=device) * self.logit_std + self.logit_mean
            )
        else:
            u = torch.rand(batch, device=device)
        return u.clamp(1e-5, 1.0 - 1e-5)

    def forward_process(
        self,
        x_0: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Linear interpolation: ``x_t = (1-t)*x_0 + t*noise``.

        Args:
            x_0: Clean signal ``[B, L, D]``.
            noise: Standard Gaussian noise, same shape.
            t: Timesteps ``[B]`` in ``(0, 1)``.

        Returns:
            Interpolated ``x_t``, same shape as ``x_0``.
        """
        t_ = _reshape_to(t, x_0)
        return (1.0 - t_) * x_0 + t_ * noise

    def velocity_target(self, x_0: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Ground-truth velocity ``v = noise - x_0``.

        The model should predict this from ``(x_t, t, cond)``.
        """
        return noise - x_0

    def x0_from_xt_and_velocity(
        self, x_t: torch.Tensor, v: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Recover ``x_0`` from the velocity prediction: ``x_0 = x_t - t*v``."""
        t_ = _reshape_to(t, x_t)
        return x_t - t_ * v

    def sample_noise(self, x_0: torch.Tensor) -> torch.Tensor:
        """Standard Gaussian noise shaped like ``x_0``."""
        return torch.randn_like(x_0)
