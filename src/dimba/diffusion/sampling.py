"""Sampling and inference for DIMBA.

Rewritten to use a correct, x0-parameterized DDIM update (Song et al., 2021) in
the diffusion *latent* space, with:

* **Clean-prefix conditioning** — the prompt latents are placed (clean) at the
  front of the sequence and held fixed every step, so the bidirectional denoiser
  attends to real prompt context exactly as during training. Only the response
  positions are denoised.
* **Classifier-free guidance** — when ``guidance_scale != 1`` we combine a
  prompt-conditioned and a null-conditioned x0 prediction.
* **Self-conditioning** — the previous x0 estimate is carried across steps.

The previous sampler used an ad-hoc update, padded per-position prompt
conditioning with zeros, and printed progress from inside the library; all fixed.
"""

import logging
import torch
import torch.nn.functional as F
from typing import Optional

logger = logging.getLogger(__name__)


def _coef(value: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Reshape a scalar schedule coefficient for broadcasting against ``like``."""
    return value.view(*([1] * like.dim())).to(like.device, like.dtype)


def _ddim_step(
    x_t: torch.Tensor,
    x0_hat: torch.Tensor,
    acp_t: torch.Tensor,
    acp_prev: torch.Tensor,
    eta: float,
) -> torch.Tensor:
    """One x0-parameterized DDIM reverse step ``x_t -> x_{prev}``.

    Args:
        x_t: Current noisy latents.
        x0_hat: Predicted clean latents.
        acp_t: alpha_cumprod at the current timestep (scalar tensor).
        acp_prev: alpha_cumprod at the next (cleaner) timestep (scalar tensor).
        eta: DDIM stochasticity (0 = deterministic).
    """
    # Final step (acp_t ~ 1): x_t is already ~clean and eps is undefined
    # (sqrt(1-acp)->0). Returning x0_hat avoids a (x_t-x0)/~0 division that can
    # overflow to Inf in fp16 and then poison the result via 0*Inf = NaN.
    if float(acp_t) >= 1.0 - 1e-6:
        return x0_hat
    sqrt_acp_t = _coef(acp_t.sqrt(), x_t)
    sqrt_om_t = _coef((1.0 - acp_t).clamp(min=1e-8).sqrt(), x_t)
    eps_hat = (x_t - sqrt_acp_t * x0_hat) / sqrt_om_t

    ratio = ((1.0 - acp_prev) / (1.0 - acp_t).clamp(min=1e-8)) * (
        1.0 - acp_t / acp_prev.clamp(min=1e-8)
    )
    sigma = float(eta) * ratio.clamp(min=0.0).sqrt()
    sigma = _coef(sigma, x_t)
    # Direction term: sqrt(1 - acp_prev - sigma^2).
    dir_coef = (_coef(1.0 - acp_prev, x_t) - sigma.pow(2)).clamp(min=0.0).sqrt()

    x_prev = _coef(acp_prev.sqrt(), x_t) * x0_hat + dir_coef * eps_hat
    if eta > 0:
        x_prev = x_prev + sigma * torch.randn_like(x_t)
    return x_prev


def _make_timesteps(total_steps: int, num_steps: int, device: torch.device) -> torch.Tensor:
    """Descending integer timestep schedule of length ``num_steps`` in ``[0, total-1]``."""
    num_steps = min(num_steps, total_steps)
    ts = torch.linspace(total_steps - 1, 0, num_steps, device=device).round().long()
    return ts


@torch.no_grad()
def sample_from_model(
    model: torch.nn.Module,
    prompt_ids: Optional[torch.Tensor],
    seq_len: int,
    num_steps: Optional[int] = None,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    guidance_scale: float = 1.0,
    eta: float = 0.0,
    clamp_to_tokens: bool = False,
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> torch.Tensor:
    """Generate ``seq_len`` response tokens, optionally conditioned on ``prompt_ids``.

    Args:
        model: A :class:`~dimba.models.diffusion.DIMBA` instance.
        prompt_ids: Prompt token IDs ``[B, P]`` (or None for unconditional).
        seq_len: Number of response tokens to generate.
        num_steps: Diffusion steps (defaults to the model's training T).
        temperature, top_k, top_p: token-sampling controls.
        guidance_scale: Classifier-free guidance weight (1.0 disables CFG).
        eta: DDIM stochasticity (0 = deterministic DDIM).
        clamp_to_tokens: Snap the predicted embedding to the nearest token
            embedding each step (the Diffusion-LM clamping trick; embedding-space only).
        device: Override device.
        verbose: Log progress.

    Returns:
        Generated token IDs ``[B, seq_len]``.
    """
    if device is None:
        device = next(model.parameters()).device
    if num_steps is None:
        num_steps = model.num_diffusion_steps
    model.eval()

    d_latent = model.d_latent
    use_cfg = abs(guidance_scale - 1.0) > 1e-6 and prompt_ids is not None

    # Prompt prefix (kept clean), conditioning, and the response noise.
    if prompt_ids is not None:
        prompt_ids = prompt_ids.to(device)
        batch_size = prompt_ids.shape[0]
        prompt_latent = model.encode_latent(model.token_embed(prompt_ids))  # [B, P, d_latent]
        prompt_len = prompt_latent.shape[1]
    else:
        batch_size = 1
        prompt_latent = None
        prompt_len = 0

    cond = model.conditioning_from_prompt(prompt_ids, batch_size, device)
    uncond = (
        model.conditioning_from_prompt(None, batch_size, device, drop_cond=True)
        if use_cfg
        else None
    )

    response = torch.randn(batch_size, seq_len, d_latent, device=device)
    if prompt_latent is not None:
        x_t = torch.cat([prompt_latent, response], dim=1)
    else:
        x_t = response

    alphas_cumprod = model.get_alphas_cumprod().to(device)
    timesteps = _make_timesteps(model.num_diffusion_steps, num_steps, device)

    x_self_cond = None
    for i in range(len(timesteps)):
        t_val = timesteps[i]
        t = torch.full((batch_size,), int(t_val.item()), dtype=torch.long, device=device)

        x0_hat = model.denoise_to_x0_latent(x_t, t, cond, x_self_cond)
        if use_cfg:
            x0_uncond = model.denoise_to_x0_latent(x_t, t, uncond, x_self_cond)
            x0_hat = x0_uncond + guidance_scale * (x0_hat - x0_uncond)
        x_self_cond = x0_hat

        if clamp_to_tokens:
            x0_hat = _clamp_latent_to_tokens(model, x0_hat)

        acp_t = alphas_cumprod[t_val]
        acp_prev = alphas_cumprod[timesteps[i + 1]] if i < len(timesteps) - 1 else torch.ones((), device=device)
        x_prev = _ddim_step(x_t, x0_hat, acp_t, acp_prev, eta)

        # Hold the prompt prefix clean.
        if prompt_latent is not None:
            x_prev[:, :prompt_len, :] = prompt_latent
        x_t = x_prev

        if verbose and (i % max(1, len(timesteps) // 10) == 0):
            logger.info("denoising step %d/%d (t=%d)", i + 1, len(timesteps), int(t_val.item()))

    # Decode the response region to logits and sample.
    response_latent = x_t[:, prompt_len:, :]
    x_dec = model.decode_latent(response_latent)
    logits = model.output_head(x_dec) / max(temperature, 1e-6)

    if top_k is not None or top_p is not None:
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

    probs = F.softmax(logits, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0)
    prob_sum = probs.sum(dim=-1, keepdim=True)
    probs = torch.where(prob_sum > 1e-6, probs / prob_sum, torch.ones_like(probs) / probs.shape[-1])
    generated = torch.multinomial(probs.view(-1, probs.shape[-1]), num_samples=1)
    return generated.view(batch_size, seq_len)


def _clamp_latent_to_tokens(model: torch.nn.Module, z0_hat: torch.Tensor) -> torch.Tensor:
    """Snap predicted latents to the nearest real token (the Diffusion-LM clamping trick).

    Decodes to embedding space, finds the nearest token embedding, and re-encodes.
    Only worthwhile for embedding-space diffusion; for a deep latent it adds cost.
    """
    emb = model.decode_latent(z0_hat)  # [B, L, d_model]
    table = model.token_embed.get_weight()  # [V, d_model]
    # Nearest neighbor by squared distance.
    dists = torch.cdist(emb, table.unsqueeze(0).expand(emb.shape[0], -1, -1))
    ids = dists.argmin(dim=-1)  # [B, L]
    snapped = model.token_embed(ids)
    return model.encode_latent(snapped)


def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: Optional[int] = 0,
    top_p: Optional[float] = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> torch.Tensor:
    """Top-k and/or top-p (nucleus) filtering on ``[..., vocab]`` logits."""
    if top_k is not None and top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, filter_value)

    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumsum_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        sorted_indices_to_remove = cumsum_probs > top_p
        # Keep the token that *crosses* top_p (standard nucleus sampling, Holtzman
        # et al. 2019): shift the removal mask right by one so the crossing token
        # stays in the nucleus. Without this the nucleus is one token too small.
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
        indices_to_remove.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, filter_value)

    return logits


def sample_timesteps(batch_size: int, num_steps: int, device: torch.device) -> torch.Tensor:
    """Sample uniform random timesteps ``[B]`` for training."""
    return torch.randint(0, num_steps, (batch_size,), device=device)


class DDIMSampler:
    """Thin OO wrapper around :func:`sample_from_model` for DDIM-style sampling."""

    def __init__(self, model: torch.nn.Module, num_steps: int = 50, ddim_eta: float = 0.0):
        self.model = model
        self.num_steps = num_steps
        self.ddim_eta = ddim_eta
        self.device = next(model.parameters()).device

    def sample(
        self,
        prompt_ids: Optional[torch.Tensor],
        seq_len: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """Generate via DDIM (see :func:`sample_from_model`)."""
        return sample_from_model(
            self.model,
            prompt_ids,
            seq_len,
            num_steps=self.num_steps,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            guidance_scale=guidance_scale,
            eta=self.ddim_eta,
            device=self.device,
        )
