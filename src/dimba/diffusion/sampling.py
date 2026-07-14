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
from typing import Callable, Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _coef(value: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Reshape a scalar schedule coefficient for broadcasting against ``like``."""
    return value.view(*([1] * like.dim())).to(like.device, like.dtype)


def _batched_cfg_denoise(
    denoise_fn: Callable[..., torch.Tensor],
    x_t: torch.Tensor,
    t: torch.Tensor,
    cond: torch.Tensor,
    cfg_cond: Optional[torch.Tensor],
    x_self_cond: Optional[torch.Tensor],
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Run conditional and unconditional predictions in one model dispatch."""
    if cfg_cond is None:
        return denoise_fn(x_t, t, cond, x_self_cond), None

    batch_size = x_t.shape[0]
    x_in = torch.cat((x_t, x_t), dim=0)
    t_in = torch.cat((t, t), dim=0)
    self_cond_in = (
        torch.cat((x_self_cond, x_self_cond), dim=0) if x_self_cond is not None else None
    )
    predictions = denoise_fn(x_in, t_in, cfg_cond, self_cond_in)
    return predictions[:batch_size], predictions[batch_size:]


def _dpmpp_step(
    x_t: torch.Tensor,
    x0_hat: torch.Tensor,
    acp_t: torch.Tensor,
    acp_prev: torch.Tensor,
    prev_x0: Optional[torch.Tensor],
    prev_h: Optional[torch.Tensor],
    is_final: bool = False,
) -> tuple:
    """One x0-parameterized DPM-Solver++(2M) reverse step ``x_t -> x_{prev}``.

    Uses the data-prediction (x0) form of DPM-Solver++ with multistep correction.

    Args:
        x_t: Current noisy latents ``[B, L, d]``.
        x0_hat: Predicted clean latents (already CFG-combined/clamped).
        acp_t: alpha_cumprod at the current (noisier) timestep.
        acp_prev: alpha_cumprod at the next (cleaner) timestep.
        prev_x0: x0 prediction from the previous step (None on first step).
        prev_h: log-SNR step size from the previous step (None on first step).

    Returns:
        ``(x_prev, x0_hat, h)`` — the denoised sample, the current x0 prediction
        (stored as ``prev_x0`` for the next call), and the current h (stored as
        ``prev_h`` for the next call).
    """
    # The caller knows the final loop iteration without reading a CUDA scalar back.
    if is_final:
        return x0_hat, x0_hat, None

    alpha_t = acp_t.sqrt()
    sigma_t = (1.0 - acp_t).clamp(min=1e-8).sqrt()
    alpha_prev = acp_prev.sqrt()
    sigma_prev = (1.0 - acp_prev).clamp(min=1e-8).sqrt()

    # log-SNR: lambda = log(alpha) - log(sigma)
    lam_t = torch.log(alpha_t) - torch.log(sigma_t)
    lam_prev = torch.log(alpha_prev) - torch.log(sigma_prev)
    h = lam_prev - lam_t  # > 0 because acp_prev > acp_t

    # DPM-Solver++(2M): multistep if we have a prior estimate, else 1st-order.
    if prev_x0 is None or prev_h is None:
        # First-order (same as DPM-Solver++ 1S / DDIM-like).
        D = x0_hat
    else:
        r = prev_h / h
        # Second-order correction blending current and previous x0 estimates.
        D = (1.0 + 1.0 / (2.0 * r)) * x0_hat - (1.0 / (2.0 * r)) * prev_x0

    coef_xt = _coef(sigma_prev / sigma_t, x_t)
    coef_D = _coef(alpha_prev * -torch.expm1(-h), x_t)
    # DPM-Solver++(2M) data-prediction: x_prev = (sigma_prev/sigma_t) x_t
    #   - alpha_prev (e^-h - 1) D  ==  (sigma_prev/sigma_t) x_t + alpha_prev (1 - e^-h) D.
    # The x0 term is ADDED (the first-order case must reduce to DDIM(eta=0)).
    x_prev = coef_xt * x_t + coef_D * D

    return x_prev, x0_hat, h


def _ddim_step(
    x_t: torch.Tensor,
    x0_hat: torch.Tensor,
    acp_t: torch.Tensor,
    acp_prev: torch.Tensor,
    eta: float,
    is_final: bool = False,
) -> torch.Tensor:
    """One x0-parameterized DDIM reverse step ``x_t -> x_{prev}``.

    Args:
        x_t: Current noisy latents.
        x0_hat: Predicted clean latents.
        acp_t: alpha_cumprod at the current timestep (scalar tensor).
        acp_prev: alpha_cumprod at the next (cleaner) timestep (scalar tensor).
        eta: DDIM stochasticity (0 = deterministic).
    """
    # The caller knows the final loop iteration without reading a CUDA scalar back.
    if is_final:
        return x0_hat

    # Keep direct calls safe when acp_t is already clean without synchronizing it
    # to Python. The safe denominator prevents an unselected fp16 Inf branch.
    is_clean = acp_t >= 1.0 - 1e-6
    sqrt_acp_t = _coef(acp_t.sqrt(), x_t)
    one_minus_t = (1.0 - acp_t).clamp(min=1e-8)
    sqrt_om_t = _coef(torch.where(is_clean, torch.ones_like(one_minus_t), one_minus_t).sqrt(), x_t)
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
    return torch.where(is_clean, x0_hat, x_prev)


def _make_timesteps(total_steps: int, num_steps: int, device: torch.device) -> torch.Tensor:
    """Descending integer timestep schedule of length ``num_steps`` in ``[0, total-1]``."""
    num_steps = min(num_steps, total_steps)
    ts = torch.linspace(total_steps - 1, 0, num_steps, device=device).round().long()
    return ts


@torch.inference_mode()
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
    clamp_mode: str = "none",
    clamp_from: float = 0.0,
    cfg_mode: str = "x0",
    device: Optional[torch.device] = None,
    verbose: bool = False,
    sampler: str = "ddim",
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
        clamp_to_tokens: Legacy flag — hard-snap every step
            (equivalent to clamp_mode="hard", clamp_from=1.0).
        clamp_mode: "none" | "hard" (nearest token, Diffusion-LM) | "soft" (expected
            token embedding under the head's softmax, DiffuSeq-v2 — less committal).
        clamp_from: apply clamping only in the final fraction of steps (0=never, 1=all);
            pulling the latent toward the token manifold late fights non-word drift.
        cfg_mode: "x0" (interpolate x0 predictions) | "eps" (interpolate in noise space —
            more uniform effective guidance across the trajectory).
        device: Override device.
        verbose: Log progress.
        sampler: ``"ddim"`` (default, unchanged behaviour) or ``"dpmpp"``
            (DPM-Solver++(2M) — data-prediction form, 10-20 steps sufficient).

    Returns:
        Generated token IDs ``[B, seq_len]``.
    """
    if sampler not in ("ddim", "dpmpp"):
        raise ValueError(f"sampler must be 'ddim' or 'dpmpp', got {sampler!r}")
    if device is None:
        device = next(model.parameters()).device
    if num_steps is None:
        num_steps = model.num_diffusion_steps
    model.eval()

    d_latent = model.d_latent
    uncond_only = abs(guidance_scale) <= 1e-6 and prompt_ids is not None
    use_cfg = (
        not uncond_only
        and abs(guidance_scale - 1.0) > 1e-6
        and prompt_ids is not None
    )

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
    uncond = None
    if uncond_only:
        cond = model.conditioning_from_prompt(None, batch_size, device, drop_cond=True)
    elif use_cfg:
        uncond = model.conditioning_from_prompt(None, batch_size, device, drop_cond=True)
    cfg_cond = torch.cat((cond, uncond), dim=0) if uncond is not None else None

    response = torch.randn(batch_size, seq_len, d_latent, device=device)
    if prompt_latent is not None:
        x_t = torch.cat([prompt_latent, response], dim=1)
    else:
        x_t = response

    alphas_cumprod = model.get_alphas_cumprod().to(device)
    timesteps = _make_timesteps(model.num_diffusion_steps, num_steps, device)
    acp = alphas_cumprod.index_select(0, timesteps)
    acp_prev = torch.cat((acp[1:], alphas_cumprod.new_ones(1)))
    t_batches = timesteps[:, None].expand(-1, batch_size)

    x_self_cond = None
    n_steps = timesteps.shape[0]
    if clamp_to_tokens and clamp_mode == "none":  # backward-compat: hard-clamp every step
        clamp_mode, clamp_from = "hard", 1.0

    # DPM-Solver++(2M) carry state.
    dpmpp_prev_x0: Optional[torch.Tensor] = None
    dpmpp_prev_h: Optional[torch.Tensor] = None

    for i in range(n_steps):
        t = t_batches[i]
        acp_t = acp[i]

        x0_hat, x0_uncond = _batched_cfg_denoise(
            model.denoise_to_x0_latent, x_t, t, cond, cfg_cond, x_self_cond
        )
        if use_cfg:
            assert x0_uncond is not None
            if cfg_mode == "eps":
                # Guide in noise space: convert both x0 estimates to eps, blend, convert
                # back. Keeps the effective guidance scale more uniform across timesteps.
                sa = acp_t.sqrt()
                so = (1.0 - acp_t).clamp(min=1e-8).sqrt()
                eps_c = (x_t - sa * x0_hat) / so
                eps_u = (x_t - sa * x0_uncond) / so
                eps_g = eps_u + guidance_scale * (eps_c - eps_u)
                x0_hat = (x_t - so * eps_g) / sa.clamp(min=1e-8)
            else:
                x0_hat = x0_uncond + guidance_scale * (x0_hat - x0_uncond)
        x_self_cond = x0_hat

        # Pull the prediction toward the token manifold in the final `clamp_from`
        # fraction of steps (soft = expected embedding, hard = nearest token).
        if clamp_mode != "none" and clamp_from > 0 and (i + 1) / n_steps >= (1.0 - clamp_from):
            if clamp_mode == "soft":
                x0_hat = _soft_clamp_latent_to_tokens(model, x0_hat, temperature)
            else:
                x0_hat = _clamp_latent_to_tokens(model, x0_hat)

        if sampler == "dpmpp":
            x_prev, dpmpp_prev_x0, dpmpp_prev_h = _dpmpp_step(
                x_t,
                x0_hat,
                acp_t,
                acp_prev[i],
                dpmpp_prev_x0,
                dpmpp_prev_h,
                is_final=i == n_steps - 1,
            )
        else:
            x_prev = _ddim_step(
                x_t, x0_hat, acp_t, acp_prev[i], eta, is_final=i == n_steps - 1
            )

        # Hold the prompt prefix clean.
        if prompt_latent is not None:
            x_prev[:, :prompt_len, :] = prompt_latent
        x_t = x_prev

        if verbose and (i % max(1, n_steps // 10) == 0):
            t_log = round((model.num_diffusion_steps - 1) * (1.0 - i / max(n_steps - 1, 1)))
            logger.info("denoising step %d/%d (t=%d)", i + 1, n_steps, t_log)

    # Decode the response region to logits and sample.
    response_latent = x_t[:, prompt_len:, :]
    x_dec = model.decode_latent(response_latent)
    logits = model.output_head(x_dec) / max(temperature, 1e-6)

    if top_k is not None or top_p is not None:
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

    probs = F.softmax(logits, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0)
    prob_sum = probs.sum(dim=-1, keepdim=True)
    probs = torch.where(
        prob_sum > 1e-6, probs / prob_sum.clamp_min(1e-6), 1.0 / probs.shape[-1]
    )
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


def _soft_clamp_latent_to_tokens(
    model: torch.nn.Module, z0_hat: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    """Soft clamping (DiffuSeq-v2): replace x0 with the *expected* token embedding under
    the head's predicted distribution, then re-encode. Less committal than hard
    nearest-neighbour, so the trajectory can still self-correct at earlier steps."""
    emb = model.decode_latent(z0_hat)             # [B, L, d_model]
    logits = model.output_head(emb)               # [B, L, V]
    probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
    table = model.token_embed.get_weight()        # [V, d_model]
    soft_emb = torch.matmul(probs, table)         # [B, L, d_model] expected embedding
    return model.encode_latent(soft_emb)


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


# ── Flow Matching ODE sampler ─────────────────────────────────────────────────

@torch.inference_mode()
def sample_from_model_flow(
    model: torch.nn.Module,
    prompt_ids: Optional[torch.Tensor],
    seq_len: int,
    num_steps: int = 20,
    sampler: str = "euler",
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    guidance_scale: float = 1.0,
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> torch.Tensor:
    """Generate ``seq_len`` tokens using flow matching ODE integration.

    Integrates ``dx/dt = v_theta(x_t, t, cond)`` from t=1 (pure noise) to t=0
    (clean data) using Euler or Heun steps.  Requires the model to have been
    trained with ``use_flow_matching=True``.

    Args:
        model: A DIMBA instance with ``use_flow_matching=True``.
        prompt_ids: Prompt token IDs ``[B, P]`` or None.
        seq_len: Number of response tokens to generate.
        num_steps: ODE integration steps (8–20 is usually sufficient).
        sampler: ``"euler"`` (1st-order, fast) or ``"heun"`` (2nd-order, better
            quality at the same step count, ~2× compute per step).
        guidance_scale: CFG weight (1.0 = off).
        temperature, top_k, top_p: Final token sampling controls.
        device: Override device.

    Returns:
        Generated token IDs ``[B, seq_len]``.
    """
    if sampler not in ("euler", "heun"):
        raise ValueError(f"sampler must be 'euler' or 'heun', got {sampler!r}")
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    d_latent = model.d_latent
    uncond_only = abs(guidance_scale) <= 1e-6 and prompt_ids is not None
    use_cfg = (
        not uncond_only
        and abs(guidance_scale - 1.0) > 1e-6
        and prompt_ids is not None
    )

    if prompt_ids is not None:
        prompt_ids = prompt_ids.to(device)
        batch_size = prompt_ids.shape[0]
        prompt_latent = model.encode_latent(model.token_embed(prompt_ids))
        prompt_len = prompt_latent.shape[1]
    else:
        batch_size = 1
        prompt_latent = None
        prompt_len = 0

    cond = model.conditioning_from_prompt(prompt_ids, batch_size, device)
    uncond = None
    if uncond_only:
        cond = model.conditioning_from_prompt(None, batch_size, device, drop_cond=True)
    elif use_cfg:
        uncond = model.conditioning_from_prompt(None, batch_size, device, drop_cond=True)
    cfg_cond = torch.cat((cond, uncond), dim=0) if uncond is not None else None

    # Start from pure noise at t=1
    response = torch.randn(batch_size, seq_len, d_latent, device=device)
    x_t = torch.cat([prompt_latent, response], dim=1) if prompt_latent is not None else response

    # Uniform time grid from t=1 → t=0
    ts = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
    t_batches = ts[:, None].expand(-1, batch_size)

    def _velocity(xt, t_batch, t_val, x_self_cond=None):
        # model's denoise_to_x0_latent returns x0 prediction; velocity = (xt - x0) / t
        x0_hat, x0_uncond = _batched_cfg_denoise(
            model.denoise_flow, xt, t_batch, cond, cfg_cond, x_self_cond
        )
        if use_cfg:
            assert x0_uncond is not None
            x0_hat = x0_uncond + guidance_scale * (x0_hat - x0_uncond)
        # v = (x_t - x0) / t  (rearranged from x_t = (1-t)*x0 + t*noise)
        v = (xt - x0_hat) / t_val.clamp_min(1e-5)
        return v, x0_hat

    x_self_cond = None

    for i in range(num_steps):
        t_cur = ts[i]
        t_nxt = ts[i + 1]
        dt = t_nxt - t_cur  # negative (integrating backward)

        v_cur, x0_hat = _velocity(x_t, t_batches[i], t_cur, x_self_cond)
        x_self_cond = x0_hat

        if sampler == "heun" and i < num_steps - 1:
            # Predictor step
            x_mid = x_t + v_cur * dt
            if prompt_latent is not None:
                x_mid[:, :prompt_len, :] = prompt_latent
            v_nxt, _ = _velocity(x_mid, t_batches[i + 1], t_nxt, x_self_cond)
            # Corrector: average the two velocities
            x_t = x_t + 0.5 * (v_cur + v_nxt) * dt
        else:
            x_t = x_t + v_cur * dt

        # Keep prompt prefix clean
        if prompt_latent is not None:
            x_t[:, :prompt_len, :] = prompt_latent

        if verbose and i % max(1, num_steps // 5) == 0:
            t_cur_log = 1.0 - i / num_steps
            t_nxt_log = 1.0 - (i + 1) / num_steps
            logger.info(
                "flow step %d/%d (t=%.3f→%.3f)",
                i + 1,
                num_steps,
                t_cur_log,
                t_nxt_log,
            )

    # Decode response region
    response_latent = x_t[:, prompt_len:, :]
    x_dec = model.decode_latent(response_latent)
    logits = model.output_head(x_dec) / max(temperature, 1e-6)

    if top_k is not None or top_p is not None:
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

    probs = F.softmax(logits, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0)
    prob_sum = probs.sum(dim=-1, keepdim=True)
    probs = torch.where(
        prob_sum > 1e-6, probs / prob_sum.clamp_min(1e-6), 1.0 / probs.shape[-1]
    )
    generated = torch.multinomial(probs.view(-1, probs.shape[-1]), num_samples=1)
    return generated.view(batch_size, seq_len)


class DDIMSampler:
    """Thin OO wrapper around :func:`sample_from_model` for DDIM-style sampling."""

    def __init__(self, model: torch.nn.Module, num_steps: int = 50, ddim_eta: float = 0.0):
        self.model = model
        self.num_steps = num_steps
        self.ddim_eta = ddim_eta
        self.device = next(model.parameters()).device

    @torch.inference_mode()
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
        )
