"""Shared, memory-efficient masked-diffusion training primitives."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fused_ce import output_head_cross_entropy


def freeze_masked_only_unused_parameters(model: nn.Module) -> Tuple[str, ...]:
    """Freeze DIMBA parameters that the masked-token forward never reads.

    Masked training gets context from the visible tokens in-sequence and calls
    ``predict_token_features``, which uses null conditioning.  The pooled prompt
    encoder belongs only to the continuous-diffusion path, so leaving it trainable
    makes DDP wait for gradients that can never exist.
    """
    prompt_encoder = getattr(model, "prompt_encoder", None)
    if prompt_encoder is None:
        return ()
    frozen = []
    for name, parameter in prompt_encoder.named_parameters():
        if parameter.requires_grad:
            parameter.requires_grad_(False)
            frozen.append(f"prompt_encoder.{name}")
    return tuple(frozen)


def sample_masked_inputs(
    input_ids: torch.Tensor,
    mask_token_id: int,
    *,
    t_min: float = 0.03,
    eligible_mask: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample LLaDA corruption and guarantee one target per eligible row."""
    batch, length = input_ids.shape
    if eligible_mask is None:
        eligible_mask = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        eligible_mask = eligible_mask.to(device=input_ids.device, dtype=torch.bool)

    t = torch.rand(batch, device=input_ids.device, generator=generator)
    t = t * (1.0 - t_min) + t_min
    target_mask = (
        torch.rand(batch, length, device=input_ids.device, generator=generator) < t[:, None]
    ) & eligible_mask

    # Avoid ``if none_masked.any()``: that innocent-looking branch synchronizes CUDA.
    none_masked = ~target_mask.any(dim=1)
    first_eligible = eligible_mask.to(torch.int64).argmax(dim=1, keepdim=True)
    fallback = torch.zeros_like(target_mask).scatter(1, first_eligible, none_masked[:, None])
    target_mask |= fallback & eligible_mask
    corrupted = input_ids.masked_fill(target_mask, mask_token_id)
    return corrupted, t, target_mask


def masked_token_loss(
    model,
    corrupted_ids: torch.Tensor,
    target_ids: torch.Tensor,
    target_mask: torch.Tensor,
    t: torch.Tensor,
    *,
    mask_token_id: Optional[int] = None,
    inverse_t_weight: bool = True,
    neighbor_unlikelihood_weight: float = 0.0,
    fused_ce_mode: str = "auto",
    ce_chunk_tokens: Optional[int] = 2048,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute CE only where tokens are masked, without a full ``[B,L,V]`` tensor.

    ``ce_chunk_tokens`` additionally chunks the selected-token vocabulary
    projection + CE so the ``[masked_tokens, vocab]`` logits are never retained
    for backward (identical math and per-token gradients; see
    :func:`dimba.training.fused_ce.output_head_cross_entropy`). ``None``
    restores the single-shot projection."""
    if fused_ce_mode not in {"auto", "on", "off"}:
        raise ValueError("fused CE mode must be 'auto', 'on', or 'off'")
    if fused_ce_mode == "on":
        raise RuntimeError(
            "Liger fused CE cannot preserve masked per-sample 1/t gradients; "
            "use fused_ce_mode='auto' or 'off'"
        )
    target_mask = target_mask.to(dtype=torch.bool)
    batch, length = target_ids.shape

    features = model.predict_token_features(corrupted_ids, t)
    selected_features = features[target_mask]
    selected_targets = target_ids[target_mask]
    standard_head = hasattr(model, "output_head") and hasattr(model, "token_embed")
    logits = None
    if neighbor_unlikelihood_weight > 0.0 or not standard_head:
        # Unlikelihood needs selected logits; avoid projecting them a second time.
        logits = model.project_token_features(selected_features)
        ce = F.cross_entropy(logits, selected_targets, reduction="none")
    else:
        ce = output_head_cross_entropy(
            model,
            selected_features,
            selected_targets,
            prepared=True,
            reduction="none",
            mode=fused_ce_mode,
            uniform_reduction=False,
            chunk_tokens=ce_chunk_tokens,
        )

    sample_ids = torch.arange(batch, device=target_ids.device)[:, None].expand(batch, length)
    selected_samples = sample_ids[target_mask]
    # Match the historical full-logit objective: reductions accumulate in fp32
    # even when the model/logits run in bf16. Hundreds of bf16 scatter additions
    # otherwise introduce measurable per-sample loss drift on long sequences.
    ce_fp32 = ce.float()
    counts = target_mask.sum(dim=1).clamp(min=1).float()
    ce_sums = torch.zeros(batch, device=ce.device, dtype=torch.float32)
    ce_sums.scatter_add_(0, selected_samples, ce_fp32)
    ce_per_sample = ce_sums / counts
    loss = (ce_per_sample / t.clamp(min=1e-6)).mean() if inverse_t_weight else ce_per_sample.mean()

    parts: Dict[str, torch.Tensor] = {
        "ce_masked": ce_per_sample.detach().mean(),
        "t_mean": t.detach().mean(),
    }

    if neighbor_unlikelihood_weight > 0.0:
        if mask_token_id is None:
            raise ValueError("mask_token_id is required for neighbor unlikelihood")
        assert logits is not None
        log_denom = torch.logsumexp(logits.float(), dim=-1)
        ul_sums = torch.zeros(batch, device=ce.device, dtype=torch.float32)
        for shift in (-1, 1):
            neighbor = torch.roll(corrupted_ids, shifts=shift, dims=1)
            valid = target_mask & (neighbor != mask_token_id) & (neighbor != target_ids)
            valid[:, -1 if shift == -1 else 0] = False
            selected_neighbor = neighbor[target_mask]
            selected_valid = valid[target_mask]
            log_p = logits.float().gather(1, selected_neighbor[:, None]).squeeze(1) - log_denom
            ul = -torch.log1p(-log_p.exp().clamp(max=1.0 - 1e-4))
            ul_sums.scatter_add_(0, selected_samples, ul * selected_valid)
        ul_per_sample = ul_sums / counts.float()
        ul_loss = ul_per_sample.mean()
        loss = loss + neighbor_unlikelihood_weight * ul_loss
        parts["ul"] = ul_loss.detach()

    return loss, parts


def compute_masked_diffusion_loss(
    model,
    input_ids: torch.Tensor,
    mask_token_id: int,
    *,
    t_min: float = 0.03,
    eligible_mask: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
    fused_ce_mode: str = "auto",
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Sample masked corruption and compute the standard ``1/t`` objective."""
    corrupted, t, target_mask = sample_masked_inputs(
        input_ids,
        mask_token_id,
        t_min=t_min,
        eligible_mask=eligible_mask,
        generator=generator,
    )
    return masked_token_loss(
        model,
        corrupted,
        input_ids,
        target_mask,
        t,
        mask_token_id=mask_token_id,
        fused_ce_mode=fused_ce_mode,
    )


class MaskedDiffusionObjective(nn.Module):
    """One DDP-safe forward covering corruption, backbone, projection, and loss."""

    def __init__(
        self,
        model: nn.Module,
        mask_token_id: int,
        *,
        t_min: float = 0.03,
        prompt_dropout: float = 0.0,
        neighbor_unlikelihood_weight: float = 0.0,
        fused_ce_mode: str = "auto",
    ) -> None:
        super().__init__()
        self.model = model
        self.mask_token_id = mask_token_id
        self.t_min = t_min
        self.prompt_dropout = prompt_dropout
        self.neighbor_unlikelihood_weight = neighbor_unlikelihood_weight
        self.fused_ce_mode = fused_ce_mode

    def forward(
        self,
        input_ids: torch.Tensor,
        prompt_lengths: Optional[torch.Tensor] = None,
        response_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if prompt_lengths is None and response_lengths is None:
            return compute_masked_diffusion_loss(
                self.model,
                input_ids,
                self.mask_token_id,
                t_min=self.t_min,
                fused_ce_mode=self.fused_ce_mode,
            )
        if prompt_lengths is None or response_lengths is None:
            raise ValueError("prompt_lengths and response_lengths must be passed together")

        batch, length = input_ids.shape
        positions = torch.arange(length, device=input_ids.device)[None, :]
        response = (positions >= prompt_lengths[:, None]) & (
            positions < response_lengths[:, None]
        )
        corrupted, t, target_mask = sample_masked_inputs(
            input_ids,
            self.mask_token_id,
            t_min=self.t_min,
            eligible_mask=response,
        )
        if self.prompt_dropout > 0.0:
            drop = torch.rand(batch, device=input_ids.device) < self.prompt_dropout
            corrupted = torch.where(
                drop[:, None] & (positions < prompt_lengths[:, None]),
                self.mask_token_id,
                corrupted,
            )
        return masked_token_loss(
            self.model,
            corrupted,
            input_ids,
            target_mask,
            t,
            mask_token_id=self.mask_token_id,
            neighbor_unlikelihood_weight=self.neighbor_unlikelihood_weight,
            fused_ce_mode=self.fused_ce_mode,
        )
