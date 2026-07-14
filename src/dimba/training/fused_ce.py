"""Optional memory-efficient output projection plus cross-entropy."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


FusedCEMode = Literal["auto", "on", "off"]


@lru_cache(maxsize=1)
def _load_liger_loss():
    try:
        from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    except (ImportError, OSError):
        return None
    return LigerFusedLinearCrossEntropyLoss


@lru_cache(maxsize=2)
def _liger_loss(reduction: str):
    loss_type = _load_liger_loss()
    return None if loss_type is None else loss_type(reduction=reduction)


def _liger_runtime_ready(features: torch.Tensor) -> bool:
    return features.device.type == "cuda"


def _validate_mode(mode: str) -> FusedCEMode:
    if mode not in {"auto", "on", "off"}:
        raise ValueError("fused CE mode must be 'auto', 'on', or 'off'")
    return mode  # type: ignore[return-value]


def _projection_parameters(model):
    head = model.output_head
    embedding_weight = model.token_embed.get_weight()
    if head.use_weight_tying:
        weight, bias = embedding_weight, None
    elif isinstance(head.projection, nn.Linear):
        weight, bias = head.projection.weight, head.projection.bias
    else:
        weight, bias = None, None
    scale = head.logit_scale.exp() if head.use_norm else None
    return head, embedding_weight, weight, bias, scale


def output_head_cross_entropy(
    model,
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    prepared: bool = False,
    active_mask: Optional[torch.Tensor] = None,
    reduction: str = "none",
    mode: FusedCEMode = "auto",
    uniform_reduction: bool = False,
) -> torch.Tensor:
    """Compute output-head CE, selecting active tokens before native projection.

    Liger is used only for a uniform ``mean`` or ``sum`` reduction. Upstream Liger
    0.8 exposes ``reduction='none'`` in its forward pass, but still pre-aggregates
    weight/bias gradients and explicitly does not support a correctly weighted
    unreduced backward. It also currently synchronizes once via ``.item()``. Until
    upstream adds tested unreduced gradients, masked, response-weighted, and
    time-faded losses stay on this exact native selected-token path.
    """
    mode = _validate_mode(mode)
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")

    head, embedding_weight, weight, bias, scale = _projection_parameters(model)
    if not prepared:
        features = head.prepare_features(features, embedding_weight)
    if features.shape[:-1] != targets.shape:
        raise ValueError("features and targets must have matching token dimensions")
    if active_mask is not None:
        if active_mask.shape != targets.shape:
            raise ValueError("active_mask must match targets")
        active_mask = active_mask.to(device=targets.device, dtype=torch.bool)
        features = features[active_mask]
        targets = targets[active_mask]
    else:
        features = features.reshape(-1, features.shape[-1])
        targets = targets.reshape(-1)

    liger_safe = uniform_reduction and reduction in {"mean", "sum"}
    if mode == "on" and not liger_safe:
        raise RuntimeError(
            "Liger fused CE cannot preserve non-uniform per-token/sample gradients; "
            "use fused_ce_mode='auto' or 'off' until upstream supports unreduced backward"
        )
    use_liger = (
        mode != "off"
        and liger_safe
        and weight is not None
        and _liger_runtime_ready(features)
        and _liger_loss(reduction) is not None
    )
    if mode == "on" and not use_liger:
        raise RuntimeError(
            "fused_ce_mode='on' requires a tied or nn.Linear output head, CUDA, and "
            "liger-kernel; install the GPU extra or use mode='auto'"
        )

    if use_liger:
        # Preserve head temperature algebraically without materializing [tokens, vocab].
        if scale is not None:
            features = features * scale
            bias = None if bias is None else bias * scale
        return _liger_loss(reduction)(weight, features, targets, bias)

    if weight is None:
        # LoRA and other linear-compatible wrappers are not accepted by Liger's
        # explicit (weight, bias) API. Keep them on the exact native module path.
        logits = head.project_features(features, embedding_weight)
        return F.cross_entropy(logits, targets, reduction=reduction)

    logits = F.linear(features, weight, bias)
    if scale is not None:
        logits = logits * scale
    return F.cross_entropy(logits, targets, reduction=reduction)


def liger_fused_ce_available() -> bool:
    """Return whether the optional Liger class imports successfully."""
    return _load_liger_loss() is not None


__all__ = ["FusedCEMode", "liger_fused_ce_available", "output_head_cross_entropy"]
