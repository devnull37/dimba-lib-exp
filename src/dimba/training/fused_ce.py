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


class _ChunkedLinearCE(torch.autograd.Function):
    """Exact linear+CE without retaining the ``[tokens, vocab]`` logits.

    Forward computes per-chunk ``F.cross_entropy(F.linear(f, W, b) * scale, t,
    reduction='none')`` -- the identical ops as the unchunked native path -- and
    frees each chunk's logits immediately. Backward recomputes each chunk's
    logits and routes the incoming per-token gradient through the same ops via
    autograd, so per-token (e.g. 1/t-weighted) gradients are preserved exactly;
    weight/bias/scale gradients accumulate across chunks in fp32.

    Peak activation memory drops from ``[N, vocab]`` (saved for backward) to
    ``[chunk, vocab]`` (transient), at the cost of recomputing the projection
    GEMM once in backward.
    """

    @staticmethod
    def forward(ctx, features, weight, bias, scale, targets, chunk):
        ce_chunks = []
        for start in range(0, features.shape[0], chunk):
            logits = F.linear(features[start : start + chunk], weight, bias)
            if scale is not None:
                logits = logits * scale
            ce_chunks.append(
                F.cross_entropy(
                    logits, targets[start : start + chunk], reduction="none"
                )
            )
        ctx.save_for_backward(features, weight, bias, scale, targets)
        ctx.chunk = chunk
        return torch.cat(ce_chunks, dim=0)

    @staticmethod
    def backward(ctx, grad_ce):
        features, weight, bias, scale, targets = ctx.saved_tensors
        needs_f, needs_w, needs_b, needs_s = ctx.needs_input_grad[:4]
        grad_f = torch.empty_like(features) if needs_f else None
        grad_w = (
            torch.zeros_like(weight, dtype=torch.float32) if needs_w else None
        )
        grad_b = (
            torch.zeros_like(bias, dtype=torch.float32)
            if (needs_b and bias is not None)
            else None
        )
        grad_s = (
            torch.zeros_like(scale, dtype=torch.float32)
            if (needs_s and scale is not None)
            else None
        )
        for start in range(0, features.shape[0], ctx.chunk):
            end = start + ctx.chunk
            inputs, live = [], []
            f = features[start:end].detach().requires_grad_(needs_f)
            w = weight.detach().requires_grad_(needs_w)
            b = None if bias is None else bias.detach().requires_grad_(needs_b)
            s = None if scale is None else scale.detach().requires_grad_(needs_s)
            with torch.enable_grad():
                logits = F.linear(f, w, b)
                if s is not None:
                    logits = logits * s
                ce = F.cross_entropy(logits, targets[start:end], reduction="none")
            for tensor, wanted in ((f, needs_f), (w, needs_w), (b, needs_b), (s, needs_s)):
                if tensor is not None and wanted:
                    inputs.append(tensor)
                    live.append(tensor)
            grads = torch.autograd.grad(ce, inputs, grad_ce[start:end])
            grads = dict(zip(live, grads))
            if needs_f:
                grad_f[start:end] = grads[f]
            if grad_w is not None:
                grad_w += grads[w].float()
            if grad_b is not None:
                grad_b += grads[b].float()
            if grad_s is not None:
                grad_s += grads[s].float()
        return (
            grad_f,
            None if grad_w is None else grad_w.to(weight.dtype),
            None if grad_b is None else grad_b.to(bias.dtype),
            None if grad_s is None else grad_s.to(scale.dtype),
            None,
            None,
        )


def _chunked_linear_ce(features, weight, bias, scale, targets, chunk):
    if not torch.is_grad_enabled():
        # Inference/eval: plain chunked loop (autograd Functions cannot save
        # inference-mode tensors, and there is no backward to feed anyway).
        out = []
        for start in range(0, features.shape[0], chunk):
            logits = F.linear(features[start : start + chunk], weight, bias)
            if scale is not None:
                logits = logits * scale
            out.append(
                F.cross_entropy(
                    logits, targets[start : start + chunk], reduction="none"
                )
            )
        return torch.cat(out, dim=0)
    return _ChunkedLinearCE.apply(features, weight, bias, scale, targets, chunk)


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
    chunk_tokens: Optional[int] = None,
) -> torch.Tensor:
    """Compute output-head CE, selecting active tokens before native projection.

    Liger is used only for a uniform ``mean`` or ``sum`` reduction. Upstream Liger
    0.8 exposes ``reduction='none'`` in its forward pass, but still pre-aggregates
    weight/bias gradients and explicitly does not support a correctly weighted
    unreduced backward. It also currently synchronizes once via ``.item()``. Until
    upstream adds tested unreduced gradients, masked, response-weighted, and
    time-faded losses stay on this exact native selected-token path.

    ``chunk_tokens`` bounds peak memory on that native unreduced path: the
    ``[tokens, vocab]`` logits are computed (and, in backward, recomputed) in
    chunks of this many tokens instead of being materialized and retained
    whole. Same ops, same per-token gradients; only the accumulation of
    weight/bias/scale gradients switches to fp32 across chunks.
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

    if (
        chunk_tokens is not None
        and reduction == "none"
        and features.shape[0] > chunk_tokens
    ):
        return _chunked_linear_ce(features, weight, bias, scale, targets, chunk_tokens)

    logits = F.linear(features, weight, bias)
    if scale is not None:
        logits = logits * scale
    return F.cross_entropy(logits, targets, reduction=reduction)


def liger_fused_ce_available() -> bool:
    """Return whether the optional Liger class imports successfully."""
    return _load_liger_loss() is not None


__all__ = ["FusedCEMode", "liger_fused_ce_available", "output_head_cross_entropy"]
