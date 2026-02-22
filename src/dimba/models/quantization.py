"""Quantization helpers for Q-LoRA style fine-tuning."""

from __future__ import annotations

from typing import Iterator, Tuple

import torch
import torch.nn as nn


_NORM_TYPES: Tuple[type[nn.Module], ...] = (
    nn.LayerNorm,
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.GroupNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
)


def _require_bitsandbytes():
    """Import bitsandbytes or raise a clear runtime error."""
    try:
        import bitsandbytes as bnb  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "bitsandbytes is required for 4-bit quantization. "
            "Install it with `pip install bitsandbytes`."
        ) from exc
    return bnb


def _iter_linear_modules(
    module: nn.Module,
) -> Iterator[tuple[nn.Module, str, nn.Linear]]:
    """Yield (parent_module, child_name, linear_child) for all linear layers."""
    for child_name, child in module.named_children():
        if isinstance(child, nn.Linear):
            yield module, child_name, child
            continue
        yield from _iter_linear_modules(child)


def _is_norm_module(module: nn.Module) -> bool:
    """Return True if module behaves like a normalization layer."""
    if isinstance(module, _NORM_TYPES):
        return True
    return "norm" in module.__class__.__name__.lower()


def _choose_compute_dtype() -> torch.dtype:
    """Choose a stable compute dtype for 4-bit operations."""
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def quantize_model_4bit(model: nn.Module) -> nn.Module:
    """Replace eligible linear layers with bitsandbytes NF4 4-bit modules in-place."""
    bnb = _require_bitsandbytes()
    bnb_nn = getattr(bnb, "nn", None)
    if bnb_nn is None:
        raise RuntimeError("bitsandbytes.nn is unavailable; cannot quantize model to 4-bit.")

    linear4bit_cls = getattr(bnb_nn, "Linear4bit", None)
    params4bit_cls = getattr(bnb_nn, "Params4bit", None)
    if linear4bit_cls is None or params4bit_cls is None:
        raise RuntimeError(
            "Installed bitsandbytes does not expose Linear4bit/Params4bit required for NF4 quantization."
        )

    compute_dtype = _choose_compute_dtype()
    linear_modules = list(_iter_linear_modules(model))

    for parent, child_name, linear in linear_modules:
        if linear.__class__.__name__.lower() == "linear4bit":
            continue

        if not torch.is_floating_point(linear.weight):
            continue

        device = linear.weight.device
        quantized = linear4bit_cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            compute_dtype=compute_dtype,
            compress_statistics=True,
            quant_type="nf4",
        ).to(device=device)

        with torch.no_grad():
            quantized.weight = params4bit_cls(
                linear.weight.detach().to(device=device).contiguous(),
                requires_grad=False,
                compress_statistics=True,
                quant_type="nf4",
            )
            if linear.bias is not None:
                quantized.bias = nn.Parameter(
                    linear.bias.detach().to(device=device).clone(),
                    requires_grad=False,
                )

        quantized.train(linear.training)
        setattr(parent, child_name, quantized)

    return model


def prepare_for_qlora(model: nn.Module) -> nn.Module:
    """Freeze base parameters and cast normalization layers to fp32 for LoRA stability."""
    for param in model.parameters():
        param.requires_grad = False

    for module in model.modules():
        if _is_norm_module(module):
            module.to(dtype=torch.float32)

    # If LoRA layers are already injected, keep them trainable.
    for name, param in model.named_parameters():
        lowered = name.lower()
        if "lora_" in lowered or ".lora" in lowered:
            param.requires_grad = True

    if hasattr(model, "enable_input_require_grads"):
        try:
            model.enable_input_require_grads()  # type: ignore[attr-defined]
        except Exception:
            pass

    return model


__all__ = ["quantize_model_4bit", "prepare_for_qlora"]
