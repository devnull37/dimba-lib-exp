"""Production backend checks shared by CUDA training launchers."""

from __future__ import annotations

import importlib
from typing import Tuple, Union

import torch
import torch.nn as nn


def configure_cuda_training(device: Union[str, torch.device]) -> None:
    """Enable CUDA math fast paths without changing CPU or MPS behavior."""
    if torch.device(device).type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def require_fast_cuda_mamba2(
    model: nn.Module,
    device: Union[str, torch.device],
) -> Tuple[str, ...]:
    """Fail a CUDA run unless every live mixer is the fused Mamba-2 backend."""
    if torch.device(device).type != "cuda":
        return ()

    base = getattr(model, "module", model)
    base = getattr(base, "_orig_mod", base)
    implementations = tuple(
        sorted(
            {
                f"{type(mixer).__module__}.{type(mixer).__name__}"
                for block in base.denoiser.blocks
                for mixer in (block.mamba_fwd, block.mamba_bwd)
                if mixer is not None
            }
        )
    )
    if not implementations or any(
        not name.startswith("mamba_ssm.") or not name.endswith(".Mamba2")
        for name in implementations
    ):
        raise RuntimeError(
            "CUDA training requires live mamba_ssm.Mamba2 mixers; "
            f"resolved {implementations or ('none',)}. Refusing the slow fallback."
        )

    try:
        importlib.import_module("causal_conv1d")
    except Exception as exc:
        raise RuntimeError(
            "CUDA training requires a working causal-conv1d build; "
            "install the pinned CUDA training extra."
        ) from exc
    return implementations


__all__ = ["configure_cuda_training", "require_fast_cuda_mamba2"]
