"""Experimental MLX backend for DIMBA (Apple Silicon).

This subpackage contains an MLX port skeleton of the Mamba-2 denoiser block and
a helper to convert PyTorch state dicts into MLX arrays. It is *experimental*
and intended for Apple-Silicon (M-series) machines where MLX can use the
unified-memory GPU.

The module imports cleanly even when ``mlx`` is not installed: in that case the
exported classes are stubs that raise a clear ``RuntimeError`` on instantiation.
Check :data:`HAS_MLX` to know whether a usable MLX runtime is present.
"""

from __future__ import annotations

from .denoiser import (
    HAS_MLX,
    MLXMamba2Block,
    MLXMamba2Denoiser,
    mlx_selective_scan_sequential,
    torch_state_dict_to_mlx,
)

__all__ = [
    "HAS_MLX",
    "MLXMamba2Block",
    "MLXMamba2Denoiser",
    "mlx_selective_scan_sequential",
    "torch_state_dict_to_mlx",
]
