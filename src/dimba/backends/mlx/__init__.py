"""MLX backend for DIMBA — Apple-Silicon GPU (Metal).

Runs DIMBA on the Apple GPU via MLX, in two layers:

* :class:`MLXMamba2Mixer` — the Mamba-2 (SSD) mixer, **weight-compatible** with
  ``mamba_ssm.Mamba2`` / :class:`dimba.models.torch_mamba2.TorchMamba2`.
* :class:`MLXDIMBA` — the **full** DIMBA diffusion sampler (token embedding, latent
  projector, FiLM conditioning, bidirectional Mamba-2 blocks, timestep embedding,
  schedule, x0-DDIM) on the GPU. ``MLXDIMBA.from_torch(model)`` copies weights from a
  PyTorch ``DIMBA`` and produces token-identical output ~17x faster than torch-MPS.

The module imports cleanly even when ``mlx`` is not installed: the exported classes are
stubs that raise a clear ``RuntimeError`` on use. Check :data:`HAS_MLX` for a usable
runtime. See ``docs/BACKENDS.md`` for benchmarks and usage.
"""

from __future__ import annotations

from .mamba2 import (
    HAS_MLX,
    MLXMamba2Mixer,
    scan_chunked_mlx,
    load_torch_mamba2_state_dict,
)
from .model import MLXDIMBA

__all__ = [
    "HAS_MLX",
    "MLXMamba2Mixer",
    "scan_chunked_mlx",
    "load_torch_mamba2_state_dict",
    "MLXDIMBA",
]
