"""MLX port skeleton of the DIMBA Mamba-2 denoiser (experimental).

Status
------
**Experimental / skeleton.** This is a structural port of the PyTorch
:class:`dimba.models.denoiser.Mamba2Block` / ``Mamba2Denoiser`` to Apple's MLX
framework, intended as a starting point for running DIMBA's CPU/MPS path on
Apple-Silicon GPUs via MLX's unified memory. It implements the *correct*
diagonal selective-scan recurrence (matching
:mod:`dimba.models.parallel_scan`) but currently only as a sequential scan, and
it has **not** been numerically validated against the PyTorch model end to end.

Performance expectations
------------------------
* The sequential scan here is a reference, not an optimized kernel; expect it to
  be slower than a fused implementation. The win from MLX comes from running on
  the Apple GPU with unified memory (no host<->device copies), which mainly
  helps the dense projections, not the O(L) scan loop. A future iteration
  should replace :func:`mlx_selective_scan_sequential` with a parallel/chunked
  scan analogous to :func:`dimba.models.parallel_scan.selective_scan`.
* :func:`torch_state_dict_to_mlx` uses NumPy as the bridge and therefore copies
  every parameter once; do it at load time, not per forward pass.

Import safety
-------------
If ``mlx`` is not installed, importing this module still succeeds. The exported
classes become stubs whose constructors raise ``RuntimeError`` and
:func:`torch_state_dict_to_mlx` still works (it only needs NumPy), so weight
conversion can be staged on non-Apple machines.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:  # MLX is only available on Apple Silicon and is an optional dependency.
    import mlx.core as mx
    import mlx.nn as mlx_nn

    HAS_MLX = True
except ImportError:  # pragma: no cover - exercised only without MLX installed
    mx = None  # type: ignore[assignment]
    mlx_nn = None  # type: ignore[assignment]
    HAS_MLX = False


__all__ = [
    "HAS_MLX",
    "MLXMamba2Block",
    "MLXMamba2Denoiser",
    "mlx_selective_scan_sequential",
    "torch_state_dict_to_mlx",
]

_NO_MLX_MSG = (
    "MLX is not installed. The MLX backend requires Apple Silicon with the "
    "'mlx' package installed (pip install mlx). Use the default torch backend "
    "instead."
)


def torch_state_dict_to_mlx(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Convert a PyTorch ``state_dict`` to a dict of MLX arrays.

    Bridges via NumPy: each tensor is detached, moved to CPU, converted to a
    NumPy array and then wrapped as an ``mlx.core.array``. Parameter *names* are
    preserved unchanged (the caller is responsible for any name remapping needed
    to match the MLX module's parameter tree). Non-tensor entries are skipped.

    This function only needs NumPy, so it works even when MLX is not installed;
    in that case the values are returned as NumPy arrays (so conversion can be
    prepared off-device). When MLX is present the values are ``mx.array``.

    Args:
        state_dict: A PyTorch ``state_dict`` (mapping names to tensors).

    Returns:
        A new dict mapping the same names to MLX arrays (or NumPy arrays if MLX
        is unavailable).
    """
    out: dict[str, Any] = {}
    for name, tensor in state_dict.items():
        # Accept torch tensors (and anything exposing detach/cpu/numpy).
        if hasattr(tensor, "detach"):
            arr = tensor.detach().cpu().numpy()
        elif isinstance(tensor, np.ndarray):
            arr = tensor
        else:
            # Skip non-array entries (e.g. metadata).
            continue
        out[name] = mx.array(arr) if HAS_MLX else arr
    return out


def mlx_selective_scan_sequential(dt, A, Bmat, C, x):  # type: ignore[no-untyped-def]
    """Reference diagonal selective scan in MLX (sequential, experimental).

    Mirrors :func:`dimba.models.parallel_scan.selective_scan_sequential` but in
    MLX. Implements the correct diagonal recurrence with the inner dimension
    kept independent::

        dA  = exp(dt[..., None] * A)
        dBx = dt[..., None] * Bmat[:, :, None, :] * x[..., None]
        h_t = dA_t * h_{t-1} + dBx_t
        y_t = sum_s C_t[s] * h_t[..., s]

    This is a correctness reference, not an optimized kernel (see module
    docstring). Requires MLX.

    Args:
        dt: Timestep deltas ``[B, L, Din]`` as an ``mx.array``.
        A: State-decay ``[Din, Dstate]`` as an ``mx.array`` (negative real).
        Bmat: Input->state projection ``[B, L, Dstate]``.
        C: State->output projection ``[B, L, Dstate]``.
        x: SSM input ``[B, L, Din]``.

    Returns:
        Output ``y`` of shape ``[B, L, Din]`` as an ``mx.array``.

    Raises:
        RuntimeError: If MLX is not installed.
    """
    if not HAS_MLX:
        raise RuntimeError(_NO_MLX_MSG)

    batch, length, d_inner = dt.shape
    d_state = A.shape[1]

    # Discretize. mx broadcasting follows NumPy semantics.
    dA = mx.exp(dt[..., None] * A)  # [B, L, Din, Dstate]
    dBx = dt[..., None] * Bmat[:, :, None, :] * x[..., None]  # [B, L, Din, Dstate]

    h = mx.zeros((batch, d_inner, d_state), dtype=dt.dtype)
    ys = []
    for t in range(length):
        h = dA[:, t] * h + dBx[:, t]  # [B, Din, Dstate]
        # y_t[i] = sum_s C_t[s] * h_t[i, s]
        y_t = mx.sum(C[:, t][:, None, :] * h, axis=-1)  # [B, Din]
        ys.append(y_t)
    return mx.stack(ys, axis=1)  # [B, L, Din]


if HAS_MLX:

    class MLXMamba2Block(mlx_nn.Module):  # type: ignore[misc]
        """Experimental MLX Mamba-2 block (skeleton).

        Structural counterpart of :class:`dimba.models.denoiser.Mamba2Block` /
        :class:`dimba.models.simple_mamba.SimpleMamba2`. Uses
        :func:`mlx_selective_scan_sequential` for the SSM core. Not numerically
        validated; see module docstring.

        Args:
            d_model: Model dimension.
            d_state: SSM state dimension.
            d_expand: Inner-dimension expansion factor.
        """

        def __init__(self, d_model: int = 512, d_state: int = 16, d_expand: int = 2):
            super().__init__()
            self.d_model = d_model
            self.d_state = d_state
            self.d_inner = int(d_model * d_expand)

            self.norm = mlx_nn.LayerNorm(d_model)
            self.in_proj = mlx_nn.Linear(d_model, 2 * self.d_inner)
            self.dt_proj = mlx_nn.Linear(d_model, self.d_inner)
            self.B_proj = mlx_nn.Linear(d_model, d_state)
            self.C_proj = mlx_nn.Linear(d_model, d_state)
            self.out_proj = mlx_nn.Linear(self.d_inner, d_model)
            # A stored as negative real state-decay [Din, Dstate].
            self.A = -mx.ones((self.d_inner, d_state))

        def __call__(self, x):  # type: ignore[no-untyped-def]
            """Forward pass.

            Args:
                x: Input ``[B, L, d_model]`` as an ``mx.array``.

            Returns:
                Output ``[B, L, d_model]`` (residual added).
            """
            x_norm = self.norm(x)
            zx = self.in_proj(x_norm)
            z, x_proj = mx.split(zx, 2, axis=-1)
            dt = mlx_nn.softplus(self.dt_proj(x_norm))
            b = self.B_proj(x_norm)
            c = self.C_proj(x_norm)

            y = mlx_selective_scan_sequential(dt, self.A, b, c, x_proj)
            y = y * mlx_nn.silu(z)
            return x + self.out_proj(y)

    class MLXMamba2Denoiser(mlx_nn.Module):  # type: ignore[misc]
        """Experimental MLX denoiser: a stack of :class:`MLXMamba2Block` (skeleton).

        Minimal port of :class:`dimba.models.denoiser.Mamba2Denoiser`. Additive
        conditioning only (timestep + prompt summed and added in), kept simple on
        purpose. Not numerically validated; see module docstring.

        Args:
            d_model: Model dimension.
            num_layers: Number of blocks.
            d_state: SSM state dimension.
            expand: Inner-dimension expansion factor.
            cond_dim: Conditioning-vector dimension.
            time_embed_dim: Timestep-embedding dimension.
        """

        def __init__(
            self,
            d_model: int = 512,
            num_layers: int = 6,
            d_state: int = 16,
            expand: int = 2,
            cond_dim: int = 512,
            time_embed_dim: int = 512,
        ):
            super().__init__()
            self.d_model = d_model
            self.num_layers = num_layers
            self.blocks = [
                MLXMamba2Block(d_model=d_model, d_state=d_state, d_expand=expand)
                for _ in range(num_layers)
            ]
            self.time_proj = mlx_nn.Linear(time_embed_dim, cond_dim)
            self.cond_proj = mlx_nn.Linear(cond_dim, d_model)

        def __call__(self, x, cond, timestep_emb):  # type: ignore[no-untyped-def]
            """Forward pass.

            Args:
                x: Noisy embeddings ``[B, L, d_model]``.
                cond: Prompt conditioning ``[B, L, cond_dim]``.
                timestep_emb: Timestep embeddings ``[B, time_embed_dim]``.

            Returns:
                Denoised embeddings ``[B, L, d_model]``.
            """
            time_cond = self.time_proj(timestep_emb)[:, None, :]  # [B, 1, cond_dim]
            combined = cond + time_cond  # broadcast over L
            cond_add = self.cond_proj(combined)  # [B, L, d_model]

            out = x
            for block in self.blocks:
                out = block(out + cond_add)
            return out

else:  # pragma: no cover - exercised only without MLX installed

    class MLXMamba2Block:  # type: ignore[no-redef]
        """Stub for :class:`MLXMamba2Block` used when MLX is not installed.

        Importing the module succeeds; constructing this class raises so the
        failure is explicit and actionable.
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError(_NO_MLX_MSG)

    class MLXMamba2Denoiser:  # type: ignore[no-redef]
        """Stub for :class:`MLXMamba2Denoiser` used when MLX is not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError(_NO_MLX_MSG)
