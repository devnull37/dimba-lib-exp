"""MLX (Apple-GPU / Metal) Mamba-2 (SSD) mixer — weight-compatible with ``mamba_ssm.Mamba2``.

This is the *correct Mamba-2 SSD* MLX port (the older :mod:`dimba.backends.mlx.denoiser`
skeleton mirrors Mamba-1 / ``SimpleMamba2`` and is **not** weight-compatible with our
kernel-trained checkpoints). It mirrors :class:`dimba.models.torch_mamba2.TorchMamba2`
exactly (same parameter tree + chunked SSD scan), so the same checkpoint weights load and
the outputs match the PyTorch reference numerically (verified ~1e-8).

It runs on the Apple GPU via MLX's unified memory (default device is the GPU). Use it as a
fast SSD primitive and the base for a full MLX inference path.

Key MLX-vs-PyTorch differences handled here (all verified against ``mlx 0.29``):
  * ``mlx.nn.Conv1d`` uses **NLC** layout (no transpose) and weight shape
    ``(out, kernel, in/groups)`` — so a torch depthwise ``conv1d.weight`` ``[C,1,k]`` is
    loaded as ``moveaxis(2,1) -> [C,k,1]``. Causality via manual left-pad + ``padding=0``.
  * ``mx.split`` takes split **indices** (cut points), not section sizes.
  * ``nn.Linear`` is ``x @ W.T`` with ``[out,in]`` weights — torch weights load directly.
  * Gated RMSNorm (``norm_before_gate=False``): gate then normalize via
    ``mx.fast.rms_norm(y * silu(z), weight, eps)``.

If ``mlx`` is not installed the module still imports; the classes become stubs that raise.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

try:  # MLX is Apple-Silicon only, optional.
    import mlx.core as mx
    import mlx.nn as mlx_nn

    HAS_MLX = True
except ImportError:  # pragma: no cover - exercised only without MLX
    mx = None  # type: ignore[assignment]
    mlx_nn = None  # type: ignore[assignment]
    HAS_MLX = False

__all__ = ["HAS_MLX", "scan_chunked_mlx", "MLXMamba2Mixer", "load_torch_mamba2_state_dict"]

_NO_MLX = "MLX not installed (pip install mlx, Apple Silicon only). Use the torch backend."


def scan_chunked_mlx(x, dt, A, Bm, Cm, chunk_size: int = 128):
    """Chunked/parallel SSD scan in MLX — verbatim port of ``TorchMamba2._scan_chunked``.

    Shapes: ``x:[B,L,h,p]``, ``dt:[B,L,h]``, ``A:[h]``, ``Bm,Cm:[B,L,h,n]`` -> ``y:[B,L,h,p]``.
    ``ngroups==1`` callers may pass ``Bm/Cm`` as ``[B,L,1,n]`` (broadcasts). Numerically
    equivalent to the sequential recurrence ``S_t = exp(dt_t*A)S_{t-1} + (dt_t*x_t)⊗B_t``.
    """
    if not HAS_MLX:
        raise RuntimeError(_NO_MLX)
    B, L, H, P = x.shape
    cs = chunk_size
    xdt = x * dt[..., None]                       # absorb dt into x
    Adt = A.reshape(1, 1, H) * dt                 # absorb dt into A
    y = mx.zeros((B, L, H, P))
    S = mx.zeros((B, H, P, Bm.shape[-1]))
    for start in range(0, L, cs):
        end = min(start + cs, L)
        cl = end - start
        xc, ac = xdt[:, start:end], Adt[:, start:end]
        bc, cc = Bm[:, start:end], Cm[:, start:end]
        a_cum = mx.cumsum(ac, axis=1)             # [B,cl,h]
        # carried-state contribution
        y_carry = mx.einsum("bthn,bhpn->bthp", cc, S) * mx.exp(a_cum)[..., None]
        # intra-chunk diagonal: mask log to -inf above diagonal BEFORE exp (no inf*0=nan)
        logdecay = a_cum[:, :, None, :] - a_cum[:, None, :, :]      # [B,cl,cl,h]
        tri = mx.tril(mx.ones((cl, cl))) > 0
        logdecay = mx.where(tri[None, :, :, None], logdecay, float("-inf"))
        decay = mx.exp(logdecay)
        cb = mx.einsum("bthn,bshn->btsh", cc, bc)
        y_diag = mx.einsum("btsh,bshp->bthp", cb * decay, xc)
        y[:, start:end] = y_carry + y_diag
        # carry state to end of chunk
        a_last = a_cum[:, -1]
        state_decay = mx.exp(a_last[:, None] - a_cum)
        dstate = mx.einsum("bthn,bthp,bth->bhpn", bc, xc, state_decay)
        S = mx.exp(a_last)[:, :, None, None] * S + dstate
    return y


if HAS_MLX:

    class MLXMamba2Mixer(mlx_nn.Module):  # type: ignore[misc]
        """MLX Mamba-2 SSD mixer; param tree matches ``mamba_ssm.Mamba2`` / ``TorchMamba2``."""

        def __init__(self, d_model=384, d_state=128, d_conv=4, expand=2,
                     headdim=64, ngroups=1, chunk_size=128):
            super().__init__()
            self.d_model, self.d_state, self.d_conv = d_model, d_state, d_conv
            self.expand, self.headdim, self.ngroups = expand, headdim, ngroups
            self.chunk_size = chunk_size
            self.d_inner = expand * d_model
            assert self.d_inner % headdim == 0
            self.nheads = self.d_inner // headdim
            self.conv_dim = self.d_inner + 2 * ngroups * d_state
            d_in_proj = 2 * self.d_inner + 2 * ngroups * d_state + self.nheads

            self.in_proj = mlx_nn.Linear(d_model, d_in_proj, bias=False)
            # depthwise, NLC; padding=0 (causality via manual left-pad in __call__)
            self.conv1d = mlx_nn.Conv1d(self.conv_dim, self.conv_dim, d_conv,
                                        groups=self.conv_dim, padding=0, bias=True)
            self.dt_bias = mx.zeros((self.nheads,))
            self.A_log = mx.zeros((self.nheads,))
            self.D = mx.ones((self.nheads,))
            self.norm = mlx_nn.RMSNorm(self.d_inner, eps=1e-5)  # weight only; gate done manually
            self.out_proj = mlx_nn.Linear(self.d_inner, d_model, bias=False)

        def __call__(self, u):  # u: [B, L, d_model] (fp32)
            B, L, _ = u.shape
            di, cd, ng, ds = self.d_inner, self.conv_dim, self.ngroups, self.d_state
            # mx.split takes cut POINTS, not sizes:
            z, xBC, dt = mx.split(self.in_proj(u), [di, di + cd], axis=-1)
            xBC = self.conv1d(mx.pad(xBC, [(0, 0), (self.d_conv - 1, 0), (0, 0)]))  # causal, NLC
            xBC = mlx_nn.silu(xBC)
            x, Bm, Cm = mx.split(xBC, [di, di + ng * ds], axis=-1)

            dt = mlx_nn.softplus(dt + self.dt_bias)            # [B,L,h]
            A = -mx.exp(self.A_log)                            # [h]
            x = x.reshape(B, L, self.nheads, self.headdim)
            Bm = Bm.reshape(B, L, ng, ds)
            Cm = Cm.reshape(B, L, ng, ds)
            if ng != self.nheads:
                Bm = mx.repeat(Bm, self.nheads // ng, axis=2)
                Cm = mx.repeat(Cm, self.nheads // ng, axis=2)

            y = scan_chunked_mlx(x, dt, A, Bm, Cm, self.chunk_size)
            y = y + self.D.reshape(1, 1, self.nheads, 1) * x
            y = y.reshape(B, L, di)
            y = mx.fast.rms_norm(y * mlx_nn.silu(z), self.norm.weight, 1e-5)  # gated RMSNorm
            return self.out_proj(y)

else:  # pragma: no cover

    class MLXMamba2Mixer:  # type: ignore[no-redef]
        def __init__(self, *a: Any, **k: Any):
            raise RuntimeError(_NO_MLX)


def load_torch_mamba2_state_dict(mixer: "MLXMamba2Mixer", torch_sd: Dict[str, Any]):
    """Load a ``mamba_ssm.Mamba2`` / ``TorchMamba2`` mixer ``state_dict`` into ``mixer``.

    ``torch_sd`` maps the bare mixer keys (``in_proj.weight``, ``conv1d.weight``,
    ``conv1d.bias``, ``dt_bias``, ``A_log``, ``D``, ``norm.weight``, ``out_proj.weight``)
    to torch tensors (or numpy arrays). Only ``conv1d.weight`` is reshaped (torch ``[C,1,k]``
    -> MLX ``[C,k,1]``); everything else loads as-is. Materializes params with ``mx.eval``.
    """
    if not HAS_MLX:
        raise RuntimeError(_NO_MLX)
    items = []
    for name, t in torch_sd.items():
        arr = t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
        arr = arr.astype(np.float32)
        if name.endswith("conv1d.weight"):
            arr = np.transpose(arr, (0, 2, 1))  # [C,1,k] -> [C,k,1] (MLX layout)
        items.append((name, mx.array(arr)))
    mixer.load_weights(items, strict=False)
    mx.eval(mixer.parameters())
    return mixer
