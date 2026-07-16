"""Manual CUDA-graph capture for fixed-shape inference callables.

The masked-diffusion sampler calls the same backbone forward (identical input
shapes) once per denoising step. Eager execution pays Python dispatch plus
~1000 kernel launches per call, which dominates wall-clock at the short
sequence lengths used for generation. Capturing the call into a
``torch.cuda.CUDAGraph`` replays the entire forward as one launch.

Unlike ``torch.compile(mode="reduce-overhead")`` this captures *through*
opaque custom kernels (mamba_ssm's Triton ops) instead of graph-breaking on
them, and warms up in a few forwards instead of a full Inductor compile.

Contract for wrapped callables:

* Inputs are positional tensors on the same CUDA device; everything else must
  be closed over (and constant across calls).
* The callable must be shape-deterministic: same input shapes -> same output
  shape, no data-dependent control flow, no host syncs, no RNG.
* The returned tensor is a static buffer that is overwritten by the next call
  with the same input shapes: consume (or clone) it before calling again.

On any capture or replay failure -- and on non-CUDA inputs -- the wrapper
falls back to the eager callable permanently, mirroring ``maybe_compile_fn``.
"""

from __future__ import annotations

import warnings
from typing import Callable

import torch

__all__ = ["GraphedFn", "cuda_graphs_supported"]


def cuda_graphs_supported() -> bool:
    """Whether this torch build/device can attempt CUDA-graph capture."""
    return (
        torch.cuda.is_available()
        and hasattr(torch.cuda, "CUDAGraph")
        and hasattr(torch.cuda, "graph")
    )


class GraphedFn:
    """Wrap ``fn(*tensors) -> tensor`` with per-shape CUDA-graph capture.

    One graph is captured (and cached) per distinct tuple of input
    ``(shape, dtype)``; the sampler's fixed-shape step reuses a single graph
    for the whole trajectory, while a shape change (new prompt length,
    CFG-truncated single-row batch) captures one additional graph.
    """

    def __init__(self, fn: Callable[..., torch.Tensor], warmup_iters: int = 3):
        self._fn = fn
        self._warmup_iters = warmup_iters
        self._graphs: dict = {}
        self._broken = not cuda_graphs_supported()

    @property
    def captured_graphs(self) -> int:
        """Number of shapes captured so far (0 => every call ran eager)."""
        return len(self._graphs)

    @property
    def eager_fallback(self) -> bool:
        """True when capture failed (or was unsupported) and calls run eager."""
        return self._broken

    def _capture(self, tensors):
        static_inputs = tuple(t.clone() for t in tensors)
        # Warm up on a side stream so lazy init (cuBLAS handles, Triton
        # autotune, allocator blocks) happens outside the capture.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(self._warmup_iters):
                self._fn(*static_inputs)
        torch.cuda.current_stream().wait_stream(side)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_output = self._fn(*static_inputs)
        return static_inputs, graph, static_output

    def __call__(self, *tensors: torch.Tensor) -> torch.Tensor:
        if self._broken or not all(
            isinstance(t, torch.Tensor) and t.is_cuda for t in tensors
        ):
            return self._fn(*tensors)
        key = tuple((tuple(t.shape), t.dtype) for t in tensors)
        entry = self._graphs.get(key)
        if entry is None:
            try:
                entry = self._capture(tensors)
            except Exception as exc:
                self._broken = True
                self._graphs.clear()
                warnings.warn(
                    f"GraphedFn: CUDA-graph capture failed ({exc!r}); "
                    "falling back to eager permanently.",
                    RuntimeWarning,
                )
                return self._fn(*tensors)
            self._graphs[key] = entry
        static_inputs, graph, static_output = entry
        for static, live in zip(static_inputs, tensors):
            static.copy_(live)
        graph.replay()
        return static_output
