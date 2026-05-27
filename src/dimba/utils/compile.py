"""torch.compile helper, guarded for CPU/MPS-only environments.

``torch.compile`` only delivers meaningful speedups (and is only reliably
available) on CUDA in many builds. On CPU-only / MPS environments compilation
can be a no-op at best and a source of breakage at worst. This helper makes
opting in safe: it compiles only when ``torch.compile`` exists *and* CUDA is
available, and it never raises -- any failure falls back to the eager module.
"""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn

__all__ = ["maybe_compile"]


def maybe_compile(
    module: nn.Module,
    *,
    enable: bool = True,
    mode: str = "reduce-overhead",
) -> nn.Module:
    """Return a ``torch.compile``-d module when it is safe, else the module.

    Compilation is applied only when all of the following hold:

    * ``enable`` is ``True``;
    * ``torch.compile`` exists in the running torch build;
    * ``torch.cuda.is_available()`` is ``True``.

    Any exception raised during the availability checks or during
    ``torch.compile`` itself is swallowed (with a warning) and the original
    eager ``module`` is returned, so calling this is always safe.

    Args:
        module: The module to (optionally) compile.
        enable: Master switch; when ``False`` the module is returned unchanged.
        mode: Compilation mode forwarded to ``torch.compile`` (e.g.
            ``"reduce-overhead"``, ``"max-autotune"``, ``"default"``).

    Returns:
        The compiled module if compilation was applied, otherwise ``module``.
    """
    if not enable:
        return module

    try:
        if not hasattr(torch, "compile"):
            return module
        if not torch.cuda.is_available():
            # No CUDA: torch.compile rarely helps and may break; skip it.
            return module
        return torch.compile(module, mode=mode)
    except Exception as exc:  # pragma: no cover - defensive guard
        warnings.warn(
            f"maybe_compile: torch.compile failed ({exc!r}); using eager module.",
            RuntimeWarning,
            stacklevel=2,
        )
        return module
