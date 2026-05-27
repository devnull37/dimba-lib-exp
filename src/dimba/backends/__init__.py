"""Compute backends for DIMBA.

The default backend is PyTorch (always available). An experimental MLX backend
for Apple Silicon lives in :mod:`dimba.backends.mlx`; it is only usable when the
``mlx`` package is installed. Use :func:`list_available_backends` to discover
which backends can be used in the current environment.
"""

from __future__ import annotations

__all__ = ["list_available_backends"]


def list_available_backends() -> list[str]:
    """Report the compute backends usable in the current environment.

    ``"torch"`` is always reported (it is a hard dependency). ``"mlx"`` is added
    only if ``mlx.core`` can be imported, i.e. on an Apple-Silicon machine with
    MLX installed.

    Returns:
        A list of backend identifier strings, e.g. ``["torch"]`` or
        ``["torch", "mlx"]``.
    """
    backends = ["torch"]
    try:
        import mlx.core  # noqa: F401

        backends.append("mlx")
    except ImportError:
        pass
    return backends
