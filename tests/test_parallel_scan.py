"""Tests for the vectorized diagonal selective scan.

Validates that the length-parallel :func:`selective_scan` matches the explicit
loop reference :func:`selective_scan_sequential`, that the bidirectional variant
behaves sensibly, and that gradients flow through the vectorized scan.
"""

import pytest
import torch

from dimba.models.parallel_scan import (
    bidirectional_selective_scan,
    selective_scan,
    selective_scan_sequential,
)


def _random_ssm_inputs(batch, length, d_inner, d_state, seed=0, dtype=torch.float64):
    """Build random SSM inputs with a physically sensible parameterization.

    ``dt`` is positive (softplus output) and ``A`` is negative real, matching the
    Mamba discretization assumptions used by the scan.

    Args:
        batch: Batch size ``B``.
        length: Sequence length ``L``.
        d_inner: Inner dimension ``Din``.
        d_state: State dimension ``Dstate``.
        seed: RNG seed for reproducibility.
        dtype: Tensor dtype (float64 by default for tight parity checks).

    Returns:
        Tuple ``(dt, A, Bmat, C, x)``.
    """
    g = torch.Generator().manual_seed(seed)

    dt = torch.nn.functional.softplus(
        torch.randn(batch, length, d_inner, generator=g, dtype=dtype)
    )
    A = -torch.rand(d_inner, d_state, generator=g, dtype=dtype) - 0.1  # negative real
    Bmat = torch.randn(batch, length, d_state, generator=g, dtype=dtype)
    C = torch.randn(batch, length, d_state, generator=g, dtype=dtype)
    x = torch.randn(batch, length, d_inner, generator=g, dtype=dtype)
    return dt, A, Bmat, C, x


SHAPES = [
    (1, 1, 1, 1),
    (2, 4, 3, 5),
    (3, 8, 6, 4),
    (2, 16, 8, 16),
    (1, 50, 4, 8),  # spans multiple chunks (chunk_size default 64 -> use small cs)
    (2, 130, 5, 3),  # > default chunk_size, exercises the chunk-carry path
]


@pytest.mark.parametrize("batch,length,d_inner,d_state", SHAPES)
@pytest.mark.parametrize("stable", [True, False])
def test_vectorized_matches_sequential(batch, length, d_inner, d_state, stable):
    """Vectorized scan must match the sequential reference within tolerance."""
    dt, A, Bmat, C, x = _random_ssm_inputs(batch, length, d_inner, d_state, seed=batch + length)

    y_ref = selective_scan_sequential(dt, A, Bmat, C, x)
    # Use a small chunk_size so the chunk-carry path is exercised even for L<64.
    y_vec = selective_scan(dt, A, Bmat, C, x, stable=stable, chunk_size=8)

    assert y_vec.shape == (batch, length, d_inner)
    assert torch.allclose(y_vec, y_ref, rtol=1e-6, atol=1e-8), (
        f"max abs diff = {(y_vec - y_ref).abs().max().item():.3e}"
    )


def test_chunk_size_invariance():
    """Result must be independent of chunk_size in stable mode."""
    dt, A, Bmat, C, x = _random_ssm_inputs(2, 40, 5, 6, seed=123)
    y_ref = selective_scan_sequential(dt, A, Bmat, C, x)
    for cs in (1, 3, 8, 16, 64):
        y = selective_scan(dt, A, Bmat, C, x, stable=True, chunk_size=cs)
        assert torch.allclose(y, y_ref, rtol=1e-6, atol=1e-8), f"chunk_size={cs}"


def test_invalid_chunk_size():
    """Non-positive chunk_size is rejected."""
    dt, A, Bmat, C, x = _random_ssm_inputs(1, 4, 2, 2)
    with pytest.raises(ValueError):
        selective_scan(dt, A, Bmat, C, x, chunk_size=0)


def test_bidirectional_sanity():
    """Bidirectional scan equals forward + reversed-backward (forward order)."""
    fwd = _random_ssm_inputs(2, 12, 4, 5, seed=7)
    bwd = _random_ssm_inputs(2, 12, 4, 5, seed=99)

    y_bi = bidirectional_selective_scan(*fwd, *bwd)

    # Reconstruct expectation manually using the sequential reference.
    y_f = selective_scan_sequential(*fwd)
    flip = lambda t: torch.flip(t, dims=[1])
    dt_b, A_b, B_b, C_b, x_b = bwd
    y_b_rev = selective_scan_sequential(flip(dt_b), A_b, flip(B_b), flip(C_b), flip(x_b))
    y_b = torch.flip(y_b_rev, dims=[1])

    assert y_bi.shape == y_f.shape
    assert torch.allclose(y_bi, y_f + y_b, rtol=1e-6, atol=1e-8)


def test_bidirectional_reduces_to_forward_when_backward_is_zero():
    """Bidirectional with a zero-input backward pass equals the forward pass.

    Setting the backward direction's input/projection (``x_bwd`` and
    ``Bmat_bwd``) to zero makes its SSM state -- and hence its output --
    identically zero, so the combined (summed) result must equal the standalone
    forward scan. This is an unambiguous structural check on the recombination.
    """
    fwd = _random_ssm_inputs(2, 11, 3, 4, seed=42)
    dt_b, A_b, _, C_b, _ = _random_ssm_inputs(2, 11, 3, 4, seed=84)
    zero_B = torch.zeros(2, 11, 4, dtype=torch.float64)
    zero_x = torch.zeros(2, 11, 3, dtype=torch.float64)

    y_bi = bidirectional_selective_scan(*fwd, dt_b, A_b, zero_B, C_b, zero_x)
    y_fwd = selective_scan(*fwd)

    assert torch.allclose(y_bi, y_fwd, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("stable", [True, False])
def test_gradients_flow(stable):
    """loss.backward() must populate grads for every input that requires grad."""
    dt, A, Bmat, C, x = _random_ssm_inputs(2, 10, 4, 5, seed=5, dtype=torch.float32)
    for t in (dt, A, Bmat, C, x):
        t.requires_grad_(True)

    y = selective_scan(dt, A, Bmat, C, x, stable=stable, chunk_size=4)
    loss = y.pow(2).mean()
    loss.backward()

    for name, t in [("dt", dt), ("A", A), ("Bmat", Bmat), ("C", C), ("x", x)]:
        assert t.grad is not None, f"no grad for {name}"
        assert torch.isfinite(t.grad).all(), f"non-finite grad for {name}"


def test_zero_input_gives_zero_output():
    """Zero SSM input (x=0, B=0) must produce zero output."""
    dt, A, Bmat, C, x = _random_ssm_inputs(2, 7, 3, 4, seed=11)
    Bmat = torch.zeros_like(Bmat)
    x = torch.zeros_like(x)
    y = selective_scan(dt, A, Bmat, C, x)
    assert torch.allclose(y, torch.zeros_like(y), atol=1e-10)
