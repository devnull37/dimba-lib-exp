"""Tests for AdaLN-Zero conditioning (conditioning_type='adaln').

Covers:
1. AdaLNZeroConditioning zero-init: scale/shift/gate are all ~0 at init.
2. DIMBA(conditioning_type='adaln') builds and forward returns finite tensor
   of the right shape.
3. Identity-at-init: with dropout off (eval mode), the denoiser output equals
   its input at init because gate=0 makes every block a no-op.
"""

import torch
import pytest

from dimba.models.embeddings import AdaLNZeroConditioning
from dimba import DIMBA


# ---------------------------------------------------------------------------
# Shared tiny-model factory (kept fast: <2s per test)
# ---------------------------------------------------------------------------

def _tiny_adaln(**kw):
    return DIMBA(
        vocab_size=64,
        d_model=64,
        d_prompt=32,
        num_diffusion_steps=16,
        num_denoiser_layers=2,
        d_state=8,
        latent_diffusion=True,
        d_latent=32,
        use_simple_mamba=True,
        conditioning_type="adaln",
        **kw,
    )


# ---------------------------------------------------------------------------
# 1. AdaLNZeroConditioning zero-init
# ---------------------------------------------------------------------------

class TestAdaLNZeroInit:
    """scale, shift, and gate must all be ~0 at initialization."""

    def test_scale_shift_gate_zero_at_init(self):
        cond_dim = 32
        target_dim = 64
        layer = AdaLNZeroConditioning(cond_dim=cond_dim, target_dim=target_dim)

        # Random conditioning; at init the linear is zero-initialized so outputs
        # should all be zero regardless of the input magnitude.
        cond = torch.randn(2, 8, cond_dim)
        scale, shift, gate = layer(cond)

        assert scale.shape == (2, 8, target_dim)
        assert shift.shape == (2, 8, target_dim)
        assert gate.shape == (2, 8, target_dim)

        assert torch.allclose(scale, torch.zeros_like(scale), atol=1e-6), (
            f"scale not ~0 at init: max abs = {scale.abs().max().item()}"
        )
        assert torch.allclose(shift, torch.zeros_like(shift), atol=1e-6), (
            f"shift not ~0 at init: max abs = {shift.abs().max().item()}"
        )
        assert torch.allclose(gate, torch.zeros_like(gate), atol=1e-6), (
            f"gate not ~0 at init: max abs = {gate.abs().max().item()}"
        )

    def test_zero_init_holds_for_large_cond(self):
        """Even with large conditioning values the output must be 0 at init."""
        layer = AdaLNZeroConditioning(cond_dim=16, target_dim=32)
        cond = torch.ones(1, 1, 16) * 1000.0
        scale, shift, gate = layer(cond)
        for name, val in [("scale", scale), ("shift", shift), ("gate", gate)]:
            assert torch.allclose(val, torch.zeros_like(val), atol=1e-5), (
                f"{name} not ~0 with large cond at init"
            )


# ---------------------------------------------------------------------------
# 2. DIMBA(conditioning_type='adaln') builds and forward is finite/right shape
# ---------------------------------------------------------------------------

class TestDIMBAAdaLNBuild:

    def test_model_builds(self):
        model = _tiny_adaln()
        assert model is not None
        assert model.denoiser.conditioning_type == "adaln"

    def test_forward_shape_and_finite(self):
        model = _tiny_adaln()
        B, L = 2, 16
        ids = torch.randint(0, 64, (B, L))
        t = torch.randint(0, 16, (B,))

        x_pred, noise, latent_info = model(ids, t)

        assert x_pred.shape == (B, L, 64), f"unexpected shape: {x_pred.shape}"
        assert torch.isfinite(x_pred).all(), "x_pred contains non-finite values"
        assert torch.isfinite(noise).all(), "noise contains non-finite values"

    def test_forward_shape_matches_d_model(self):
        model = _tiny_adaln()
        ids = torch.randint(0, 64, (3, 12))
        t = torch.randint(0, 16, (3,))
        x_pred, _, _ = model(ids, t)
        assert x_pred.shape == (3, 12, model.d_model)


# ---------------------------------------------------------------------------
# 3. Identity-at-init: denoiser output == input when gate=0
# ---------------------------------------------------------------------------

class TestAdaLNIdentityAtInit:
    """At init, gate=0 -> every AdaLN block is a no-op: denoiser(z) == z."""

    def test_denoiser_identity(self):
        model = _tiny_adaln(dropout=0.0)
        model.eval()

        B, L = 2, 12
        d_latent = model.d_latent  # 32
        device = next(model.parameters()).device

        z = torch.randn(B, L, d_latent, device=device)
        cond = model.conditioning_from_prompt(None, B, device)
        t = torch.zeros(B, dtype=torch.long, device=device)
        t_emb = model.timestep_embed(t)

        with torch.no_grad():
            out = model.denoiser(z, cond, t_emb)

        assert out.shape == z.shape, f"shape mismatch: {out.shape} vs {z.shape}"
        assert torch.allclose(out, z, atol=1e-5), (
            f"Denoiser is not identity at init. "
            f"Max abs diff: {(out - z).abs().max().item():.2e}"
        )

    def test_denoiser_identity_multiple_batches(self):
        """Identity should hold for different batch shapes."""
        model = _tiny_adaln(dropout=0.0)
        model.eval()

        device = next(model.parameters()).device

        for B, L in [(1, 8), (4, 20)]:
            d_latent = model.d_latent
            z = torch.randn(B, L, d_latent, device=device)
            cond = model.conditioning_from_prompt(None, B, device)
            t = torch.zeros(B, dtype=torch.long, device=device)
            t_emb = model.timestep_embed(t)

            with torch.no_grad():
                out = model.denoiser(z, cond, t_emb)

            assert torch.allclose(out, z, atol=1e-5), (
                f"Identity failed for B={B}, L={L}. "
                f"Max abs diff: {(out - z).abs().max().item():.2e}"
            )
