"""Tests for the self-conditioning feature in DIMBA.

Covers:
- DIMBA(self_conditioning=True) forward with x_self_cond=None and explicit x_self_cond.
"""

import torch

from dimba import DIMBA


# ------------------------------------------------------------------ tiny model factory

def _tiny_model(self_conditioning: bool) -> DIMBA:
    """Build a minimal DIMBA suitable for fast CPU tests.

    dropout=0.0 to avoid stochastic NaNs in tiny latent projectors under
    random perturbations; the models are sized for speed, not capacity.
    """
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
        self_conditioning=self_conditioning,
        dropout=0.0,
    )


# ------------------------------------------------------------------ forward tests

class TestSelfConditioningForward:
    """DIMBA forward runs without error when self_conditioning=True."""

    def test_forward_no_self_cond(self):
        """x_self_cond=None (the default path): output is finite."""
        model = _tiny_model(self_conditioning=True)
        model.eval()
        B, L = 2, 16
        input_ids = torch.randint(0, 64, (B, L))
        t = torch.randint(0, 16, (B,))

        with torch.no_grad():
            x_pred, noise, info = model(input_ids, t, x_self_cond=None)

        assert x_pred.shape == (B, L, model.d_model), "x_pred shape mismatch"
        assert torch.isfinite(x_pred).all(), "x_pred contains non-finite values"
        assert torch.isfinite(info["z0_hat"]).all(), "z0_hat contains non-finite values"

    def test_forward_with_explicit_self_cond(self):
        """Explicit x_self_cond tensor: output is finite."""
        model = _tiny_model(self_conditioning=True)
        model.eval()
        B, L = 2, 16
        input_ids = torch.randint(0, 64, (B, L))
        t = torch.randint(0, 16, (B,))
        x_self_cond = torch.randn(B, L, model.d_latent)

        with torch.no_grad():
            x_pred, noise, info = model(input_ids, t, x_self_cond=x_self_cond)

        assert x_pred.shape == (B, L, model.d_model), "x_pred shape mismatch"
        assert torch.isfinite(x_pred).all(), "x_pred with explicit self_cond contains non-finite values"
        assert torch.isfinite(info["z0_hat"]).all(), "z0_hat with explicit self_cond contains non-finite values"

    def test_self_cond_proj_initialised_as_identity(self):
        """self_cond_proj is initialised so the model ignores the zero prior -> identity-like."""
        model = _tiny_model(self_conditioning=True)
        assert model.self_cond_proj is not None
        B, L, d = 1, 8, model.d_latent
        x = torch.randn(B, L, d)
        zeros = torch.zeros(B, L, d)
        fused = model.self_cond_proj(torch.cat([x, zeros], dim=-1))
        # The identity init means fused ~= x (within floating-point tolerance).
        assert torch.allclose(fused, x, atol=1e-6), (
            "self_cond_proj did not initialise to near-identity on zero prior"
        )

    def test_self_cond_proj_absent_when_disabled(self):
        """self_cond_proj should be None when self_conditioning=False."""
        model = _tiny_model(self_conditioning=False)
        assert model.self_cond_proj is None
