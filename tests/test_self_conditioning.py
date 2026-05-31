"""Tests for the self-conditioning feature in DIMBA.

Covers:
- DIMBA(self_conditioning=True) forward with x_self_cond=None and explicit x_self_cond.
- compute_dimba_losses on a self_conditioning=True model over ~15 steps with backward.
- Regression: self_conditioning=False model still works through compute_dimba_losses.
"""

import torch
import pytest

from dimba import DIMBA
from dimba.training.trainer import compute_dimba_losses


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


# ------------------------------------------------------------------ loss + backward tests

class TestSelfConditioningLoss:
    """compute_dimba_losses exercises the 50/50 self-cond branch; loss must be finite."""

    def _run_steps(self, model: DIMBA, num_steps: int = 15) -> None:
        """Run num_steps of compute_dimba_losses with backward, assert finite loss each time.

        Uses Adam + gradient clipping to match real training stability.
        """
        torch.manual_seed(0)
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        B, L = 4, 12

        for step in range(num_steps):
            input_ids = torch.randint(0, 64, (B, L))
            t = torch.randint(0, model.num_diffusion_steps, (B,))

            loss, parts = compute_dimba_losses(model, input_ids, t)

            assert torch.isfinite(loss), (
                f"Loss is not finite at step {step}: {loss.item()}"
            )
            for name, val in parts.items():
                assert torch.isfinite(val), (
                    f"Loss component '{name}' is not finite at step {step}: {val.item()}"
                )

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    def test_self_conditioning_true_15_steps(self):
        """~15 training steps through the self-cond branch: every loss is finite."""
        torch.manual_seed(42)
        model = _tiny_model(self_conditioning=True)
        self._run_steps(model, num_steps=15)

    def test_self_conditioning_false_regression(self):
        """self_conditioning=False model: compute_dimba_losses still produces finite loss."""
        torch.manual_seed(42)
        model = _tiny_model(self_conditioning=False)
        self._run_steps(model, num_steps=15)

    def test_both_branches_hit(self):
        """Over enough steps, both the self-cond branch and the skip branch are exercised.

        We verify this by counting how many times the no-grad pre-pass is triggered.
        We patch torch.rand to alternate deterministically so both code paths run.
        """
        import unittest.mock as mock

        model = _tiny_model(self_conditioning=True)
        model.train()

        # Alternate torch.rand(()) return value: 0.3 triggers the branch, 0.7 skips it.
        side_effects = [torch.tensor(0.3), torch.tensor(0.7)] * 10
        B, L = 2, 8
        input_ids = torch.randint(0, 64, (B, L))
        t = torch.randint(0, 16, (B,))

        branch_triggered = 0
        for val in side_effects[:8]:
            with mock.patch("torch.rand", return_value=val):
                loss, _ = compute_dimba_losses(model, input_ids, t)
                assert torch.isfinite(loss), f"Loss not finite with rand={val.item()}"
                if float(val) < 0.5:
                    branch_triggered += 1

        assert branch_triggered == 4, (
            f"Expected 4 self-cond branch hits, got {branch_triggered}"
        )
