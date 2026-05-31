"""Tests for latent normalization in LatentProjector and DIMBA."""

import torch
import pytest
from dimba.models.embeddings import LatentProjector
from dimba import DIMBA


def test_latent_projector_norm_unit_variance():
    """LatentProjector with latent_norm=True has ~unit std on encoder output."""
    proj = LatentProjector(input_dim=64, latent_dim=32, latent_norm=True)
    proj.eval()
    with torch.no_grad():
        x = torch.randn(8, 16, 64)
        z = proj.encode(x)
    std = z.std().item()
    assert 0.7 < std < 1.4, f"Expected std ~1.0, got {std}"


def test_latent_projector_no_norm_runs():
    """LatentProjector with latent_norm=False runs without error (no variance constraint)."""
    proj = LatentProjector(input_dim=64, latent_dim=32, latent_norm=False)
    proj.eval()
    with torch.no_grad():
        x = torch.randn(8, 16, 64)
        z = proj.encode(x)
    assert z.shape == (8, 16, 32)
    assert torch.isfinite(z).all()


def test_dimba_latent_norm_builds_and_forward():
    """DIMBA(latent_norm=True) builds, forward is finite."""
    model = DIMBA(
        vocab_size=64,
        d_model=64,
        d_prompt=32,
        num_diffusion_steps=16,
        num_denoiser_layers=2,
        d_state=8,
        latent_diffusion=True,
        d_latent=32,
        use_simple_mamba=True,
        latent_norm=True,
    )
    model.eval()
    B, L = 2, 8
    input_ids = torch.randint(0, 64, (B, L))
    t = torch.randint(0, 16, (B,))
    with torch.no_grad():
        x_pred, noise, _ = model(input_ids, t)
    assert torch.isfinite(x_pred).all(), "x_pred has non-finite values"


def test_dimba_latent_norm_encode_decode_roundtrip():
    """DIMBA(latent_norm=True) encode_latent/decode_latent runs without error."""
    model = DIMBA(
        vocab_size=64,
        d_model=64,
        d_prompt=32,
        num_diffusion_steps=16,
        num_denoiser_layers=2,
        d_state=8,
        latent_diffusion=True,
        d_latent=32,
        use_simple_mamba=True,
        latent_norm=True,
    )
    model.eval()
    B, L = 2, 8
    with torch.no_grad():
        x = torch.randn(B, L, 64)
        z = model.encode_latent(x)
        x_rec = model.decode_latent(z)
    assert z.shape == (B, L, 32), f"Unexpected latent shape {z.shape}"
    assert x_rec.shape == (B, L, 64), f"Unexpected reconstructed shape {x_rec.shape}"
    assert torch.isfinite(z).all(), "Latent z has non-finite values"
    assert torch.isfinite(x_rec).all(), "Reconstructed x has non-finite values"
