"""Regression tests for the v2 overhaul.

Covers the correctness fixes and new capabilities: zero-terminal-SNR schedule,
FiLM identity init, the 3-tuple forward, prompt-mask (clean-prefix) conditioning,
self-conditioning / CFG / v-prediction / latent modes, sampler shapes, config
round-trip, and the combined training loss.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dimba.models.diffusion import DIMBA
from dimba.diffusion.schedules import CosineNoiseSchedule
from dimba.diffusion.sampling import sample_from_model


def tiny(**kw):
    return DIMBA(
        vocab_size=40,
        d_model=16,
        d_prompt=16,
        num_diffusion_steps=20,
        num_denoiser_layers=2,
        d_state=8,
        expand=2,
        use_simple_mamba=True,
        **kw,
    )


def test_zero_terminal_snr():
    s = CosineNoiseSchedule(num_steps=50, zero_terminal_snr=True)
    acp = s.get_alphas_cumprod()
    assert float(acp[0]) == pytest.approx(1.0, abs=1e-4)
    assert float(acp[-1]) == pytest.approx(0.0, abs=1e-6)
    # Without the fix, terminal SNR is (incorrectly) nonzero.
    s2 = CosineNoiseSchedule(num_steps=50, zero_terminal_snr=False)
    assert float(s2.get_alphas_cumprod()[-1]) > 0.0


def test_film_identity_init():
    from dimba.models.embeddings import FiLMConditioning

    f = FiLMConditioning(8, 8)
    x = torch.randn(2, 5, 8)
    cond = torch.randn(2, 5, 8)
    # gamma=1, beta=0 at init => identity (independent of conditioning).
    assert torch.allclose(f(x, cond), x, atol=1e-5)


def test_forward_returns_three_tuple():
    m = tiny()
    out = m(torch.randint(0, 40, (2, 6)), torch.randint(0, 20, (2,)))
    assert isinstance(out, tuple) and len(out) == 3


@pytest.mark.parametrize(
    "kw",
    [
        {},
        {"self_conditioning": True},
        {"latent_diffusion": True, "d_latent": 8},
        {
            "latent_diffusion": True,
            "d_latent": 8,
            "self_conditioning": True,
            "prediction_type": "v",
        },
        {"conditioning_type": "additive"},
    ],
)
def test_forward_and_backward(kw):
    m = tiny(**kw)
    ids = torch.randint(0, 40, (2, 6))
    t = torch.randint(0, 20, (2,))
    xp, _noise, info = m(ids, t)
    assert xp.shape == (2, 6, 16)
    loss = ((info["z0_hat"] - info["z_0"]) ** 2).mean()
    loss.backward()
    assert torch.isfinite(loss)


def test_prompt_mask_keeps_prefix_clean():
    m = tiny()
    ids = torch.randint(0, 40, (2, 6))
    t = torch.randint(0, 20, (2,))
    pm = torch.zeros(2, 6, dtype=torch.bool)
    pm[:, :3] = True
    _xp, _noise, info = m(ids, t, prompt_mask=pm)
    assert info["diffuse_mask"] is not None
    # Prompt positions are not noised: x_t == z_0 there.
    assert torch.allclose(info["x_t"][:, :3], info["z_0"][:, :3], atol=1e-5)


def test_sampling_shapes_and_cfg():
    m = tiny()
    ids = torch.randint(0, 40, (2, 4))
    assert sample_from_model(m, ids, seq_len=5, num_steps=5, top_k=10).shape == (2, 5)
    assert sample_from_model(m, ids, seq_len=5, num_steps=5, guidance_scale=2.0).shape == (2, 5)


def test_config_roundtrip():
    m = tiny(self_conditioning=True, latent_diffusion=True, d_latent=8)
    m2 = DIMBA(**m.config)
    assert m2.self_conditioning and m2.latent_diffusion and m2.d_latent == 8


def test_latent_scale_calibration():
    # Embedding mode: default scale = 1/embed_init_std = 50 -> ~unit-variance signal.
    m = tiny()
    assert m.latent_scale == pytest.approx(50.0, rel=1e-3)
    x = m.token_embed(torch.randint(0, 40, (2, 5)))
    s = m.encode_latent(x)
    assert 0.5 < float(s.std()) < 2.0
    assert torch.allclose(m.decode_latent(s), x, atol=1e-4)  # round-trips exactly
    new = m.calibrate_latent_scale(torch.randint(0, 40, (4, 8)))
    assert new > 0 and m.config["latent_scale"] == pytest.approx(new)


def test_combined_loss():
    pytest.importorskip("pytorch_lightning")
    from dimba.training.trainer import compute_dimba_losses

    m = tiny(latent_diffusion=True, d_latent=8)
    ids = torch.randint(0, 40, (2, 6))
    t = torch.randint(0, 20, (2,))
    loss, parts = compute_dimba_losses(m, ids, t)
    assert torch.isfinite(loss)
    assert "diff_loss" in parts and "ce_loss" in parts
    loss.backward()
