"""Tests for the research-driven diffusion improvements (all flag-gated, default-off)."""

import math
import torch

from dimba import DIMBA
from dimba.diffusion.schedules import CosineNoiseSchedule
from dimba.diffusion.sampling import sample_from_model


def _tiny(**kw):
    cfg = dict(vocab_size=48, d_model=48, d_prompt=24, num_diffusion_steps=16,
               num_denoiser_layers=2, d_state=8, latent_diffusion=True, d_latent=24,
               use_simple_mamba=True)
    cfg.update(kw)
    return DIMBA(**cfg)


# ---- timestep sampling ----
def test_timestep_modes_in_range():
    sch = CosineNoiseSchedule(num_steps=64)
    for mode in ("uniform", "logit_normal", "logsnr_uniform"):
        t = sch.sample_timesteps(128, torch.device("cpu"), mode=mode)
        assert t.dtype == torch.long
        assert int(t.min()) >= 0 and int(t.max()) <= 63
        assert torch.isfinite(t.float()).all()


def test_logit_normal_concentrates_mid():
    sch = CosineNoiseSchedule(num_steps=64)
    g = torch.Generator()  # determinism not critical; use many samples
    u = sch.sample_timesteps(8000, torch.device("cpu"), mode="uniform")
    ln = sch.sample_timesteps(8000, torch.device("cpu"), mode="logit_normal")
    mid = lambda t: ((t >= 16) & (t < 48)).float().mean().item()
    assert mid(ln) > mid(u) + 0.1  # logit-normal puts more mass on the middle


def test_antithetic_and_exclude_zero():
    sch = CosineNoiseSchedule(num_steps=64)
    t = sch.sample_timesteps(64, torch.device("cpu"), mode="uniform", antithetic=True)
    assert len(t) == 64
    # antithetic pairs mirror around T-1
    half = 32
    assert torch.all(t[:half] + t[half:half * 2] == 63)
    t0 = sch.sample_timesteps(500, torch.device("cpu"), mode="uniform", exclude_zero=True)
    assert int(t0.min()) >= 1


# ---- non-Gaussian noise ----
def test_student_t_noise_unit_variance():
    sch = CosineNoiseSchedule(num_steps=16, noise_dist="student_t", noise_df=5.0)
    n = sch.sample_noise(torch.zeros(5000, 8))
    assert torch.isfinite(n).all()
    assert 0.8 < float(n.std()) < 1.25  # rescaled to ~unit variance


def test_dimba_student_t_forward():
    m = _tiny(noise_dist="student_t")
    assert m.config.get("noise_dist") == "student_t"
    xp, _, _ = m(torch.randint(0, 48, (2, 6)), torch.randint(0, 16, (2,)))
    assert torch.isfinite(xp).all()


# ---- sampling: clamping + CFG mode ----
def test_soft_and_hard_clamp_run():
    m = _tiny().eval()
    for mode in ("soft", "hard"):
        out = sample_from_model(m, None, seq_len=8, num_steps=8,
                                clamp_mode=mode, clamp_from=0.5, device="cpu")
        assert out.shape == (1, 8)
        assert int(out.min()) >= 0 and int(out.max()) < 48
    # legacy flag still works (== hard, every step)
    out = sample_from_model(m, None, seq_len=8, num_steps=8, clamp_to_tokens=True, device="cpu")
    assert out.shape == (1, 8)


def test_cfg_eps_mode_runs():
    m = _tiny().eval()
    prompt = torch.randint(0, 48, (1, 4))
    out = sample_from_model(m, prompt, seq_len=8, num_steps=8,
                            guidance_scale=2.0, cfg_mode="eps", device="cpu")
    assert out.shape == (1, 8) and torch.isfinite(out.float()).all()


# ---- MLX v2 support ----
def test_mlx_v2_supported():
    """MLX backend now supports the v2 config (adaln / v-pred / head-norm).

    The old guard (NotImplementedError) has been replaced with real v2 support.
    This test confirms from_torch succeeds and produces finite logits.

    NOTE: _tiny() uses use_simple_mamba=True (SimpleMamba2), which has a
    different state_dict layout from TorchMamba2/MLXMamba2Mixer.  To keep this
    test fast and decoupled from mixer weight-format details we just verify that
    construction and a single-step forward pass succeed without error.
    """
    import pytest
    pytest.importorskip("mlx.core")
    import numpy as np
    import mlx.core as mx
    from dimba.backends.mlx.model import MLXDIMBA
    # _tiny uses d_latent=24 which is too small for TorchMamba2 (needs d_inner >= headdim=64).
    # Override d_latent=64 and use_simple_mamba=False so mixer weights are
    # TorchMamba2-compatible (required by load_torch_mamba2_state_dict).
    v2 = _tiny(conditioning_type="adaln", prediction_type="v", use_head_norm=True,
               latent_norm=True, self_conditioning=True, use_simple_mamba=False,
               d_latent=64)
    v2.eval()
    mlx_m = MLXDIMBA.from_torch(v2)
    assert mlx_m.conditioning_type == "adaln"
    assert mlx_m.prediction_type == "v"
    assert mlx_m.use_head_norm is True
    # Quick smoke: sample_logits returns finite values.
    noise = np.random.RandomState(0).randn(1, 4, v2.d_latent).astype(np.float32)
    logits = mlx_m.sample_logits(noise, num_steps=2)
    assert np.isfinite(np.array(logits)).all()
