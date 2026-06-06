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


# ---- MLX v2 guard ----
def test_mlx_guard_rejects_v2():
    import pytest
    pytest.importorskip("mlx.core")
    from dimba.backends.mlx.model import MLXDIMBA
    v2 = _tiny(conditioning_type="adaln", prediction_type="v", use_head_norm=True)
    with pytest.raises(NotImplementedError):
        MLXDIMBA.from_torch(v2)
