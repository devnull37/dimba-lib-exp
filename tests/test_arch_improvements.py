"""Tests for RMSNorm, AdaLN-Zero 6-param, and Flow Matching."""
import math
import pytest
import torch
import torch.nn as nn


# ── RMSNorm ───────────────────────────────────────────────────────────────────

def test_rmsnorm_output_shape():
    from dimba.models.denoiser import RMSNorm
    rn = RMSNorm(32)
    x = torch.randn(2, 10, 32)
    assert rn(x).shape == x.shape


def test_rmsnorm_no_bias():
    from dimba.models.denoiser import RMSNorm
    rn = RMSNorm(16)
    param_names = [n for n, _ in rn.named_parameters()]
    assert "bias" not in param_names
    assert "weight" in param_names


def test_rmsnorm_unit_rms():
    """Output should have RMS ≈ weight (ones init → RMS ≈ 1)."""
    from dimba.models.denoiser import RMSNorm
    rn = RMSNorm(64)
    x = torch.randn(4, 8, 64) * 5.0  # large magnitude input
    y = rn(x)
    rms = y.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=0.05)


def test_rmsnorm_used_in_mamba2block():
    """Mamba2Block should use RMSNorm, not LayerNorm."""
    from dimba.models.denoiser import Mamba2Block, RMSNorm
    block = Mamba2Block(d_model=64, use_simple_mamba=True)
    assert isinstance(block.norm, RMSNorm), f"Expected RMSNorm, got {type(block.norm)}"


def test_rmsnorm_used_in_block_ffn():
    from dimba.models.denoiser import Mamba2Block, RMSNorm
    block = Mamba2Block(d_model=64, use_simple_mamba=True, block_ffn=True)
    assert isinstance(block.norm2, RMSNorm)


# ── AdaLN-Zero 6-param ────────────────────────────────────────────────────────

def test_adaln_zero_3param_shapes():
    from dimba.models.embeddings import AdaLNZeroConditioning
    mod = AdaLNZeroConditioning(cond_dim=32, target_dim=16, has_ffn=False)
    cond = torch.randn(2, 1, 32)
    out = mod(cond)
    assert len(out) == 3
    for t in out:
        assert t.shape == (2, 1, 16)


def test_adaln_zero_6param_shapes():
    from dimba.models.embeddings import AdaLNZeroConditioning
    mod = AdaLNZeroConditioning(cond_dim=32, target_dim=16, has_ffn=True)
    cond = torch.randn(2, 1, 32)
    out = mod(cond)
    assert len(out) == 6
    for t in out:
        assert t.shape == (2, 1, 16)


def test_adaln_zero_zero_init():
    """All outputs should be ~0 at init (zero-initialized linear)."""
    from dimba.models.embeddings import AdaLNZeroConditioning
    for has_ffn in (False, True):
        mod = AdaLNZeroConditioning(64, 32, has_ffn=has_ffn)
        cond = torch.randn(4, 1, 64)
        out = mod(cond)
        for t in out:
            assert t.abs().max().item() < 1e-6, "AdaLN output not zero at init"


def test_adaln_zero_block_identity_at_init():
    """With zero modulation, forward_adaln should equal plain forward."""
    from dimba.models.denoiser import Mamba2Block
    block = Mamba2Block(d_model=64, use_simple_mamba=True)
    x = torch.randn(2, 8, 64)
    scale = torch.zeros(2, 1, 64)
    shift = torch.zeros(2, 1, 64)
    gate  = torch.zeros(2, 1, 64)
    # gate=0 → x + 0*mix(h) = x  (identity)
    out_adaln = block.forward_adaln(x, scale, shift, gate)
    assert torch.allclose(out_adaln, x, atol=1e-6)


def test_adaln_6param_block_forward_ffn():
    """forward_adaln with 6 params should differ from 3-param when gate2 ≠ 0."""
    from dimba.models.denoiser import Mamba2Block
    block = Mamba2Block(d_model=64, use_simple_mamba=True, block_ffn=True)
    x = torch.randn(2, 8, 64)
    zeros = torch.zeros(2, 1, 64)
    ones  = torch.ones(2, 1, 64) * 0.1
    # 3-param (no FFN conditioning)
    out3 = block.forward_adaln(x, zeros, zeros, zeros)
    # 6-param (with FFN conditioning, gate2 nonzero)
    out6 = block.forward_adaln(x, zeros, zeros, zeros, zeros, zeros, ones)
    # They should differ because gate2=0.1 ≠ 0
    assert not torch.allclose(out3, out6)


def test_denoiser_adaln_default():
    """Mamba2Denoiser should default to adaln conditioning."""
    from dimba.models.denoiser import Mamba2Denoiser
    d = Mamba2Denoiser(d_model=64, num_layers=2, cond_dim=64, time_embed_dim=64,
                       use_simple_mamba=True)
    assert d.conditioning_type == "adaln"


def test_denoiser_adaln_6param_wired():
    """When block_ffn=True, conditioning layers should output 6 params."""
    from dimba.models.denoiser import Mamba2Denoiser
    from dimba.models.embeddings import AdaLNZeroConditioning
    d = Mamba2Denoiser(d_model=64, num_layers=2, cond_dim=64, time_embed_dim=64,
                       use_simple_mamba=True, block_ffn=True, conditioning_type="adaln")
    for cond_layer in d.conditioning:
        assert isinstance(cond_layer, AdaLNZeroConditioning)
        assert cond_layer.has_ffn is True


def test_denoiser_adaln_forward():
    from dimba.models.denoiser import Mamba2Denoiser
    d = Mamba2Denoiser(d_model=64, num_layers=2, cond_dim=64, time_embed_dim=64,
                       use_simple_mamba=True, block_ffn=True, conditioning_type="adaln")
    x = torch.randn(2, 10, 64)
    cond = torch.randn(2, 1, 64)
    t_emb = torch.randn(2, 64)
    out = d(x, cond, t_emb)
    assert out.shape == x.shape
    assert out.isfinite().all()


# ── Flow Matching Schedule ────────────────────────────────────────────────────

def test_flow_schedule_forward_process_endpoints():
    """x_t at t=0 should be x0; at t=1 should be noise."""
    from dimba.diffusion.schedules import FlowMatchingSchedule
    sched = FlowMatchingSchedule()
    x0 = torch.randn(2, 5, 8)
    noise = torch.randn_like(x0)
    t0 = torch.zeros(2)
    t1 = torch.ones(2)
    assert torch.allclose(sched.forward_process(x0, noise, t0), x0, atol=1e-6)
    assert torch.allclose(sched.forward_process(x0, noise, t1), noise, atol=1e-6)


def test_flow_schedule_velocity_target_shape():
    from dimba.diffusion.schedules import FlowMatchingSchedule
    sched = FlowMatchingSchedule()
    x0 = torch.randn(3, 7, 16)
    noise = torch.randn_like(x0)
    v = sched.velocity_target(x0, noise)
    assert v.shape == x0.shape
    assert torch.allclose(v, noise - x0)


def test_flow_schedule_x0_recovery():
    """x0 recovered from x_t and velocity should equal original x0."""
    from dimba.diffusion.schedules import FlowMatchingSchedule
    sched = FlowMatchingSchedule()
    x0 = torch.randn(2, 6, 8)
    noise = torch.randn_like(x0)
    t = torch.tensor([0.3, 0.7])
    x_t = sched.forward_process(x0, noise, t)
    v = sched.velocity_target(x0, noise)
    x0_rec = sched.x0_from_xt_and_velocity(x_t, v, t)
    assert torch.allclose(x0_rec, x0, atol=1e-5)


def test_flow_schedule_logit_normal_sampling():
    from dimba.diffusion.schedules import FlowMatchingSchedule
    sched = FlowMatchingSchedule(logit_normal_sampling=True)
    t = sched.sample_timesteps(1000, torch.device("cpu"))
    assert t.shape == (1000,)
    assert (t > 0).all() and (t < 1).all()
    # Should be concentrated around 0.5 (logit-normal with mean=0)
    assert 0.4 < t.mean().item() < 0.6


def test_flow_schedule_uniform_sampling():
    from dimba.diffusion.schedules import FlowMatchingSchedule
    sched = FlowMatchingSchedule(logit_normal_sampling=False)
    t = sched.sample_timesteps(500, torch.device("cpu"))
    assert (t > 0).all() and (t < 1).all()


# ── Flow Matching in DIMBA ────────────────────────────────────────────────────

@pytest.fixture
def tiny_flow_model():
    from dimba.models.diffusion import DIMBA
    return DIMBA(
        vocab_size=100, d_model=64, d_prompt=64,
        num_diffusion_steps=100, num_denoiser_layers=2,
        use_simple_mamba=True, use_flow_matching=True,
        conditioning_type="adaln",
    )


def test_flow_model_has_flow_schedule(tiny_flow_model):
    from dimba.diffusion.schedules import FlowMatchingSchedule
    assert tiny_flow_model.use_flow_matching is True
    assert isinstance(tiny_flow_model.flow_schedule, FlowMatchingSchedule)


def test_flow_model_forward_finite(tiny_flow_model):
    model = tiny_flow_model
    ids = torch.randint(0, 100, (2, 16))
    t = torch.randint(0, 100, (2,))
    x_pred, noise, info = model(ids, t)
    assert x_pred.isfinite().all()
    assert noise.isfinite().all()


def test_flow_model_denoise_flow(tiny_flow_model):
    model = tiny_flow_model
    model.eval()
    x_t = torch.randn(2, 16, model.d_latent)
    t_cont = torch.tensor([0.3, 0.7])
    ids = torch.randint(0, 100, (2, 8))
    cond = model.conditioning_from_prompt(ids, 2, x_t.device)
    x0_hat = model.denoise_flow(x_t, t_cont, cond)
    assert x0_hat.shape == x_t.shape
    assert x0_hat.isfinite().all()


def test_flow_sampler_runs(tiny_flow_model):
    from dimba.diffusion.sampling import sample_from_model_flow
    model = tiny_flow_model
    model.eval()
    prompt = torch.randint(0, 100, (1, 4))
    out = sample_from_model_flow(model, prompt, seq_len=8, num_steps=4, sampler="euler")
    assert out.shape == (1, 8)


def test_heun_sampler_runs(tiny_flow_model):
    from dimba.diffusion.sampling import sample_from_model_flow
    model = tiny_flow_model
    model.eval()
    out = sample_from_model_flow(tiny_flow_model, None, seq_len=6, num_steps=4, sampler="heun")
    assert out.shape == (1, 6)


def test_flow_config_round_trip():
    """DIMBA(**model.config) should produce an identical architecture."""
    from dimba.models.diffusion import DIMBA
    m = DIMBA(vocab_size=80, d_model=64, d_prompt=64, num_denoiser_layers=2,
              use_simple_mamba=True, use_flow_matching=True)
    m2 = DIMBA(**m.config)
    assert m2.use_flow_matching is True
