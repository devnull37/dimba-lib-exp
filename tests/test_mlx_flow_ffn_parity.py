"""MLX flow-matching + per-block FFN (block_ffn) parity tests.

The MLX backend gained two new capabilities required by the current v2 checkpoints
(``use_flow_matching=True`` with ``block_ffn=True`` / SwiGLU):

  1. ``block_ffn``: the inherited per-block FFN sub-layer (6-param adaln path:
     norm->modulate->mix->gated residual, then norm2->modulate2->FFN->gated residual).
  2. a flow-matching Euler/Heun ODE sampler (``dx/dt = v_theta`` from t=1 -> t=0).

These tests build a SMALL torch DIMBA in the same config family as the real base
checkpoint (adaln + block_ffn=swiglu + flow), convert it with
``MLXDIMBA.from_torch``, and assert the per-block denoiser forward (the x0 / velocity
prediction) matches the torch reference on the same fixed input.

Skipped automatically when mlx is not installed.
"""

import numpy as np
import pytest
import torch

mx = pytest.importorskip("mlx.core")  # skips the whole module without MLX

from dimba.models.diffusion import DIMBA
from dimba.backends.mlx.model import MLXDIMBA

_SEED = 11
_BATCH = 2
_SEQ_LEN = 8

# Small DIMBA mirroring the real base config family:
#   conditioning_type="adaln", block_ffn=True, ffn_type="swiglu",
#   use_flow_matching=True, latent_diffusion=False (embedding-space).
_CFG_FLOW_FFN = dict(
    vocab_size=64,
    d_model=128,
    d_prompt=96,                  # != d_model so cond_dim != d_latent is exercised
    num_diffusion_steps=100,
    num_denoiser_layers=3,
    d_state=16,
    expand=2,
    conditioning_type="adaln",
    prediction_type="x0",
    latent_diffusion=False,       # embedding-space (like the base checkpoint)
    use_simple_mamba=False,       # TorchMamba2: weight-compatible with MLXMamba2Mixer
    use_weight_tying=True,
    block_ffn=True,
    ffn_type="swiglu",
    ffn_mult=4,
    use_flow_matching=True,
    flow_logit_normal=True,
    dropout=0.0,
)

# mlp-FFN variant (block_ffn with the GPT-style ff2(gelu(ff1(x))) shape).
_CFG_FLOW_MLP = dict(_CFG_FLOW_FFN, ffn_type="mlp")


def _build_torch_model(cfg: dict) -> DIMBA:
    torch.manual_seed(_SEED)
    m = DIMBA(**cfg)
    # Randomise weights so we aren't testing on the zero-init adaln modulation
    # (which would make every block an identity — a trivial pass).
    for p in m.parameters():
        torch.nn.init.normal_(p, std=0.05)
    m.eval()
    return m


def _fixed_noise(d_latent: int) -> np.ndarray:
    rng = np.random.RandomState(_SEED)
    return rng.randn(_BATCH, _SEQ_LEN, d_latent).astype(np.float32)


@torch.no_grad()
def _torch_denoiser_x0(model: DIMBA, x_t: np.ndarray, t_cont: float) -> np.ndarray:
    """Torch reference: denoiser x0 prediction at continuous flow-time ``t_cont``.

    Mirrors ``denoise_flow`` (the per-step model call inside sample_from_model_flow):
    null conditioning, continuous t mapped to the discrete timestep-embedding index.
    """
    device = next(model.parameters()).device
    xt = torch.from_numpy(x_t).to(device)
    cond = model.conditioning_from_prompt(None, _BATCH, device)  # null cond
    t = torch.full((_BATCH,), float(t_cont), device=device)
    x0 = model.denoise_flow(xt, t, cond, None)  # raw == x0 for flow matching
    return x0.detach().cpu().numpy()


def _mlx_denoiser_x0(mlx_model: MLXDIMBA, x_t: np.ndarray, t_cont: float) -> np.ndarray:
    """MLX denoiser x0 prediction at continuous flow-time ``t_cont`` (null cond)."""
    B = x_t.shape[0]
    null = mlx_model.p["null_cond"].reshape(1, 1, -1)
    null = mlx_model._project_cond(null)
    cond = mx.broadcast_to(null, (B, 1, mlx_model.cond_dim))
    t_idx = mlx_model._flow_t_idx(t_cont)
    x0 = mlx_model._denoiser(mx.array(x_t), cond, t_idx, None)
    mx.eval(x0)
    return np.array(x0)


@pytest.fixture(scope="module")
def swiglu_models():
    tm = _build_torch_model(_CFG_FLOW_FFN)
    mm = MLXDIMBA.from_torch(tm)
    return tm, mm


@pytest.fixture(scope="module")
def mlp_models():
    tm = _build_torch_model(_CFG_FLOW_MLP)
    mm = MLXDIMBA.from_torch(tm)
    return tm, mm


class TestFlowFFNParity:
    """Denoiser (velocity / x0) forward parity for the flow + block_ffn config."""

    def test_from_torch_builds(self, swiglu_models):
        _, mm = swiglu_models
        assert mm.use_flow_matching is True
        assert mm.block_ffn is True
        assert mm.ffn_type == "swiglu"
        assert mm.conditioning_type == "adaln"
        assert mm.cond_dim == _CFG_FLOW_FFN["d_prompt"]  # 96 != d_latent 128

    @pytest.mark.parametrize("t_cont", [0.99, 0.75, 0.5, 0.25, 0.05])
    def test_x0_parity_swiglu(self, swiglu_models, t_cont):
        tm, mm = swiglu_models
        x_t = _fixed_noise(_CFG_FLOW_FFN["d_model"])
        pt = _torch_denoiser_x0(tm, x_t, t_cont)
        mxo = _mlx_denoiser_x0(mm, x_t, t_cont)
        assert np.isfinite(mxo).all(), "MLX x0 has NaN/Inf"
        max_delta = np.abs(pt - mxo).max()
        print(f"\n[flow+swiglu t={t_cont}] max |Δx0| = {max_delta:.6e}")
        assert max_delta < 1e-2, f"x0 delta too large: {max_delta:.6e} >= 1e-2"

    @pytest.mark.parametrize("t_cont", [0.9, 0.5, 0.1])
    def test_x0_parity_mlp(self, mlp_models, t_cont):
        tm, mm = mlp_models
        x_t = _fixed_noise(_CFG_FLOW_MLP["d_model"])
        pt = _torch_denoiser_x0(tm, x_t, t_cont)
        mxo = _mlx_denoiser_x0(mm, x_t, t_cont)
        max_delta = np.abs(pt - mxo).max()
        print(f"\n[flow+mlp t={t_cont}] max |Δx0| = {max_delta:.6e}")
        assert max_delta < 1e-2, f"mlp x0 delta too large: {max_delta:.6e} >= 1e-2"

    def test_velocity_parity_full_step(self, swiglu_models):
        """Parity of the derived velocity v=(x_t-x0)/t, the actual ODE rhs."""
        tm, mm = swiglu_models
        x_t = _fixed_noise(_CFG_FLOW_FFN["d_model"])
        t_cont = 0.5
        pt_x0 = _torch_denoiser_x0(tm, x_t, t_cont)
        mx_x0 = _mlx_denoiser_x0(mm, x_t, t_cont)
        pt_v = (x_t - pt_x0) / t_cont
        mx_v = (x_t - mx_x0) / t_cont
        max_delta = np.abs(pt_v - mx_v).max()
        print(f"\n[flow velocity t=0.5] max |Δv| = {max_delta:.6e}")
        assert max_delta < 1e-2, f"velocity delta too large: {max_delta:.6e}"

    def test_flow_sampler_runs_and_finite(self, swiglu_models):
        """End-to-end MLX Euler flow sampler produces finite logits of right shape."""
        _, mm = swiglu_models
        noise = _fixed_noise(_CFG_FLOW_FFN["d_model"])
        logits = np.array(mm.sample_logits_flow(noise, num_steps=8, sampler="euler"))
        assert logits.shape == (_BATCH, _SEQ_LEN, _CFG_FLOW_FFN["vocab_size"])
        assert np.isfinite(logits).all()

    def test_heun_sampler_runs(self, swiglu_models):
        _, mm = swiglu_models
        noise = _fixed_noise(_CFG_FLOW_FFN["d_model"])
        logits = np.array(mm.sample_logits_flow(noise, num_steps=6, sampler="heun"))
        assert logits.shape == (_BATCH, _SEQ_LEN, _CFG_FLOW_FFN["vocab_size"])
        assert np.isfinite(logits).all()
