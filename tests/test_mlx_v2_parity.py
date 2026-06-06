"""MLX v2 parity tests: MLXDIMBA.from_torch must match PyTorch DIMBA numerically.

Tests the full v2 config:
  conditioning_type="adaln", prediction_type="v", use_head_norm=True,
  latent_norm=True, self_conditioning=True, head_type="attn"

NOTE: use_simple_mamba=False is intentional. SimpleMamba2 has a different state_dict
structure (in_proj+B_proj+C_proj+dt_proj vs in_proj+conv1d+dt_bias+A_log+D) that is
NOT weight-compatible with MLXMamba2Mixer (which mirrors mamba_ssm.Mamba2 / TorchMamba2).
The MLX backend uses the TorchMamba2-compatible format so the PyTorch side must match.

Skipped automatically when mlx is not installed.
"""

import numpy as np
import pytest
import torch

mx = pytest.importorskip("mlx.core")  # skips the whole module without MLX

from dimba.models.diffusion import DIMBA
from dimba.backends.mlx.model import MLXDIMBA

# --------------------------------------------------------------------------- config
# Small enough to run fast; d_latent * expand >= headdim (default 64 in MLXMamba2Mixer).
# d_latent=64, expand=2 -> d_inner=128 >= 64: fine.
_CFG_V2 = dict(
    vocab_size=128,
    d_model=64,
    d_prompt=64,
    num_diffusion_steps=10,
    num_denoiser_layers=2,
    d_latent=64,
    latent_diffusion=True,
    conditioning_type="adaln",
    prediction_type="v",
    use_head_norm=True,
    latent_norm=True,
    self_conditioning=True,
    head_type="attn",
    head_attn_layers=2,
    head_attn_heads=4,
    use_simple_mamba=False,  # TorchMamba2: weight-compatible with MLXMamba2Mixer
    use_weight_tying=True,
    dropout=0.0,
)

# v1 baseline: film/x0/linear/no-norm/no-self-cond (backward-compat check)
_CFG_V1 = dict(
    vocab_size=128,
    d_model=64,
    d_prompt=64,
    num_diffusion_steps=10,
    num_denoiser_layers=2,
    d_latent=64,
    latent_diffusion=True,
    conditioning_type="film",
    prediction_type="x0",
    use_head_norm=False,
    latent_norm=False,
    self_conditioning=False,
    head_type="linear",
    use_simple_mamba=False,
    use_weight_tying=True,
    dropout=0.0,
)

_SEQ_LEN = 8
_BATCH = 2
_SEED = 7


def _build_torch_model(cfg: dict) -> DIMBA:
    torch.manual_seed(_SEED)
    m = DIMBA(**cfg)
    # Randomise weights so we are not testing on zeroed parameters.
    for p in m.parameters():
        torch.nn.init.normal_(p, std=0.05)
    # Ensure logit_scale is a reasonable positive scalar (exp(.) must not overflow).
    if cfg.get("use_head_norm") and hasattr(m.output_head, "logit_scale"):
        with torch.no_grad():
            m.output_head.logit_scale.fill_(0.1)
    m.eval()
    return m


def _fixed_noise(batch: int, seq_len: int, d_latent: int) -> np.ndarray:
    """Deterministic noise array for reproducible cross-backend comparisons."""
    rng = np.random.RandomState(_SEED)
    return rng.randn(batch, seq_len, d_latent).astype(np.float32)


@torch.no_grad()
def _torch_sample_logits(model: DIMBA, noise: np.ndarray, num_steps: int) -> np.ndarray:
    """Run the full DDIM loop on the PyTorch model and return the final logits.

    Mirrors sample_from_model (sampling.py) but returns raw logits instead of
    token IDs so we can compare with the MLX float output.
    """
    device = next(model.parameters()).device
    T = model.num_diffusion_steps
    ns = min(num_steps, T)
    timesteps = torch.linspace(T - 1, 0, ns, device=device).round().long()
    acp = model.get_alphas_cumprod().to(device)

    B, seq_len, d_lat = noise.shape
    x_t = torch.from_numpy(noise).to(device)

    cond = model.conditioning_from_prompt(None, B, device)  # null cond

    x_self_cond = None
    for i in range(ns):
        t_val = timesteps[i]
        t = t_val.expand(B)
        acp_t = acp[t_val]

        x0_hat = model.denoise_to_x0_latent(x_t, t, cond, x_self_cond)
        if model.self_conditioning:
            x_self_cond = x0_hat

        acp_prev = acp[timesteps[i + 1]] if i < ns - 1 else torch.ones((), device=device)
        sqrt_acp_t = acp_t.sqrt()
        sqrt_om_t = (1.0 - acp_t).clamp(min=1e-8).sqrt()
        eps_hat = (x_t - sqrt_acp_t * x0_hat) / sqrt_om_t
        dir_coef = (1.0 - acp_prev).clamp(min=0.0).sqrt()
        x_t = acp_prev.sqrt() * x0_hat + dir_coef * eps_hat

    x_dec = model.decode_latent(x_t)
    logits = model.output_head(x_dec)   # [B, L, V]
    return logits.detach().cpu().numpy()


# --------------------------------------------------------------------------- fixtures

@pytest.fixture(scope="module")
def v2_torch_model():
    return _build_torch_model(_CFG_V2)


@pytest.fixture(scope="module")
def v2_mlx_model(v2_torch_model):
    return MLXDIMBA.from_torch(v2_torch_model)


@pytest.fixture(scope="module")
def v1_torch_model():
    return _build_torch_model(_CFG_V1)


@pytest.fixture(scope="module")
def v1_mlx_model(v1_torch_model):
    return MLXDIMBA.from_torch(v1_torch_model)


# --------------------------------------------------------------------------- v2 parity

class TestV2Parity:
    """Full v2 config: adaln + v-pred + latent-norm + head-norm + self-cond + attn head."""

    def test_from_torch_builds(self, v2_mlx_model):
        assert v2_mlx_model is not None
        assert v2_mlx_model.conditioning_type == "adaln"
        assert v2_mlx_model.prediction_type == "v"
        assert v2_mlx_model.use_head_norm is True
        assert v2_mlx_model.latent_norm is True
        assert v2_mlx_model.self_conditioning is True
        assert v2_mlx_model.head_type == "attn"

    def test_logits_shape(self, v2_mlx_model):
        noise = _fixed_noise(_BATCH, _SEQ_LEN, _CFG_V2["d_latent"])
        logits = np.array(v2_mlx_model.sample_logits(noise, num_steps=_CFG_V2["num_diffusion_steps"]))
        assert logits.shape == (_BATCH, _SEQ_LEN, _CFG_V2["vocab_size"])

    def test_logits_finite(self, v2_mlx_model):
        noise = _fixed_noise(_BATCH, _SEQ_LEN, _CFG_V2["d_latent"])
        logits = np.array(v2_mlx_model.sample_logits(noise, num_steps=_CFG_V2["num_diffusion_steps"]))
        assert np.isfinite(logits).all(), "MLX v2 logits contain NaN/Inf"

    def test_parity_with_torch(self, v2_torch_model, v2_mlx_model):
        """MLX and PyTorch must agree to < 1e-2 max logit delta on the same noise."""
        noise = _fixed_noise(_BATCH, _SEQ_LEN, _CFG_V2["d_latent"])
        ns = _CFG_V2["num_diffusion_steps"]

        # PyTorch reference
        pt_logits = _torch_sample_logits(v2_torch_model, noise, ns)

        # MLX replica
        mx_logits = np.array(v2_mlx_model.sample_logits(noise, num_steps=ns))

        max_delta = np.abs(pt_logits - mx_logits).max()
        print(f"\n[v2 parity] max |Δlogits| = {max_delta:.6f}")
        assert max_delta < 1e-2, (
            f"MLX v2 logit delta too large: {max_delta:.6f} >= 1e-2"
        )

    def test_argmax_agreement_with_torch(self, v2_torch_model, v2_mlx_model):
        """100% argmax-token agreement between MLX and PyTorch."""
        noise = _fixed_noise(_BATCH, _SEQ_LEN, _CFG_V2["d_latent"])
        ns = _CFG_V2["num_diffusion_steps"]

        pt_logits = _torch_sample_logits(v2_torch_model, noise, ns)
        mx_logits = np.array(v2_mlx_model.sample_logits(noise, num_steps=ns))

        pt_tokens = pt_logits.argmax(-1)   # [B, L]
        mx_tokens = mx_logits.argmax(-1)

        agreement = (pt_tokens == mx_tokens).mean()
        print(f"\n[v2 parity] argmax agreement = {agreement:.4f}")
        assert agreement == 1.0, (
            f"MLX v2 argmax agreement only {agreement:.4f}, expected 1.0\n"
            f"mismatch positions:\n{np.where(pt_tokens != mx_tokens)}"
        )


# --------------------------------------------------------------------------- v1 regression

class TestV1Regression:
    """Ensure the v1 (film/x0) path still matches after the v2 refactor."""

    def test_from_torch_builds(self, v1_mlx_model):
        assert v1_mlx_model.conditioning_type == "film"
        assert v1_mlx_model.prediction_type == "x0"
        assert v1_mlx_model.use_head_norm is False
        assert v1_mlx_model.head_type == "linear"

    def test_parity_with_torch(self, v1_torch_model, v1_mlx_model):
        noise = _fixed_noise(_BATCH, _SEQ_LEN, _CFG_V1["d_latent"])
        ns = _CFG_V1["num_diffusion_steps"]

        pt_logits = _torch_sample_logits(v1_torch_model, noise, ns)
        mx_logits = np.array(v1_mlx_model.sample_logits(noise, num_steps=ns))

        max_delta = np.abs(pt_logits - mx_logits).max()
        print(f"\n[v1 parity] max |Δlogits| = {max_delta:.6f}")
        assert max_delta < 1e-2, (
            f"MLX v1 regression: logit delta {max_delta:.6f} >= 1e-2"
        )

    def test_argmax_agreement_with_torch(self, v1_torch_model, v1_mlx_model):
        noise = _fixed_noise(_BATCH, _SEQ_LEN, _CFG_V1["d_latent"])
        ns = _CFG_V1["num_diffusion_steps"]

        pt_logits = _torch_sample_logits(v1_torch_model, noise, ns)
        mx_logits = np.array(v1_mlx_model.sample_logits(noise, num_steps=ns))

        agreement = (pt_logits.argmax(-1) == mx_logits.argmax(-1)).mean()
        print(f"\n[v1 parity] argmax agreement = {agreement:.4f}")
        assert agreement == 1.0, f"MLX v1 argmax agreement only {agreement:.4f}"


# --------------------------------------------------------------------------- guard for additive cond

def test_additive_conditioning_raises():
    """Additive conditioning is not yet supported; from_torch must raise NotImplementedError."""
    torch.manual_seed(0)
    m = DIMBA(
        vocab_size=64, d_model=64, d_prompt=64, num_diffusion_steps=4,
        num_denoiser_layers=1, d_latent=64, latent_diffusion=True,
        conditioning_type="additive", dropout=0.0,
    )
    m.eval()
    with pytest.raises(NotImplementedError):
        MLXDIMBA.from_torch(m)
