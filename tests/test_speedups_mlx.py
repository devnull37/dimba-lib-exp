"""MLX-side speedup tests: half-precision weights and threshold commits.

Skipped automatically when mlx is not installed (Apple Silicon only).
"""

import numpy as np
import pytest
import torch

mx = pytest.importorskip("mlx.core")  # skips the whole module without MLX

from dimba.models.diffusion import DIMBA
from dimba.backends.mlx.model import MLXDIMBA

_MASK_ID = 63  # last vocab slot plays [MASK]

# Same config family as hr-diffuse-1-nano (mirrors test_mlx_masked_parity).
_CFG = dict(
    vocab_size=64,
    d_model=128,
    d_prompt=96,
    num_diffusion_steps=100,
    num_denoiser_layers=3,
    d_state=16,
    expand=2,
    conditioning_type="adaln",
    prediction_type="x0",
    latent_diffusion=False,
    use_simple_mamba=False,
    use_weight_tying=True,
    block_ffn=True,
    ffn_type="swiglu",
    ffn_mult=4,
    dropout=0.0,
)


@pytest.fixture(scope="module")
def mlx_pair():
    torch.manual_seed(31)
    torch_model = DIMBA(**_CFG).eval()
    return (
        MLXDIMBA.from_torch(torch_model),
        MLXDIMBA.from_torch(torch_model, dtype=mx.float16),
    )


def test_fp16_weights_are_half_and_islands_stay_fp32(mlx_pair) -> None:
    _, fp16 = mlx_pair
    assert fp16.p["token_embed_w"].dtype == mx.float16
    assert fp16.p["sqrt_acp"].dtype == mx.float32  # schedule island
    mixer = fp16.mixers_fwd[0]
    assert mixer.in_proj.weight.dtype == mx.float16
    assert mixer.A_log.dtype == mx.float32  # time-constant island


def test_fp16_argmax_agreement(mlx_pair) -> None:
    fp32, fp16 = mlx_pair
    rng = np.random.default_rng(31)
    ids = rng.integers(0, _MASK_ID, size=(2, 12))
    ids[:, 6:] = _MASK_ID  # mask the response half

    ref = np.array(fp32.predict_token_logits(ids, 0.5))
    half = np.array(fp16.predict_token_logits(ids, 0.5))
    assert np.isfinite(half).all()
    agreement = (ref.argmax(-1) == half.argmax(-1)).mean()
    assert agreement >= 0.95, f"fp16 argmax agreement {agreement:.3f}"


def test_fp16_sample_masked_finishes(mlx_pair) -> None:
    _, fp16 = mlx_pair
    prompt = np.array([[1, 2, 3, 4]], dtype=np.int32)
    out = fp16.sample_masked(prompt, _MASK_ID, gen_len=6, steps=8, seed=3)
    assert out.shape == (1, 10)
    assert np.isfinite(out).all()


def test_commit_threshold_above_one_is_identical(mlx_pair) -> None:
    fp32, _ = mlx_pair
    prompt = np.array([[1, 2, 3, 4]], dtype=np.int32)
    kwargs = dict(gen_len=6, steps=8, seed=5)
    baseline = fp32.sample_masked(prompt, _MASK_ID, **kwargs)
    thresholded = fp32.sample_masked(
        prompt, _MASK_ID, commit_threshold=1.5, **kwargs
    )
    assert (baseline == thresholded).all()


def test_commit_threshold_zero_exits_after_one_step(mlx_pair) -> None:
    fp32, _ = mlx_pair
    prompt = np.array([[1, 2, 3, 4]], dtype=np.int32)
    steps_seen = []
    fp32.sample_masked(
        prompt, _MASK_ID, gen_len=6, steps=16, seed=5, commit_threshold=0.0,
        on_step=lambda ids, still, s, total: steps_seen.append(s),
    )
    assert len(steps_seen) == 1
