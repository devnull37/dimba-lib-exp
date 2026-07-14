"""MLX masked-diffusion parity and device-residency tests.

The MLX backend gained ``MLXDIMBA.predict_token_logits`` for the masked discrete
diffusion track (hr-diffuse-1-nano): embed -> encode -> denoise at t -> decode ->
token logits, mirroring ``DIMBA.predict_token_logits``. The production sampler
also keeps its entire MaskGIT trajectory on MLX and projects only unresolved
positions. Batches larger than 2 are chunked into two-row calls (mlx 0.29.3
fused-graph NaN workaround), which the B=4 case exercises.

Skipped automatically when mlx is not installed.
"""

import numpy as np
import pytest
import torch

mx = pytest.importorskip("mlx.core")  # skips the whole module without MLX

from dimba.models.diffusion import DIMBA
from dimba.backends.mlx.model import MLXDIMBA
import dimba.backends.mlx.model as mlx_model_module

_SEED = 11
_SEQ_LEN = 12
_MASK_ID = 63  # last vocab slot plays [MASK]

# Same config family as hr-diffuse-1-nano: adaln + block_ffn/swiglu + x0 +
# weight tying, embedding-space diffusion.
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
    use_simple_mamba=False,       # TorchMamba2: weight-compatible with MLXMamba2Mixer
    use_weight_tying=True,
    block_ffn=True,
    ffn_type="swiglu",
    ffn_mult=4,
    use_flow_matching=True,
    flow_logit_normal=True,
    dropout=0.0,
)


@pytest.fixture(scope="module")
def models():
    torch.manual_seed(_SEED)
    m = DIMBA(**_CFG).eval()
    return m, MLXDIMBA.from_torch(m)


def _ids(batch):
    rng = np.random.default_rng(_SEED)
    ids = rng.integers(0, _MASK_ID, size=(batch, _SEQ_LEN))
    ids[:, _SEQ_LEN // 2:] = _MASK_ID  # masked response region
    return ids


@pytest.mark.parametrize("batch", [1, 2, 4])  # 4 exercises the >2 chunking path
@pytest.mark.parametrize("t", [1.0, 0.5, 0.03])
def test_predict_token_logits_parity(models, batch, t):
    torch_model, mlx_model = models
    ids = _ids(batch)
    with torch.no_grad():
        ref = torch_model.predict_token_logits(torch.tensor(ids), t).numpy()
    out = np.array(mlx_model.predict_token_logits(ids, t))
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out, ref, atol=1e-4)
    assert (out.argmax(-1) == ref.argmax(-1)).all()


def test_selected_feature_cfg_matches_torch(models):
    torch_model, mlx_model = models
    ids = _ids(2)
    prompt_len = _SEQ_LEN // 2
    positions = np.array([[6, 8, 10], [7, 9, 11]], dtype=np.int32)
    uncond = ids.copy()
    uncond[:, :prompt_len] = _MASK_ID
    guidance = 1.7

    with torch.no_grad():
        features = torch_model.predict_token_features(
            torch.tensor(np.concatenate([ids, uncond])), 0.5
        )
        cond, null = features.chunk(2)
        ref = torch_model.project_token_features(
            null + guidance * (cond - null), torch.tensor(positions)
        ).numpy()

    out = np.array(
        mlx_model.guided_token_logits(
            ids, _MASK_ID, prompt_len, 0.5, guidance, positions
        )
    )
    np.testing.assert_allclose(out, ref, atol=1e-4)
    assert (out.argmax(-1) == ref.argmax(-1)).all()


def test_one_step_resident_sample_matches_torch_argmax(models):
    torch_model, mlx_model = models
    prompt_len = _SEQ_LEN // 2
    prompt = _ids(2)[:, :prompt_len]
    gen_len = 4
    ids = np.concatenate(
        [prompt, np.full((len(prompt), gen_len), _MASK_ID, dtype=np.int32)], axis=1
    )
    uncond = ids.copy()
    uncond[:, :prompt_len] = _MASK_ID

    with torch.no_grad():
        features = torch_model.predict_token_features(
            torch.tensor(np.concatenate([ids, uncond])), gen_len / ids.shape[1]
        )
        cond, null = features.chunk(2)
        positions = torch.arange(prompt_len, ids.shape[1]).expand(len(prompt), -1)
        logits = torch_model.project_token_features(
            null + 1.7 * (cond - null), positions
        )
    expected = ids.copy()
    expected[:, prompt_len:] = logits.argmax(-1).numpy()

    out = mlx_model.sample_masked(
        prompt,
        _MASK_ID,
        gen_len=gen_len,
        steps=1,
        top_k=1,
        guidance=1.7,
        seed=3,
    )
    np.testing.assert_array_equal(out, expected)


def test_resident_gap_score_matches_torch(models):
    from scripts.generate import gap_score

    torch_model, mlx_model = models
    prompt_len = _SEQ_LEN // 2
    ids = _ids(3)
    rng = np.random.default_rng(_SEED + 1)
    ids[:, prompt_len:] = rng.integers(3, _MASK_ID, size=ids[:, prompt_len:].shape)
    ids[0, -1] = 2

    expected = gap_score(
        torch_model, _MASK_ID, 2, torch.tensor(ids), prompt_len, K=4
    ).numpy()
    actual = mlx_model.masked_gap_score(ids, _MASK_ID, 2, prompt_len, groups=4)

    np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=2e-4)


def test_multistep_sampler_converts_only_final_ids(monkeypatch):
    sampler = object.__new__(MLXDIMBA)
    sampler.vocab_size = 8
    seen = []

    def guided(ids, mask_id, prompt_len, t, guidance, positions):
        assert isinstance(ids, mx.array)
        assert isinstance(positions, mx.array)
        seen.append(positions.shape[1])
        vocab = mx.arange(sampler.vocab_size)[None, None, :]
        targets = (positions % (sampler.vocab_size - 1))[..., None]
        return -mx.abs(vocab - targets).astype(mx.float32)

    sampler.guided_token_logits = guided
    original_array = mlx_model_module.np.array
    conversions = []

    def counted_array(*args, **kwargs):
        conversions.append(type(args[0]))
        return original_array(*args, **kwargs)

    monkeypatch.setattr(mlx_model_module.np, "array", counted_array)
    out = sampler.sample_masked(
        np.asarray([[1, 2]], dtype=np.int32),
        7,
        gen_len=4,
        steps=5,
        top_k=3,
        seed=9,
    )
    conversion_count = len(conversions)

    assert out.shape == (1, 6)
    assert seen == sorted(seen, reverse=True)
    assert conversion_count == 1
