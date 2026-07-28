"""Parity tests for the general-speedup pass.

Covers the CUDA-graph-ready feature fast path in ``scripts/generate.py``, the
confidence-threshold commit override, and the ``GraphedFn`` eager fallback.
CUDA-graph capture itself requires CUDA; this file exercises the portable
fallback and the graph-ready feature boundary.
"""

import pytest
import torch

from dimba.models.diffusion import DIMBA
from dimba.utils.cuda_graphs import GraphedFn
from scripts.generate import _make_cfg_features, generate, guided_logits


def _tiny_model(**overrides) -> DIMBA:
    cfg = dict(
        vocab_size=31,
        d_model=16,
        d_prompt=16,
        num_diffusion_steps=8,
        num_denoiser_layers=1,
        d_state=4,
        d_conv=2,
        dropout=0.0,
        use_simple_mamba=True,
    )
    cfg.update(overrides)
    return DIMBA(**cfg).eval()


# --------------------------------------------------------------------------- #
# Feature-space CFG fast path (the CUDA-graph capture target).
# --------------------------------------------------------------------------- #
@torch.inference_mode()
@pytest.mark.parametrize("guidance", [0.0, 1.0, 2.0])
def test_cfg_features_matches_guided_logits(guidance) -> None:
    torch.manual_seed(21)
    model = _tiny_model()
    mask_id, prompt_len = 30, 3
    ids = torch.randint(3, 29, (2, 9))
    ids[:, prompt_len:] = mask_id
    positions = torch.arange(prompt_len, 9).unsqueeze(0).expand(2, -1)
    t = 0.7

    expected = guided_logits(model, mask_id, ids, prompt_len, t, guidance, positions)
    features = _make_cfg_features(model, mask_id, prompt_len, guidance)(
        ids, torch.tensor(t)
    )
    actual = model.project_token_features(features, positions=positions).float()

    torch.testing.assert_close(actual, expected)


def test_generate_matches_logits_only_model() -> None:
    """The feature fast path must sample the same tokens as the generic
    predict_token_logits dispatch (logit- vs feature-space CFG combination is
    algebraically identical for the linear head)."""
    torch.manual_seed(22)
    model = _tiny_model()

    class LogitsOnly:
        predict_token_logits = staticmethod(model.predict_token_logits)

    prompt = torch.randint(3, 29, (2, 4))
    kwargs = dict(gen_len=6, steps=8, guidance=2.0, use_compile=False, seed=7)
    fast = generate(model, 30, prompt, **kwargs)
    generic = generate(LogitsOnly(), 30, prompt, **kwargs)
    assert torch.equal(fast, generic)


# --------------------------------------------------------------------------- #
# Confidence-threshold commits.
# --------------------------------------------------------------------------- #
def test_commit_threshold_above_one_is_identical() -> None:
    """tau >= 1 never overrides the cosine schedule: identical trajectory."""
    torch.manual_seed(23)
    model = _tiny_model()
    prompt = torch.randint(3, 29, (2, 4))
    kwargs = dict(gen_len=6, steps=8, guidance=2.0, use_compile=False, seed=9)
    baseline = generate(model, 30, prompt, **kwargs)
    thresholded = generate(model, 30, prompt, commit_threshold=1.5, **kwargs)
    assert torch.equal(baseline, thresholded)


def test_commit_threshold_zero_commits_everything_immediately() -> None:
    torch.manual_seed(24)
    model = _tiny_model()
    prompt = torch.randint(3, 29, (1, 4))
    steps_seen, still_states = [], []
    generate(
        model, 30, prompt, gen_len=6, steps=16, guidance=2.0,
        use_compile=False, seed=9, commit_threshold=0.0,
        on_step=lambda ids, still, s, steps: (
            steps_seen.append(s), still_states.append(still.clone())
        ),
    )
    assert len(steps_seen) == 1  # every confidence > 0 -> all committed, exit
    assert not bool(still_states[-1].any())  # nothing left masked


# --------------------------------------------------------------------------- #
# Adaptive CFG truncation.
# --------------------------------------------------------------------------- #
def test_cfg_drop_switches_to_single_row_forwards() -> None:
    torch.manual_seed(25)
    model = _tiny_model()
    calls = []
    original = model.predict_token_features

    def recording(input_ids, t):
        calls.append(input_ids.shape[0])
        return original(input_ids, t)

    model.predict_token_features = recording
    prompt = torch.randint(3, 29, (1, 4))
    generate(model, 30, prompt, gen_len=6, steps=8, guidance=2.0,
             use_compile=False, seed=11, cfg_drop=0.5)
    assert calls[0] == 2  # paired CFG rows while mostly masked
    assert calls[-1] == 1  # single conditional row after the drop point
    model.predict_token_features = original


# --------------------------------------------------------------------------- #
# GraphedFn fallback.
# --------------------------------------------------------------------------- #
def test_graphed_fn_is_eager_without_cuda() -> None:
    fn = GraphedFn(lambda x, t: x * 2.0 + t)
    x, t = torch.randn(3, 4), torch.tensor(1.0)
    torch.testing.assert_close(fn(x, t), x * 2.0 + t)
    # Second call with the same shapes must not be a stale static buffer.
    y = torch.randn(3, 4)
    torch.testing.assert_close(fn(y, t), y * 2.0 + t)
