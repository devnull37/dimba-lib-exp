"""Parity tests for the general-speedup pass.

Covers the CUDA-graph-ready feature fast path in ``scripts/generate.py``, the
confidence-threshold commit override, the chunked linear-CE that avoids
retaining ``[tokens, vocab]`` logits, and the ``GraphedFn`` eager fallback.
The CUDA-graph capture itself only runs on CUDA and is exercised by
``scripts/benchmark_h100.py``.
"""

import pytest
import torch

from dimba.models.diffusion import DIMBA
from dimba.training.fused_ce import output_head_cross_entropy
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
# Chunked linear cross-entropy.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "head_overrides",
    [dict(), dict(use_weight_tying=True, use_head_norm=True)],
    ids=["linear-head", "tied-normed-head"],
)
def test_chunked_ce_matches_unchunked_values_and_grads(head_overrides) -> None:
    torch.manual_seed(26)
    model = _tiny_model(**head_overrides)
    n_tokens, d_model = 64, 16
    targets = torch.randint(0, 31, (n_tokens,))

    def run(chunk_tokens):
        model.zero_grad(set_to_none=True)
        features = torch.randn(
            n_tokens, d_model, generator=torch.Generator().manual_seed(5)
        ).requires_grad_()
        prepared = model.output_head.prepare_features(
            features, model.token_embed.get_weight()
        )
        ce = output_head_cross_entropy(
            model, prepared, targets, prepared=True, reduction="none",
            chunk_tokens=chunk_tokens,
        )
        # Non-uniform per-token weights: the case Liger cannot express and the
        # chunked backward must preserve exactly.
        weights = torch.linspace(0.1, 2.0, n_tokens)
        (ce * weights).sum().backward()
        grads = {
            name: p.grad.clone()
            for name, p in model.named_parameters()
            if p.grad is not None
        }
        return ce.detach().clone(), features.grad.clone(), grads

    ce_ref, feat_grad_ref, grads_ref = run(chunk_tokens=None)
    ce_chunk, feat_grad_chunk, grads_chunk = run(chunk_tokens=16)

    torch.testing.assert_close(ce_chunk, ce_ref)
    torch.testing.assert_close(feat_grad_chunk, feat_grad_ref, atol=1e-6, rtol=1e-5)
    assert grads_chunk.keys() == grads_ref.keys()
    for name in grads_ref:
        # Weight grads accumulate across chunks in fp32; tolerance covers the
        # accumulation-order difference vs the single-matmul reference.
        torch.testing.assert_close(
            grads_chunk[name], grads_ref[name], atol=1e-5, rtol=1e-4,
            msg=lambda m, n=name: f"{n}: {m}",
        )


def test_chunked_ce_inference_mode() -> None:
    torch.manual_seed(27)
    model = _tiny_model()
    targets = torch.randint(0, 31, (48,))
    with torch.inference_mode():
        features = torch.randn(48, 16)
        prepared = model.output_head.prepare_features(
            features, model.token_embed.get_weight()
        )
        ce_ref = output_head_cross_entropy(
            model, prepared, targets, prepared=True, reduction="none",
        )
        ce_chunk = output_head_cross_entropy(
            model, prepared, targets, prepared=True, reduction="none",
            chunk_tokens=16,
        )
    torch.testing.assert_close(ce_chunk, ce_ref)


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
