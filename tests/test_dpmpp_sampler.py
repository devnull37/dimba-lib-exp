"""Tests for the DPM-Solver++(2M) sampler added to sample_from_model.

Uses the same tiny DIMBA configuration as test_smoke.py so the whole file
runs in well under 30 s on CPU.
"""

import pytest
import torch

from dimba.diffusion.sampling import sample_from_model
from dimba.models.diffusion import DIMBA

# Tiny configuration -------------------------------------------------------
VOCAB_SIZE = 256
D_MODEL = 64
NUM_DENOISER_LAYERS = 2
NUM_DIFFUSION_STEPS = 20
SEQ_LEN = 16
BATCH_SIZE = 2


def _build_tiny_model(seed: int = 0) -> DIMBA:
    torch.manual_seed(seed)
    return DIMBA(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        d_prompt=D_MODEL,
        num_diffusion_steps=NUM_DIFFUSION_STEPS,
        num_denoiser_layers=NUM_DENOISER_LAYERS,
        use_simple_mamba=True,
    )


@pytest.fixture(scope="module")
def tiny_model() -> DIMBA:
    return _build_tiny_model(seed=7)


# ---------------------------------------------------------------------------
# Core correctness tests
# ---------------------------------------------------------------------------

def test_dpmpp_returns_valid_token_ids(tiny_model: DIMBA) -> None:
    """DPM-Solver++(2M) with 15 steps returns [1, seq_len] valid token IDs."""
    torch.manual_seed(42)
    generated = sample_from_model(
        tiny_model,
        prompt_ids=None,
        seq_len=SEQ_LEN,
        num_steps=15,
        device=torch.device("cpu"),
        sampler="dpmpp",
    )

    assert generated.shape == (1, SEQ_LEN), f"unexpected shape {generated.shape}"
    assert generated.dtype == torch.long
    assert int(generated.min()) >= 0, "token id below 0"
    assert int(generated.max()) < VOCAB_SIZE, "token id >= vocab_size"
    assert torch.isfinite(generated.float()).all(), "non-finite token ids"


def test_dpmpp_with_prompt_returns_valid_tokens(tiny_model: DIMBA) -> None:
    """DPM-Solver++(2M) with a prompt prefix returns valid tokens."""
    torch.manual_seed(43)
    prompt_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, 4))
    generated = sample_from_model(
        tiny_model,
        prompt_ids=prompt_ids,
        seq_len=SEQ_LEN,
        num_steps=15,
        device=torch.device("cpu"),
        sampler="dpmpp",
    )

    assert generated.shape == (BATCH_SIZE, SEQ_LEN)
    assert generated.dtype == torch.long
    assert int(generated.min()) >= 0
    assert int(generated.max()) < VOCAB_SIZE
    assert torch.isfinite(generated.float()).all()


def test_ddim_default_still_works(tiny_model: DIMBA) -> None:
    """Default sampler='ddim' (unchanged behaviour) still produces valid output."""
    torch.manual_seed(44)
    prompt_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, 4))
    generated = sample_from_model(
        tiny_model,
        prompt_ids=prompt_ids,
        seq_len=SEQ_LEN,
        num_steps=4,
        device=torch.device("cpu"),
        # sampler defaults to "ddim"
    )

    assert generated.shape == (BATCH_SIZE, SEQ_LEN)
    assert generated.dtype == torch.long
    assert int(generated.min()) >= 0
    assert int(generated.max()) < VOCAB_SIZE


def test_dpmpp_cfg_eps_finite(tiny_model: DIMBA) -> None:
    """DPM-Solver++ with CFG (cfg_mode='eps', guidance_scale=2) runs without NaN/Inf."""
    torch.manual_seed(45)
    prompt_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, 4))
    generated = sample_from_model(
        tiny_model,
        prompt_ids=prompt_ids,
        seq_len=SEQ_LEN,
        num_steps=15,
        guidance_scale=2.0,
        cfg_mode="eps",
        device=torch.device("cpu"),
        sampler="dpmpp",
    )

    assert generated.shape == (BATCH_SIZE, SEQ_LEN)
    assert generated.dtype == torch.long
    assert int(generated.min()) >= 0
    assert int(generated.max()) < VOCAB_SIZE
    assert torch.isfinite(generated.float()).all(), "CFG + dpmpp produced non-finite tokens"


def test_dpmpp_first_order_matches_ddim() -> None:
    """The first-order DPM-Solver++ step (no history) MUST reduce to DDIM(eta=0).

    Regression guard for the data-term sign: a sign flip on the x0 term still
    produces finite, in-range token ids (so the other tests pass) but yields a
    completely wrong reverse trajectory. This catches it.
    """
    from dimba.diffusion.sampling import _dpmpp_step, _ddim_step

    torch.manual_seed(0)
    x_t = torch.randn(2, 5, 8)
    x0 = torch.randn(2, 5, 8)
    for acp_t, acp_prev in [(0.4, 0.7), (0.1, 0.9), (0.5, 0.51)]:
        ddim = _ddim_step(x_t, x0, torch.tensor(acp_t), torch.tensor(acp_prev), 0.0)
        dpmpp, _, _ = _dpmpp_step(x_t, x0, torch.tensor(acp_t), torch.tensor(acp_prev), None, None)
        assert torch.allclose(ddim, dpmpp, atol=1e-5), (
            f"dpmpp 1st-order step != DDIM(eta=0) at acp=({acp_t},{acp_prev})"
        )


def test_dpmpp_converges_to_ddim_at_high_steps(tiny_model: DIMBA) -> None:
    """At full step count both samplers approximate the same ODE -> high token agreement."""
    torch.manual_seed(1)
    a = sample_from_model(tiny_model, None, seq_len=SEQ_LEN, num_steps=NUM_DIFFUSION_STEPS,
                          sampler="ddim", temperature=1e-9, device=torch.device("cpu"))
    torch.manual_seed(1)
    b = sample_from_model(tiny_model, None, seq_len=SEQ_LEN, num_steps=NUM_DIFFUSION_STEPS,
                          sampler="dpmpp", temperature=1e-9, device=torch.device("cpu"))
    agree = (a == b).float().mean().item()
    assert agree > 0.6, f"dpmpp vs ddim token agreement only {agree:.0%} -- trajectory bug?"


def test_invalid_sampler_raises() -> None:
    """Passing an unknown sampler name raises ValueError."""
    model = _build_tiny_model(seed=99)
    with pytest.raises(ValueError, match="sampler must be"):
        sample_from_model(
            model,
            prompt_ids=None,
            seq_len=8,
            num_steps=4,
            device=torch.device("cpu"),
            sampler="bogus",
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
