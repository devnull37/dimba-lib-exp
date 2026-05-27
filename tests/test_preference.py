"""Unit tests for DIMBA preference optimization and pluggable rewards.

Covers:
    - ``dpo_loss`` decreases as the chosen-vs-rejected log-prob margin grows.
    - ``ipo_loss`` / ``simpo_loss`` basic monotonicity and shapes.
    - ``sequence_logprob`` masking and ``elbo_sequence_logprob`` gradient flow.
    - Reward classes returning expected values on crafted strings.

All tests use tiny tensors / short strings and run on CPU.
"""

from __future__ import annotations

import re

import pytest
import torch

from dimba.training.preference import (
    antithetic_timesteps,
    dpo_loss,
    elbo_sequence_logprob,
    ipo_loss,
    sequence_logprob,
    simpo_loss,
)
from dimba.training.rewards import (
    CodeExecReward,
    CompositeReward,
    ExactMatchReward,
    LengthPenaltyReward,
    NumericAnswerReward,
    RegexMatchReward,
    Reward,
    RewardModelReward,
    TokenOverlapReward,
    get_reward,
)


# --------------------------------------------------------------------------- #
# preference.py: log-prob primitives
# --------------------------------------------------------------------------- #
def test_sequence_logprob_respects_mask() -> None:
    """Masked-out positions must not contribute to the summed log-prob."""
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 5)
    labels = torch.randint(0, 5, (2, 4))

    full_mask = torch.ones(2, 4)
    half_mask = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])

    lp_full = sequence_logprob(logits, labels, full_mask)
    lp_half = sequence_logprob(logits, labels, half_mask)
    zero_lp = sequence_logprob(logits, labels, torch.zeros(2, 4))

    assert lp_full.shape == (2,)
    # Log-probs are negative; summing more (negative) terms => more negative.
    assert (lp_full <= lp_half + 1e-6).all()
    assert torch.allclose(zero_lp, torch.zeros(2))


def test_dpo_loss_decreases_as_margin_grows() -> None:
    """DPO loss is monotonically decreasing in the chosen-vs-rejected margin."""
    ref_chosen = torch.tensor([0.0, 0.0])
    ref_rejected = torch.tensor([0.0, 0.0])
    rejected = torch.tensor([0.0, 0.0])

    margins = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
    losses = []
    for m in margins:
        chosen = torch.tensor([m, m])
        loss, chosen_reward, rejected_reward = dpo_loss(
            chosen, rejected, ref_chosen, ref_rejected, beta=0.1
        )
        losses.append(loss.item())
        assert chosen_reward.shape == (2,)
        assert rejected_reward.shape == (2,)

    # Strictly decreasing as the (positive) margin increases.
    for earlier, later in zip(losses, losses[1:]):
        assert later < earlier


def test_dpo_implicit_rewards_match_formula() -> None:
    """Implicit rewards equal beta * (policy_lp - ref_lp)."""
    pi_c = torch.tensor([2.0])
    pi_r = torch.tensor([1.0])
    ref_c = torch.tensor([0.5])
    ref_r = torch.tensor([0.25])
    beta = 0.2

    _, chosen_reward, rejected_reward = dpo_loss(pi_c, pi_r, ref_c, ref_r, beta=beta)
    assert torch.allclose(chosen_reward, beta * (pi_c - ref_c))
    assert torch.allclose(rejected_reward, beta * (pi_r - ref_r))


def test_dpo_label_smoothing_changes_loss() -> None:
    """cDPO label smoothing yields a different (regularized) loss value."""
    args = (torch.tensor([1.0]), torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]))
    base, _, _ = dpo_loss(*args, beta=0.1, label_smoothing=0.0)
    smoothed, _, _ = dpo_loss(*args, beta=0.1, label_smoothing=0.2)
    assert not torch.allclose(base, smoothed)


def test_ipo_loss_minimized_at_target_margin() -> None:
    """IPO squared loss is smallest when the margin hits 1/(2*beta)."""
    beta = 0.5
    target = 1.0 / (2.0 * beta)  # == 1.0
    ref = torch.tensor([0.0])
    rejected = torch.tensor([0.0])

    loss_at_target, _, _ = ipo_loss(torch.tensor([target]), rejected, ref, ref, beta=beta)
    loss_off_target, _, _ = ipo_loss(torch.tensor([target + 2.0]), rejected, ref, ref, beta=beta)
    assert loss_at_target.item() < loss_off_target.item()
    assert loss_at_target.item() == pytest.approx(0.0, abs=1e-6)


def test_simpo_loss_reference_free_and_decreases() -> None:
    """SimPO (reference-free) loss decreases as the length-normalized gap grows."""
    chosen_len = torch.tensor([4.0])
    rejected_len = torch.tensor([4.0])
    rejected = torch.tensor([0.0])

    small, _, _ = simpo_loss(torch.tensor([4.0]), rejected, chosen_len, rejected_len, beta=2.0, gamma=1.0)
    large, _, _ = simpo_loss(torch.tensor([40.0]), rejected, chosen_len, rejected_len, beta=2.0, gamma=1.0)
    assert large.item() < small.item()


def test_antithetic_timesteps_are_mirrored() -> None:
    """Antithetic partner satisfies t + t' = T - 1 and stays in range."""
    T = 100
    t, t_anti = antithetic_timesteps(64, T)
    assert t.shape == (64,)
    assert torch.all((t + t_anti) == (T - 1))
    assert int(t.min()) >= 0 and int(t.max()) < T


def test_elbo_sequence_logprob_grad_flows() -> None:
    """ELBO surrogate is differentiable wrt model params and shaped [batch]."""
    torch.manual_seed(1)
    vocab = 6
    seq = 4

    class StubDiffusion(torch.nn.Module):
        num_diffusion_steps = 8

        def __init__(self) -> None:
            super().__init__()
            self.head = torch.nn.Linear(1, vocab)

    def logits_fn(model, input_ids, t):
        feat = (t.float() / model.num_diffusion_steps).view(-1, 1, 1)
        feat = feat.expand(input_ids.shape[0], input_ids.shape[1], 1)
        return model.head(feat)

    model = StubDiffusion()
    input_ids = torch.randint(0, vocab, (2, seq))
    labels = torch.randint(0, vocab, (2, seq))
    mask = torch.tensor([[0.0, 1.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]])

    lp = elbo_sequence_logprob(
        model, input_ids, labels, mask, num_mc_samples=2, antithetic=True, logits_fn=logits_fn
    )
    assert lp.shape == (2,)
    assert lp.requires_grad
    lp.sum().backward()
    assert model.head.weight.grad is not None


def test_elbo_end_to_end_dpo_gradient() -> None:
    """ELBO log-probs feed DPO and produce a finite, differentiable loss."""
    torch.manual_seed(2)
    vocab = 5

    class StubDiffusion(torch.nn.Module):
        num_diffusion_steps = 4

        def __init__(self) -> None:
            super().__init__()
            self.head = torch.nn.Linear(1, vocab)

    def logits_fn(model, input_ids, t):
        feat = (t.float() / model.num_diffusion_steps).view(-1, 1, 1)
        return model.head(feat.expand(input_ids.shape[0], input_ids.shape[1], 1))

    policy = StubDiffusion()
    fixed_t = torch.zeros(2, dtype=torch.long)
    ids = torch.randint(0, vocab, (2, 3))
    labels_c = torch.randint(0, vocab, (2, 3))
    labels_r = torch.randint(0, vocab, (2, 3))
    mask = torch.ones(2, 3)

    pi_c = elbo_sequence_logprob(policy, ids, labels_c, mask, timesteps=fixed_t, logits_fn=logits_fn)
    pi_r = elbo_sequence_logprob(policy, ids, labels_r, mask, timesteps=fixed_t, logits_fn=logits_fn)
    ref = torch.zeros(2)
    loss, _, _ = dpo_loss(pi_c, pi_r, ref, ref, beta=0.1)
    assert torch.isfinite(loss)
    loss.backward()
    assert policy.head.weight.grad is not None


# --------------------------------------------------------------------------- #
# rewards.py: verifiable and proxy rewards
# --------------------------------------------------------------------------- #
def test_reward_protocol_runtime_checkable() -> None:
    """Concrete rewards satisfy the runtime-checkable Reward protocol."""
    assert isinstance(ExactMatchReward(), Reward)
    assert isinstance(NumericAnswerReward(), Reward)
    assert isinstance(TokenOverlapReward(), Reward)


def test_exact_match_reward() -> None:
    """Exact match is case/punctuation-insensitive by default."""
    reward = ExactMatchReward()
    assert reward("q", "Paris.", "paris") == 1.0
    assert reward("q", "  PARIS ", "paris") == 1.0
    assert reward("q", "London", "paris") == 0.0
    assert reward("q", "anything", None) == 0.0


def test_numeric_answer_reward_gsm8k_style() -> None:
    """Final-number extraction handles markers, commas, and trailing text."""
    reward = NumericAnswerReward()
    # GSM8K-style marker in the reference, final number in the completion.
    assert reward("q", "The answer is 42.", "#### 42") == 1.0
    # Comma grouping and last-number fallback.
    assert reward("q", "so we get 1,024 widgets", "1024") == 1.0
    # Wrong number.
    assert reward("q", "the result is 7", "8") == 0.0
    # No number present.
    assert reward("q", "no digits here", "5") == 0.0


def test_numeric_answer_reward_tolerance() -> None:
    """Absolute tolerance allows near-equal floats."""
    reward = NumericAnswerReward(abs_tol=0.01)
    assert reward("q", "3.141", "3.14") == 1.0
    assert reward("q", "3.20", "3.14") == 0.0


def test_regex_match_reward() -> None:
    """Regex reward fires only when the pattern is present."""
    boxed = RegexMatchReward(pattern=r"\\boxed\{.*\}")
    assert boxed("q", r"final \boxed{42}", None) == 1.0
    assert boxed("q", "no box here", None) == 0.0

    # Per-example pattern via the reference field.
    dynamic = RegexMatchReward(use_reference_as_pattern=True)
    assert dynamic("q", "hello world", r"he\w+o") == 1.0
    assert dynamic("q", "nope", r"\d{3}") == 0.0


def test_length_penalty_reward() -> None:
    """Length penalty is zero inside the window and negative outside."""
    reward = LengthPenaltyReward(target_length=5, tolerance=1, penalty_per_token=0.1)
    in_window = reward("q", "a b c d e", None)  # 5 tokens
    assert in_window == 0.0
    long_completion = reward("q", " ".join(["w"] * 20), None)  # far over window
    assert long_completion < 0.0


def test_reward_model_reward_wraps_callable() -> None:
    """RewardModelReward scales and clips an external scorer."""
    reward = RewardModelReward(scorer=lambda p, c, r: 10.0, scale=0.5, clip=(0.0, 3.0))
    assert reward("q", "c", None) == 3.0  # 10*0.5=5 clipped to 3
    reward2 = RewardModelReward(scorer=lambda p, c, r: 2.0, scale=0.5)
    assert reward2("q", "c", None) == pytest.approx(1.0)


def test_composite_reward_weighted_sum() -> None:
    """CompositeReward sums weighted component rewards."""
    composite = CompositeReward(
        components=[(NumericAnswerReward(), 1.0), (LengthPenaltyReward(target_length=1, tolerance=0, penalty_per_token=0.1, max_penalty=1.0), 1.0)]
    )
    # Correct number (+1.0) but long completion (negative length penalty).
    value = composite("q", "the answer is 42 and then a lot more words here", "42")
    assert value < 1.0  # penalty pulled it below the pure correctness reward
    assert value > -1.0


def test_token_overlap_reward_rewards_copying() -> None:
    """TokenOverlapReward (weak proxy) scores high for verbatim copies."""
    reward = TokenOverlapReward()
    identical = reward("q", "the cat sat on the mat", "the cat sat on the mat")
    disjoint = reward("q", "completely different words entirely", "the cat sat on the mat")
    assert identical == pytest.approx(1.0, abs=1e-6)
    assert disjoint < identical
    assert reward("q", "anything", None) == 0.0


def test_code_exec_reward_is_safe_stub() -> None:
    """CodeExecReward must NOT execute code; it raises NotImplementedError."""
    reward = CodeExecReward(unit_tests="assert solve() == 1", timeout_s=1.0)
    with pytest.raises(NotImplementedError):
        reward("write solve()", "def solve(): return 1", None)


def test_get_reward_registry() -> None:
    """get_reward constructs registered rewards and rejects unknown names."""
    assert isinstance(get_reward("numeric"), NumericAnswerReward)
    assert isinstance(get_reward("token_overlap"), TokenOverlapReward)
    with pytest.raises(KeyError):
        get_reward("does_not_exist")
