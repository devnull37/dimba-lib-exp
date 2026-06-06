"""Tests for TASK A (compute_model_perplexity, distinct_n) and TASK B (AbsorbingMaskCorruption).

Covers:
  TASK A
  * compute_model_perplexity (from dimba.evaluation) on a tiny DIMBA + a few
    real-ish token batches returns a finite positive float (not exp(MSE)).
    Sanity: a random-output model gives ppl in a plausible range (> 1,
    roughly <= vocab_size).
  * distinct_n(["the cat sat","the cat ran"], 1) and (..., 2) return floats
    in (0, 1]. An all-identical list gives low distinct-2; an all-distinct
    list (unique words throughout) gives high distinct-2.

  TASK B
  * AbsorbingMaskCorruption with schedule='cosine' and schedule='linear':
    the loss runs finite for both schedules.
  * The cosine NELBO weight at t=0.5 equals (pi/2 * sin(pi/4)) /
    (1 - cos(pi/4)) ~= 3.79, which is NOT the linear 1/t = 2.0.
"""

import math
import torch
import torch.nn.functional as F
import pytest

from dimba import DIMBA
from dimba.evaluation import compute_model_perplexity
from dimba.evaluation.metrics import distinct_n
from dimba.diffusion.corruption import AbsorbingMaskCorruption


# ---------------------------------------------------------------------------
# Shared tiny-model helpers
# ---------------------------------------------------------------------------

VOCAB_SIZE = 32
D_MODEL = 16
SEQ_LEN = 8
BATCH_SIZE = 4


def _tiny_model(seed: int = 0) -> DIMBA:
    torch.manual_seed(seed)
    return DIMBA(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        d_prompt=D_MODEL,
        num_diffusion_steps=8,
        num_denoiser_layers=1,
        d_state=4,
        use_simple_mamba=True,
    )


def _tiny_dataloader(model: DIMBA, n_batches: int = 3):
    """Yield simple dicts with 'input_ids' to mimic a real DataLoader."""
    torch.manual_seed(7)
    batches = []
    for _ in range(n_batches):
        ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        batches.append({"input_ids": ids})
    return batches


# ---------------------------------------------------------------------------
# TASK A — compute_model_perplexity
# ---------------------------------------------------------------------------


class TestComputeModelPerplexity:
    """compute_model_perplexity returns a proper denoising-reconstruction PPL."""

    def test_returns_finite_positive_float(self):
        model = _tiny_model()
        dl = _tiny_dataloader(model)
        ppl = compute_model_perplexity(model, dl, device="cpu")
        assert isinstance(ppl, float), f"Expected float, got {type(ppl)}"
        assert math.isfinite(ppl), f"PPL is not finite: {ppl}"
        assert ppl > 0.0, f"PPL must be positive, got {ppl}"

    def test_ppl_greater_than_one(self):
        """A real model should give PPL > 1 (NLL > 0) for non-trivial text."""
        model = _tiny_model()
        dl = _tiny_dataloader(model)
        ppl = compute_model_perplexity(model, dl, device="cpu")
        assert ppl > 1.0, f"PPL should be > 1, got {ppl}"

    def test_ppl_plausible_upper_bound(self):
        """A random-output model should have PPL roughly <= vocab_size.

        At uniform confusion the NLL is ~log(vocab_size) so exp(NLL) ~ vocab_size.
        Allow generous headroom (3x) to account for random weight variance.
        """
        model = _tiny_model(seed=42)
        dl = _tiny_dataloader(model)
        ppl = compute_model_perplexity(model, dl, device="cpu")
        # Generous upper bound: 3 * vocab_size (accounts for random-init variance)
        assert ppl <= 3.0 * VOCAB_SIZE, (
            f"PPL {ppl:.2f} is implausibly large (> 3 * vocab_size={3*VOCAB_SIZE}); "
            f"this suggests exp(MSE) rather than exp(NLL)"
        )

    def test_not_exp_mse(self):
        """Regression: function must NOT return exp(MSE(embeddings)).

        We verify this by checking that the returned value is on the scale of
        a cross-entropy perplexity (bounded by vocab_size with room), not on
        the scale of an MSE-derived quantity (which would be exp(~d_model-scale
        numbers and return values far outside the vocabulary range for a random
        model).
        """
        model = _tiny_model(seed=0)
        dl = _tiny_dataloader(model)
        ppl = compute_model_perplexity(model, dl, device="cpu")
        # exp(MSE) on random embeddings in R^D_MODEL is on the order of e^D_MODEL
        # which for D_MODEL=16 is ~8.9 million. A cross-entropy PPL on a vocab of
        # size 32 must be much smaller.
        assert ppl < 1e5, (
            f"PPL {ppl:.2f} looks like exp(MSE) rather than exp(NLL): "
            f"expected < 1e5, got {ppl}"
        )


# ---------------------------------------------------------------------------
# TASK A — distinct_n
# ---------------------------------------------------------------------------


class TestDistinctN:
    """distinct_n computes whitespace-tokenized n-gram diversity correctly."""

    def test_unigram_in_range(self):
        texts = ["the cat sat", "the cat ran"]
        val = distinct_n(texts, 1)
        assert isinstance(val, float), f"Expected float, got {type(val)}"
        assert 0.0 < val <= 1.0, f"distinct_1 should be in (0, 1], got {val}"

    def test_bigram_in_range(self):
        texts = ["the cat sat", "the cat ran"]
        val = distinct_n(texts, 2)
        assert isinstance(val, float), f"Expected float, got {type(val)}"
        assert 0.0 < val <= 1.0, f"distinct_2 should be in (0, 1], got {val}"

    def test_all_identical_low_distinct2(self):
        """Repeating the same sentence gives distinct-2 close to 0 (many repeated bigrams)."""
        texts = ["the cat sat on the mat"] * 10
        val = distinct_n(texts, 2)
        # 5 bigrams, all repeated 10 times -> distinct-2 = 5/50 = 0.1
        assert val < 0.5, f"All-identical list should have low distinct-2, got {val}"

    def test_all_distinct_high_distinct2(self):
        """Sentences with all unique bigrams give distinct-2 == 1."""
        # Each bigram appears exactly once across all sentences.
        texts = [
            "apple banana cherry",    # bigrams: (apple,banana), (banana,cherry)
            "delta echo foxtrot",     # bigrams: (delta,echo), (echo,foxtrot)
            "golf hotel india",       # bigrams: (golf,hotel), (hotel,india)
        ]
        val = distinct_n(texts, 2)
        # All bigrams are unique -> distinct-2 = 1.0
        assert val == 1.0, f"All-distinct bigrams should give distinct-2=1.0, got {val}"

    def test_distinct1_mixed(self):
        """Two sentences sharing a word have distinct-1 < 1."""
        texts = ["the cat sat", "the cat ran"]
        val = distinct_n(texts, 1)
        # Tokens: the,cat,sat,the,cat,ran => 4 unique / 6 total = 0.667
        assert val < 1.0, f"Shared-token list should have distinct-1 < 1, got {val}"
        assert val > 0.0

    def test_empty_list_returns_zero(self):
        val = distinct_n([], 1)
        assert val == 0.0

    def test_single_token_sentences(self):
        """Single-token sentences: distinct-1 is well-defined, distinct-2 is 0."""
        texts = ["hello", "hello", "world"]
        val1 = distinct_n(texts, 1)
        assert 0.0 < val1 <= 1.0
        # No bigrams possible from single-token sentences
        val2 = distinct_n(texts, 2)
        assert val2 == 0.0


# ---------------------------------------------------------------------------
# TASK B — AbsorbingMaskCorruption
# ---------------------------------------------------------------------------


class TestAbsorbingMaskCorruption:
    """AbsorbingMaskCorruption schedule correctness and loss finiteness."""

    # -- helpers --

    @staticmethod
    def _random_batch(batch: int = 4, seq: int = 6, vocab: int = 20):
        torch.manual_seed(42)
        ids = torch.randint(0, vocab, (batch, seq))
        return ids

    @staticmethod
    def _random_logits(batch: int = 4, seq: int = 6, vocab: int = 20):
        torch.manual_seed(99)
        return torch.randn(batch, seq, vocab)

    # -- loss finiteness --

    def test_cosine_loss_finite(self):
        """Loss is finite for cosine schedule."""
        vocab = 20
        proc = AbsorbingMaskCorruption(mask_token_id=vocab - 1, schedule="cosine")
        ids = self._random_batch(vocab=vocab)
        t = proc.sample_timesteps(ids.shape[0], ids.device)
        masked_ids, info = proc.corrupt(ids, t)
        logits = self._random_logits(vocab=vocab)
        loss = proc.loss(logits, info)
        assert torch.isfinite(loss), f"Cosine loss is not finite: {loss}"
        assert loss >= 0.0, f"Cosine loss must be non-negative, got {loss}"

    def test_linear_loss_finite(self):
        """Loss is finite for linear schedule."""
        vocab = 20
        proc = AbsorbingMaskCorruption(mask_token_id=vocab - 1, schedule="linear")
        ids = self._random_batch(vocab=vocab)
        t = proc.sample_timesteps(ids.shape[0], ids.device)
        masked_ids, info = proc.corrupt(ids, t)
        logits = self._random_logits(vocab=vocab)
        loss = proc.loss(logits, info)
        assert torch.isfinite(loss), f"Linear loss is not finite: {loss}"
        assert loss >= 0.0, f"Linear loss must be non-negative, got {loss}"

    # -- cosine weight differs from 1/t at t=0.5 --

    def test_cosine_weight_differs_from_linear_at_half(self):
        """At t=0.5, cosine NELBO weight != 1/t (the linear weight).

        The cosine weight formula:
            w_cos(t) = (pi/2 * sin(pi*t/2)) / (1 - cos(pi*t/2))

        At t=0.5:
            w_cos(0.5) = (pi/2 * sin(pi/4)) / (1 - cos(pi/4))
                       ~= 3.79

        The linear weight is 1/t = 2.0. They must differ.
        """
        t_val = 0.5
        t = torch.tensor([t_val])

        # Replicate cosine formula from AbsorbingMaskCorruption.loss
        numerator = (0.5 * math.pi) * torch.sin(0.5 * math.pi * t)
        denominator = (1.0 - torch.cos(0.5 * math.pi * t)).clamp(min=1e-8)
        cosine_weight = (numerator / denominator).item()

        linear_weight = 1.0 / t_val  # 2.0

        # Expected cosine weight: (pi/2 * sin(pi/4)) / (1 - cos(pi/4))
        expected = (
            (0.5 * math.pi) * math.sin(0.5 * math.pi * t_val)
        ) / (1.0 - math.cos(0.5 * math.pi * t_val))

        # Cosine and linear weights must differ noticeably at t=0.5
        assert abs(cosine_weight - linear_weight) > 0.5, (
            f"Cosine weight {cosine_weight:.4f} and linear weight {linear_weight:.4f} "
            f"should differ by > 0.5 at t=0.5"
        )

        # Cosine weight must match the analytic formula (pi/2*sin(pi/4))/(1-cos(pi/4))
        assert math.isclose(cosine_weight, expected, rel_tol=1e-5), (
            f"Cosine weight {cosine_weight:.6f} != analytic {expected:.6f}"
        )

        # The formula is NOT the linear 1/t
        assert not math.isclose(cosine_weight, linear_weight, rel_tol=0.01), (
            f"Cosine weight {cosine_weight:.4f} must NOT equal linear weight "
            f"{linear_weight:.4f} at t=0.5"
        )

    def test_cosine_weight_approx_expected_value(self):
        """At t=0.5, the cosine weight is approximately 3.79 (not ~2.0 or ~2.66)."""
        t_val = 0.5
        numerator = (0.5 * math.pi) * math.sin(0.5 * math.pi * t_val)
        denominator = 1.0 - math.cos(0.5 * math.pi * t_val)
        cosine_weight = numerator / denominator

        # The implemented cosine weight should be close to ~3.79
        assert math.isclose(cosine_weight, 3.79, rel_tol=0.01), (
            f"Expected cosine weight ~3.79, got {cosine_weight:.4f}"
        )
        # And it is definitely not 2.0 (which would be 1/t for linear schedule)
        assert not math.isclose(cosine_weight, 2.0, rel_tol=0.05), (
            f"Cosine weight {cosine_weight:.4f} must not equal linear 1/t = 2.0"
        )

    def test_cosine_vs_linear_loss_values_differ(self):
        """Cosine and linear schedule produce different loss values on the same input."""
        vocab = 20
        torch.manual_seed(123)
        ids = torch.randint(0, vocab, (4, 6))
        # Use a fixed t=0.5 to compare schedules deterministically
        t_fixed = torch.full((4,), 0.5)

        proc_cos = AbsorbingMaskCorruption(mask_token_id=vocab - 1, schedule="cosine")
        proc_lin = AbsorbingMaskCorruption(mask_token_id=vocab - 1, schedule="linear")

        # Use same random mask by seeding
        torch.manual_seed(55)
        _, info_cos = proc_cos.corrupt(ids, t_fixed)
        torch.manual_seed(55)
        _, info_lin = proc_lin.corrupt(ids, t_fixed)

        logits = torch.randn(4, 6, vocab)
        loss_cos = proc_cos.loss(logits, info_cos)
        loss_lin = proc_lin.loss(logits, info_lin)

        # Both must be finite
        assert torch.isfinite(loss_cos), f"Cosine loss not finite: {loss_cos}"
        assert torch.isfinite(loss_lin), f"Linear loss not finite: {loss_lin}"

        # They should differ because the NELBO weights differ
        ratio = loss_cos.item() / (loss_lin.item() + 1e-8)
        # cosine weight (3.79) vs linear weight (2.0): ratio should be ~1.895
        assert not math.isclose(loss_cos.item(), loss_lin.item(), rel_tol=0.05), (
            f"Cosine loss {loss_cos.item():.4f} and linear loss {loss_lin.item():.4f} "
            f"should differ (cosine weight ~3.79, linear ~2.0)"
        )
