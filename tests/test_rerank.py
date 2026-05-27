"""Tests for best-of-K diffusion sample reranking (:mod:`dimba.diffusion.rerank`).

These tests use tiny tensors and run in well under a second on CPU. They cover:

* :func:`rerank_candidates` -- picks the unambiguously best candidate under a toy
  ``score_fn``, breaks ties at the lowest index, returns scores on request, and
  validates its inputs.
* :func:`best_of_k` -- generates ``k`` candidates and returns the max-scoring one,
  with ``return_all`` exposing every candidate/score.
* :func:`diffusion_elbo_score` -- runs against a tiny dummy ``model_forward`` under
  both supported return contracts ((x0_pred, x0_target) and scalar MSE) and both
  weightings, returns a finite scalar, and ranks a near-perfect denoiser above a
  random one. Also validates its input checks.
"""

import math

import pytest
import torch

from dimba.diffusion.rerank import (
    best_of_k,
    diffusion_elbo_score,
    rerank_candidates,
    sequence_logprob_score,
)


def _toy_alphas_cumprod(num_steps: int = 50) -> torch.Tensor:
    """Monotonically decreasing cosine-like ``alphas_cumprod`` for tests."""
    t = torch.arange(num_steps, dtype=torch.float32)
    acp = torch.cos(0.5 * math.pi * (t / num_steps + 0.008) / 1.008) ** 2
    return torch.clamp(acp, 1e-4, 1 - 1e-4)


# ---------------------------------------------------------------------------
# rerank_candidates
# ---------------------------------------------------------------------------


class TestRerankCandidates:
    def test_picks_unambiguous_best(self):
        # Candidates are token-id tensors; the toy score prefers larger sums.
        candidates = [
            torch.tensor([1, 1, 1]),
            torch.tensor([9, 9, 9]),  # unambiguously best under "sum"
            torch.tensor([2, 0, 1]),
        ]

        def score_fn(c: torch.Tensor) -> torch.Tensor:
            return c.sum()

        best = rerank_candidates(candidates, score_fn)
        assert torch.equal(best, candidates[1])

    def test_returns_scores_in_order(self):
        candidates = [torch.tensor([0.0]), torch.tensor([5.0]), torch.tensor([2.0])]
        best, scores = rerank_candidates(
            candidates, lambda c: c.item(), return_scores=True
        )
        assert scores == [0.0, 5.0, 2.0]
        assert torch.equal(best, candidates[1])

    def test_lower_is_better_via_negation(self):
        # Reranking maximizes; to pick the minimum, negate inside score_fn.
        candidates = [torch.tensor([3.0]), torch.tensor([1.0]), torch.tensor([7.0])]
        best = rerank_candidates(candidates, lambda c: -c.item())
        assert best.item() == 1.0

    def test_tie_breaks_to_lowest_index(self):
        candidates = ["a", "b", "c"]
        # All equal score -> stable argmax returns the first.
        best = rerank_candidates(candidates, lambda c: 1.0)
        assert best == "a"

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            rerank_candidates([], lambda c: 0.0)

    def test_non_scalar_score_raises(self):
        with pytest.raises(ValueError):
            rerank_candidates(
                [torch.tensor([1, 2, 3])], lambda c: c.float()  # returns a vector
            )


# ---------------------------------------------------------------------------
# best_of_k
# ---------------------------------------------------------------------------


class TestBestOfK:
    def test_returns_max_scoring_candidate(self):
        # Deterministic generator yields a known increasing sequence of values;
        # best_of_k must return the largest.
        torch.manual_seed(0)
        produced = [torch.tensor([float(v)]) for v in (3.0, 1.0, 8.0, 2.0)]
        it = iter(produced)

        def generate_fn() -> torch.Tensor:
            return next(it)

        best = best_of_k(generate_fn, lambda c: c.item(), k=4)
        assert best.item() == 8.0

    def test_return_all_exposes_candidates_and_scores(self):
        produced = [torch.tensor([float(v)]) for v in (4.0, 9.0, 1.0)]
        it = iter(produced)
        best, candidates, scores = best_of_k(
            lambda: next(it), lambda c: c.item(), k=3, return_all=True
        )
        assert best.item() == 9.0
        assert [c.item() for c in candidates] == [4.0, 9.0, 1.0]
        assert scores == [4.0, 9.0, 1.0]

    def test_k_one_returns_only_candidate(self):
        best = best_of_k(lambda: torch.tensor([42.0]), lambda c: c.item(), k=1)
        assert best.item() == 42.0

    def test_invalid_k_raises(self):
        with pytest.raises(ValueError):
            best_of_k(lambda: torch.tensor([0.0]), lambda c: c.item(), k=0)

    def test_composes_with_elbo_score(self):
        # End-to-end: generate two id sequences and rank by a dummy ELBO score
        # whose error is lower for a "good" sequence (id 0) than a "bad" one.
        acp = _toy_alphas_cumprod(40)

        good = torch.zeros(1, 6, dtype=torch.long)
        bad = torch.ones(1, 6, dtype=torch.long)
        it = iter([good, bad])

        def generate_fn() -> torch.Tensor:
            return next(it)

        def model_forward(input_ids, t):
            # Error depends only on the token id: id 0 -> tiny error, id 1 -> big.
            err = input_ids.float().mean()  # 0.0 for `good`, 1.0 for `bad`
            return err

        def score_fn(c):
            return diffusion_elbo_score(model_forward, c, acp, num_mc=3)

        best = best_of_k(generate_fn, score_fn, k=2)
        assert torch.equal(best, good)


# ---------------------------------------------------------------------------
# diffusion_elbo_score
# ---------------------------------------------------------------------------


class TestDiffusionElboScore:
    def test_runs_and_returns_finite_scalar_tuple_contract(self):
        torch.manual_seed(0)
        acp = _toy_alphas_cumprod(50)
        ids = torch.randint(0, 10, (2, 8))

        def model_forward(input_ids, t):
            # Tuple contract: return (x0_pred, x0_target). Random pred -> finite err.
            x0_target = torch.randn(input_ids.shape[0], input_ids.shape[1], 4)
            x0_pred = torch.randn_like(x0_target)
            return x0_pred, x0_target

        score = diffusion_elbo_score(model_forward, ids, acp, num_mc=4)
        assert score.shape == ()
        assert torch.isfinite(score)
        # Score is a NEGATIVE error -> non-positive.
        assert score.item() <= 0.0

    def test_scalar_mse_contract(self):
        acp = _toy_alphas_cumprod(50)
        ids = torch.randint(0, 10, (1, 5))

        def model_forward(input_ids, t):
            # Scalar contract: return a single non-negative MSE.
            return torch.tensor(0.25)

        score = diffusion_elbo_score(model_forward, ids, acp, num_mc=5)
        assert torch.isfinite(score)
        # Constant 0.25 error every draw -> score == -0.25 exactly.
        assert math.isclose(score.item(), -0.25, abs_tol=1e-6)

    def test_better_denoiser_scores_higher(self):
        torch.manual_seed(0)
        acp = _toy_alphas_cumprod(50)
        ids = torch.randint(0, 10, (1, 8))

        def good_forward(input_ids, t):
            return torch.tensor(0.01)  # near-perfect reconstruction

        def bad_forward(input_ids, t):
            return torch.tensor(1.0)  # poor reconstruction

        good = diffusion_elbo_score(good_forward, ids, acp, num_mc=4)
        bad = diffusion_elbo_score(bad_forward, ids, acp, num_mc=4)
        assert good.item() > bad.item()

    def test_1d_input_uses_single_timestep_row(self):
        acp = _toy_alphas_cumprod(30)
        ids = torch.randint(0, 10, (7,))  # 1-D [seq]
        seen_shapes = []

        def model_forward(input_ids, t):
            seen_shapes.append(tuple(t.shape))
            return torch.tensor(0.5)

        score = diffusion_elbo_score(model_forward, ids, acp, num_mc=2)
        assert torch.isfinite(score)
        # 1-D input -> batch of 1 timestep.
        assert all(s == (1,) for s in seen_shapes)

    def test_snr_weighting_runs_finite(self):
        torch.manual_seed(0)
        acp = _toy_alphas_cumprod(50)
        ids = torch.randint(0, 10, (2, 6))

        def model_forward(input_ids, t):
            x0_target = torch.randn(input_ids.shape[0], input_ids.shape[1], 3)
            return torch.randn_like(x0_target), x0_target

        score = diffusion_elbo_score(
            model_forward, ids, acp, num_mc=4, weighting="snr"
        )
        assert torch.isfinite(score)
        assert score.item() <= 0.0

    def test_shared_timesteps_are_paired_with_generator(self):
        # With a shared generator + shared_timesteps, the timesteps drawn for two
        # different candidates are identical (paired comparison).
        acp = _toy_alphas_cumprod(50)
        gen = torch.Generator()
        seen = []

        def model_forward(input_ids, t):
            seen.append(t.clone())
            return torch.tensor(0.3)

        ids_a = torch.randint(0, 10, (1, 4))
        ids_b = torch.randint(0, 10, (1, 4))
        diffusion_elbo_score(model_forward, ids_a, acp, num_mc=3, generator=gen)
        first = list(seen)
        seen.clear()
        diffusion_elbo_score(model_forward, ids_b, acp, num_mc=3, generator=gen)
        second = list(seen)

        assert len(first) == len(second) == 3
        for a, b in zip(first, second):
            assert torch.equal(a, b)

    def test_timesteps_within_requested_range(self):
        acp = _toy_alphas_cumprod(100)
        ids = torch.randint(0, 10, (3, 5))
        lo, hi = 10, 40

        def model_forward(input_ids, t):
            assert (t >= lo).all() and (t < hi).all()
            return torch.tensor(0.2)

        score = diffusion_elbo_score(
            model_forward, ids, acp, num_mc=6, t_min=lo, t_max=hi
        )
        assert torch.isfinite(score)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"num_mc": 0},
            {"weighting": "bogus"},
            {"t_min": 5, "t_max": 5},  # empty range
            {"t_max": 999},  # out of range (> T)
        ],
    )
    def test_invalid_args_raise(self, kwargs):
        acp = _toy_alphas_cumprod(50)
        ids = torch.randint(0, 10, (1, 4))
        with pytest.raises(ValueError):
            diffusion_elbo_score(
                lambda i, t: torch.tensor(0.1), ids, acp, **kwargs
            )

    def test_non_1d_schedule_raises(self):
        ids = torch.randint(0, 10, (1, 4))
        bad_acp = torch.randn(5, 5)  # 2-D
        with pytest.raises(ValueError):
            diffusion_elbo_score(lambda i, t: torch.tensor(0.1), ids, bad_acp)


# ---------------------------------------------------------------------------
# sequence_logprob_score adapter
# ---------------------------------------------------------------------------


class TestSequenceLogprobScore:
    def test_passes_through_and_ranks(self):
        c_hi = "high"
        c_lo = "low"
        scores = {"high": -1.0, "low": -5.0}

        s_hi = sequence_logprob_score(lambda c: scores[c], c_hi)
        s_lo = sequence_logprob_score(lambda c: scores[c], c_lo)
        assert torch.isfinite(s_hi) and torch.isfinite(s_lo)
        assert s_hi.item() > s_lo.item()

        # Drops into rerank_candidates as a higher-is-better score.
        best = rerank_candidates(
            [c_lo, c_hi], lambda c: sequence_logprob_score(lambda x: scores[x], c)
        )
        assert best == c_hi


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
