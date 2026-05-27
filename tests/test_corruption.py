"""Tests for diffusion corruption processes and masked-diffusion sampling.

These tests use tiny tensors and run in well under 20 seconds on CPU. They cover:

* Gaussian embedding corruption: shape/parameterization + finite, positive MSE
  loss (with and without min-SNR weighting).
* Absorbing-mask corruption: empirical mask fraction tracks the schedule, and
  the masked cross-entropy NELBO loss is finite and > 0.
* Hybrid corruption: produces both masked (discrete) and noised (continuous)
  positions, and yields a finite, positive combined loss.
* Masked-diffusion sampler: ends fully unmasked and keeps prompt tokens fixed,
  including with the low-confidence remasking variant.

References: MDLM (arXiv:2406.07524), LLaDA (arXiv:2502.09992).
"""

import math

import pytest
import torch

from dimba.diffusion.corruption import (
    AbsorbingMaskCorruption,
    GaussianEmbeddingCorruption,
    HybridCorruption,
    _mask_prob,
)
from dimba.diffusion.masked_sampling import masked_diffusion_sample


def _toy_alphas_cumprod(num_steps: int = 100) -> torch.Tensor:
    """Build a monotonically decreasing cosine-like ``alphas_cumprod`` schedule."""
    t = torch.arange(num_steps, dtype=torch.float32)
    acp = torch.cos(0.5 * math.pi * (t / num_steps + 0.008) / 1.008) ** 2
    return torch.clamp(acp, 1e-4, 1 - 1e-4)


# ---------------------------------------------------------------------------
# Gaussian embedding corruption.
# ---------------------------------------------------------------------------


class TestGaussianEmbeddingCorruption:
    def test_corrupt_shapes_and_parameterization(self):
        torch.manual_seed(0)
        acp = _toy_alphas_cumprod(100)
        proc = GaussianEmbeddingCorruption(acp)

        x0 = torch.randn(4, 8, 16)
        t = torch.tensor([10, 30, 50, 70])
        noise = torch.randn_like(x0)

        x_t, info = proc.corrupt(x0, t, noise=noise)

        assert x_t.shape == x0.shape
        assert torch.equal(info["noise"], noise)
        assert torch.equal(info["x0"], x0)
        # Recompute x_t by hand and compare.
        a = acp[t].view(-1, 1, 1)
        expected = torch.sqrt(a) * x0 + torch.sqrt(1 - a) * noise
        assert torch.allclose(x_t, expected, atol=1e-5)

    def test_loss_finite_and_positive(self):
        torch.manual_seed(0)
        proc = GaussianEmbeddingCorruption(_toy_alphas_cumprod(100))
        x0 = torch.randn(4, 8, 16)
        t = torch.tensor([5, 25, 55, 95])
        _, info = proc.corrupt(x0, t)

        prediction = torch.randn_like(x0)  # wrong on purpose -> positive loss
        loss = proc.loss(prediction, info)
        assert torch.isfinite(loss)
        assert loss.item() > 0

    def test_min_snr_weighting_finite(self):
        torch.manual_seed(0)
        proc = GaussianEmbeddingCorruption(_toy_alphas_cumprod(100))
        x0 = torch.randn(3, 6, 12)
        t = torch.tensor([10, 50, 90])
        _, info = proc.corrupt(x0, t)
        prediction = torch.randn_like(x0)

        loss_plain = proc.loss(prediction, info)
        loss_weighted = proc.loss(prediction, info, min_snr_gamma=5.0)
        assert torch.isfinite(loss_weighted)
        assert loss_weighted.item() > 0
        # Perfect prediction -> ~zero loss even with weighting.
        zero_loss = proc.loss(x0.clone(), info, min_snr_gamma=5.0)
        assert zero_loss.item() < 1e-6


# ---------------------------------------------------------------------------
# Absorbing-mask (MDLM/LLaDA) corruption.
# ---------------------------------------------------------------------------


class TestAbsorbingMaskCorruption:
    def test_mask_prob_endpoints(self):
        t0 = torch.tensor([0.0])
        t1 = torch.tensor([1.0])
        for sched in ("linear", "cosine"):
            assert torch.allclose(_mask_prob(t0, sched), torch.tensor([0.0]), atol=1e-6)
            assert torch.allclose(_mask_prob(t1, sched), torch.tensor([1.0]), atol=1e-6)

    @pytest.mark.parametrize("schedule", ["linear", "cosine"])
    def test_mask_fraction_matches_schedule(self, schedule):
        torch.manual_seed(0)
        proc = AbsorbingMaskCorruption(mask_token_id=99, schedule=schedule)
        # Large batch/seq so the empirical mask fraction concentrates.
        ids = torch.randint(0, 50, (256, 128))
        t = torch.full((256,), 0.5)
        masked_ids, info = proc.corrupt(ids, t)

        expected = proc.mask_prob(t)[0].item()
        empirical = info["masked_positions"].float().mean().item()
        assert abs(empirical - expected) < 0.03
        # Masked positions actually carry the mask token id.
        assert (masked_ids[info["masked_positions"]] == 99).all()
        # Unmasked positions are unchanged.
        unmasked = ~info["masked_positions"]
        assert torch.equal(masked_ids[unmasked], ids[unmasked])

    def test_masked_ce_loss_finite_positive(self):
        torch.manual_seed(0)
        vocab = 50
        proc = AbsorbingMaskCorruption(mask_token_id=vocab - 1)
        ids = torch.randint(0, vocab - 1, (4, 16))
        t = torch.full((4,), 0.6)
        _, info = proc.corrupt(ids, t)

        logits = torch.randn(4, 16, vocab)
        loss = proc.loss(logits, info)
        assert torch.isfinite(loss)
        assert loss.item() > 0

    def test_sample_timesteps_range(self):
        proc = AbsorbingMaskCorruption(mask_token_id=99)
        t = proc.sample_timesteps(64, torch.device("cpu"))
        assert t.shape == (64,)
        assert (t > 0).all() and (t <= 1).all()


# ---------------------------------------------------------------------------
# Hybrid corruption (novel).
# ---------------------------------------------------------------------------


class TestHybridCorruption:
    def _embed_fn(self, vocab, dim):
        emb = torch.nn.Embedding(vocab, dim)
        torch.nn.init.normal_(emb.weight, std=0.02)
        return emb

    def test_yields_both_masked_and_noised_positions(self):
        torch.manual_seed(0)
        vocab, dim = 40, 8
        emb = self._embed_fn(vocab, dim)
        proc = HybridCorruption(
            mask_token_id=vocab - 1,
            alphas_cumprod=_toy_alphas_cumprod(100),
            embed_fn=emb,
            mask_weight=0.5,
        )
        ids = torch.randint(0, vocab - 1, (8, 64))
        t = torch.full((8,), 0.8)  # high t -> many masks in the discrete channel
        corrupted, info = proc.corrupt(ids, t)

        assert corrupted.shape == (8, 64, dim)
        # Both channels are populated.
        assert info["discrete_channel"].any()
        assert info["continuous_channel"].any()
        # Discrete and continuous channels partition the positions.
        assert torch.equal(
            info["discrete_channel"] ^ info["continuous_channel"],
            torch.ones_like(info["discrete_channel"]),
        )
        # At high t there is at least one actually-masked position.
        assert info["masked_positions"].any()
        # Masked positions are a subset of the discrete channel.
        assert (info["masked_positions"] & ~info["discrete_channel"]).sum() == 0

    def test_combined_loss_finite_positive(self):
        torch.manual_seed(0)
        vocab, dim = 40, 8
        emb = self._embed_fn(vocab, dim)
        proc = HybridCorruption(
            mask_token_id=vocab - 1,
            alphas_cumprod=_toy_alphas_cumprod(100),
            embed_fn=emb,
            mask_weight=0.5,
        )
        ids = torch.randint(0, vocab - 1, (8, 64))
        t = torch.full((8,), 0.7)
        _, info = proc.corrupt(ids, t)

        logits = torch.randn(8, 64, vocab)
        x0_pred = torch.randn(8, 64, dim)
        loss = proc.loss(logits, info, x0_prediction=x0_pred)
        assert torch.isfinite(loss)
        assert loss.item() > 0

        # CE-only path (no regression head) is also valid and positive.
        loss_ce_only = proc.loss(logits, info)
        assert torch.isfinite(loss_ce_only)
        assert loss_ce_only.item() > 0


# ---------------------------------------------------------------------------
# Masked-diffusion sampler.
# ---------------------------------------------------------------------------


class TestMaskedDiffusionSample:
    def _make_predict_logits(self, vocab):
        """A deterministic toy model: confidently predicts a fixed target id."""
        target = 7

        def predict_logits(ids, t):
            batch, seq = ids.shape
            logits = torch.zeros(batch, seq, vocab)
            logits[:, :, target] = 10.0  # high confidence on `target`
            return logits

        return predict_logits, target

    def test_ends_fully_unmasked_and_keeps_prompt(self):
        torch.manual_seed(0)
        vocab = 20
        mask_id = vocab - 1
        predict_logits, target = self._make_predict_logits(vocab)

        prompt = torch.tensor([[1, 2, 3], [4, 5, 6]])
        gen_len = 10
        out = masked_diffusion_sample(
            predict_logits=predict_logits,
            prompt_ids=prompt,
            gen_len=gen_len,
            mask_token_id=mask_id,
            num_steps=5,
        )

        assert out.shape == (2, gen_len)
        # Fully unmasked: no mask tokens remain.
        assert (out != mask_id).all()
        # The toy model is confident on `target`, so everything resolves to it.
        assert (out == target).all()

    def test_prompt_unchanged_with_remasking(self):
        torch.manual_seed(0)
        vocab = 20
        mask_id = vocab - 1
        target = 7

        prompt = torch.tensor([[1, 2, 3, 4]])
        prompt_len = prompt.shape[1]
        gen_len = 8

        # Capturing model: records the prompt prefix it is handed on every call so
        # we can assert the sampler never overwrites prompt tokens (incl. during
        # low-confidence remasking).
        seen_prompts = []

        def predict_logits(ids, t):
            seen_prompts.append(ids[:, :prompt_len].clone())
            batch, seq = ids.shape
            logits = torch.zeros(batch, seq, vocab)
            logits[:, :, target] = 10.0
            return logits

        out = masked_diffusion_sample(
            predict_logits=predict_logits,
            prompt_ids=prompt,
            gen_len=gen_len,
            mask_token_id=mask_id,
            num_steps=4,
            remask=True,
            remask_fraction=0.25,
        )
        assert out.shape == (1, gen_len)
        assert (out != mask_id).all()
        # The prompt prefix must equal the original prompt on every model call.
        assert len(seen_prompts) > 0
        for seen in seen_prompts:
            assert torch.equal(seen, prompt)

    def test_single_step_reveals_everything(self):
        torch.manual_seed(0)
        vocab = 12
        mask_id = vocab - 1
        predict_logits, target = self._make_predict_logits(vocab)
        prompt = torch.tensor([[2, 3]])
        out = masked_diffusion_sample(
            predict_logits=predict_logits,
            prompt_ids=prompt,
            gen_len=5,
            mask_token_id=mask_id,
            num_steps=1,
        )
        assert (out != mask_id).all()
        assert (out == target).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
