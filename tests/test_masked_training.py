"""Parity checks for memory-efficient masked training."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from dimba.training.masked import masked_token_loss, sample_masked_inputs


class _TinyMaskedModel(nn.Module):
    def __init__(self, vocab: int = 17, dim: int = 8):
        super().__init__()
        self.embedding = nn.Embedding(vocab, dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        self.projected_tokens = 0

    def predict_token_features(self, ids, t):
        return self.embedding(ids)

    def project_token_features(self, features, positions=None):
        assert positions is None
        self.projected_tokens = features.shape[0]
        return self.head(features)


def test_masked_loss_matches_full_projection_reference():
    torch.manual_seed(0)
    model = _TinyMaskedModel()
    targets = torch.randint(0, 17, (3, 6))
    mask = torch.tensor(
        [
            [True, False, True, False, False, True],
            [False, True, False, True, False, False],
            [True, True, False, False, True, False],
        ]
    )
    corrupted = targets.masked_fill(mask, 16)
    t = torch.tensor([0.5, 0.7, 0.9])

    loss, _ = masked_token_loss(model, corrupted, targets, mask, t)

    full_logits = model.head(model.embedding(corrupted))
    full_ce = F.cross_entropy(
        full_logits.flatten(0, 1), targets.flatten(), reduction="none"
    ).view_as(targets)
    expected_per_sample = (full_ce * mask).sum(1) / mask.sum(1)
    expected = (expected_per_sample / t).mean()

    assert torch.allclose(loss, expected, atol=1e-6, rtol=1e-6)
    assert model.projected_tokens == int(mask.sum())


def test_bfloat16_masked_loss_accumulates_like_float32_reference():
    """Selected projection must not change the original fp32 loss reduction."""
    torch.manual_seed(3)
    model = _TinyMaskedModel(vocab=31, dim=16).to(torch.bfloat16)
    targets = torch.randint(0, 31, (8, 256))
    mask = torch.rand_like(targets, dtype=torch.float32) < 0.8
    corrupted = targets.masked_fill(mask, 30)
    t = torch.linspace(0.1, 0.9, targets.shape[0])

    loss, _ = masked_token_loss(model, corrupted, targets, mask, t)

    full_logits = model.head(model.embedding(corrupted))
    token_ce = F.cross_entropy(
        full_logits.flatten(0, 1), targets.flatten(), reduction="none"
    ).view_as(targets)
    expected_per_sample = (token_ce.float() * mask.float()).sum(1) / mask.sum(1).float()
    expected = (expected_per_sample / t).mean()

    assert loss.dtype == torch.float32
    assert torch.allclose(loss, expected, atol=1e-6, rtol=1e-6)


def test_mask_sampling_guarantees_one_eligible_target_without_host_branch():
    ids = torch.arange(12).view(2, 6)
    eligible = torch.tensor(
        [[False, False, True, True, False, False], [False, True, True, False, False, False]]
    )
    generator = torch.Generator().manual_seed(0)
    corrupted, t, mask = sample_masked_inputs(
        ids, 99, t_min=1e-9, eligible_mask=eligible, generator=generator
    )

    assert mask.any(dim=1).all()
    assert not (mask & ~eligible).any()
    assert torch.equal(corrupted.masked_select(mask), torch.full((int(mask.sum()),), 99))
    assert (t > 0).all()


def test_neighbor_unlikelihood_is_finite_and_backpropagates():
    torch.manual_seed(1)
    model = _TinyMaskedModel()
    targets = torch.randint(0, 16, (2, 5))
    mask = torch.tensor([[False, True, True, False, False], [True, False, True, False, True]])
    corrupted = targets.masked_fill(mask, 16)
    loss, parts = masked_token_loss(
        model,
        corrupted,
        targets,
        mask,
        torch.tensor([0.4, 0.8]),
        mask_token_id=16,
        neighbor_unlikelihood_weight=0.5,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(parts["ul"])
    assert model.head.weight.grad is not None
