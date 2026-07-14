"""Parity checks for optional Liger fused output-head cross-entropy."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

from dimba.models.diffusion import DIMBA
from dimba.training import fused_ce
from dimba.training.fused_ce import output_head_cross_entropy
from dimba.training.masked import masked_token_loss
from dimba.training.trainer import compute_dimba_losses


def _tiny(**kwargs) -> DIMBA:
    return DIMBA(
        vocab_size=23,
        d_model=8,
        d_prompt=8,
        num_diffusion_steps=8,
        num_denoiser_layers=1,
        d_state=2,
        d_conv=2,
        dropout=0.0,
        use_simple_mamba=True,
        **kwargs,
    )


class _FakeLiger:
    calls = []

    def __init__(self, reduction="mean") -> None:
        self.reduction = reduction

    def __call__(self, weight, features, targets, bias=None):
        self.calls.append((self.reduction, features.shape, bias is not None))
        return F.cross_entropy(
            F.linear(features, weight, bias),
            targets,
            reduction=self.reduction,
        )


@pytest.fixture
def fake_liger(monkeypatch):
    _FakeLiger.calls.clear()
    fused_ce._liger_loss.cache_clear()
    monkeypatch.setattr(fused_ce, "_load_liger_loss", lambda: _FakeLiger)
    monkeypatch.setattr(fused_ce, "_liger_runtime_ready", lambda _features: True)
    yield _FakeLiger
    fused_ce._liger_loss.cache_clear()


@pytest.mark.parametrize(
    "config",
    [
        {"use_weight_tying": False, "use_head_norm": False},
        {"use_weight_tying": True, "use_head_norm": True},
        {
            "use_weight_tying": False,
            "use_head_norm": True,
            "head_type": "attn",
            "head_attn_layers": 1,
            "head_attn_heads": 2,
        },
    ],
)
def test_fake_liger_matches_head_loss_and_backprop_on_cpu(config, fake_liger):
    torch.manual_seed(4)
    reference = _tiny(**config)
    fused = _tiny(**config)
    fused.load_state_dict(reference.state_dict())
    base = torch.randn(2, 5, 8)
    reference_features = base.detach().clone().requires_grad_(True)
    fused_features = base.detach().clone().requires_grad_(True)
    targets = torch.randint(0, 23, (2, 5))

    logits = reference.output_head(reference_features, reference.token_embed.get_weight())
    expected = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
    actual = output_head_cross_entropy(
        fused,
        fused_features,
        targets,
        reduction="mean",
        mode="on",
        uniform_reduction=True,
    )
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)

    expected.backward()
    actual.backward()
    torch.testing.assert_close(fused_features.grad, reference_features.grad, rtol=3e-5, atol=3e-6)
    reference_grads = dict(reference.named_parameters())
    for name, parameter in fused.named_parameters():
        expected_grad = reference_grads[name].grad
        if parameter.grad is None or expected_grad is None:
            assert parameter.grad is expected_grad
        else:
            torch.testing.assert_close(parameter.grad, expected_grad, rtol=3e-5, atol=3e-6)
    assert fake_liger.calls == [("mean", torch.Size([10, 8]), not config["use_weight_tying"])]


def test_uniform_continuous_loss_routes_through_fake_liger(fake_liger):
    model = _tiny(use_weight_tying=True, use_head_norm=True)
    ids = torch.randint(0, 23, (2, 5))
    loss, _ = compute_dimba_losses(
        model,
        ids,
        torch.tensor([2, 4]),
        fused_ce_mode="on",
    )
    loss.backward()
    assert fake_liger.calls == [("mean", torch.Size([10, 8]), False)]
    assert model.token_embed.get_weight().grad is not None


@pytest.mark.parametrize("tied", [False, True])
def test_native_real_head_selects_mask_before_projection_and_preserves_scale(
    tied, monkeypatch
):
    torch.manual_seed(9)
    model = _tiny(use_weight_tying=tied, use_head_norm=True)
    features = torch.randn(2, 5, 8, requires_grad=True)
    targets = torch.randint(0, 23, (2, 5))
    active = torch.tensor(
        [[True, False, True, False, False], [False, True, True, False, True]]
    )
    full_logits = model.output_head(features, model.token_embed.get_weight())
    expected = F.cross_entropy(full_logits[active], targets[active], reduction="none")

    original_linear = F.linear
    projected_rows = []

    def recording_linear(values, weight, bias=None):
        projected_rows.append(values.shape[0])
        return original_linear(values, weight, bias)

    monkeypatch.setattr(fused_ce.F, "linear", recording_linear)
    actual = output_head_cross_entropy(
        model,
        features,
        targets,
        active_mask=active,
        reduction="none",
        mode="off",
    )
    torch.testing.assert_close(actual, expected)
    assert projected_rows == [int(active.sum())]
    actual.mean().backward()
    assert torch.isfinite(features.grad).all()


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_continuous_native_mask_and_time_fade_match_full_logits(tied, dtype):
    torch.manual_seed(13)
    model = _tiny(use_weight_tying=tied, use_head_norm=True).to(dtype)
    ids = torch.randint(0, 23, (2, 6))
    timesteps = torch.tensor([1, 6])
    prompt_mask = torch.tensor(
        [[True, True, False, False, False, False], [True, False, False, False, False, False]]
    )
    loss_mask = torch.tensor(
        [[1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 1, 0]], dtype=torch.bool
    )

    torch.manual_seed(22)
    loss, parts = compute_dimba_losses(
        model,
        ids,
        timesteps,
        prompt_mask=prompt_mask,
        loss_mask=loss_mask,
        ce_time_fade=True,
        fused_ce_mode="off",
    )
    torch.manual_seed(22)
    x_pred, _noise, info = model(ids, timesteps, prompt_mask=prompt_mask)
    effective = info["diffuse_mask"].float() * loss_mask.float()
    logits = model.output_head(x_pred)
    full_ce = F.cross_entropy(logits.flatten(0, 1), ids.flatten(), reduction="none").view_as(ids)
    ce_per_sample = (full_ce * effective).sum(1) / effective.sum(1).clamp(min=1.0)
    fade = 1.0 - timesteps.float() / (model.num_diffusion_steps - 1)
    expected_ce = (ce_per_sample * fade).mean()

    torch.testing.assert_close(parts["ce_loss"], expected_ce, rtol=1e-5, atol=1e-6)
    assert parts["ce_loss"].dtype == torch.float32
    loss.backward()
    assert torch.isfinite(loss)


def test_masked_unlikelihood_stays_native_and_mode_on_rejects_unsafe_gradients(fake_liger):
    model = _tiny(use_weight_tying=True, use_head_norm=True)
    targets = torch.randint(0, 22, (2, 5))
    target_mask = torch.tensor(
        [[True, False, True, False, True], [False, True, False, True, False]]
    )
    corrupted = targets.masked_fill(target_mask, 22)
    loss, _ = masked_token_loss(
        model,
        corrupted,
        targets,
        target_mask,
        torch.tensor([0.5, 0.75]),
        mask_token_id=22,
        neighbor_unlikelihood_weight=0.5,
        fused_ce_mode="auto",
    )
    loss.backward()
    assert not fake_liger.calls

    with pytest.raises(RuntimeError, match="cannot preserve masked per-sample"):
        masked_token_loss(
            model,
            corrupted,
            targets,
            target_mask,
            torch.tensor([0.5, 0.75]),
            fused_ce_mode="on",
        )


def test_mode_on_rejects_upstream_unreduced_backward_even_with_fake_liger(fake_liger):
    model = _tiny()
    with pytest.raises(RuntimeError, match="non-uniform"):
        output_head_cross_entropy(
            model,
            torch.randn(3, 8),
            torch.randint(0, 23, (3,)),
            prepared=True,
            reduction="none",
            mode="on",
            uniform_reduction=False,
        )
    assert not fake_liger.calls


def test_wrapped_projection_uses_native_fallback(fake_liger):
    class WrappedLinear(nn.Module):
        def __init__(self, linear: nn.Linear) -> None:
            super().__init__()
            self.linear = linear

        def forward(self, values: torch.Tensor) -> torch.Tensor:
            return self.linear(values)

    torch.manual_seed(31)
    model = _tiny(use_weight_tying=False, use_head_norm=True)
    model.output_head.projection = WrappedLinear(model.output_head.projection)
    features = torch.randn(2, 4, 8, requires_grad=True)
    targets = torch.randint(0, 23, (2, 4))
    expected = F.cross_entropy(model.output_head(features).flatten(0, 1), targets.flatten())
    actual = output_head_cross_entropy(
        model,
        features,
        targets,
        reduction="mean",
        mode="auto",
        uniform_reduction=True,
    )
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(features.grad).all()
    assert not fake_liger.calls

    with pytest.raises(RuntimeError, match="tied or nn.Linear"):
        output_head_cross_entropy(
            model,
            features.detach(),
            targets,
            reduction="mean",
            mode="on",
            uniform_reduction=True,
        )
