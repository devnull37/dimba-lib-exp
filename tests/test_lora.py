"""Unit tests for LoRA finetuning utilities."""

import pytest
import torch
import torch.nn as nn

from dimba.models.lora import (
    LoRALinear,
    inject_lora_to_model,
    load_lora_weights,
    merge_lora_weights,
    save_lora_weights,
)


class TinyBlock(nn.Module):
    """Small block with both target and non-target linear layers."""

    def __init__(self) -> None:
        super().__init__()
        self.out_proj = nn.Linear(8, 8)
        self.ff = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff(torch.relu(self.out_proj(x)))


class TinyModel(nn.Module):
    """Small model used for LoRA injection and roundtrip checks."""

    def __init__(self) -> None:
        super().__init__()
        self.in_proj = nn.Linear(8, 8)
        self.block = TinyBlock()
        self.final = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        x = self.block(x)
        return self.final(x)


def _build_model(seed: int) -> TinyModel:
    torch.manual_seed(seed)
    return TinyModel()


def test_inject_lora_wraps_targets_and_preserves_forward_shape() -> None:
    model = _build_model(seed=123)
    x = torch.randn(2, 5, 8)
    baseline = model(x)

    wrapped = inject_lora_to_model(
        model,
        target_modules=("in_proj", "out_proj"),
        r=2,
        alpha=4.0,
        dropout=0.0,
    )

    assert set(wrapped) == {"in_proj", "block.out_proj"}
    assert isinstance(model.in_proj, LoRALinear)
    assert isinstance(model.block.out_proj, LoRALinear)
    assert isinstance(model.block.ff, nn.Linear)
    assert isinstance(model.final, nn.Linear)

    after_injection = model(x)
    assert after_injection.shape == baseline.shape
    # LoRA B is zero-initialized, so injection should not change outputs initially.
    assert torch.allclose(after_injection, baseline, atol=1e-6)

    trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]
    assert trainable_names
    assert all(name.endswith(("lora_A", "lora_B")) for name in trainable_names)


def test_lora_save_load_and_merge_roundtrip(tmp_path) -> None:
    x = torch.randn(2, 3, 8)

    source_model = _build_model(seed=7)
    inject_lora_to_model(source_model, target_modules=("in_proj",), r=2, alpha=6.0)
    source_lora = source_model.in_proj
    assert isinstance(source_lora, LoRALinear)

    with torch.no_grad():
        source_lora.lora_A.copy_(
            torch.arange(source_lora.lora_A.numel(), dtype=torch.float32).view_as(source_lora.lora_A) / 10.0
        )
        source_lora.lora_B.copy_(
            torch.arange(source_lora.lora_B.numel(), dtype=torch.float32).view_as(source_lora.lora_B) / 20.0
        )

    expected = source_model(x)
    ckpt_path = tmp_path / "lora.pt"
    saved = save_lora_weights(source_model, ckpt_path)
    assert ckpt_path.exists()
    assert "in_proj.lora_A" in saved
    assert "in_proj.lora_B" in saved

    target_model = _build_model(seed=7)
    inject_lora_to_model(target_model, target_modules=("in_proj",), r=2, alpha=1.0)
    report = load_lora_weights(target_model, ckpt_path, strict=True)
    assert report == {
        "loaded": ["in_proj"],
        "missing": [],
        "unexpected": [],
        "mismatched": [],
    }

    target_lora = target_model.in_proj
    assert isinstance(target_lora, LoRALinear)
    assert torch.allclose(target_lora.lora_A, source_lora.lora_A, atol=0, rtol=0)
    assert torch.allclose(target_lora.lora_B, source_lora.lora_B, atol=0, rtol=0)
    assert target_lora.alpha == pytest.approx(source_lora.alpha)

    loaded_output = target_model(x)
    assert torch.allclose(loaded_output, expected, atol=1e-6)

    pre_merge = target_model(x)
    merged = merge_lora_weights(target_model)
    post_merge = target_model(x)

    assert merged == ["in_proj"]
    assert target_lora.merged is True
    assert torch.allclose(post_merge, pre_merge, atol=1e-6)


def test_lora_load_accepts_module_prefix_mismatch(tmp_path) -> None:
    x = torch.randn(2, 3, 8)

    source_model = _build_model(seed=17)
    inject_lora_to_model(source_model, target_modules=("in_proj",), r=2, alpha=5.0)
    source_lora = source_model.in_proj
    assert isinstance(source_lora, LoRALinear)

    with torch.no_grad():
        source_lora.lora_A.copy_(
            torch.arange(source_lora.lora_A.numel(), dtype=torch.float32).view_as(source_lora.lora_A) / 13.0
        )
        source_lora.lora_B.copy_(
            torch.arange(source_lora.lora_B.numel(), dtype=torch.float32).view_as(source_lora.lora_B) / 17.0
        )

    expected = source_model(x)
    ckpt_path = tmp_path / "lora_module_prefixed.pt"
    raw_state = save_lora_weights(source_model, ckpt_path)
    module_prefixed_state = {f"module.{key}": value for key, value in raw_state.items()}
    torch.save({"lora_state_dict": module_prefixed_state}, ckpt_path)

    target_model = _build_model(seed=17)
    inject_lora_to_model(target_model, target_modules=("in_proj",), r=2, alpha=1.0)
    report = load_lora_weights(target_model, ckpt_path, strict=True)

    assert report == {
        "loaded": ["in_proj"],
        "missing": [],
        "unexpected": [],
        "mismatched": [],
    }

    target_lora = target_model.in_proj
    assert isinstance(target_lora, LoRALinear)
    assert torch.allclose(target_lora.lora_A, source_lora.lora_A, atol=0, rtol=0)
    assert torch.allclose(target_lora.lora_B, source_lora.lora_B, atol=0, rtol=0)
    assert target_lora.alpha == pytest.approx(source_lora.alpha)
    assert torch.allclose(target_model(x), expected, atol=1e-6)
