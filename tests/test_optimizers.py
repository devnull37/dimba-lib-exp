"""Focused checks for the production Muon optimizer path."""

import copy

import torch
import torch.nn as nn

from dimba.models.diffusion import DIMBA
from dimba.training.optimizers import HybridMuon, build_optimizer, split_muon_parameters


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.A_log = nn.Parameter(torch.zeros(2, 2))
        self.lora_A = nn.Parameter(torch.randn(2, 6))
        self.token_embed = nn.Embedding(11, 4)
        self.hidden = nn.Linear(4, 6)
        self.norm = nn.LayerNorm(6)
        self.output_head = nn.Linear(6, 11)


def _set_gradients(model: nn.Module, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    for parameter in model.parameters():
        parameter.grad = torch.randn(
            parameter.shape,
            dtype=parameter.dtype,
            device=parameter.device,
            generator=generator,
        )


def test_split_keeps_only_hidden_matrices_on_muon() -> None:
    muon, adamw = split_muon_parameters(_ToyModel())
    muon_names = {name for name, _ in muon}
    adamw_names = {name for name, _ in adamw}

    assert muon_names == {"lora_A", "hidden.weight"}
    assert {
        "A_log",
        "token_embed.weight",
        "hidden.bias",
        "norm.weight",
        "norm.bias",
        "output_head.weight",
        "output_head.bias",
    } <= adamw_names


def test_dimba_attention_head_hidden_matrices_use_muon_not_vocab_projection() -> None:
    model = DIMBA(
        vocab_size=32,
        d_model=16,
        d_prompt=16,
        num_diffusion_steps=8,
        num_denoiser_layers=1,
        d_state=4,
        d_conv=2,
        dropout=0.0,
        use_simple_mamba=True,
        head_type="attn",
        head_attn_layers=1,
        head_attn_heads=4,
    )
    muon, adamw = split_muon_parameters(model)
    muon_names = {name for name, _ in muon}
    adamw_names = {name for name, _ in adamw}

    assert "output_head.attn_blocks.0.q_proj.weight" in muon_names
    assert "output_head.projection.weight" in adamw_names


def test_hybrid_muon_steps_and_round_trips_standard_optimizer_state() -> None:
    torch.manual_seed(7)
    first = _ToyModel()
    first_optimizer = build_optimizer(
        first,
        name="muon",
        lr=1e-3,
        weight_decay=0.01,
        fused=False,
    )
    assert isinstance(first_optimizer, HybridMuon)

    before = {name: parameter.detach().clone() for name, parameter in first.named_parameters()}
    _set_gradients(first, seed=11)
    first_optimizer.step()
    assert all(
        not torch.equal(before[name], parameter)
        for name, parameter in first.named_parameters()
    )

    second = _ToyModel()
    second.load_state_dict(first.state_dict())
    second_optimizer = build_optimizer(
        second,
        name="muon",
        lr=1e-3,
        weight_decay=0.01,
        fused=False,
    )
    second_optimizer.load_state_dict(copy.deepcopy(first_optimizer.state_dict()))

    _set_gradients(first, seed=23)
    _set_gradients(second, seed=23)
    first_optimizer.step()
    second_optimizer.step()

    for first_parameter, second_parameter in zip(first.parameters(), second.parameters()):
        assert torch.equal(first_parameter, second_parameter)


def test_hybrid_muon_exposes_both_groups_to_one_scheduler() -> None:
    optimizer = build_optimizer(
        _ToyModel(),
        name="muon",
        lr=1e-3,
        fused=False,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 0.25)

    assert len(optimizer.param_groups) == 2
    assert all(group["lr"] == 2.5e-4 for group in optimizer.param_groups)
    assert optimizer.muon.param_groups[0] is optimizer.param_groups[0]
    assert optimizer.adamw.param_groups[0] is optimizer.param_groups[1]
    optimizer.step()
    scheduler.step()
    assert all(group["lr"] == 2.5e-4 for group in optimizer.param_groups)


def test_adamw_remains_the_default_builder_path() -> None:
    optimizer = build_optimizer(_ToyModel(), lr=1e-3)
    assert isinstance(optimizer, torch.optim.AdamW)
