"""Trace-level regression for the canonical GRPO training scorer."""

import copy

import pytest
import torch

from dimba.models.diffusion import DIMBA
from dimba.training import fused_ce
from dimba.training import grpo as grpo_module
from dimba.training.grpo import GRPOConfig, GRPOTrainer


def test_train_step_shares_nonclean_trajectories_and_selected_projection(monkeypatch) -> None:
    torch.manual_seed(23)
    policy = DIMBA(
        vocab_size=20,
        d_model=12,
        d_prompt=12,
        num_diffusion_steps=8,
        num_denoiser_layers=1,
        d_state=3,
        dropout=0.8,
        use_simple_mamba=True,
    )
    reference = copy.deepcopy(policy)
    config = GRPOConfig(
        group_size=2,
        mc_samples=2,
        antithetic=True,
        use_mgpo=False,
        sampler="ddim",
        response_len=2,
        warmup_steps=0,
        total_steps=2,
        log_interval=999,
        save_interval=999,
    )
    trainer = GRPOTrainer(
        policy,
        reward_fn=lambda *_args: 0.0,
        tokenizer=object(),
        config=config,
        ref_model=reference,
    )

    monkeypatch.setattr(
        trainer,
        "_encode",
        lambda text: [1] if text == "short" else [2, 3, 4],
    )
    monkeypatch.setattr(
        trainer,
        "_score_completions",
        lambda *_args: (
            torch.tensor([1.0, 0.0, 0.0, 1.0]),
            torch.tensor([1.0, 0.0, 0.0, 1.0]),
        ),
    )

    sampled_prompts = []

    def fake_block_sample(_model, prompts, *_args, **_kwargs):
        sampled_prompts.append(prompts.detach().clone())
        first = (prompts[:, 0] % 5) + 10
        response = torch.stack((first, first + 1), dim=1)
        return {
            "full_ids": torch.cat((prompts, response), dim=1),
            "response": response,
            "n_think_blocks": 0,
        }

    monkeypatch.setattr(grpo_module, "block_sample_from_model", fake_block_sample)

    policy_calls = []
    reference_calls = []

    def record(calls):
        def hook(module, args, kwargs, _output):
            calls.append(
                {
                    "timesteps": args[1].detach().clone(),
                    "noise": kwargs["noise"].detach().clone(),
                    "prompt_mask": kwargs["prompt_mask"].detach().clone(),
                    "training": module.training,
                }
            )

        return hook

    policy.register_forward_hook(record(policy_calls), with_kwargs=True)
    reference.register_forward_hook(record(reference_calls), with_kwargs=True)

    projected_shapes = []
    original_cross_entropy = fused_ce.F.cross_entropy

    def recording_cross_entropy(logits, targets, **kwargs):
        projected_shapes.append(tuple(logits.shape))
        return original_cross_entropy(logits, targets, **kwargs)

    monkeypatch.setattr(fused_ce.F, "cross_entropy", recording_cross_entropy)

    metrics = trainer.train_step(["short", "long"], ["a", "b"])

    assert [prompt.shape[1] for prompt in sampled_prompts] == [1, 1, 3, 3]
    assert sampled_prompts[0].tolist() == sampled_prompts[1].tolist() == [[1]]
    assert sampled_prompts[2].tolist() == sampled_prompts[3].tolist() == [[2, 3, 4]]

    assert len(policy_calls) == len(reference_calls) == config.mc_samples
    for policy_call, reference_call in zip(policy_calls, reference_calls):
        assert not policy_call["training"] and not reference_call["training"]
        assert torch.all((policy_call["timesteps"] >= 1) & (policy_call["timesteps"] < 8))
        torch.testing.assert_close(policy_call["timesteps"], reference_call["timesteps"])
        torch.testing.assert_close(policy_call["noise"], reference_call["noise"])
        torch.testing.assert_close(policy_call["prompt_mask"], reference_call["prompt_mask"])
        assert policy_call["prompt_mask"].tolist() == [
            [True, False, False, False, False],
            [True, False, False, False, False],
            [True, True, True, False, False],
            [True, True, True, False, False],
        ]

    assert projected_shapes == [(8, policy.vocab_size)] * (config.mc_samples * 2)
    assert metrics["kl"] == pytest.approx(0.0, abs=1e-7)
    assert policy.training
    assert not reference.training
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in policy.parameters()
    )
    assert all(parameter.grad is None for parameter in reference.parameters())
