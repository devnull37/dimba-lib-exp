"""Regression tests for GRPO generation's shared-sampler delegation."""

import copy
from types import SimpleNamespace

import pytest
import torch

from scripts.finetuning import finetune_grpo


class _StubModel:
    def __init__(self, *, flow: bool) -> None:
        self.use_flow_matching = flow
        self.eval_called = False

    def eval(self):
        self.eval_called = True
        return self

    def encode_prompt(self, *_args, **_kwargs):
        raise AssertionError("the retired GRPO sampler path was used")

    def denoise_step(self, *_args, **_kwargs):
        raise AssertionError("the retired GRPO sampler path was used")


def test_generate_quiet_delegates_to_ddim_with_clean_prompts(monkeypatch) -> None:
    calls = []

    def fake_ddim(model, prompt_ids, seq_len, **kwargs):
        calls.append((model, prompt_ids.clone(), seq_len, kwargs))
        token = 40 + prompt_ids.shape[1]
        return torch.full(
            (prompt_ids.shape[0], seq_len),
            token,
            dtype=torch.long,
            device=prompt_ids.device,
        )

    def unexpected_flow(*_args, **_kwargs):
        raise AssertionError("flow sampler used for a diffusion model")

    monkeypatch.setattr(finetune_grpo, "sample_from_model", fake_ddim)
    monkeypatch.setattr(finetune_grpo, "sample_from_model_flow", unexpected_flow)

    model = _StubModel(flow=False)
    original_prompts = torch.tensor([[1, 2, 0], [3, 4, 5]])
    prompt_lens = torch.tensor([2, 3])
    repeated_prompts = original_prompts.repeat_interleave(2, dim=0)
    repeated_lens = prompt_lens.repeat_interleave(2)

    generated = finetune_grpo.generate_quiet(
        model,
        repeated_prompts,
        repeated_lens,
        max_new_tokens=2,
        max_seq_len=5,
        num_steps=7,
        temperature=0.8,
        top_k=11,
        top_p=0.9,
        device=torch.device("cpu"),
        pad_token_id=0,
    )

    assert model.eval_called
    assert len(calls) == 2
    assert calls[0][1].tolist() == [[1, 2], [1, 2]]
    assert calls[1][1].tolist() == [[3, 4, 5], [3, 4, 5]]
    for _, clean_prompts, response_len, kwargs in calls:
        assert clean_prompts.shape[1] in {2, 3}
        assert response_len == 2
        assert kwargs == {
            "num_steps": 7,
            "temperature": 0.8,
            "top_k": 11,
            "top_p": 0.9,
            "guidance_scale": 1.0,
            "eta": 0.0,
            "device": torch.device("cpu"),
            "sampler": "ddim",
        }

    assert generated.tolist() == [
        [1, 2, 42, 42, 0],
        [1, 2, 42, 42, 0],
        [3, 4, 5, 43, 43],
        [3, 4, 5, 43, 43],
    ]

    eval_ids, completion_mask, completions = finetune_grpo.build_eval_inputs(
        original_prompts,
        prompt_lens,
        generated,
        num_generations=2,
        max_new_tokens=2,
        max_seq_len=5,
        pad=0,
    )
    assert eval_ids.tolist() == generated.tolist()
    assert completion_mask.tolist() == [
        [0.0, 0.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 1.0, 1.0],
    ]
    assert completions == [[42, 42], [42, 42], [43, 43], [43, 43]]


def test_generate_quiet_delegates_to_flow_and_honors_sequence_cap(monkeypatch) -> None:
    calls = []

    def fake_flow(model, prompt_ids, seq_len, **kwargs):
        calls.append((model, prompt_ids.clone(), seq_len, kwargs))
        return torch.full(
            (prompt_ids.shape[0], seq_len),
            9,
            dtype=torch.long,
            device=prompt_ids.device,
        )

    def unexpected_ddim(*_args, **_kwargs):
        raise AssertionError("DDIM sampler used for a flow model")

    monkeypatch.setattr(finetune_grpo, "sample_from_model_flow", fake_flow)
    monkeypatch.setattr(finetune_grpo, "sample_from_model", unexpected_ddim)

    model = _StubModel(flow=True)
    generated = finetune_grpo.generate_quiet(
        model,
        torch.tensor([[1, 2, 3], [4, 5, 6]]),
        torch.tensor([3, 3]),
        max_new_tokens=4,
        max_seq_len=5,
        num_steps=0,
        temperature=1.0,
        top_k=None,
        top_p=None,
        device=torch.device("cpu"),
        pad_token_id=0,
    )

    assert generated.tolist() == [[1, 2, 3, 9, 9], [4, 5, 6, 9, 9]]
    assert len(calls) == 1
    _, clean_prompts, response_len, kwargs = calls[0]
    assert clean_prompts.tolist() == [[1, 2, 3], [4, 5, 6]]
    assert response_len == 2
    assert kwargs == {
        "num_steps": 1,
        "sampler": "euler",
        "temperature": 1.0,
        "top_k": None,
        "top_p": None,
        "guidance_scale": 1.0,
        "device": torch.device("cpu"),
    }


def test_completion_scoring_has_no_response_conditioning_leak(monkeypatch) -> None:
    torch.manual_seed(9)
    model = finetune_grpo.DIMBA(
        vocab_size=24,
        d_model=16,
        d_prompt=16,
        num_diffusion_steps=8,
        num_denoiser_layers=1,
        d_state=4,
        dropout=0.0,
        use_simple_mamba=True,
    ).eval()
    first = torch.tensor([[1, 2, 7, 8, 0], [3, 9, 10, 0, 0]])
    second = torch.tensor([[1, 2, 11, 12, 0], [3, 13, 14, 0, 0]])
    prompt_mask = torch.tensor(
        [[True, True, False, False, False], [True, False, False, False, False]]
    )
    completion_mask = torch.tensor(
        [[False, False, True, True, False], [False, True, True, False, False]]
    )
    noise = torch.randn(2, 5, model.d_latent)
    timesteps = torch.tensor([3, 6])

    pooled_conditions = []
    projected_shapes = []
    original_pool = model._pooled_prompt
    original_project = model.output_head.project_features

    def recording_pool(input_ids, mask):
        pooled = original_pool(input_ids, mask)
        pooled_conditions.append(pooled.detach().clone())
        return pooled

    def recording_project(features, embedding_weight=None, positions=None):
        projected_shapes.append(tuple(features.shape))
        return original_project(features, embedding_weight, positions)

    monkeypatch.setattr(model, "_pooled_prompt", recording_pool)
    monkeypatch.setattr(model.output_head, "project_features", recording_project)

    first_logits, first_noise = finetune_grpo.model_completion_logits(
        model,
        first,
        prompt_mask,
        completion_mask,
        timesteps,
        noise=noise,
    )
    second_logits, second_noise = finetune_grpo.model_completion_logits(
        model,
        second,
        prompt_mask,
        completion_mask,
        timesteps,
        noise=noise,
    )

    assert first_logits.shape == second_logits.shape == (4, model.vocab_size)
    assert projected_shapes == [(4, model.d_model), (4, model.d_model)]
    torch.testing.assert_close(pooled_conditions[0], pooled_conditions[1])
    torch.testing.assert_close(first_noise, noise)
    torch.testing.assert_close(second_noise, noise)


def test_grpo_timesteps_exclude_clean_self_copy_endpoint(monkeypatch) -> None:
    model = finetune_grpo.DIMBA(
        vocab_size=16,
        d_model=8,
        d_prompt=8,
        num_diffusion_steps=4,
        num_denoiser_layers=1,
        d_state=2,
        dropout=0.0,
        use_simple_mamba=True,
    ).eval()
    call = {}

    def recording_randint(low, high, size, *, device):
        call.update(low=low, high=high, size=size, device=device)
        return torch.ones(size, dtype=torch.long, device=device)

    monkeypatch.setattr(torch, "randint", recording_randint)
    sampled = finetune_grpo.sample_grpo_timesteps(model, 3, torch.device("cpu"))

    assert call == {"low": 1, "high": 4, "size": (3,), "device": torch.device("cpu")}
    assert sampled.tolist() == [1, 1, 1]


def test_completion_scoring_matches_frozen_reference_with_dropout_disabled() -> None:
    torch.manual_seed(12)
    policy = finetune_grpo.DIMBA(
        vocab_size=20,
        d_model=12,
        d_prompt=12,
        num_diffusion_steps=6,
        num_denoiser_layers=1,
        d_state=3,
        dropout=0.8,
        use_simple_mamba=True,
    ).train()
    reference = copy.deepcopy(policy).eval()
    input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 0]])
    prompt_mask = torch.tensor(
        [[True, True, False, False], [True, False, False, False]]
    )
    completion_mask = torch.tensor(
        [[False, False, True, True], [False, True, True, False]]
    )
    timesteps = torch.tensor([2, 5])
    noise = torch.randn(2, 4, policy.d_latent)

    policy_logits, used_noise = finetune_grpo.model_completion_logits(
        policy, input_ids, prompt_mask, completion_mask, timesteps, noise=noise
    )
    with torch.no_grad():
        reference_logits, _ = finetune_grpo.model_completion_logits(
            reference, input_ids, prompt_mask, completion_mask, timesteps, noise=used_noise
        )

    torch.testing.assert_close(policy_logits, reference_logits)
    assert policy.training
    assert not reference.training
    policy_logits.sum().backward()
    gradient = policy.output_head.projection.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()


def test_packed_grpo_loss_matches_dense_values_and_gradients() -> None:
    torch.manual_seed(21)
    batch_size, seq_len, vocab_size = 3, 5, 11
    dense_policy = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
    dense_ref = torch.randn(batch_size, seq_len, vocab_size)
    target_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    completion_mask = torch.tensor(
        [
            [False, True, True, False, False],
            [False, False, True, True, True],
            [True, False, False, False, False],
        ]
    )
    advantage = torch.tensor([0.7, -1.2, 0.4])
    beta = 0.13

    dense_policy_lp = torch.log_softmax(dense_policy.float(), dim=-1)
    dense_ref_lp = torch.log_softmax(dense_ref.float(), dim=-1)
    dense_token_lp = dense_policy_lp.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    den = completion_mask.sum(dim=1).clamp_min(1)
    dense_seq_lp = (dense_token_lp * completion_mask).sum(dim=1) / den
    dense_token_kl = (
        dense_policy_lp.exp() * (dense_policy_lp - dense_ref_lp)
    ).sum(dim=-1)
    dense_seq_kl = (dense_token_kl * completion_mask).sum(dim=1) / den
    dense_loss = -(advantage * dense_seq_lp - beta * dense_seq_kl).mean()
    dense_loss.backward()
    expected_gradient = dense_policy.grad.detach().clone()

    packed_source = dense_policy.detach().clone().requires_grad_(True)
    sequence_ids = completion_mask.nonzero(as_tuple=True)[0]
    packed_loss, stats = finetune_grpo.grpo_loss(
        packed_source[completion_mask],
        dense_ref[completion_mask],
        target_ids[completion_mask],
        sequence_ids,
        advantage,
        beta,
    )
    packed_loss.backward()

    torch.testing.assert_close(packed_loss, dense_loss.detach())
    torch.testing.assert_close(stats["logp"], dense_seq_lp.mean().detach())
    torch.testing.assert_close(stats["kl"], dense_seq_kl.mean().detach())
    torch.testing.assert_close(stats["adv_abs"], advantage.abs().mean())
    torch.testing.assert_close(packed_source.grad, expected_gradient)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"num_generations": 1}, "num-generations"),
        ({"max_new_tokens": 0}, "max-new-tokens"),
        ({"max_new_tokens": 32}, "max-new-tokens"),
        ({"sampling_steps": 0}, "sampling-steps"),
        ({"learning_rate": 0.0}, "learning-rate"),
        ({"temperature": 0.0}, "temperature"),
        ({"top_p": 0.0}, "top-p"),
    ],
)
def test_grpo_rejects_noop_or_invalid_arguments(override, message) -> None:
    values = {
        "num_generations": 4,
        "max_new_tokens": 8,
        "max_seq_len": 32,
        "batch_size": 2,
        "epochs": 1,
        "sampling_steps": 8,
        "log_every": 1,
        "save_every": 10,
        "learning_rate": 1e-5,
        "temperature": 1.0,
        "beta": 0.1,
        "top_p": 0.95,
    }
    values.update(override)
    with pytest.raises(ValueError, match=message):
        finetune_grpo.validate_args(SimpleNamespace(**values))
