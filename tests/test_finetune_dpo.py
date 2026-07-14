"""Focused regressions for leak-free, response-only DPO scoring."""

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from dimba.training import preference
from dimba.training.preference import elbo_sequence_logprob
from scripts.finetuning import finetune_dpo


class _TokenEmbedding(nn.Embedding):
    def get_weight(self) -> torch.Tensor:
        return self.weight


class _Head(nn.Module):
    def __init__(self, width: int, vocab_size: int) -> None:
        super().__init__()
        self.use_weight_tying = False
        self.use_norm = False
        self.projection = nn.Linear(width, vocab_size)

    def prepare_features(self, features: torch.Tensor, _weight: torch.Tensor) -> torch.Tensor:
        return features

    def forward(self, features: torch.Tensor, **_kwargs) -> torch.Tensor:
        return self.projection(self.prepare_features(features, torch.empty(0)))


class _RecordingDIMBA(nn.Module):
    num_diffusion_steps = 8

    def __init__(self, vocab_size: int = 11, width: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.d_latent = width
        self.token_embed = _TokenEmbedding(vocab_size, width)
        self.output_head = _Head(width, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.prompt_tokens = []
        self.prompt_masks = []

    def forward(
        self,
        input_ids: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        noise=None,
        prompt_mask=None,
        return_latent_info=True,
    ):
        del return_latent_info
        assert prompt_mask is not None
        prompt_mask = prompt_mask.bool()
        self.prompt_masks.append(prompt_mask.detach().clone())
        self.prompt_tokens.append(
            [input_ids[row][prompt_mask[row]].tolist() for row in range(input_ids.shape[0])]
        )
        features = self.dropout(self.token_embed(input_ids))
        features = features + timesteps.to(features.dtype).view(-1, 1, 1) / self.num_diffusion_steps
        return features, noise, {}


def _inputs():
    input_ids = torch.tensor([[1, 2, 3, 4, 0], [5, 6, 7, 0, 0]])
    prompt_mask = torch.tensor(
        [[True, True, False, False, False], [True, False, False, False, False]]
    )
    response_mask = torch.tensor(
        [[False, False, True, True, False], [False, True, True, False, False]]
    )
    labels = input_ids.clone()
    labels[~response_mask] = -100
    return input_ids, labels, prompt_mask, response_mask


def test_policy_logprob_conditions_only_on_prompt_and_projects_only_response(monkeypatch) -> None:
    model = _RecordingDIMBA()
    input_ids, labels, prompt_mask, response_mask = _inputs()
    projected_shapes = []
    cross_entropy = preference.F.cross_entropy

    def recording_cross_entropy(logits, targets, **kwargs):
        projected_shapes.append(tuple(logits.shape))
        return cross_entropy(logits, targets, **kwargs)

    monkeypatch.setattr(preference.F, "cross_entropy", recording_cross_entropy)
    score = finetune_dpo.policy_logprob(
        model,
        input_ids,
        labels,
        prompt_mask,
        response_mask,
        num_mc_samples=1,
        antithetic=False,
    )

    assert score.shape == (2,)
    assert len(model.prompt_masks) == 1
    assert torch.equal(model.prompt_masks[0], prompt_mask)
    assert model.prompt_tokens == [[[1, 2], [5]]]
    assert projected_shapes == [
        (int(response_mask.sum()), model.output_head.projection.out_features)
    ]
    score.sum().backward()
    assert model.output_head.projection.weight.grad is not None

    changed = input_ids.clone()
    changed[response_mask] = torch.tensor([8, 9, 10, 8])
    changed_labels = changed.clone()
    changed_labels[~response_mask] = -100
    finetune_dpo.policy_logprob(
        model,
        changed,
        changed_labels,
        prompt_mask,
        response_mask,
        num_mc_samples=1,
        antithetic=False,
    )
    assert model.prompt_tokens[-1] == [[1, 2], [5]]


def test_selected_response_projection_matches_dense_reference() -> None:
    torch.manual_seed(0)
    model = _RecordingDIMBA()
    input_ids, labels, prompt_mask, response_mask = _inputs()
    safe_labels = labels.masked_fill(~response_mask, 0)
    timesteps = torch.tensor([2, 6])

    selected = elbo_sequence_logprob(
        model,
        input_ids,
        safe_labels,
        response_mask,
        timesteps=timesteps,
        prompt_mask=prompt_mask,
    )

    def dense_logits(m, ids, t):
        features, _, _ = m(ids, t, prompt_mask=prompt_mask, return_latent_info=True)
        return m.output_head(features, embedding_weight=m.token_embed.get_weight())

    dense = elbo_sequence_logprob(
        model,
        input_ids,
        safe_labels,
        response_mask,
        timesteps=timesteps,
        logits_fn=dense_logits,
    )
    assert torch.allclose(selected, dense, atol=1e-6)

    selected.sum().backward()
    selected_grads = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    model.zero_grad(set_to_none=True)
    dense.sum().backward()
    parameters = dict(model.named_parameters())
    for name, expected in selected_grads.items():
        torch.testing.assert_close(parameters[name].grad, expected)


def test_preference_draws_are_nonclean_antithetic_pairs() -> None:
    model = _RecordingDIMBA()
    input_ids, _, _, _ = _inputs()
    timesteps, noises = preference.sample_elbo_trajectories(
        model, input_ids, num_mc_samples=4, antithetic=True
    )

    assert timesteps.shape == (4, input_ids.shape[0])
    assert noises.shape == (4, input_ids.shape[0], input_ids.shape[1], model.d_latent)
    assert torch.all((timesteps >= 1) & (timesteps < model.num_diffusion_steps))
    assert torch.equal(timesteps[0] + timesteps[1], torch.full_like(timesteps[0], 8))
    assert torch.equal(timesteps[2] + timesteps[3], torch.full_like(timesteps[2], 8))


def test_dpo_batch_keeps_reference_out_of_autograd() -> None:
    policy = _RecordingDIMBA()
    reference = _RecordingDIMBA()
    input_ids, labels, prompt_mask, response_mask = _inputs()
    batch = {
        "chosen_input_ids": input_ids,
        "chosen_labels": labels,
        "chosen_prompt_mask": prompt_mask,
        "chosen_response_mask": response_mask.float(),
        "rejected_input_ids": input_ids.roll(1, dims=1),
        "rejected_labels": labels.roll(1, dims=1),
        "rejected_prompt_mask": prompt_mask.roll(1, dims=1),
        "rejected_response_mask": response_mask.roll(1, dims=1).float(),
    }
    args = SimpleNamespace(
        loss_type="dpo",
        mc_samples=1,
        antithetic=False,
        beta=0.1,
        gamma=1.0,
        label_smoothing=0.0,
    )

    loss, _ = finetune_dpo.compute_dpo_batch_loss(policy, reference, batch, args)
    loss.backward()
    assert any(parameter.grad is not None for parameter in policy.parameters())
    assert all(parameter.grad is None for parameter in reference.parameters())


def test_identical_dropout_models_have_zero_implicit_reward_with_shared_draws(
    monkeypatch,
) -> None:
    torch.manual_seed(17)
    policy = _RecordingDIMBA(dropout=0.75).train()
    reference = copy.deepcopy(policy).eval()
    input_ids, labels, prompt_mask, response_mask = _inputs()
    batch = {
        "chosen_input_ids": input_ids,
        "chosen_labels": labels,
        "chosen_prompt_mask": prompt_mask,
        "chosen_response_mask": response_mask.float(),
        "rejected_input_ids": input_ids.flip(1),
        "rejected_labels": labels.flip(1),
        "rejected_prompt_mask": prompt_mask.flip(1),
        "rejected_response_mask": response_mask.flip(1).float(),
    }
    args = SimpleNamespace(
        loss_type="dpo",
        mc_samples=2,
        antithetic=True,
        beta=0.1,
        gamma=1.0,
        label_smoothing=0.0,
    )
    captured = {}
    original_dpo_loss = finetune_dpo.dpo_loss

    def recording_dpo_loss(pi_c, pi_r, ref_c, ref_r, **kwargs):
        captured.update(pi_c=pi_c.detach(), pi_r=pi_r.detach(), ref_c=ref_c, ref_r=ref_r)
        return original_dpo_loss(pi_c, pi_r, ref_c, ref_r, **kwargs)

    monkeypatch.setattr(finetune_dpo, "dpo_loss", recording_dpo_loss)
    loss, metrics = finetune_dpo.compute_dpo_batch_loss(policy, reference, batch, args)

    torch.testing.assert_close(captured["pi_c"], captured["ref_c"])
    torch.testing.assert_close(captured["pi_r"], captured["ref_r"])
    torch.testing.assert_close(loss, torch.log(torch.tensor(2.0)))
    torch.testing.assert_close(metrics["reward_margin"], torch.tensor(0.0))
    assert policy.training
    assert not reference.training

    loss.backward()
    policy_grad = policy.output_head.projection.weight.grad
    assert policy_grad is not None and torch.isfinite(policy_grad).all()
    assert policy_grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in reference.parameters())


def test_collate_discards_only_batch_wide_trailing_padding() -> None:
    input_ids, labels, prompt_mask, response_mask = _inputs()
    items = []
    for row in range(2):
        item = {}
        for side in ("chosen", "rejected"):
            item[f"{side}_input_ids"] = input_ids[row]
            item[f"{side}_labels"] = labels[row]
            item[f"{side}_prompt_mask"] = prompt_mask[row]
            item[f"{side}_response_mask"] = response_mask[row].float()
        items.append(item)

    batch = finetune_dpo.collate_triplets(items)
    assert batch["chosen_input_ids"].shape == (2, 4)
    assert batch["rejected_input_ids"].shape == (2, 4)
    assert batch["chosen_prompt_mask"].tolist() == prompt_mask[:, :4].tolist()
    assert batch["chosen_response_mask"].tolist() == response_mask[:, :4].float().tolist()


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"mc_samples": 0}, "mc-samples"),
        ({"mc_samples": 3, "antithetic": True}, "even"),
        ({"grad_accumulation_steps": 0}, "accumulation"),
        ({"learning_rate": 0.0}, "learning-rate"),
        ({"beta": 0.0}, "beta"),
        ({"label_smoothing": 0.5}, "label-smoothing"),
    ],
)
def test_dpo_rejects_noop_or_invalid_arguments(override, message) -> None:
    values = {
        "mc_samples": 2,
        "antithetic": True,
        "batch_size": 2,
        "grad_accumulation_steps": 1,
        "num_epochs": 1,
        "max_seq_length": 32,
        "log_every": 1,
        "learning_rate": 1e-6,
        "beta": 0.1,
        "label_smoothing": 0.0,
    }
    values.update(override)
    with pytest.raises(ValueError, match=message):
        finetune_dpo.validate_args(SimpleNamespace(**values))
