"""Parity checks for the optimized masked-generation verifier."""

import torch
import torch.nn.functional as F

from dimba.models.diffusion import DIMBA
from scripts.generate import gap_score, guided_logits


def _tiny_model() -> DIMBA:
    return DIMBA(
        vocab_size=31,
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
    ).eval()


@torch.inference_mode()
def _gap_reference(model, mask_id, eos, ids, prompt_len, groups):
    batch, length = ids.shape
    generated = torch.zeros_like(ids, dtype=torch.bool)
    generated[:, prompt_len:] = (
        (ids[:, prompt_len:] != eos) & (ids[:, prompt_len:] != mask_id)
    )
    positions = torch.arange(length)[None, :].expand(batch, length)
    total = torch.zeros(batch)
    count = torch.zeros(batch)
    for remainder in range(groups):
        masked = generated & (positions % groups == remainder)
        corrupted = ids.masked_fill(masked, mask_id)
        cond = model.predict_token_logits(corrupted, 0.25).float()
        uncond_ids = corrupted.clone()
        uncond_ids[:, :prompt_len] = mask_id
        uncond = model.predict_token_logits(uncond_ids, 0.25).float()
        targets = ids.unsqueeze(-1)
        delta = (
            F.log_softmax(cond, -1).gather(-1, targets)
            - F.log_softmax(uncond, -1).gather(-1, targets)
        ).squeeze(-1)
        total += (delta * masked).sum(1)
        count += masked.sum(1)
    return total / count.clamp(min=1)


def test_batched_gap_score_matches_leave_group_out_reference() -> None:
    torch.manual_seed(12)
    model = _tiny_model()
    mask_id, eos, prompt_len = 30, 2, 3
    ids = torch.randint(3, 29, (3, 9))
    ids[0, -1] = eos
    ids[1, -2] = mask_id

    expected = _gap_reference(model, mask_id, eos, ids, prompt_len, groups=7)
    actual = gap_score(model, mask_id, eos, ids, prompt_len, K=7)

    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_gap_score_uses_one_feature_dispatch() -> None:
    torch.manual_seed(13)
    model = _tiny_model()
    ids = torch.randint(3, 29, (2, 10))
    calls = []
    original = model.predict_token_features

    def recording_features(input_ids, t):
        calls.append(input_ids.shape[0])
        return original(input_ids, t)

    model.predict_token_features = recording_features
    gap_score(model, 30, 2, ids, prompt_len=4, K=4)

    assert calls == [2 * 4 * ids.shape[0]]


def test_guidance_zero_and_one_skip_the_doubled_batch() -> None:
    torch.manual_seed(14)
    model = _tiny_model()
    ids = torch.randint(3, 29, (2, 8))
    positions = torch.tensor([[4, 6], [4, 6]])
    calls = []
    original = model.predict_token_features

    def recording_features(input_ids, t):
        calls.append(input_ids.shape[0])
        return original(input_ids, t)

    model.predict_token_features = recording_features
    guided_logits(model, 30, ids, 4, 0.25, 0.0, positions)
    guided_logits(model, 30, ids, 4, 0.25, 1.0, positions)
    guided_logits(model, 30, ids, 4, 0.25, 2.0, positions)

    assert calls == [2, 2, 4]
