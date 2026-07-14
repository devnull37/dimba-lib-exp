"""Regression tests for the optimized continuous samplers."""

import pytest
import torch
from torch import nn

from dimba.diffusion.sampling import (
    _batched_cfg_denoise,
    sample_from_model,
    sample_from_model_flow,
)
from dimba.models.diffusion import DIMBA


class _ToySamplerModel(nn.Module):
    """Tiny deterministic model that exposes sampler dispatch sizes."""

    def __init__(self, *, flow: bool = False) -> None:
        super().__init__()
        self.d_latent = 4
        self.num_diffusion_steps = 6
        self.use_flow_matching = flow
        self.token_embed = nn.Embedding(17, self.d_latent)
        self.output_head = nn.Linear(self.d_latent, 17, bias=False)
        self.dispatch_batches = []
        self.inference_flags = []

    def encode_latent(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def decode_latent(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def get_alphas_cumprod(self) -> torch.Tensor:
        return torch.linspace(
            0.99, 0.05, self.num_diffusion_steps, device=self.token_embed.weight.device
        )

    def conditioning_from_prompt(
        self,
        prompt_ids=None,
        batch_size=None,
        device=None,
        drop_cond=False,
    ) -> torch.Tensor:
        if drop_cond or prompt_ids is None:
            return torch.zeros(batch_size, 1, self.d_latent, device=device)
        return self.token_embed(prompt_ids).mean(dim=1, keepdim=True)

    def _denoise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        x_self_cond=None,
    ) -> torch.Tensor:
        self.dispatch_batches.append(x_t.shape[0])
        self.inference_flags.append(torch.is_inference_mode_enabled())
        time = t.to(x_t.dtype).view(-1, 1, 1) / max(self.num_diffusion_steps - 1, 1)
        result = 0.25 * x_t + 0.5 * cond + 0.01 * time
        return result if x_self_cond is None else result + 0.1 * x_self_cond

    denoise_to_x0_latent = _denoise
    denoise_flow = _denoise


def test_batched_cfg_matches_two_sequential_denoiser_calls() -> None:
    torch.manual_seed(0)
    model = _ToySamplerModel()
    batch_size = 3
    x_t = torch.randn(batch_size, 5, model.d_latent)
    t = torch.arange(batch_size, dtype=torch.long)
    cond = torch.randn(batch_size, 1, model.d_latent)
    uncond = torch.randn_like(cond)
    x_self_cond = torch.randn_like(x_t)

    with torch.inference_mode():
        actual_cond, actual_uncond = _batched_cfg_denoise(
            model.denoise_to_x0_latent,
            x_t,
            t,
            cond,
            torch.cat((cond, uncond), dim=0),
            x_self_cond,
        )
        assert model.dispatch_batches == [2 * batch_size]

        model.dispatch_batches.clear()
        expected_cond = model.denoise_to_x0_latent(x_t, t, cond, x_self_cond)
        expected_uncond = model.denoise_to_x0_latent(x_t, t, uncond, x_self_cond)

    assert model.dispatch_batches == [batch_size, batch_size]
    assert actual_uncond is not None
    torch.testing.assert_close(actual_cond, expected_cond)
    torch.testing.assert_close(actual_uncond, expected_uncond)


def test_batched_cfg_matches_real_dimba() -> None:
    torch.manual_seed(2)
    model = DIMBA(
        vocab_size=32,
        d_model=32,
        d_prompt=32,
        num_diffusion_steps=8,
        num_denoiser_layers=1,
        d_state=4,
        dropout=0.0,
        use_simple_mamba=True,
        self_conditioning=True,
    ).eval()
    batch_size = 2
    prompt = torch.randint(0, 32, (batch_size, 3))
    cond = model.conditioning_from_prompt(prompt, batch_size, torch.device("cpu"))
    uncond = model.conditioning_from_prompt(
        None, batch_size, torch.device("cpu"), drop_cond=True
    )
    x_t = torch.randn(batch_size, 6, model.d_latent)
    x_self_cond = torch.randn_like(x_t)
    t = torch.full((batch_size,), 4, dtype=torch.long)

    with torch.inference_mode():
        actual_cond, actual_uncond = _batched_cfg_denoise(
            model.denoise_to_x0_latent,
            x_t,
            t,
            cond,
            torch.cat((cond, uncond), dim=0),
            x_self_cond,
        )
        expected_cond = model.denoise_to_x0_latent(x_t, t, cond, x_self_cond)
        expected_uncond = model.denoise_to_x0_latent(x_t, t, uncond, x_self_cond)

    assert actual_uncond is not None
    torch.testing.assert_close(actual_cond, expected_cond)
    torch.testing.assert_close(actual_uncond, expected_uncond)


@pytest.mark.parametrize("kind", ["ddim", "flow"])
def test_cfg_sampler_shape_and_single_dispatch_per_step(kind: str) -> None:
    torch.manual_seed(1)
    batch_size, seq_len, num_steps = 2, 5, 4
    model = _ToySamplerModel(flow=kind == "flow")
    prompt_ids = torch.randint(0, 17, (batch_size, 3))

    if kind == "flow":
        generated = sample_from_model_flow(
            model,
            prompt_ids,
            seq_len,
            num_steps=num_steps,
            guidance_scale=2.0,
        )
    else:
        generated = sample_from_model(
            model,
            prompt_ids,
            seq_len,
            num_steps=num_steps,
            guidance_scale=2.0,
        )

    assert generated.shape == (batch_size, seq_len)
    assert generated.dtype == torch.long
    assert model.dispatch_batches == [2 * batch_size] * num_steps
    assert all(model.inference_flags)


@pytest.mark.parametrize("kind", ["ddim", "flow"])
def test_zero_guidance_runs_only_the_unconditional_batch(kind: str) -> None:
    torch.manual_seed(3)
    batch_size, seq_len, num_steps = 2, 5, 4
    model = _ToySamplerModel(flow=kind == "flow")
    prompt_ids = torch.randint(0, 17, (batch_size, 3))

    if kind == "flow":
        sample_from_model_flow(
            model,
            prompt_ids,
            seq_len,
            num_steps=num_steps,
            guidance_scale=0.0,
        )
    else:
        sample_from_model(
            model,
            prompt_ids,
            seq_len,
            num_steps=num_steps,
            guidance_scale=0.0,
        )

    assert model.dispatch_batches == [batch_size] * num_steps
