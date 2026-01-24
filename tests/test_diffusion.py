"""Tests for diffusion components."""

import pytest
import torch
import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0] + '/src')

from dimba.diffusion.schedules import CosineNoiseSchedule
from dimba.diffusion.sampling import sample_timesteps, top_k_top_p_filtering


class TestNoiseSchedules:
    """Test noise schedule implementations."""

    def test_cosine_schedule_initialization(self):
        schedule = CosineNoiseSchedule(num_steps=1000, s=0.008)

        assert schedule.alphas_cumprod.shape == (1000,)
        assert schedule.betas.shape == (1000,)
        assert schedule.sqrt_alphas_cumprod.shape == (1000,)

    def test_cosine_schedule_add_noise(self):
        schedule = CosineNoiseSchedule(num_steps=100)

        x_0 = torch.randn(4, 32, 64)
        t = torch.tensor([10, 25, 50, 75])
        noise = torch.randn_like(x_0)

        x_t, used_noise = schedule.add_noise(x_0, t, noise)

        assert x_t.shape == x_0.shape
        assert used_noise.shape == x_0.shape
        assert torch.allclose(used_noise, noise)

    def test_cosine_schedule_alpha_ranges(self):
        schedule = CosineNoiseSchedule(num_steps=1000)

        alphas = schedule.get_alphas_cumprod()
        assert (alphas >= 0).all()
        assert (alphas <= 1).all()
        # Should be decreasing over time
        assert (alphas[:-1] >= alphas[1:]).all()



    def test_noise_without_predefined(self):
        """Test noise generation when not provided."""
        schedule = CosineNoiseSchedule(num_steps=100)

        x_0 = torch.randn(4, 32, 64)
        t = torch.tensor([10, 50, 99, 0])

        x_t1, noise1 = schedule.add_noise(x_0, t)
        x_t2, noise2 = schedule.add_noise(x_0, t)

        # Different noise should be generated each time
        assert not torch.allclose(noise1, noise2)

    def test_buffer_device_handling(self):
        """Test that buffers are properly managed."""
        schedule = CosineNoiseSchedule(num_steps=100)

        if torch.cuda.is_available():
            schedule = schedule.cuda()

            alphas = schedule.get_alphas_cumprod()
            assert alphas.device.type == 'cuda'


class TestSampling:
    """Test sampling utilities."""

    def test_sample_timesteps(self):
        batch_size = 32
        num_steps = 1000

        t = sample_timesteps(batch_size, num_steps, torch.device('cpu'))

        assert t.shape == (batch_size,)
        assert (t >= 0).all()
        assert (t < num_steps).all()

    def test_top_k_filtering(self):
        logits = torch.randn(4, 32, 1000)

        filtered = top_k_top_p_filtering(logits, top_k=50, top_p=1.0)

        assert filtered.shape == logits.shape
        # Check that some values are set to -inf
        assert (filtered == float('-inf')).any()

    def test_top_p_filtering(self):
        logits = torch.randn(4, 32, 1000)

        filtered = top_k_top_p_filtering(logits, top_k=0, top_p=0.9)

        assert filtered.shape == logits.shape

    def test_top_k_top_p_combined(self):
        logits = torch.randn(4, 32, 1000)

        filtered = top_k_top_p_filtering(logits, top_k=100, top_p=0.95)

        assert filtered.shape == logits.shape

    def test_filtering_preserves_shape(self):
        """Test that filtering preserves tensor shape."""
        logits = torch.randn(2, 16, 500)

        for top_k in [None, 10, 50]:
            for top_p in [None, 0.9, 0.95]:
                if top_k is None and top_p is None:
                    continue

                filtered = top_k_top_p_filtering(
                    logits,
                    top_k=top_k if top_k else 0,
                    top_p=top_p if top_p else 1.0,
                )

                assert filtered.shape == logits.shape


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
