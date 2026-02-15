"""Tests for DIMBA model components."""

import pytest
import torch

from dimba.models.embeddings import (
    TokenEmbedding,
    TimestepEmbedding,
    PromptEncoder,
    FiLMConditioning,
    AdditiveConditioning,
)
from dimba.models.denoiser import Mamba2Block, Mamba2Denoiser, DenoisingHead
from dimba.models.diffusion import DIMBA


class TestEmbeddings:
    """Test embedding layers."""

    def test_token_embedding(self):
        vocab_size = 1000
        embed_dim = 64
        embedding = TokenEmbedding(vocab_size, embed_dim, padding_idx=0)

        # Test shape
        input_ids = torch.randint(0, vocab_size, (4, 32))
        output = embedding(input_ids)
        assert output.shape == (4, 32, embed_dim)

    def test_timestep_embedding(self):
        time_embed = TimestepEmbedding(time_embed_dim=128, out_dim=256)

        # Test shape
        t = torch.randint(0, 1000, (4,))
        output = time_embed(t)
        assert output.shape == (4, 256)

    def test_prompt_encoder(self):
        encoder = PromptEncoder(
            input_dim=64,
            hidden_dim=128,
            output_dim=64,
            num_layers=2,
        )

        x = torch.randn(4, 32, 64)
        output = encoder(x)
        assert output.shape == (4, 32, 64)

    def test_film_conditioning(self):
        film = FiLMConditioning(cond_dim=64, target_dim=128)

        x = torch.randn(4, 32, 128)
        cond = torch.randn(4, 32, 64)
        output = film(x, cond)

        assert output.shape == x.shape

    def test_additive_conditioning(self):
        additive = AdditiveConditioning(cond_dim=64, target_dim=64)

        x = torch.randn(4, 32, 64)
        cond = torch.randn(4, 32, 64)
        output = additive(x, cond)

        assert output.shape == x.shape


class TestDenoiser:
    """Test denoiser components."""

    def test_mamba2_block(self):
        block = Mamba2Block(d_model=64, d_state=8)

        x = torch.randn(4, 32, 64)
        output = block(x)

        assert output.shape == x.shape

    def test_mamba2_denoiser(self):
        denoiser = Mamba2Denoiser(
            d_model=64,
            num_layers=2,
            d_state=8,
            cond_dim=64,
            time_embed_dim=128,
        )

        x_t = torch.randn(4, 32, 64)
        cond = torch.randn(4, 32, 64)
        time_emb = torch.randn(4, 128)

        output = denoiser(x_t, cond, time_emb)
        assert output.shape == x_t.shape

    def test_denoising_head(self):
        vocab_size = 1000
        head = DenoisingHead(d_model=64, vocab_size=vocab_size)

        x = torch.randn(4, 32, 64)
        logits = head(x)

        assert logits.shape == (4, 32, vocab_size)


class TestDIMBA:
    """Test main DIMBA model."""

    @pytest.fixture
    def model(self):
        return DIMBA(
            vocab_size=1000,
            d_model=64,
            d_prompt=64,
            num_diffusion_steps=100,
            num_denoiser_layers=2,
        )

    def test_model_initialization(self, model):
        """Test model initializes correctly."""
        assert model.vocab_size == 1000
        assert model.d_model == 64
        assert model.num_diffusion_steps == 100

    def test_forward_pass(self, model):
        """Test forward pass during training."""
        input_ids = torch.randint(0, 1000, (4, 32))
        t = torch.randint(0, 100, (4,))

        x_pred, noise = model(input_ids, t)

        assert x_pred.shape == (4, 32, 64)
        assert noise.shape == (4, 32, 64)

    def test_prompt_encoding(self, model):
        """Test prompt encoding."""
        input_ids = torch.randint(0, 1000, (4, 32))
        cond = model.encode_prompt(input_ids)

        assert cond.shape == (4, 32, 64)

    def test_denoise_step(self, model):
        """Test single denoising step."""
        x_t = torch.randn(4, 32, 64)
        t = torch.full((4,), 50, dtype=torch.long)
        prompt_cond = torch.randn(4, 32, 64)

        x_pred = model.denoise_step(x_t, t, prompt_cond)

        assert x_pred.shape == x_t.shape

    def test_device_handling(self, model):
        """Test model device handling."""
        if torch.cuda.is_available():
            model = model.cuda()
            input_ids = torch.randint(0, 1000, (2, 16)).cuda()
            t = torch.randint(0, 100, (2,)).cuda()

            x_pred, _ = model(input_ids, t)
            assert x_pred.device.type == 'cuda'

    def test_output_projection(self, model):
        """Test output projection."""
        x_pred = torch.randn(4, 32, 64)
        logits = model.output_head(x_pred)

        assert logits.shape == (4, 32, 1000)

    def test_get_alphas_cumprod(self, model):
        """Test getting cumulative alphas."""
        alphas = model.get_alphas_cumprod()
        assert alphas.shape == (model.num_diffusion_steps,)
        assert (alphas >= 0).all() and (alphas <= 1).all()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
