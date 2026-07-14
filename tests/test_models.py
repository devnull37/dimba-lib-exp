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

    def test_backend_is_mamba2_family(self):
        block = Mamba2Block(d_model=64, d_state=8)
        assert type(block.mamba_fwd).__name__ in {"Mamba2", "TorchMamba2", "SimpleMamba2"}
        assert type(block.mamba_fwd).__name__ != "Mamba"

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

    def test_denoising_head_selected_positions_match_full_projection(self):
        head = DenoisingHead(d_model=16, vocab_size=32).eval()
        x = torch.randn(2, 7, 16)
        positions = torch.tensor([[1, 4, 6], [0, 2, 5]])
        with torch.inference_mode():
            full = head(x)
            selected = head(x, positions=positions)
        expected = full.gather(1, positions.unsqueeze(-1).expand(-1, -1, full.shape[-1]))
        assert torch.allclose(selected, expected, atol=1e-6, rtol=1e-6)


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

        # DIMBA.forward returns (x_pred, noise, latent_info).
        x_pred, noise, latent_info = model(input_ids, t)

        assert x_pred.shape == (4, 32, 64)
        assert noise.shape == (4, 32, 64)
        assert isinstance(latent_info, dict)

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

    def test_selected_masked_logits_and_feature_cfg_match_full_logits(self, model):
        model.eval()
        ids = torch.randint(0, model.vocab_size, (2, 8))
        uncond = ids.clone()
        uncond[:, :3] = 0
        both = torch.cat([ids, uncond])
        positions = torch.tensor([[3, 5, 7], [3, 4, 6]])
        guidance = 1.7

        with torch.inference_mode():
            full = model.predict_token_logits(both, 0.5)
            cond_logits, uncond_logits = full.chunk(2)
            expected_cfg = uncond_logits + guidance * (cond_logits - uncond_logits)
            expected_selected = expected_cfg.gather(
                1, positions.unsqueeze(-1).expand(-1, -1, expected_cfg.shape[-1])
            )

            features = model.predict_token_features(both, 0.5)
            cond_features, uncond_features = features.chunk(2)
            selected_cfg = model.project_token_features(
                uncond_features + guidance * (cond_features - uncond_features),
                positions,
            )

        assert torch.allclose(selected_cfg, expected_selected, atol=1e-5, rtol=1e-5)

    def test_get_alphas_cumprod(self, model):
        """Test getting cumulative alphas."""
        alphas = model.get_alphas_cumprod()
        assert alphas.shape == (model.num_diffusion_steps,)
        assert (alphas >= 0).all() and (alphas <= 1).all()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
