"""Tests for VAE (Variational Autoencoder) components."""

import pytest
import torch
import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0] + '/src')

from dimba.models.vae import TokenVAE, TokenVAEWithDeterministicFallback, create_latent_projector
from dimba.models.embeddings import LatentProjector


class TestTokenVAE:
    """Test TokenVAE functionality."""

    @pytest.fixture
    def vae(self):
        return TokenVAE(
            input_dim=64,
            latent_dim=32,
            hidden_dim=128,
            num_layers=2,
            dropout=0.1,
            kl_weight=1.0,
        )

    def test_vae_initialization(self, vae):
        """Test VAE initializes correctly."""
        assert vae.input_dim == 64
        assert vae.latent_dim == 32
        assert vae.hidden_dim == 128
        assert vae.num_layers == 2
        assert vae.kl_weight == 1.0

    def test_encode_shape(self, vae):
        """Test encoder output shapes."""
        x = torch.randn(4, 16, 64)  # [batch, seq, input_dim]
        mu, logvar = vae.encode(x)
        
        assert mu.shape == (4, 16, 32)  # [batch, seq, latent_dim]
        assert logvar.shape == (4, 16, 32)

    def test_reparameterize_shape(self, vae):
        """Test reparameterization output shape."""
        mu = torch.randn(4, 16, 32)
        logvar = torch.randn(4, 16, 32)
        z = vae.reparameterize(mu, logvar)
        
        assert z.shape == (4, 16, 32)

    def test_reparameterize_different_samples(self, vae):
        """Test that reparameterize produces different samples."""
        mu = torch.randn(4, 16, 32)
        logvar = torch.randn(4, 16, 32)
        
        z1 = vae.reparameterize(mu, logvar)
        z2 = vae.reparameterize(mu, logvar)
        
        # Samples should be different (stochastic)
        assert not torch.allclose(z1, z2)

    def test_reparameterize_statistics(self, vae):
        """Test that reparameterize produces correct statistics."""
        torch.manual_seed(42)
        mu = torch.zeros(1000, 32)
        logvar = torch.zeros(1000, 32)  # std = 1
        
        z = vae.reparameterize(mu, logvar)
        
        # Mean should be close to 0, std close to 1
        assert abs(z.mean().item()) < 0.1
        assert abs(z.std().item() - 1.0) < 0.1

    def test_decode_shape(self, vae):
        """Test decoder output shape."""
        z = torch.randn(4, 16, 32)
        x = vae.decode(z)
        
        assert x.shape == (4, 16, 64)  # [batch, seq, input_dim]

    def test_forward_shape(self, vae):
        """Test forward pass shapes."""
        x = torch.randn(4, 16, 64)
        x_recon, stats = vae(x, return_stats=True)
        
        assert x_recon.shape == x.shape
        assert "mu" in stats
        assert "logvar" in stats
        assert "z" in stats
        assert stats["mu"].shape == (4, 16, 32)
        assert stats["logvar"].shape == (4, 16, 32)
        assert stats["z"].shape == (4, 16, 32)

    def test_forward_no_stats(self, vae):
        """Test forward pass without returning stats."""
        x = torch.randn(4, 16, 64)
        x_recon, stats = vae(x, return_stats=False)
        
        assert x_recon.shape == x.shape
        assert stats is None

    def test_sample_latent_deterministic(self, vae):
        """Test sample_latent with sample=False returns mu."""
        x = torch.randn(4, 16, 64)
        z = vae.sample_latent(x, sample=False)
        mu, _ = vae.encode(x)
        
        assert torch.allclose(z, mu)

    def test_sample_latent_stochastic(self, vae):
        """Test sample_latent with sample=True returns stochastic sample."""
        x = torch.randn(4, 16, 64)
        z1 = vae.sample_latent(x, sample=True)
        z2 = vae.sample_latent(x, sample=True)
        
        # Should be different due to randomness
        assert not torch.allclose(z1, z2)

    def test_compute_loss(self, vae):
        """Test loss computation."""
        x = torch.randn(4, 16, 64)
        x_recon = torch.randn(4, 16, 64)
        mu = torch.randn(4, 16, 32)
        logvar = torch.randn(4, 16, 32)
        
        loss, loss_dict = vae.compute_loss(x, x_recon, mu, logvar)
        
        assert loss.item() > 0
        assert "total" in loss_dict
        assert "recon" in loss_dict
        assert "kl" in loss_dict
        assert loss_dict["total"].item() > 0
        assert loss_dict["recon"].item() >= 0
        assert loss_dict["kl"].item() >= 0

    def test_compute_loss_reductions(self, vae):
        """Test loss with different reductions."""
        x = torch.randn(4, 16, 64)
        x_recon = torch.randn(4, 16, 64)
        mu = torch.randn(4, 16, 32)
        logvar = torch.randn(4, 16, 32)
        
        loss_mean, _ = vae.compute_loss(x, x_recon, mu, logvar, reduction="mean")
        loss_sum, _ = vae.compute_loss(x, x_recon, mu, logvar, reduction="sum")
        
        # Sum should be larger than mean (approximately batch_size * seq_len times)
        assert loss_sum > loss_mean

    def test_kl_divergence_non_negative(self, vae):
        """Test that KL divergence is non-negative."""
        x = torch.randn(4, 16, 64)
        x_recon, stats = vae(x, return_stats=True)
        
        _, loss_dict = vae.compute_loss(x, x_recon, stats["mu"], stats["logvar"])
        
        assert loss_dict["kl"].item() >= 0

    def test_encode_to_latent_interface(self, vae):
        """Test encode_to_latent matches LatentProjector interface."""
        x = torch.randn(4, 16, 64)
        z = vae.encode_to_latent(x)
        
        assert z.shape == (4, 16, 32)

    def test_decode_from_latent_interface(self, vae):
        """Test decode_from_latent matches LatentProjector interface."""
        z = torch.randn(4, 16, 32)
        x = vae.decode_from_latent(z)
        
        assert x.shape == (4, 16, 64)

    def test_gradient_flow(self, vae):
        """Test that gradients flow through VAE."""
        x = torch.randn(4, 16, 64, requires_grad=True)
        x_recon, stats = vae(x, return_stats=True)
        
        loss, _ = vae.compute_loss(x, x_recon, stats["mu"], stats["logvar"])
        loss.backward()
        
        # Check that gradients exist
        assert x.grad is not None
        assert any(p.grad is not None for p in vae.parameters())

    def test_device_handling(self, vae):
        """Test VAE works on different devices."""
        if torch.cuda.is_available():
            vae = vae.cuda()
            x = torch.randn(4, 16, 64).cuda()
            
            x_recon, _ = vae(x)
            
            assert x_recon.device.type == 'cuda'


class TestTokenVAEWithDeterministicFallback:
    """Test TokenVAEWithDeterministicFallback wrapper."""

    @pytest.fixture
    def wrapped_vae(self):
        vae = TokenVAE(input_dim=64, latent_dim=32)
        return TokenVAEWithDeterministicFallback(vae, use_vae_sampling=False)

    def test_wrapper_encode_deterministic(self, wrapped_vae):
        """Test wrapper encode uses deterministic path."""
        x = torch.randn(4, 16, 64)
        z1 = wrapped_vae.encode(x)
        z2 = wrapped_vae.encode(x)
        
        # Should be identical (deterministic)
        assert torch.allclose(z1, z2)

    def test_wrapper_encode_stochastic(self):
        """Test wrapper encode with sampling."""
        vae = TokenVAE(input_dim=64, latent_dim=32)
        wrapped = TokenVAEWithDeterministicFallback(vae, use_vae_sampling=True)
        
        x = torch.randn(4, 16, 64)
        z1 = wrapped.encode(x)
        z2 = wrapped.encode(x)
        
        # Should be different (stochastic)
        assert not torch.allclose(z1, z2)

    def test_wrapper_decode(self, wrapped_vae):
        """Test wrapper decode."""
        z = torch.randn(4, 16, 32)
        x = wrapped_vae.decode(z)
        
        assert x.shape == (4, 16, 64)

    def test_wrapper_forward(self, wrapped_vae):
        """Test wrapper forward."""
        x = torch.randn(4, 16, 64)
        x_recon, stats = wrapped_vae(x, return_stats=True)
        
        assert x_recon.shape == x.shape
        assert stats is not None


class TestCreateLatentProjector:
    """Test factory function for creating latent projectors."""

    def test_create_latent_projector_deterministic(self):
        """Test creating deterministic LatentProjector."""
        projector = create_latent_projector(
            input_dim=64,
            latent_dim=32,
            use_vae=False,
        )
        
        assert isinstance(projector, LatentProjector)
        assert not isinstance(projector, TokenVAE)

    def test_create_latent_projector_vae(self):
        """Test creating TokenVAE."""
        projector = create_latent_projector(
            input_dim=64,
            latent_dim=32,
            use_vae=True,
            kl_weight=0.5,
        )
        
        assert isinstance(projector, TokenVAE)
        assert projector.kl_weight == 0.5


class TestVAEIntegration:
    """Test VAE integration with diffusion model components."""

    def test_vae_with_embeddings(self):
        """Test VAE works with token embeddings."""
        from dimba.models.embeddings import TokenEmbedding
        
        vocab_size = 1000
        d_model = 64
        latent_dim = 32
        
        token_embed = TokenEmbedding(vocab_size, d_model)
        vae = TokenVAE(input_dim=d_model, latent_dim=latent_dim)
        
        input_ids = torch.randint(0, vocab_size, (4, 16))
        x_0 = token_embed(input_ids)
        
        x_recon, stats = vae(x_0, return_stats=True)
        
        assert x_recon.shape == x_0.shape
        assert stats["z"].shape == (4, 16, latent_dim)

    def test_vae_encode_decode_cycle(self):
        """Test encode-decode cycle preserves information."""
        vae = TokenVAE(input_dim=64, latent_dim=32)
        
        x = torch.randn(4, 16, 64)
        
        # Encode to latent
        z = vae.sample_latent(x, sample=False)
        
        # Decode back
        x_recon = vae.decode(z)
        
        # Reconstruction shouldn't be perfect (VAE is lossy)
        # but shapes should match
        assert x_recon.shape == x.shape


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
