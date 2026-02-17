"""Variational Autoencoder (VAE) for token embeddings in DIMBA.

This module provides a TokenVAE class that encodes token embeddings into
a probabilistic latent space and decodes them back, enabling latent diffusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

from .embeddings import LatentProjector


class TokenVAE(nn.Module):
    """Variational Autoencoder for token embeddings.

    Maps token embeddings to a probabilistic latent space using an encoder
    that outputs mean (mu) and log-variance (logvar) parameters. The decoder
    maps sampled latent vectors back to the embedding space.

    Args:
        input_dim: Input embedding dimension
        latent_dim: Latent dimension (will be doubled for mu/logvar)
        hidden_dim: Hidden dimension for MLP layers
        num_layers: Number of layers in encoder/decoder
        dropout: Dropout rate
        kl_weight: Weight for KL divergence loss (beta-VAE)
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        dropout: float = 0.1,
        kl_weight: float = 1.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim or max(input_dim, latent_dim)
        self.num_layers = num_layers
        self.kl_weight = kl_weight

        # Encoder outputs 2 * latent_dim (mu and logvar)
        self.encoder = self._build_mlp(
            input_dim=input_dim,
            output_dim=2 * latent_dim,
            hidden_dim=self.hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        # Decoder maps latent z back to input space
        self.decoder = self._build_mlp(
            input_dim=latent_dim,
            output_dim=input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

    @staticmethod
    def _build_mlp(
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ):
        """Build an MLP with GELU activations."""
        layers = []
        for i in range(num_layers):
            in_features = input_dim if i == 0 else hidden_dim
            out_features = output_dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_features, out_features))
            if i < num_layers - 1:
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
        return nn.Sequential(*layers)

    def encode(self, x: torch.Tensor):
        """Encode embeddings into latent distribution parameters.

        Args:
            x: Embeddings [batch_size, seq_len, input_dim]

        Returns:
            mu: Mean of latent distribution [batch_size, seq_len, latent_dim]
            logvar: Log-variance of latent distribution [batch_size, seq_len, latent_dim]
        """
        h = self.encoder(x)  # [batch_size, seq_len, 2 * latent_dim]
        mu, logvar = torch.chunk(h, 2, dim=-1)  # Split into mu and logvar
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        """Reparameterization trick for sampling from N(mu, var).

        z = mu + sigma * epsilon, where epsilon ~ N(0, I)

        Args:
            mu: Mean [batch_size, seq_len, latent_dim]
            logvar: Log-variance [batch_size, seq_len, latent_dim]

        Returns:
            z: Sampled latent vector [batch_size, seq_len, latent_dim]
        """
        std = torch.exp(0.5 * logvar)  # sigma = sqrt(exp(logvar)) = exp(0.5 * logvar)
        eps = torch.randn_like(std)  # epsilon ~ N(0, I)
        z = mu + std * eps  # Reparameterization trick
        return z

    def decode(self, z: torch.Tensor):
        """Decode latent representations back to embedding space.

        Args:
            z: Latent embeddings [batch_size, seq_len, latent_dim]

        Returns:
            embeddings: [batch_size, seq_len, input_dim]
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor, return_stats: bool = False):
        """Forward pass through VAE.

        Args:
            x: Input embeddings [batch_size, seq_len, input_dim]
            return_stats: Whether to return intermediate statistics

        Returns:
            x_recon: Reconstructed embeddings [batch_size, seq_len, input_dim]
            stats: Dictionary containing mu, logvar, z (if return_stats=True)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)

        if return_stats:
            stats = {"mu": mu, "logvar": logvar, "z": z}
            return x_recon, stats
        return x_recon, None

    def sample_latent(self, x: torch.Tensor, sample: bool = True):
        """Get latent representation (either mu or sampled z).

        Args:
            x: Input embeddings [batch_size, seq_len, input_dim]
            sample: If True, sample from distribution; else return mu

        Returns:
            z: Latent representation [batch_size, seq_len, latent_dim]
        """
        mu, logvar = self.encode(x)
        if sample:
            return self.reparameterize(mu, logvar)
        return mu

    def compute_loss(self, x, x_recon, mu, logvar, reduction="mean"):
        """Compute VAE loss (reconstruction + KL divergence).

        Args:
            x: Original embeddings [batch_size, seq_len, input_dim]
            x_recon: Reconstructed embeddings [batch_size, seq_len, input_dim]
            mu: Mean [batch_size, seq_len, latent_dim]
            logvar: Log-variance [batch_size, seq_len, latent_dim]
            reduction: 'mean', 'sum', or 'none'

        Returns:
            loss: Total ELBO loss
            loss_dict: Dictionary with 'total', 'recon', 'kl' losses
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(x_recon, x, reduction=reduction)

        # KL divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)

        if reduction == "mean":
            kl_loss = kl_loss.mean()
        elif reduction == "sum":
            kl_loss = kl_loss.sum()

        # Total ELBO loss (negative ELBO = recon_loss + beta * KL)
        total_loss = recon_loss + self.kl_weight * kl_loss

        loss_dict = {"total": total_loss, "recon": recon_loss, "kl": kl_loss}
        return total_loss, loss_dict

    def encode_to_latent(self, x: torch.Tensor):
        """Convenience method: encode embeddings to latent space.

        This matches the LatentProjector.encode interface.

        Args:
            x: Embeddings [batch_size, seq_len, input_dim]

        Returns:
            z: Latent embeddings [batch_size, seq_len, latent_dim]
        """
        return self.sample_latent(x, sample=False)  # Use mu for deterministic encoding

    def decode_from_latent(self, z: torch.Tensor):
        """Convenience method: decode latent to embeddings.

        This matches the LatentProjector.decode interface.

        Args:
            z: Latent embeddings [batch_size, seq_len, latent_dim]

        Returns:
            x: Embeddings [batch_size, seq_len, input_dim]
        """
        return self.decode(z)


class TokenVAEWithDeterministicFallback(nn.Module):
    """Wrapper that uses VAE during training but can use deterministic projection during inference.

    This provides compatibility with the existing LatentProjector interface while
    enabling VAE training.

    Args:
        vae: TokenVAE instance
        use_vae_sampling: Whether to use VAE sampling during encoding
    """

    def __init__(self, vae: TokenVAE, use_vae_sampling: bool = False):
        super().__init__()
        self.vae = vae
        self.use_vae_sampling = use_vae_sampling
        self.input_dim = vae.input_dim
        self.latent_dim = vae.latent_dim

    def encode(self, x: torch.Tensor):
        """Encode to latent space.

        Args:
            x: Embeddings [batch_size, seq_len, input_dim]

        Returns:
            z: Latent embeddings [batch_size, seq_len, latent_dim]
        """
        return self.vae.sample_latent(x, sample=self.use_vae_sampling)

    def decode(self, z: torch.Tensor):
        """Decode from latent space.

        Args:
            z: Latent embeddings [batch_size, seq_len, latent_dim]

        Returns:
            x: Embeddings [batch_size, seq_len, input_dim]
        """
        return self.vae.decode(z)

    def forward(self, x: torch.Tensor, return_stats: bool = False):
        """Forward through VAE."""
        return self.vae(x, return_stats=return_stats)

    def compute_loss(self, x, x_recon, mu, logvar):
        """Compute VAE loss."""
        return self.vae.compute_loss(x, x_recon, mu, logvar)


def create_latent_projector(
    input_dim: int,
    latent_dim: int,
    use_vae: bool = False,
    hidden_dim: Optional[int] = None,
    num_layers: int = 2,
    dropout: float = 0.1,
    kl_weight: float = 1.0,
):
    """Factory function to create either LatentProjector or TokenVAE.

    Args:
        input_dim: Input embedding dimension
        latent_dim: Latent dimension
        use_vae: If True, create TokenVAE; else create LatentProjector
        hidden_dim: Hidden dimension for MLP layers
        num_layers: Number of layers
        dropout: Dropout rate
        kl_weight: KL weight for VAE

    Returns:
        projector: LatentProjector or TokenVAE instance
    """
    if use_vae:
        return TokenVAE(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            kl_weight=kl_weight,
        )
    else:
        return LatentProjector(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
