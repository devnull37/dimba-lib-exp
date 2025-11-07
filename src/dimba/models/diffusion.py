"""Core DIMBA diffusion model."""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from ..diffusion.schedules import CosineNoiseSchedule
from .embeddings import TokenEmbedding, TimestepEmbedding, PromptEncoder
from .denoiser import Mamba2Denoiser, DenoisingHead


class DIMBA(nn.Module):
    """DIMBA: Diffusion-based Mamba for non-autoregressive text generation.

    Combines diffusion process with Mamba-2 denoiser for parallel text generation.

    Args:
        vocab_size: Size of vocabulary
        d_model: Hidden dimension (default: 512)
        d_prompt: Prompt conditioning dimension (default: 512)
        num_diffusion_steps: Number of diffusion steps T (default: 1000)
        num_denoiser_layers: Number of Mamba-2 layers (default: 6)
        d_state: SSM state size (default: 16)
        d_conv: Convolution kernel size (default: 4)
        expand: Expansion factor for Mamba (default: 2)
        conditioning_type: 'film' or 'additive' (default: 'film')
        dropout: Dropout rate (default: 0.1)
        use_weight_tying: Whether to tie embedding and output weights (default: False)
        padding_idx: Padding token index (default: None)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        d_prompt: int = 512,
        num_diffusion_steps: int = 1000,
        num_denoiser_layers: int = 6,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        conditioning_type: str = "film",
        dropout: float = 0.1,
        use_weight_tying: bool = False,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_prompt = d_prompt
        self.num_diffusion_steps = num_diffusion_steps
        self.use_weight_tying = use_weight_tying

        # Token embeddings
        self.token_embed = TokenEmbedding(vocab_size, d_model, padding_idx=padding_idx)

        # Prompt encoder
        self.prompt_encoder = PromptEncoder(
            input_dim=d_model,
            hidden_dim=d_model * 2,
            output_dim=d_prompt,
            num_layers=2,
            dropout=dropout,
        )

        # Timestep embeddings
        self.timestep_embed = TimestepEmbedding(time_embed_dim=128, out_dim=512)

        # Diffusion schedule
        self.noise_schedule = CosineNoiseSchedule(num_steps=num_diffusion_steps)

        # Mamba-2 denoiser
        self.denoiser = Mamba2Denoiser(
            d_model=d_model,
            num_layers=num_denoiser_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            conditioning_type=conditioning_type,
            cond_dim=d_prompt,
            time_embed_dim=512,
            dropout=dropout,
        )

        # Output head
        if use_weight_tying:
            self.output_head = DenoisingHead(
                d_model=d_model,
                vocab_size=vocab_size,
                use_weight_tying=True,
                embedding_weight=self.token_embed.get_weight(),
            )
        else:
            self.output_head = DenoisingHead(
                d_model=d_model,
                vocab_size=vocab_size,
                use_weight_tying=False,
            )

    def encode_prompt(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Encode prompt to conditioning vectors.

        Args:
            input_ids: Prompt token IDs [batch_size, seq_len]

        Returns:
            conditioning: Prompt conditioning [batch_size, seq_len, d_prompt]
        """
        # Get embeddings
        embeddings = self.token_embed(input_ids)  # [batch_size, seq_len, d_model]

        # Encode to conditioning dimension
        conditioning = self.prompt_encoder(embeddings)  # [batch_size, seq_len, d_prompt]

        return conditioning

    def forward(
        self,
        input_ids: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass during training.

        Adds noise to input at timestep t and predicts clean embeddings.

        Args:
            input_ids: Target token IDs [batch_size, seq_len]
            t: Timesteps [batch_size], values in [0, num_diffusion_steps-1]
            noise: Optional predefined noise, otherwise sampled

        Returns:
            predicted_embeddings: Predicted clean embeddings [batch_size, seq_len, d_model]
            noise: The noise that was used for noising
        """
        # Get clean embeddings
        x_0 = self.token_embed(input_ids)  # [batch_size, seq_len, d_model]

        # Add noise according to schedule
        x_t, noise = self.noise_schedule.add_noise(x_0, t, noise)

        # Encode prompt from same input (in practice, could be different)
        cond = self.encode_prompt(input_ids)  # [batch_size, seq_len, d_prompt]

        # Get timestep embeddings
        time_emb = self.timestep_embed(t)  # [batch_size, 512]

        # Denoise
        x_pred = self.denoiser(x_t, cond, time_emb)  # [batch_size, seq_len, d_model]

        return x_pred, noise

    def denoise_step(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        prompt_cond: torch.Tensor,
    ) -> torch.Tensor:
        """Single denoising step.

        Used during inference to iteratively denoise.

        Args:
            x_t: Noisy embeddings [batch_size, seq_len, d_model]
            t: Current timestep [batch_size]
            prompt_cond: Prompt conditioning [batch_size, seq_len, d_prompt]

        Returns:
            x_pred: Predicted previous step [batch_size, seq_len, d_model]
        """
        # Get timestep embeddings
        time_emb = self.timestep_embed(t)  # [batch_size, 512]

        # Denoise
        x_pred = self.denoiser(x_t, prompt_cond, time_emb)

        return x_pred

    def get_noise_schedule(self):
        """Get access to noise schedule (useful for inference)."""
        return self.noise_schedule

    def get_alphas_cumprod(self) -> torch.Tensor:
        """Get cumulative alphas from noise schedule."""
        return self.noise_schedule.get_alphas_cumprod()
