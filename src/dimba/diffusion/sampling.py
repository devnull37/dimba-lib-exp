"""Sampling and inference procedures for DIMBA."""

import torch
import torch.nn.functional as F
from typing import Optional, List


def sample_from_model(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    seq_len: int,
    num_steps: Optional[int] = None,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Generate text from the DIMBA model.

    Iteratively refines embeddings from noise using the denoiser.

    Args:
        model: DIMBA model instance
        prompt_ids: Prompt token IDs [batch_size, prompt_len]
        seq_len: Length of text to generate
        num_steps: Number of diffusion steps (uses model's default if None)
        temperature: Sampling temperature for logit rescaling
        top_k: Top-k sampling parameter (None for no filtering)
        top_p: Top-p (nucleus) sampling parameter (None for no filtering)
        device: Device to run on (defaults to model's device)

    Returns:
        generated_ids: Generated token IDs [batch_size, seq_len]
    """
    if device is None:
        device = next(model.parameters()).device

    if num_steps is None:
        num_steps = model.num_diffusion_steps

    batch_size = prompt_ids.shape[0]
    model.eval()

    with torch.no_grad():
        # Encode prompt to conditioning
        prompt_cond = model.encode_prompt(prompt_ids.to(device))  # [batch_size, prompt_len, d_prompt]

        # Pad/extend conditioning to match generation length
        d_prompt = prompt_cond.shape[-1]
        if prompt_cond.shape[1] < seq_len:
            # Pad with zeros
            pad_size = seq_len - prompt_cond.shape[1]
            padding = torch.zeros(batch_size, pad_size, d_prompt, device=device)
            cond = torch.cat([prompt_cond, padding], dim=1)
        else:
            cond = prompt_cond[:, :seq_len, :]

        # Initialize with noise
        x_t = torch.randn(batch_size, seq_len, model.d_model, device=device)

        # Get noise schedule
        noise_schedule = model.get_noise_schedule()
        alphas_cumprod = model.get_alphas_cumprod().to(device)

        # Iterative denoising loop
        timesteps = torch.linspace(num_steps - 1, 0, num_steps, dtype=torch.long, device=device)

        for i, t_continuous in enumerate(timesteps):
            # Print progress every 10 steps
            if i % max(1, num_steps // 10) == 0:
                print(f"  Denoising step {i+1}/{num_steps}")
            # Get discrete timestep
            t = torch.full((batch_size,), t_continuous.item(), dtype=torch.long, device=device)

            # Denoise step
            x_pred = model.denoise_step(x_t, t, cond)

            # Compute previous timestep noise
            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1].long()
                alpha_t = alphas_cumprod[t]  # [batch_size]
                alpha_prev = alphas_cumprod[t_prev]  # [batch_size]

                # Simple denoising: interpolate towards cleaner prediction
                sigma_t = torch.sqrt((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev))
                sigma_t = sigma_t.view(-1, 1, 1)

                noise = torch.randn_like(x_t)
                x_t = (x_pred + sigma_t * noise) * torch.sqrt(alpha_prev / alpha_t).view(-1, 1, 1)
            else:
                x_t = x_pred

        # Project to logits and sample
        logits = model.output_head(x_t)  # [batch_size, seq_len, vocab_size]

        # Apply temperature
        logits = logits / temperature

        # Apply top-k and top-p filtering
        if top_k is not None or top_p is not None:
            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

        # Sample tokens
        probs = F.softmax(logits, dim=-1)
        # Handle potential NaN values from -inf logits
        probs = torch.nan_to_num(probs, nan=0.0)
        # Renormalize in case filtering produced NaNs
        prob_sum = probs.sum(dim=-1, keepdim=True)
        # If sum is effectively zero, use uniform distribution
        probs = torch.where(prob_sum > 1e-6, probs / prob_sum, torch.ones_like(probs) / probs.shape[-1])
        generated_ids = torch.multinomial(probs.view(-1, probs.shape[-1]), num_samples=1)
        generated_ids = generated_ids.view(batch_size, seq_len)

    return generated_ids


def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: Optional[int] = 0,
    top_p: Optional[float] = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> torch.Tensor:
    """Filter a distribution of logits using top-k and/or top-p filtering.

    Args:
        logits: Logits distribution [batch_size, seq_len, vocab_size]
        top_k: Keep only top k tokens with highest probability (None to disable)
        top_p: Keep the top tokens with cumulative probability >= top_p (None to disable)
        filter_value: Value to use for filtered tokens
        min_tokens_to_keep: Minimum number of tokens to keep per sample

    Returns:
        filtered_logits: Filtered logits with same shape as input
    """
    if top_k is not None and top_k > 0:
        # Top-k filtering
        indices_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, filter_value)

    if top_p is not None and top_p < 1.0:
        # Top-p (nucleus) filtering
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumsum_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)

        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumsum_probs > top_p

        # Keep at least min_tokens_to_keep
        sorted_indices_to_remove[..., :min_tokens_to_keep] = 0

        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
        indices_to_remove.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, filter_value)

    return logits


def sample_timesteps(
    batch_size: int,
    num_steps: int,
    device: torch.device,
) -> torch.Tensor:
    """Sample random timesteps for training.

    Args:
        batch_size: Batch size
        num_steps: Total number of diffusion steps
        device: Device to create tensor on

    Returns:
        timesteps: Random timesteps [batch_size]
    """
    return torch.randint(0, num_steps, (batch_size,), device=device)


class DDIMSampler:
    """DDIM-style accelerated sampling.

    Accelerates inference by skipping denoising steps while maintaining quality.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        num_steps: int = 50,
        ddim_eta: float = 0.0,
    ):
        self.model = model
        self.num_steps = num_steps
        self.ddim_eta = ddim_eta
        self.device = next(model.parameters()).device

    def sample(
        self,
        prompt_ids: torch.Tensor,
        seq_len: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Generate text using DDIM sampling.

        Args:
            prompt_ids: Prompt token IDs
            seq_len: Length of text to generate
            temperature: Sampling temperature

        Returns:
            generated_ids: Generated token IDs
        """
        batch_size = prompt_ids.shape[0]
        self.model.eval()

        with torch.no_grad():
            # Encode prompt
            prompt_cond = self.model.encode_prompt(prompt_ids.to(self.device))

            # Pad conditioning
            d_prompt = prompt_cond.shape[-1]
            if prompt_cond.shape[1] < seq_len:
                pad_size = seq_len - prompt_cond.shape[1]
                padding = torch.zeros(batch_size, pad_size, d_prompt, device=self.device)
                cond = torch.cat([prompt_cond, padding], dim=1)
            else:
                cond = prompt_cond[:, :seq_len, :]

            # Initialize with noise
            x_t = torch.randn(batch_size, seq_len, self.model.d_model, device=self.device)

            # Get noise schedule
            alphas = self.model.get_alphas_cumprod().to(self.device)

            # DDIM timestep schedule: uniformly spaced subset
            total_steps = self.model.num_diffusion_steps
            skip = total_steps // self.num_steps
            timesteps = list(range(0, total_steps, skip))[:self.num_steps]
            timesteps = sorted(timesteps, reverse=True)

            for i, t in enumerate(timesteps):
                t_tensor = torch.full((batch_size,), t, dtype=torch.long, device=self.device)

                # Denoise
                x_pred = self.model.denoise_step(x_t, t_tensor, cond)

                if i < len(timesteps) - 1:
                    t_next = timesteps[i + 1]
                    alpha_t = alphas[t]
                    alpha_next = alphas[t_next]

                    # DDIM update
                    sigma = self.ddim_eta * torch.sqrt((1 - alpha_next) / (1 - alpha_t) * (1 - alpha_t / alpha_next))
                    sigma = sigma.view(-1, 1, 1)

                    noise = torch.randn_like(x_t)
                    x_t = x_pred + sigma * noise
                else:
                    x_t = x_pred

            # Project to logits and sample
            logits = self.model.output_head(x_t) / temperature
            probs = F.softmax(logits, dim=-1)
            generated_ids = torch.multinomial(probs.view(-1, probs.shape[-1]), num_samples=1)
            generated_ids = generated_ids.view(batch_size, seq_len)

        return generated_ids
