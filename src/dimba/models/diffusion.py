"""Core DIMBA diffusion model.

DIMBA performs continuous Gaussian diffusion in a learned **latent** space (a VAE
or deterministic projector over token embeddings; raw-embedding diffusion is the
``latent_diffusion=False`` special case), denoised by a bidirectional Mamba
backbone for non-autoregressive, parallel text generation.

Key correctness changes vs. the original implementation (and vs. the v1 paper):

* **No conditioning leak.** The original built the prompt conditioning from the
  *clean target itself* (``C = PromptEncoder(X_0)``), so training could trivially
  read the answer through the conditioning path while inference saw a different
  prompt. We now condition only on the prompt: (a) a pooled prompt summary
  (never the response), and (b) when a ``prompt_mask`` is given, the prompt tokens
  are kept *clean in-sequence* and only the response is noised — so the bidirectional
  denoiser attends to real prompt context, exactly as at inference.
* **Consistent return.** ``forward`` always returns ``(x_pred, noise, latent_info)``
  (the trainer already unpacks three values).
* **Self-conditioning & classifier-free guidance hooks** (opt-in via
  ``self_conditioning`` and the ``drop_cond`` argument).
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Union

from ..diffusion.schedules import CosineNoiseSchedule
from .embeddings import TokenEmbedding, TimestepEmbedding, PromptEncoder, LatentProjector
from .denoiser import Mamba2Denoiser, DenoisingHead
from .vae import TokenVAE, TokenVAEWithDeterministicFallback


class DIMBA(nn.Module):
    """DIMBA: Diffusion-based Mamba for non-autoregressive text generation.

    Args:
        vocab_size: Size of vocabulary.
        d_model: Token embedding dimension (default 512).
        d_prompt: Prompt conditioning dimension (default 512).
        num_diffusion_steps: Number of diffusion steps T (default 1000).
        num_denoiser_layers: Number of Mamba blocks (default 6).
        d_state, d_conv, expand: SSM hyperparameters.
        conditioning_type: 'film' or 'additive'.
        dropout: Dropout rate.
        use_weight_tying: Tie embedding and output-head weights.
        padding_idx: Padding token index.
        use_simple_mamba: Force pure-PyTorch SSM (CPU/MPS).
        latent_diffusion: Diffuse in a projected latent space.
        d_latent: Latent dimension (defaults to d_model // 2 when latent).
        latent_projector_depth, latent_loss_weight, recon_loss_weight: latent options.
        use_vae_latent, vae_kl_weight, vae_checkpoint_path: VAE latent options.
        bidirectional: Use bidirectional Mamba scans (default True).
        self_conditioning: Feed the previous x0 estimate back into the denoiser.
        prediction_type: 'x0' (default) or 'v' (v-prediction).
        zero_terminal_snr: Enforce zero terminal SNR in the schedule (default True).
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
        use_simple_mamba: bool = False,
        use_gradient_checkpointing: bool = False,
        latent_diffusion: bool = False,
        d_latent: Optional[int] = None,
        latent_projector_depth: int = 2,
        latent_loss_weight: float = 1.0,
        recon_loss_weight: float = 1.0,
        use_vae_latent: bool = False,
        vae_kl_weight: float = 1.0,
        vae_checkpoint_path: Optional[str] = None,
        bidirectional: bool = True,
        self_conditioning: bool = False,
        prediction_type: str = "x0",
        zero_terminal_snr: bool = True,
        embed_init_std: float = 0.02,
        latent_scale: Optional[float] = None,
    ):
        super().__init__()

        if prediction_type not in ("x0", "v"):
            raise ValueError(f"prediction_type must be 'x0' or 'v', got {prediction_type}")

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_prompt = d_prompt
        self.num_diffusion_steps = num_diffusion_steps
        self.use_weight_tying = use_weight_tying
        self.latent_diffusion = latent_diffusion
        self.latent_loss_weight = latent_loss_weight
        self.recon_loss_weight = recon_loss_weight
        self.use_vae_latent = use_vae_latent
        self.bidirectional = bidirectional
        self.self_conditioning = self_conditioning
        self.prediction_type = prediction_type
        self.vae_kl_weight = vae_kl_weight

        if self.latent_diffusion:
            if d_latent is None:
                d_latent = max(1, d_model // 2)
            self.d_latent = d_latent
        else:
            self.d_latent = d_model

        # Scale applied to the encoded signal so the diffused tensor is ~unit
        # variance (standard diffusion assumes this for a calibrated SNR). For the
        # embedding path the signal is the token embedding (std ~= embed_init_std),
        # so the default brings it to ~unit variance. For the projector/VAE path the
        # output std is data-dependent -> default 1.0; call calibrate_latent_scale on
        # a representative batch before training. (cf. Stable Diffusion's 0.18215.)
        if latent_scale is None:
            latent_scale = (1.0 / embed_init_std) if not self.latent_diffusion else 1.0
        self.latent_scale = float(latent_scale)
        self.embed_init_std = float(embed_init_std)

        # Token embeddings
        self.token_embed = TokenEmbedding(
            vocab_size, d_model, padding_idx=padding_idx, init_std=embed_init_std
        )

        # Prompt encoder (used to build a pooled, response-free conditioning summary)
        self.prompt_encoder = PromptEncoder(
            input_dim=d_model,
            hidden_dim=d_model * 2,
            output_dim=d_prompt,
            num_layers=2,
            dropout=dropout,
        )

        # Latent projector (VAE or deterministic)
        self.latent_projector = None
        self.cond_projector = None
        if self.latent_diffusion:
            if self.use_vae_latent:
                vae = TokenVAE(
                    input_dim=d_model,
                    latent_dim=self.d_latent,
                    hidden_dim=max(d_model, self.d_latent),
                    num_layers=latent_projector_depth,
                    dropout=dropout,
                    kl_weight=vae_kl_weight,
                )
                if vae_checkpoint_path is not None:
                    checkpoint = torch.load(vae_checkpoint_path, map_location="cpu")
                    if "vae_state_dict" in checkpoint:
                        vae.load_state_dict(checkpoint["vae_state_dict"])
                    else:
                        vae.load_state_dict(checkpoint)
                    print(f"Loaded VAE checkpoint from {vae_checkpoint_path}")
                self.latent_projector = TokenVAEWithDeterministicFallback(
                    vae=vae,
                    use_vae_sampling=False,
                )
            else:
                self.latent_projector = LatentProjector(
                    input_dim=d_model,
                    latent_dim=self.d_latent,
                    hidden_dim=max(d_model, self.d_latent),
                    num_layers=latent_projector_depth,
                    dropout=dropout,
                )

            if d_prompt != self.d_latent:
                self.cond_projector = nn.Linear(d_prompt, self.d_latent)

        # Conditioning dimension fed to the denoiser.
        self.cond_dim = self.d_latent if self.latent_diffusion else d_prompt

        # Learned "null" conditioning for classifier-free guidance / unconditional use.
        self.null_cond = nn.Parameter(torch.zeros(d_prompt))

        # Self-conditioning fusion: [x_t ; x0_hat_prev] -> d_latent.
        if self.self_conditioning:
            self.self_cond_proj = nn.Linear(2 * self.d_latent, self.d_latent)
            self._init_self_cond_proj()
        else:
            self.self_cond_proj = None

        # Timestep embeddings
        self.timestep_embed = TimestepEmbedding(time_embed_dim=128, out_dim=512)

        # Diffusion schedule (now with a real zero-terminal-SNR option)
        self.noise_schedule = CosineNoiseSchedule(
            num_steps=num_diffusion_steps, zero_terminal_snr=zero_terminal_snr
        )

        # Mamba denoiser (bidirectional by default)
        self.denoiser = Mamba2Denoiser(
            d_model=self.d_latent,
            num_layers=num_denoiser_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            conditioning_type=conditioning_type,
            cond_dim=self.cond_dim,
            time_embed_dim=512,
            dropout=dropout,
            bidirectional=bidirectional,
            use_simple_mamba=use_simple_mamba,
            use_gradient_checkpointing=use_gradient_checkpointing,
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
                d_model=d_model, vocab_size=vocab_size, use_weight_tying=False
            )

        # Full constructor config, stored for faithful replicas (EMA / reload).
        # (vae_checkpoint_path is intentionally omitted; replicas copy weights.)
        self._config = dict(
            vocab_size=vocab_size,
            d_model=d_model,
            d_prompt=d_prompt,
            num_diffusion_steps=num_diffusion_steps,
            num_denoiser_layers=num_denoiser_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            conditioning_type=conditioning_type,
            dropout=dropout,
            use_weight_tying=use_weight_tying,
            padding_idx=padding_idx,
            use_simple_mamba=use_simple_mamba,
            use_gradient_checkpointing=use_gradient_checkpointing,
            latent_diffusion=latent_diffusion,
            d_latent=self.d_latent,
            latent_projector_depth=latent_projector_depth,
            latent_loss_weight=latent_loss_weight,
            recon_loss_weight=recon_loss_weight,
            use_vae_latent=use_vae_latent,
            vae_kl_weight=vae_kl_weight,
            bidirectional=bidirectional,
            self_conditioning=self_conditioning,
            prediction_type=prediction_type,
            zero_terminal_snr=zero_terminal_snr,
            embed_init_std=embed_init_std,
            latent_scale=self.latent_scale,
        )

    @property
    def config(self) -> dict:
        """Return a copy of the constructor configuration (for building replicas)."""
        return dict(self._config)

    def _init_self_cond_proj(self) -> None:
        """Initialize the self-conditioning fusion to ignore the (zero) prior estimate.

        At init, ``self_cond_proj([x_t ; 0]) == x_t`` so the model behaves like the
        non-self-conditioned version until it learns to use the prior estimate.
        """
        d = self.d_latent
        with torch.no_grad():
            w = torch.zeros(d, 2 * d)
            w[:, :d] = torch.eye(d)
            self.self_cond_proj.weight.copy_(w)
            self.self_cond_proj.bias.zero_()

    # ------------------------------------------------------------------ helpers

    def encode_prompt(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Encode tokens to per-position conditioning ``[B, L, d_prompt]`` (kept for compat)."""
        return self.prompt_encoder(self.token_embed(input_ids))

    def project_conditioning(self, conditioning: torch.Tensor) -> torch.Tensor:
        """Project conditioning from ``d_prompt`` into the latent space if needed."""
        if self.cond_projector is None:
            return conditioning
        return self.cond_projector(conditioning)

    def encode_latent(self, x_0: torch.Tensor) -> torch.Tensor:
        """Encode embeddings into the (scaled, ~unit-variance) diffusion signal."""
        z = x_0 if self.latent_projector is None else self.latent_projector.encode(x_0)
        return self.latent_scale * z

    def decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        """Invert :meth:`encode_latent`: unscale, then project back to embedding space."""
        z = z / self.latent_scale
        if self.latent_projector is None:
            return z
        return self.latent_projector.decode(z)

    @torch.no_grad()
    def calibrate_latent_scale(
        self, input_ids_or_embeds: torch.Tensor, target_std: float = 1.0
    ) -> float:
        """Set ``latent_scale`` so the encoded signal has ~``target_std`` per element.

        Call once on a representative batch *before* training (especially in
        latent/VAE mode, where the projector output std is data-dependent). This is
        the standard "measure the latent std, divide it out" calibration used by
        latent diffusion models so the noise schedule's SNR is meaningful.

        Args:
            input_ids_or_embeds: Token ids ``[B, L]`` or embeddings ``[B, L, d_model]``.
            target_std: Desired per-element std of the diffused signal (default 1.0).

        Returns:
            The new ``latent_scale``.
        """
        if input_ids_or_embeds.dim() == 2 and not torch.is_floating_point(input_ids_or_embeds):
            x_0 = self.token_embed(input_ids_or_embeds)
        else:
            x_0 = input_ids_or_embeds
        raw = x_0 if self.latent_projector is None else self.latent_projector.encode(x_0)
        std = raw.float().std().clamp(min=1e-6).item()
        self.latent_scale = float(target_std / std)
        self._config["latent_scale"] = self.latent_scale
        return self.latent_scale

    def _pooled_prompt(self, ids: torch.Tensor, prompt_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Mean-pool the prompt-encoder output over prompt positions -> ``[B, d_prompt]``."""
        cond = self.prompt_encoder(self.token_embed(ids))  # [B, L, d_prompt]
        if prompt_mask is not None:
            m = prompt_mask.to(cond.dtype).unsqueeze(-1)  # [B, L, 1]
            denom = m.sum(dim=1).clamp(min=1.0)
            return (cond * m).sum(dim=1) / denom
        return cond.mean(dim=1)

    def _build_conditioning(
        self,
        pooled: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Turn a pooled prompt (or the null embedding) into denoiser conditioning ``[B, 1, cond_dim]``."""
        if pooled is None:
            pooled = self.null_cond.unsqueeze(0).expand(batch_size, -1).to(device)
        cond = self.project_conditioning(pooled)  # [B, cond_dim]
        return cond.unsqueeze(1)  # broadcast over sequence length

    def conditioning_from_prompt(
        self,
        prompt_ids: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
        device: Optional[torch.device] = None,
        drop_cond: bool = False,
    ) -> torch.Tensor:
        """Public helper for samplers: build conditioning from a prompt (or null)."""
        if drop_cond or prompt_ids is None:
            assert batch_size is not None and device is not None
            return self._build_conditioning(None, batch_size, device)
        pooled = self._pooled_prompt(prompt_ids, prompt_mask=None)
        return self._build_conditioning(pooled, prompt_ids.shape[0], prompt_ids.device)

    def _denoiser_raw(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        x_self_cond: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Run the denoiser and return its *raw* prediction (x0 or v per prediction_type)."""
        if self.self_conditioning and self.self_cond_proj is not None:
            sc = x_self_cond if x_self_cond is not None else torch.zeros_like(x_t)
            denoiser_in = self.self_cond_proj(torch.cat([x_t, sc], dim=-1))
        else:
            denoiser_in = x_t
        return self.denoiser(denoiser_in, cond, self.timestep_embed(t))

    def _to_x0_latent(self, x_t: torch.Tensor, raw: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Convert a raw denoiser prediction to a clean-latent (x0) estimate."""
        if self.prediction_type == "v":
            return self.noise_schedule.predict_x0_from_v(x_t, raw, t)
        return raw

    def denoise_to_x0_latent(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        x_self_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Single denoise -> predicted clean latent ``z0_hat`` (used by samplers)."""
        raw = self._denoiser_raw(x_t, t, cond, x_self_cond)
        return self._to_x0_latent(x_t, raw, t)

    # --------------------------------------------------------------- forward

    def forward(
        self,
        input_ids: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        prompt_mask: Optional[torch.Tensor] = None,
        x_self_cond: Optional[torch.Tensor] = None,
        drop_cond: bool = False,
        return_latent_info: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict]]:
        """Training forward pass.

        Args:
            input_ids: Token IDs ``[B, L]`` (full sequence; prompt + response).
            t: Timesteps ``[B]`` in ``[0, num_diffusion_steps - 1]``.
            noise: Optional pre-sampled noise.
            prompt_mask: Optional bool ``[B, L]``, True where a position is *clean
                prompt context* (not noised, not part of the loss). None -> the whole
                sequence is diffused (unconditional / LM pretraining).
            x_self_cond: Optional previous ``z0_hat`` for self-conditioning.
            drop_cond: If True, use the null conditioning (for classifier-free guidance training).
            return_latent_info: kept for API compatibility; the 3-tuple is always returned.

        Returns:
            ``(x_pred, noise, latent_info)`` where ``x_pred`` is the predicted clean
            embedding ``[B, L, d_model]`` and ``latent_info`` carries tensors the
            trainer needs (raw prediction, clean latent, x_t, diffuse_mask, ...).
        """
        batch_size = input_ids.shape[0]
        x_0 = self.token_embed(input_ids)
        z_0 = self.encode_latent(x_0)

        # VAE KL (computed on the clean embeddings) if using a VAE latent.
        vae_kl_loss = None
        if self.use_vae_latent and self.latent_projector is not None:
            _, vae_stats = self.latent_projector(x_0, return_stats=True)
            if vae_stats is not None:
                vae_kl_loss = -0.5 * torch.sum(
                    1 + vae_stats["logvar"] - vae_stats["mu"].pow(2) - vae_stats["logvar"].exp()
                )

        # Forward diffusion; keep prompt positions clean when a prompt_mask is given.
        x_t, noise = self.noise_schedule.add_noise(z_0, t, noise)
        diffuse_mask = None
        if prompt_mask is not None:
            keep = prompt_mask.unsqueeze(-1)  # [B, L, 1] bool
            x_t = torch.where(keep, z_0, x_t)
            diffuse_mask = ~prompt_mask

        # Conditioning: pooled prompt (response-free) or null. Never the target.
        if drop_cond or prompt_mask is None:
            pooled = None  # unconditional / CFG-dropped -> null embedding
        else:
            pooled = self._pooled_prompt(input_ids, prompt_mask)
        cond = self._build_conditioning(pooled, batch_size, input_ids.device)

        # Denoise -> raw prediction -> clean latent -> decode to embedding space.
        raw = self._denoiser_raw(x_t, t, cond, x_self_cond)
        z0_hat = self._to_x0_latent(x_t, raw, t)
        x_pred = self.decode_latent(z0_hat)

        latent_info: Dict[str, Optional[torch.Tensor]] = {
            "pred_raw": raw,
            "z0_hat": z0_hat,
            "z_0": z_0,
            "x_t": x_t,
            "diffuse_mask": diffuse_mask,
            "noise": noise,
        }
        if self.latent_diffusion:
            # Backwards-compatible keys for the existing latent loss in the trainer.
            latent_info["z_pred"] = z0_hat
        if vae_kl_loss is not None:
            latent_info["vae_kl_loss"] = vae_kl_loss

        return x_pred, noise, latent_info

    def denoise_step(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        prompt_cond: torch.Tensor,
        x_self_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Single denoising step for inference (predicts the clean latent ``z0_hat``).

        ``prompt_cond`` is the ``[B, 1, cond_dim]`` conditioning from
        :meth:`conditioning_from_prompt`.
        """
        return self.denoise_to_x0_latent(x_t, t, prompt_cond, x_self_cond)

    def _to_timestep_index(
        self, t, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        """Coerce a timestep (int index, float in (0,1], scalar, or [B]) to a long [B] index."""
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=device)
        t = t.to(device)
        if t.dim() == 0:
            t = t.expand(batch_size)
        if torch.is_floating_point(t):
            # Masked-diffusion continuous time in (0, 1] -> discrete schedule index.
            t = (t.clamp(0.0, 1.0) * (self.num_diffusion_steps - 1)).round()
        return t.long()

    def predict_token_logits(self, input_ids: torch.Tensor, t) -> torch.Tensor:
        """Per-position token logits for the discrete / masked-diffusion track.

        Unlike :meth:`forward` (which adds Gaussian noise to latents), the masked
        track corrupts by replacing tokens with ``[MASK]``: the (already-masked)
        ``input_ids`` are embedded, denoised conditioned on the timestep, and
        projected to vocabulary logits. Prompt context comes from the *unmasked*
        tokens already present in ``input_ids`` (the bidirectional denoiser attends
        to them), so no separate prompt conditioning is required.

        Args:
            input_ids: Possibly-masked token ids ``[B, L]``.
            t: Timestep(s): an int/long index in ``[0, T)`` or a float in ``(0, 1]``
                (masked-diffusion continuous time); scalar or ``[B]``.

        Returns:
            Token logits ``[B, L, vocab_size]``.
        """
        batch_size = input_ids.shape[0]
        z = self.encode_latent(self.token_embed(input_ids))
        cond = self._build_conditioning(None, batch_size, input_ids.device)
        t_idx = self._to_timestep_index(t, batch_size, input_ids.device)
        raw = self._denoiser_raw(z, t_idx, cond, None)
        x_dec = self.decode_latent(raw)
        return self.output_head(x_dec, embedding_weight=self.token_embed.get_weight())

    def get_noise_schedule(self) -> CosineNoiseSchedule:
        """Access the noise schedule."""
        return self.noise_schedule

    def get_alphas_cumprod(self) -> torch.Tensor:
        """Cumulative alphas from the noise schedule."""
        return self.noise_schedule.get_alphas_cumprod()
