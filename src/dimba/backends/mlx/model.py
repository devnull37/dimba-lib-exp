"""Full DIMBA inference on the Apple GPU via MLX.

This runs the *entire* Shakespeare sampling loop — token embedding, latent projector,
FiLM conditioning, the 12 bidirectional Mamba-2 (SSD) blocks, the timestep embedding, the
cosine/zero-terminal-SNR schedule and the x0-parameterized DDIM update — on the Apple GPU
through MLX, instead of only the mixer. The PyTorch model (``dimba.models.diffusion.DIMBA``)
stays the source of truth: :meth:`MLXDIMBA.from_torch` copies its weights and the result
matches the PyTorch sampler's logits numerically.

Scope: the ``dimbapeare1-30m`` configuration — ``latent_diffusion=True``, ``conditioning=film``,
``use_weight_tying=True``, ``prediction_type='x0'``, ``self_conditioning=False``,
``bidirectional=True``. Both unconditional and prompt-conditioned sampling are supported.

If ``mlx`` is missing the module imports but :class:`MLXDIMBA` raises on construction.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .mamba2 import HAS_MLX, MLXMamba2Mixer, load_torch_mamba2_state_dict

try:
    import mlx.core as mx
    import mlx.nn as mlx_nn
except ImportError:  # pragma: no cover
    mx = None  # type: ignore
    mlx_nn = None  # type: ignore

__all__ = ["MLXDIMBA"]

_NO_MLX = "MLX not installed (pip install mlx, Apple Silicon only)."


def _to_mx(t) -> "mx.array":
    arr = t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
    return mx.array(arr.astype(np.float32))


def _linear(x, w, b=None):
    """torch ``nn.Linear``: ``x @ w.T (+ b)`` with ``w`` shape ``[out, in]``."""
    y = mx.matmul(x, w.T)
    return y + b if b is not None else y


def _flip_seq(x):
    """Reverse along the sequence axis (this MLX build lacks ``mx.flip``)."""
    idx = mx.array(np.arange(x.shape[1] - 1, -1, -1, dtype=np.int32))
    return mx.take(x, idx, axis=1)


class MLXDIMBA:
    """MLX (Apple-GPU) replica of a PyTorch :class:`DIMBA` for inference/sampling."""

    def __init__(self, config: Dict[str, Any]):
        if not HAS_MLX:
            raise RuntimeError(_NO_MLX)
        self.cfg = dict(config)
        self.d_latent = config["d_latent"]
        self.d_model = config["d_model"]
        self.vocab_size = config["vocab_size"]
        self.num_layers = config["num_denoiser_layers"]
        self.num_diffusion_steps = config["num_diffusion_steps"]
        self.latent_scale = float(config.get("latent_scale", 1.0))
        # 12 forward + 12 backward SSD mixers (mixer d_model == d_latent).
        mk = lambda: MLXMamba2Mixer(
            d_model=self.d_latent, d_state=config["d_state"],
            d_conv=config["d_conv"], expand=config["expand"],
        )
        self.mixers_fwd: List[MLXMamba2Mixer] = [mk() for _ in range(self.num_layers)]
        self.mixers_bwd: List[MLXMamba2Mixer] = [mk() for _ in range(self.num_layers)]
        # Plain-array parameters (filled by from_torch).
        self.p: Dict[str, Any] = {}

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_torch(cls, torch_model) -> "MLXDIMBA":
        """Build from a constructed PyTorch ``DIMBA`` (weights copied to MLX)."""
        cfg = dict(torch_model.config)
        cfg["latent_scale"] = float(torch_model.latent_scale)
        self = cls(cfg)
        sd = {k: v for k, v in torch_model.state_dict().items()}
        g = lambda k: sd[k]
        p = self.p
        # embeddings / conditioning / schedule
        p["token_embed_w"] = _to_mx(g("token_embed.embedding.weight"))   # [V, d_model]
        p["null_cond"] = _to_mx(g("null_cond"))                          # [d_prompt]
        p["ts_pe"] = _to_mx(g("timestep_embed.position_encoding"))       # [10000, 128]
        p["ts_w0"], p["ts_b0"] = _to_mx(g("timestep_embed.mlp.0.weight")), _to_mx(g("timestep_embed.mlp.0.bias"))
        p["ts_w2"], p["ts_b2"] = _to_mx(g("timestep_embed.mlp.2.weight")), _to_mx(g("timestep_embed.mlp.2.bias"))
        p["time_proj_w"], p["time_proj_b"] = _to_mx(g("denoiser.time_proj.weight")), _to_mx(g("denoiser.time_proj.bias"))
        p["acp"] = _to_mx(g("noise_schedule.alphas_cumprod"))           # [T]
        # latent projector decoder (decode_latent) + encoder (encode, for prompts)
        p["dec0_w"], p["dec0_b"] = _to_mx(g("latent_projector.decoder.0.weight")), _to_mx(g("latent_projector.decoder.0.bias"))
        p["dec3_w"], p["dec3_b"] = _to_mx(g("latent_projector.decoder.3.weight")), _to_mx(g("latent_projector.decoder.3.bias"))
        p["enc0_w"], p["enc0_b"] = _to_mx(g("latent_projector.encoder.0.weight")), _to_mx(g("latent_projector.encoder.0.bias"))
        p["enc3_w"], p["enc3_b"] = _to_mx(g("latent_projector.encoder.3.weight")), _to_mx(g("latent_projector.encoder.3.bias"))
        # prompt encoder (for conditional sampling)
        p["pe0_w"], p["pe0_b"] = _to_mx(g("prompt_encoder.mlp.0.weight")), _to_mx(g("prompt_encoder.mlp.0.bias"))
        p["pe3_w"], p["pe3_b"] = _to_mx(g("prompt_encoder.mlp.3.weight")), _to_mx(g("prompt_encoder.mlp.3.bias"))
        # per-layer: FiLM gamma/beta, block LayerNorm, and the two mixers
        p["film_g_w"], p["film_g_b"], p["film_b_w"], p["film_b_b"] = [], [], [], []
        p["ln_w"], p["ln_b"] = [], []
        for i in range(self.num_layers):
            c = f"denoiser.conditioning.{i}."
            p["film_g_w"].append(_to_mx(g(c + "gamma_proj.weight"))); p["film_g_b"].append(_to_mx(g(c + "gamma_proj.bias")))
            p["film_b_w"].append(_to_mx(g(c + "beta_proj.weight"))); p["film_b_b"].append(_to_mx(g(c + "beta_proj.bias")))
            b = f"denoiser.blocks.{i}."
            p["ln_w"].append(_to_mx(g(b + "norm.weight"))); p["ln_b"].append(_to_mx(g(b + "norm.bias")))
            fwd = {k[len(b + "mamba_fwd."):]: v for k, v in sd.items() if k.startswith(b + "mamba_fwd.")}
            bwd = {k[len(b + "mamba_bwd."):]: v for k, v in sd.items() if k.startswith(b + "mamba_bwd.")}
            load_torch_mamba2_state_dict(self.mixers_fwd[i], fwd)
            load_torch_mamba2_state_dict(self.mixers_bwd[i], bwd)
        mx.eval([v for v in p.values() if isinstance(v, mx.array)])
        return self

    # ------------------------------------------------------------------ pieces
    def _timestep_emb(self, t_idx: int):
        p = self.p
        pe = p["ts_pe"][t_idx]                                   # [128]
        h = _linear(pe, p["ts_w0"], p["ts_b0"])
        h = mlx_nn.silu(h)
        h = _linear(h, p["ts_w2"], p["ts_b2"])                  # [512]
        return h

    def _denoiser(self, x_t, cond, t_idx):
        """x0 prediction. x_t:[B,L,d_latent], cond:[B,1,d_latent]."""
        p = self.p
        t_emb = self._timestep_emb(t_idx)                       # [512]
        time_cond = _linear(t_emb, p["time_proj_w"], p["time_proj_b"]).reshape(1, 1, -1)
        combined = cond + time_cond                              # [B,1,d_latent]
        x = x_t
        for i in range(self.num_layers):
            gamma = _linear(combined, p["film_g_w"][i], p["film_g_b"][i])
            beta = _linear(combined, p["film_b_w"][i], p["film_b_b"][i])
            conditioned = gamma * x + beta                       # FiLM, broadcast over L
            h = mx.fast.layer_norm(conditioned, p["ln_w"][i], p["ln_b"][i], 1e-5)
            y = self.mixers_fwd[i](h)
            y = y + _flip_seq(self.mixers_bwd[i](_flip_seq(h)))
            x = conditioned + y
        return x

    def _decode_logits(self, z):
        """decode_latent -> output head (weight-tied). z:[B,L,d_latent] -> [B,L,V]."""
        p = self.p
        h = z / self.latent_scale
        h = _linear(h, p["dec0_w"], p["dec0_b"])
        h = mlx_nn.gelu(h)
        h = _linear(h, p["dec3_w"], p["dec3_b"])                # [B,L,d_model]
        return mx.matmul(h, p["token_embed_w"].T)               # [B,L,V]

    def _encode_prompt_latent(self, prompt_ids: np.ndarray):
        """Embed + project prompt ids -> clean latent prefix. ids:[B,P] -> [B,P,d_latent]."""
        p = self.p
        emb = p["token_embed_w"][mx.array(prompt_ids.astype(np.int32))]  # [B,P,d_model]
        h = _linear(emb, p["enc0_w"], p["enc0_b"]); h = mlx_nn.gelu(h)
        h = _linear(h, p["enc3_w"], p["enc3_b"])                # [B,P,d_latent]
        return self.latent_scale * h, emb

    def _pooled_cond(self, prompt_emb):
        """Mean-pooled prompt-encoder summary -> [B,1,d_prompt]."""
        p = self.p
        h = _linear(prompt_emb, p["pe0_w"], p["pe0_b"]); h = mlx_nn.gelu(h)
        h = _linear(h, p["pe3_w"], p["pe3_b"])                  # [B,P,d_prompt]
        return mx.mean(h, axis=1, keepdims=True)                # [B,1,d_prompt]

    # ------------------------------------------------------------------ sampling
    def sample_logits(self, noise: np.ndarray, num_steps: Optional[int] = None,
                      prompt_ids: Optional[np.ndarray] = None):
        """Deterministic DDIM (eta=0) -> final response logits ``[B, seq_len, V]`` (mx.array).

        ``noise`` is the initial response latent ``[B, seq_len, d_latent]`` (pass the *same*
        array used by the torch sampler for a faithful comparison).
        """
        T = self.num_diffusion_steps
        num_steps = num_steps or T
        ns = min(num_steps, T)
        timesteps = np.linspace(T - 1, 0, ns).round().astype(int)
        acp = self.p["acp"]
        B = noise.shape[0]

        # conditioning + (optional) clean prompt prefix
        if prompt_ids is not None:
            prompt_latent, prompt_emb = self._encode_prompt_latent(prompt_ids)
            prompt_len = prompt_latent.shape[1]
            cond = self._pooled_cond(prompt_emb)                # [B,1,d_prompt(=d_latent)]
            x_t = mx.concatenate([prompt_latent, mx.array(noise)], axis=1)
        else:
            prompt_latent, prompt_len = None, 0
            cond = mx.broadcast_to(self.p["null_cond"].reshape(1, 1, -1), (B, 1, self.d_latent))
            x_t = mx.array(noise)

        for i in range(ns):
            t_val = int(timesteps[i])
            z0 = self._denoiser(x_t, cond, t_val)               # x0 prediction
            acp_t = acp[t_val]
            acp_prev = acp[int(timesteps[i + 1])] if i < ns - 1 else mx.array(1.0)
            sqrt_acp_t = mx.sqrt(acp_t)
            sqrt_om_t = mx.sqrt(mx.maximum(1.0 - acp_t, 1e-8))
            eps_hat = (x_t - sqrt_acp_t * z0) / sqrt_om_t
            dir_coef = mx.sqrt(mx.maximum(1.0 - acp_prev, 0.0))  # eta=0 -> sigma=0
            x_t = mx.sqrt(acp_prev) * z0 + dir_coef * eps_hat
            if prompt_latent is not None:                        # hold prefix clean
                x_t = mx.concatenate([prompt_latent, x_t[:, prompt_len:, :]], axis=1)
            mx.eval(x_t)

        response = x_t[:, prompt_len:, :]
        logits = self._decode_logits(response)
        mx.eval(logits)
        return logits

    def sample(self, seq_len: int, num_steps: Optional[int] = None, temperature: float = 1.0,
               prompt_ids: Optional[np.ndarray] = None, noise: Optional[np.ndarray] = None,
               seed: int = 0) -> np.ndarray:
        """Generate token ids ``[B, seq_len]`` (greedy by default; set temperature for sampling)."""
        if noise is None:
            rng = np.random.default_rng(seed)
            B = 1 if prompt_ids is None else prompt_ids.shape[0]
            noise = rng.standard_normal((B, seq_len, self.d_latent)).astype(np.float32)
        logits = np.array(self.sample_logits(noise, num_steps, prompt_ids))
        logits = logits / max(temperature, 1e-6)
        if temperature <= 1e-6:
            return logits.argmax(-1)
        # softmax sample
        logits = logits - logits.max(-1, keepdims=True)
        probs = np.exp(logits); probs /= probs.sum(-1, keepdims=True)
        rng = np.random.default_rng(seed + 1)
        B, L, V = probs.shape
        out = np.empty((B, L), dtype=np.int64)
        for b in range(B):
            for l in range(L):
                out[b, l] = rng.choice(V, p=probs[b, l])
        return out
