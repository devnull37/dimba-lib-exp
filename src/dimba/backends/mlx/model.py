"""Full DIMBA inference on the Apple GPU via MLX.

This runs the *entire* Shakespeare sampling loop — token embedding, latent projector,
FiLM/AdaLN conditioning, the bidirectional Mamba-2 (SSD) blocks, the timestep embedding,
the cosine/zero-terminal-SNR schedule and the x0-/v-parameterized DDIM update — on the
Apple GPU through MLX, instead of only the mixer. The PyTorch model
(``dimba.models.diffusion.DIMBA``) stays the source of truth:
:meth:`MLXDIMBA.from_torch` copies its weights and the result matches the PyTorch
sampler's logits numerically.

Supported feature matrix (v1 = dimbapeare1-30m baseline; v2 = full new config):
  * conditioning_type: "film" (v1) and "adaln" (v2)
  * prediction_type:   "x0" (v1) and "v" (v2)
  * use_head_norm:     False (v1) and True (v2)
  * latent_norm:       False (v1) and True (v2)
  * self_conditioning: False (v1) and True (v2)
  * head_type:         "linear" (v1) and "attn" (v2)
  * noise_dist:        "gaussian" (both; student_t not supported)
  * bidirectional:     True (both)
  * use_weight_tying:  True (both)
  * latent_diffusion:  False (embedding-space, current base) and True (projector)
  * use_flow_matching: False (DDIM/DPM cosine schedule) and True (Euler/Heun ODE)
  * block_ffn:         False and True (per-block SwiGLU/MLP FFN, 6-param adaln only)

Two samplers are wired: a deterministic DDIM path (cosine schedule) and a
flow-matching Euler/Heun ODE path. ``sample`` auto-selects flow when the model
config has ``use_flow_matching=True``.
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


def _layer_norm(x, w, b, eps=1e-5):
    """Manual LayerNorm: mirrors ``mx.fast.layer_norm`` but always available."""
    try:
        return mx.fast.layer_norm(x, w, b, eps)
    except Exception:
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.mean((x - mean) ** 2, axis=-1, keepdims=True)
        return w * (x - mean) / mx.sqrt(var + eps) + b


def _rms_norm(x, w, eps=1e-6):
    """RMSNorm: no mean subtraction, no bias (mirrors PyTorch RMSNorm in denoiser)."""
    rms = mx.sqrt(mx.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return (x / rms) * w


def _norm(x, w, b, eps=1e-5):
    """Dispatch to RMSNorm (b is None) or LayerNorm (b is an array)."""
    if b is None:
        return _rms_norm(x, w, eps)
    return _layer_norm(x, w, b, eps)


class MLXDIMBA:
    """MLX (Apple-GPU) replica of a PyTorch :class:`DIMBA` for inference/sampling.

    Supports both the v1 config (film/x0/linear head/no norm/no self-cond) and the
    full v2 config (adaln/v-pred/attn head/latent-norm/head-norm/self-conditioning).
    """

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
        self.conditioning_type = config.get("conditioning_type", "film")
        self.prediction_type = config.get("prediction_type", "x0")
        self.use_head_norm = bool(config.get("use_head_norm", False))
        self.latent_norm = bool(config.get("latent_norm", False))
        self.self_conditioning = bool(config.get("self_conditioning", False))
        self.head_type = config.get("head_type", "linear")
        self.head_attn_layers = int(config.get("head_attn_layers", 2))
        self.use_weight_tying = bool(config.get("use_weight_tying", False))
        # cond_projector: maps d_prompt -> d_latent when they differ
        self.d_prompt = config.get("d_prompt", config["d_model"])
        self.latent_diffusion = bool(config.get("latent_diffusion", False))
        self.has_cond_projector = (self.d_prompt != self.d_latent and self.latent_diffusion)
        # Conditioning dimension fed to the denoiser (mirrors DIMBA.cond_dim):
        #   latent space -> d_latent ;  embedding space -> d_prompt.
        self.cond_dim = self.d_latent if self.latent_diffusion else self.d_prompt
        # Flow matching + per-block FFN (block_ffn).
        self.use_flow_matching = bool(config.get("use_flow_matching", False))
        self.block_ffn = bool(config.get("block_ffn", False))
        self.ffn_type = config.get("ffn_type", "mlp")

        # 12 forward + 12 backward SSD mixers (mixer d_model == d_latent).
        # headdim must divide d_inner = d_latent * expand.  Default 64; shrink if needed.
        d_inner = self.d_latent * config["expand"]
        headdim = min(64, d_inner)
        mk = lambda: MLXMamba2Mixer(
            d_model=self.d_latent, d_state=config["d_state"],
            d_conv=config["d_conv"], expand=config["expand"],
            headdim=headdim,
        )
        self.mixers_fwd: List[MLXMamba2Mixer] = [mk() for _ in range(self.num_layers)]
        self.mixers_bwd: List[MLXMamba2Mixer] = [mk() for _ in range(self.num_layers)]
        # Plain-array parameters (filled by from_torch).
        self.p: Dict[str, Any] = {}

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_torch(cls, torch_model) -> "MLXDIMBA":
        """Build from a constructed PyTorch ``DIMBA`` (weights copied to MLX).

        Supports both v1 (film/x0) and v2 (adaln/v-pred/head-norm/latent-norm/
        self-conditioning/attn-head) configs. Only ``noise_dist != gaussian`` and
        ``bidirectional=False`` remain unsupported.
        """
        cfg = dict(torch_model.config)
        _unsupported = []
        if cfg.get("noise_dist", "gaussian") != "gaussian":
            _unsupported.append(f"noise_dist={cfg.get('noise_dist')}")
        if not cfg.get("bidirectional", True):
            _unsupported.append("bidirectional=False")
        if cfg.get("conditioning_type", "film") not in ("film", "adaln"):
            _unsupported.append(f"conditioning_type={cfg.get('conditioning_type')}")
        if cfg.get("head_type", "linear") not in ("linear", "attn"):
            _unsupported.append(f"head_type={cfg.get('head_type')}")
        # block_ffn is only supported on the adaln (6-param) path: the FFN modulation
        # comes from the second (scale2, shift2, gate2) triplet of AdaLNZeroConditioning.
        if cfg.get("block_ffn", False) and cfg.get("conditioning_type", "film") != "adaln":
            _unsupported.append(
                f"block_ffn=True with conditioning_type={cfg.get('conditioning_type')} "
                "(only adaln is wired)"
            )
        if cfg.get("ffn_type", "mlp") not in ("mlp", "swiglu"):
            _unsupported.append(f"ffn_type={cfg.get('ffn_type')}")
        if _unsupported:
            raise NotImplementedError(
                f"MLXDIMBA does not support: {_unsupported}. "
                "Use the PyTorch backend for these configs."
            )

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
        p["acp"] = _to_mx(g("noise_schedule.alphas_cumprod"))            # [T]
        p["sqrt_acp"] = _to_mx(g("noise_schedule.sqrt_alphas_cumprod"))  # [T]
        p["sqrt_om"] = _to_mx(g("noise_schedule.sqrt_one_minus_alphas_cumprod"))  # [T]

        # latent projector decoder (decode_latent) + encoder (encode, for prompts).
        # Only present when latent_diffusion=True; otherwise encode/decode are the
        # identity (modulo the latent_scale) and these keys do not exist.
        if self.latent_diffusion:
            p["dec0_w"], p["dec0_b"] = _to_mx(g("latent_projector.decoder.0.weight")), _to_mx(g("latent_projector.decoder.0.bias"))
            p["dec3_w"], p["dec3_b"] = _to_mx(g("latent_projector.decoder.3.weight")), _to_mx(g("latent_projector.decoder.3.bias"))
            p["enc0_w"], p["enc0_b"] = _to_mx(g("latent_projector.encoder.0.weight")), _to_mx(g("latent_projector.encoder.0.bias"))
            p["enc3_w"], p["enc3_b"] = _to_mx(g("latent_projector.encoder.3.weight")), _to_mx(g("latent_projector.encoder.3.bias"))

            # latent_norm (v2): LayerNorm on the encoder output before *latent_scale
            if self.latent_norm:
                p["latent_norm_w"] = _to_mx(g("latent_projector.latent_norm.weight"))
                p["latent_norm_b"] = _to_mx(g("latent_projector.latent_norm.bias"))

        # prompt encoder (for conditional sampling)
        p["pe0_w"], p["pe0_b"] = _to_mx(g("prompt_encoder.mlp.0.weight")), _to_mx(g("prompt_encoder.mlp.0.bias"))
        p["pe3_w"], p["pe3_b"] = _to_mx(g("prompt_encoder.mlp.3.weight")), _to_mx(g("prompt_encoder.mlp.3.bias"))

        # cond_projector: maps pooled [B, d_prompt] -> [B, d_latent] when they differ
        if self.has_cond_projector:
            p["cp_w"] = _to_mx(g("cond_projector.weight"))
            p["cp_b"] = _to_mx(g("cond_projector.bias"))

        # self-conditioning (v2): fuse [x_t ; x0_hat_prev] -> d_latent
        if self.self_conditioning:
            p["sc_w"] = _to_mx(g("self_cond_proj.weight"))
            p["sc_b"] = _to_mx(g("self_cond_proj.bias"))

        # per-layer conditioning + block norms + mixers
        ctype = self.conditioning_type
        if ctype == "film":
            p["film_g_w"], p["film_g_b"] = [], []
            p["film_b_w"], p["film_b_b"] = [], []
        elif ctype == "adaln":
            p["adaln_w"], p["adaln_b"] = [], []

        p["ln_w"], p["ln_b"] = [], []

        # Per-block FFN sub-layer (block_ffn): norm2 + (swiglu | mlp) weights.
        if self.block_ffn:
            p["ln2_w"], p["ln2_b"] = [], []
            if self.ffn_type == "swiglu":
                p["ffn_gate_w"], p["ffn_up_w"], p["ffn_down_w"] = [], [], []
            else:  # mlp
                p["ffn_ff1_w"], p["ffn_ff1_b"] = [], []
                p["ffn_ff2_w"], p["ffn_ff2_b"] = [], []

        for i in range(self.num_layers):
            c = f"denoiser.conditioning.{i}."
            if ctype == "film":
                p["film_g_w"].append(_to_mx(g(c + "gamma_proj.weight")))
                p["film_g_b"].append(_to_mx(g(c + "gamma_proj.bias")))
                p["film_b_w"].append(_to_mx(g(c + "beta_proj.weight")))
                p["film_b_b"].append(_to_mx(g(c + "beta_proj.bias")))
            elif ctype == "adaln":
                # AdaLNZeroConditioning: SiLU -> Linear(cond_dim, n_params*d_latent).
                # n_params is 3 (mixer only) or 6 (mixer + FFN, block_ffn=True).
                # state dict key: denoiser.conditioning.{i}.modulation.1.weight/bias
                p["adaln_w"].append(_to_mx(g(c + "modulation.1.weight")))
                p["adaln_b"].append(_to_mx(g(c + "modulation.1.bias")))

            b = f"denoiser.blocks.{i}."
            p["ln_w"].append(_to_mx(g(b + "norm.weight")))
            # RMSNorm has no bias; store None so forward uses _rms_norm
            bias_key = b + "norm.bias"
            p["ln_b"].append(_to_mx(sd[bias_key]) if bias_key in sd else None)

            if self.block_ffn:
                # norm2 (RMSNorm, no bias) + FFN weights (all bias-free per _BlockFFN).
                p["ln2_w"].append(_to_mx(g(b + "norm2.weight")))
                n2bias = b + "norm2.bias"
                p["ln2_b"].append(_to_mx(sd[n2bias]) if n2bias in sd else None)
                if self.ffn_type == "swiglu":
                    p["ffn_gate_w"].append(_to_mx(g(b + "ffn.gate_proj.weight")))
                    p["ffn_up_w"].append(_to_mx(g(b + "ffn.up_proj.weight")))
                    p["ffn_down_w"].append(_to_mx(g(b + "ffn.down_proj.weight")))
                else:  # mlp: ff2(gelu(ff1(x))), both Linear with bias
                    p["ffn_ff1_w"].append(_to_mx(g(b + "ffn.ff1.weight")))
                    p["ffn_ff1_b"].append(_to_mx(g(b + "ffn.ff1.bias")))
                    p["ffn_ff2_w"].append(_to_mx(g(b + "ffn.ff2.weight")))
                    p["ffn_ff2_b"].append(_to_mx(g(b + "ffn.ff2.bias")))

            fwd = {k[len(b + "mamba_fwd."):]: v for k, v in sd.items() if k.startswith(b + "mamba_fwd.")}
            bwd = {k[len(b + "mamba_bwd."):]: v for k, v in sd.items() if k.startswith(b + "mamba_bwd.")}
            load_torch_mamba2_state_dict(self.mixers_fwd[i], fwd)
            load_torch_mamba2_state_dict(self.mixers_bwd[i], bwd)

        # head_norm (v2): DenoisingHead LayerNorm + logit_scale
        if self.use_head_norm:
            p["head_norm_w"] = _to_mx(g("output_head.norm.weight"))
            p["head_norm_b"] = _to_mx(g("output_head.norm.bias"))
            p["logit_scale"] = _to_mx(g("output_head.logit_scale"))  # scalar

        # head_type=="attn": load _AttnBlock weights
        if self.head_type == "attn":
            p["attn_blocks"] = []
            for i in range(self.head_attn_layers):
                pfx = f"output_head.attn_blocks.{i}."
                blk = {}
                blk["norm1_w"] = _to_mx(g(pfx + "norm1.weight"))
                blk["norm1_b"] = _to_mx(g(pfx + "norm1.bias"))
                blk["q_w"] = _to_mx(g(pfx + "q_proj.weight"))   # [d_model, d_model]
                blk["k_w"] = _to_mx(g(pfx + "k_proj.weight"))
                blk["v_w"] = _to_mx(g(pfx + "v_proj.weight"))
                blk["o_w"] = _to_mx(g(pfx + "o_proj.weight"))
                blk["norm2_w"] = _to_mx(g(pfx + "norm2.weight"))
                blk["norm2_b"] = _to_mx(g(pfx + "norm2.bias"))
                blk["ff1_w"] = _to_mx(g(pfx + "ff1.weight"))
                blk["ff1_b"] = _to_mx(g(pfx + "ff1.bias"))
                blk["ff2_w"] = _to_mx(g(pfx + "ff2.weight"))
                blk["ff2_b"] = _to_mx(g(pfx + "ff2.bias"))
                p["attn_blocks"].append(blk)
            # nhead for the attn blocks (read from torch model)
            p["attn_nhead"] = torch_model.output_head.attn_blocks[0].nhead

        # output projection (non-weight-tied only)
        if not self.use_weight_tying:
            p["proj_w"] = _to_mx(g("output_head.projection.weight"))
            p["proj_b"] = _to_mx(g("output_head.projection.bias"))

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

    def _denoiser(self, x_t, cond, t_idx, x_self_cond=None):
        """Denoiser raw output. x_t:[B,L,d_latent], cond:[B,1,d_latent] -> [B,L,d_latent]."""
        p = self.p

        # self-conditioning fuse: concat [x_t, x0_hat_prev or zeros] -> d_latent
        if self.self_conditioning:
            sc = x_self_cond if x_self_cond is not None else mx.zeros_like(x_t)
            denoiser_in = _linear(
                mx.concatenate([x_t, sc], axis=-1), p["sc_w"], p["sc_b"]
            )
        else:
            denoiser_in = x_t

        t_emb = self._timestep_emb(t_idx)                        # [512]
        time_cond = _linear(t_emb, p["time_proj_w"], p["time_proj_b"]).reshape(1, 1, -1)
        combined = cond + time_cond                               # [B,1,cond_dim]

        x = denoiser_in
        ctype = self.conditioning_type

        for i in range(self.num_layers):
            if ctype == "film":
                # FiLM: gamma * x + beta  (before layernorm)
                gamma = _linear(combined, p["film_g_w"][i], p["film_g_b"][i])
                beta  = _linear(combined, p["film_b_w"][i], p["film_b_b"][i])
                conditioned = gamma * x + beta
                h = _norm(conditioned, p["ln_w"][i], p["ln_b"][i])
                y = self.mixers_fwd[i](h)
                y = y + _flip_seq(self.mixers_bwd[i](_flip_seq(h)))
                x = conditioned + y
            elif ctype == "adaln":
                # AdaLN-Zero: SiLU(combined) -> Linear -> (scale, shift, gate[, scale2, shift2, gate2]).
                # Mirrors Mamba2Block.forward_adaln: norm->modulate->mix->gated residual,
                # then (block_ffn) norm2->modulate2->FFN->gated residual.
                silu_cond = mlx_nn.silu(combined)
                mod = _linear(silu_cond, p["adaln_w"][i], p["adaln_b"][i])  # [B,1,n*d_latent]
                d = self.d_latent
                scale = mod[..., :d]
                shift = mod[..., d:2*d]
                gate  = mod[..., 2*d:3*d]
                h = _norm(x, p["ln_w"][i], p["ln_b"][i])
                h = h * (1.0 + scale) + shift
                y = self.mixers_fwd[i](h)
                y = y + _flip_seq(self.mixers_bwd[i](_flip_seq(h)))
                x = x + gate * y
                if self.block_ffn:
                    scale2 = mod[..., 3*d:4*d]
                    shift2 = mod[..., 4*d:5*d]
                    gate2  = mod[..., 5*d:6*d]
                    h2 = _norm(x, p["ln2_w"][i], p["ln2_b"][i])
                    h2 = h2 * (1.0 + scale2) + shift2
                    x = x + gate2 * self._block_ffn(h2, i)

        return x

    def _block_ffn(self, h, i):
        """Per-block FFN sub-layer (mirrors _BlockFFN.forward).

        swiglu: down(silu(gate(x)) * up(x))   (all bias-free Linears).
        mlp:    ff2(gelu(ff1(x)))              (Linear with bias).
        """
        p = self.p
        if self.ffn_type == "swiglu":
            gated = mlx_nn.silu(_linear(h, p["ffn_gate_w"][i])) * _linear(h, p["ffn_up_w"][i])
            return _linear(gated, p["ffn_down_w"][i])
        ff = mlx_nn.gelu(_linear(h, p["ffn_ff1_w"][i], p["ffn_ff1_b"][i]))
        return _linear(ff, p["ffn_ff2_w"][i], p["ffn_ff2_b"][i])

    def _raw_to_x0(self, x_t, raw, t_idx):
        """Convert raw denoiser output to x0 estimate depending on prediction_type."""
        if self.prediction_type == "v":
            # x0 = sqrt(acp_t) * x_t - sqrt(1-acp_t) * v
            sa = self.p["sqrt_acp"][t_idx]
            so = self.p["sqrt_om"][t_idx]
            return sa * x_t - so * raw
        return raw  # x0 prediction: raw == x0

    def _attn_block(self, x, blk):
        """Mirror _AttnBlock.forward in MLX (bidirectional self-attention + FFN)."""
        B, L, D = x.shape
        nhead = self.p["attn_nhead"]
        head_dim = D // nhead
        scale = head_dim ** -0.5

        # --- self-attention branch ---
        h = _layer_norm(x, blk["norm1_w"], blk["norm1_b"])
        # Linear projections (no bias for q/k/v/o)
        q = _linear(h, blk["q_w"])                               # [B, L, D]
        k = _linear(h, blk["k_w"])
        v = _linear(h, blk["v_w"])
        # Reshape to [B, L, nhead, head_dim]
        q = q.reshape(B, L, nhead, head_dim)
        k = k.reshape(B, L, nhead, head_dim)
        v = v.reshape(B, L, nhead, head_dim)
        # Transpose to [B, nhead, L, head_dim]
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        # Scaled dot-product attention (no mask -> bidirectional)
        attn_w = mx.matmul(q, k.transpose(0, 1, 3, 2)) * scale   # [B, nhead, L, L]
        attn_w = mx.softmax(attn_w, axis=-1)
        a = mx.matmul(attn_w, v)                                  # [B, nhead, L, head_dim]
        a = a.transpose(0, 2, 1, 3).reshape(B, L, D)             # [B, L, D]
        x = x + _linear(a, blk["o_w"])

        # --- feed-forward branch ---
        h = _layer_norm(x, blk["norm2_w"], blk["norm2_b"])
        ff = mlx_nn.gelu(_linear(h, blk["ff1_w"], blk["ff1_b"]))
        x = x + _linear(ff, blk["ff2_w"], blk["ff2_b"])
        return x

    def _decode_logits(self, z):
        """decode_latent -> optional attn-head -> optional head-norm -> output head.

        z:[B,L,d_latent] -> [B,L,V]
        """
        p = self.p
        h = z / self.latent_scale
        # decode_latent: identity when there is no latent projector (latent_diffusion=False).
        if self.latent_diffusion:
            h = _linear(h, p["dec0_w"], p["dec0_b"])
            h = mlx_nn.gelu(h)
            h = _linear(h, p["dec3_w"], p["dec3_b"])            # [B,L,d_model]

        # Attention rounding head (v2, head_type=="attn")
        if self.head_type == "attn":
            for blk in p["attn_blocks"]:
                h = self._attn_block(h, blk)

        # Head norm + logit scale (v2, use_head_norm==True)
        if self.use_head_norm:
            h = _layer_norm(h, p["head_norm_w"], p["head_norm_b"])

        if self.use_weight_tying:
            logits = mx.matmul(h, p["token_embed_w"].T)          # [B,L,V]
        else:
            logits = _linear(h, p["proj_w"], p["proj_b"])

        if self.use_head_norm:
            logits = logits * mx.exp(p["logit_scale"])

        return logits

    def _encode_latent(self, emb):
        """Encode token embeddings -> scaled latent. emb:[B,P,d_model] -> [B,P,d_latent]."""
        p = self.p
        # encode_latent: identity (modulo scale) when there is no latent projector.
        if not self.latent_diffusion:
            return self.latent_scale * emb
        h = _linear(emb, p["enc0_w"], p["enc0_b"])
        h = mlx_nn.gelu(h)
        h = _linear(h, p["enc3_w"], p["enc3_b"])                # [B,P,d_latent]
        if self.latent_norm:
            h = _layer_norm(h, p["latent_norm_w"], p["latent_norm_b"])
        return self.latent_scale * h

    def _encode_prompt_latent(self, prompt_ids: np.ndarray):
        """Embed + project prompt ids -> clean latent prefix. ids:[B,P] -> [B,P,d_latent]."""
        p = self.p
        emb = p["token_embed_w"][mx.array(prompt_ids.astype(np.int32))]  # [B,P,d_model]
        latent = self._encode_latent(emb)
        return latent, emb

    def _project_cond(self, pooled):
        """Project pooled conditioning [B,1,d_prompt] -> [B,1,cond_dim].

        Mirrors DIMBA.project_conditioning: applies cond_projector when d_prompt != d_latent.
        """
        if self.has_cond_projector:
            return _linear(pooled, self.p["cp_w"], self.p["cp_b"])
        return pooled

    def _pooled_cond(self, prompt_emb):
        """Mean-pooled prompt-encoder summary -> [B,1,cond_dim]."""
        p = self.p
        h = _linear(prompt_emb, p["pe0_w"], p["pe0_b"])
        h = mlx_nn.gelu(h)
        h = _linear(h, p["pe3_w"], p["pe3_b"])                  # [B,P,d_prompt]
        pooled = mx.mean(h, axis=1, keepdims=True)               # [B,1,d_prompt]
        return self._project_cond(pooled)                        # [B,1,cond_dim]

    # ------------------------------------------------------------------ sampling
    def _setup_conditioning(self, noise: np.ndarray, prompt_ids: Optional[np.ndarray]):
        """Build (cond, x_t, prompt_latent, prompt_len) shared by every sampler.

        Mirrors the prompt-prefix + pooled/null conditioning logic of
        ``sample_from_model`` / ``sample_from_model_flow`` (sampling.py).
        """
        B = noise.shape[0]
        if prompt_ids is not None:
            prompt_latent, prompt_emb = self._encode_prompt_latent(prompt_ids)
            prompt_len = prompt_latent.shape[1]
            cond = self._pooled_cond(prompt_emb)                # [B,1,cond_dim]
            x_t = mx.concatenate([prompt_latent, mx.array(noise)], axis=1)
        else:
            prompt_latent, prompt_len = None, 0
            # null_cond is d_prompt; project to cond_dim if needed (no-op when equal)
            null = self.p["null_cond"].reshape(1, 1, -1)        # [1,1,d_prompt]
            null = self._project_cond(null)                      # [1,1,cond_dim]
            cond = mx.broadcast_to(null, (B, 1, self.cond_dim))
            x_t = mx.array(noise)
        return cond, x_t, prompt_latent, prompt_len

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

        cond, x_t, prompt_latent, prompt_len = self._setup_conditioning(noise, prompt_ids)

        x_self_cond = None  # carries x0_hat across steps for self-conditioning

        for i in range(ns):
            t_val = int(timesteps[i])
            raw = self._denoiser(x_t, cond, t_val, x_self_cond)  # raw output
            z0 = self._raw_to_x0(x_t, raw, t_val)                # x0 estimate

            # carry x0 for self-conditioning
            if self.self_conditioning:
                x_self_cond = z0

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

    def _flow_t_idx(self, t_val: float) -> int:
        """Continuous t in (0,1] -> discrete timestep-embedding index (mirrors denoise_flow)."""
        T = self.num_diffusion_steps
        idx = int(round(min(max(t_val, 0.0), 1.0) * (T - 1)))
        return max(0, min(T - 1, idx))

    def sample_logits_flow(self, noise: np.ndarray, num_steps: int = 20,
                           sampler: str = "euler",
                           prompt_ids: Optional[np.ndarray] = None):
        """Flow-matching ODE sampler -> final response logits ``[B, seq_len, V]`` (mx.array).

        Integrates ``dx/dt = v_theta(x_t, t, cond)`` from t=1 (pure noise) to t=0
        (clean data) with Euler or Heun steps.  Mirrors ``sample_from_model_flow``
        (diffusion/sampling.py): the denoiser's raw output is the x0 prediction and
        the velocity is derived as ``v = (x_t - x0) / t``.

        ``noise`` is the initial response latent ``[B, seq_len, d_latent]`` (the t=1
        state).  Pass the same array used by the torch flow sampler for a faithful
        comparison.
        """
        if sampler not in ("euler", "heun"):
            raise ValueError(f"sampler must be 'euler' or 'heun', got {sampler!r}")

        cond, x_t, prompt_latent, prompt_len = self._setup_conditioning(noise, prompt_ids)

        # Uniform time grid t=1 -> t=0 (num_steps+1 points), as in sample_from_model_flow.
        ts = np.linspace(1.0, 0.0, num_steps + 1).astype(np.float64)

        def _velocity(xt, t_val, x_self_cond):
            # Flow raw output is always x0 (velocity derived outside, like denoise_flow).
            t_idx = self._flow_t_idx(t_val)
            x0_hat = self._denoiser(xt, cond, t_idx, x_self_cond)
            # v = (x_t - x0) / t   (rearranged from x_t = (1-t)*x0 + t*noise)
            v = (xt - x0_hat) / max(t_val, 1e-5)
            return v, x0_hat

        x_self_cond = None
        for i in range(num_steps):
            t_cur = float(ts[i])
            t_nxt = float(ts[i + 1])
            dt = t_nxt - t_cur                                   # negative (backward)

            v_cur, x0_hat = _velocity(x_t, t_cur, x_self_cond)
            if self.self_conditioning:
                x_self_cond = x0_hat

            if sampler == "heun" and i < num_steps - 1:
                # Predictor.
                x_mid = x_t + v_cur * dt
                if prompt_latent is not None:
                    x_mid = mx.concatenate(
                        [prompt_latent, x_mid[:, prompt_len:, :]], axis=1)
                v_nxt, _ = _velocity(x_mid, t_nxt, x_self_cond)
                # Corrector: average the two velocities.
                x_t = x_t + 0.5 * (v_cur + v_nxt) * dt
            else:
                x_t = x_t + v_cur * dt

            if prompt_latent is not None:                        # hold prefix clean
                x_t = mx.concatenate(
                    [prompt_latent, x_t[:, prompt_len:, :]], axis=1)
            mx.eval(x_t)

        response = x_t[:, prompt_len:, :]
        logits = self._decode_logits(response)
        mx.eval(logits)
        return logits

    def sample(self, seq_len: int, num_steps: Optional[int] = None, temperature: float = 1.0,
               prompt_ids: Optional[np.ndarray] = None, noise: Optional[np.ndarray] = None,
               seed: int = 0, sampler: Optional[str] = None) -> np.ndarray:
        """Generate token ids ``[B, seq_len]`` (greedy by default; set temperature for sampling).

        Auto-selects the flow-matching Euler/Heun ODE sampler when the model was
        trained with ``use_flow_matching=True`` (``sampler`` then chooses euler/heun,
        default euler); otherwise runs the DDIM path.  ``num_steps`` is the number of
        DDIM / Euler steps.
        """
        if noise is None:
            rng = np.random.default_rng(seed)
            B = 1 if prompt_ids is None else prompt_ids.shape[0]
            noise = rng.standard_normal((B, seq_len, self.d_latent)).astype(np.float32)
        if self.use_flow_matching:
            ns = num_steps if num_steps is not None else 20
            logits = np.array(self.sample_logits_flow(
                noise, num_steps=ns, sampler=(sampler or "euler"), prompt_ids=prompt_ids))
        else:
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
