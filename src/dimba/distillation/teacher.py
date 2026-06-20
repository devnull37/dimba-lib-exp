"""TeacherWrapper: wraps a pretrained HuggingFace Transformer for cross-architecture KD.

Provides a unified interface over multiple Transformer families (Llama/Qwen/Mistral,
GPT-2, GPT-NeoX/Pythia, BERT/RoBERTa) so that DIMBA's distillation pipeline can
query hidden states, attention matrices, FFN weights, and vocabulary projections
without caring about the underlying architecture.

All forward passes are wrapped with ``@torch.no_grad()`` so the teacher never
accumulates gradients. Weight accessors return tensors in ``nn.Linear`` convention
(i.e. ``weight[out_features, in_features]``) with GPT-2 Conv1D weights transposed.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------


@dataclass
class TeacherOutputs:
    """Structured outputs from a teacher forward pass.

    Attributes:
        hidden_states: Tuple of length ``num_layers + 1``; index 0 is the token
            embedding output, index ``k`` is the output of teacher layer ``k-1``.
            Each tensor has shape ``[B, L, d_teacher]``.
        attentions: Tuple of length ``num_layers``; element ``k`` is the attention
            weight matrix of teacher layer ``k``, shape ``[B, num_heads, L, L]``.
        logits: Vocabulary logits ``[B, L, vocab_size]``, or ``None`` when the
            teacher does not produce them (e.g. BERT with no LM head).
    """

    hidden_states: Tuple
    attentions: Tuple
    logits: Optional[torch.Tensor]


# ---------------------------------------------------------------------------
# Architecture-family helpers
# ---------------------------------------------------------------------------


def _is_conv1d(module: nn.Module) -> bool:
    """Return True if *module* is a GPT-2-style Conv1D (not nn.Conv1d)."""
    return type(module).__name__ == "Conv1D"


def _conv1d_weight(module: nn.Module) -> torch.Tensor:
    """Extract weight from Conv1D as ``[out, in]`` (transpose of Conv1D.weight)."""
    # transformers Conv1D stores weight as [in_features, out_features]
    return module.weight.t().contiguous()


def _get_attr_path(obj: object, *attrs: str) -> Optional[object]:
    """Walk a dotted attribute path, returning None if any step is missing."""
    cur = obj
    for attr in attrs:
        cur = getattr(cur, attr, None)
        if cur is None:
            return None
    return cur


def _resolve_decoder_layers(model: nn.Module) -> Optional[List[nn.Module]]:
    """Return the list of decoder/encoder blocks for known HF architectures.

    Tries common attribute paths in order. Returns ``None`` when nothing matches.
    """
    # Llama/Qwen/Mistral/Gemma: model.model.layers
    layers = _get_attr_path(model, "model", "layers")
    if layers is not None:
        return list(layers)

    # GPT-2: model.transformer.h
    layers = _get_attr_path(model, "transformer", "h")
    if layers is not None:
        return list(layers)

    # GPT-NeoX / Pythia: model.gpt_neox.layers
    layers = _get_attr_path(model, "gpt_neox", "layers")
    if layers is not None:
        return list(layers)

    # BERT / RoBERTa (encoder-only): model.encoder.layer
    layers = _get_attr_path(model, "encoder", "layer")
    if layers is not None:
        return list(layers)

    # BertModel nested inside a larger model (e.g. BertForMaskedLM)
    layers = _get_attr_path(model, "bert", "encoder", "layer")
    if layers is not None:
        return list(layers)

    # RobertaModel nested
    layers = _get_attr_path(model, "roberta", "encoder", "layer")
    if layers is not None:
        return list(layers)

    return None


def _detect_ffn_type_and_hidden(
    layer: nn.Module,
) -> Tuple[str, int]:
    """Detect FFN type and intermediate hidden size from a single decoder layer.

    Returns:
        (ffn_type, hidden_size) where ffn_type is ``'swiglu'`` or ``'mlp'``.

    Raises:
        ValueError: If the layer's FFN structure is not recognised.
    """
    # --- Llama / Qwen / Mistral / Gemma (SwiGLU) ---
    # layer.mlp has gate_proj, up_proj, down_proj
    mlp = getattr(layer, "mlp", None)
    if mlp is not None:
        if (
            hasattr(mlp, "gate_proj")
            and hasattr(mlp, "up_proj")
            and hasattr(mlp, "down_proj")
        ):
            hidden = mlp.gate_proj.out_features
            return "swiglu", hidden

        # GPT-NeoX / Pythia: mlp.dense_h_to_4h / dense_4h_to_h
        if hasattr(mlp, "dense_h_to_4h") and hasattr(mlp, "dense_4h_to_h"):
            proj = mlp.dense_h_to_4h
            if _is_conv1d(proj):
                hidden = proj.weight.shape[1]  # Conv1D: [in, out]
            else:
                hidden = proj.out_features
            return "mlp", hidden

        # OPT / Falcon: mlp.fc1/fc2
        if hasattr(mlp, "fc1") and hasattr(mlp, "fc2"):
            hidden = mlp.fc1.out_features
            return "mlp", hidden

    # --- GPT-2: layer.mlp.c_fc / c_proj (Conv1D) ---
    gpt2_mlp = getattr(layer, "mlp", None)
    if gpt2_mlp is not None and hasattr(gpt2_mlp, "c_fc") and hasattr(gpt2_mlp, "c_proj"):
        c_fc = gpt2_mlp.c_fc
        if _is_conv1d(c_fc):
            hidden = c_fc.weight.shape[1]  # Conv1D weight: [in, out]
        else:
            hidden = c_fc.out_features
        return "mlp", hidden

    # --- BERT / RoBERTa: layer.intermediate.dense + layer.output.dense ---
    intermediate = getattr(layer, "intermediate", None)
    if intermediate is not None and hasattr(intermediate, "dense"):
        dense = intermediate.dense
        hidden = dense.out_features
        return "mlp", hidden

    raise ValueError(
        f"Cannot detect FFN type from layer {type(layer).__name__!r}. "
        "Supported families: Llama/Qwen/Mistral (swiglu), GPT-2 (c_fc/c_proj), "
        "GPT-NeoX/Pythia (dense_h_to_4h), BERT/RoBERTa (intermediate.dense)."
    )


# ---------------------------------------------------------------------------
# TeacherWrapper
# ---------------------------------------------------------------------------


class TeacherWrapper(nn.Module):
    """Wraps a pretrained HuggingFace model as a DIMBA distillation teacher.

    Handles output collection (hidden states + attention matrices), weight
    extraction in a normalised form, and architecture detection across four
    LLM families: Llama/Qwen/Mistral, GPT-2, GPT-NeoX/Pythia, BERT/RoBERTa.

    Args:
        model_id_or_path: HuggingFace model identifier or local path.
        teacher_type: ``'causal'`` (decoder-only) or ``'masked'`` (encoder-only
            / bidirectional, e.g. BERT).  Affects how stage-1 attention targets
            are built.
        device: Device string (``'cpu'``, ``'cuda'``, etc.).
        dtype: Dtype for the teacher weights.
        trust_remote_code: Pass ``trust_remote_code=True`` to ``from_pretrained``.
    """

    def __init__(
        self,
        model_id_or_path: str,
        teacher_type: str = "causal",
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        trust_remote_code: bool = False,
    ) -> None:
        super().__init__()
        self._teacher_type = teacher_type
        self._device_str = device
        self._dtype = dtype

        from transformers import AutoModel, AutoModelForCausalLM, AutoModelForMaskedLM

        load_kwargs: Dict = dict(
            output_hidden_states=True,
            output_attentions=True,
            trust_remote_code=trust_remote_code,
            torch_dtype=dtype,
            attn_implementation="eager",
        )

        def _load_model(auto_cls, fallback_cls, path, kwargs):
            """Try loading with eager attn; retry without it for custom models."""
            try:
                return auto_cls.from_pretrained(path, **kwargs)
            except Exception:
                pass
            # Some custom/remote-code models reject attn_implementation; retry without.
            kwargs_no_eager = {k: v for k, v in kwargs.items() if k != "attn_implementation"}
            try:
                return auto_cls.from_pretrained(path, **kwargs_no_eager)
            except Exception:
                return fallback_cls.from_pretrained(path, **kwargs_no_eager)

        if teacher_type == "causal":
            self._hf_model = _load_model(
                AutoModelForCausalLM, AutoModel, model_id_or_path, load_kwargs
            )
        else:
            self._hf_model = _load_model(
                AutoModelForMaskedLM, AutoModel, model_id_or_path, load_kwargs
            )

        self._hf_model.to(device=device, dtype=dtype)
        self._hf_model.eval()

        # Cache decoder layers for weight accessors.
        self._layers: Optional[List[nn.Module]] = _resolve_decoder_layers(self._hf_model)

        # Detect FFN type from the first layer (if layers are available).
        self._ffn_type: str = "mlp"
        self._ffn_hidden: int = 0
        if self._layers:
            try:
                self._ffn_type, self._ffn_hidden = _detect_ffn_type_and_hidden(self._layers[0])
            except ValueError as exc:
                warnings.warn(
                    f"TeacherWrapper: could not auto-detect FFN type: {exc}. "
                    "Defaulting to 'mlp'.",
                    RuntimeWarning,
                )
                cfg = self._hf_config
                self._ffn_hidden = getattr(cfg, "intermediate_size", None) or getattr(
                    cfg, "ffn_dim", None
                ) or (self.d_model * 4)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def _hf_config(self):
        return self._hf_model.config

    @property
    def num_layers(self) -> int:
        """Number of transformer layers (attention + FFN blocks)."""
        cfg = self._hf_config
        for attr in ("num_hidden_layers", "n_layer", "num_layers"):
            val = getattr(cfg, attr, None)
            if val is not None:
                return int(val)
        if self._layers is not None:
            return len(self._layers)
        raise ValueError("Cannot determine num_layers from model config.")

    @property
    def num_heads(self) -> int:
        """Number of attention heads."""
        cfg = self._hf_config
        for attr in ("num_attention_heads", "n_head", "num_heads"):
            val = getattr(cfg, attr, None)
            if val is not None:
                return int(val)
        raise ValueError("Cannot determine num_heads from model config.")

    @property
    def d_model(self) -> int:
        """Hidden dimension of the teacher."""
        cfg = self._hf_config
        for attr in ("hidden_size", "d_model", "n_embd"):
            val = getattr(cfg, attr, None)
            if val is not None:
                return int(val)
        raise ValueError("Cannot determine d_model from model config.")

    @property
    def vocab_size(self) -> int:
        """Vocabulary size."""
        cfg = self._hf_config
        for attr in ("vocab_size",):
            val = getattr(cfg, attr, None)
            if val is not None:
                return int(val)
        raise ValueError("Cannot determine vocab_size from model config.")

    @property
    def is_causal(self) -> bool:
        """Whether this teacher is a causal (decoder-only) language model."""
        return self._teacher_type == "causal"

    @property
    def ffn_type(self) -> str:
        """FFN type: ``'swiglu'`` or ``'mlp'``."""
        return self._ffn_type

    @property
    def ffn_hidden(self) -> int:
        """Teacher MLP intermediate size."""
        return self._ffn_hidden

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def unload(self) -> None:
        """Move teacher weights to CPU and free GPU memory.

        The wrapper object remains alive so metadata properties
        (num_layers, d_model, num_heads, …) continue to work — they
        read from self._hf_model.config which is a Python object.
        """
        if next(self._hf_model.parameters(), None) is not None:
            self._hf_model.to("cpu")
        import gc
        gc.collect()
        try:
            import torch as _torch
            _torch.cuda.empty_cache()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> TeacherOutputs:
        """Run a no-grad forward pass and collect hidden states + attentions.

        Args:
            input_ids: Token id tensor ``[B, L]``.
            attention_mask: Optional padding mask ``[B, L]``.

        Returns:
            :class:`TeacherOutputs` with hidden_states (length num_layers+1),
            attentions (length num_layers), and logits if available.
        """
        input_ids = input_ids.to(self._hf_model.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._hf_model.device)

        kwargs: Dict = dict(
            input_ids=input_ids,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True,
        )
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask

        outputs = self._hf_model(**kwargs)

        hidden_states: Tuple = outputs.hidden_states  # tuple length L+1
        attentions = outputs.attentions  # tuple length L, or None when backend returns nothing

        logits: Optional[torch.Tensor] = getattr(outputs, "logits", None)

        # Guard against the whole attentions field being None (SDPA/Flash backend even
        # with output_attentions=True on some recent transformers versions).
        B, L = input_ids.shape
        if attentions is None:
            attentions = (None,) * self.num_layers

        # Some models return None attentions per layer (e.g. Flash-Attention variants).
        # Replace None entries with zero tensors of the expected shape.
        cleaned_attentions: List[torch.Tensor] = []
        for attn in attentions:
            if attn is None:
                nh = self.num_heads
                cleaned_attentions.append(
                    torch.zeros(B, nh, L, L, dtype=self._dtype, device=input_ids.device)
                )
            else:
                cleaned_attentions.append(attn)

        return TeacherOutputs(
            hidden_states=hidden_states,
            attentions=tuple(cleaned_attentions),
            logits=logits,
        )

    # ------------------------------------------------------------------
    # Weight accessors
    # ------------------------------------------------------------------

    def _require_layers(self, context: str) -> List[nn.Module]:
        if self._layers is None:
            raise ValueError(
                f"TeacherWrapper.{context}: cannot resolve decoder layers for "
                f"model type {type(self._hf_model).__name__!r}."
            )
        return self._layers

    def input_embedding_weight(self) -> torch.Tensor:
        """Return the input token embedding matrix ``[vocab_size, d_model]``.

        Raises:
            ValueError: If the embedding table cannot be found.
        """
        # Llama / Qwen / Mistral: model.model.embed_tokens
        m = _get_attr_path(self._hf_model, "model", "embed_tokens")
        if m is not None and hasattr(m, "weight"):
            return m.weight.detach()

        # GPT-2: model.transformer.wte
        m = _get_attr_path(self._hf_model, "transformer", "wte")
        if m is not None and hasattr(m, "weight"):
            return m.weight.detach()

        # GPT-NeoX / Pythia: model.gpt_neox.embed_in
        m = _get_attr_path(self._hf_model, "gpt_neox", "embed_in")
        if m is not None and hasattr(m, "weight"):
            return m.weight.detach()

        # BERT / RoBERTa: model.embeddings.word_embeddings
        m = _get_attr_path(self._hf_model, "embeddings", "word_embeddings")
        if m is not None and hasattr(m, "weight"):
            return m.weight.detach()

        # nested bert/roberta
        for prefix in ("bert", "roberta"):
            m = _get_attr_path(self._hf_model, prefix, "embeddings", "word_embeddings")
            if m is not None and hasattr(m, "weight"):
                return m.weight.detach()

        # Generic fallback: look for an Embedding with the right vocab_size
        for module in self._hf_model.modules():
            if isinstance(module, nn.Embedding) and module.num_embeddings == self.vocab_size:
                return module.weight.detach()

        raise ValueError(
            f"Cannot locate input embedding weight for {type(self._hf_model).__name__!r}."
        )

    def output_head_weight(self) -> Optional[torch.Tensor]:
        """Return the output LM head weight ``[vocab_size, d_model]``, or None.

        Returns ``None`` when the model has no LM head (e.g. base BERT without MLM).
        """
        # CausalLM head: lm_head (nn.Linear or tied to embedding)
        lm_head = getattr(self._hf_model, "lm_head", None)
        if lm_head is not None and hasattr(lm_head, "weight"):
            return lm_head.weight.detach()

        # BERT MLM: cls.predictions.decoder
        m = _get_attr_path(self._hf_model, "cls", "predictions", "decoder")
        if m is not None and hasattr(m, "weight"):
            return m.weight.detach()

        # Roberta MLM: lm_head.decoder
        m = _get_attr_path(self._hf_model, "lm_head", "decoder")
        if m is not None and hasattr(m, "weight"):
            return m.weight.detach()

        return None

    def final_norm_state(self) -> Optional[Dict[str, torch.Tensor]]:
        """Return the state_dict of the final layer norm, or None if not found.

        Returns:
            Mapping of parameter name to tensor, or ``None``.
        """
        # Llama / Mistral: model.model.norm
        m = _get_attr_path(self._hf_model, "model", "norm")
        if m is not None and isinstance(m, nn.Module):
            return {k: v.detach() for k, v in m.state_dict().items()}

        # GPT-2: model.transformer.ln_f
        m = _get_attr_path(self._hf_model, "transformer", "ln_f")
        if m is not None and isinstance(m, nn.Module):
            return {k: v.detach() for k, v in m.state_dict().items()}

        # GPT-NeoX / Pythia: model.gpt_neox.final_layer_norm
        m = _get_attr_path(self._hf_model, "gpt_neox", "final_layer_norm")
        if m is not None and isinstance(m, nn.Module):
            return {k: v.detach() for k, v in m.state_dict().items()}

        # BERT: model.pooler or encoder final norm doesn't really apply, skip
        return None

    def layer_ffn_state(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return a normalised state-dict for the FFN sub-layer at layer *idx*.

        Weights are returned in ``nn.Linear`` convention (``[out, in]``).  GPT-2
        Conv1D weights are transposed.  Keys match :class:`_BlockFFN`:

        * For ``'mlp'``:   ``ff1.weight``, ``ff1.bias``, ``ff2.weight``, ``ff2.bias``.
        * For ``'swiglu'``: ``gate_proj.weight``, ``up_proj.weight``, ``down_proj.weight``.

        Args:
            idx: Zero-based layer index.

        Returns:
            Dict of parameter name to detached tensor.

        Raises:
            ValueError: If the architecture is not supported or ``idx`` is out of range.
        """
        layers = self._require_layers("layer_ffn_state")
        if idx < 0 or idx >= len(layers):
            raise ValueError(f"Layer index {idx} out of range [0, {len(layers)}).")
        layer = layers[idx]

        # --- Llama / Qwen / Mistral / Gemma (SwiGLU) ---
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and hasattr(mlp, "gate_proj"):
            return {
                "gate_proj.weight": mlp.gate_proj.weight.detach(),
                "up_proj.weight": mlp.up_proj.weight.detach(),
                "down_proj.weight": mlp.down_proj.weight.detach(),
            }

        # --- GPT-2: layer.mlp.c_fc / c_proj (Conv1D -> transpose) ---
        if mlp is not None and hasattr(mlp, "c_fc") and hasattr(mlp, "c_proj"):
            c_fc = mlp.c_fc
            c_proj = mlp.c_proj
            w1 = _conv1d_weight(c_fc) if _is_conv1d(c_fc) else c_fc.weight.detach()
            w2 = _conv1d_weight(c_proj) if _is_conv1d(c_proj) else c_proj.weight.detach()
            state: Dict[str, torch.Tensor] = {
                "ff1.weight": w1,
                "ff2.weight": w2,
            }
            # Conv1D bias lives on the module
            b1 = getattr(c_fc, "bias", None)
            b2 = getattr(c_proj, "bias", None)
            if b1 is not None:
                state["ff1.bias"] = b1.detach()
            if b2 is not None:
                state["ff2.bias"] = b2.detach()
            return state

        # --- GPT-NeoX / Pythia: layer.mlp.dense_h_to_4h / dense_4h_to_h ---
        if mlp is not None and hasattr(mlp, "dense_h_to_4h") and hasattr(mlp, "dense_4h_to_h"):
            proj_in = mlp.dense_h_to_4h
            proj_out = mlp.dense_4h_to_h
            if _is_conv1d(proj_in):
                w1 = _conv1d_weight(proj_in)
                w2 = _conv1d_weight(proj_out)
            else:
                w1 = proj_in.weight.detach()
                w2 = proj_out.weight.detach()
            state = {"ff1.weight": w1, "ff2.weight": w2}
            b1 = getattr(proj_in, "bias", None)
            b2 = getattr(proj_out, "bias", None)
            if b1 is not None:
                state["ff1.bias"] = b1.detach()
            if b2 is not None:
                state["ff2.bias"] = b2.detach()
            return state

        # --- OPT / Falcon: mlp.fc1/fc2 ---
        if mlp is not None and hasattr(mlp, "fc1") and hasattr(mlp, "fc2"):
            state = {
                "ff1.weight": mlp.fc1.weight.detach(),
                "ff2.weight": mlp.fc2.weight.detach(),
            }
            if mlp.fc1.bias is not None:
                state["ff1.bias"] = mlp.fc1.bias.detach()
            if mlp.fc2.bias is not None:
                state["ff2.bias"] = mlp.fc2.bias.detach()
            return state

        # --- BERT / RoBERTa: layer.intermediate.dense + layer.output.dense ---
        intermediate = getattr(layer, "intermediate", None)
        bert_output = getattr(layer, "output", None)
        if intermediate is not None and hasattr(intermediate, "dense"):
            state = {
                "ff1.weight": intermediate.dense.weight.detach(),
                "ff2.weight": bert_output.dense.weight.detach(),
            }
            if intermediate.dense.bias is not None:
                state["ff1.bias"] = intermediate.dense.bias.detach()
            if bert_output.dense.bias is not None:
                state["ff2.bias"] = bert_output.dense.bias.detach()
            return state

        raise ValueError(
            f"Cannot extract FFN state from layer {idx} of "
            f"{type(self._hf_model).__name__!r}. "
            "Supported: Llama/Qwen/Mistral (gate_proj/up_proj/down_proj), "
            "GPT-2 (c_fc/c_proj Conv1D), GPT-NeoX/Pythia (dense_h_to_4h), "
            "BERT/RoBERTa (intermediate.dense + output.dense)."
        )

    def layer_attention_qkvo(self, idx: int) -> Dict[str, Optional[torch.Tensor]]:
        """Return Q/K/V/O projection weights for the self-attention at layer *idx*.

        Returned dict has keys ``'q'``, ``'k'``, ``'v'``, ``'o'``; values are
        ``[d_model, d_model]`` tensors or ``None`` when the weight cannot be
        separated or found.  GPT-2 fused ``c_attn`` Conv1D is split into Q/K/V.

        Args:
            idx: Zero-based layer index.

        Returns:
            Dict with keys ``'q'``, ``'k'``, ``'v'``, ``'o'``.
        """
        layers = self._require_layers("layer_attention_qkvo")
        if idx < 0 or idx >= len(layers):
            raise ValueError(f"Layer index {idx} out of range [0, {len(layers)}).")
        layer = layers[idx]

        result: Dict[str, Optional[torch.Tensor]] = {"q": None, "k": None, "v": None, "o": None}

        # --- Llama / Qwen / Mistral: layer.self_attn.{q,k,v,o}_proj ---
        attn = getattr(layer, "self_attn", None)
        if attn is not None:
            q_proj = getattr(attn, "q_proj", None)
            k_proj = getattr(attn, "k_proj", None)
            v_proj = getattr(attn, "v_proj", None)
            o_proj = getattr(attn, "o_proj", None)
            if q_proj is not None:
                result["q"] = q_proj.weight.detach() if hasattr(q_proj, "weight") else None
                result["k"] = k_proj.weight.detach() if (k_proj is not None and hasattr(k_proj, "weight")) else None
                result["v"] = v_proj.weight.detach() if (v_proj is not None and hasattr(v_proj, "weight")) else None
                result["o"] = o_proj.weight.detach() if (o_proj is not None and hasattr(o_proj, "weight")) else None
                return result

        # --- GPT-2: layer.attn.c_attn (fused QKV) + c_proj ---
        gpt2_attn = getattr(layer, "attn", None)
        if gpt2_attn is not None:
            c_attn = getattr(gpt2_attn, "c_attn", None)
            c_proj = getattr(gpt2_attn, "c_proj", None)
            if c_attn is not None:
                w = _conv1d_weight(c_attn) if _is_conv1d(c_attn) else c_attn.weight.detach()
                # w: [3*d_model, d_model] — split into Q, K, V
                d = w.shape[0] // 3
                result["q"] = w[:d].contiguous()
                result["k"] = w[d : 2 * d].contiguous()
                result["v"] = w[2 * d :].contiguous()
                if c_proj is not None:
                    w_o = _conv1d_weight(c_proj) if _is_conv1d(c_proj) else c_proj.weight.detach()
                    result["o"] = w_o
                return result

        # --- GPT-NeoX / Pythia: layer.attention.query_key_value + dense ---
        neox_attn = getattr(layer, "attention", None)
        if neox_attn is not None:
            qkv = getattr(neox_attn, "query_key_value", None)
            dense = getattr(neox_attn, "dense", None)
            if qkv is not None:
                w = _conv1d_weight(qkv) if _is_conv1d(qkv) else qkv.weight.detach()
                d = w.shape[0] // 3
                result["q"] = w[:d].contiguous()
                result["k"] = w[d : 2 * d].contiguous()
                result["v"] = w[2 * d :].contiguous()
                if dense is not None:
                    result["o"] = dense.weight.detach()
                return result

        # --- BERT / RoBERTa: layer.attention.self.{query,key,value} + output.dense ---
        bert_attn = getattr(layer, "attention", None)
        if bert_attn is not None:
            self_attn = getattr(bert_attn, "self", None)
            bert_out = getattr(bert_attn, "output", None)
            if self_attn is not None:
                q = getattr(self_attn, "query", None)
                k = getattr(self_attn, "key", None)
                v = getattr(self_attn, "value", None)
                result["q"] = q.weight.detach() if (q is not None and hasattr(q, "weight")) else None
                result["k"] = k.weight.detach() if (k is not None and hasattr(k, "weight")) else None
                result["v"] = v.weight.detach() if (v is not None and hasattr(v, "weight")) else None
                if bert_out is not None and hasattr(bert_out, "dense"):
                    result["o"] = bert_out.dense.weight.detach()
                return result

        # Unknown architecture — return dict with None values; caller must handle.
        warnings.warn(
            f"TeacherWrapper.layer_attention_qkvo: cannot resolve Q/K/V/O for "
            f"layer {idx} of {type(self._hf_model).__name__!r}. Returning Nones.",
            RuntimeWarning,
        )
        return result
