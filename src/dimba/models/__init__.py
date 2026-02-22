"""Models module for DIMBA."""

from .diffusion import DIMBA
from .denoiser import Mamba2Denoiser, Mamba2Block, DenoisingHead
from .embeddings import (
    TokenEmbedding,
    TimestepEmbedding,
    PromptEncoder,
    FiLMConditioning,
    AdditiveConditioning,
)
from .simple_mamba import SimpleMamba2, SimpleMamba2Block
from .vae import TokenVAE, TokenVAEWithDeterministicFallback, create_latent_projector
from .lora import (
    DEFAULT_LORA_TARGET_MODULES,
    LoRALinear,
    inject_lora_to_model,
    merge_lora_weights,
    save_lora_weights,
    load_lora_weights,
)
from .quantization import quantize_model_4bit, prepare_for_qlora

__all__ = [
    "DIMBA",
    "Mamba2Denoiser",
    "Mamba2Block",
    "DenoisingHead",
    "TokenEmbedding",
    "TimestepEmbedding",
    "PromptEncoder",
    "FiLMConditioning",
    "AdditiveConditioning",
    "SimpleMamba2",
    "SimpleMamba2Block",
    "TokenVAE",
    "TokenVAEWithDeterministicFallback",
    "create_latent_projector",
    "DEFAULT_LORA_TARGET_MODULES",
    "LoRALinear",
    "inject_lora_to_model",
    "merge_lora_weights",
    "save_lora_weights",
    "load_lora_weights",
    "quantize_model_4bit",
    "prepare_for_qlora",
]
