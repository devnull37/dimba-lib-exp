#!/usr/bin/env python3
"""Supervised fine-tuning (SFT) script for DIMBA.

Supports:
- Full finetuning
- LoRA finetuning
- Q-LoRA finetuning (via repo quantization helper when available)

Core behavior:
- Formats examples with an instruction template.
- Computes loss only on response tokens by masking prompt tokens with ignore_index.
- Saves full checkpoint and LoRA adapter weights (when LoRA/Q-LoRA is used).
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import math
import random
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from datasets import load_dataset, load_from_disk
except ImportError:
    load_dataset = None
    load_from_disk = None

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

# Add local src/ to import path
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = (SCRIPT_DIR / ".." / ".." / "src").resolve()
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dimba.models.diffusion import DIMBA
from dimba.tokenizers.simple import SimpleCharacterTokenizer


RESPONSE_SENTINEL = "<|DIMBA_RESPONSE|>"
DEFAULT_INSTRUCTION_TEMPLATE = "{instruction}\n\n{input}\n\n{response}"


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def optional_import(module_name: str) -> Optional[ModuleType]:
    """Import a module if present, otherwise return None."""
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


def call_with_supported_kwargs(fn: Callable[..., Any], **kwargs: Any) -> Any:
    """Call fn with only kwargs it accepts."""
    sig = inspect.signature(fn)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return fn(**kwargs)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(**filtered)


def filter_kwargs_for_callable(fn: Callable[..., Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Filter kwargs to keys accepted by fn."""
    sig = inspect.signature(fn)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def choose_device(device_arg: str) -> torch.device:
    """Resolve training device from CLI argument."""
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    if device.type == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available.")
    return device


def strip_prefix_if_present(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    """Strip prefix from keys when present."""
    out: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            out[key[len(prefix) :]] = value
        else:
            out[key] = value
    return out


def normalize_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Normalize common checkpoint key prefixes."""
    normalized = dict(state_dict)
    candidate_prefixes = ("module.model.", "model.", "module.")
    for prefix in candidate_prefixes:
        starts = sum(k.startswith(prefix) for k in normalized.keys())
        if starts > len(normalized) // 2:
            normalized = strip_prefix_if_present(normalized, prefix)
    return normalized


def infer_dimba_config_from_state(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """Infer DIMBA config from a model state_dict when metadata is unavailable."""
    if "token_embed.embedding.weight" not in state_dict:
        raise ValueError("Cannot infer model config: missing token embedding weights.")

    embed_weight = state_dict["token_embed.embedding.weight"]
    vocab_size = int(embed_weight.shape[0])
    d_model = int(embed_weight.shape[1])

    d_prompt = d_model
    prompt_weight_keys = [
        key
        for key in state_dict.keys()
        if key.startswith("prompt_encoder.mlp.") and key.endswith(".weight")
    ]
    if prompt_weight_keys:
        prompt_indices = [
            int(key.split(".")[2])
            for key in prompt_weight_keys
            if key.split(".")[2].isdigit()
        ]
        if prompt_indices:
            last_key = f"prompt_encoder.mlp.{max(prompt_indices)}.weight"
            d_prompt = int(state_dict[last_key].shape[0])

    num_diffusion_steps = 1000
    if "noise_schedule.alphas_cumprod" in state_dict:
        num_diffusion_steps = int(state_dict["noise_schedule.alphas_cumprod"].shape[0])

    block_ids = set()
    for key in state_dict.keys():
        if key.startswith("denoiser.blocks."):
            parts = key.split(".")
            if len(parts) > 2 and parts[2].isdigit():
                block_ids.add(int(parts[2]))
    num_denoiser_layers = max(block_ids) + 1 if block_ids else 6

    conditioning_type = "film"
    if any(".conditioning." in k and ".proj." in k for k in state_dict.keys()):
        conditioning_type = "additive"
    if any(".conditioning." in k and ".gamma_proj." in k for k in state_dict.keys()):
        conditioning_type = "film"

    latent_dim = d_model
    if "denoiser.blocks.0.norm.weight" in state_dict:
        latent_dim = int(state_dict["denoiser.blocks.0.norm.weight"].shape[0])
    elif "denoiser.blocks.0.mamba.norm.weight" in state_dict:
        latent_dim = int(state_dict["denoiser.blocks.0.mamba.norm.weight"].shape[0])

    latent_diffusion = any(k.startswith("latent_projector.") for k in state_dict.keys())
    if latent_dim != d_model:
        latent_diffusion = True

    use_simple_mamba = any(k.startswith("denoiser.blocks.0.mamba.A") for k in state_dict.keys())
    d_state = 16
    expand = 2
    if "denoiser.blocks.0.mamba.A" in state_dict:
        a = state_dict["denoiser.blocks.0.mamba.A"]
        d_state = int(a.shape[-1])
        d_inner = int(a.shape[-2])
        if latent_dim > 0:
            expand = max(1, int(round(d_inner / float(latent_dim))))

    use_weight_tying = "output_head.projection.weight" not in state_dict
    use_vae_latent = any(k.startswith("latent_projector.vae.") for k in state_dict.keys())

    model_config: Dict[str, Any] = {
        "d_model": d_model,
        "d_prompt": d_prompt,
        "num_diffusion_steps": num_diffusion_steps,
        "num_denoiser_layers": num_denoiser_layers,
        "d_state": d_state,
        "d_conv": 4,
        "expand": expand,
        "conditioning_type": conditioning_type,
        "dropout": 0.1,
        "use_weight_tying": use_weight_tying,
        "use_simple_mamba": use_simple_mamba,
        "latent_diffusion": latent_diffusion,
        "use_vae_latent": use_vae_latent,
    }
    if latent_diffusion:
        model_config["d_latent"] = latent_dim

    return {"vocab_size": vocab_size, "model_config": model_config}


def load_dimba_checkpoint(checkpoint_path: str, map_location: str = "cpu") -> Tuple[DIMBA, Dict[str, Any]]:
    """Load DIMBA from Lightning checkpoint or plain PyTorch checkpoint."""
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint_obj = torch.load(ckpt_path, map_location=map_location)
    if not isinstance(checkpoint_obj, dict):
        raise ValueError("Unsupported checkpoint format: expected a dict-like torch checkpoint.")

    state_dict: Optional[Dict[str, torch.Tensor]] = None
    model_config: Dict[str, Any] = {}
    vocab_size: Optional[int] = None

    if "state_dict" in checkpoint_obj:
        state_dict = checkpoint_obj["state_dict"]
    elif "model_state_dict" in checkpoint_obj:
        state_dict = checkpoint_obj["model_state_dict"]
    elif all(torch.is_tensor(v) for v in checkpoint_obj.values()):
        state_dict = checkpoint_obj
    else:
        raise ValueError("Could not find model state_dict in checkpoint.")

    state_dict = normalize_state_dict_keys(state_dict)

    hyper = checkpoint_obj.get("hyper_parameters", {})
    if not isinstance(hyper, dict):
        hyper = {}
    top_model_config = checkpoint_obj.get("model_config", {})
    if isinstance(top_model_config, dict):
        model_config.update(top_model_config)
    hyper_model_config = hyper.get("model_config", {})
    if isinstance(hyper_model_config, dict):
        model_config.update(hyper_model_config)

    if "vocab_size" in checkpoint_obj:
        vocab_size = int(checkpoint_obj["vocab_size"])
    if "vocab_size" in hyper:
        vocab_size = int(hyper["vocab_size"])

    if vocab_size is None or not model_config:
        inferred = infer_dimba_config_from_state(state_dict)
        if vocab_size is None:
            vocab_size = inferred["vocab_size"]
        if not model_config:
            model_config = inferred["model_config"]

    dimba_init_kwargs = filter_kwargs_for_callable(DIMBA.__init__, model_config)
    model = DIMBA(vocab_size=vocab_size, **dimba_init_kwargs)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    load_info = {
        "checkpoint_path": str(ckpt_path),
        "vocab_size": vocab_size,
        "model_config": dimba_init_kwargs,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }
    return model, load_info


def parse_template(
    template: str,
    instruction: str,
    input_text: str,
    response: str,
) -> Tuple[str, str]:
    """Build full text and prompt-only prefix from template."""
    if "{response}" not in template:
        raise ValueError("--instruction-template must include '{response}'.")

    marked = template.format(
        instruction=instruction,
        input=input_text,
        response=RESPONSE_SENTINEL,
    )
    if RESPONSE_SENTINEL not in marked:
        raise ValueError("Failed to locate response slot in instruction template.")
    prefix, suffix = marked.split(RESPONSE_SENTINEL, 1)
    full_text = prefix + response + suffix
    return full_text, prefix


def encode_text(
    tokenizer: Any,
    text: str,
    max_length: int,
    pad_token_id: int,
    pad_to_max_length: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tokenize text with either HF tokenizer API or DIMBA custom tokenizer API."""
    if callable(tokenizer):
        encoded = tokenizer(
            text,
            max_length=max_length,
            padding="max_length" if pad_to_max_length else "do_not_pad",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0).long()
        if "attention_mask" in encoded:
            attention_mask = encoded["attention_mask"].squeeze(0).long()
        else:
            attention_mask = (input_ids != pad_token_id).long()
        return input_ids, attention_mask

    if not hasattr(tokenizer, "encode"):
        raise TypeError("Tokenizer must be callable or provide encode(text).")

    token_ids = tokenizer.encode(text)
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()
    token_ids = list(token_ids)[:max_length]

    if pad_to_max_length:
        attention = [1] * len(token_ids)
        if len(token_ids) < max_length:
            token_ids = token_ids + [pad_token_id] * (max_length - len(token_ids))
            attention = attention + [0] * (max_length - len(attention))
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(attention, dtype=torch.long)

    return torch.tensor(token_ids, dtype=torch.long), torch.ones(len(token_ids), dtype=torch.long)


def as_text(value: Any) -> str:
    """Convert potentially missing fields to text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def pick_first_non_empty(example: Dict[str, Any], keys: Sequence[str]) -> str:
    """Pick first non-empty text field from example."""
    for key in keys:
        if key in example:
            value = as_text(example[key]).strip()
            if value:
                return value
    return ""


def extract_instruction_triplet(
    example: Dict[str, Any],
    instruction_column: str,
    input_column: str,
    response_column: str,
) -> Tuple[str, str, str]:
    """Extract instruction, input, response from an example."""
    instruction = pick_first_non_empty(
        example,
        [instruction_column, "instruction", "prompt", "question", "task", "query"],
    )
    input_text = pick_first_non_empty(
        example,
        [input_column, "input", "context", "source"],
    )
    response = pick_first_non_empty(
        example,
        [response_column, "response", "output", "answer", "completion", "target"],
    )

    if not response and isinstance(example.get("messages"), list):
        user_messages: List[str] = []
        assistant_messages: List[str] = []
        for msg in example["messages"]:
            if not isinstance(msg, dict):
                continue
            role = as_text(msg.get("role")).strip().lower()
            content = as_text(msg.get("content")).strip()
            if not content:
                continue
            if role == "assistant":
                assistant_messages.append(content)
            elif role in {"user", "system"}:
                user_messages.append(content)
        if not instruction and user_messages:
            instruction = user_messages[0]
            if len(user_messages) > 1:
                input_text = "\n".join(user_messages[1:])
        if not response and assistant_messages:
            response = assistant_messages[-1]

    return instruction, input_text, response


class InstructionSFTDataset(Dataset):
    """Instruction SFT dataset with response-only labels."""

    def __init__(
        self,
        records: Sequence[Dict[str, Any]],
        tokenizer: Any,
        max_seq_length: int,
        instruction_template: str,
        instruction_column: str,
        input_column: str,
        response_column: str,
        pad_token_id: int,
        ignore_index: int,
    ):
        self.records = records
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.instruction_template = instruction_template
        self.instruction_column = instruction_column
        self.input_column = input_column
        self.response_column = response_column
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.records[idx]
        instruction, input_text, response = extract_instruction_triplet(
            example=example,
            instruction_column=self.instruction_column,
            input_column=self.input_column,
            response_column=self.response_column,
        )

        full_text, prompt_prefix = parse_template(
            template=self.instruction_template,
            instruction=instruction,
            input_text=input_text,
            response=response,
        )

        input_ids, attention_mask = encode_text(
            tokenizer=self.tokenizer,
            text=full_text,
            max_length=self.max_seq_length,
            pad_token_id=self.pad_token_id,
            pad_to_max_length=True,
        )
        prompt_ids, _ = encode_text(
            tokenizer=self.tokenizer,
            text=prompt_prefix,
            max_length=self.max_seq_length,
            pad_token_id=self.pad_token_id,
            pad_to_max_length=False,
        )
        prompt_len = int(min(prompt_ids.shape[0], self.max_seq_length))

        labels = input_ids.clone()
        labels[:prompt_len] = self.ignore_index
        labels[attention_mask == 0] = self.ignore_index

        prompt_input_ids = input_ids.clone()
        prompt_input_ids[prompt_len:] = self.pad_token_id
        prompt_input_ids[attention_mask == 0] = self.pad_token_id

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prompt_input_ids": prompt_input_ids,
            "labels": labels,
        }


def default_sft_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Stack tensor batch."""
    out: Dict[str, torch.Tensor] = {}
    for key in batch[0].keys():
        out[key] = torch.stack([item[key] for item in batch], dim=0)
    return out


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL file into list of dictionaries."""
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_num}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object per line in {path}, got {type(row)}")
            records.append(row)
    return records


def load_records(args: argparse.Namespace) -> Sequence[Dict[str, Any]]:
    """Load dataset records from path or HF dataset name."""
    dataset_arg = args.dataset
    ds_path = Path(dataset_arg)

    if ds_path.exists():
        if ds_path.is_file():
            if ds_path.suffix.lower() == ".jsonl":
                records = read_jsonl(ds_path)
            elif ds_path.suffix.lower() == ".json":
                with ds_path.open("r", encoding="utf-8") as f:
                    obj = json.load(f)
                if isinstance(obj, list):
                    records = obj
                elif isinstance(obj, dict):
                    if args.dataset_split in obj and isinstance(obj[args.dataset_split], list):
                        records = obj[args.dataset_split]
                    else:
                        raise ValueError(
                            f"JSON file {ds_path} is a dict but does not contain split '{args.dataset_split}'."
                        )
                else:
                    raise ValueError(f"Unsupported JSON dataset format in {ds_path}.")
            else:
                if load_dataset is None:
                    raise ImportError("datasets library is required for non-JSON/JSONL files.")
                ds = load_dataset("json", data_files=str(ds_path), split="train")
                records = [ds[i] for i in range(len(ds))]
        else:
            if load_from_disk is not None and (ds_path / "dataset_info.json").exists():
                loaded = load_from_disk(str(ds_path))
                if isinstance(loaded, dict):
                    if args.dataset_split not in loaded:
                        raise ValueError(
                            f"Split '{args.dataset_split}' not found in dataset saved at {ds_path}."
                        )
                    ds = loaded[args.dataset_split]
                else:
                    ds = loaded
                records = [ds[i] for i in range(len(ds))]
            else:
                split_file_candidates = [
                    ds_path / f"{args.dataset_split}.jsonl",
                    ds_path / f"{args.dataset_split}.json",
                    ds_path / "train.jsonl",
                    ds_path / "train.json",
                ]
                for candidate in split_file_candidates:
                    if candidate.exists():
                        if candidate.suffix == ".jsonl":
                            records = read_jsonl(candidate)
                        else:
                            with candidate.open("r", encoding="utf-8") as f:
                                records = json.load(f)
                        break
                else:
                    raise ValueError(
                        f"Could not find dataset files in {ds_path}. "
                        "Expected HuggingFace disk dataset or JSON/JSONL split files."
                    )
    else:
        if load_dataset is None:
            raise ImportError("datasets library is required for Hugging Face datasets.")
        ds = load_dataset(
            path=dataset_arg,
            name=args.dataset_config,
            split=args.dataset_split,
        )
        records = [ds[i] for i in range(len(ds))]

    if args.max_train_samples > 0:
        records = records[: args.max_train_samples]

    filtered = [row for row in records if isinstance(row, dict)]
    if not filtered:
        raise ValueError("Loaded dataset is empty after parsing.")
    return filtered


def maybe_build_dataset_from_repo_helper(
    args: argparse.Namespace,
    tokenizer: Any,
    pad_token_id: int,
    ignore_index: int,
) -> Tuple[Optional[Dataset], Optional[Callable[..., Any]], Optional[ModuleType]]:
    """Try building dataset via src/dimba/data/finetuning.py helper."""
    module = optional_import("dimba.data.finetuning")
    if module is None:
        return None, None, None

    common_kwargs = {
        "dataset": args.dataset,
        "dataset_name": args.dataset,
        "dataset_path": args.dataset,
        "dataset_split": args.dataset_split,
        "split": args.dataset_split,
        "dataset_config": args.dataset_config,
        "tokenizer": tokenizer,
        "max_seq_length": args.max_seq_length,
        "instruction_template": args.instruction_template,
        "instruction_column": args.instruction_column,
        "input_column": args.input_column,
        "response_column": args.response_column,
        "pad_token_id": pad_token_id,
        "ignore_index": ignore_index,
        "max_train_samples": args.max_train_samples,
    }

    builder_candidates = [
        "build_sft_dataset",
        "create_sft_dataset",
        "load_sft_dataset",
        "build_finetuning_dataset",
        "create_finetuning_dataset",
    ]
    for fn_name in builder_candidates:
        fn = getattr(module, fn_name, None)
        if fn is None or not callable(fn):
            continue
        try:
            result = call_with_supported_kwargs(fn, **common_kwargs)
            if result is None:
                continue
            if isinstance(result, tuple):
                if len(result) == 2:
                    ds, collate = result
                elif len(result) >= 1:
                    ds, collate = result[0], None
                else:
                    continue
            else:
                ds, collate = result, None
            if hasattr(ds, "__len__") and hasattr(ds, "__getitem__"):
                return ds, collate, module
        except Exception:
            continue

    dataset_class_candidates = [
        "InstructionSFTDataset",
        "SFTDataset",
        "FinetuningDataset",
        "InstructionFinetuningDataset",
    ]
    for cls_name in dataset_class_candidates:
        cls = getattr(module, cls_name, None)
        if cls is None or not inspect.isclass(cls):
            continue
        try:
            ds = call_with_supported_kwargs(cls, **common_kwargs)
            if hasattr(ds, "__len__") and hasattr(ds, "__getitem__"):
                return ds, None, module
        except Exception:
            continue

    return None, None, module


def load_tokenizer(args: argparse.Namespace, vocab_size_hint: int) -> Tuple[Any, int]:
    """Load tokenizer from explicit arg, checkpoint dir, or fallback char tokenizer."""
    candidate_sources: List[str] = []
    if args.tokenizer:
        candidate_sources.append(args.tokenizer)

    ckpt_path = Path(args.base_checkpoint)
    if ckpt_path.is_dir():
        candidate_sources.append(str(ckpt_path))
    else:
        candidate_sources.append(str(ckpt_path.parent))

    if AutoTokenizer is not None:
        for source in candidate_sources:
            try:
                tok = AutoTokenizer.from_pretrained(source, trust_remote_code=args.trust_remote_code)
                if tok.pad_token_id is None:
                    if tok.eos_token is not None:
                        tok.pad_token = tok.eos_token
                    else:
                        tok.add_special_tokens({"pad_token": "<pad>"})
                vocab_size = int(getattr(tok, "vocab_size", vocab_size_hint))
                return tok, vocab_size
            except Exception:
                continue

    for source in candidate_sources:
        source_path = Path(source)
        tokenizer_json = source_path / "tokenizer.json" if source_path.is_dir() else source_path
        if tokenizer_json.exists():
            tok = SimpleCharacterTokenizer(vocab_size=vocab_size_hint)
            try:
                tok.load(str(tokenizer_json))
                return tok, tok.vocab_size
            except Exception:
                continue

    tok = SimpleCharacterTokenizer(vocab_size=vocab_size_hint)
    return tok, tok.vocab_size


def get_pad_token_id(tokenizer: Any) -> int:
    """Resolve tokenizer pad token id."""
    pad = getattr(tokenizer, "pad_token_id", None)
    if pad is not None:
        return int(pad)
    return 0


def get_parent_module(root: nn.Module, module_name: str) -> Tuple[nn.Module, str]:
    """Get parent module and final attribute/index token."""
    tokens = module_name.split(".")
    parent = root
    for token in tokens[:-1]:
        if token.isdigit():
            parent = parent[int(token)]  # type: ignore[index]
        else:
            parent = getattr(parent, token)
    return parent, tokens[-1]


def set_child_module(root: nn.Module, module_name: str, new_module: nn.Module) -> None:
    """Replace a child module by dotted name."""
    parent, last_token = get_parent_module(root, module_name)
    if last_token.isdigit():
        parent[int(last_token)] = new_module  # type: ignore[index]
    else:
        setattr(parent, last_token, new_module)


class LoRALinear(nn.Module):
    """Minimal LoRA wrapper for nn.Linear."""

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float):
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank r must be > 0.")

        self.base = base
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        in_features = base.in_features
        out_features = base.out_features
        self.lora_A = nn.Linear(in_features, self.r, bias=False)
        self.lora_B = nn.Linear(self.r, out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        return base_out + lora_out


def parse_target_modules(raw: Optional[str]) -> Optional[List[str]]:
    """Parse comma-separated target module patterns.

    Returns None when no explicit targets are provided, allowing the centralized
    repo LoRA helper to use its own default target module set.
    """
    if raw is None:
        return None
    modules = [part.strip() for part in raw.split(",") if part.strip()]
    if not modules:
        return None
    return modules


def apply_builtin_lora(
    model: nn.Module,
    target_modules: Sequence[str],
    lora_r: int,
    lora_alpha: float,
    lora_dropout: float,
) -> List[str]:
    """Apply local LoRA implementation to target nn.Linear layers."""
    matched: List[str] = []
    named_modules = list(model.named_modules())
    for module_name, module in named_modules:
        if not isinstance(module, nn.Linear):
            continue
        if not any(target in module_name for target in target_modules):
            continue
        replacement = LoRALinear(
            base=module,
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
        )
        set_child_module(model, module_name, replacement)
        matched.append(module_name)

    if not matched:
        raise ValueError(
            "No nn.Linear modules matched --lora-target-modules. "
            "Update the patterns to match this model's module names."
        )

    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad = True
    return matched


def maybe_apply_repo_quantization_helper(
    model: nn.Module,
    args: argparse.Namespace,
) -> Tuple[nn.Module, bool, Optional[ModuleType]]:
    """Apply centralized repo quantization helpers for Q-LoRA when available."""
    _ = args  # kept for CLI parity with existing call site
    module = optional_import("dimba.models.quantization")
    if module is None:
        return model, False, None

    quantize_fn = getattr(module, "quantize_model_4bit", None)
    prepare_fn = getattr(module, "prepare_for_qlora", None)
    if not callable(quantize_fn) or not callable(prepare_fn):
        return model, False, None

    try:
        quantized = quantize_fn(model)
        if isinstance(quantized, nn.Module):
            model = quantized

        prepared = prepare_fn(model)
        if isinstance(prepared, nn.Module):
            model = prepared
        return model, True, module
    except Exception:
        return model, False, None


def maybe_apply_repo_lora_helper(
    model: nn.Module,
    args: argparse.Namespace,
    target_modules: Optional[Sequence[str]],
) -> Tuple[nn.Module, bool, Optional[ModuleType]]:
    """Apply centralized repo LoRA helper when available."""
    module = optional_import("dimba.models.lora")
    if module is None:
        return model, False, None

    inject_fn = getattr(module, "inject_lora_to_model", None)
    if not callable(inject_fn):
        return model, False, None

    try:
        inject_fn(
            model=model,
            target_modules=list(target_modules) if target_modules is not None else None,
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
        )
        return model, True, module
    except Exception:
        return model, False, None


def extract_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Extract only LoRA adapter parameters from model state."""
    lora_state: Dict[str, torch.Tensor] = {}
    for name, tensor in model.state_dict().items():
        if "lora_A" in name or "lora_B" in name:
            lora_state[name] = tensor.detach().cpu()
    return lora_state


def maybe_save_lora_with_repo_helper(
    lora_module: Optional[ModuleType],
    model: nn.Module,
    output_dir: Path,
) -> bool:
    """Save adapters via centralized repo LoRA helper when available."""
    if lora_module is None:
        return False

    save_fn = getattr(lora_module, "save_lora_weights", None)
    if save_fn is None or not callable(save_fn):
        return False

    adapter_dir = output_dir / "lora_adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = adapter_dir / "adapter_model.pt"
    try:
        save_fn(model=model, path=adapter_path)
        return True
    except Exception:
        return False


def standardize_batch(
    batch: Dict[str, torch.Tensor],
    pad_token_id: int,
    ignore_index: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize incoming batch keys expected by training loop."""
    if "input_ids" not in batch:
        raise ValueError("Batch must include 'input_ids'.")

    input_ids = batch["input_ids"].long()
    attention_mask = batch.get("attention_mask")
    if attention_mask is None:
        attention_mask = (input_ids != pad_token_id).long()
    else:
        attention_mask = attention_mask.long()

    labels = batch.get("labels")
    if labels is None:
        labels = input_ids.clone()
        if "loss_mask" in batch:
            labels[batch["loss_mask"] == 0] = ignore_index
        elif "prompt_mask" in batch:
            labels[batch["prompt_mask"] != 0] = ignore_index
        else:
            labels[attention_mask == 0] = ignore_index
    else:
        labels = labels.long()

    prompt_input_ids = batch.get("prompt_input_ids")
    if prompt_input_ids is None:
        if "prompt_ids" in batch:
            prompt_input_ids = batch["prompt_ids"].long()
        else:
            prompt_input_ids = input_ids.clone()
            prompt_input_ids[labels != ignore_index] = pad_token_id
    else:
        prompt_input_ids = prompt_input_ids.long()

    return input_ids, prompt_input_ids, attention_mask, labels


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Return (trainable_params, total_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> LambdaLR:
    """Linear warmup then linear decay scheduler."""
    total_steps = max(total_steps, 1)
    warmup_steps = max(min(warmup_steps, total_steps), 0)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        if total_steps == warmup_steps:
            return 1.0
        decay_progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 1.0 - decay_progress)

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def diffusion_logits_forward(
    model: DIMBA,
    input_ids: torch.Tensor,
    prompt_input_ids: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    """Forward pass that conditions denoising on prompt-only tokens."""
    x_0 = model.token_embed(input_ids)
    z_0 = model.encode_latent(x_0)
    x_t, _ = model.noise_schedule.add_noise(z_0, timesteps)

    cond = model.encode_prompt(prompt_input_ids)
    cond = model.project_conditioning(cond)
    time_emb = model.timestep_embed(timesteps)

    z_pred = model.denoiser(x_t, cond, time_emb)
    x_pred = model.decode_latent(z_pred)
    logits = model.output_head(x_pred, embedding_weight=model.token_embed.get_weight())
    return logits


def compute_response_only_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute CE loss averaged over non-ignored response tokens only."""
    vocab_size = logits.shape[-1]
    flat_logits = logits.reshape(-1, vocab_size)
    flat_labels = labels.reshape(-1)

    loss_sum = F.cross_entropy(
        flat_logits,
        flat_labels,
        ignore_index=ignore_index,
        reduction="sum",
    )
    token_count = (flat_labels != ignore_index).sum()
    loss = loss_sum / token_count.clamp(min=1)
    return loss, token_count


def save_checkpoint(
    path: Path,
    model: DIMBA,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    step: int,
    args: argparse.Namespace,
    tokenizer_vocab_size: int,
) -> None:
    """Save training checkpoint."""
    payload = {
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "global_step": step,
        "args": vars(args),
        "vocab_size": model.vocab_size,
        "tokenizer_vocab_size": tokenizer_vocab_size,
        "model_config": filter_kwargs_for_callable(
            DIMBA.__init__,
            {
                "d_model": model.d_model,
                "d_prompt": model.d_prompt,
                "num_diffusion_steps": model.num_diffusion_steps,
                "num_denoiser_layers": getattr(model.denoiser, "num_layers", None),
                "latent_diffusion": model.latent_diffusion,
                "d_latent": getattr(model, "d_latent", None),
                "use_weight_tying": model.use_weight_tying,
                "use_vae_latent": model.use_vae_latent,
            },
        ),
    }
    torch.save(payload, path)


def save_tokenizer(tokenizer: Any, output_dir: Path) -> Optional[Path]:
    """Save tokenizer artifact if supported."""
    if hasattr(tokenizer, "save_pretrained") and callable(tokenizer.save_pretrained):
        tok_dir = output_dir / "tokenizer"
        tok_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(str(tok_dir))
        return tok_dir

    if hasattr(tokenizer, "save") and callable(tokenizer.save):
        tok_path = output_dir / "tokenizer.json"
        tokenizer.save(str(tok_path))
        return tok_path

    return None


def parse_args() -> argparse.Namespace:
    """CLI arguments."""
    parser = argparse.ArgumentParser(description="SFT fine-tuning for DIMBA")

    parser.add_argument("--base-checkpoint", type=str, required=True, help="Path to DIMBA checkpoint (.ckpt/.pt)")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset path or HF dataset name")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save outputs")

    parser.add_argument("--max-seq-length", type=int, default=512, help="Maximum tokenized sequence length")
    parser.add_argument(
        "--instruction-template",
        type=str,
        default=DEFAULT_INSTRUCTION_TEMPLATE,
        help="Instruction template using {instruction}, {input}, {response}",
    )

    parser.add_argument("--use-lora", action="store_true", help="Enable LoRA finetuning")
    parser.add_argument("--use-qlora", action="store_true", help="Enable Q-LoRA finetuning")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=float, default=32.0, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument(
        "--lora-target-modules",
        type=str,
        default=None,
        help=(
            "Comma-separated module name patterns for LoRA injection. "
            "When omitted, uses centralized helper defaults."
        ),
    )

    parser.add_argument("--dataset-split", type=str, default="train", help="Dataset split")
    parser.add_argument("--dataset-config", type=str, default=None, help="Optional HF dataset config")
    parser.add_argument("--instruction-column", type=str, default="instruction", help="Instruction column name")
    parser.add_argument("--input-column", type=str, default="input", help="Input/context column name")
    parser.add_argument("--response-column", type=str, default="response", help="Response/output column name")
    parser.add_argument("--max-train-samples", type=int, default=-1, help="Limit number of training samples")

    parser.add_argument("--batch-size", type=int, default=4, help="Per-step batch size")
    parser.add_argument("--grad-accumulation-steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--num-epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--max-steps", type=int, default=-1, help="Max update steps (overrides epochs when > 0)")
    parser.add_argument("--warmup-steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--ignore-index", type=int, default=-100, help="Ignore index for masked labels")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cpu, cuda, cuda:0, mps")
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        choices=["fp32", "fp16", "bf16"],
        help="Compute precision",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log-every", type=int, default=10, help="Log every N optimizer steps")

    parser.add_argument("--tokenizer", type=str, default=None, help="Tokenizer name/path (optional)")
    parser.add_argument("--trust-remote-code", action="store_true", help="HF trust_remote_code for tokenizer")

    return parser.parse_args()


def main() -> None:
    """Main entrypoint."""
    args = parse_args()

    if args.use_qlora:
        args.use_lora = True

    if args.max_seq_length <= 0:
        raise ValueError("--max-seq-length must be > 0.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0.")
    if args.grad_accumulation_steps <= 0:
        raise ValueError("--grad-accumulation-steps must be > 0.")
    if args.max_steps <= 0 and args.num_epochs <= 0:
        raise ValueError("Set --num-epochs > 0 or --max-steps > 0.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = choose_device(args.device)

    print("=" * 80)
    print("DIMBA SFT")
    print("=" * 80)
    print(f"Base checkpoint: {args.base_checkpoint}")
    print(f"Dataset: {args.dataset} (split={args.dataset_split})")
    print(f"Output dir: {output_dir}")
    print(f"Mode: {'Q-LoRA' if args.use_qlora else ('LoRA' if args.use_lora else 'Full finetune')}")
    print(f"Device: {device}")

    model, load_info = load_dimba_checkpoint(args.base_checkpoint, map_location="cpu")
    print(f"Loaded model with vocab_size={load_info['vocab_size']}")
    if load_info["missing_keys"]:
        print(f"Checkpoint load warning: {len(load_info['missing_keys'])} missing keys")
    if load_info["unexpected_keys"]:
        print(f"Checkpoint load warning: {len(load_info['unexpected_keys'])} unexpected keys")

    tokenizer, tokenizer_vocab_size = load_tokenizer(args, vocab_size_hint=model.vocab_size)
    pad_token_id = get_pad_token_id(tokenizer)
    print(f"Tokenizer vocab size: {tokenizer_vocab_size}, pad_token_id={pad_token_id}")

    helper_dataset, helper_collate, helper_module = maybe_build_dataset_from_repo_helper(
        args=args,
        tokenizer=tokenizer,
        pad_token_id=pad_token_id,
        ignore_index=args.ignore_index,
    )
    if helper_dataset is not None:
        dataset = helper_dataset
        collate_fn = helper_collate if callable(helper_collate) else default_sft_collate_fn
        print("Using finetuning dataset helper: dimba.data.finetuning")
    else:
        records = load_records(args)
        dataset = InstructionSFTDataset(
            records=records,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            instruction_template=args.instruction_template,
            instruction_column=args.instruction_column,
            input_column=args.input_column,
            response_column=args.response_column,
            pad_token_id=pad_token_id,
            ignore_index=args.ignore_index,
        )
        collate_fn = default_sft_collate_fn
        print(f"Built fallback SFT dataset with {len(dataset)} samples.")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    quant_helper_module: Optional[ModuleType] = None
    if args.use_qlora:
        model, quant_applied, quant_helper_module = maybe_apply_repo_quantization_helper(model, args)
        if quant_applied:
            print(f"Applied quantization helper from module: {quant_helper_module.__name__}")
        else:
            print(
                "Quantization helper not found. Falling back to frozen-base LoRA without true 4-bit quantization."
            )
            for param in model.parameters():
                param.requires_grad = False

    lora_helper_module: Optional[ModuleType] = None
    lora_targets = parse_target_modules(args.lora_target_modules)
    fallback_lora_targets = lora_targets if lora_targets is not None else ["denoiser"]
    if args.use_lora:
        model, used_repo_lora, lora_helper_module = maybe_apply_repo_lora_helper(
            model=model,
            args=args,
            target_modules=lora_targets,
        )
        if used_repo_lora:
            print("Applied LoRA via dimba.models.lora helper.")
        else:
            matched = apply_builtin_lora(
                model=model,
                target_modules=fallback_lora_targets,
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
            )
            print(f"Applied built-in LoRA to {len(matched)} linear layers.")
    else:
        for param in model.parameters():
            param.requires_grad = True

    model.to(device)
    model.train()

    trainable_params, total_params = count_parameters(model)
    if trainable_params == 0:
        raise RuntimeError("No trainable parameters found.")
    print(f"Trainable params: {trainable_params:,} / {total_params:,} ({100.0 * trainable_params / total_params:.2f}%)")

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    updates_per_epoch = math.ceil(len(dataloader) / args.grad_accumulation_steps)
    total_update_steps = (
        args.max_steps if args.max_steps > 0 else max(1, updates_per_epoch * args.num_epochs)
    )
    scheduler = create_scheduler(
        optimizer=optimizer,
        warmup_steps=args.warmup_steps,
        total_steps=total_update_steps,
    )

    use_autocast = device.type == "cuda" and args.precision in {"fp16", "bf16"}
    autocast_dtype = torch.float16 if args.precision == "fp16" else torch.bfloat16
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and args.precision == "fp16"))

    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    running_loss = 0.0
    running_tokens = 0
    log_counter = 0

    stop_training = False
    for epoch in range(max(1, args.num_epochs)):
        if stop_training:
            break

        epoch_iterator = tqdm(dataloader, desc=f"Epoch {epoch + 1}", leave=False)
        for batch_idx, batch in enumerate(epoch_iterator):
            input_ids, prompt_input_ids, _attention_mask, labels = standardize_batch(
                batch=batch,
                pad_token_id=pad_token_id,
                ignore_index=args.ignore_index,
            )

            input_ids = input_ids.to(device, non_blocking=True)
            prompt_input_ids = prompt_input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            bsz = input_ids.size(0)
            timesteps = torch.randint(
                low=0,
                high=model.num_diffusion_steps,
                size=(bsz,),
                device=device,
                dtype=torch.long,
            )

            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=autocast_dtype)
                if use_autocast
                else nullcontext()
            )
            with autocast_ctx:
                logits = diffusion_logits_forward(
                    model=model,
                    input_ids=input_ids,
                    prompt_input_ids=prompt_input_ids,
                    timesteps=timesteps,
                )
                loss, token_count = compute_response_only_loss(
                    logits=logits,
                    labels=labels,
                    ignore_index=args.ignore_index,
                )
                loss = loss / args.grad_accumulation_steps

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            running_loss += loss.detach().float().item()
            running_tokens += int(token_count.item())

            is_update_step = ((batch_idx + 1) % args.grad_accumulation_steps == 0) or (
                (batch_idx + 1) == len(dataloader)
            )
            if is_update_step:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                if args.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)

                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                scheduler.step()
                global_step += 1
                log_counter += 1

                if global_step % max(1, args.log_every) == 0:
                    mean_loss = running_loss / max(1, log_counter)
                    lr = optimizer.param_groups[0]["lr"]
                    print(
                        f"step={global_step}/{total_update_steps} "
                        f"loss={mean_loss:.6f} "
                        f"lr={lr:.3e} "
                        f"response_tokens={running_tokens}"
                    )
                    running_loss = 0.0
                    running_tokens = 0
                    log_counter = 0

                if args.max_steps > 0 and global_step >= args.max_steps:
                    stop_training = True
                    break

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    final_ckpt_path = output_dir / "sft_model.pt"
    save_checkpoint(
        path=final_ckpt_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=global_step,
        args=args,
        tokenizer_vocab_size=tokenizer_vocab_size,
    )
    print(f"Saved model checkpoint: {final_ckpt_path}")

    if args.use_lora:
        saved_by_helper = maybe_save_lora_with_repo_helper(
            lora_module=lora_helper_module,
            model=model,
            output_dir=output_dir,
        )
        if not saved_by_helper:
            lora_state = extract_lora_state_dict(model)
            adapter_dir = output_dir / "lora_adapter"
            adapter_dir.mkdir(parents=True, exist_ok=True)
            adapter_path = adapter_dir / "adapter_model.pt"
            adapter_meta_path = adapter_dir / "adapter_config.json"
            torch.save({"state_dict": lora_state}, adapter_path)
            with adapter_meta_path.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "base_checkpoint": args.base_checkpoint,
                        "use_qlora": args.use_qlora,
                        "lora_r": args.lora_r,
                        "lora_alpha": args.lora_alpha,
                        "lora_dropout": args.lora_dropout,
                        "lora_target_modules": (
                            lora_targets if lora_targets is not None else "repo_helper_default"
                        ),
                    },
                    f,
                    indent=2,
                )
            print(f"Saved LoRA adapter weights: {adapter_path}")

    tokenizer_path = save_tokenizer(tokenizer, output_dir)
    if tokenizer_path is not None:
        print(f"Saved tokenizer: {tokenizer_path}")

    run_meta = {
        "global_step": global_step,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "used_data_helper": helper_module is not None and helper_dataset is not None,
        "used_lora_helper": lora_helper_module is not None and args.use_lora,
        "used_quantization_helper": quant_helper_module is not None and args.use_qlora,
    }
    with (output_dir / "run_meta.json").open("w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    print("=" * 80)
    print("SFT complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
