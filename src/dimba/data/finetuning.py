"""Finetuning dataset utilities for DIMBA.

This module provides:
1. Structured datasets for instruction tuning, preference tuning, and chat data.
2. A sequence packing dataset for efficient training on short tokenized samples.
3. Dataset-specific formatters for common public finetuning datasets.
4. Helper loaders for suggested aliases, arbitrary Hugging Face datasets, and
   local ``.json``/``.jsonl`` files.

The returned dataset items are tokenized tensors that follow a practical causal
training shape:
    - ``input_ids``
    - ``attention_mask``
    - ``labels`` (prompt and padding masked with ``-100`` when applicable)
    - ``loss_mask`` (1 where loss is applied, else 0)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import Dataset

try:
    from datasets import load_dataset
except Exception:  # pragma: no cover - handled at runtime if datasets is missing.
    load_dataset = None


JSONLike = Mapping[str, Any]
TokenLike = Union[Sequence[int], torch.Tensor]


def _as_text(value: Any) -> str:
    """Convert a value to string text safely."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _first_present(example: Mapping[str, Any], keys: Sequence[str], default: Any = "") -> Any:
    """Return the first existing key value from a mapping."""
    for key in keys:
        if key in example and example[key] is not None:
            return example[key]
    return default


def _canonical_name(name: str) -> str:
    """Normalize dataset aliases for robust resolution."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


_ROLE_MAP = {
    "assistant": "assistant",
    "gpt": "assistant",
    "bot": "assistant",
    "model": "assistant",
    "user": "user",
    "human": "user",
    "instruction": "user",
    "system": "system",
    "context": "system",
}


def _normalize_role(role: Any) -> str:
    """Normalize heterogeneous role labels into user/assistant/system."""
    normalized = _canonical_name(_as_text(role))
    return _ROLE_MAP.get(normalized, "user")


def _normalize_messages(raw_messages: Any) -> List[Dict[str, str]]:
    """Normalize chat messages into ``[{role, content}, ...]``."""
    if raw_messages is None:
        return []
    if isinstance(raw_messages, Mapping):
        raw_messages = raw_messages.get("messages") or raw_messages.get("conversations")
    if not isinstance(raw_messages, Sequence) or isinstance(raw_messages, (str, bytes)):
        return []

    normalized: List[Dict[str, str]] = []
    for message in raw_messages:
        if isinstance(message, Mapping):
            role = _normalize_role(
                message.get("role", message.get("from", message.get("speaker", "user")))
            )
            content = _as_text(
                message.get("content", message.get("value", message.get("text", "")))
            ).strip()
        elif isinstance(message, Sequence) and not isinstance(message, (str, bytes)):
            if len(message) < 2:
                continue
            role = _normalize_role(message[0])
            content = _as_text(message[1]).strip()
        else:
            continue

        if content:
            normalized.append({"role": role, "content": content})
    return normalized


def _render_chat_prompt(messages: Sequence[Mapping[str, str]], add_generation_prompt: bool = True) -> str:
    """Render messages into a simple role-tagged prompt text."""
    chunks: List[str] = []
    for msg in messages:
        role = _normalize_role(msg.get("role", "user"))
        content = _as_text(msg.get("content", "")).strip()
        if content:
            chunks.append(f"<|{role}|>\n{content}\n")
    if add_generation_prompt:
        chunks.append("<|assistant|>\n")
    return "\n".join(chunks)


def _split_chat_prompt_response(messages: Sequence[Mapping[str, str]]) -> Tuple[str, str]:
    """Use the last assistant turn as target response and preceding turns as prompt."""
    normalized = _normalize_messages(messages)
    if not normalized:
        raise ValueError("Conversation has no valid messages.")

    last_assistant_idx = None
    for idx in range(len(normalized) - 1, -1, -1):
        if normalized[idx]["role"] == "assistant":
            last_assistant_idx = idx
            break

    if last_assistant_idx is None:
        if len(normalized) < 2:
            raise ValueError("Conversation has no assistant turn to use as target.")
        last_assistant_idx = len(normalized) - 1

    prompt_messages = normalized[:last_assistant_idx]
    response = normalized[last_assistant_idx]["content"]
    prompt = _render_chat_prompt(prompt_messages, add_generation_prompt=True)
    return prompt, response


def _extract_ids(encoded: Any) -> List[int]:
    """Extract flat token ids from tokenizer outputs."""
    if isinstance(encoded, Mapping):
        encoded = encoded.get("input_ids")
    if isinstance(encoded, torch.Tensor):
        encoded = encoded.tolist()
    if encoded is None:
        return []
    if isinstance(encoded, Sequence) and encoded and isinstance(encoded[0], Sequence):
        encoded = encoded[0]
    return [int(x) for x in encoded]


def _encode_text(tokenizer: Any, text: str, add_special_tokens: bool = False) -> List[int]:
    """Encode text into token ids using either HF-style or custom tokenizers."""
    if tokenizer is None:
        raise ValueError("A tokenizer is required for finetuning datasets.")

    if hasattr(tokenizer, "encode"):
        try:
            return _extract_ids(tokenizer.encode(text, add_special_tokens=add_special_tokens))
        except TypeError:
            return _extract_ids(tokenizer.encode(text))

    if callable(tokenizer):
        try:
            return _extract_ids(tokenizer(text, add_special_tokens=add_special_tokens))
        except TypeError:
            return _extract_ids(tokenizer(text))

    raise TypeError("Tokenizer must be callable or have an encode(text, ...) method.")


def _resolve_pad_token_id(tokenizer: Any, pad_token_id: Optional[int]) -> int:
    """Resolve pad token id from explicit argument or tokenizer fallback."""
    if pad_token_id is not None:
        return int(pad_token_id)
    if tokenizer is not None and hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None:
        return int(tokenizer.pad_token_id)
    return 0


def _resolve_eos_token_id(tokenizer: Any, eos_token_id: Optional[int]) -> Optional[int]:
    """Resolve eos token id from explicit argument or tokenizer fallback."""
    if eos_token_id is not None:
        return int(eos_token_id)
    if tokenizer is not None and hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
        return int(tokenizer.eos_token_id)
    return None


def _truncate_prompt_response(
    prompt_ids: List[int], response_ids: List[int], max_length: int
) -> Tuple[List[int], List[int]]:
    """Trim prompt first, then response if needed, to preserve supervision."""
    if max_length <= 0:
        raise ValueError("max_length must be > 0.")

    if len(response_ids) >= max_length:
        return [], response_ids[:max_length]

    total = len(prompt_ids) + len(response_ids)
    if total <= max_length:
        return prompt_ids, response_ids

    overflow = total - max_length
    if overflow >= len(prompt_ids):
        return [], response_ids
    return prompt_ids[overflow:], response_ids


def _build_causal_tensors(
    *,
    tokenizer: Any,
    prompt: str,
    response: str,
    max_length: int,
    ignore_prompt_loss: bool,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Build tokenized tensors with optional prompt loss masking."""
    pad_id = _resolve_pad_token_id(tokenizer, pad_token_id)
    eos_id = _resolve_eos_token_id(tokenizer, eos_token_id)

    prompt_ids = _encode_text(tokenizer, prompt, add_special_tokens=False)
    response_ids = _encode_text(tokenizer, response, add_special_tokens=False)
    if eos_id is not None and (not response_ids or response_ids[-1] != eos_id):
        response_ids.append(eos_id)

    prompt_ids, response_ids = _truncate_prompt_response(prompt_ids, response_ids, max_length=max_length)
    input_ids = prompt_ids + response_ids
    attention_mask = [1] * len(input_ids)
    prompt_loss_value = 0 if ignore_prompt_loss else 1
    loss_mask = [prompt_loss_value] * len(prompt_ids) + [1] * len(response_ids)

    if len(input_ids) < max_length:
        pad_size = max_length - len(input_ids)
        input_ids.extend([pad_id] * pad_size)
        attention_mask.extend([0] * pad_size)
        loss_mask.extend([0] * pad_size)

    labels = []
    for token_id, attend, include_in_loss in zip(input_ids, attention_mask, loss_mask):
        if attend == 1 and include_in_loss == 1:
            labels.append(token_id)
        else:
            labels.append(-100)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "loss_mask": torch.tensor(loss_mask, dtype=torch.long),
    }


class BaseDiffusionFormatter:
    """Base class for converting raw examples into DIMBA-friendly schemas."""

    def format_for_diffusion(self, example: JSONLike) -> Dict[str, Any]:
        raise NotImplementedError


class UltraChatFormatter(BaseDiffusionFormatter):
    """Formatter for UltraChat-style multi-turn conversations."""

    def format_for_diffusion(self, example: JSONLike) -> Dict[str, Any]:
        raw_messages = _first_present(example, ("messages", "conversation", "conversations"), default=None)
        messages = _normalize_messages(raw_messages if raw_messages is not None else example)
        if not messages:
            raise ValueError("UltraChat example does not contain a valid conversation.")
        return {"messages": messages}


class CodeFeedbackFormatter(BaseDiffusionFormatter):
    """Formatter for Code-Feedback variants (instruction or preference)."""

    def format_for_diffusion(self, example: JSONLike) -> Dict[str, Any]:
        prompt = _as_text(_first_present(example, ("prompt", "query", "question", "instruction"), default=""))
        chosen = _as_text(
            _first_present(
                example,
                ("chosen", "chosen_response", "preferred", "better_response", "response_a"),
                default="",
            )
        )
        rejected = _as_text(
            _first_present(
                example,
                ("rejected", "rejected_response", "other_response", "response_b"),
                default="",
            )
        )
        if chosen and rejected:
            return {"prompt": prompt, "chosen": chosen, "rejected": rejected}

        messages = _normalize_messages(
            _first_present(example, ("messages", "conversation", "conversations"), default=None)
        )
        if messages:
            return {"messages": messages}

        instruction = _as_text(
            _first_present(example, ("instruction", "prompt", "query", "question", "task"), default="")
        )
        input_text = _as_text(
            _first_present(example, ("input", "context", "starter_code", "code_context"), default="")
        )
        output = _as_text(
            _first_present(
                example,
                ("output", "response", "answer", "completion", "solution", "target"),
                default="",
            )
        )
        if not output:
            raise ValueError("Code-Feedback example missing output/response text.")
        return {"instruction": instruction, "input": input_text, "output": output}


class OpenHermes25Formatter(BaseDiffusionFormatter):
    """Formatter for OpenHermes-2.5 conversational data."""

    def format_for_diffusion(self, example: JSONLike) -> Dict[str, Any]:
        raw_messages = _first_present(example, ("conversations", "messages", "conversation"), default=None)
        messages = _normalize_messages(raw_messages)
        if messages:
            return {"messages": messages}

        instruction = _as_text(_first_present(example, ("instruction", "prompt"), default=""))
        input_text = _as_text(_first_present(example, ("input", "context"), default=""))
        output = _as_text(_first_present(example, ("output", "response", "answer"), default=""))
        if output:
            return {"instruction": instruction, "input": input_text, "output": output}

        raise ValueError("OpenHermes example has no recognized fields.")


class FeedbackCollectionFormatter(BaseDiffusionFormatter):
    """Formatter for preference-style feedback collection datasets."""

    def _from_ranked_lists(self, example: JSONLike) -> Tuple[str, str]:
        responses = _first_present(
            example, ("responses", "completions", "candidate_responses", "answers"), default=[]
        )
        scores = _first_present(example, ("scores", "ratings", "preferences"), default=[])
        if not isinstance(responses, Sequence) or isinstance(responses, (str, bytes)):
            return "", ""
        if not isinstance(scores, Sequence) or isinstance(scores, (str, bytes)):
            return "", ""
        if len(responses) < 2 or len(scores) != len(responses):
            return "", ""

        scored_pairs = []
        for response, score in zip(responses, scores):
            try:
                scored_pairs.append((_as_text(response), float(score)))
            except (TypeError, ValueError):
                return "", ""
        if len(scored_pairs) < 2:
            return "", ""

        scored_pairs.sort(key=lambda item: item[1])
        rejected = scored_pairs[0][0]
        chosen = scored_pairs[-1][0]
        return chosen, rejected

    def format_for_diffusion(self, example: JSONLike) -> Dict[str, Any]:
        prompt = _as_text(_first_present(example, ("prompt", "instruction", "query", "question"), default=""))
        chosen = _as_text(
            _first_present(
                example,
                ("chosen", "chosen_response", "accepted", "preferred", "response_j"),
                default="",
            )
        )
        rejected = _as_text(
            _first_present(
                example,
                ("rejected", "rejected_response", "discarded", "response_k"),
                default="",
            )
        )

        if not (chosen and rejected):
            ranked_chosen, ranked_rejected = self._from_ranked_lists(example)
            if ranked_chosen and ranked_rejected:
                chosen, rejected = ranked_chosen, ranked_rejected

        if not (chosen and rejected):
            raise ValueError("feedback-collection example missing chosen/rejected responses.")
        return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


@dataclass(frozen=True)
class SuggestedDatasetSpec:
    """Metadata for resolving a suggested finetuning dataset alias."""

    canonical_name: str
    aliases: Tuple[str, ...]
    hf_candidates: Tuple[str, ...]
    default_split: str
    formatter_cls: type


_SUGGESTED_DATASET_SPECS: Tuple[SuggestedDatasetSpec, ...] = (
    SuggestedDatasetSpec(
        canonical_name="ultrachat_200k",
        aliases=("ultrachat_200k", "ultrachat-200k", "ultrachat"),
        hf_candidates=("HuggingFaceH4/ultrachat_200k",),
        default_split="train_sft",
        formatter_cls=UltraChatFormatter,
    ),
    SuggestedDatasetSpec(
        canonical_name="code-feedback",
        aliases=("code-feedback", "code_feedback", "codefeedback"),
        hf_candidates=(
            "m-a-p/CodeFeedback-Filtered-Instruction",
            "m-a-p/CodeFeedback-Filtered-Preference",
        ),
        default_split="train",
        formatter_cls=CodeFeedbackFormatter,
    ),
    SuggestedDatasetSpec(
        canonical_name="openhermes-2.5",
        aliases=("openhermes-2.5", "openhermes2.5", "openhermes25", "openhermes_2_5"),
        hf_candidates=("teknium/OpenHermes-2.5",),
        default_split="train",
        formatter_cls=OpenHermes25Formatter,
    ),
    SuggestedDatasetSpec(
        canonical_name="feedback-collection",
        aliases=("feedback-collection", "feedback_collection", "feedbackcollection"),
        hf_candidates=("argilla/feedback-collection",),
        default_split="train",
        formatter_cls=FeedbackCollectionFormatter,
    ),
)


SUGGESTED_DATASETS: Dict[str, SuggestedDatasetSpec] = {}
for _spec in _SUGGESTED_DATASET_SPECS:
    for _alias in _spec.aliases:
        SUGGESTED_DATASETS[_canonical_name(_alias)] = _spec
    SUGGESTED_DATASETS[_canonical_name(_spec.canonical_name)] = _spec


def resolve_suggested_dataset(name: str) -> Optional[SuggestedDatasetSpec]:
    """Resolve a suggested dataset alias into a concrete specification."""
    return SUGGESTED_DATASETS.get(_canonical_name(name))


def load_local_records(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load a local ``.json`` or ``.jsonl`` dataset into a list of dict examples."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Local dataset not found: {file_path}")

    suffix = file_path.suffix.lower()
    records: List[Dict[str, Any]]
    if suffix == ".jsonl":
        records = []
        with file_path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {line_num} in {file_path}") from exc
                if not isinstance(value, Mapping):
                    raise ValueError(f"JSONL line {line_num} in {file_path} is not an object.")
                records.append(dict(value))
        return records

    if suffix == ".json":
        with file_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            if not all(isinstance(item, Mapping) for item in payload):
                raise ValueError(f"{file_path} must contain a list of objects.")
            return [dict(item) for item in payload]
        if isinstance(payload, Mapping):
            for key in ("data", "examples", "records", "items", "train"):
                maybe_records = payload.get(key)
                if isinstance(maybe_records, list) and all(
                    isinstance(item, Mapping) for item in maybe_records
                ):
                    return [dict(item) for item in maybe_records]
            raise ValueError(
                f"{file_path} is a JSON object but no list field was found under "
                "'data/examples/records/items/train'."
            )
        raise ValueError(f"{file_path} must be a JSON list or object containing a list.")

    raise ValueError(f"Unsupported local dataset format: {file_path.suffix}. Use .json or .jsonl")


def _parse_hf_dataset_ref(source: str) -> Tuple[str, Optional[str]]:
    """Parse ``dataset[:config]`` syntax for Hugging Face datasets."""
    if ":" not in source:
        return source, None
    dataset_name, maybe_config = source.split(":", 1)
    if not dataset_name or not maybe_config:
        return source, None
    return dataset_name, maybe_config


def _load_hf_dataset(
    dataset_name: str,
    split: Optional[str],
    streaming: bool,
    dataset_config: Optional[str] = None,
    **load_kwargs: Any,
) -> Any:
    """Load a Hugging Face dataset with local validation and clear errors."""
    if load_dataset is None:
        raise ImportError("datasets package is required to load Hugging Face datasets.")
    return load_dataset(
        dataset_name,
        name=dataset_config,
        split=split or "train",
        streaming=streaming,
        **load_kwargs,
    )


def resolve_dataset_source(source: str) -> Dict[str, Any]:
    """Resolve source metadata without loading data.

    Returns:
        A dictionary with ``source_type`` and resolved fields.
        ``source_type`` is one of ``local``, ``suggested``, or ``hf``.
    """
    source = source.strip()
    local_path = Path(source)
    if (
        local_path.exists()
        and local_path.is_file()
        and local_path.suffix.lower() in {".json", ".jsonl"}
    ):
        return {"source_type": "local", "path": str(local_path.resolve())}

    suggested = resolve_suggested_dataset(source)
    if suggested is not None:
        return {
            "source_type": "suggested",
            "canonical_name": suggested.canonical_name,
            "hf_candidates": list(suggested.hf_candidates),
            "default_split": suggested.default_split,
            "formatter": suggested.formatter_cls.__name__,
        }

    dataset_name, dataset_config = _parse_hf_dataset_ref(source)
    return {
        "source_type": "hf",
        "dataset_name": dataset_name,
        "dataset_config": dataset_config,
    }


def load_finetuning_source(
    source: str,
    split: Optional[str] = None,
    streaming: bool = False,
    **load_kwargs: Any,
) -> Tuple[Any, Optional[BaseDiffusionFormatter], Dict[str, Any]]:
    """Load a source from local files, suggested aliases, or HF dataset names.

    Returns:
        ``(records_or_dataset, formatter_or_none, metadata)``
    """
    source_info = resolve_dataset_source(source)
    source_type = source_info["source_type"]

    if source_type == "local":
        path = source_info["path"]
        records = load_local_records(path)
        return records, None, source_info

    if source_type == "suggested":
        spec = resolve_suggested_dataset(source)
        if spec is None:
            raise ValueError(f"Could not resolve suggested dataset alias: {source}")

        requested_split = split or spec.default_split
        last_error: Optional[Exception] = None
        for candidate in spec.hf_candidates:
            try:
                dataset = _load_hf_dataset(
                    candidate,
                    split=requested_split,
                    streaming=streaming,
                    **load_kwargs,
                )
                formatter = spec.formatter_cls()
                metadata = {
                    "source_type": "suggested",
                    "canonical_name": spec.canonical_name,
                    "resolved_hf_dataset": candidate,
                    "split": requested_split,
                    "streaming": streaming,
                }
                return dataset, formatter, metadata
            except Exception as exc:  # pragma: no cover - depends on external datasets.
                last_error = exc

        raise ValueError(
            f"Unable to load suggested dataset '{source}'. "
            f"Tried candidates: {spec.hf_candidates}"
        ) from last_error

    dataset_name = source_info["dataset_name"]
    dataset_config = source_info.get("dataset_config")
    dataset = _load_hf_dataset(
        dataset_name,
        split=split or "train",
        streaming=streaming,
        dataset_config=dataset_config,
        **load_kwargs,
    )
    metadata = {
        "source_type": "hf",
        "dataset_name": dataset_name,
        "dataset_config": dataset_config,
        "split": split or "train",
        "streaming": streaming,
    }
    return dataset, None, metadata


def _iter_examples(records_or_dataset: Any) -> Iterable[Dict[str, Any]]:
    """Iterate over list-like and HF datasets as dict examples."""
    if isinstance(records_or_dataset, list):
        for record in records_or_dataset:
            if not isinstance(record, Mapping):
                raise ValueError("Local records must be dictionaries.")
            yield dict(record)
        return

    for record in records_or_dataset:
        if not isinstance(record, Mapping):
            raise ValueError("Dataset examples must be dictionaries.")
        yield dict(record)


def load_and_format_finetuning_records(
    source: str,
    split: Optional[str] = None,
    formatter: Optional[BaseDiffusionFormatter] = None,
    max_examples: Optional[int] = None,
    streaming: bool = False,
    strict: bool = True,
    **load_kwargs: Any,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Load data and optionally apply a formatter with ``format_for_diffusion``.

    Args:
        source: Suggested alias, HF dataset name, or local path.
        split: Dataset split (ignored for local files).
        formatter: Explicit formatter; if omitted and source is suggested, the
            source-specific formatter is used.
        max_examples: Optional cap on returned examples.
        streaming: Whether to use streaming when loading HF datasets.
        strict: If ``True``, formatting errors are raised; otherwise invalid
            rows are skipped.

    Returns:
        ``(formatted_records, metadata)``
    """
    if streaming and max_examples is None:
        raise ValueError(
            "max_examples must be provided when streaming=True to avoid unbounded accumulation."
        )

    records_or_dataset, inferred_formatter, metadata = load_finetuning_source(
        source=source,
        split=split,
        streaming=streaming,
        **load_kwargs,
    )

    active_formatter = formatter or inferred_formatter
    formatted: List[Dict[str, Any]] = []
    for record in _iter_examples(records_or_dataset):
        if max_examples is not None and len(formatted) >= max_examples:
            break
        if active_formatter is None:
            formatted.append(record)
            continue
        try:
            formatted.append(active_formatter.format_for_diffusion(record))
        except Exception:
            if strict:
                raise
            continue

    metadata = dict(metadata)
    metadata["num_examples"] = len(formatted)
    metadata["formatter"] = active_formatter.__class__.__name__ if active_formatter else None
    return formatted, metadata


class InstructionDataset(Dataset):
    """Dataset for ``{instruction, input, output}`` style supervised finetuning."""

    DEFAULT_PROMPT_TEMPLATE = (
        "### Instruction:\n{instruction}\n\n"
        "{input_block}"
        "### Response:\n"
    )

    def __init__(
        self,
        data: Sequence[Mapping[str, Any]],
        tokenizer: Any,
        max_length: int = 1024,
        formatter: Optional[BaseDiffusionFormatter] = None,
        prompt_template: Optional[str] = None,
        ignore_prompt_loss: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> None:
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.formatter = formatter
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT_TEMPLATE
        self.ignore_prompt_loss = bool(ignore_prompt_loss)
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id

        if self.max_length <= 0:
            raise ValueError("max_length must be > 0.")

    def __len__(self) -> int:
        return len(self.data)

    def _normalize_instruction_example(self, example: Mapping[str, Any]) -> Dict[str, str]:
        if self.formatter is not None:
            example = self.formatter.format_for_diffusion(example)

        instruction = _as_text(_first_present(example, ("instruction", "prompt"), default="")).strip()
        input_text = _as_text(_first_present(example, ("input", "context"), default="")).strip()
        output = _as_text(
            _first_present(example, ("output", "response", "answer", "completion"), default="")
        ).strip()

        if not output:
            messages = _normalize_messages(
                _first_present(example, ("messages", "conversation", "conversations"), default=None)
            )
            if messages:
                prompt, response = _split_chat_prompt_response(messages)
                return {"prompt": prompt, "response": response}
            raise ValueError("Instruction example requires an output/response field.")

        input_block = f"### Input:\n{input_text}\n\n" if input_text else ""
        prompt = self.prompt_template.format(instruction=instruction, input=input_text, input_block=input_block)
        return {"prompt": prompt, "response": output}

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self._normalize_instruction_example(self.data[idx])
        return _build_causal_tensors(
            tokenizer=self.tokenizer,
            prompt=example["prompt"],
            response=example["response"],
            max_length=self.max_length,
            ignore_prompt_loss=self.ignore_prompt_loss,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
        )


class PreferenceDataset(Dataset):
    """Dataset for ``{prompt, chosen, rejected}`` style preference tuning."""

    def __init__(
        self,
        data: Sequence[Mapping[str, Any]],
        tokenizer: Any,
        max_length: int = 1024,
        formatter: Optional[BaseDiffusionFormatter] = None,
        ignore_prompt_loss: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> None:
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.formatter = formatter
        self.ignore_prompt_loss = bool(ignore_prompt_loss)
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id

        if self.max_length <= 0:
            raise ValueError("max_length must be > 0.")

    def __len__(self) -> int:
        return len(self.data)

    def _normalize_preference_example(self, example: Mapping[str, Any]) -> Dict[str, str]:
        if self.formatter is not None:
            example = self.formatter.format_for_diffusion(example)

        prompt = _as_text(_first_present(example, ("prompt", "instruction", "query", "question"), default="")).strip()
        chosen = _as_text(
            _first_present(
                example,
                ("chosen", "chosen_response", "preferred", "accepted"),
                default="",
            )
        ).strip()
        rejected = _as_text(
            _first_present(
                example,
                ("rejected", "rejected_response", "other_response", "discarded"),
                default="",
            )
        ).strip()

        if not (chosen and rejected):
            responses = _first_present(example, ("responses", "completions"), default=[])
            scores = _first_present(example, ("scores", "ratings"), default=[])
            if (
                isinstance(responses, Sequence)
                and not isinstance(responses, (str, bytes))
                and isinstance(scores, Sequence)
                and not isinstance(scores, (str, bytes))
                and len(responses) == len(scores)
                and len(responses) >= 2
            ):
                pairs: List[Tuple[float, str]] = []
                for response, score in zip(responses, scores):
                    try:
                        pairs.append((float(score), _as_text(response).strip()))
                    except (TypeError, ValueError):
                        pairs = []
                        break
                if pairs:
                    pairs.sort(key=lambda item: item[0])
                    rejected = pairs[0][1]
                    chosen = pairs[-1][1]

        if not (chosen and rejected):
            raise ValueError("Preference example requires chosen and rejected responses.")
        return {"prompt": prompt, "chosen": chosen, "rejected": rejected}

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self._normalize_preference_example(self.data[idx])
        chosen_tensors = _build_causal_tensors(
            tokenizer=self.tokenizer,
            prompt=example["prompt"],
            response=example["chosen"],
            max_length=self.max_length,
            ignore_prompt_loss=self.ignore_prompt_loss,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
        )
        rejected_tensors = _build_causal_tensors(
            tokenizer=self.tokenizer,
            prompt=example["prompt"],
            response=example["rejected"],
            max_length=self.max_length,
            ignore_prompt_loss=self.ignore_prompt_loss,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
        )

        return {
            "chosen_input_ids": chosen_tensors["input_ids"],
            "chosen_attention_mask": chosen_tensors["attention_mask"],
            "chosen_labels": chosen_tensors["labels"],
            "chosen_loss_mask": chosen_tensors["loss_mask"],
            "rejected_input_ids": rejected_tensors["input_ids"],
            "rejected_attention_mask": rejected_tensors["attention_mask"],
            "rejected_labels": rejected_tensors["labels"],
            "rejected_loss_mask": rejected_tensors["loss_mask"],
        }


class ChatDataset(Dataset):
    """Dataset for multi-turn conversations with ``{role, content}`` messages."""

    def __init__(
        self,
        data: Sequence[Mapping[str, Any]],
        tokenizer: Any,
        max_length: int = 1024,
        formatter: Optional[BaseDiffusionFormatter] = None,
        ignore_prompt_loss: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> None:
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.formatter = formatter
        self.ignore_prompt_loss = bool(ignore_prompt_loss)
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id

        if self.max_length <= 0:
            raise ValueError("max_length must be > 0.")

    def __len__(self) -> int:
        return len(self.data)

    def _normalize_chat_example(self, example: Mapping[str, Any]) -> Dict[str, str]:
        if self.formatter is not None:
            example = self.formatter.format_for_diffusion(example)

        raw_messages = _first_present(example, ("messages", "conversation", "conversations"), default=None)
        messages = _normalize_messages(raw_messages if raw_messages is not None else example)
        if messages:
            prompt, response = _split_chat_prompt_response(messages)
            return {"prompt": prompt, "response": response}

        instruction = _as_text(_first_present(example, ("instruction", "prompt"), default="")).strip()
        input_text = _as_text(_first_present(example, ("input", "context"), default="")).strip()
        output = _as_text(_first_present(example, ("output", "response", "answer"), default="")).strip()
        if not output:
            raise ValueError("Chat example must contain messages or output-like fields.")
        user_text = instruction
        if input_text:
            user_text = f"{instruction}\n\n{input_text}" if instruction else input_text
        prompt = _render_chat_prompt(
            [{"role": "user", "content": user_text}] if user_text else [],
            add_generation_prompt=True,
        )
        return {"prompt": prompt, "response": output}

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self._normalize_chat_example(self.data[idx])
        return _build_causal_tensors(
            tokenizer=self.tokenizer,
            prompt=example["prompt"],
            response=example["response"],
            max_length=self.max_length,
            ignore_prompt_loss=self.ignore_prompt_loss,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
        )


class ConcatenatedSequenceDataset(Dataset):
    """Pack short tokenized examples into fixed-length sequences.

    This dataset is useful for reducing padding waste during training by
    concatenating multiple short examples into a single sequence of
    ``max_length`` tokens.
    """

    def __init__(
        self,
        examples: Sequence[Union[TokenLike, Mapping[str, Any]]],
        max_length: int = 1024,
        sequence_key: str = "input_ids",
        attention_mask_key: Optional[str] = "attention_mask",
        pad_token_id: int = 0,
        eos_token_id: Optional[int] = None,
        drop_remainder: bool = False,
        min_sequence_length: int = 1,
    ) -> None:
        if max_length <= 0:
            raise ValueError("max_length must be > 0.")
        if min_sequence_length <= 0:
            raise ValueError("min_sequence_length must be > 0.")

        self.max_length = int(max_length)
        self.sequence_key = sequence_key
        self.attention_mask_key = attention_mask_key
        self.pad_token_id = int(pad_token_id)
        self.eos_token_id = eos_token_id if eos_token_id is None else int(eos_token_id)
        self.drop_remainder = bool(drop_remainder)
        self.min_sequence_length = int(min_sequence_length)

        self._packed_ids: List[List[int]] = []
        self._packed_lengths: List[int] = []
        self._pack_examples(examples)

    def __len__(self) -> int:
        return len(self._packed_ids)

    def _normalize_token_sequence(self, value: Union[TokenLike, Mapping[str, Any]]) -> List[int]:
        if isinstance(value, Mapping):
            if self.sequence_key not in value:
                raise KeyError(f"Missing '{self.sequence_key}' in tokenized example.")
            token_ids = _extract_ids(value[self.sequence_key])
            if self.attention_mask_key and self.attention_mask_key in value:
                attention_mask = _extract_ids(value[self.attention_mask_key])
                valid_len = min(len(token_ids), int(sum(1 for token in attention_mask if int(token) != 0)))
                token_ids = token_ids[:valid_len]
        else:
            token_ids = _extract_ids(value)

        if len(token_ids) < self.min_sequence_length:
            return []
        return token_ids

    def _pack_examples(self, examples: Sequence[Union[TokenLike, Mapping[str, Any]]]) -> None:
        buffer: List[int] = []
        for example in examples:
            token_ids = self._normalize_token_sequence(example)
            if not token_ids:
                continue
            if self.eos_token_id is not None and token_ids[-1] != self.eos_token_id:
                token_ids = token_ids + [self.eos_token_id]

            cursor = 0
            while cursor < len(token_ids):
                remaining = self.max_length - len(buffer)
                take = min(remaining, len(token_ids) - cursor)
                buffer.extend(token_ids[cursor : cursor + take])
                cursor += take

                if len(buffer) == self.max_length:
                    self._packed_ids.append(buffer)
                    self._packed_lengths.append(self.max_length)
                    buffer = []

        if buffer and not self.drop_remainder:
            valid_len = len(buffer)
            padded = buffer + [self.pad_token_id] * (self.max_length - valid_len)
            self._packed_ids.append(padded)
            self._packed_lengths.append(valid_len)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        token_ids = self._packed_ids[idx]
        valid_len = self._packed_lengths[idx]
        attention_mask = [1] * valid_len + [0] * (self.max_length - valid_len)
        labels = [
            token_id if position < valid_len else -100
            for position, token_id in enumerate(token_ids)
        ]
        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "loss_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


__all__ = [
    "BaseDiffusionFormatter",
    "UltraChatFormatter",
    "CodeFeedbackFormatter",
    "OpenHermes25Formatter",
    "FeedbackCollectionFormatter",
    "SuggestedDatasetSpec",
    "SUGGESTED_DATASETS",
    "resolve_suggested_dataset",
    "resolve_dataset_source",
    "load_local_records",
    "load_finetuning_source",
    "load_and_format_finetuning_records",
    "InstructionDataset",
    "PreferenceDataset",
    "ChatDataset",
    "ConcatenatedSequenceDataset",
]
