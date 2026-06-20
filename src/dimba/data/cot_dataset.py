"""CoT dataset loaders for block-sequential thinking training.

Recommended datasets (all freely available on HuggingFace):

  SmolTalk     HuggingFaceTB/smoltalk           ~1M  general reasoning
               Subset "smol-magpie-ultra" is best for instruction-following.
               Matches SmolLM-135M teacher (same training family) — biggest
               boost for Mode-A distilled models.

  Orca-Math    microsoft/orca-math-word-problems-200k   200k  math
               Concise step-by-step word problem solutions. Short enough to
               fit in 2 think blocks at block_size=64. Best math data for
               sub-200M models (NuminaMath-CoT traces are too long).

Both loaders return dicts with "prompt", "response", and "reasoning" keys.
``BlockCoTDataset`` wraps either, splits reasoning into blocks, and returns
tokenized tensors ready for SFT or GRPO.
"""
from __future__ import annotations

import re
from typing import Optional, List, Dict, Any, Callable

import torch
from torch.utils.data import Dataset


# ── text utilities ─────────────────────────────────────────────────────────────

def _split_into_blocks(text: str, num_blocks: int, min_chars: int = 20) -> List[str]:
    """Split *text* into *num_blocks* roughly equal chunks at sentence boundaries."""
    # Try to split at sentence-ending punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s for s in sentences if len(s) >= min_chars]
    if not sentences:
        # Fallback: split by chars
        chunk = len(text) // num_blocks
        return [text[i * chunk:(i + 1) * chunk].strip() for i in range(num_blocks)]

    if len(sentences) <= num_blocks:
        # Pad with empty if fewer sentences than blocks (skip empty blocks during training)
        return sentences + [""] * (num_blocks - len(sentences))

    # Distribute sentences evenly across blocks
    per_block = len(sentences) // num_blocks
    blocks = []
    for i in range(num_blocks):
        start = i * per_block
        end = start + per_block if i < num_blocks - 1 else len(sentences)
        blocks.append(" ".join(sentences[start:end]))
    return blocks


# ── raw dataset wrappers ──────────────────────────────────────────────────────

class SmolTalkDataset(Dataset):
    """Loads HuggingFaceTB/smoltalk and normalises to {prompt, reasoning, response}.

    Args:
        split: Dataset split ("train" / "test").
        subset: HuggingFace config name. "smol-magpie-ultra" gives the best
            instruction-following data; "all" includes all subsets.
        max_examples: Truncate for fast debugging.
    """

    def __init__(
        self,
        split: str = "train",
        subset: str = "smol-magpie-ultra",
        max_examples: Optional[int] = None,
    ) -> None:
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise ImportError("pip install datasets") from e

        ds = load_dataset("HuggingFaceTB/smoltalk", subset, split=split,
                          trust_remote_code=True)
        if max_examples is not None:
            ds = ds.select(range(min(max_examples, len(ds))))
        self._data = ds

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        row = self._data[idx]
        messages = row.get("messages", [])
        prompt, response = "", ""
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                prompt = content
            elif role == "assistant":
                response = content
        return {"prompt": prompt, "reasoning": "", "response": response}


class OrcaMathDataset(Dataset):
    """Loads microsoft/orca-math-word-problems-200k.

    Each example has a "question" and a "answer" with step-by-step working.
    The working is treated as reasoning; the final numeric line is the response.

    Args:
        split: "train" (the only split available).
        max_examples: Truncate for fast debugging.
    """

    def __init__(self, split: str = "train", max_examples: Optional[int] = None) -> None:
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise ImportError("pip install datasets") from e

        ds = load_dataset("microsoft/orca-math-word-problems-200k", split=split,
                          trust_remote_code=True)
        if max_examples is not None:
            ds = ds.select(range(min(max_examples, len(ds))))
        self._data = ds

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        row = self._data[idx]
        question = row.get("question", "")
        answer_full = row.get("answer", "")
        # Last line is typically the numeric answer; everything before = reasoning
        lines = [l.strip() for l in answer_full.splitlines() if l.strip()]
        if len(lines) > 1:
            reasoning = " ".join(lines[:-1])
            response = lines[-1]
        else:
            reasoning = ""
            response = answer_full
        return {"prompt": question, "reasoning": reasoning, "response": response}


# ── block-CoT wrapper ─────────────────────────────────────────────────────────

class BlockCoTDataset(Dataset):
    """Tokenizes and formats examples for block-CoT SFT or GRPO.

    Takes any dataset that yields ``{prompt, reasoning, response}`` dicts and
    converts them into token tensors in the format:

        [prompt_tokens] [think_start] [block_1] [think_end]
                        [think_start] [block_2] [think_end]
                        [response_tokens]

    Special tokens are inserted only when *think_start_id* / *think_end_id*
    are provided. When *reasoning* is empty, no think blocks are emitted and
    the example is treated as a direct-answer example (this is intentional —
    training on a mix of "think" and "direct" examples prevents the model from
    always overthinking easy prompts).

    Args:
        base_dataset: Any Dataset yielding ``{prompt, reasoning, response}``.
        tokenizer: A callable ``text -> List[int]`` (or ``encode()`` method).
        max_prompt_len: Truncate prompt to this many tokens.
        block_size: Tokens per think block (should match inference block_size).
        num_think_blocks: How many blocks to split the reasoning into.
        response_len: Truncate/pad response to this length.
        think_start_id: Token ID for ``<think>``.
        think_end_id: Token ID for ``</think>``.
        pad_id: Padding token ID.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        tokenizer: Any,
        max_prompt_len: int = 128,
        block_size: int = 64,
        num_think_blocks: int = 2,
        response_len: int = 128,
        think_start_id: Optional[int] = None,
        think_end_id: Optional[int] = None,
        pad_id: int = 0,
    ) -> None:
        self.base = base_dataset
        self.tok = tokenizer
        self.max_prompt_len = max_prompt_len
        self.block_size = block_size
        self.num_think_blocks = num_think_blocks
        self.response_len = response_len
        self.think_start_id = think_start_id
        self.think_end_id = think_end_id
        self.pad_id = pad_id

    # ------------------------------------------------------------------
    def _encode(self, text: str) -> List[int]:
        if hasattr(self.tok, "encode"):
            return self.tok.encode(text)
        return self.tok(text)

    def _pad_or_trunc(self, ids: List[int], length: int) -> List[int]:
        if len(ids) >= length:
            return ids[:length]
        return ids + [self.pad_id] * (length - len(ids))

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.base[idx]
        prompt_ids = self._pad_or_trunc(self._encode(item["prompt"]), self.max_prompt_len)
        response_ids = self._pad_or_trunc(self._encode(item["response"]), self.response_len)

        reasoning = item.get("reasoning", "")
        think_ids_list: List[List[int]] = []

        if reasoning.strip():
            blocks_text = _split_into_blocks(reasoning, self.num_think_blocks)
            for bt in blocks_text:
                if bt.strip():
                    think_ids_list.append(
                        self._pad_or_trunc(self._encode(bt), self.block_size)
                    )

        # Pad missing blocks with zeros (masked out during loss computation)
        while len(think_ids_list) < self.num_think_blocks:
            think_ids_list.append([self.pad_id] * self.block_size)

        # Build full sequence for SFT: [prompt | think*N | response]
        full: List[int] = list(prompt_ids)
        for block_ids in think_ids_list:
            if self.think_start_id is not None:
                full.append(self.think_start_id)
            full.extend(block_ids)
            if self.think_end_id is not None:
                full.append(self.think_end_id)
        full.extend(response_ids)

        # Response mask: 1 on response positions (for loss computation)
        prefix_len = len(full) - self.response_len
        response_mask = [0] * prefix_len + [1] * self.response_len

        return {
            "input_ids": torch.tensor(full, dtype=torch.long),
            "prompt_ids": torch.tensor(prompt_ids, dtype=torch.long),
            "response_ids": torch.tensor(response_ids, dtype=torch.long),
            "think_ids": torch.tensor(think_ids_list, dtype=torch.long),  # [num_blocks, block_size]
            "response_mask": torch.tensor(response_mask, dtype=torch.float),
            "has_reasoning": torch.tensor(bool(reasoning.strip()), dtype=torch.bool),
        }
