"""Frontier-model CoT trace loaders.

All four classes return ``{"prompt": str, "reasoning": str, "response": str}``
so they drop directly into :class:`~dimba.data.cot_dataset.BlockCoTDataset`.

Dataset summary
---------------
+------------------------------------------+--------+-----------------------------------+
| Dataset                                  | Rows   | CoT format                        |
+------------------------------------------+--------+-----------------------------------+
| kelexine/fable-5-sft-traces              |  ~885* | ``thinking`` field + ``response`` |
| Jackrong/GLM-5.1-Reasoning-1M-Cleaned   | 746 K  | inline ``<think>...</think>``     |
| TheFusionCube/Fable-5-CoT-Traces        |   468  | inline (no guaranteed tags)       |
| ansulev/Opus-4.7-Reasoning-CoT-4800x    |  2410  | inline ``<think>...</think>``     |
+------------------------------------------+--------+-----------------------------------+
* after filtering to task_type == "reasoning" (81% of the dataset is agentic/tool-use)

``FrontierCoTMix`` returns a ready-to-use ``torch.utils.data.ConcatDataset`` of all
four sources with sensible default subsampling.
"""
from __future__ import annotations

import re
from typing import Dict, Optional

import torch
from torch.utils.data import ConcatDataset, Dataset

# ── helpers ───────────────────────────────────────────────────────────────────

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def _extract_think_response(text: str) -> tuple[str, str]:
    """Split inline ``<think>...</think>thinking</think>response`` into parts.

    Returns ``(reasoning, response)``.  If no think block is found, returns
    ``("", text)`` so the example is treated as a direct-answer sample.
    """
    m = _THINK_RE.search(text)
    if m:
        reasoning = m.group(1).strip()
        response = _THINK_RE.sub("", text).strip()
        return reasoning, response
    return "", text.strip()


# ── dataset classes ───────────────────────────────────────────────────────────

class KelexineFableTracesDataset(Dataset):
    """``kelexine/fable-5-sft-traces`` — Fable-5 SFT traces (reasoning subset).

    The dataset is 81% agentic (tool-use) and 19% reasoning.  We filter to
    ``task_type == "reasoning"`` only; the ``thinking`` field is already split
    from the response so no regex extraction is needed.

    Args:
        split: HuggingFace split (only "train" exists).
        max_examples: Optional cap for fast debugging.
    """

    def __init__(self, split: str = "train", max_examples: Optional[int] = None) -> None:
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise ImportError("pip install datasets") from e

        ds = load_dataset("kelexine/fable-5-sft-traces", split=split,
                          trust_remote_code=True)
        ds = ds.filter(lambda r: r.get("task_type") == "reasoning")
        if max_examples is not None:
            ds = ds.select(range(min(max_examples, len(ds))))
        self._data = ds

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        row = self._data[idx]
        # Prompt from the last human turn in messages
        prompt = ""
        for msg in row.get("messages", []):
            if msg.get("role") == "user":
                prompt = msg.get("content", "")
        return {
            "prompt": prompt,
            "reasoning": row.get("thinking", "").strip(),
            "response": row.get("response", "").strip(),
        }


class FusionCubeFableCoTDataset(Dataset):
    """``TheFusionCube/Fable-5-CoT-Traces`` — Fable-5 CoT traces.

    468 rows.  CoT may be inline in the ``response`` field with or without
    ``<think>`` tags.  We attempt extraction; fall back to empty reasoning.

    Args:
        split: HuggingFace split ("train" is the only one).
        max_examples: Optional cap.
    """

    def __init__(self, split: str = "train", max_examples: Optional[int] = None) -> None:
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise ImportError("pip install datasets") from e

        ds = load_dataset("TheFusionCube/Fable-5-CoT-Traces", split=split,
                          trust_remote_code=True)
        # Drop truncated examples (incomplete CoT)
        ds = ds.filter(lambda r: not r.get("truncated", False))
        if max_examples is not None:
            ds = ds.select(range(min(max_examples, len(ds))))
        self._data = ds

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        row = self._data[idx]
        reasoning, response = _extract_think_response(row.get("response", ""))
        return {
            "prompt": row.get("prompt", "").strip(),
            "reasoning": reasoning,
            "response": response,
        }


class GLMReasoningDataset(Dataset):
    """``Jackrong/GLM-5.1-Reasoning-1M-Cleaned`` — 746 K GLM-4 reasoning traces.

    Each row has ``input`` (prompt) and ``output`` with inline
    ``<think>...</think>`` then the answer.  Default cap is 100 K to avoid
    this single source dominating the SFT mix.

    Args:
        split: "train" (only split).
        max_examples: Default 100_000.  Pass ``None`` for the full 746 K.
    """

    def __init__(self, split: str = "train",
                 max_examples: Optional[int] = 100_000) -> None:
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise ImportError("pip install datasets") from e

        ds = load_dataset("Jackrong/GLM-5.1-Reasoning-1M-Cleaned", split=split,
                          trust_remote_code=True)
        if max_examples is not None:
            ds = ds.select(range(min(max_examples, len(ds))))
        self._data = ds

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        row = self._data[idx]
        prompt = row.get("input", "").strip()
        output = row.get("output", "")
        reasoning, response = _extract_think_response(output)
        return {"prompt": prompt, "reasoning": reasoning, "response": response}


class OpusCoTDataset(Dataset):
    """``ansulev/Opus-4.7-Reasoning-CoT-4800x`` — 2 410 Opus-4.7 CoT traces.

    Each row has ``messages`` list: ``[{role: user, content: prompt},
    {role: assistant, content: <think>...</think>answer}]``.

    Args:
        split: "train" / "validation" / "test".
        max_examples: Optional cap.
    """

    def __init__(self, split: str = "train",
                 max_examples: Optional[int] = None) -> None:
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise ImportError("pip install datasets") from e

        ds = load_dataset("ansulev/Opus-4.7-Reasoning-CoT-4800x", split=split,
                          trust_remote_code=True)
        if max_examples is not None:
            ds = ds.select(range(min(max_examples, len(ds))))
        self._data = ds

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        row = self._data[idx]
        messages = row.get("messages", [])
        prompt, full_response = "", ""
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                prompt = content.strip()
            elif role == "assistant":
                full_response = content.strip()
        reasoning, response = _extract_think_response(full_response)
        return {"prompt": prompt, "reasoning": reasoning, "response": response}


# ── convenience mix ───────────────────────────────────────────────────────────

def FrontierCoTMix(
    glm_max: int = 100_000,
    include_fable_kelexine: bool = True,
    include_fable_fusioncube: bool = True,
    include_glm: bool = True,
    include_opus: bool = True,
) -> ConcatDataset:
    """Return a ``ConcatDataset`` of all four frontier CoT sources.

    Default mix (rows):
      Fable-5 (kelexine, reasoning-only)     ~885
      Fable-5 CoT traces (FusionCube)        ~468
      GLM-5.1 (subsampled)                100 000
      Opus-4.7 CoT                          2 170
                                         --------
                                         ~103 500

    Args:
        glm_max: Row cap for the large GLM dataset (default 100 K).
        include_*: Toggle individual sources for ablations.

    Returns:
        A ``ConcatDataset`` ready to wrap in ``BlockCoTDataset``.
    """
    parts = []
    if include_fable_kelexine:
        parts.append(KelexineFableTracesDataset())
    if include_fable_fusioncube:
        parts.append(FusionCubeFableCoTDataset())
    if include_glm:
        parts.append(GLMReasoningDataset(max_examples=glm_max))
    if include_opus:
        parts.append(OpusCoTDataset())
    if not parts:
        raise ValueError("At least one source must be enabled in FrontierCoTMix.")
    return ConcatDataset(parts)
