"""Frontier-model CoT trace loaders.

All classes return ``{"prompt": str, "reasoning": str, "response": str}``
and drop directly into :class:`~dimba.data.cot_dataset.BlockCoTDataset`.

Dataset summary
---------------
Source                                    Rows     Model              Domain
----------------------------------------  -------  -----------------  ----------------------
kelexine/fable-5-sft-traces               ~885*    Claude Fable-5     general reasoning
TheFusionCube/Fable-5-CoT-Traces           468     Claude Fable-5     general
ansulev/Opus-4.7-Reasoning-CoT-4800x      2 410   Claude Opus-4.7    math/science
Jackrong/GLM-5.1-Reasoning-1M-Cleaned   100 000†  GLM-5.1            math/STEM
bespokelabs/Bespoke-Stratos-17k          16 710   DeepSeek-R1        math+code+science ★
open-thoughts/OpenThoughts-114k         114 000   DeepSeek-R1        math+code+science ★
glaiveai/reasoning-v1-20m               50 000†  R1-Distill-70B     general (non-math) ★

★ = new additions recommended after research sweep
† = subsampled; full datasets are 746K / 22M respectively

``FrontierCoTMix()`` returns a ``ConcatDataset`` of all sources (~285K rows by
default) ready for wrapping in ``BlockCoTDataset``.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

import torch
from torch.utils.data import ConcatDataset, Dataset

# ── helpers ───────────────────────────────────────────────────────────────────

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def _extract_think_response(text: str) -> tuple[str, str]:
    """Split ``<think>...</think>response`` into ``(reasoning, response)``.

    Falls back to ``("", text)`` when no think block is found — the example
    is then treated as a direct-answer sample by ``BlockCoTDataset``.
    """
    m = _THINK_RE.search(text)
    if m:
        reasoning = m.group(1).strip()
        response = _THINK_RE.sub("", text).strip()
        return reasoning, response
    return "", text.strip()


def _parse_conversations(conversations: List[Dict]) -> tuple[str, str]:
    """Extract (prompt, full_assistant_response) from a conversations list.

    Handles both ``{"from": "human/gpt"}`` (ShareGPT style) and
    ``{"role": "user/assistant"}`` (OpenAI style).
    """
    prompt, assistant = "", ""
    for turn in conversations:
        role = turn.get("from", turn.get("role", "")).lower()
        value = turn.get("value", turn.get("content", "")).strip()
        if role in ("human", "user"):
            prompt = value
        elif role in ("gpt", "assistant"):
            assistant = value
    return prompt, assistant


# ── original four sources ─────────────────────────────────────────────────────

class KelexineFableTracesDataset(Dataset):
    """``kelexine/fable-5-sft-traces`` — Fable-5 SFT traces (reasoning subset).

    Filters to ``task_type == "reasoning"`` (~885 rows from 4665 total).
    The ``thinking`` field is already split from ``response``.
    """

    def __init__(self, split: str = "train",
                 max_examples: Optional[int] = None) -> None:
        from datasets import load_dataset
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
    """``TheFusionCube/Fable-5-CoT-Traces`` — 468 Fable-5 CoT traces.

    Drops truncated examples; extracts ``<think>`` inline when present.
    """

    def __init__(self, split: str = "train",
                 max_examples: Optional[int] = None) -> None:
        from datasets import load_dataset
        ds = load_dataset("TheFusionCube/Fable-5-CoT-Traces", split=split,
                          trust_remote_code=True)
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

    Default cap 100 K to avoid this source dominating the mix.
    ``<think>...</think>`` is inline in the ``output`` field.
    """

    def __init__(self, split: str = "train",
                 max_examples: Optional[int] = 100_000) -> None:
        from datasets import load_dataset
        ds = load_dataset("Jackrong/GLM-5.1-Reasoning-1M-Cleaned", split=split,
                          trust_remote_code=True)
        if max_examples is not None:
            ds = ds.select(range(min(max_examples, len(ds))))
        self._data = ds

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        row = self._data[idx]
        reasoning, response = _extract_think_response(row.get("output", ""))
        return {
            "prompt": row.get("input", "").strip(),
            "reasoning": reasoning,
            "response": response,
        }


class OpusCoTDataset(Dataset):
    """``ansulev/Opus-4.7-Reasoning-CoT-4800x`` — 2 410 Opus-4.7 CoT traces.

    ``messages[1].content`` contains ``<think>...</think>answer``.
    """

    def __init__(self, split: str = "train",
                 max_examples: Optional[int] = None) -> None:
        from datasets import load_dataset
        ds = load_dataset("ansulev/Opus-4.7-Reasoning-CoT-4800x", split=split,
                          trust_remote_code=True)
        if max_examples is not None:
            ds = ds.select(range(min(max_examples, len(ds))))
        self._data = ds

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        row = self._data[idx]
        prompt, full = _parse_conversations(row.get("messages", []))
        reasoning, response = _extract_think_response(full)
        return {"prompt": prompt, "reasoning": reasoning, "response": response}


# ── new additions (code + math focus) ────────────────────────────────────────

class BespokeStratosDataset(Dataset):
    """``bespokelabs/Bespoke-Stratos-17k`` — 16 710 DeepSeek-R1 traces.

    10 K math + 5 K code + 1 K science.  Rejection-sampled for correctness —
    highest signal-to-noise of the small frontier datasets.  ShareGPT
    ``conversations`` format; ``<think>`` is inline in the assistant turn.
    """

    def __init__(self, split: str = "train",
                 max_examples: Optional[int] = None) -> None:
        from datasets import load_dataset
        ds = load_dataset("bespokelabs/Bespoke-Stratos-17k", split=split,
                          trust_remote_code=True)
        if max_examples is not None:
            ds = ds.select(range(min(max_examples, len(ds))))
        self._data = ds

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        row = self._data[idx]
        prompt, full = _parse_conversations(row.get("conversations", []))
        reasoning, response = _extract_think_response(full)
        return {"prompt": prompt, "reasoning": reasoning, "response": response}


class OpenThoughtsDataset(Dataset):
    """``open-thoughts/OpenThoughts-114k`` — 114 K DeepSeek-R1 traces.

    Math (89 K) + code (20 K) + science (4 K) + puzzles (1 K).  Verified
    correct answers.  ShareGPT ``conversations`` format with ``<think>`` inline.
    The dataset also exposes a ``metadata`` subset with ``deepseek_reasoning``
    as a clean field; we use the standard split here for simplicity.

    Args:
        max_examples: Default None (all 114 K).  Set to e.g. 50_000 if VRAM
            or time is a constraint.
    """

    def __init__(self, split: str = "train",
                 max_examples: Optional[int] = None) -> None:
        from datasets import load_dataset
        ds = load_dataset("open-thoughts/OpenThoughts-114k", split=split,
                          trust_remote_code=True)
        if max_examples is not None:
            ds = ds.select(range(min(max_examples, len(ds))))
        self._data = ds

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        row = self._data[idx]
        prompt, full = _parse_conversations(row.get("conversations", []))
        reasoning, response = _extract_think_response(full)
        return {"prompt": prompt, "reasoning": reasoning, "response": response}


class GlaiveReasoningDataset(Dataset):
    """``glaiveai/reasoning-v1-20m`` — 22 M R1-Distill-70B traces (sampled).

    The only large-scale **general-domain** dataset with clean ``<think>``
    traces — covers science, philosophy, law, history, engineering.  Default
    cap 50 K to balance the math-heavy sources without blowing up SFT time.
    ``prompt`` / ``response`` columns; ``<think>`` is inline in ``response``.

    Args:
        max_examples: Default 50_000.  Pass None for the full 22 M (slow).
    """

    def __init__(self, split: str = "train",
                 max_examples: Optional[int] = 50_000) -> None:
        from datasets import load_dataset
        # Stream to avoid downloading all 22M rows before selecting.
        ds_stream = load_dataset("glaiveai/reasoning-v1-20m", split=split,
                                 trust_remote_code=True, streaming=True)
        rows = []
        for i, row in enumerate(ds_stream):
            if max_examples is not None and i >= max_examples:
                break
            rows.append(row)
        self._data = rows

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


# ── convenience mix ───────────────────────────────────────────────────────────

def FrontierCoTMix(
    glm_max: int = 100_000,
    glaive_max: int = 50_000,
    openthoughts_max: Optional[int] = None,
    include_fable_kelexine: bool = True,
    include_fable_fusioncube: bool = True,
    include_glm: bool = True,
    include_opus: bool = True,
    include_stratos: bool = True,
    include_openthoughts: bool = True,
    include_glaive: bool = True,
) -> ConcatDataset:
    """Return a ``ConcatDataset`` of frontier CoT sources.

    Default mix                          Rows     Model / domain
    -----------------------------------  -------  --------------------------
    Fable-5 kelexine (reasoning only)      ~885   Claude Fable-5 / general
    Fable-5 FusionCube                      468   Claude Fable-5 / general
    Opus-4.7 CoT                          2 170   Claude Opus / math+sci
    GLM-5.1 (capped)                    100 000   GLM / math+STEM
    Bespoke-Stratos-17k                  16 710   DeepSeek-R1 / math+code+sci
    OpenThoughts-114k                   114 000   DeepSeek-R1 / math+code+sci
    Glaive reasoning (capped)            50 000   R1-Distill-70B / general
                                        -------
    Total                              ~284 000

    Args:
        glm_max: Row cap for GLM-5.1 (default 100 K).
        glaive_max: Row cap for Glaive (default 50 K).
        openthoughts_max: Row cap for OpenThoughts (default None = all 114 K).
        include_*: Toggle individual sources for ablations.

    Returns:
        A ``ConcatDataset`` ready to wrap in ``BlockCoTDataset``.
    """
    import logging as _logging
    _log = _logging.getLogger(__name__)

    def _try_load(name: str, factory):
        try:
            ds = factory()
            _log.info("  %-30s %d rows", name, len(ds))
            return ds
        except Exception as exc:
            _log.warning("FrontierCoTMix: skipping %s — %s", name, exc)
            return None

    candidates = []
    if include_fable_kelexine:
        candidates.append(("kelexine/fable-5-sft-traces", KelexineFableTracesDataset))
    if include_fable_fusioncube:
        candidates.append(("TheFusionCube/Fable-5-CoT-Traces", FusionCubeFableCoTDataset))
    if include_opus:
        candidates.append(("ansulev/Opus-4.7-Reasoning-CoT-4800x", OpusCoTDataset))
    if include_glm:
        candidates.append(("Jackrong/GLM-5.1-Reasoning-1M-Cleaned",
                           lambda: GLMReasoningDataset(max_examples=glm_max)))
    if include_stratos:
        candidates.append(("bespokelabs/Bespoke-Stratos-17k", BespokeStratosDataset))
    if include_openthoughts:
        candidates.append(("open-thoughts/OpenThoughts-114k",
                           lambda: OpenThoughtsDataset(max_examples=openthoughts_max)))
    if include_glaive:
        candidates.append(("glaiveai/reasoning-v1-20m",
                           lambda: GlaiveReasoningDataset(max_examples=glaive_max)))

    parts: list = [_try_load(name, f) for name, f in candidates]
    parts = [p for p in parts if p is not None]
    if not parts:
        raise ValueError("All FrontierCoTMix sources failed to load — check your network/HF auth.")
    _log.info("FrontierCoTMix: %d sources, %d total rows",
              len(parts), sum(len(p) for p in parts))
    return ConcatDataset(parts)
