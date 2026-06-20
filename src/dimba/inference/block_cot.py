"""Block-sequential chain-of-thought sampling for DIMBA.

Generates thinking in discrete sequential blocks (each block = one full diffusion
pass), then generates the final response conditioned on all thinking.

    [prompt] → diffuse [think_1]
             → diffuse [think_2 | prompt, think_1]   (prefix fixed, clean)
             → ...
             → diffuse [response | prompt, think_1, …, think_N]

Within each block DIMBA's bidirectional scan attends across ALL prefix positions
(better than causal-within-block). Blocks are causal to each other, so later
thinking is genuinely conditioned on earlier thinking.

Anti-overthinking controls (see BlockCoTSampler):
  - Hard cap: max_think_blocks (default 2 for small models)
  - Length penalty: charged per think-token in GRPO reward shaping
  - Adaptive stop: skip remaining blocks when a block is highly repetitive

References
    Block Diffusion (ICLR 2025 Oral):  arXiv:2503.09573
    Test-Time Scaling for Block Diff:  arXiv:2602.09555
    d1 / diffu-GRPO:                   arXiv:2504.12216
"""
from __future__ import annotations

import torch
from typing import Optional, Dict, Any

from ..diffusion.sampling import sample_from_model


# ── helpers ──────────────────────────────────────────────────────────────────

def _repetition_ratio(block: torch.Tensor, prev_blocks: list[torch.Tensor]) -> float:
    """Fraction of tokens in *block* that also appear (majority) in *prev_blocks*."""
    if not prev_blocks:
        return 0.0
    flat = block.reshape(-1)
    if flat.numel() == 0:
        return 0.0
    all_prev = torch.cat([b.reshape(-1) for b in prev_blocks])
    prev_set = set(all_prev.tolist())
    matches = sum(1 for t in flat.tolist() if t in prev_set)
    return matches / flat.numel()


def _is_degenerate(block: torch.Tensor, eos_id: Optional[int], rep_threshold: float = 0.85,
                   prev_blocks: Optional[list] = None) -> bool:
    """True if the block looks like padding/repetition and thinking should stop."""
    flat = block.reshape(-1)
    if flat.numel() == 0:
        return True
    # Mostly EOS/padding
    if eos_id is not None:
        eos_frac = (flat == eos_id).float().mean().item()
        if eos_frac > 0.6:
            return True
    # Highly repetitive vs previous blocks
    if prev_blocks and _repetition_ratio(block, prev_blocks) >= rep_threshold:
        return True
    return False


# ── public API ───────────────────────────────────────────────────────────────

@torch.no_grad()
def block_sample_from_model(
    model: torch.nn.Module,
    prompt_ids: Optional[torch.Tensor],
    block_size: int,
    num_think_blocks: int,
    response_len: int,
    *,
    think_start_id: Optional[int] = None,
    think_end_id: Optional[int] = None,
    eos_id: Optional[int] = None,
    adaptive_stop: bool = True,
    rep_threshold: float = 0.85,
    **sample_kwargs: Any,
) -> Dict[str, torch.Tensor]:
    """Generate *response_len* tokens after up to *num_think_blocks* thinking blocks.

    Args:
        model: A :class:`~dimba.models.diffusion.DIMBA` instance.
        prompt_ids: Prompt token IDs ``[B, P]`` or None.
        block_size: Tokens per thinking block.
        num_think_blocks: Maximum number of thinking blocks to generate.
        response_len: Length of the final response block.
        think_start_id: Optional token ID prepended before each block.
        think_end_id: Optional token ID appended after each block.
        eos_id: EOS token; used for adaptive stopping (degenerate-block check).
        adaptive_stop: Stop early when a generated block looks degenerate.
        rep_threshold: Repetition fraction above which a block is considered
            degenerate (only used when ``adaptive_stop=True``).
        **sample_kwargs: Forwarded to :func:`~dimba.diffusion.sampling.sample_from_model`.

    Returns:
        Dict with keys:
          ``think_blocks``  — ``[B, n_actual, block_size]`` (only generated blocks)
          ``response``      — ``[B, response_len]``
          ``full_ids``      — ``[B, P + overhead + n*block_size + response_len]``
          ``n_think_blocks``— int, how many blocks were actually generated
    """
    device = next(model.parameters()).device

    if prompt_ids is not None:
        prompt_ids = prompt_ids.to(device)
        batch_size = prompt_ids.shape[0]
    else:
        batch_size = 1

    # Running prefix (will grow as we generate blocks)
    prefix = prompt_ids  # [B, P] or None

    think_blocks: list[torch.Tensor] = []   # list of [B, block_size]
    n_generated = 0

    for _ in range(num_think_blocks):
        # Optionally prepend <think> to the prefix before sampling this block
        gen_prefix = prefix
        if think_start_id is not None:
            start_tok = torch.full((batch_size, 1), think_start_id,
                                   dtype=torch.long, device=device)
            gen_prefix = torch.cat([gen_prefix, start_tok], dim=1) if gen_prefix is not None else start_tok

        block = sample_from_model(model, gen_prefix, block_size, **sample_kwargs)  # [B, block_size]

        if adaptive_stop and _is_degenerate(block, eos_id, rep_threshold, think_blocks):
            break

        think_blocks.append(block)
        n_generated += 1

        # Extend prefix with [<think>] block [</think>]
        parts = [prefix] if prefix is not None else []
        if think_start_id is not None:
            parts.append(torch.full((batch_size, 1), think_start_id, dtype=torch.long, device=device))
        parts.append(block)
        if think_end_id is not None:
            parts.append(torch.full((batch_size, 1), think_end_id, dtype=torch.long, device=device))
        prefix = torch.cat(parts, dim=1)

    # Generate final response conditioned on prefix (prompt + all think blocks)
    response = sample_from_model(model, prefix, response_len, **sample_kwargs)  # [B, response_len]

    full_ids = torch.cat([prefix, response], dim=1) if prefix is not None else response

    if think_blocks:
        think_tensor = torch.stack(think_blocks, dim=1)  # [B, n, block_size]
    else:
        think_tensor = torch.zeros(batch_size, 0, block_size, dtype=torch.long, device=device)

    return {
        "think_blocks": think_tensor,
        "response": response,
        "full_ids": full_ids,
        "n_think_blocks": n_generated,
    }


class BlockCoTSampler:
    """Object-oriented wrapper around :func:`block_sample_from_model`.

    Designed to be the drop-in replacement for the plain
    :class:`~dimba.diffusion.sampling.DDIMSampler` when you want CoT reasoning.

    Args:
        model: DIMBA model.
        block_size: Tokens per think block (keep small: 64–128 for 135M models).
        num_think_blocks: Max blocks (2 is enough for most tasks at this scale).
        response_len: Response length.
        think_start_id: ``<think>`` token ID (from your tokenizer).
        think_end_id: ``</think>`` token ID (from your tokenizer).
        eos_id: EOS token ID for adaptive stopping.
        adaptive_stop: Enable degenerate-block detection.
        thinking_length_weight: Per-think-token penalty to report in metrics
            (not applied here — subtract from the GRPO reward externally using
            ``result["n_think_blocks"] * block_size * weight``).
        **sample_kwargs: Defaults forwarded to the inner sampler (num_steps,
            sampler, guidance_scale, temperature, etc.).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        block_size: int = 64,
        num_think_blocks: int = 2,
        response_len: int = 128,
        think_start_id: Optional[int] = None,
        think_end_id: Optional[int] = None,
        eos_id: Optional[int] = None,
        adaptive_stop: bool = True,
        thinking_length_weight: float = 0.0,
        **sample_kwargs: Any,
    ) -> None:
        self.model = model
        self.block_size = block_size
        self.num_think_blocks = num_think_blocks
        self.response_len = response_len
        self.think_start_id = think_start_id
        self.think_end_id = think_end_id
        self.eos_id = eos_id
        self.adaptive_stop = adaptive_stop
        self.thinking_length_weight = thinking_length_weight
        self.sample_kwargs = sample_kwargs

    def sample(
        self,
        prompt_ids: Optional[torch.Tensor],
        *,
        num_think_blocks: Optional[int] = None,
        response_len: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Generate thinking + response.

        Returns the same dict as :func:`block_sample_from_model`, plus
        ``think_token_count`` (int) for reward shaping.
        """
        result = block_sample_from_model(
            self.model,
            prompt_ids,
            self.block_size,
            num_think_blocks if num_think_blocks is not None else self.num_think_blocks,
            response_len if response_len is not None else self.response_len,
            think_start_id=self.think_start_id,
            think_end_id=self.think_end_id,
            eos_id=self.eos_id,
            adaptive_stop=self.adaptive_stop,
            **self.sample_kwargs,
        )
        result["think_token_count"] = result["n_think_blocks"] * self.block_size
        result["thinking_length_penalty"] = (
            result["think_token_count"] * self.thinking_length_weight
        )
        return result
