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

from ..diffusion.sampling import sample_from_model, sample_from_model_flow


# ── helpers ──────────────────────────────────────────────────────────────────

def _degenerate_mask(
    block: torch.Tensor,
    eos_id: Optional[int],
    rep_threshold: float,
    prev_blocks: list[torch.Tensor],
) -> torch.Tensor:
    """Return a [B] bool tensor: True where row *i* of *block* looks degenerate.

    Per-row degeneracy is checked independently so a single bad prompt cannot
    terminate thinking for the whole batch.
    """
    B = block.shape[0]
    mask = torch.zeros(B, dtype=torch.bool, device=block.device)

    if block.shape[1] == 0:
        return torch.ones(B, dtype=torch.bool, device=block.device)

    # Per-row EOS fraction
    if eos_id is not None:
        eos_frac = (block == eos_id).float().mean(dim=1)  # [B]
        mask = mask | (eos_frac > 0.6)

    # Per-row repetition against each row's own previous tokens
    if prev_blocks:
        prev = torch.cat(prev_blocks, dim=1)  # [B, prev_len]
        rep = torch.stack([
            torch.isin(block[i], prev[i]).float().mean()
            for i in range(B)
        ])  # [B]
        mask = mask | (rep >= rep_threshold)

    return mask


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
          ``n_think_blocks``— ``[B]`` long tensor, per-row count of generated blocks
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
    # Per-row counters and active mask (bugs 1, 2, 3: replace scalar with [B] tensors)
    n_generated = torch.zeros(batch_size, dtype=torch.long, device=device)
    active = torch.ones(batch_size, dtype=torch.bool, device=device)
    fill_id = eos_id if eos_id is not None else 0

    # Resolve sampler name per regime once, before the loop (bug 4)
    use_flow = getattr(model, "use_flow_matching", False)
    req_sampler = sample_kwargs.get("sampler")
    if use_flow:
        resolved_sampler = req_sampler if req_sampler in ("euler", "heun") else "euler"
    else:
        resolved_sampler = req_sampler if req_sampler in ("ddim", "dpmpp") else "ddim"
    # Build a clean kwargs dict for the DDIM branch with the resolved sampler
    ddim_kwargs = {**sample_kwargs, "sampler": resolved_sampler}

    for _ in range(num_think_blocks):
        if not active.any():
            break

        # Optionally prepend <think> to the prefix before sampling this block
        gen_prefix = prefix
        if think_start_id is not None:
            start_tok = torch.full((batch_size, 1), think_start_id,
                                   dtype=torch.long, device=device)
            gen_prefix = torch.cat([gen_prefix, start_tok], dim=1) if gen_prefix is not None else start_tok

        if use_flow:
            block = sample_from_model_flow(
                model, gen_prefix, seq_len=block_size,
                num_steps=sample_kwargs.get("num_steps", 20),
                sampler=resolved_sampler,
            )
        else:
            block = sample_from_model(model, gen_prefix, block_size, **ddim_kwargs)  # [B, block_size]

        if adaptive_stop:
            deg = _degenerate_mask(block, eos_id, rep_threshold, think_blocks)  # [B] bool
            active = active & ~deg

        if not active.any():
            break

        # Freeze inactive rows: replace their tokens with fill_id so they
        # do not corrupt the shared prefix for still-active rows.
        block = torch.where(active.unsqueeze(1), block,
                            torch.full_like(block, fill_id))

        n_generated = n_generated + active.long()
        think_blocks.append(block)

        # Extend prefix with [<think>] block [</think>]
        parts = [prefix] if prefix is not None else []
        if think_start_id is not None:
            parts.append(torch.full((batch_size, 1), think_start_id, dtype=torch.long, device=device))
        parts.append(block)
        if think_end_id is not None:
            parts.append(torch.full((batch_size, 1), think_end_id, dtype=torch.long, device=device))
        prefix = torch.cat(parts, dim=1)

    # Generate final response conditioned on prefix (prompt + all think blocks)
    if use_flow:
        response = sample_from_model_flow(
            model, prefix, seq_len=response_len,
            num_steps=sample_kwargs.get("num_steps", 20),
            sampler=resolved_sampler,
        )
    else:
        response = sample_from_model(model, prefix, response_len, **ddim_kwargs)

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
        ``think_token_count`` ([B] long tensor) for reward shaping.
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
