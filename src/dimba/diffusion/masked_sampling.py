"""Iterative decoding for discrete masked (absorbing-state) diffusion.

This implements LLaDA-style *confidence-based* iterative unmasking for masked
diffusion language models (LLaDA, arXiv:2502.09992; MDLM, arXiv:2406.07524).

The decoder is **model-agnostic**: it never touches the model object directly.
Instead the caller passes a callable ``predict_logits(ids, t) -> logits`` that
maps a (partially masked) id sequence and a scalar timestep to vocabulary
logits. This keeps the sampler decoupled from the (concurrently refactored) core
model and trivially reusable for the continuous, masked, or hybrid heads.

Algorithm (conditional generation done right)
----------------------------------------------
1. The response region is initialised fully ``[MASK]``; the prompt tokens are
   placed verbatim and **never** overwritten.
2. At each of ``num_steps`` reverse steps we predict logits for the whole
   sequence, take the arg-max token and its softmax probability (the
   *confidence*) at each currently-masked response position.
3. We *commit* (unmask) the highest-confidence positions, scheduling the number
   committed per step so that all response positions are revealed by the final
   step.
4. Optionally (``remask=True``) we additionally re-mask the lowest-confidence
   *already-committed* positions each step (LLaDA's low-confidence remasking),
   letting the model revise earlier mistakes.
"""

from __future__ import annotations

import inspect
import math
from typing import Callable, Optional

import torch

# predict_logits(ids: [batch, seq] long, t: float) -> logits: [batch, seq, vocab]
PredictLogits = Callable[..., torch.Tensor]


def _max_softmax_confidence(logits: torch.Tensor):
    """Return argmax probability in fp32 without materializing full probabilities."""
    # Native bf16/fp16 logsumexp loses enough precision to reorder confidence-ranked
    # positions. The sampler's reveal trajectory is accuracy-sensitive, so perform
    # both reductions in fp32 while still avoiding a second [B, K, vocab] softmax.
    values = logits.float()
    max_logits, pred_ids = values.max(dim=-1)
    return (max_logits - torch.logsumexp(values, dim=-1)).exp(), pred_ids


def _unmask_count_schedule(gen_len: int, num_steps: int) -> list[int]:
    """How many positions to reveal at each step so all ``gen_len`` are revealed.

    Distributes ``gen_len`` reveals across ``num_steps`` as evenly as possible
    (front-loading any remainder), guaranteeing the sum equals ``gen_len`` and
    every step reveals at least zero (and the schedule is non-increasing).

    Args:
        gen_len: Number of positions to reveal in total.
        num_steps: Number of reverse diffusion steps.

    Returns:
        A list of length ``num_steps`` of non-negative ints summing to
        ``gen_len``.
    """
    base = gen_len // num_steps
    remainder = gen_len % num_steps
    # Front-load the remainder so early (most-masked) steps commit slightly more.
    return [base + (1 if i < remainder else 0) for i in range(num_steps)]


@torch.inference_mode()
def masked_diffusion_sample(
    predict_logits: PredictLogits,
    prompt_ids: torch.Tensor,
    gen_len: int,
    mask_token_id: int,
    num_steps: int,
    temperature: float = 1.0,
    remask: bool = False,
    remask_fraction: float = 0.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Generate a response by LLaDA-style confidence-based iterative unmasking.

    Args:
        predict_logits: Model-agnostic callable ``predict_logits(ids, t)`` where
            ``ids`` is ``[batch, prompt_len + gen_len]`` (long) and ``t`` is the
            current scalar timestep in ``(0, 1]`` (``1`` fully masked, ``-> 0``
            clean). Must return logits ``[batch, prompt_len + gen_len, vocab]``.
            Callables exposing a ``positions=[B,K]`` parameter automatically use
            DIMBA's selected-projection fast path and return ``[B,K,vocab]``.
        prompt_ids: Conditioning prompt ids ``[batch, prompt_len]`` (long). Kept
            fixed and unmasked throughout (correct conditional generation).
        gen_len: Number of response tokens to generate (appended after prompt).
        mask_token_id: Vocabulary id of the ``[MASK]`` token.
        num_steps: Number of reverse diffusion steps.
        temperature: Softmax temperature applied before sampling/arg-max. ``<= 0``
            and ``== 1`` both mean greedy-equivalent scaling is skipped for
            ``temperature == 1``; values ``> 0`` rescale logits. Tokens are taken
            greedily (arg-max); temperature only affects the confidence score.
        remask: If ``True``, enable LLaDA low-confidence remasking: after
            committing, re-mask the lowest-confidence committed positions so the
            model can revise them on later steps.
        remask_fraction: Fraction of currently-committed positions to re-mask per
            step when ``remask`` is enabled (e.g. ``0.1``). Ignored on the final
            step so the output is always fully unmasked.
        device: Device for computation; defaults to ``prompt_ids.device``.

    Returns:
        Generated response ids ``[batch, gen_len]`` (long), fully unmasked.
    """
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1.")
    if device is None:
        device = prompt_ids.device

    prompt_ids = prompt_ids.to(device)
    batch, prompt_len = prompt_ids.shape
    total_len = prompt_len + gen_len

    # Build the working sequence: [prompt | all-MASK response].
    ids = torch.full((batch, total_len), mask_token_id, dtype=torch.long, device=device)
    ids[:, :prompt_len] = prompt_ids

    # Track which response positions are still masked (True == masked).
    # Prompt positions are never masked.
    response_masked = torch.ones((batch, gen_len), dtype=torch.bool, device=device)

    reveal_schedule = _unmask_count_schedule(gen_len, num_steps)
    try:
        selected_projection = not remask and "positions" in inspect.signature(
            predict_logits
        ).parameters
    except (TypeError, ValueError):
        selected_projection = False
    active_positions = (
        torch.arange(prompt_len, total_len, device=device).unsqueeze(0).expand(batch, -1)
    )
    n_active = gen_len
    n_committed = 0

    for step in range(num_steps):
        # Continuous time goes 1 -> ~0 across steps (fully masked -> clean).
        t = 1.0 - step / num_steps
        t = max(t, 1e-3)

        if selected_projection:
            # The denoiser still sees the complete sequence; only its final, expensive
            # hidden-to-vocabulary projection is gathered to unresolved positions.
            resp_logits = predict_logits(ids, t, positions=active_positions)
        else:
            logits = predict_logits(ids, t)  # [batch, total_len, vocab]
            resp_logits = logits[:, prompt_len:, :]  # [batch, gen_len, vocab]

        if temperature != 1.0 and temperature > 0:
            resp_logits = resp_logits / temperature

        confidence, pred_ids = _max_softmax_confidence(resp_logits)

        # The selected path contains only unresolved positions. The generic path
        # masks committed positions out of the confidence ranking.
        select_conf = (
            confidence
            if selected_projection
            else confidence.masked_fill(~response_masked, float("-inf"))
        )

        # Number of positions to reveal this step.
        n_reveal = reveal_schedule[step]
        # On the last step, force-reveal everything still masked.
        if step == num_steps - 1:
            n_reveal = gen_len

        if selected_projection and n_reveal > 0:
            k = min(n_reveal, n_active)
            reveal_local = torch.topk(select_conf, k=k, dim=1).indices
            reveal_positions = active_positions.gather(1, reveal_local)
            reveal_tokens = pred_ids.gather(1, reveal_local)
            ids.scatter_(1, reveal_positions, reveal_tokens)
            response_masked.scatter_(1, reveal_positions - prompt_len, False)

            n_remaining = n_active - k
            if n_remaining > 0:
                remaining_conf = select_conf.scatter(
                    1,
                    reveal_local,
                    torch.full_like(reveal_local, float("inf"), dtype=select_conf.dtype),
                )
                keep_local = torch.topk(
                    remaining_conf, k=n_remaining, dim=1, largest=False
                ).indices
                active_positions = active_positions.gather(1, keep_local)
            else:
                active_positions = active_positions[:, :0]
            n_active = n_remaining
            n_committed += k
        elif n_reveal > 0:
            # Per-row top-k highest-confidence masked positions.
            k = min(n_reveal, gen_len)
            topk = torch.topk(select_conf, k=k, dim=1).indices  # [batch, k]
            reveal_mask = torch.zeros_like(response_masked)
            reveal_mask.scatter_(1, topk, True)
            # Do not "reveal" positions that were already committed (-inf conf):
            reveal_mask &= response_masked
            # Commit predicted tokens at revealed positions.
            new_resp = ids[:, prompt_len:].clone()
            new_resp = torch.where(reveal_mask, pred_ids, new_resp)
            ids[:, prompt_len:] = new_resp
            response_masked &= ~reveal_mask
            n_committed += min(k, gen_len - n_committed)

        # Optional low-confidence remasking (skip on the final step).
        if remask and remask_fraction > 0 and step < num_steps - 1:
            committed = ~response_masked  # [batch, gen_len]
            n_remask = int(math.floor(remask_fraction * n_committed))
            if n_remask > 0:
                # Lowest confidence among committed positions -> re-mask.
                remask_conf = confidence.masked_fill(~committed, float("inf"))
                bottomk = torch.topk(
                    remask_conf, k=min(n_remask, gen_len), dim=1, largest=False
                ).indices
                remask_sel = torch.zeros_like(response_masked)
                remask_sel.scatter_(1, bottomk, True)
                remask_sel &= committed
                ids[:, prompt_len:] = torch.where(
                    remask_sel,
                    torch.full_like(ids[:, prompt_len:], mask_token_id),
                    ids[:, prompt_len:],
                )
                response_masked |= remask_sel
                n_committed -= min(n_remask, n_committed)

        if not remask and n_committed == gen_len:
            break

    return ids[:, prompt_len:]
