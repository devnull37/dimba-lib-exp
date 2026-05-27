"""Preference-optimization objectives for DIMBA (DPO/IPO/SimPO) with diffusion surrogates.

This module implements *direct preference optimization* losses for the DIMBA
non-autoregressive diffusion language model. The central difficulty is that a
masked / continuous diffusion LM does **not** expose an exact, cheap sequence
log-likelihood ``log p(y | x)`` the way an autoregressive model does: the true
marginal requires integrating over the diffusion trajectory and is intractable.
We therefore optimize over a *variational lower bound* (ELBO) surrogate for the
sequence log-probability, following the diffusion-DPO literature.

References
    - DPO: "Direct Preference Optimization" (Rafailov et al., 2023),
      arXiv:2305.18290. Bradley-Terry preference loss expressed directly over a
      reference-anchored policy.
    - Diffusion-DPO: "Diffusion Model Alignment Using Direct Preference
      Optimization" (Wallace et al., 2023), arXiv:2311.12908. Replaces the exact
      log-likelihood in DPO with a per-timestep ELBO / denoising-error surrogate.
    - LLaDA 1.5 / VRPO: "LLaDA 1.5: Variance-Reduced Preference Optimization for
      Large Language Diffusion Models" (Zhu et al., 2025), arXiv:2505.19223.
      Motivates Monte-Carlo ELBO estimation of diffusion log-probabilities and
      antithetic-sampling variance reduction for the preference gradient.
    - IPO: "A General Theoretical Paradigm to Understand Learning from Human
      Preferences" (Azar et al., 2023). Squared-loss variant that avoids the
      Bradley-Terry over-fitting failure mode.
    - SimPO: "Simple Preference Optimization with a Reference-Free Reward"
      (Meng et al., 2024). Length-normalized, reference-free margin objective.

Design notes
    - ``sequence_logprob`` computes an *exact* autoregressive-style summed
      log-prob over masked positions. It is the right primitive when logits are
      produced from a single forward pass and you treat each response position
      as a categorical (this is how DIMBA's GRPO path already scores sequences:
      ``log_softmax`` then ``gather`` over realized tokens).
    - ``elbo_sequence_logprob`` is the diffusion-aware surrogate. For DIMBA
      (masked / mean-field decoding) it is a *one-step* ELBO: a denoising
      forward at a sampled timestep yields per-position token logits, and the
      masked summed log-prob is a one-sample estimate of the ELBO term. Average
      over several timesteps (or use :func:`antithetic_timesteps`) for a
      lower-variance Monte-Carlo estimate.

All log-probabilities are returned in *nats* and summed (not averaged) over the
masked response positions, matching the DPO derivation where the implicit reward
is ``beta * (log pi(y|x) - log pi_ref(y|x))``.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

__all__ = [
    "sequence_logprob",
    "elbo_sequence_logprob",
    "antithetic_timesteps",
    "dpo_loss",
    "ipo_loss",
    "simpo_loss",
]


def _coerce_mask(mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Coerce a label/attention mask to a float mask broadcastable over ``reference``.

    Args:
        mask: Boolean, integer, or float mask of shape ``[batch, seq_len]``.
        reference: Tensor whose dtype/device the mask should follow.

    Returns:
        Float mask of shape ``[batch, seq_len]`` with 1.0 on positions to score.
    """
    if mask.dtype != reference.dtype:
        mask = mask.to(reference.dtype)
    return mask


def sequence_logprob(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Per-sequence summed log-probability over masked (response) positions.

    Computes ``sum_t mask_t * log softmax(logits_t)[labels_t]`` for each sequence
    in the batch. Masked-out positions (``mask == 0``) contribute nothing, which
    is how we restrict the DPO/IPO/SimPO signal to *response* tokens only and
    ignore the prompt and padding.

    Args:
        logits: Unnormalized token logits ``[batch, seq_len, vocab_size]``.
        labels: Realized/target token ids ``[batch, seq_len]``. Values at masked
            positions are ignored, so they may be arbitrary (e.g. ``0`` or a
            padding id) as long as they are valid indices into ``vocab_size``.
        mask: Response mask ``[batch, seq_len]`` (bool/int/float); 1 marks a
            position that should contribute to the log-prob.

    Returns:
        Summed log-probability per sequence, shape ``[batch]`` (in nats).

    Note:
        For DIMBA this is exact only if ``logits`` already represent the model's
        token distribution at the positions being scored. Because diffusion
        log-likelihoods are intractable, prefer :func:`elbo_sequence_logprob`
        as the surrogate when the logits come from a noised denoising pass.
    """
    if logits.dim() != 3:
        raise ValueError(f"Expected logits [batch, seq, vocab], got shape {tuple(logits.shape)}.")
    if labels.shape != logits.shape[:2]:
        raise ValueError(
            f"labels shape {tuple(labels.shape)} incompatible with logits "
            f"{tuple(logits.shape[:2])}."
        )

    log_probs = F.log_softmax(logits, dim=-1)
    # Gather the log-prob of each realized token: [batch, seq_len].
    token_log_probs = torch.gather(log_probs, dim=-1, index=labels.long().unsqueeze(-1)).squeeze(-1)
    float_mask = _coerce_mask(mask, token_log_probs)
    return (token_log_probs * float_mask).sum(dim=-1)


def antithetic_timesteps(
    batch_size: int,
    num_diffusion_steps: int,
    *,
    device: Optional[torch.device] = None,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample diffusion timesteps together with their antithetic partners (VRPO).

    Antithetic sampling is a classic Monte-Carlo variance-reduction technique:
    instead of drawing two independent timesteps to estimate the ELBO, draw one
    timestep ``t`` and pair it with its "mirror" ``T - 1 - t``. Because the
    denoising error is (approximately) monotone in the noise level, ``t`` and its
    partner are *negatively correlated*. The average of two negatively-correlated
    estimators has lower variance than the average of two independent ones:

        ``Var((A + B) / 2) = (Var(A) + Var(B) + 2 Cov(A, B)) / 4``

    so a negative ``Cov(A, B)`` shrinks the variance of the ELBO estimate, and
    therefore the variance of the DPO gradient. This mirrors the variance-reduced
    preference optimization recipe of LLaDA 1.5 / VRPO (arXiv:2505.19223), which
    couples the timestep draws used to score chosen and rejected completions.

    Args:
        batch_size: Number of timestep pairs to draw.
        num_diffusion_steps: Total diffusion steps ``T`` (timesteps in ``[0, T)``).
        device: Device for the returned tensors.
        generator: Optional RNG for reproducible draws.

    Returns:
        Tuple ``(t, t_antithetic)``, each of shape ``[batch_size]`` and dtype
        ``long``, with ``t_antithetic = (T - 1) - t``.
    """
    if num_diffusion_steps <= 0:
        raise ValueError("num_diffusion_steps must be > 0.")
    t = torch.randint(
        low=0,
        high=num_diffusion_steps,
        size=(batch_size,),
        device=device,
        generator=generator,
        dtype=torch.long,
    )
    t_antithetic = (num_diffusion_steps - 1) - t
    return t, t_antithetic


def elbo_sequence_logprob(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    *,
    timesteps: Optional[torch.Tensor] = None,
    num_mc_samples: int = 1,
    antithetic: bool = False,
    logits_fn: Optional[Callable[[torch.nn.Module, torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Diffusion-aware ELBO surrogate for the per-sequence response log-prob.

    Exact ``log p(y | x)`` for a diffusion LM requires marginalizing over the
    full denoising trajectory and is intractable. Following Diffusion-DPO
    (arXiv:2311.12908) and VRPO/LLaDA 1.5 (arXiv:2505.19223), we substitute a
    variational lower bound (ELBO) estimated by Monte-Carlo over timesteps.

    For DIMBA, which performs *masked / mean-field* denoising, we use a
    **one-step ELBO surrogate**: at a sampled timestep ``t`` the model runs a
    denoising forward (conditioned on the prompt embedded in ``input_ids``) and
    emits per-position token logits; the masked summed log-prob of the realized
    tokens is a single-sample estimate of the relevant ELBO term. We average
    over ``num_mc_samples`` timesteps to reduce variance.

    Per-timestep Monte-Carlo note (continuous / score-based diffusion):
        For a continuous diffusion model the ELBO has the integral form
        ``E_{t ~ U(0,T)}[ w(t) * || eps_theta(x_t, t) - eps ||^2 ]`` (up to a
        constant). One would estimate it by sampling ``t`` and the injected noise
        ``eps``, computing the reweighted denoising MSE, and averaging over MC
        samples; the negated, reweighted error then plays the role of
        ``log p(y | x)`` inside the DPO objective. The discrete/categorical
        surrogate used here (token-level ``log_softmax`` after a noised forward)
        is the masked-diffusion analogue of that construction.

    Args:
        model: A DIMBA-like module. The default ``logits_fn`` expects
            ``model.forward(input_ids, t, return_latent_info=True) ->
            (x_pred, ...)`` plus ``model.output_head`` and
            ``model.token_embed.get_weight()`` (matching the existing GRPO path),
            and ``model.num_diffusion_steps``.
        input_ids: Full sequence token ids ``[batch, seq_len]`` (prompt + response).
        labels: Realized response token ids ``[batch, seq_len]`` to score.
        mask: Response mask ``[batch, seq_len]``; 1 on positions to score.
        timesteps: Optional explicit timesteps ``[batch]``. When ``None`` they are
            sampled uniformly (optionally antithetically) per MC sample.
        num_mc_samples: Number of timestep samples to average the ELBO over.
        antithetic: If ``True`` and ``num_mc_samples`` is even, draw timesteps in
            antithetic pairs via :func:`antithetic_timesteps` for variance
            reduction. Ignored when ``timesteps`` is provided.
        logits_fn: Optional override ``(model, input_ids, t) -> logits`` so callers
            can plug in a custom diffusion-conditioned forward without depending
            on DIMBA internals (useful for tests with tiny stub modules).
        generator: Optional RNG for reproducible timestep sampling.

    Returns:
        ELBO-surrogate summed log-prob per sequence, shape ``[batch]`` (nats).
        Gradients flow through ``model`` so this can be used directly inside the
        DPO/IPO/SimPO losses.
    """
    if num_mc_samples < 1:
        raise ValueError("num_mc_samples must be >= 1.")

    batch_size = input_ids.shape[0]
    device = input_ids.device

    if logits_fn is None:
        logits_fn = _default_diffusion_logits_fn

    num_steps = int(getattr(model, "num_diffusion_steps", 1000))

    if timesteps is not None:
        timestep_draws = [timesteps.to(device=device, dtype=torch.long)]
    elif antithetic and num_mc_samples % 2 == 0:
        timestep_draws = []
        for _ in range(num_mc_samples // 2):
            t, t_anti = antithetic_timesteps(
                batch_size, num_steps, device=device, generator=generator
            )
            timestep_draws.append(t)
            timestep_draws.append(t_anti)
    else:
        timestep_draws = [
            torch.randint(
                low=0,
                high=num_steps,
                size=(batch_size,),
                device=device,
                generator=generator,
                dtype=torch.long,
            )
            for _ in range(num_mc_samples)
        ]

    estimates = []
    for t in timestep_draws:
        logits = logits_fn(model, input_ids, t)
        estimates.append(sequence_logprob(logits, labels, mask))

    # Monte-Carlo average over timesteps: mean of stacked [batch] estimates.
    return torch.stack(estimates, dim=0).mean(dim=0)


def _default_diffusion_logits_fn(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    """Default DIMBA forward producing token logits at the given timesteps.

    Mirrors the existing GRPO scoring path: a denoising forward at timestep ``t``
    followed by the (optionally weight-tied) output head.

    Args:
        model: DIMBA-like module.
        input_ids: Token ids ``[batch, seq_len]``.
        timesteps: Diffusion timesteps ``[batch]``.

    Returns:
        Token logits ``[batch, seq_len, vocab_size]``.
    """
    x_pred, _, _ = model(input_ids, timesteps, return_latent_info=True)
    embedding_weight = model.token_embed.get_weight()
    return model.output_head(x_pred, embedding_weight=embedding_weight)


def dpo_loss(
    pi_chosen_lp: torch.Tensor,
    pi_rejected_lp: torch.Tensor,
    ref_chosen_lp: torch.Tensor,
    ref_rejected_lp: torch.Tensor,
    beta: float = 0.1,
    *,
    label_smoothing: float = 0.0,
    reduction: str = "mean",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Standard Bradley-Terry DPO loss with reference anchoring.

    Implements the DPO objective (Rafailov et al., 2023, arXiv:2305.18290):

        ``L = -E[ log sigmoid( beta * ((pi_c - ref_c) - (pi_r - ref_r)) ) ]``

    where ``pi_*`` / ``ref_*`` are summed response log-probs under the policy and
    frozen reference, and ``beta`` controls the KL penalty implied by the closed-
    form optimal policy. For DIMBA all four log-probs are ELBO surrogates from
    :func:`elbo_sequence_logprob` (diffusion log-likelihoods are intractable, per
    Diffusion-DPO arXiv:2311.12908 / VRPO arXiv:2505.19223).

    Optional ``label_smoothing`` implements the conservative (cDPO) variant,
    interpolating toward assuming the preference label is flipped with small
    probability.

    Args:
        pi_chosen_lp: Policy summed log-prob of chosen response ``[batch]``.
        pi_rejected_lp: Policy summed log-prob of rejected response ``[batch]``.
        ref_chosen_lp: Reference summed log-prob of chosen response ``[batch]``.
        ref_rejected_lp: Reference summed log-prob of rejected response ``[batch]``.
        beta: Inverse temperature / KL strength (default: 0.1).
        label_smoothing: cDPO label-smoothing in ``[0, 0.5)`` (default: 0.0).
        reduction: ``"mean"``, ``"sum"``, or ``"none"``.

    Returns:
        Tuple ``(loss, chosen_reward, rejected_reward)`` where the implicit
        rewards are ``beta * (pi_* - ref_*)`` detached for logging, shape
        ``[batch]`` (or scalar loss after reduction).
    """
    pi_logratios = pi_chosen_lp - pi_rejected_lp
    ref_logratios = ref_chosen_lp - ref_rejected_lp
    logits = beta * (pi_logratios - ref_logratios)

    if label_smoothing > 0.0:
        # Conservative DPO: assume label is correct w.p. (1 - eps).
        per_example = (
            -F.logsigmoid(logits) * (1.0 - label_smoothing)
            - F.logsigmoid(-logits) * label_smoothing
        )
    else:
        per_example = -F.logsigmoid(logits)

    chosen_reward = (beta * (pi_chosen_lp - ref_chosen_lp)).detach()
    rejected_reward = (beta * (pi_rejected_lp - ref_rejected_lp)).detach()

    loss = _reduce(per_example, reduction)
    return loss, chosen_reward, rejected_reward


def ipo_loss(
    pi_chosen_lp: torch.Tensor,
    pi_rejected_lp: torch.Tensor,
    ref_chosen_lp: torch.Tensor,
    ref_rejected_lp: torch.Tensor,
    beta: float = 0.1,
    *,
    reduction: str = "mean",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """IPO (Identity Preference Optimization) loss.

    From Azar et al. (2023). Replaces the Bradley-Terry log-sigmoid with a
    squared loss that regresses the policy-vs-reference log-ratio difference
    toward a fixed margin ``1 / (2 * beta)``:

        ``L = E[ ( (pi_c - ref_c) - (pi_r - ref_r) - 1/(2*beta) )^2 ]``

    This avoids DPO's tendency to drive the implicit reward gap to infinity (and
    thus over-fit / collapse) when preferences are deterministic.

    Args:
        pi_chosen_lp: Policy summed log-prob of chosen response ``[batch]``.
        pi_rejected_lp: Policy summed log-prob of rejected response ``[batch]``.
        ref_chosen_lp: Reference summed log-prob of chosen response ``[batch]``.
        ref_rejected_lp: Reference summed log-prob of rejected response ``[batch]``.
        beta: Controls the target margin ``1/(2*beta)`` (default: 0.1).
        reduction: ``"mean"``, ``"sum"``, or ``"none"``.

    Returns:
        Tuple ``(loss, chosen_reward, rejected_reward)`` with detached implicit
        rewards ``beta * (pi_* - ref_*)``.
    """
    pi_logratios = pi_chosen_lp - pi_rejected_lp
    ref_logratios = ref_chosen_lp - ref_rejected_lp
    margin = pi_logratios - ref_logratios
    per_example = (margin - 1.0 / (2.0 * beta)) ** 2

    chosen_reward = (beta * (pi_chosen_lp - ref_chosen_lp)).detach()
    rejected_reward = (beta * (pi_rejected_lp - ref_rejected_lp)).detach()

    loss = _reduce(per_example, reduction)
    return loss, chosen_reward, rejected_reward


def simpo_loss(
    pi_chosen_lp: torch.Tensor,
    pi_rejected_lp: torch.Tensor,
    chosen_lengths: torch.Tensor,
    rejected_lengths: torch.Tensor,
    beta: float = 2.0,
    gamma: float = 1.0,
    *,
    reduction: str = "mean",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """SimPO (reference-free, length-normalized) preference loss.

    From Meng et al. (2024). SimPO removes the reference model entirely and uses
    a length-normalized average log-prob as an implicit reward, with a target
    reward margin ``gamma``:

        ``r(y) = (beta / |y|) * sum_t log pi(y_t | ...)``
        ``L = -E[ log sigmoid( r(chosen) - r(rejected) - gamma ) ]``

    Because there is no reference term, only the *policy* log-probs are needed,
    which is convenient for DIMBA where computing reference ELBO surrogates
    doubles the forward cost. Length normalization counteracts the diffusion
    decoder's bias toward longer or shorter completions.

    Args:
        pi_chosen_lp: Policy summed log-prob of chosen response ``[batch]``.
        pi_rejected_lp: Policy summed log-prob of rejected response ``[batch]``.
        chosen_lengths: Number of scored chosen tokens per example ``[batch]``
            (the sum of the chosen response mask). Used for length normalization.
        rejected_lengths: Number of scored rejected tokens per example ``[batch]``.
        beta: Reward scaling (default: 2.0, per the SimPO paper's typical range).
        gamma: Target reward margin subtracted before the sigmoid (default: 1.0).
        reduction: ``"mean"``, ``"sum"``, or ``"none"``.

    Returns:
        Tuple ``(loss, chosen_reward, rejected_reward)`` with detached
        length-normalized implicit rewards.
    """
    chosen_len = chosen_lengths.to(pi_chosen_lp.dtype).clamp(min=1.0)
    rejected_len = rejected_lengths.to(pi_rejected_lp.dtype).clamp(min=1.0)

    chosen_reward = beta * (pi_chosen_lp / chosen_len)
    rejected_reward = beta * (pi_rejected_lp / rejected_len)

    per_example = -F.logsigmoid(chosen_reward - rejected_reward - gamma)

    loss = _reduce(per_example, reduction)
    return loss, chosen_reward.detach(), rejected_reward.detach()


def _reduce(values: torch.Tensor, reduction: str) -> torch.Tensor:
    """Apply a reduction to a per-example tensor.

    Args:
        values: Per-example values ``[batch]``.
        reduction: One of ``"mean"``, ``"sum"``, ``"none"``.

    Returns:
        Reduced scalar tensor, or the unchanged tensor for ``"none"``.
    """
    if reduction == "mean":
        return values.mean()
    if reduction == "sum":
        return values.sum()
    if reduction == "none":
        return values
    raise ValueError(f"Unknown reduction '{reduction}'. Use 'mean', 'sum', or 'none'.")
