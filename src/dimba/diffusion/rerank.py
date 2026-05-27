"""Best-of-K self-reranking of parallel diffusion samples for DIMBA.

Non-autoregressive diffusion LMs generate every token in parallel, so a single
sample is often locally inconsistent. A cheap and broadly applicable remedy is
**best-of-K**: draw ``K`` independent candidates and keep the one the model
itself scores highest. This module provides the scoring/selection plumbing,
kept deliberately **model-agnostic** (callables + plain tensors) so it composes
with the concurrently-refactored core model, the continuous sampler
(:func:`dimba.diffusion.sampling.sample_from_model`), and the masked sampler
(:func:`dimba.diffusion.masked_sampling.masked_diffusion_sample`).

Three pieces
------------
* :func:`rerank_candidates` -- given candidates and a ``score_fn`` returning a
  per-candidate scalar (**higher is better**), return the best candidate (and,
  optionally, all scores).
* :func:`diffusion_elbo_score` -- a self-supervised, *training-free* quality
  score for a token sequence under a DIMBA-style continuous (latent) diffusion
  model. It is a negative Monte-Carlo estimate of the denoising / reconstruction
  error (a proxy for the diffusion ELBO): sample a few timesteps, add noise to
  the clean signal, denoise, and measure how well the model reconstructs it.
  **Higher (less negative) = lower error = better.**
* :func:`best_of_k` -- glue: call a ``generate_fn`` ``k`` times and return the
  best candidate under a ``score_fn``.

Why an ELBO/denoising-error score (and its bias)
-------------------------------------------------
For continuous Gaussian diffusion with a clean signal :math:`x_0`, the variational
bound decomposes into per-timestep denoising terms. With the **predict-**:math:`x_0`
parameterization that DIMBA uses (see
:class:`dimba.diffusion.corruption.GaussianEmbeddingCorruption` and
``DIMBA.forward``), each term is, up to an SNR-dependent weight, a
mean-squared reconstruction error

.. math::
    \\mathcal L_t = \\mathbb E_{\\varepsilon}\\big[\\, w(t)\\,
    \\lVert \\hat x_0(x_t, t) - x_0 \\rVert^2 \\,\\big],
    \\qquad x_t = \\sqrt{\\bar\\alpha_t}\\,x_0 + \\sqrt{1-\\bar\\alpha_t}\\,\\varepsilon.

:func:`diffusion_elbo_score` returns the **negative** of a Monte-Carlo average of
such per-timestep errors and is therefore a *score* (higher is better). It is an
intentionally cheap proxy, **not** the exact ELBO/NELBO, and the user should be
aware of three sources of bias/variance:

* **Weighting bias.** We default to an *unweighted* mean of squared errors
  (``weighting="uniform"``). The true bound uses an SNR-dependent weight; pass
  ``weighting="snr"`` (weight ``1 / (1 - acp_t)``, the predict-:math:`x_0` MSE
  coefficient up to a constant) to approximate it more faithfully. Neither equals
  the exact bound's constant, but for *ranking* candidates only relative scores
  matter.
* **Monte-Carlo variance.** With ``num_mc`` timestep/noise draws the estimate is
  unbiased *for the chosen weighting* but noisy; reuse a fixed ``generator`` and
  the same timesteps across candidates (``shared_timesteps=True``, the default)
  to make comparisons paired and low-variance.
* **Latent vs. token error.** The score measures reconstruction error in
  whatever space ``model_forward`` operates (latent or embedding). It does **not**
  include the discrete decoding/rounding term, so it can prefer a sequence whose
  embeddings denoise cleanly even if argmax-decoding would pick a different token.
  For masked/discrete models, score with a likelihood-based ``score_fn`` instead
  (see :func:`sequence_logprob_score`).

All math uses ``black`` line-length 100 and Google-style docstrings.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch

# A candidate is any object the caller understands (commonly a [seq] or
# [batch, seq] long tensor of token ids, but it can be a string, a tuple, etc.).
Candidate = object

# score_fn(candidate) -> scalar score, higher is better.
ScoreFn = Callable[[Candidate], Union[float, torch.Tensor]]

# generate_fn() -> candidate. Called once per requested sample.
GenerateFn = Callable[[], Candidate]


def _as_float(score: Union[float, torch.Tensor]) -> float:
    """Coerce a scalar score (Python number or 0-/1-element tensor) to ``float``.

    Args:
        score: A Python number, or a tensor containing exactly one element.

    Returns:
        The score as a Python ``float``.

    Raises:
        ValueError: If ``score`` is a tensor with more than one element.
    """
    if isinstance(score, torch.Tensor):
        if score.numel() != 1:
            raise ValueError(
                f"score_fn must return a scalar; got tensor with {score.numel()} elements."
            )
        return float(score.detach().reshape(()).item())
    return float(score)


def rerank_candidates(
    candidates: Sequence[Candidate],
    score_fn: ScoreFn,
    *,
    return_scores: bool = False,
) -> Union[Candidate, Tuple[Candidate, List[float]]]:
    """Return the single best candidate under ``score_fn`` (**higher is better**).

    Ties are broken by the lowest index (stable ``argmax``), so the result is
    deterministic given deterministic scores.

    Args:
        candidates: A non-empty sequence of candidate objects. They are treated
            opaquely; only ``score_fn`` interprets them.
        score_fn: Callable mapping a candidate to a scalar score (Python number
            or single-element tensor). Larger scores are better. To rank by an
            *error* or *loss* (lower better), negate it inside ``score_fn`` (or
            use :func:`diffusion_elbo_score`, which already returns a negative
            error so that higher is better).
        return_scores: If ``True``, also return the list of per-candidate scores
            (as ``float`` s, in input order).

    Returns:
        The best candidate, or ``(best_candidate, scores)`` if
        ``return_scores=True``.

    Raises:
        ValueError: If ``candidates`` is empty.
    """
    candidates = list(candidates)
    if not candidates:
        raise ValueError("rerank_candidates requires at least one candidate.")

    scores = [_as_float(score_fn(c)) for c in candidates]
    # Stable argmax: first index achieving the maximum.
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    best = candidates[best_idx]

    if return_scores:
        return best, scores
    return best


# model_forward(input_ids, t) contract: see diffusion_elbo_score docstring.
# Returns either (x0_pred, x0_target) [both float tensors of equal shape] or a
# single scalar per-draw MSE tensor.
ModelForward = Callable[
    [torch.Tensor, torch.Tensor],
    Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
]


def diffusion_elbo_score(
    model_forward: ModelForward,
    input_ids: torch.Tensor,
    schedule_alphas_cumprod: torch.Tensor,
    *,
    num_mc: int = 8,
    weighting: str = "uniform",
    t_min: int = 0,
    t_max: Optional[int] = None,
    shared_timesteps: bool = True,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    r"""Negative Monte-Carlo denoising-error score (an ELBO proxy) for a sequence.

    This is a *training-free, self-supervised* quality signal for a continuous
    (latent) diffusion model in the predict-:math:`x_0` parameterization (DIMBA's
    setup). For ``num_mc`` random timesteps it (1) noises the clean signal with
    the schedule's forward process, (2) denoises with ``model_forward``, and
    (3) accumulates the (optionally SNR-weighted) mean-squared reconstruction
    error. The returned score is the **negative** of that average, so **higher is
    better** and it can be used directly with :func:`rerank_candidates`.

    The ``model_forward`` contract
    ------------------------------
    ``model_forward(input_ids, t)`` is called with ``input_ids`` exactly as passed
    here and a ``t`` tensor of timestep indices, and must return **one** of:

    * ``(x0_pred, x0_target)``: two float tensors of identical shape
      ``[..., dim]`` -- the model's predicted clean signal and the clean target
      it was reconstructing (latent or embedding space; this function does not
      care which). The per-draw error is ``mean((x0_pred - x0_target) ** 2)``.
      The callable is responsible for embedding ``input_ids``, adding noise at
      ``t`` (e.g. via ``schedule.add_noise``), and denoising. This is the
      recommended contract because the *callable* owns the noising, so it can use
      the model's own embedding table and latent encoder.
    * a single scalar tensor: a precomputed per-draw MSE (or any non-negative
      error). Use this when the caller would rather compute the error itself.

    The ``t`` passed to ``model_forward`` has shape ``[B]`` if ``input_ids`` is
    ``[B, seq]`` (one timestep per batch row) and shape ``[1]`` if ``input_ids``
    is 1-D ``[seq]``. Timesteps are integer indices in
    ``[t_min, t_max)`` suitable for indexing ``schedule_alphas_cumprod``.

    Args:
        model_forward: Callable implementing the contract above. It must not
            require gradients; this function runs under ``torch.no_grad``.
        input_ids: Token ids for the candidate, shape ``[seq]`` or ``[B, seq]``
            (long). Passed through unchanged to ``model_forward``.
        schedule_alphas_cumprod: 1-D tensor :math:`\bar\alpha_t` of shape ``[T]``
            (e.g. ``model.get_alphas_cumprod()``), used only to determine the
            valid timestep range ``[0, T)`` and the SNR weight. It is **not** used
            to noise anything (the callable owns noising), so its device/dtype are
            irrelevant beyond providing ``T`` and the weights.
        num_mc: Number of Monte-Carlo timestep/noise draws. More draws reduce
            variance at linear cost. Defaults to ``8``.
        weighting: ``"uniform"`` (default) averages the raw per-draw MSE;
            ``"snr"`` weights draw ``i`` by ``1 / (1 - acp_{t_i})`` (the
            predict-:math:`x_0` MSE coefficient, up to a constant) to better track
            the variational bound. Unknown values raise ``ValueError``.
        t_min: Inclusive lower bound on sampled timestep indices (default ``0``).
        t_max: Exclusive upper bound on sampled timestep indices; defaults to
            ``T = len(schedule_alphas_cumprod)``. Restricting the range (e.g. to
            mid/low-noise steps) often gives a more discriminative score.
        shared_timesteps: If ``True`` (default), the *same* sampled timesteps are
            reused across calls **with the same** ``generator`` state, which makes
            best-of-K comparisons paired (lower-variance). Set ``False`` for fully
            independent draws each call.
        generator: Optional ``torch.Generator`` for reproducible timestep
            sampling. Pass one shared generator across all candidates for paired
            comparisons. Reseeding before each candidate guarantees identical
            timesteps regardless of ``shared_timesteps``.

    Returns:
        A scalar tensor: the **negative** mean (weighted) reconstruction error.
        Higher is better. Finite for finite model outputs.

    Raises:
        ValueError: If ``schedule_alphas_cumprod`` is not 1-D, ``num_mc < 1``, the
            timestep range is empty, ``weighting`` is unknown, or ``model_forward``
            returns an unexpected type/shape.

    Note:
        This is a *proxy*, not the exact NELBO. See the module docstring for the
        weighting, Monte-Carlo, and latent-vs-token biases. For masked/discrete
        models use a likelihood score (:func:`sequence_logprob_score`) instead.
    """
    if schedule_alphas_cumprod.dim() != 1:
        raise ValueError("schedule_alphas_cumprod must be a 1D tensor of shape [T].")
    if num_mc < 1:
        raise ValueError("num_mc must be >= 1.")
    if weighting not in ("uniform", "snr"):
        raise ValueError(f"Unknown weighting {weighting!r}; expected 'uniform' or 'snr'.")

    num_steps = int(schedule_alphas_cumprod.shape[0])
    hi = num_steps if t_max is None else int(t_max)
    lo = int(t_min)
    if not (0 <= lo < hi <= num_steps):
        raise ValueError(
            f"Invalid timestep range [t_min, t_max) = [{lo}, {hi}) for T={num_steps}."
        )

    device = input_ids.device
    # One timestep per batch row (or a single row for 1-D input).
    batch = input_ids.shape[0] if input_ids.dim() >= 2 else 1

    # Reproducible, optionally-paired timestep sampling. When a generator is
    # provided and shared_timesteps is True, reseed it to a fixed value so every
    # candidate sees identical timesteps (paired comparison); otherwise advance
    # the generator so draws are independent across calls.
    gen = generator
    if gen is not None and shared_timesteps:
        gen.manual_seed(0)

    total = input_ids.new_zeros((), dtype=torch.float32)
    weight_sum = input_ids.new_zeros((), dtype=torch.float32)

    acp = schedule_alphas_cumprod.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        for _ in range(num_mc):
            t = torch.randint(lo, hi, (batch,), device=device, generator=gen)

            out = model_forward(input_ids, t)

            if isinstance(out, tuple):
                if len(out) != 2:
                    raise ValueError(
                        "model_forward returning a tuple must return exactly "
                        "(x0_pred, x0_target)."
                    )
                x0_pred, x0_target = out
                if x0_pred.shape != x0_target.shape:
                    raise ValueError(
                        "x0_pred and x0_target must have the same shape; got "
                        f"{tuple(x0_pred.shape)} vs {tuple(x0_target.shape)}."
                    )
                mse = ((x0_pred - x0_target) ** 2).mean().to(torch.float32)
            elif isinstance(out, torch.Tensor):
                if out.numel() != 1:
                    raise ValueError(
                        "model_forward returning a single tensor must return a "
                        f"scalar MSE; got tensor with {out.numel()} elements."
                    )
                mse = out.reshape(()).to(torch.float32)
            else:
                raise ValueError(
                    "model_forward must return (x0_pred, x0_target) or a scalar "
                    f"MSE tensor; got {type(out)!r}."
                )

            if weighting == "snr":
                # Predict-x0 MSE coefficient is proportional to 1 / (1 - acp_t).
                # Use the mean over the batch's timesteps for a single scalar weight.
                one_minus = (1.0 - acp[t]).clamp(min=1e-8)
                w = (1.0 / one_minus).mean()
            else:
                w = total.new_ones(())

            total = total + w * mse
            weight_sum = weight_sum + w

    mean_error = total / weight_sum.clamp(min=1e-8)
    # Negate so that LOWER error -> HIGHER score (rerank picks the max).
    return -mean_error


def sequence_logprob_score(
    logprob_fn: Callable[[Candidate], Union[float, torch.Tensor]],
    candidate: Candidate,
) -> torch.Tensor:
    """Thin adapter exposing a (higher-is-better) log-probability as a score.

    For masked/discrete DIMBA (``AbsorbingMaskCorruption`` head) the natural
    self-score is the model's (pseudo-)log-likelihood of the committed sequence,
    which is already "higher is better". This helper merely coerces it to a scalar
    tensor so it drops into :func:`rerank_candidates` like the ELBO score.

    Args:
        logprob_fn: Callable mapping a candidate to its scalar log-probability
            (Python number or single-element tensor). Higher is better.
        candidate: The candidate to score.

    Returns:
        A scalar tensor equal to ``logprob_fn(candidate)`` (higher is better).
    """
    return torch.as_tensor(_as_float(logprob_fn(candidate)), dtype=torch.float32)


def best_of_k(
    generate_fn: GenerateFn,
    score_fn: ScoreFn,
    k: int,
    *,
    return_all: bool = False,
) -> Union[Candidate, Tuple[Candidate, List[Candidate], List[float]]]:
    """Generate ``k`` candidates and return the best one under ``score_fn``.

    This is the high-level entry point for inference: it draws ``k`` independent
    samples (e.g. ``k`` runs of
    :func:`dimba.diffusion.sampling.sample_from_model` with different seeds) and
    keeps the highest-scoring one. The score is typically
    :func:`diffusion_elbo_score` (continuous DIMBA) or a log-likelihood
    (:func:`sequence_logprob_score`, masked DIMBA).

    Args:
        generate_fn: Zero-argument callable returning one fresh candidate per
            call. Make it stochastic (different RNG state per call) so the ``k``
            candidates differ; otherwise best-of-K is a no-op.
        score_fn: Callable mapping a candidate to a scalar score (higher better),
            as in :func:`rerank_candidates`.
        k: Number of candidates to generate; must be ``>= 1``.
        return_all: If ``True``, also return the list of all generated candidates
            and their scores (useful for logging / debugging).

    Returns:
        The best candidate, or ``(best, candidates, scores)`` if
        ``return_all=True``.

    Raises:
        ValueError: If ``k < 1``.
    """
    if k < 1:
        raise ValueError("k must be >= 1.")

    candidates = [generate_fn() for _ in range(k)]
    best, scores = rerank_candidates(candidates, score_fn, return_scores=True)

    if return_all:
        return best, candidates, scores
    return best
