"""Corruption (forward) processes for DIMBA diffusion.

This module defines a small, model-agnostic abstraction for the *forward*
("corruption") process of a diffusion language model, together with three
concrete implementations:

* :class:`GaussianEmbeddingCorruption` -- continuous Gaussian diffusion over
  token embeddings, mirroring the existing DIMBA model (predict-``x0``). This is
  the classic Diffusion-LM / continuous-latent recipe.
* :class:`AbsorbingMaskCorruption` -- discrete *masked* (absorbing-state)
  diffusion, i.e. the MDLM / LLaDA recipe that scales for text
  (MDLM, arXiv:2406.07524; LLaDA, arXiv:2502.09992).
* :class:`HybridCorruption` -- **novel, experimental** per-token mixture of the
  two above: each token is either replaced by ``[MASK]`` (discrete) or has its
  embedding perturbed with Gaussian noise (continuous). This forms a continuum
  between Diffusion-LM and MDLM.

Design notes
------------
The classes here deliberately depend only on *plain tensors and callables* so
they can be wired into the (concurrently refactored) core model without coupling
to its exact signatures. In particular:

* Embedding lookups are passed as a callable ``embed_fn(ids) -> Tensor`` rather
  than a concrete ``nn.Module``.
* Model predictions are passed into :meth:`CorruptionProcess.loss` as plain
  tensors (either predicted ``x0`` embeddings or vocabulary ``logits``), so the
  loss never calls back into the model.

All math uses ``black`` line-length 100 and Google-style docstrings.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

# An info dictionary carries everything ``loss`` needs about a corruption draw.
InfoDict = Dict[str, torch.Tensor]


class CorruptionProcess(ABC):
    """Abstract forward (corruption) process for a diffusion language model.

    A corruption process defines how a *clean* example is degraded at a sampled
    timestep, and how the corresponding training objective is computed from a
    model prediction. Concrete subclasses may operate on either continuous token
    embeddings (Gaussian) or discrete token ids (absorbing mask), or a mixture.

    Contract
    --------
    Subclasses must implement three methods:

    ``sample_timesteps(batch, device) -> Tensor``
        Draw one timestep per batch element. The *type and range* of timesteps
        is process-specific: continuous Gaussian diffusion uses integer indices
        in ``[0, T)`` (to match the precomputed schedule buffers), while masked
        diffusion uses continuous ``t`` in ``(0, 1]``. Each subclass documents
        its own convention.

    ``corrupt(x, t, ...) -> (corrupted, info)``
        Apply the forward process to ``x`` at timestep ``t``. Returns the
        corrupted tensor (what the model consumes) and an ``info`` dict holding
        the targets and any quantities required to weight the loss. The exact
        keys are documented per subclass; ``loss`` consumes them.

    ``loss(prediction, info, ...) -> Tensor``
        Compute the (scalar) training objective from a *model prediction* and
        the ``info`` returned by ``corrupt``. ``prediction`` is a plain tensor
        whose meaning is process-specific (predicted ``x0`` embeddings for the
        Gaussian process; vocabulary logits for the masked process). The method
        never calls back into the model.

    These three calls are designed to be used together in a training step::

        t = process.sample_timesteps(batch, device)
        corrupted, info = process.corrupt(clean, t)
        prediction = model(corrupted, t, ...)   # model-specific
        loss = process.loss(prediction, info)
    """

    @abstractmethod
    def sample_timesteps(self, batch: int, device: torch.device) -> torch.Tensor:
        """Sample one timestep per batch element.

        Args:
            batch: Number of independent examples in the batch.
            device: Device on which to allocate the returned tensor.

        Returns:
            A tensor of shape ``[batch]``. Dtype and value range are
            process-specific (see subclass docstrings).
        """
        raise NotImplementedError

    @abstractmethod
    def corrupt(
        self, x: torch.Tensor, t: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, InfoDict]:
        """Apply the forward corruption process at timestep ``t``.

        Args:
            x: The clean input. Its meaning is process-specific (embeddings for
                the Gaussian process, token ids for the masked process).
            t: Timesteps of shape ``[batch]`` as produced by
                :meth:`sample_timesteps`.
            **kwargs: Optional process-specific arguments (e.g. a fixed
                ``noise`` tensor).

        Returns:
            A tuple ``(corrupted, info)`` where ``corrupted`` is what the model
            consumes and ``info`` carries targets / weighting metadata for
            :meth:`loss`.
        """
        raise NotImplementedError

    @abstractmethod
    def loss(self, prediction: torch.Tensor, info: InfoDict, **kwargs) -> torch.Tensor:
        """Compute the scalar training objective.

        Args:
            prediction: A model prediction whose semantics are process-specific.
            info: The ``info`` dict returned by :meth:`corrupt`.
            **kwargs: Optional process-specific arguments.

        Returns:
            A scalar loss tensor.
        """
        raise NotImplementedError


def _broadcast_to(coef: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Reshape a per-batch coefficient ``[batch]`` to broadcast against ``x``.

    Args:
        coef: Per-batch coefficients of shape ``[batch]``.
        x: Reference tensor of shape ``[batch, ...]``.

    Returns:
        ``coef`` viewed as ``[batch, 1, 1, ...]`` so it broadcasts over ``x``.
    """
    return coef.view(coef.shape[0], *([1] * (x.dim() - 1)))


# ---------------------------------------------------------------------------
# Continuous Gaussian embedding diffusion (Diffusion-LM style; predict-x0).
# ---------------------------------------------------------------------------


class GaussianEmbeddingCorruption(CorruptionProcess):
    r"""Continuous Gaussian diffusion over token embeddings (predict-``x0``).

    Mirrors the existing DIMBA forward process
    (:class:`dimba.diffusion.schedules.CosineNoiseSchedule`):

    .. math::
        x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \varepsilon,
        \qquad \varepsilon \sim \mathcal N(0, I).

    The model is trained to predict the clean embeddings ``x0``; the loss is the
    mean-squared error between the prediction and ``x0`` with an optional
    *min-SNR-gamma* reweighting (Hang et al., 2023).

    Timestep convention:
        Integer indices in ``[0, T)`` (``T = len(alphas_cumprod)``), matching the
        precomputed schedule buffers. A continuous-time variant can be obtained
        by precomputing ``alphas_cumprod`` on a finer grid; the API is unchanged.

    Args:
        alphas_cumprod: Precomputed cumulative product schedule
            :math:`\bar\alpha_t` of shape ``[T]`` (e.g.
            ``CosineNoiseSchedule.alphas_cumprod``). Stored by reference; moved
            onto the input's device lazily inside :meth:`corrupt`.
    """

    def __init__(self, alphas_cumprod: torch.Tensor):
        if alphas_cumprod.dim() != 1:
            raise ValueError("alphas_cumprod must be a 1D tensor of shape [T].")
        self.alphas_cumprod = alphas_cumprod
        self.num_steps = int(alphas_cumprod.shape[0])

    def sample_timesteps(self, batch: int, device: torch.device) -> torch.Tensor:
        """Sample integer timesteps uniformly in ``[0, T)``.

        Args:
            batch: Number of examples.
            device: Device for the returned tensor.

        Returns:
            Long tensor of shape ``[batch]`` with values in ``[0, T)``.
        """
        return torch.randint(0, self.num_steps, (batch,), device=device)

    def corrupt(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, InfoDict]:
        r"""Forward-noise clean embeddings at timestep ``t``.

        Args:
            x: Clean embeddings ``x0`` of shape ``[batch, seq, dim]``.
            t: Long timesteps of shape ``[batch]`` with values in ``[0, T)``.
            noise: Optional fixed noise of shape ``x`` for reproducibility; if
                ``None``, standard Gaussian noise is sampled.

        Returns:
            A tuple ``(x_t, info)`` where ``x_t`` is the noised embedding and
            ``info`` has keys:

            * ``"noise"``: the noise tensor used (shape ``x``).
            * ``"x0"``: the clean embeddings (shape ``x``).
            * ``"t"``: the timesteps (shape ``[batch]``).
        """
        if noise is None:
            noise = torch.randn_like(x)

        acp = self.alphas_cumprod.to(device=x.device, dtype=x.dtype)[t]  # [batch]
        sqrt_acp = _broadcast_to(torch.sqrt(acp), x)
        sqrt_one_minus = _broadcast_to(torch.sqrt(1.0 - acp), x)

        x_t = sqrt_acp * x + sqrt_one_minus * noise
        info: InfoDict = {"noise": noise, "x0": x, "t": t}
        return x_t, info

    def loss(
        self,
        prediction: torch.Tensor,
        info: InfoDict,
        min_snr_gamma: Optional[float] = None,
    ) -> torch.Tensor:
        r"""Predict-``x0`` mean-squared error with optional min-SNR weighting.

        The base objective is :math:`\lVert \hat x_0 - x_0 \rVert^2` averaged over
        all elements. When ``min_snr_gamma`` is provided, each example's
        contribution is weighted by

        .. math::
            w(t) = \min(\mathrm{SNR}(t), \gamma),
            \qquad \mathrm{SNR}(t) = \frac{\bar\alpha_t}{1 - \bar\alpha_t},

        which is the correct min-SNR weight for the **x0-prediction**
        parameterization (Hang et al., 2023, "Efficient Diffusion Training via
        Min-SNR Weighting Strategy"). For the eps/v parameterizations the weight
        differs; we implement the x0 form because DIMBA predicts ``x0``.

        Args:
            prediction: Predicted clean embeddings ``x0_hat`` of shape ``x``.
            info: Info dict from :meth:`corrupt` (uses ``"x0"`` and, if weighting
                is requested, ``"t"``).
            min_snr_gamma: Optional truncation constant :math:`\gamma` (e.g.
                ``5.0``). ``None`` disables weighting (plain MSE).

        Returns:
            Scalar loss tensor.
        """
        x0 = info["x0"]
        # Per-element squared error, then mean over feature dims -> [batch, seq].
        sq_err = (prediction - x0) ** 2
        per_token = sq_err.mean(dim=-1)  # [batch, seq]

        if min_snr_gamma is None:
            return per_token.mean()

        t = info["t"]
        acp = self.alphas_cumprod.to(device=x0.device, dtype=x0.dtype)[t]  # [batch]
        snr = acp / (1.0 - acp).clamp(min=1e-8)
        weight = torch.clamp(snr, max=float(min_snr_gamma))  # [batch]
        weight = weight.view(weight.shape[0], *([1] * (per_token.dim() - 1)))
        weighted = per_token * weight
        # Normalize by the mean weight so the loss scale is comparable to plain MSE.
        return weighted.mean() / weight.mean().clamp(min=1e-8)


# ---------------------------------------------------------------------------
# Discrete absorbing-state (masked) diffusion -- MDLM / LLaDA recipe.
# ---------------------------------------------------------------------------


def _mask_prob(t: torch.Tensor, schedule: str) -> torch.Tensor:
    r"""Marginal masking probability ``alpha-bar`` complement at time ``t``.

    For absorbing diffusion the forward marginal keeps a token with probability
    :math:`\alpha(t)` and replaces it with ``[MASK]`` with probability
    :math:`1 - \alpha(t)`. We define the *masking* probability directly:

    * ``"linear"``:  :math:`p_{\text{mask}}(t) = t`.
    * ``"cosine"``:  :math:`p_{\text{mask}}(t) = 1 - \cos(\tfrac{\pi}{2} t)`,
      i.e. the keep-rate is :math:`\cos(\tfrac{\pi}{2} t)`. This mirrors the
      cosine schedule used elsewhere in DIMBA and masks slowly for small ``t``.

    Both satisfy ``p_mask(0)=0`` and ``p_mask(1)=1``.

    Args:
        t: Continuous timesteps in ``(0, 1]`` (any broadcastable shape).
        schedule: ``"linear"`` or ``"cosine"``.

    Returns:
        Masking probabilities with the same shape as ``t``.
    """
    if schedule == "linear":
        return t
    if schedule == "cosine":
        return 1.0 - torch.cos(0.5 * math.pi * t)
    raise ValueError(f"Unknown schedule {schedule!r}; expected 'linear' or 'cosine'.")


class AbsorbingMaskCorruption(CorruptionProcess):
    r"""Discrete masked (absorbing-state) diffusion -- the MDLM / LLaDA recipe.

    Each token is independently replaced by ``mask_token_id`` with probability
    :math:`p_{\text{mask}}(t)` (see :func:`_mask_prob`). The model receives the
    partially-masked ids and predicts a categorical distribution (logits) over
    the vocabulary at every position; the loss is a cross-entropy on the *masked*
    positions only, reweighted by the MDLM continuous-time NELBO weight.

    References:
        MDLM (Sahoo et al., 2024, arXiv:2406.07524); LLaDA (Nie et al., 2025,
        arXiv:2502.09992).

    Timestep convention:
        Continuous ``t`` in ``(0, 1]`` (``t -> 0`` is clean, ``t = 1`` fully
        masked).

    Args:
        mask_token_id: Vocabulary id of the absorbing ``[MASK]`` token.
        schedule: Masking schedule, ``"cosine"`` (default) or ``"linear"``.
    """

    def __init__(self, mask_token_id: int, schedule: str = "cosine"):
        if schedule not in ("cosine", "linear"):
            raise ValueError(f"Unknown schedule {schedule!r}.")
        self.mask_token_id = int(mask_token_id)
        self.schedule = schedule

    def sample_timesteps(self, batch: int, device: torch.device) -> torch.Tensor:
        """Sample continuous timesteps uniformly in ``(0, 1]``.

        Args:
            batch: Number of examples.
            device: Device for the returned tensor.

        Returns:
            Float tensor of shape ``[batch]`` with values in ``(0, 1]``. We
            sample in ``[eps, 1]`` (``eps = 1e-3``) to keep the ``1/t`` NELBO
            weight finite.
        """
        eps = 1e-3
        return torch.rand(batch, device=device) * (1.0 - eps) + eps

    def mask_prob(self, t: torch.Tensor) -> torch.Tensor:
        """Return the marginal masking probability for timesteps ``t``.

        Args:
            t: Continuous timesteps in ``(0, 1]`` of shape ``[batch]``.

        Returns:
            Masking probabilities of shape ``[batch]``.
        """
        return _mask_prob(t, self.schedule)

    def corrupt(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, InfoDict]:
        """Independently mask tokens with probability ``mask_prob(t)``.

        Args:
            x: Clean token ids of shape ``[batch, seq]`` (long).
            t: Continuous timesteps in ``(0, 1]`` of shape ``[batch]``.

        Returns:
            A tuple ``(masked_ids, info)`` where ``masked_ids`` has masked
            positions set to ``mask_token_id`` and ``info`` has keys:

            * ``"masked_positions"``: bool tensor ``[batch, seq]``, ``True`` where
              the token was replaced by ``[MASK]``.
            * ``"targets"``: the original ids ``[batch, seq]``.
            * ``"t"``: timesteps ``[batch]`` (used for the NELBO weight).
        """
        p = self.mask_prob(t)  # [batch]
        p = p.view(p.shape[0], *([1] * (x.dim() - 1)))  # broadcast over seq
        rand = torch.rand(x.shape, device=x.device)
        masked_positions = rand < p  # [batch, seq] bool

        masked_ids = torch.where(
            masked_positions,
            torch.full_like(x, self.mask_token_id),
            x,
        )
        info: InfoDict = {
            "masked_positions": masked_positions,
            "targets": x,
            "t": t,
        }
        return masked_ids, info

    def loss(self, prediction: torch.Tensor, info: InfoDict) -> torch.Tensor:
        r"""Masked cross-entropy with the MDLM continuous-time NELBO weight.

        Only masked positions contribute. For the absorbing diffusion with the
        forward marginal of :func:`_mask_prob`, the continuous-time NELBO reduces
        to a reconstruction term whose per-token weight is

        .. math::
            w(t) = \frac{\alpha'(t)}{1 - \alpha(t)},

        where :math:`\alpha(t) = 1 - p_{\text{mask}}(t)` is the keep-rate. For the
        **linear** schedule (:math:`p_{\text{mask}}(t)=t`, so :math:`\alpha=1-t`,
        :math:`\alpha'=-1`) this is exactly :math:`w(t) = 1/t` (the well-known
        MDLM weight; MDLM arXiv:2406.07524, LLaDA arXiv:2502.09992). We therefore
        weight each masked token by :math:`1/t` of its example. The masked-CE is
        summed over masked positions, ``1/t``-weighted per example, then divided
        by the total number of masked tokens to yield a stable scalar.

        Args:
            prediction: Vocabulary logits of shape ``[batch, seq, vocab]``.
            info: Info dict from :meth:`corrupt` (uses ``"masked_positions"``,
                ``"targets"``, ``"t"``).

        Returns:
            Scalar loss tensor. Returns ``0`` (with grad) if no position was
            masked in the batch.
        """
        masked_positions = info["masked_positions"]  # [batch, seq] bool
        targets = info["targets"]  # [batch, seq]
        t = info["t"]  # [batch]

        batch, seq, vocab = prediction.shape
        # Per-token CE over the whole grid, then keep masked positions only.
        ce = F.cross_entropy(
            prediction.reshape(-1, vocab),
            targets.reshape(-1),
            reduction="none",
        ).reshape(batch, seq)

        # NELBO weight: 1/t for linear schedule; for cosine schedule the correct
        # continuous-time weight is (pi/2 * sin(pi*t/2)) / (1 - cos(pi*t/2)).
        if self.schedule == "cosine":
            numerator = (0.5 * math.pi) * torch.sin(0.5 * math.pi * t)
            denominator = (1.0 - torch.cos(0.5 * math.pi * t)).clamp(min=1e-8)
            nelbo_weight = numerator / denominator
        else:
            nelbo_weight = 1.0 / t
        weight = nelbo_weight.view(batch, *([1] * (ce.dim() - 1)))
        weighted = ce * weight * masked_positions.to(ce.dtype)

        denom = masked_positions.sum().clamp(min=1).to(ce.dtype)
        return weighted.sum() / denom


# ---------------------------------------------------------------------------
# Hybrid mask + Gaussian corruption -- NOVEL / experimental headline method.
# ---------------------------------------------------------------------------


class HybridCorruption(CorruptionProcess):
    r"""**Experimental** per-token hybrid of masked and Gaussian diffusion.

    *Headline contribution.* This process interpolates between the two dominant
    text-diffusion paradigms:

    * **Diffusion-LM / continuous** (Gaussian noise on embeddings, predict-``x0``)
    * **MDLM / LLaDA / discrete** (absorbing ``[MASK]`` + categorical denoising)

    For every token we flip a Bernoulli with probability ``mask_weight`` to pick a
    *channel*:

    * **Discrete channel** (prob ``mask_weight``): the token *may* be replaced by
      ``[MASK]`` with the absorbing schedule probability ``mask_prob(t)``; the
      model must predict its identity (categorical CE).
    * **Continuous channel** (prob ``1 - mask_weight``): the token's embedding is
      perturbed with Gaussian noise at level ``t`` (using the same
      ``sqrt(acp)*x0 + sqrt(1-acp)*noise`` parameterization); the model must
      predict the clean embedding (MSE).

    Intuition: ``mask_weight = 1`` recovers pure MDLM (every token is in the
    discrete channel), ``mask_weight = 0`` recovers pure continuous Diffusion-LM,
    and intermediate values let a *single* denoiser learn both a categorical head
    and an embedding-regression head, sharing representation across the two noise
    geometries. The shared timestep ``t`` controls the overall corruption level
    for both channels so the difficulty stays coupled.

    Because the model must consume a single corrupted embedding sequence, the
    discrete channel's tokens (masked or not) are embedded via ``embed_fn`` and
    concatenated with the noised continuous-channel embeddings into one
    ``[batch, seq, dim]`` tensor. The model then needs *two* heads at training
    time: a vocab-logits head (scored on discrete-channel tokens that were
    actually masked) and an ``x0`` head (scored on continuous-channel tokens).

    Args:
        mask_token_id: Vocabulary id of the ``[MASK]`` token.
        alphas_cumprod: Cumulative product schedule ``[T]`` for the Gaussian
            channel (same object the continuous model uses).
        embed_fn: Callable ``embed_fn(ids) -> Tensor`` mapping ids
            ``[batch, seq]`` to embeddings ``[batch, seq, dim]``. Passed in
            (rather than an ``nn.Embedding``) to avoid coupling to the model.
        mask_weight: Probability a token uses the discrete channel, in ``[0, 1]``.
        schedule: Schedule name shared by both channels (``"cosine"`` default;
            the discrete channel uses :func:`_mask_prob` and the continuous
            channel indexes ``alphas_cumprod``).
    """

    def __init__(
        self,
        mask_token_id: int,
        alphas_cumprod: torch.Tensor,
        embed_fn: Callable[[torch.Tensor], torch.Tensor],
        mask_weight: float = 0.5,
        schedule: str = "cosine",
    ):
        if not 0.0 <= mask_weight <= 1.0:
            raise ValueError("mask_weight must be in [0, 1].")
        if alphas_cumprod.dim() != 1:
            raise ValueError("alphas_cumprod must be a 1D tensor of shape [T].")
        self.mask_token_id = int(mask_token_id)
        self.alphas_cumprod = alphas_cumprod
        self.num_steps = int(alphas_cumprod.shape[0])
        self.embed_fn = embed_fn
        self.mask_weight = float(mask_weight)
        self.schedule = schedule
        # Reuse the discrete schedule helper for the masking probability.
        self._absorbing = AbsorbingMaskCorruption(mask_token_id, schedule=schedule)

    def sample_timesteps(self, batch: int, device: torch.device) -> torch.Tensor:
        """Sample continuous timesteps in ``(0, 1]`` shared by both channels.

        Args:
            batch: Number of examples.
            device: Device for the returned tensor.

        Returns:
            Float tensor of shape ``[batch]`` in ``(0, 1]``.
        """
        return self._absorbing.sample_timesteps(batch, device)

    def _t_to_index(self, t: torch.Tensor) -> torch.Tensor:
        """Map continuous ``t`` in ``(0, 1]`` to a Gaussian-schedule index.

        Args:
            t: Continuous timesteps ``[batch]`` in ``(0, 1]``.

        Returns:
            Long indices ``[batch]`` in ``[0, T)`` for ``alphas_cumprod`` lookup.
        """
        idx = (t * (self.num_steps - 1)).round().long()
        return idx.clamp(0, self.num_steps - 1)

    def corrupt(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, InfoDict]:
        """Apply the per-token hybrid corruption.

        Args:
            x: Clean token ids of shape ``[batch, seq]`` (long).
            t: Continuous timesteps ``[batch]`` in ``(0, 1]``.
            noise: Optional fixed Gaussian noise of shape
                ``[batch, seq, dim]`` for the continuous channel.

        Returns:
            A tuple ``(corrupted_embeds, info)``. ``corrupted_embeds`` has shape
            ``[batch, seq, dim]``: discrete-channel positions hold the embedding
            of either ``[MASK]`` or the (unmasked) original token; continuous-
            channel positions hold the Gaussian-noised clean embedding. ``info``
            keys:

            * ``"discrete_channel"``: bool ``[batch, seq]`` -- token uses the
              discrete channel.
            * ``"masked_positions"``: bool ``[batch, seq]`` -- discrete-channel
              token actually replaced by ``[MASK]`` (subset of
              ``discrete_channel``); cross-entropy is scored here.
            * ``"continuous_channel"``: bool ``[batch, seq]`` -- complement of
              ``discrete_channel``; MSE is scored here.
            * ``"targets"``: original ids ``[batch, seq]`` (CE targets).
            * ``"x0"``: clean embeddings ``[batch, seq, dim]`` (MSE targets).
            * ``"noise"``: the Gaussian noise used ``[batch, seq, dim]``.
            * ``"t"``: timesteps ``[batch]``.
        """
        device = x.device
        x0_embeds = self.embed_fn(x)  # [batch, seq, dim]
        dim = x0_embeds.shape[-1]

        if noise is None:
            noise = torch.randn_like(x0_embeds)

        # 1) Channel assignment per token.
        discrete_channel = torch.rand(x.shape, device=device) < self.mask_weight
        continuous_channel = ~discrete_channel

        # 2) Discrete channel: mask with prob mask_prob(t) within the channel.
        p = self._absorbing.mask_prob(t)  # [batch]
        p = p.view(p.shape[0], *([1] * (x.dim() - 1)))
        masked_positions = discrete_channel & (torch.rand(x.shape, device=device) < p)

        # Discrete-channel ids: [MASK] where masked, original otherwise.
        discrete_ids = torch.where(
            masked_positions, torch.full_like(x, self.mask_token_id), x
        )
        discrete_embeds = self.embed_fn(discrete_ids)  # [batch, seq, dim]

        # 3) Continuous channel: Gaussian-noise the clean embedding at level t.
        idx = self._t_to_index(t)  # [batch]
        acp = self.alphas_cumprod.to(device=device, dtype=x0_embeds.dtype)[idx]
        sqrt_acp = _broadcast_to(torch.sqrt(acp), x0_embeds)
        sqrt_one_minus = _broadcast_to(torch.sqrt(1.0 - acp), x0_embeds)
        noised_embeds = sqrt_acp * x0_embeds + sqrt_one_minus * noise

        # 4) Combine into a single embedding sequence the model consumes.
        chan = discrete_channel.unsqueeze(-1).expand(-1, -1, dim)
        corrupted_embeds = torch.where(chan, discrete_embeds, noised_embeds)

        info: InfoDict = {
            "discrete_channel": discrete_channel,
            "continuous_channel": continuous_channel,
            "masked_positions": masked_positions,
            "targets": x,
            "x0": x0_embeds,
            "noise": noise,
            "t": t,
        }
        return corrupted_embeds, info

    def loss(
        self,
        prediction: torch.Tensor,
        info: InfoDict,
        x0_prediction: Optional[torch.Tensor] = None,
        ce_weight: float = 1.0,
        mse_weight: float = 1.0,
    ) -> torch.Tensor:
        r"""Combined masked-CE (discrete) + MSE (continuous) objective.

        Args:
            prediction: Vocabulary logits ``[batch, seq, vocab]`` from the
                categorical head. Cross-entropy is scored only on
                ``info["masked_positions"]`` (with the ``1/t`` MDLM weight).
            info: Info dict from :meth:`corrupt`.
            x0_prediction: Predicted clean embeddings ``[batch, seq, dim]`` from
                the regression head. MSE is scored only on
                ``info["continuous_channel"]``. If ``None``, the MSE term is
                skipped (useful for smoke tests with a single head).
            ce_weight: Scalar multiplier on the masked-CE term.
            mse_weight: Scalar multiplier on the MSE term.

        Returns:
            Scalar combined loss tensor.
        """
        masked_positions = info["masked_positions"]
        targets = info["targets"]
        t = info["t"]

        batch, seq, vocab = prediction.shape
        ce = F.cross_entropy(
            prediction.reshape(-1, vocab),
            targets.reshape(-1),
            reduction="none",
        ).reshape(batch, seq)
        # NELBO weight: 1/t for linear schedule; for cosine schedule the correct
        # continuous-time weight is (pi/2 * sin(pi*t/2)) / (1 - cos(pi*t/2)).
        if self.schedule == "cosine":
            numerator = (0.5 * math.pi) * torch.sin(0.5 * math.pi * t)
            denominator = (1.0 - torch.cos(0.5 * math.pi * t)).clamp(min=1e-8)
            nelbo_weight = numerator / denominator
        else:
            nelbo_weight = 1.0 / t
        ce_w = nelbo_weight.view(batch, *([1] * (ce.dim() - 1)))
        ce_term = (ce * ce_w * masked_positions.to(ce.dtype)).sum()
        ce_term = ce_term / masked_positions.sum().clamp(min=1).to(ce.dtype)

        total = ce_weight * ce_term

        if x0_prediction is not None:
            continuous_channel = info["continuous_channel"]
            x0 = info["x0"]
            sq = ((x0_prediction - x0) ** 2).mean(dim=-1)  # [batch, seq]
            mse_term = (sq * continuous_channel.to(sq.dtype)).sum()
            mse_term = mse_term / continuous_channel.sum().clamp(min=1).to(sq.dtype)
            total = total + mse_weight * mse_term

        return total
