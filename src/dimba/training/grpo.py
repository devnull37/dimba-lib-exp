"""GRPO / MGPO training for DIMBA block-CoT models.

Implements diffu-GRPO (d1, arXiv:2504.12216) with three improvements from
VibeThinker (arXiv:2606.16140):

  MGPO weighting      Prompts where the model's group accuracy p(q) ≈ 0.5
                      (capability boundary) are upweighted.  Easy wins
                      (p≈1) and hopeless prompts (p≈0) contribute little.
                      w(q) = exp(-γ · (p(q) - 0.5)² / 0.25)

  Long2Short reward   Within each group, correct-but-shorter completions get
                      a normalised brevity bonus instead of a fixed per-token
                      penalty.  Only applied to completions where r > 0.
                      bonus_i = λ · (s_i - s̄) / (max|s_j - s̄| + ε)
                      where s_i = 1/L_i  (inverse total length).

  LR warmup           Linear warmup then cosine decay — previously missing
                      from the GRPO optimizer.

References
    d1 / diffu-GRPO:        arXiv:2504.12216
    GRPO / DeepSeekMath:    arXiv:2402.03300
    LLaDA 1.5 / VRPO:       arXiv:2505.19223  (antithetic timesteps)
    VibeThinker / MGPO:     arXiv:2606.16140
"""
from __future__ import annotations

import copy
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .preference import elbo_sequence_logprob, antithetic_timesteps
from .rewards import CompositeReward, NumericAnswerReward, ExactMatchReward, RegexMatchReward, LengthPenaltyReward, Reward
from ..inference.block_cot import block_sample_from_model

logger = logging.getLogger(__name__)


# ── config ────────────────────────────────────────────────────────────────────

@dataclass
class GRPOConfig:
    """Hyperparameters for GRPO/MGPO training.

    MGPO knobs
        use_mgpo:       Enable max-entropy prompt weighting (VibeThinker).
        mgpo_gamma:     Sharpness of the capability-boundary gate (2.0 is good).
                        Higher = only train on very-boundary prompts.

    Long2Short knobs
        long2short_lambda:  Normalised brevity bonus weight (0.2 from VibeThinker).
                            Set to 0 to use the old fixed thinking_length_weight.
        thinking_length_weight: Legacy fixed penalty per think token.  Ignored
                            when long2short_lambda > 0.

    GRPO core
        group_size:     Completions per prompt (G).
        kl_coeff:       KL(policy ‖ ref) weight β.
        mc_samples:     ELBO MC samples for log-prob (2 is a good tradeoff).
        antithetic:     Antithetic timestep pairs for variance reduction (VRPO).
    """
    # ── generation ────────────────────────────────────────────────────────────
    block_size: int = 64
    max_think_blocks: int = 2
    response_len: int = 128
    think_start_id: Optional[int] = None
    think_end_id: Optional[int] = None
    eos_id: Optional[int] = None
    adaptive_stop: bool = True
    num_diffusion_steps_inference: int = 15
    sampler: str = "euler"          # flow matching default

    # ── MGPO (VibeThinker max-entropy weighting) ──────────────────────────────
    use_mgpo: bool = True
    mgpo_gamma: float = 2.0         # higher = tighter gate around p(q)=0.5

    # ── Long2Short (normalised brevity bonus) ─────────────────────────────────
    long2short_lambda: float = 0.2  # 0 = disable, use thinking_length_weight instead
    thinking_length_weight: float = 0.02   # legacy; only used when long2short_lambda==0
    min_response_reward_bonus: float = 0.05  # small bonus for correct direct answers

    # ── GRPO core ─────────────────────────────────────────────────────────────
    group_size: int = 8
    kl_coeff: float = 0.04
    mc_samples: int = 2
    antithetic: bool = True
    advantage_eps: float = 1e-6

    # ── optimization ─────────────────────────────────────────────────────────
    lr: float = 5e-6
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 50
    total_steps: int = 2000         # for cosine schedule
    bf16: bool = True

    # ── logging / checkpointing ───────────────────────────────────────────────
    log_interval: int = 10
    save_interval: int = 200
    save_dir: str = "./checkpoints/grpo"
    state_file: str = "./training_state.json"


# ── core utilities ────────────────────────────────────────────────────────────

def _compute_elbo_logprob(
    model: nn.Module,
    input_ids: torch.Tensor,
    response_mask: torch.Tensor,
    mc_samples: int,
    antithetic: bool,
    generator: Optional[torch.Generator] = None,
    prompt_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """ELBO-surrogate log-prob for response positions of *input_ids*. Returns [B].

    Bug 3 fix: when prompt_mask is supplied, a custom logits_fn is built that
    passes it into model.forward so that (a) prompt tokens are kept clean during
    diffusion and (b) pooled conditioning is built from the prompt prefix only.
    Without this, model(input_ids, t, return_latent_info=True) with prompt_mask=None
    noises the entire sequence (including the prompt) and uses no conditioning.
    """
    B = input_ids.shape[0]
    T = model.num_diffusion_steps
    total_lp = torch.zeros(B, device=input_ids.device)

    if prompt_mask is not None:
        # Capture prompt_mask in a closure for the logits_fn signature required
        # by elbo_sequence_logprob: (model, input_ids, t) -> logits.
        _pm = prompt_mask
        def _logits_fn(m: nn.Module, ids: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            x_pred, _, _ = m(ids, t, prompt_mask=_pm, return_latent_info=True)
            return m.output_head(x_pred, embedding_weight=m.token_embed.get_weight())
    else:
        _logits_fn = None  # elbo_sequence_logprob uses its default forward

    if antithetic and mc_samples < 2:
        antithetic = False  # can't do antithetic pairs with fewer than 2 samples
    draws = mc_samples // 2 if antithetic else mc_samples
    draws = max(1, draws)
    for _ in range(draws):
        if antithetic:
            t, t_ant = antithetic_timesteps(B, T, device=input_ids.device, generator=generator)
            for tt in (t, t_ant):
                lp = elbo_sequence_logprob(model, input_ids, input_ids, response_mask,
                                           timesteps=tt, num_mc_samples=1,
                                           logits_fn=_logits_fn,
                                           generator=generator)
                total_lp = total_lp + lp
        else:
            lp = elbo_sequence_logprob(model, input_ids, input_ids, response_mask,
                                       timesteps=None, num_mc_samples=1,
                                       logits_fn=_logits_fn,
                                       generator=generator)
            total_lp = total_lp + lp

    n_total = draws * (2 if antithetic else 1)
    return total_lp / n_total


def _group_advantages(rewards: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Group-normalize rewards → advantages. *rewards*: [B, G] → [B, G].

    Uses population std (correction=0) to avoid NaN when G=1 (Bessel-corrected
    std of a single element is undefined).  When G=1 or all rewards are equal,
    sigma=0 and advantages become 0 (via eps), which is harmless.
    """
    mu = rewards.mean(dim=1, keepdim=True)
    sigma = rewards.std(dim=1, keepdim=True, correction=0)
    return (rewards - mu) / (sigma + eps)


def _mgpo_weights(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    """MGPO prompt weights [B] from group rewards [B, G].

    w(q) = exp(-γ · (p(q) - 0.5)² / 0.25)

    Peaks at p(q)=0.5 (half the group correct — capability boundary).
    Prompts trivially solved (p≈1) or impossible (p≈0) get near-zero weight.
    """
    p_q = (rewards > 0).float().mean(dim=1)        # [B] group accuracy
    deviation = (p_q - 0.5).pow(2) / 0.25          # normalized distance from 0.5
    return torch.exp(-gamma * deviation)             # [B]


def _long2short_bonus(
    full_lengths: torch.Tensor,    # [B*G] total sequence lengths
    raw_rewards: torch.Tensor,     # [B*G] rewards before bonus
    G: int,
    lam: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Normalised brevity bonus (VibeThinker Long2Short).

    Within each group of G completions:
      s_i = 1/L_i  (brevity score; shorter = higher)
      bonus_i = λ · (s_i - s̄) / (max|s_j - s̄| + ε)

    Only applied to correct completions (raw_reward > 0) — we don't want to
    reward short wrong answers.

    Returns a bonus tensor [B*G] to add to raw_rewards.
    """
    B = full_lengths.shape[0] // G
    s = 1.0 / full_lengths.float().clamp(min=1)    # [B*G] brevity scores
    s = s.reshape(B, G)
    correct = (raw_rewards.reshape(B, G) > 0).float()

    s_bar = s.mean(dim=1, keepdim=True)             # [B, 1]
    deviation = s - s_bar                           # [B, G]
    max_dev = deviation.abs().max(dim=1, keepdim=True).values.clamp(min=eps)
    normalised = deviation / max_dev                # [B, G] in [-1, 1]

    bonus = lam * normalised * correct              # only for correct completions
    return bonus.reshape(B * G)


# ── trainer ──────────────────────────────────────────────────────────────────

class GRPOTrainer:
    """GRPO/MGPO trainer for a DIMBA model with block-CoT generation.

    Args:
        model:     Policy DIMBA model (trained in-place).
        reward_fn: A :class:`~dimba.training.rewards.Reward` callable.
        tokenizer: Object with ``.encode`` / ``.decode``.
        config:    :class:`GRPOConfig`.
        ref_model: Frozen reference model for KL penalty (deep copy if None).
    """

    def __init__(
        self,
        model: nn.Module,
        reward_fn: Reward,
        tokenizer: Any,
        config: Optional[GRPOConfig] = None,
        ref_model: Optional[nn.Module] = None,
    ) -> None:
        self.model = model
        self.reward_fn = reward_fn
        self.tok = tokenizer
        self.cfg = config or GRPOConfig()

        self.ref_model = ref_model if ref_model is not None else copy.deepcopy(model)
        for p in self.ref_model.parameters():
            p.requires_grad_(False)
        self.ref_model.eval()

        # Bug 4 fix: validate that cfg.sampler is compatible with the model's
        # sampling path.  'euler'/'heun' are only valid on the flow-matching path;
        # 'ddim'/'dpmpp' are only valid on the non-flow path.  Surface the mismatch
        # at construction time rather than crashing on the first generation call.
        _use_flow = getattr(model, "use_flow_matching", False)
        if _use_flow:
            if self.cfg.sampler not in ("euler", "heun"):
                raise ValueError(
                    f"flow-matching model requires sampler in {{'euler', 'heun'}}, "
                    f"got {self.cfg.sampler!r}"
                )
        else:
            if self.cfg.sampler not in ("ddim", "dpmpp"):
                raise ValueError(
                    f"non-flow model requires sampler in {{'ddim', 'dpmpp'}}, "
                    f"got {self.cfg.sampler!r}"
                )

        _fused = next(model.parameters()).is_cuda
        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
            fused=_fused,
        )

        # Cosine LR schedule with linear warmup
        def _lr_lambda(step: int) -> float:
            if step < self.cfg.warmup_steps:
                return step / max(1, self.cfg.warmup_steps)
            progress = (step - self.cfg.warmup_steps) / max(
                1, self.cfg.total_steps - self.cfg.warmup_steps
            )
            return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, _lr_lambda)
        self.step = 0
        self._state: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    def _encode(self, text: str) -> List[int]:
        if hasattr(self.tok, "encode"):
            return self.tok.encode(text)
        return self.tok(text)

    def _decode(self, ids: torch.Tensor) -> List[str]:
        ids_list = ids.tolist()
        if hasattr(self.tok, "decode"):
            return [self.tok.decode(row) for row in ids_list]
        return [" ".join(map(str, row)) for row in ids_list]

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _generate_group(
        self, prompt_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate G completions per prompt.

        Returns:
            full_ids     [B*G, L]
            response_ids [B*G, response_len]
            think_counts [B*G]
            true_lens    [B*G]  pre-pad length of each row (independent of token values)
        """
        B = prompt_ids.shape[0]
        G = self.cfg.group_size
        device = prompt_ids.device

        full_list, resp_list, count_list, true_lens_list = [], [], [], []
        for _ in range(G):
            out = block_sample_from_model(
                self.model,
                prompt_ids,
                self.cfg.block_size,
                self.cfg.max_think_blocks,
                self.cfg.response_len,
                think_start_id=self.cfg.think_start_id,
                think_end_id=self.cfg.think_end_id,
                eos_id=self.cfg.eos_id,
                adaptive_stop=self.cfg.adaptive_stop,
                num_steps=self.cfg.num_diffusion_steps_inference,
                sampler=self.cfg.sampler,
            )
            f = out["full_ids"]
            # Record the true pre-pad column count for every row in this call.
            # f.shape[1] is the exact sequence length before padding, independent
            # of any token id values (including id 0) inside the sequence.
            true_lens_list.append(
                torch.full((B,), f.shape[1], dtype=torch.long, device=device)
            )
            full_list.append(f)
            resp_list.append(out["response"])
            # Bug 6 fix: block_sample_from_model returns n_think_blocks as a scalar
            # int (adaptive_stop breaks globally for the whole batch via
            # _is_degenerate in block_cot, not per-row).  The old isinstance branch
            # claiming per-row tensor tracking was dead and documented a guarantee
            # that does not exist.  Broadcast the scalar across the B rows of this
            # call to form the [B] count tensor.
            n_think = out["n_think_blocks"]
            count_list.append(
                torch.full((B,), int(n_think), dtype=torch.long, device=device)
            )

        max_len = max(f.shape[1] for f in full_list)
        padded = [F.pad(f, (0, max_len - f.shape[1])) for f in full_list]
        full_ids = torch.stack(padded, dim=1).reshape(B * G, max_len)
        response_ids = torch.stack(resp_list, dim=1).reshape(B * G, self.cfg.response_len)
        think_counts = torch.stack(count_list, dim=1).reshape(B * G)
        true_lens = torch.stack(true_lens_list, dim=1).reshape(B * G)
        return full_ids, response_ids, think_counts, true_lens

    # ------------------------------------------------------------------
    def _score_completions(
        self,
        prompt_strs: List[str],
        full_ids: torch.Tensor,
        response_ids: torch.Tensor,
        think_counts: torch.Tensor,
        true_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Score completions and apply Long2Short brevity bonus.

        Returns:
            raw_rewards    [B*G]  correctness reward before Long2Short shaping.
            bonused_rewards [B*G] rewards after Long2Short shaping (used for advantages).
        """
        G = self.cfg.group_size
        resp_strs = self._decode(response_ids)
        raw_list = []

        for i, (resp, n_think) in enumerate(zip(resp_strs, think_counts.tolist())):
            prompt_str = prompt_strs[i // G]
            ref_str = getattr(self, "_references", [None] * len(prompt_strs))[i // G]

            r = float(self.reward_fn(prompt_str, resp, ref_str))

            if self.cfg.long2short_lambda == 0:
                # Legacy: fixed per-token penalty
                r -= n_think * self.cfg.block_size * self.cfg.thinking_length_weight

            if n_think == 0 and r > 0:
                r += self.cfg.min_response_reward_bonus

            raw_list.append(r)

        raw_rewards = torch.tensor(raw_list, dtype=torch.float, device=full_ids.device)

        if self.cfg.long2short_lambda > 0:
            # Use true_lens (pre-pad sequence lengths) for the brevity score so
            # that token-id-0 occurrences inside the sequence don't distort L_i.
            full_lengths = true_lens.float()
            bonus = _long2short_bonus(
                full_lengths, raw_rewards, G,
                lam=self.cfg.long2short_lambda,
            )
            bonused_rewards = raw_rewards + bonus
        else:
            bonused_rewards = raw_rewards

        return raw_rewards, bonused_rewards

    # ------------------------------------------------------------------
    def _build_response_mask(
        self,
        full_ids: torch.Tensor,
        response_ids: torch.Tensor,
        true_lens: torch.Tensor,
    ) -> torch.Tensor:
        """[B*G, L] mask for response positions.

        With adaptive_stop=True different group members generate different numbers
        of think blocks, so rows have different actual lengths and are right-padded
        with zeros to align to max_len.  mask[:, -R:] would mark padding as
        response tokens for shorter rows.  We instead use the true pre-pad length
        recorded during generation (independent of token id values) and mark
        exactly the R tokens before the trailing padding.
        """
        B_G, L = full_ids.shape
        R = response_ids.shape[1]
        # true_lens[i] is the exact pre-pad column count for row i, passed in
        # from _generate_group.  This is correct regardless of token id values
        # (including id 0 inside prompt / think / response blocks).
        actual_len = true_lens  # [B*G]
        # Vectorised: for each row build a range mask for [start, start+R)
        row_idx = torch.arange(L, device=full_ids.device).unsqueeze(0)  # [1, L]
        start = (actual_len - R).clamp(min=0).unsqueeze(1)              # [B*G, 1]
        end = start + R                                                   # [B*G, 1]
        mask = ((row_idx >= start) & (row_idx < end)).float()
        return mask

    # ------------------------------------------------------------------
    def train_step(
        self,
        prompt_strs: List[str],
        reference_strs: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """One GRPO/MGPO update from a list of prompt strings.

        Args:
            prompt_strs:    Prompts (length B).
            reference_strs: Ground-truth answers for verifiable reward.

        Returns:
            Dict of scalar metrics.
        """
        cfg = self.cfg
        G = cfg.group_size
        B = len(prompt_strs)
        device = next(self.model.parameters()).device

        # Stash references so _score_completions can see them
        self._references = reference_strs or [None] * B

        # Tokenize prompts → [B, 128]
        prompt_ids = torch.zeros(B, 128, dtype=torch.long, device=device)
        for i, ps in enumerate(prompt_strs):
            enc = self._encode(ps)[:128]
            prompt_ids[i, :len(enc)] = torch.tensor(enc, dtype=torch.long)

        # ── 1. Generate ────────────────────────────────────────────────────────
        self.model.eval()
        full_ids, response_ids, think_counts, true_lens = self._generate_group(prompt_ids)

        # ── 2. Score (+ Long2Short bonus) ─────────────────────────────────────
        # raw_rewards: correctness signal before Long2Short shaping
        # bonused_rewards: shaped rewards used for advantage computation
        raw_rewards_flat, bonused_rewards_flat = self._score_completions(
            prompt_strs, full_ids, response_ids, think_counts, true_lens
        )
        raw_rewards = raw_rewards_flat.reshape(B, G)          # [B, G] correctness only
        bonused_rewards = bonused_rewards_flat.reshape(B, G)  # [B, G] for advantages
        advantages = _group_advantages(bonused_rewards, eps=cfg.advantage_eps)  # [B, G]

        # ── 3. MGPO: upweight capability-boundary prompts ─────────────────────
        # Use raw correctness rewards so the brevity bonus doesn't corrupt p(q).
        if cfg.use_mgpo:
            w = _mgpo_weights(raw_rewards, cfg.mgpo_gamma)    # [B]
            advantages = advantages * w.unsqueeze(1)           # broadcast to [B, G]

        advantages_flat = advantages.reshape(B * G)

        # ── Bug 1 fix: skip optimizer step when no group carries learning signal ──
        # With B=1 and a homogeneous group (all-correct or all-wrong), every
        # advantage is zero after group normalization.  pg_loss is then exactly 0
        # and the step wastes the generation budget with a pure-KL gradient only.
        # Detect this condition and skip optimizer.step()/scheduler.step().
        _adv_tol = cfg.advantage_eps * 10  # slightly above the normalization floor
        _homogeneous = advantages_flat.abs().max().item() < _adv_tol
        if not hasattr(self, "_skipped_steps"):
            self._skipped_steps = 0

        # ── 4. Policy / ref log-probs ─────────────────────────────────────────
        self.model.train()
        response_mask = self._build_response_mask(full_ids, response_ids, true_lens)
        response_len = response_mask.sum(dim=1).clamp(min=1)  # [B*G] for per-token norm

        # Bug 3 fix: build a prompt_mask so model.forward receives clean prompt
        # tokens (no noise) and builds pooled conditioning from the prompt prefix.
        # Computed from true_lens (not ~response_mask) to exclude right-padding.
        R = response_ids.shape[1]
        _prompt_start = (true_lens - R).clamp(min=0).unsqueeze(1)   # [B*G, 1]
        _col = torch.arange(full_ids.shape[1], device=full_ids.device).unsqueeze(0)
        prompt_mask = (_col < _prompt_start)                          # bool [B*G, L]

        policy_lp = _compute_elbo_logprob(
            self.model, full_ids, response_mask, cfg.mc_samples, cfg.antithetic,
            prompt_mask=prompt_mask,
        )
        with torch.no_grad():
            ref_lp = _compute_elbo_logprob(
                self.ref_model, full_ids, response_mask, cfg.mc_samples, cfg.antithetic,
                prompt_mask=prompt_mask,
            )

        # Per-token normalisation (VibeThinker: 1/|y_i| in the loss)
        policy_lp_norm = policy_lp / response_len
        ref_lp_norm = ref_lp / response_len

        # ── 5. GRPO loss ───────────────────────────────────────────────────────
        pg_loss = -(advantages_flat * policy_lp_norm).mean()
        # k3 KL estimator: E[exp(log r) - log r - 1] where r = ref/policy.
        # Always >= 0, unbiased, avoids the anti-regularization caused by the
        # raw log-prob difference which can be negative and reduce the loss.
        # Bug 2 fix: cast to fp32 and clamp before exp() to prevent bf16 overflow.
        logr = (ref_lp_norm - policy_lp_norm).float()   # log(ref/policy); fp32 to avoid bf16 overflow
        logr = logr.clamp(min=-10.0, max=10.0)          # bound k3 to ~2.2e4, preserve sign
        kl = (logr.exp() - logr - 1).mean()             # k3 estimator: always >= 0
        loss = pg_loss + cfg.kl_coeff * kl

        # ── 6. Optimise ────────────────────────────────────────────────────────
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
        if _homogeneous:
            # Homogeneous group: no policy learning signal; skip optimizer/scheduler
            # to avoid wasting an update step on a pure-KL gradient.
            self._skipped_steps += 1
        else:
            self.optimizer.step()
            used_lr = self.scheduler.get_last_lr()[0]   # LR actually applied by optimizer.step()
            self.scheduler.step()
        self.step += 1

        # ── 7. Metrics ────────────────────────────────────────────────────────
        # group_acc uses raw correctness rewards (not the Long2Short-shaped ones)
        # so it reflects true solve rate, not brevity-adjusted reward.
        group_acc = (raw_rewards > 0).float().mean(dim=1)     # [B]
        if _homogeneous:
            # LR didn't advance; report the current scheduler LR for consistency.
            used_lr = self.scheduler.get_last_lr()[0]
        metrics = {
            "step": self.step,
            "loss": loss.item(),
            "pg_loss": pg_loss.item(),
            "kl": kl.item(),
            "mean_reward": bonused_rewards.mean().item(),      # what the policy optimizes
            "reward_std": bonused_rewards.std().item(),
            "mean_think_blocks": think_counts.float().mean().item(),
            "group_accuracy": group_acc.mean().item(),
            "lr": used_lr,
            "skipped_steps": self._skipped_steps,
        }
        if cfg.use_mgpo:
            # Use raw_rewards to keep mgpo_weight consistent with the weight used
            # for advantages (computed from raw_rewards above).
            metrics["mgpo_weight_mean"] = _mgpo_weights(raw_rewards, cfg.mgpo_gamma).mean().item()

        if self.step % cfg.log_interval == 0:
            logger.info(
                "step %d | loss=%.4f | reward=%.3f±%.3f | think=%.2f | "
                "kl=%.4f | acc=%.2f | lr=%.2e",
                self.step, metrics["loss"], metrics["mean_reward"],
                metrics["reward_std"], metrics["mean_think_blocks"],
                metrics["kl"], metrics["group_accuracy"], metrics["lr"],
            )
            self._write_state(metrics)

        if self.step % cfg.save_interval == 0:
            self.save_checkpoint()

        return metrics

    # ------------------------------------------------------------------
    def _write_state(self, metrics: Dict[str, float]) -> None:
        state = {**metrics, "stage": "grpo", "timestamp": time.time()}
        os.makedirs(os.path.dirname(self.cfg.state_file) or ".", exist_ok=True)
        with open(self.cfg.state_file, "w") as f:
            json.dump(state, f, indent=2)

    def save_checkpoint(self, path: Optional[str] = None) -> str:
        os.makedirs(self.cfg.save_dir, exist_ok=True)
        path = path or os.path.join(self.cfg.save_dir, f"grpo_step{self.step}.pt")
        _base_model = getattr(self.model, "_orig_mod", self.model)
        # Pin the backend: DIMBA.config omits force_torch_mixer so a CUDA checkpoint
        # reloads on the fast kernel, but if this model runs on the pure-PyTorch
        # TorchMamba2 backend the next loader must rebuild on it to load the weights.
        cfg_dict = dict(getattr(_base_model, "config", None) or {})
        try:
            if type(_base_model.denoiser.blocks[0].mamba_fwd).__name__ == "TorchMamba2":
                cfg_dict["force_torch_mixer"] = True
        except Exception:  # noqa: BLE001 — never let a probe break checkpointing
            pass
        torch.save({
            "model_state_dict": _base_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "step": self.step,
            "grpo_config": asdict(self.cfg),
            "config": cfg_dict or None,  # real DIMBA model config (+ backend pin)
        }, path)
        logger.info("saved → %s", path)
        return path

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location="cpu")
        msd = ckpt["model_state_dict"]
        # Tolerate torch.compile prefix differences between the saving and loading model.
        try:
            self.model.load_state_dict(msd)
        except RuntimeError:
            from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
            consume_prefix_in_state_dict_if_present(msd, "_orig_mod.")
            target = getattr(self.model, "_orig_mod", self.model)
            target.load_state_dict(msd)
        # Optimizer and scheduler state are only present in GRPO-saved checkpoints;
        # SFT/distill checkpoints are model-only and do not carry these keys.
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.step = ckpt.get("step", 0)
        logger.info("loaded <- %s (step %d)", path, self.step)


# ── factory helpers ───────────────────────────────────────────────────────────

def build_math_grpo_trainer(
    model: nn.Module,
    tokenizer: Any,
    config: Optional[GRPOConfig] = None,
    ref_model: Optional[nn.Module] = None,
) -> GRPOTrainer:
    """GRPO trainer with numeric-answer verifiable reward (OrcaMath / NuminaMath)."""
    reward = CompositeReward([
        (NumericAnswerReward(), 0.9),
        (LengthPenaltyReward(target_length=80, tolerance=24, penalty_per_token=0.005), 0.1),
    ])
    return GRPOTrainer(model, reward, tokenizer, config, ref_model)


def build_code_grpo_trainer(
    model: nn.Module,
    tokenizer: Any,
    config: Optional[GRPOConfig] = None,
    ref_model: Optional[nn.Module] = None,
) -> GRPOTrainer:
    """GRPO trainer for code reasoning (Bespoke-Stratos / OpenThoughts code split).

    Bespoke-Stratos is math-heavy (~10 K math / ~5 K code). ExactMatch against the
    full long reference answer is essentially always 0, giving no gradient signal.
    Use NumericAnswerReward (covers the math rows) + a boxed-answer regex reward
    (rewards the model for producing a \\boxed{} final answer) + a length penalty.
    """
    reward = CompositeReward([
        (NumericAnswerReward(), 0.7),
        (RegexMatchReward(pattern=r"\\boxed\{[^}]*\}"), 0.2),
        (LengthPenaltyReward(target_length=96, tolerance=32, penalty_per_token=0.005), 0.1),
    ])
    return GRPOTrainer(model, reward, tokenizer, config, ref_model)


def build_general_grpo_trainer(
    model: nn.Module,
    tokenizer: Any,
    config: Optional[GRPOConfig] = None,
    ref_model: Optional[nn.Module] = None,
) -> GRPOTrainer:
    """GRPO trainer with exact-match + length reward (SmolTalk / general)."""
    reward = CompositeReward([
        (ExactMatchReward(), 0.85),
        (LengthPenaltyReward(target_length=64, tolerance=20, penalty_per_token=0.005), 0.15),
    ])
    return GRPOTrainer(model, reward, tokenizer, config, ref_model)
