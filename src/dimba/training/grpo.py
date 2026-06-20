"""GRPO (Group Relative Policy Optimization) for DIMBA block-CoT models.

Implements diffu-GRPO (d1 paper, arXiv:2504.12216) adapted for DIMBA's
Gaussian continuous diffusion and block-CoT generation format.

Algorithm per step
    1. For each prompt in the batch, generate G completions via block CoT.
    2. Score completions with a verifiable reward function.
    3. Compute group-relative advantages: A_i = (r_i - μ) / (σ + ε).
    4. Estimate policy log-prob of each completion using the ELBO surrogate
       (antithetic timestep sampling for variance reduction, VRPO-style).
    5. GRPO loss = -mean(A_i * lp_i) + β * KL(policy ∥ ref).
    6. Anti-overthinking: thinking-length penalty subtracted from reward.

References
    d1 / diffu-GRPO:      arXiv:2504.12216
    GRPO / DeepSeekMath:  arXiv:2402.03300
    LLaDA 1.5 / VRPO:     arXiv:2505.19223  (antithetic timesteps)
"""
from __future__ import annotations

import copy
import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .preference import elbo_sequence_logprob, antithetic_timesteps
from .rewards import CompositeReward, NumericAnswerReward, ExactMatchReward, LengthPenaltyReward, Reward
from ..inference.block_cot import block_sample_from_model

logger = logging.getLogger(__name__)


# ── config ────────────────────────────────────────────────────────────────────

@dataclass
class GRPOConfig:
    """Hyperparameters for GRPO training.

    Anti-overthinking knobs
        max_think_blocks:       hard cap on thinking blocks generated (2 is good for 135M).
        thinking_length_weight: reward penalty per think token (0.01–0.05 range).
        min_response_reward:    completions with no think blocks get a small bonus if
                                they answer correctly — encourages direct answers for easy
                                prompts (set to 0 to disable).

    GRPO core
        group_size:   number of completions to sample per prompt (G; 8 is standard).
        kl_coeff:     KL(policy ∥ ref) weight (β; 0.01–0.1).
        mc_samples:   ELBO MC timestep samples for log-prob estimation (2 = good tradeoff).
        antithetic:   use antithetic timestep pairs for variance reduction (VRPO).
    """
    # ── generation ────────────────────────────────────────────────────────────
    block_size: int = 64
    max_think_blocks: int = 2
    response_len: int = 128
    think_start_id: Optional[int] = None
    think_end_id: Optional[int] = None
    eos_id: Optional[int] = None
    adaptive_stop: bool = True
    num_diffusion_steps_inference: int = 50
    sampler: str = "dpmpp"         # fast; switch to "ddim" if dpmpp is unstable early

    # ── anti-overthinking ─────────────────────────────────────────────────────
    thinking_length_weight: float = 0.02     # penalty per think token; 0 to disable
    min_response_reward_bonus: float = 0.05  # tiny bonus for direct (no-think) correct answer

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
    bf16: bool = True

    # ── logging / checkpointing ───────────────────────────────────────────────
    log_interval: int = 10
    save_interval: int = 200
    save_dir: str = "./checkpoints/grpo"
    state_file: str = "./training_state.json"   # for /loop monitor


# ── core utilities ────────────────────────────────────────────────────────────

def _compute_elbo_logprob(
    model: nn.Module,
    input_ids: torch.Tensor,
    response_mask: torch.Tensor,
    mc_samples: int,
    antithetic: bool,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """ELBO-surrogate log-prob for the response positions of *input_ids*. [B]."""
    B = input_ids.shape[0]
    T = model.num_diffusion_steps
    total_lp = torch.zeros(B, device=input_ids.device)

    draws = mc_samples // 2 if antithetic else mc_samples
    draws = max(1, draws)

    for _ in range(draws):
        if antithetic:
            t, t_ant = antithetic_timesteps(B, T, device=input_ids.device, generator=generator)
            for tt in (t, t_ant):
                lp = elbo_sequence_logprob(model, input_ids, input_ids, response_mask,
                                           timesteps=tt, num_mc_samples=1,
                                           generator=generator)
                total_lp = total_lp + lp
            n_draws = 2
        else:
            lp = elbo_sequence_logprob(model, input_ids, input_ids, response_mask,
                                       timesteps=None, num_mc_samples=1,
                                       generator=generator)
            total_lp = total_lp + lp
            n_draws = 1

    return total_lp / (draws * (2 if antithetic else 1))


def _group_advantages(rewards: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Group-normalize rewards → advantages. *rewards* shape: [B, G]."""
    mu = rewards.mean(dim=1, keepdim=True)
    sigma = rewards.std(dim=1, keepdim=True)
    return (rewards - mu) / (sigma + eps)


# ── trainer ──────────────────────────────────────────────────────────────────

class GRPOTrainer:
    """GRPO trainer for a DIMBA model with block-CoT generation.

    Args:
        model:     The policy DIMBA model (will be trained in-place).
        reward_fn: A :class:`~dimba.training.rewards.Reward` callable.
                   For math: ``NumericAnswerReward()``.
                   For general: ``CompositeReward([(ExactMatchReward(), 0.8),
                                                   (LengthPenaltyReward(target_length=64), 0.2)])``.
        tokenizer: Any object with ``.encode(str) -> List[int]`` and
                   ``.decode(List[int]) -> str``.
        config:    :class:`GRPOConfig` instance.
        ref_model: Frozen reference model for KL penalty.  When *None*, a deep
                   copy of *model* at construction time is used.
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

        # Frozen reference model (deep copy if not provided)
        if ref_model is not None:
            self.ref_model = ref_model
        else:
            self.ref_model = copy.deepcopy(model)
        for p in self.ref_model.parameters():
            p.requires_grad_(False)
        self.ref_model.eval()

        # Optimizer
        _fused = next(model.parameters()).is_cuda
        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
            fused=_fused,
        )

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
        self,
        prompt_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate G completions per prompt.

        Returns:
            full_ids      [B*G, seq_len]
            response_ids  [B*G, response_len]
            think_counts  [B*G]  (number of think blocks generated)
        """
        B = prompt_ids.shape[0]
        G = self.cfg.group_size
        device = prompt_ids.device

        full_list, resp_list, count_list = [], [], []
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
            full_list.append(out["full_ids"])    # [B, seq_len]
            resp_list.append(out["response"])    # [B, response_len]
            count_list.append(
                torch.full((B,), out["n_think_blocks"], dtype=torch.long, device=device)
            )

        # Pad to same length (sequences may differ by overhead tokens)
        max_len = max(f.shape[1] for f in full_list)
        padded = [
            F.pad(f, (0, max_len - f.shape[1])) for f in full_list
        ]
        full_ids = torch.stack(padded, dim=1).reshape(B * G, max_len)  # [B*G, L]
        response_ids = torch.stack(resp_list, dim=1).reshape(B * G, self.cfg.response_len)
        think_counts = torch.stack(count_list, dim=1).reshape(B * G)
        return full_ids, response_ids, think_counts

    # ------------------------------------------------------------------
    def _score_completions(
        self,
        prompt_strs: List[str],
        response_ids: torch.Tensor,
        reference_strs: Optional[List[str]],
        think_counts: torch.Tensor,
    ) -> torch.Tensor:
        """Return reward tensor [B*G]."""
        B_G = response_ids.shape[0]
        G = self.cfg.group_size
        B = B_G // G
        device = response_ids.device

        resp_strs = self._decode(response_ids)
        rewards = []

        for i, (resp, n_think) in enumerate(zip(resp_strs, think_counts.tolist())):
            prompt_str = prompt_strs[i // G]
            ref_str = reference_strs[i // G] if reference_strs else None

            # Core task reward
            r = float(self.reward_fn(prompt_str, resp, ref_str))

            # Anti-overthinking: penalize each think token
            r -= n_think * self.cfg.block_size * self.cfg.thinking_length_weight

            # Small bonus for correct direct (no-think) answers
            if n_think == 0 and r > 0:
                r += self.cfg.min_response_reward_bonus

            rewards.append(r)

        return torch.tensor(rewards, dtype=torch.float, device=device)

    # ------------------------------------------------------------------
    def _build_response_mask(
        self, full_ids: torch.Tensor, response_ids: torch.Tensor
    ) -> torch.Tensor:
        """Build a mask [B*G, L] marking the response positions in full_ids."""
        B_G, L = full_ids.shape
        R = response_ids.shape[1]
        mask = torch.zeros(B_G, L, device=full_ids.device)
        # Response tokens are always the last R tokens of full_ids
        mask[:, -R:] = 1.0
        return mask

    # ------------------------------------------------------------------
    def train_step(
        self,
        prompt_strs: List[str],
        reference_strs: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """One GRPO update from a list of prompt strings.

        Args:
            prompt_strs: Prompts for this mini-batch (length B).
            reference_strs: Ground-truth answers for verifiable reward.

        Returns:
            Dict of scalar metrics for logging / monitoring.
        """
        cfg = self.cfg
        G = cfg.group_size
        B = len(prompt_strs)
        device = next(self.model.parameters()).device
        dtype = torch.bfloat16 if cfg.bf16 and device.type == "cuda" else torch.float32

        # Tokenize prompts
        prompt_ids = torch.zeros(B, 128, dtype=torch.long, device=device)
        for i, ps in enumerate(prompt_strs):
            enc = self._encode(ps)[:128]
            prompt_ids[i, :len(enc)] = torch.tensor(enc, dtype=torch.long)

        # ── 1. Generate G completions per prompt ─────────────────────────────
        self.model.eval()
        full_ids, response_ids, think_counts = self._generate_group(prompt_ids)

        # ── 2. Score ─────────────────────────────────────────────────────────
        rewards_flat = self._score_completions(
            prompt_strs, response_ids, reference_strs, think_counts
        )
        rewards = rewards_flat.reshape(B, G)    # [B, G]
        advantages = _group_advantages(rewards, eps=cfg.advantage_eps)  # [B, G]
        advantages_flat = advantages.reshape(B * G)

        # ── 3. Policy log-probs ───────────────────────────────────────────────
        self.model.train()
        response_mask = self._build_response_mask(full_ids, response_ids)

        with torch.autocast(device_type=device.type, dtype=dtype, enabled=cfg.bf16):
            policy_lp = _compute_elbo_logprob(
                self.model, full_ids, response_mask,
                cfg.mc_samples, cfg.antithetic,
            )
            ref_lp = _compute_elbo_logprob(
                self.ref_model, full_ids, response_mask,
                cfg.mc_samples, cfg.antithetic,
            )

            # ── 4. GRPO loss = policy-gradient term + KL penalty ─────────────
            pg_loss = -(advantages_flat * policy_lp).mean()
            kl = (policy_lp - ref_lp).mean()
            loss = pg_loss + cfg.kl_coeff * kl

        # ── 5. Optimise ───────────────────────────────────────────────────────
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
        self.optimizer.step()
        self.step += 1

        metrics = {
            "step": self.step,
            "loss": loss.item(),
            "pg_loss": pg_loss.item(),
            "kl": kl.item(),
            "mean_reward": rewards.mean().item(),
            "mean_think_blocks": think_counts.float().mean().item(),
            "reward_std": rewards.std().item(),
        }

        if self.step % cfg.log_interval == 0:
            logger.info(
                "step %d | loss=%.4f | reward=%.3f | think_blocks=%.2f | kl=%.4f",
                self.step, metrics["loss"], metrics["mean_reward"],
                metrics["mean_think_blocks"], metrics["kl"],
            )
            self._write_state(metrics)

        if self.step % cfg.save_interval == 0:
            self.save_checkpoint()

        return metrics

    # ------------------------------------------------------------------
    def _write_state(self, metrics: Dict[str, float]) -> None:
        """Write metrics to *state_file* so the /loop monitor can read them."""
        state = {
            **metrics,
            "stage": "grpo",
            "timestamp": time.time(),
        }
        os.makedirs(os.path.dirname(self.cfg.state_file) or ".", exist_ok=True)
        with open(self.cfg.state_file, "w") as f:
            json.dump(state, f, indent=2)

    def save_checkpoint(self, path: Optional[str] = None) -> str:
        os.makedirs(self.cfg.save_dir, exist_ok=True)
        path = path or os.path.join(self.cfg.save_dir, f"grpo_step{self.step}.pt")
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "step": self.step,
            "config": asdict(self.cfg),
        }, path)
        logger.info("saved checkpoint → %s", path)
        return path

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location="cpu")
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.step = ckpt.get("step", 0)
        logger.info("loaded checkpoint ← %s (step %d)", path, self.step)


# ── factory helper ────────────────────────────────────────────────────────────

def build_math_grpo_trainer(
    model: nn.Module,
    tokenizer: Any,
    config: Optional[GRPOConfig] = None,
    ref_model: Optional[nn.Module] = None,
) -> GRPOTrainer:
    """Convenience: GRPO trainer with a numeric-answer verifiable reward.

    Suitable for Orca-Math / NuminaMath style training.  Composed reward:
    - 0.9 × NumericAnswerReward  (correct final number)
    - 0.1 × LengthPenaltyReward  (discourages length blow-ups)
    """
    reward = CompositeReward([
        (NumericAnswerReward(), 0.9),
        (LengthPenaltyReward(target_length=80, tolerance=24, penalty_per_token=0.005), 0.1),
    ])
    return GRPOTrainer(model, reward, tokenizer, config, ref_model)


def build_general_grpo_trainer(
    model: nn.Module,
    tokenizer: Any,
    config: Optional[GRPOConfig] = None,
    ref_model: Optional[nn.Module] = None,
) -> GRPOTrainer:
    """Convenience: GRPO trainer with exact-match + length reward.

    Suitable for SmolTalk / general instruction-following.
    """
    reward = CompositeReward([
        (ExactMatchReward(), 0.85),
        (LengthPenaltyReward(target_length=64, tolerance=20, penalty_per_token=0.005), 0.15),
    ])
    return GRPOTrainer(model, reward, tokenizer, config, ref_model)
