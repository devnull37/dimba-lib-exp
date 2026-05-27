"""Pluggable, verifiable reward functions for DIMBA GRPO.

GRPO (Group Relative Policy Optimization; Shao et al., 2024, DeepSeekMath
arXiv:2402.03300) estimates advantages from the *relative* reward of multiple
sampled completions per prompt. The quality of the resulting policy is therefore
bounded entirely by the quality of the reward signal. The d1 / diffu-GRPO work
("d1: Scaling Reasoning in Diffusion LLMs via Reinforcement Learning",
arXiv:2504.12216) shows that adapting GRPO to masked diffusion LMs works well
*precisely because* the reward is a verifiable, rule-based check (e.g. exact
match on a math answer) rather than a soft text-overlap heuristic.

This module provides a small, composable set of rewards behind a single
:class:`Reward` protocol so the GRPO training script can select a reward at the
command line. Prefer the *verifiable* rewards (:class:`ExactMatchReward`,
:class:`NumericAnswerReward`, :class:`RegexMatchReward`) whenever a ground-truth
reference is available; they cannot be gamed by copying the prompt the way a
token-overlap reward can.

References
    - GRPO / DeepSeekMath: arXiv:2402.03300.
    - d1 / diffu-GRPO for diffusion LLMs: arXiv:2504.12216.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

__all__ = [
    "Reward",
    "ExactMatchReward",
    "NumericAnswerReward",
    "RegexMatchReward",
    "LengthPenaltyReward",
    "RewardModelReward",
    "CompositeReward",
    "CodeExecReward",
    "TokenOverlapReward",
    "get_reward",
    "REWARD_REGISTRY",
]


@runtime_checkable
class Reward(Protocol):
    """Protocol for a scalar reward over a generated completion.

    A reward maps a ``(prompt, completion, reference)`` triple to a float. The
    ``reference`` (ground-truth / gold answer) is optional so the same protocol
    covers both *verifiable* rewards (need a reference) and *reference-free*
    rewards (e.g. length penalties, reward models).

    Implementations must be deterministic and side-effect free given their
    inputs, and should return a finite float. Higher is better.
    """

    def __call__(self, prompt: str, completion: str, reference: Optional[str]) -> float:
        """Score a completion.

        Args:
            prompt: The input prompt the completion responds to.
            completion: The model-generated text to score.
            reference: Optional gold/reference answer.

        Returns:
            A scalar reward (higher is better).
        """
        ...


def _normalize_text(text: str, *, lower: bool = True, strip_punct: bool = False) -> str:
    """Normalize text for robust string comparison.

    Args:
        text: Input text.
        lower: Lowercase the text.
        strip_punct: Remove non-alphanumeric (keeping whitespace) characters.

    Returns:
        Whitespace-collapsed, optionally lowercased/depunctuated text.
    """
    out = text.strip()
    if lower:
        out = out.lower()
    if strip_punct:
        out = re.sub(r"[^\w\s]", "", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out


@dataclass
class ExactMatchReward:
    """Reward 1.0 when the normalized completion equals the reference, else 0.0.

    A verifiable reward suitable for short-answer / classification style tasks.

    Args:
        lower: Case-insensitive comparison (default: True).
        strip_punct: Strip punctuation before comparing (default: True).
        positive: Reward returned on a match (default: 1.0).
        negative: Reward returned on a mismatch (default: 0.0).
    """

    lower: bool = True
    strip_punct: bool = True
    positive: float = 1.0
    negative: float = 0.0

    def __call__(self, prompt: str, completion: str, reference: Optional[str]) -> float:
        if reference is None:
            return self.negative
        pred = _normalize_text(completion, lower=self.lower, strip_punct=self.strip_punct)
        gold = _normalize_text(reference, lower=self.lower, strip_punct=self.strip_punct)
        return self.positive if pred == gold else self.negative


# Matches integers, decimals, signed numbers, and simple comma grouping, e.g.
# "-12", "3.14", "1,024", "+0.5".
_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
# GSM8K-style explicit final answer marker "#### <answer>".
_GSM8K_MARKER_RE = re.compile(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)")


def _extract_final_number(text: str) -> Optional[float]:
    """Extract the final numeric answer from text (GSM8K-style).

    Prefers an explicit ``#### <number>`` marker (the GSM8K gold format); falls
    back to the last number appearing in the text. Comma thousands-separators are
    removed before parsing.

    Args:
        text: Text to extract a number from.

    Returns:
        The parsed float, or ``None`` if no number is present.
    """
    marker = _GSM8K_MARKER_RE.search(text)
    candidate: Optional[str] = None
    if marker:
        candidate = marker.group(1)
    else:
        matches = _NUMBER_RE.findall(text)
        if matches:
            candidate = matches[-1]
    if candidate is None:
        return None
    try:
        return float(candidate.replace(",", ""))
    except ValueError:
        return None


@dataclass
class NumericAnswerReward:
    """Verifiable reward that compares the *final numeric answer* (GSM8K-style).

    Extracts the last number (or the number after a ``####`` marker) from both the
    completion and the reference and rewards a numerically-close match. This is the
    canonical verifiable reward for math reasoning used by diffu-GRPO / d1
    (arXiv:2504.12216) and DeepSeekMath GRPO (arXiv:2402.03300).

    Args:
        rel_tol: Relative tolerance for the match (default: 0.0, exact).
        abs_tol: Absolute tolerance for the match (default: 1e-6).
        positive: Reward on a match (default: 1.0).
        negative: Reward when the numbers differ or are missing (default: 0.0).
    """

    rel_tol: float = 0.0
    abs_tol: float = 1e-6
    positive: float = 1.0
    negative: float = 0.0

    def __call__(self, prompt: str, completion: str, reference: Optional[str]) -> float:
        if reference is None:
            return self.negative
        pred = _extract_final_number(completion)
        gold = _extract_final_number(reference)
        if pred is None or gold is None:
            return self.negative
        tol = max(self.abs_tol, self.rel_tol * abs(gold))
        return self.positive if abs(pred - gold) <= tol else self.negative


@dataclass
class RegexMatchReward:
    """Verifiable reward that checks whether the completion matches a regex.

    Useful for format/structure constraints (e.g. "answer must contain
    ``\\boxed{...}``", "must be valid JSON-ish") that can be verified without a
    reference.

    Args:
        pattern: Regular expression to search for in the completion.
        flags: ``re`` flags (default: 0).
        positive: Reward on a match (default: 1.0).
        negative: Reward when there is no match (default: 0.0).
        use_reference_as_pattern: When True and a reference is given, the
            reference string is used as the pattern instead of ``pattern``
            (lets each example carry its own expected pattern).
    """

    pattern: str = ""
    flags: int = 0
    positive: float = 1.0
    negative: float = 0.0
    use_reference_as_pattern: bool = False
    _compiled: Optional[re.Pattern] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.pattern:
            self._compiled = re.compile(self.pattern, self.flags)

    def __call__(self, prompt: str, completion: str, reference: Optional[str]) -> float:
        if self.use_reference_as_pattern and reference is not None:
            matcher: Optional[re.Pattern] = re.compile(reference, self.flags)
        else:
            matcher = self._compiled
        if matcher is None:
            return self.negative
        return self.positive if matcher.search(completion) is not None else self.negative


@dataclass
class LengthPenaltyReward:
    """Reference-free reward that targets a desired completion length.

    Returns a non-positive penalty proportional to the deviation (in tokens, by
    whitespace split) from a target length window. Commonly *composed* with a
    verifiable correctness reward to discourage degenerate short/rambling outputs
    without dominating the correctness signal.

    Args:
        target_length: Desired completion length in tokens (default: 64).
        tolerance: No penalty inside ``target_length +/- tolerance`` (default: 16).
        penalty_per_token: Penalty magnitude per token outside the window
            (default: 0.01).
        max_penalty: Clamp on the total penalty magnitude (default: 1.0).
    """

    target_length: int = 64
    tolerance: int = 16
    penalty_per_token: float = 0.01
    max_penalty: float = 1.0

    def __call__(self, prompt: str, completion: str, reference: Optional[str]) -> float:
        length = len(completion.split())
        deviation = max(0, abs(length - self.target_length) - self.tolerance)
        penalty = min(self.max_penalty, deviation * self.penalty_per_token)
        return -penalty


@dataclass
class RewardModelReward:
    """Wrap an external scoring callable / reward model behind the protocol.

    The wrapped ``scorer`` may be any callable ``(prompt, completion, reference)
    -> float`` (for example a learned reward model's ``.score`` method, an LLM
    judge, or a heuristic). This keeps learned reward models pluggable without
    importing heavy dependencies into this module.

    Args:
        scorer: Callable returning a float score for a completion.
        scale: Multiplicative scaling applied to the raw score (default: 1.0).
        clip: Optional ``(low, high)`` clamp on the scaled score.
    """

    scorer: Callable[[str, str, Optional[str]], float]
    scale: float = 1.0
    clip: Optional[Tuple[float, float]] = None

    def __call__(self, prompt: str, completion: str, reference: Optional[str]) -> float:
        score = float(self.scorer(prompt, completion, reference)) * self.scale
        if self.clip is not None:
            low, high = self.clip
            score = max(low, min(high, score))
        return score


@dataclass
class CompositeReward:
    """Weighted sum of several rewards.

    Lets a verifiable correctness reward be combined with auxiliary shaping
    rewards (length, format). For example
    ``CompositeReward([(NumericAnswerReward(), 1.0), (LengthPenaltyReward(), 0.1)])``
    rewards correct answers while gently discouraging length blow-ups.

    Args:
        components: Sequence of ``(reward, weight)`` pairs.
    """

    components: Sequence[Tuple[Reward, float]]

    def __call__(self, prompt: str, completion: str, reference: Optional[str]) -> float:
        total = 0.0
        for reward, weight in self.components:
            total += weight * float(reward(prompt, completion, reference))
        return total


class CodeExecReward:
    """SAFE STUB: interface for reward-by-unit-test of generated code.

    Code-execution rewards (run generated code against hidden unit tests and
    reward the fraction of passing tests) are the gold standard for code-RL, but
    executing model-generated code is **untrusted code execution** and must never
    run inside the training process or on the host unsandboxed.

    This class intentionally **does not execute anything**. It only defines the
    interface and documents how to implement it safely. ``__call__`` raises
    :class:`NotImplementedError`.

    How to implement safely (do this in a separate, sandboxed service):
        1. Run each candidate in an isolated, ephemeral sandbox with **no
           network**, a read-only / throwaway filesystem, and a hard wall-clock
           and memory limit (e.g. a locked-down container, gVisor/Firecracker
           microVM, nsjail/bubblewrap, or a remote code-execution sandbox).
        2. Drop privileges; disable subprocess spawning and dangerous syscalls
           via a seccomp profile. Never use bare ``exec``/``eval`` or
           ``subprocess`` on the host.
        3. Provide the unit tests as fixed, trusted inputs; capture only the
           pass/fail count and stdout, never let the candidate write outside the
           sandbox.
        4. Map results to a reward, e.g. ``fraction_tests_passed`` (optionally
           ``1.0`` only if all tests pass), with a timeout/crash mapped to the
           minimum reward.

    Args:
        unit_tests: Per-example test code (trusted) keyed by example, or a single
            test harness string. Stored only; never executed here.
        timeout_s: Intended per-candidate wall-clock budget for a real sandbox.
        pass_all_required: If True, a correct implementation must pass all tests
            to receive ``1.0`` (otherwise fractional credit is intended).
    """

    def __init__(
        self,
        unit_tests: Optional[object] = None,
        timeout_s: float = 5.0,
        pass_all_required: bool = False,
    ) -> None:
        self.unit_tests = unit_tests
        self.timeout_s = timeout_s
        self.pass_all_required = pass_all_required

    def __call__(self, prompt: str, completion: str, reference: Optional[str]) -> float:
        raise NotImplementedError(
            "CodeExecReward is a safety stub and does not execute code. Implement "
            "test execution in an isolated, network-disabled, resource-limited "
            "sandbox (container/microVM/nsjail) in a separate service, then map "
            "the passing-test fraction to a reward. See the class docstring for "
            "the required sandboxing controls."
        )


def _strip_punct_tokens(text: str) -> List[str]:
    """Tokenize text into lowercased word tokens for overlap metrics.

    Args:
        text: Input text.

    Returns:
        List of lowercase alphanumeric tokens.
    """
    return re.findall(r"\w+", text.lower())


@dataclass
class TokenOverlapReward:
    """DEPRECATED WEAK PROXY: ``0.7 * token_F1 + 0.3 * bigram_precision``.

    WARNING:
        This reward measures surface token overlap between the completion and the
        reference. It is a **weak proxy** that primarily rewards *copying*: a model
        can maximize it by echoing reference (or prompt) tokens without producing a
        correct or coherent answer. It carries no notion of correctness, reasoning,
        or factuality. It is retained only for backward compatibility with the
        original DIMBA GRPO reward and for ablations. **Prefer a verifiable reward**
        (:class:`NumericAnswerReward`, :class:`ExactMatchReward`,
        :class:`RegexMatchReward`) or a learned :class:`RewardModelReward`.

    The score is ``0.7 * unigram_token_F1 + 0.3 * bigram_precision`` between the
    completion and the reference, in ``[0, 1]``. Returns ``0.0`` when no reference
    is provided or either side is empty.

    Args:
        f1_weight: Weight on unigram token F1 (default: 0.7).
        bigram_weight: Weight on bigram precision (default: 0.3).
    """

    f1_weight: float = 0.7
    bigram_weight: float = 0.3

    def __call__(self, prompt: str, completion: str, reference: Optional[str]) -> float:
        if reference is None:
            return 0.0
        pred = _strip_punct_tokens(completion)
        gold = _strip_punct_tokens(reference)
        if not pred or not gold:
            return 0.0
        return self.f1_weight * _token_f1(pred, gold) + self.bigram_weight * _bigram_precision(
            pred, gold
        )


def _token_f1(pred: Sequence[str], gold: Sequence[str]) -> float:
    """Unigram token F1 (multiset overlap) between two token sequences."""
    from collections import Counter

    if not pred or not gold:
        return 0.0
    cp, cg = Counter(pred), Counter(gold)
    overlap = sum((cp & cg).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(gold)
    return 2.0 * precision * recall / max(1e-8, precision + recall)


def _bigram_precision(pred: Sequence[str], gold: Sequence[str]) -> float:
    """Bigram precision of ``pred`` against ``gold``."""
    from collections import Counter

    if len(pred) < 2 or len(gold) < 2:
        return 0.0
    bp = Counter(tuple(pred[i : i + 2]) for i in range(len(pred) - 1))
    bg = Counter(tuple(gold[i : i + 2]) for i in range(len(gold) - 1))
    overlap = sum((bp & bg).values())
    return overlap / max(1, sum(bp.values()))


# Registry of zero-argument reward factories for CLI selection. Verifiable
# rewards are listed first and are the recommended defaults.
REWARD_REGISTRY: dict = {
    "exact_match": ExactMatchReward,
    "numeric": NumericAnswerReward,
    "regex": RegexMatchReward,
    "length_penalty": LengthPenaltyReward,
    "token_overlap": TokenOverlapReward,
}


def get_reward(name: str, **kwargs: object) -> Reward:
    """Construct a reward by name from :data:`REWARD_REGISTRY`.

    Args:
        name: Registry key (e.g. ``"numeric"``, ``"exact_match"``,
            ``"token_overlap"``).
        **kwargs: Forwarded to the reward constructor.

    Returns:
        An instantiated :class:`Reward`.

    Raises:
        KeyError: If ``name`` is not registered.
    """
    if name not in REWARD_REGISTRY:
        raise KeyError(
            f"Unknown reward '{name}'. Available: {sorted(REWARD_REGISTRY)}."
        )
    return REWARD_REGISTRY[name](**kwargs)
