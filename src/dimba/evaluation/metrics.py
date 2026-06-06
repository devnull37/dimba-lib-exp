"""Evaluation metrics for DIMBA."""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional
import warnings

try:
    from sacrebleu import corpus_bleu
except ImportError:
    corpus_bleu = None

try:
    from rouge_score import rouge_scorer
except ImportError:
    rouge_scorer = None

try:
    from nltk.translate.meteor_score import meteor_score
except ImportError:
    meteor_score = None


def compute_perplexity(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    ignore_index: int = -100,
) -> float:
    """Compute perplexity from logits and targets.

    Args:
        logits: Model logits [batch_size, seq_len, vocab_size]
        target_ids: Target token IDs [batch_size, seq_len]
        ignore_index: Index to ignore in loss computation

    Returns:
        Perplexity score
    """
    # Reshape for loss computation
    logits = logits.view(-1, logits.size(-1))
    target_ids = target_ids.view(-1)

    # Compute cross-entropy loss
    loss = F.cross_entropy(
        logits,
        target_ids,
        ignore_index=ignore_index,
        reduction="mean",
    )

    # Perplexity is e^loss
    perplexity = torch.exp(loss).item()

    return perplexity


def compute_bleu(
    predictions: List[str],
    references: List[List[str]],
    max_n: int = 4,
    smooth_method: str = "exp",
) -> float:
    """Compute BLEU score.

    Args:
        predictions: List of generated sequences
        references: List of reference sequences (multiple references per prediction)
        max_n: Maximum n-gram order (default: 4 for BLEU-4)
        smooth_method: Smoothing method

    Returns:
        BLEU score (0-1)
    """
    if corpus_bleu is None:
        warnings.warn("sacrebleu not installed. Install with: pip install sacrebleu")
        return 0.0

    try:
        score = corpus_bleu(predictions, [references], max_n=max_n, smooth_method=smooth_method)
        return score.score / 100.0  # Convert to 0-1 range
    except Exception as e:
        warnings.warn(f"Failed to compute BLEU: {e}")
        return 0.0


def compute_rouge(
    predictions: List[str],
    references: List[str],
    rouge_types: List[str] = ["rouge1", "rouge2", "rougeL"],
) -> Dict[str, float]:
    """Compute ROUGE scores.

    Args:
        predictions: List of generated sequences
        references: List of reference sequences
        rouge_types: ROUGE variants to compute

    Returns:
        Dictionary of ROUGE scores
    """
    if rouge_scorer is None:
        warnings.warn("rouge_score not installed. Install with: pip install rouge_score")
        return {}

    scorer = rouge_scorer.RougeScorer(rouge_types=rouge_types, use_stemmer=True)

    scores = {rouge_type: [] for rouge_type in rouge_types}

    for pred, ref in zip(predictions, references):
        try:
            result = scorer.score(ref, pred)
            for rouge_type in rouge_types:
                scores[rouge_type].append(result[rouge_type].fmeasure)
        except Exception as e:
            warnings.warn(f"Failed to compute ROUGE for pair: {e}")

    # Average scores
    avg_scores = {
        rouge_type: np.mean(scores[rouge_type]) if scores[rouge_type] else 0.0
        for rouge_type in rouge_types
    }

    return avg_scores


def compute_meteor(
    predictions: List[str],
    references: List[str],
) -> float:
    """Compute METEOR score.

    Args:
        predictions: List of generated sequences
        references: List of reference sequences

    Returns:
        Average METEOR score
    """
    if meteor_score is None:
        warnings.warn("nltk not installed. Install with: pip install nltk")
        return 0.0

    try:
        scores = []
        for pred, ref in zip(predictions, references):
            pred_tokens = pred.split()
            ref_tokens = ref.split()
            score = meteor_score([ref_tokens], pred_tokens)
            scores.append(score)

        return np.mean(scores) if scores else 0.0
    except Exception as e:
        warnings.warn(f"Failed to compute METEOR: {e}")
        return 0.0


class MetricsLogger:
    """Simple logger for tracking multiple metrics during training/evaluation.

    Useful for organizing and printing metrics results.
    """

    def __init__(self):
        self.metrics = {}

    def log_scalar(self, name: str, value: float):
        """Log a scalar metric."""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)

    def get_average(self, name: str) -> float:
        """Get average value for a metric."""
        if name not in self.metrics or not self.metrics[name]:
            return 0.0
        return np.mean(self.metrics[name])

    def get_all_averages(self) -> Dict[str, float]:
        """Get averages for all metrics."""
        return {name: self.get_average(name) for name in self.metrics}

    def reset(self):
        """Reset all metrics."""
        self.metrics = {}

    def print_summary(self):
        """Print summary of all metrics."""
        print("\n=== Metrics Summary ===")
        for name, avg in self.get_all_averages().items():
            print(f"{name}: {avg:.4f}")
        print("=" * 22 + "\n")


def evaluate_generation(
    predictions: List[str],
    references: List[str],
    with_bleu: bool = True,
    with_rouge: bool = True,
    with_meteor: bool = False,
) -> Dict[str, float]:
    """Compute multiple evaluation metrics for generated text.

    Args:
        predictions: List of generated sequences
        references: List of reference sequences
        with_bleu: Whether to compute BLEU
        with_rouge: Whether to compute ROUGE
        with_meteor: Whether to compute METEOR

    Returns:
        Dictionary of all computed metrics
    """
    results = {}

    if with_bleu:
        bleu_score = compute_bleu(predictions, [[ref] for ref in references])
        results["bleu"] = bleu_score

    if with_rouge:
        rouge_scores = compute_rouge(predictions, references)
        results.update(rouge_scores)

    if with_meteor:
        meteor_avg = compute_meteor(predictions, references)
        results["meteor"] = meteor_avg

    return results


def compute_model_perplexity(
    model: torch.nn.Module,
    dataloader,
    device: str = "cuda",
) -> float:
    """Compute denoising-NLL perplexity of a DIMBA model on a dataset.

    For each batch, samples random diffusion timesteps, runs the forward
    denoising pass, projects to token logits via output_head, and computes
    cross-entropy against the true token ids. Returns exp(mean NLL).

    Args:
        model: DIMBA model
        dataloader: DataLoader yielding dicts with "input_ids"
        device: Device to run on

    Returns:
        exp(mean denoising NLL) -- denoising-reconstruction perplexity
    """
    from ..diffusion.sampling import sample_timesteps

    model.eval()
    total_nll = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            batch_size = input_ids.shape[0]

            # Sample random diffusion timesteps
            t = sample_timesteps(batch_size, model.num_diffusion_steps, torch.device(device))

            # Forward denoising pass
            x_pred, _, _ = model(input_ids, t)

            # Project to token logits
            logits = model.output_head(x_pred, embedding_weight=model.token_embed.get_weight())

            # Cross-entropy NLL vs true token ids
            nll = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                input_ids.reshape(-1),
                reduction="sum",
            )
            total_nll += nll.item()
            total_tokens += input_ids.numel()

    avg_nll = total_nll / max(1, total_tokens)
    return float(np.exp(avg_nll))


def distinct_n(texts: List[str], n: int) -> float:
    """Compute distinct-n diversity metric.

    Ratio of unique n-grams (whitespace-tokenized) to total n-grams across
    all texts. Returns 0.0 if there are no n-grams.

    Args:
        texts: List of generated text strings
        n: n-gram order (1 for unigrams, 2 for bigrams, etc.)

    Returns:
        Ratio of unique n-grams to total n-grams in [0, 1]
    """
    all_ngrams = []
    for text in texts:
        tokens = text.split()
        ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
        all_ngrams.extend(ngrams)
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


def self_bleu(texts: List[str]) -> Optional[float]:
    """Compute self-BLEU diversity metric (mean pairwise BLEU).

    Lower self-BLEU indicates higher diversity. Requires sacrebleu.
    Returns None if sacrebleu is not available.

    Args:
        texts: List of generated text strings

    Returns:
        Mean pairwise corpus BLEU score (0-1), or None if sacrebleu unavailable
    """
    if corpus_bleu is None:
        warnings.warn("sacrebleu not installed; self_bleu returning None. Install with: pip install sacrebleu")
        return None

    if len(texts) < 2:
        return 0.0

    scores = []
    try:
        for i, hypothesis in enumerate(texts):
            references = [t for j, t in enumerate(texts) if j != i]
            score = corpus_bleu([hypothesis], [references])
            scores.append(score.score / 100.0)
        return float(np.mean(scores)) if scores else 0.0
    except Exception as e:
        warnings.warn(f"Failed to compute self-BLEU: {e}")
        return None
