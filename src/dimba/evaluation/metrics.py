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
    compute_bleu: bool = True,
    compute_rouge: bool = True,
    compute_meteor: bool = False,
) -> Dict[str, float]:
    """Compute multiple evaluation metrics for generated text.

    Args:
        predictions: List of generated sequences
        references: List of reference sequences
        compute_bleu: Whether to compute BLEU
        compute_rouge: Whether to compute ROUGE
        compute_meteor: Whether to compute METEOR

    Returns:
        Dictionary of all computed metrics
    """
    results = {}

    if compute_bleu:
        bleu_score = compute_bleu(predictions, [[ref] for ref in references])
        results["bleu"] = bleu_score

    if compute_rouge:
        rouge_scores = compute_rouge(predictions, references)
        results.update(rouge_scores)

    if compute_meteor:
        meteor_avg = compute_meteor(predictions, references)
        results["meteor"] = meteor_avg

    return results


def compute_model_perplexity(
    model: torch.nn.Module,
    dataloader,
    device: str = "cuda",
) -> float:
    """Compute perplexity of model on a dataset.

    Args:
        model: DIMBA model
        dataloader: DataLoader with batches
        device: Device to run on

    Returns:
        Average perplexity across dataset
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            batch_size = input_ids.shape[0]

            # Sample random timesteps
            from ..diffusion.sampling import sample_timesteps
            t = sample_timesteps(batch_size, model.num_diffusion_steps, torch.device(device))

            # Forward pass
            x_pred, _ = model(input_ids, t)

            # Get clean embeddings
            x_0 = model.token_embed(input_ids)

            # Compute loss
            loss = torch.nn.functional.mse_loss(x_pred, x_0)
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / max(1, num_batches)
    # Convert MSE loss to approximate perplexity
    perplexity = np.exp(min(avg_loss, 10))  # Cap to prevent overflow

    return perplexity
