#!/usr/bin/env python3
"""Evaluation script for DIMBA model."""

import argparse
import torch
import sys
import time

sys.path.insert(0, str(__file__).rsplit('/', 1)[0] + '/../src')

from dimba import DIMBA
from dimba.data import DummyDataset, collate_fn
from dimba.evaluation import compute_model_perplexity, MetricsLogger
from torch.utils.data import DataLoader


def load_checkpoint(checkpoint_path: str, vocab_size: int, device: str):
    """Load model from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}...")

    try:
        state_dict = torch.load(checkpoint_path, map_location=device)

        # Extract model state dict if it's a Lightning checkpoint
        if 'state_dict' in state_dict:
            model_state = {}
            for k, v in state_dict['state_dict'].items():
                if k.startswith('model.'):
                    model_state[k[6:]] = v
                else:
                    model_state[k] = v
        else:
            model_state = state_dict

    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        model_state = torch.load(checkpoint_path, map_location=device)

    # Create model
    model = DIMBA(vocab_size=vocab_size)
    model.load_state_dict(model_state, strict=False)
    model = model.to(device)
    model.eval()

    return model


def evaluate_inference_speed(model: DIMBA, seq_len: int = 256, num_runs: int = 10, device: str = 'cuda'):
    """Measure inference speed with different numbers of diffusion steps."""
    print("\nMeasuring inference speed...")
    print("-" * 60)

    # Dummy prompt
    batch_size = 1
    prompt_ids = torch.randint(0, model.vocab_size, (batch_size, 10)).to(device)

    step_counts = [10, 25, 50, 100]
    results = {}

    for num_steps in step_counts:
        if num_steps > model.num_diffusion_steps:
            continue

        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.time()

                # Encode prompt
                prompt_cond = model.encode_prompt(prompt_ids)

                # Initialize noise
                x_t = torch.randn(batch_size, seq_len, model.d_model, device=device)

                # Simple denoising loop
                alphas = model.get_alphas_cumprod().to(device)
                timesteps = torch.linspace(num_steps - 1, 0, num_steps, dtype=torch.long, device=device)

                for t in timesteps[:5]:  # Only a few steps for speed test
                    t_tensor = torch.full((batch_size,), t.item(), dtype=torch.long, device=device)
                    x_pred = model.denoise_step(x_t, t_tensor, prompt_cond)
                    x_t = x_pred

                elapsed = time.time() - start
                times.append(elapsed)

        avg_time = sum(times) / len(times)
        results[num_steps] = avg_time
        print(f"Steps: {num_steps:3d} | Avg time: {avg_time:.4f}s | Throughput: {batch_size / avg_time:.2f} samples/s")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate DIMBA model")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--vocab-size', type=int, default=50000, help='Vocabulary size')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--seq-length', type=int, default=256, help='Sequence length')
    parser.add_argument('--num-batches', type=int, default=10, help='Number of batches to evaluate')
    parser.add_argument('--eval-speed', action='store_true', help='Evaluate inference speed')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    print("=" * 60)
    print("DIMBA Model Evaluation")
    print("=" * 60)

    # Load model
    model = load_checkpoint(args.checkpoint, args.vocab_size, args.device)
    print(f"\nModel loaded successfully!")
    print(f"Vocab size: {args.vocab_size}")
    print(f"Model dimension: {model.d_model}")
    print(f"Device: {args.device}")

    # Create evaluation dataset
    print(f"\nPreparing evaluation dataset...")
    eval_dataset = DummyDataset(
        size=args.num_batches * args.batch_size,
        vocab_size=args.vocab_size,
        seq_length=args.seq_length,
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Compute perplexity
    print("\nEvaluating model perplexity on test set...")
    print("-" * 60)
    ppl = compute_model_perplexity(model, eval_loader, device=args.device)
    print(f"Average Perplexity: {ppl:.4f}")

    # Model statistics
    print("\nModel Statistics:")
    print("-" * 60)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Evaluate inference speed
    if args.eval_speed:
        evaluate_inference_speed(model, seq_len=args.seq_length, device=args.device)

    print("\n" + "=" * 60)
    print("Evaluation complete!")


if __name__ == '__main__':
    main()
