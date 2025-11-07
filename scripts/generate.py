#!/usr/bin/env python3
"""Generation script for DIMBA model."""

import argparse
import torch
import yaml
import sys

sys.path.insert(0, str(__file__).rsplit('/', 1)[0] + '/../src')

from dimba import DIMBA, sample_from_model, DDIMSampler


def load_checkpoint(checkpoint_path: str, vocab_size: int, device: str):
    """Load model from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}...")

    # Try to load as Lightning checkpoint first
    try:
        from pytorch_lightning import LightningModule
        import pytorch_lightning as pl

        # Load Lightning checkpoint
        state_dict = torch.load(checkpoint_path, map_location=device)

        # Extract model state dict if it's a Lightning checkpoint
        if 'state_dict' in state_dict:
            model_state = {}
            for k, v in state_dict['state_dict'].items():
                # Remove 'model.' prefix from Lightning
                if k.startswith('model.'):
                    model_state[k[6:]] = v
                else:
                    model_state[k] = v
        else:
            model_state = state_dict

    except Exception as e:
        print(f"Failed to load as Lightning checkpoint: {e}")
        # Try loading as raw model checkpoint
        model_state = torch.load(checkpoint_path, map_location=device)

    # Create model
    model = DIMBA(vocab_size=vocab_size)
    model.load_state_dict(model_state, strict=False)
    model = model.to(device)
    model.eval()

    return model


def main():
    parser = argparse.ArgumentParser(description="Generate text with DIMBA")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--prompt', type=str, default='The quick brown fox', help='Prompt text')
    parser.add_argument('--length', type=int, default=100, help='Length of text to generate')
    parser.add_argument('--num-steps', type=int, default=50, help='Number of denoising steps')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--top-k', type=int, default=None, help='Top-k sampling')
    parser.add_argument('--top-p', type=float, default=0.95, help='Top-p (nucleus) sampling')
    parser.add_argument('--use-ddim', action='store_true', help='Use DDIM sampling')
    parser.add_argument('--vocab-size', type=int, default=50000, help='Vocabulary size')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num-samples', type=int, default=1, help='Number of samples to generate')

    args = parser.parse_args()

    print("=" * 50)
    print("DIMBA Generation")
    print("=" * 50)

    # Load model
    model = load_checkpoint(args.checkpoint, args.vocab_size, args.device)

    print(f"\nModel loaded successfully!")
    print(f"Prompt: {args.prompt}")
    print(f"Generation length: {args.length}")
    print(f"Number of samples: {args.num_samples}")
    print("-" * 50)

    # Simple tokenizer (just split by spaces for demo)
    def tokenize(text):
        return [ord(c) % args.vocab_size for c in text]

    def detokenize(token_ids):
        # For demo, just convert to readable format
        return f"Generated text (IDs: {token_ids.tolist()[:20]}...)"

    # Tokenize prompt
    prompt_ids = torch.tensor([tokenize(args.prompt)], device=args.device)

    # Generate
    with torch.no_grad():
        if args.use_ddim:
            sampler = DDIMSampler(model, num_steps=args.num_steps, ddim_eta=0.0)
            generated = sampler.sample(
                prompt_ids,
                seq_len=args.length,
                temperature=args.temperature,
            )
        else:
            generated = sample_from_model(
                model,
                prompt_ids,
                seq_len=args.length,
                num_steps=args.num_steps,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                device=args.device,
            )

    print("\nGenerated samples:")
    print("-" * 50)
    for i, gen in enumerate(generated):
        print(f"\nSample {i + 1}:")
        print(detokenize(gen))

    print("\n" + "=" * 50)


if __name__ == '__main__':
    main()
