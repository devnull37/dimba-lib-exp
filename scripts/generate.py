#!/usr/bin/env python3
"""Generation script for DIMBA model.

Usage:
    # Generate with default settings
    python scripts/generate.py --checkpoint checkpoints/dimba.ckpt --prompt "Hello world"

    # Generate with DDIM sampling
    python scripts/generate.py --checkpoint checkpoints/dimba.ckpt --prompt "Hello world" --use-ddim

    # Generate multiple samples
    python scripts/generate.py --checkpoint checkpoints/dimba.ckpt --prompt "Hello world" --num-samples 3
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml

# Add src to path
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = (SCRIPT_DIR / ".." / "src").resolve()
sys.path.insert(0, str(SRC_DIR))

from dimba import DIMBA, DDIMSampler, sample_from_model


def load_checkpoint(checkpoint_path: str, vocab_size: int, device: str, config_path: str = None):
    """Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        vocab_size: Vocabulary size (used if config not provided)
        device: Device to load model on
        config_path: Optional path to config.yaml with model hyperparameters
    """
    print(f"Loading checkpoint from {checkpoint_path}...")

    # Try to load as Lightning checkpoint first
    try:
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

    # Load config if provided
    model_kwargs = {'vocab_size': vocab_size}
    if config_path:
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            if 'model' in config:
                model_config = config['model']
                model_kwargs.update({
                    'd_model': model_config.get('d_model', 512),
                    'd_prompt': model_config.get('d_prompt', 512),
                    'num_diffusion_steps': model_config.get('num_diffusion_steps', 1000),
                    'num_denoiser_layers': model_config.get('num_denoiser_layers', 6),
                    'd_state': model_config.get('d_state', 16),
                    'd_conv': model_config.get('d_conv', 4),
                    'expand': model_config.get('expand', 2),
                    'conditioning_type': model_config.get('conditioning_type', 'film'),
                    'dropout': model_config.get('dropout', 0.1),
                    'use_weight_tying': model_config.get('use_weight_tying', False),
                })
                print(f"Loaded model config: d_model={model_kwargs['d_model']}, vocab_size={vocab_size}")
        except Exception as e:
            print(f"Warning: Failed to load config from {config_path}: {e}")
            print("Using default model parameters")

    # Create model with loaded config
    model = DIMBA(**model_kwargs)
    model.load_state_dict(model_state, strict=False)
    model = model.to(device)
    model.eval()

    return model


def main():
    parser = argparse.ArgumentParser(description="Generate text with DIMBA")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config.yaml file')
    parser.add_argument('--prompt', type=str, default='The quick brown fox', help='Prompt text')
    parser.add_argument('--length', type=int, default=100, help='Length of text to generate')
    parser.add_argument('--num-steps', type=int, default=50, help='Number of denoising steps')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--top-k', type=int, default=None, help='Top-k sampling')
    parser.add_argument('--top-p', type=float, default=0.95, help='Top-p (nucleus) sampling')
    parser.add_argument('--use-ddim', action='store_true', help='Use DDIM sampling')
    parser.add_argument('--vocab-size', type=int, default=10000, help='Vocabulary size')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num-samples', type=int, default=1, help='Number of samples to generate')

    args = parser.parse_args()

    print("=" * 50)
    print("DIMBA Generation")
    print("=" * 50)

    # Load model
    model = load_checkpoint(args.checkpoint, args.vocab_size, args.device, args.config)

    print(f"\nModel loaded successfully!")
    print(f"Prompt: {args.prompt}")
    print(f"Generation length: {args.length}")
    print(f"Number of samples: {args.num_samples}")
    print("-" * 50)

    # Simple tokenizer (just split by spaces for demo)
    def tokenize(text):
        return [ord(c) % args.vocab_size for c in text]

    def detokenize(token_ids):
        # Convert token IDs back to characters (inverse of tokenize)
        # Since tokenize uses ord(c) % vocab_size, try to recover characters
        try:
            chars = [chr(int(t)) for t in token_ids.tolist() if 32 <= int(t) < 127]
            if chars:
                text = ''.join(chars)
            else:
                text = f"<non-printable tokens>"
        except:
            text = f"<decoding error>"

        # Show both text and token IDs
        return f"{text}\n  (Token IDs: {token_ids.tolist()[:30]}...)"

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
