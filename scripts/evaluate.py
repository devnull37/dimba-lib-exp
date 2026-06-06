#!/usr/bin/env python3
"""Evaluation script for DIMBA model.

Usage:
    # Basic evaluation (requires a real text file and tokenizer)
    python scripts/evaluate.py --checkpoint checkpoints/dimba.ckpt \\
        --data data/shakespeare.txt --tokenizer checkpoints/shakespeare/tokenizer.json

    # Evaluate with speed benchmarking
    python scripts/evaluate.py --checkpoint checkpoints/dimba.ckpt \\
        --data data/shakespeare.txt --tokenizer checkpoints/shakespeare/tokenizer.json \\
        --eval-speed

    # Evaluate on CPU
    python scripts/evaluate.py --checkpoint checkpoints/dimba.ckpt \\
        --data data/shakespeare.txt --tokenizer checkpoints/shakespeare/tokenizer.json \\
        --device cpu
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

# Add src to path
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = (SCRIPT_DIR / ".." / "src").resolve()
sys.path.insert(0, str(SRC_DIR))

from dimba import DIMBA
from dimba.evaluation import compute_model_perplexity
from dimba.evaluation.metrics import distinct_n, self_bleu


def _load_tokenizer_auto(path):
    """Auto-detect tokenizer type from JSON and load it."""
    import json
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "char_to_id" in data:
        from dimba.tokenizers.simple import SimpleCharacterTokenizer
        tok = SimpleCharacterTokenizer()
        tok.load(path)
        return tok
    from dimba.tokenizers.bpe import BPETokenizer
    tok = BPETokenizer()
    tok.load(path)
    return tok


def load_checkpoint(checkpoint_path: str, device: str):
    """Load DIMBA model from checkpoint using stored hyper_parameters."""
    print(f"Loading checkpoint from {checkpoint_path}...")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hp = ckpt.get("hyper_parameters", {})

    if not hp:
        raise ValueError(
            "Checkpoint has no 'hyper_parameters'. Cannot rebuild model with correct architecture. "
            "Use --vocab-size and ensure the checkpoint was saved by DIMBA Lightning trainer."
        )

    vocab_size = hp["vocab_size"]
    model_config = dict(hp["model_config"])
    model = DIMBA(vocab_size=vocab_size, **model_config)

    # Extract model weights from Lightning state_dict
    sd = {k[len("model."):]: v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    real_missing = [m for m in missing if not m.startswith("noise_schedule.")]
    if real_missing or unexpected:
        print(f"  [warn] missing={real_missing[:6]} unexpected={list(unexpected)[:6]}")

    model.to(device).eval()
    return model


def build_eval_dataloader(text: str, tokenizer, seq_length: int, batch_size: int, val_frac: float = 0.1):
    """Tokenize text, take last val_frac as held-out, return a DataLoader."""
    ids = tokenizer.encode(text)
    split = int(len(ids) * (1 - val_frac))
    val_ids = ids[split:]

    n = (len(val_ids) // seq_length) * seq_length
    if n == 0:
        raise ValueError(
            f"Not enough held-out tokens ({len(val_ids)}) to form a single window of {seq_length}. "
            "Use a larger --data file or smaller --seq-length."
        )
    val_ids = val_ids[:n]
    windows = torch.tensor(val_ids, dtype=torch.long).view(-1, seq_length)

    dataset = TensorDataset(windows)

    def collate(batch):
        return {"input_ids": torch.stack([b[0] for b in batch])}

    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)


def evaluate_inference_speed(model: DIMBA, seq_len: int = 256, num_runs: int = 10, device: str = 'cuda'):
    """Measure inference speed with different numbers of diffusion steps."""
    print("\nMeasuring inference speed...")
    print("-" * 60)

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

                prompt_cond = model.encode_prompt(prompt_ids)
                x_t = torch.randn(batch_size, seq_len, model.d_model, device=device)
                alphas = model.get_alphas_cumprod().to(device)
                timesteps = torch.linspace(num_steps - 1, 0, num_steps, dtype=torch.long, device=device)

                for t in timesteps[:5]:
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
    parser.add_argument('--data', type=str, required=True, help='Path to held-out text file for evaluation')
    parser.add_argument('--tokenizer', type=str, required=True, help='Path to tokenizer JSON file')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--seq-length', type=int, default=256, help='Sequence length for evaluation windows')
    parser.add_argument('--val-frac', type=float, default=0.1, help='Fraction of text to use as held-out set')
    parser.add_argument('--eval-speed', action='store_true', help='Evaluate inference speed')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    print("=" * 60)
    print("DIMBA Model Evaluation")
    print("=" * 60)

    # Load model from checkpoint using stored hyper_parameters
    model = load_checkpoint(args.checkpoint, args.device)
    print(f"\nModel loaded successfully!")
    print(f"Vocab size: {model.vocab_size}")
    print(f"Model dimension: {model.d_model}")
    print(f"Device: {args.device}")

    # Load tokenizer
    print(f"\nLoading tokenizer from {args.tokenizer}...")
    tokenizer = _load_tokenizer_auto(args.tokenizer)

    # Build held-out evaluation dataloader from real text
    print(f"\nPreparing held-out evaluation dataset from {args.data}...")
    text = Path(args.data).read_text()
    eval_loader = build_eval_dataloader(
        text, tokenizer,
        seq_length=args.seq_length,
        batch_size=args.batch_size,
        val_frac=args.val_frac,
    )
    print(f"  Evaluation windows: {len(eval_loader.dataset)}")

    # Compute denoising-NLL perplexity on real held-out data
    print("\nEvaluating model perplexity on held-out set...")
    print("-" * 60)
    ppl = compute_model_perplexity(model, eval_loader, device=args.device)
    print(f"Denoising-NLL Perplexity: {ppl:.4f}")

    # Decode a sample of held-out windows and compute diversity metrics
    print("\nDiversity metrics (held-out reconstructions):")
    print("-" * 60)
    sample_texts = []
    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch["input_ids"].to(args.device)
            t = torch.zeros(input_ids.shape[0], dtype=torch.long, device=args.device)  # t=0: clean
            x_pred, _, _ = model(input_ids, t)
            logits = model.output_head(x_pred, embedding_weight=model.token_embed.get_weight())
            pred_ids = logits.argmax(dim=-1)
            for seq in pred_ids:
                try:
                    decoded = tokenizer.decode(seq.tolist())
                    sample_texts.append(decoded)
                except Exception:
                    pass
            if len(sample_texts) >= 100:
                break

    if sample_texts:
        d1 = distinct_n(sample_texts, 1)
        d2 = distinct_n(sample_texts, 2)
        print(f"Distinct-1: {d1:.4f}")
        print(f"Distinct-2: {d2:.4f}")
        sb = self_bleu(sample_texts[:50])
        if sb is not None:
            print(f"Self-BLEU:  {sb:.4f}  (lower = more diverse)")
    else:
        print("  (no decodeable samples)")

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
