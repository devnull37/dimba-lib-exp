#!/usr/bin/env python3
"""Honest held-out perplexity for a DIMBA continuous-latent diffusion checkpoint.

Why this script exists
----------------------
DIMBA is *not* autoregressive, so there is no ``p(token | previous tokens)`` to
read off and multiply -- the standard GPT-style perplexity is undefined here.
What we *can* measure cleanly is the model's **token-reconstruction perplexity
across noise levels**: for held-out text, embed -> encode to latent -> add the
schedule's Gaussian noise at level ``t`` -> denoise -> decode -> softmax head ->
cross-entropy vs. the true tokens. Averaged over the noise schedule this is an
honest measure of held-out fit (it is the per-timestep denoising NLL whose
schedule-weighted sum is the diffusion ELBO term).

Caveats (read before quoting the number):
  * This is a *proxy*, not the gold-standard autoregressive perplexity. At low
    noise it is trivially easy (the model nearly sees clean tokens); at high
    noise it approaches the data prior. The single headline number is the
    *uniform-over-timesteps* average -- "expected denoising difficulty".
  * The repo's own ``compute_model_perplexity`` returns ``exp(MSE(embed))``,
    which is dimensionally meaningless; this replaces it.

For interpretation we also print trivial baselines on the *same* held-out text
(uniform-over-vocab and a unigram char model), so the number actually means
something.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = (SCRIPT_DIR / ".." / "src").resolve()
sys.path.insert(0, str(SRC_DIR))

from dimba import DIMBA  # noqa: E402


def _load_tokenizer_auto(path):
    import json
    with open(path) as f: data = json.load(f)
    if isinstance(data, dict) and "char_to_id" in data:
        from dimba.tokenizers.simple import SimpleCharacterTokenizer
        tok = SimpleCharacterTokenizer(); tok.load(path); return tok
    from dimba.tokenizers.bpe import BPETokenizer
    tok = BPETokenizer(); tok.load(path); return tok


def load_dimba(ckpt_path: str, device: str) -> DIMBA:
    """Rebuild DIMBA from the checkpoint's stored hparams and load weights strictly."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = ckpt.get("hyper_parameters", {})
    vocab_size = hp["vocab_size"]
    model_config = dict(hp["model_config"])
    model = DIMBA(vocab_size=vocab_size, **model_config)
    # Pull the (non-EMA) model weights out of the Lightning state_dict.
    sd = {k[len("model."):]: v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # Buffers (schedule) are fine to be "missing" if recomputed; flag real gaps.
    real_missing = [m for m in missing if not m.startswith("noise_schedule.")]
    if real_missing or unexpected:
        print(f"  [warn] missing={real_missing[:6]} unexpected={list(unexpected)[:6]}")
    model.to(device).eval()
    return model


def load_tokenizer(tok_path: str):
    return _load_tokenizer_auto(tok_path)


@torch.no_grad()
def denoising_ppl(model, ids, device, n_noise=2, t_stride=1, seed=0):
    """Return per-timestep mean NLL (nats/token) over the held-out windows.

    ids: LongTensor [N, L] of held-out token windows.
    Returns dict: {t_index: mean_nll}, evaluated at every ``t_stride``-th step.
    """
    T = model.num_diffusion_steps
    gen = torch.Generator(device="cpu").manual_seed(seed)
    timesteps = list(range(0, T, t_stride))
    per_t = {}
    N, L = ids.shape
    BS = 128
    for t_idx in timesteps:
        tot_nll, tot_tok = 0.0, 0
        for _ in range(n_noise):
            for b0 in range(0, N, BS):
                batch = ids[b0 : b0 + BS].to(device)
                t = torch.full((batch.shape[0],), t_idx, dtype=torch.long, device=device)
                noise = torch.randn(*batch.shape, model.d_latent, generator=gen).to(device)
                x_pred, _, _ = model(batch, t, noise=noise)          # decoded embedding [B,L,d_model]
                logits = model.output_head(x_pred)                   # [B,L,V] (mirrors sampler)
                nll = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), batch.reshape(-1), reduction="sum"
                )
                tot_nll += nll.item()
                tot_tok += batch.numel()
        per_t[t_idx] = tot_nll / tot_tok
    return per_t


def baselines(train_ids, val_ids, vocab_size):
    """Uniform and unigram char baselines (nats/token) on the val set."""
    counts = torch.bincount(train_ids.reshape(-1), minlength=vocab_size).float()
    probs = (counts + 1.0) / (counts.sum() + vocab_size)            # Laplace-smoothed
    val_nll = -torch.log(probs[val_ids.reshape(-1)]).mean().item()
    used = int((counts > 0).sum().item())
    return {
        "uniform_full_vocab": math.log(vocab_size),
        "uniform_used_chars": math.log(used),
        "unigram": val_nll,
    }


def chunk(ids_list, L):
    n = (len(ids_list) // L) * L
    return torch.tensor(ids_list[:n], dtype=torch.long).view(-1, L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/shakespeare/shakespeare1.ckpt")
    ap.add_argument("--tokenizer", default="checkpoints/shakespeare/tokenizer.json")
    ap.add_argument("--data", default="data/shakespeare.txt")
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--max-windows", type=int, default=1000)
    ap.add_argument("--n-noise", type=int, default=2)
    ap.add_argument("--t-stride", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    print(f"Loading {args.ckpt} ...")
    model = load_dimba(args.ckpt, args.device)
    tok = load_tokenizer(args.tokenizer)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  d_model={model.d_model} d_latent={model.d_latent} T={model.num_diffusion_steps} "
          f"learnable_params={n_params:,}")

    text = Path(args.data).read_text()
    ids = tok.encode(text)
    split = int(len(ids) * (1 - args.val_frac))
    train_ids = torch.tensor(ids[:split], dtype=torch.long)
    val = chunk(ids[split:], args.seq_len)
    if val.shape[0] > args.max_windows:
        val = val[: args.max_windows]
    print(f"  held-out: {val.shape[0]} windows x {args.seq_len} chars "
          f"(last {args.val_frac:.0%} of corpus)\n")

    bl = baselines(train_ids, val, model.vocab_size)
    per_t = denoising_ppl(model, val, args.device, n_noise=args.n_noise, t_stride=args.t_stride)

    ts = sorted(per_t)
    avg_nll = sum(per_t[t] for t in ts) / len(ts)     # uniform-over-timesteps
    low_nll = per_t[ts[0]]                              # cleanest level (round-trip)
    hi_nll = per_t[ts[-1]]                              # near pure noise

    def ppl(n):
        return math.exp(n)

    def bpc(n):
        return n / math.log(2)

    print("=== Baselines (held-out, nats/char -> ppl) ===")
    for k, v in bl.items():
        print(f"  {k:20s}  nll={v:.3f}  ppl={ppl(v):7.2f}  bpc={bpc(v):.2f}")
    print("\n=== DIMBA denoising reconstruction (held-out) ===")
    print(f"  low-noise  t={ts[0]:<3d}  nll={low_nll:.3f}  ppl={ppl(low_nll):7.2f}  bpc={bpc(low_nll):.2f}   (round-trip fidelity)")
    print(f"  high-noise t={ts[-1]:<3d}  nll={hi_nll:.3f}  ppl={ppl(hi_nll):7.2f}  bpc={bpc(hi_nll):.2f}   (near pure noise)")
    print(f"  TIME-AVG          nll={avg_nll:.3f}  ppl={ppl(avg_nll):7.2f}  bpc={bpc(avg_nll):.2f}   <-- headline proxy")

    # Compact curve
    print("\n  denoising curve (ppl vs t):")
    for t in ts[:: max(1, len(ts) // 12)]:
        bar = "#" * int(min(60, ppl(per_t[t])))
        print(f"    t={t:>3d}  ppl={ppl(per_t[t]):7.2f}  {bar}")

    print("\nNOTE: this is a denoising-reconstruction proxy, not autoregressive PPL.")
    print(json.dumps({"time_avg_ppl": ppl(avg_nll), "low_noise_ppl": ppl(low_nll),
                      "unigram_ppl": ppl(bl['unigram'])}, indent=0))


if __name__ == "__main__":
    main()
