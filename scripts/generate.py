#!/usr/bin/env python3
"""DIMBA generation with an accuracy-vs-speed slider.

A single ``--quality`` knob (Q in [0, 1]) trades wall-clock time for output
quality by scaling BOTH the number of iterative unmasking steps AND the number
of best-of-N candidates that get ranked by a guidance-gap verifier.

Production sampler: MaskGIT-style iterative unmasking with a cosine remask
schedule, classifier-free guidance (2.0), an exempt-first frequency penalty
(0.7), temperature (0.7), and top-k (20) multinomial sampling. The logic here
mirrors the validated reference implementations in
``scripts/experiments/selfcorrect_test.py`` (sampler), ``bestofn2.py``
(guidance-gap verifier), and ``critic_bon.py`` (critic head).

CLI:
    python scripts/generate.py "What is the capital of France?" --quality 0.5

Python API:
    from generate import slider_generate
    out = slider_generate(model, tokenizer, mask_id, "...", quality=0.5)
"""
from __future__ import annotations

import argparse
import inspect
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

# Allow ``PYTHONPATH=src`` runs as well as direct ``python scripts/generate.py``.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from dimba import DIMBA  # noqa: E402

# --------------------------------------------------------------------------- #
# Constants (validated defaults; overridable via CLI expert flags).
# --------------------------------------------------------------------------- #
DEVICE = "cuda"
DTYPE = torch.bfloat16
T_MIN = 0.03                # floor for the fraction-masked t passed to the model
DEFAULT_GEN_LEN = 40
DEFAULT_TEMPERATURE = 0.7
DEFAULT_GUIDANCE = 2.0
DEFAULT_TOP_K = 20
DEFAULT_FREQ_PEN = 0.7
DEFAULT_CKPT = os.path.join(
    _REPO_ROOT, "checkpoints", "mdm_sft_cfg2", "mdm_sft_final.pt"
)
TOKENIZER_NAME = "HuggingFaceTB/SmolLM-135M"
CRITIC_CKPT = os.path.join(_REPO_ROOT, "checkpoints", "critic_head.pt")
CRITIC_TIE_BAND = 0.02      # gap within this band -> break tie with critic


# --------------------------------------------------------------------------- #
# Loading.
# --------------------------------------------------------------------------- #
def load_model(path: str = DEFAULT_CKPT):
    """Load the flagship DIMBA checkpoint. Returns (model, mask_id)."""
    ck = torch.load(path, map_location="cpu")
    cfg, mask_id = dict(ck["config"]), ck["mask_id"]
    sig = set(inspect.signature(DIMBA.__init__).parameters) - {"self"}
    model = DIMBA(**{k: v for k, v in cfg.items() if k in sig})
    model.load_state_dict(ck["model_state_dict"], strict=False)
    model = model.to(DEVICE).to(DTYPE).eval()
    return model, mask_id


def load_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TOKENIZER_NAME)


def load_critic():
    """Load the trained critic head MLP: Linear(dim,256)+GELU+Linear(256,1)."""
    ck = torch.load(CRITIC_CKPT, map_location="cpu")
    dim = ck["dim"]
    critic = torch.nn.Sequential(
        torch.nn.Linear(dim, 256), torch.nn.GELU(), torch.nn.Linear(256, 1)
    )
    state = {k.replace("net.", ""): v for k, v in ck["critic_state_dict"].items()}
    critic.load_state_dict(state)
    return critic.to(DEVICE).to(DTYPE).eval()


# --------------------------------------------------------------------------- #
# Sampler (replicates selfcorrect_test.py: guided_logits + generate).
# --------------------------------------------------------------------------- #
@torch.no_grad()
def guided_logits(model, mask_id, ids, prompt_len, t, guidance):
    """Classifier-free guidance: lu + guidance * (lc - lu)."""
    lc = model.predict_token_logits(ids, t).float()
    u = ids.clone()
    u[:, :prompt_len] = mask_id
    lu = model.predict_token_logits(u, t).float()
    return lu + guidance * (lc - lu)


@torch.no_grad()
def generate(
    model,
    mask_id,
    prompt_ids,
    gen_len=DEFAULT_GEN_LEN,
    steps=128,
    temperature=DEFAULT_TEMPERATURE,
    top_k=DEFAULT_TOP_K,
    freq_pen=DEFAULT_FREQ_PEN,
    guidance=DEFAULT_GUIDANCE,
):
    """MaskGIT-style iterative unmasking with a cosine remask schedule."""
    B, P = prompt_ids.shape
    ids = torch.cat(
        [
            prompt_ids,
            torch.full((B, gen_len), mask_id, dtype=torch.long, device=DEVICE),
        ],
        dim=1,
    )
    still = torch.zeros(B, P + gen_len, dtype=torch.bool, device=DEVICE)
    still[:, P:] = True
    for s in range(steps):
        frac = still.float().mean().item()
        logits = guided_logits(model, mask_id, ids, P, max(frac, T_MIN), guidance)
        # Exempt-first frequency penalty over already-committed generated tokens.
        for b in range(B):
            comm = ids[b, P:][~still[b, P:]]
            if comm.numel():
                uniq, cnt = comm.unique(return_counts=True)
                logits[b, :, uniq] -= freq_pen * (cnt - 1).clamp(min=0).float()
        # Temperature AFTER penalty/guidance, then top-k multinomial sampling.
        logits = logits / temperature
        kth = logits.topk(top_k, dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
        probs = F.softmax(logits, dim=-1)
        sampled = torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(B, -1)
        conf = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
        conf = conf.masked_fill(~still, float("inf"))
        n_keep = int(gen_len * math.cos(math.pi / 2 * (s + 1) / steps))
        ids = torch.where(still, sampled, ids)
        if n_keep > 0:
            remask = torch.zeros_like(still)
            remask.scatter_(1, conf.argsort(dim=1)[:, :n_keep], True)
            remask &= still
            ids = torch.where(remask, mask_id, ids)
            still = remask
        else:
            break
    return ids


# --------------------------------------------------------------------------- #
# Verifiers.
# --------------------------------------------------------------------------- #
@torch.no_grad()
def gap_score(model, mask_id, eos, ids, prompt_len, K=4):
    """Guidance-gap verifier (bestofn2.py): mean over generated positions of
    [logp_cond - logp_uncond] for the committed token, with the scored
    positions MASKED via K interleaved leave-k-out passes at t=0.25.
    Higher = better."""
    B, L = ids.shape
    gen = torch.zeros_like(ids, dtype=torch.bool)
    gen[:, prompt_len:] = (ids[:, prompt_len:] != eos) & (ids[:, prompt_len:] != mask_id)
    pos = torch.arange(L, device=ids.device)[None, :].expand(B, L)
    tot = torch.zeros(B, device=ids.device)
    n = torch.zeros(B, device=ids.device)
    for r in range(K):
        m = gen & (pos % K == r)
        if not m.any():
            continue
        x = ids.masked_fill(m, mask_id)
        lc = F.log_softmax(model.predict_token_logits(x, 0.25).float(), -1)
        u = x.clone()
        u[:, :prompt_len] = mask_id
        lu = F.log_softmax(model.predict_token_logits(u, 0.25).float(), -1)
        tokp = ids.unsqueeze(-1)
        gap = (lc.gather(-1, tokp) - lu.gather(-1, tokp)).squeeze(-1)
        tot += (gap * m).sum(1)
        n += m.sum(1)
    return tot / n.clamp(min=1)


@torch.no_grad()
def _xdec(model, ids, t):
    """Decoded x0 latent features ("x_dec") used by the critic head."""
    B = ids.shape[0]
    z = model.encode_latent(model.token_embed(ids))
    cond = model._build_conditioning(None, B, ids.device)
    ti = model._to_timestep_index(t, B, ids.device)
    raw = model._denoiser_raw(z, ti, cond, None)
    return model.decode_latent(model._to_x0_latent(z, raw, ti))


@torch.no_grad()
def critic_score(model, critic, mask_id, eos, ids, prompt_len):
    """Mean predicted wrongness over generated tokens (lower = better)."""
    gen = torch.zeros_like(ids, dtype=torch.bool)
    gen[:, prompt_len:] = (ids[:, prompt_len:] != mask_id) & (ids[:, prompt_len:] != eos)
    s = torch.sigmoid(critic(_xdec(model, ids, 0.15)).squeeze(-1).float())
    return (s * gen).sum(1) / gen.sum(1).clamp(min=1)


# --------------------------------------------------------------------------- #
# Quality slider mapping.
# --------------------------------------------------------------------------- #
def _geom_interp(q, lo, mid, hi):
    """Geometric interpolation lo (q=0) -> mid (q=0.5) -> hi (q=1.0)."""
    q = max(0.0, min(1.0, q))
    if q <= 0.5:
        f = q / 0.5
        return lo * (mid / lo) ** f
    f = (q - 0.5) / 0.5
    return mid * (hi / mid) ** f


def steps_for_quality(q):
    return int(round(_geom_interp(q, 16.0, 64.0, 256.0)))


def n_for_quality(q):
    if q < 0.35:
        return 1
    if q < 0.6:
        return 2
    if q < 0.85:
        return 4
    return 8


# --------------------------------------------------------------------------- #
# Decoding.
# --------------------------------------------------------------------------- #
def decode(tokenizer, ids_row, prompt_len, mask_id, eos):
    """Drop tokens >= mask_id and truncate at the first EOS."""
    toks = [x for x in ids_row[prompt_len:].tolist() if x < mask_id]
    if eos in toks:
        toks = toks[: toks.index(eos)]
    return tokenizer.decode(toks)


# --------------------------------------------------------------------------- #
# Public API.
# --------------------------------------------------------------------------- #
@torch.no_grad()
def slider_generate(model, tokenizer, mask_id, question, quality=0.5,
                    critic=None, **overrides):
    """Generate an answer with the accuracy-vs-speed slider.

    Args:
        model, tokenizer, mask_id: as loaded by ``load_*``.
        question: the raw question string.
        quality: Q in [0, 1] mapping to (steps, N).
        critic: optional loaded critic head; used to break near-ties in gap.
        overrides: any of steps, n, temperature, guidance, seed, gen_len,
            top_k, freq_pen.

    Returns:
        dict with keys: text, steps, n, seconds, scores (list, empty when N==1),
        plus best (index) and all_texts.
    """
    eos = tokenizer.eos_token_id
    # ponytail: `or` treats falsy overrides (e.g. --guidance 0) as unset; use None-check like freq_pen did.
    ov = lambda k, default: default if overrides.get(k) is None else overrides.get(k)
    steps = int(ov("steps", steps_for_quality(quality)))
    n = int(ov("n", n_for_quality(quality)))
    gen_len = int(ov("gen_len", DEFAULT_GEN_LEN))
    temperature = float(ov("temperature", DEFAULT_TEMPERATURE))
    guidance = float(ov("guidance", DEFAULT_GUIDANCE))
    top_k = int(ov("top_k", DEFAULT_TOP_K))
    freq_pen = float(ov("freq_pen", DEFAULT_FREQ_PEN))
    seed = overrides.get("seed")

    if seed is not None:
        torch.manual_seed(int(seed))

    prompt = f"Question: {question}\nAnswer:"
    prompt_ids = torch.tensor(
        [tokenizer.encode(prompt, add_special_tokens=False)] * n, device=DEVICE
    )
    P = prompt_ids.shape[1]

    torch.cuda.synchronize()
    t0 = time.time()

    ids = generate(
        model, mask_id, prompt_ids, gen_len=gen_len, steps=steps,
        temperature=temperature, top_k=top_k, freq_pen=freq_pen, guidance=guidance,
    )

    scores = []
    if n == 1:
        best = 0
    else:
        gap = gap_score(model, mask_id, eos, ids, P)
        scores = gap.tolist()
        order = sorted(range(n), key=lambda i: -scores[i])
        best = order[0]
        if critic is not None:
            # Break near-ties (gap within band) by lower critic wrongness.
            top_gap = scores[best]
            tied = [i for i in range(n) if top_gap - scores[i] <= CRITIC_TIE_BAND]
            if len(tied) > 1:
                cs = critic_score(model, critic, mask_id, eos, ids, P).tolist()
                best = min(tied, key=lambda i: cs[i])

    torch.cuda.synchronize()
    seconds = time.time() - t0

    text = decode(tokenizer, ids[best], P, mask_id, eos)
    all_texts = [decode(tokenizer, ids[i], P, mask_id, eos) for i in range(n)]
    return {
        "text": text,
        "steps": steps,
        "n": n,
        "seconds": seconds,
        "scores": scores,
        "best": best,
        "all_texts": all_texts,
    }


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser(
        description="DIMBA generation with an accuracy-vs-speed slider."
    )
    p.add_argument("question", help="the question to answer")
    p.add_argument("--quality", type=float, default=0.5,
                   help="Q in [0,1]: higher = slower & better (default 0.5)")
    p.add_argument("--checkpoint", default=DEFAULT_CKPT, help="model checkpoint")
    # Expert overrides.
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--n", type=int, default=None, help="best-of-N candidates")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--guidance", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--gen-len", type=int, default=None, dest="gen_len")
    p.add_argument("--critic", action="store_true",
                   help="load the critic head to break near-ties in gap")
    p.add_argument("--show-all", action="store_true",
                   help="print all candidates with scores")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    model, mask_id = load_model(args.checkpoint)
    tokenizer = load_tokenizer()
    critic = load_critic() if args.critic else None

    overrides = {}
    for k in ("steps", "n", "temperature", "guidance", "seed", "gen_len"):
        v = getattr(args, k)
        if v is not None:
            overrides[k] = v

    out = slider_generate(
        model, tokenizer, mask_id, args.question,
        quality=args.quality, critic=critic, **overrides,
    )

    if args.show_all and out["n"] > 1:
        print("Candidates:")
        for i in range(out["n"]):
            score = out["scores"][i] if out["scores"] else float("nan")
            mark = " *" if i == out["best"] else "  "
            print(f"{mark}[{i}] gap={score:+.3f}  {out['all_texts'][i]}")
        print()

    print(f"quality      : {args.quality}")
    print(f"steps        : {out['steps']}")
    print(f"N candidates : {out['n']}")
    print(f"wall-clock   : {out['seconds']:.2f} s")
    print(f"answer       : {out['text']}")


if __name__ == "__main__":
    main()
