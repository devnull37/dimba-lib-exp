"""Masked discrete diffusion (LLaDA/MDLM-style) conversion of the DIMBA backbone.

The continuous-latent track is a triangulated negative at 135M (fixed loss,
SFT, GRPO, noisy-t KD all failed to produce commitment from pure noise — see
docs/ROOT_CAUSE_BUDGET28.md and the KD run). This switches the *objective*:
corrupt by replacing tokens with a new [MASK] token, predict the originals
with plain cross-entropy on masked positions (LLaDA 1/t weighting). No SNR
weighting, no latent blending — both proven pathologies vanish by
construction. The bidirectional Mamba backbone and its 28B-token English
knowledge are reused: init from checkpoints/repair/repaired_final.pt, with
one extra embedding row for [MASK] (init = mean embedding). The repo's
predict_token_logits() is the intended entry point for exactly this track
(mask-ratio in (0,1] is mapped onto the timestep conditioning).

Generation: MaskGIT-style iterative unmasking (confidence-based, cosine
schedule). Probes + infill-recovery gate + checkpoint every EVAL_EVERY steps.

Usage: masked_diffusion_finetune.py [--steps N] [--batch B] [--smoke]
"""
import argparse
import inspect
import math
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from dimba import DIMBA

CKPT = "checkpoints/masked_diffusion/mdm_final.pt"
OUT_DIR = "checkpoints/masked_diffusion2"
SEQ_LEN = 512
LR = 1e-4
WARMUP = 200
EVAL_EVERY = 500
T_MIN = 0.03          # avoid the 1/t blowup; LLaDA clamps similarly
DEVICE = "cuda"

EVAL_PROMPTS = [
    "The capital of France is",
    "Once upon a time there",
]
GATE_TEXT = ("The quick brown fox jumps over the lazy dog while the sun sets "
             "behind the mountains and the river flows quietly through the valley.")


class PackedChunks(Dataset):
    def __init__(self, stream: torch.Tensor, seq_len: int):
        self.stream, self.seq_len = stream, seq_len
        self.n = max(0, stream.numel() // seq_len)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.stream[i * self.seq_len:(i + 1) * self.seq_len]


def build_stream(tokenizer, n_docs: int, skip_docs: int = 0) -> torch.Tensor:
    from datasets import load_dataset
    print(f"streaming FineWeb sample-10BT, skipping {skip_docs}, "
          f"caching {n_docs} docs ...", flush=True)
    ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT",
                      split="train", streaming=True)
    if skip_docs:
        ds = ds.skip(skip_docs)
    eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    chunks = []
    for i, row in enumerate(ds):
        if i >= n_docs:
            break
        enc = tokenizer.encode(row["text"], add_special_tokens=False)
        enc.append(eos)
        chunks.append(torch.tensor(enc, dtype=torch.long))
        if (i + 1) % 20000 == 0:
            print(f"  {i + 1}/{n_docs} docs", flush=True)
    stream = torch.cat(chunks)
    print(f"cache ready: {stream.numel() / 1e6:.1f}M tokens", flush=True)
    return stream


def masked_diffusion_loss(model, input_ids, mask_id: int):
    """LLaDA-style: mask each token w.p. t (t ~ U[T_MIN, 1]), CE on masked, 1/t weight."""
    B, L = input_ids.shape
    t = torch.rand(B, device=input_ids.device) * (1.0 - T_MIN) + T_MIN
    mask = torch.rand(B, L, device=input_ids.device) < t[:, None]
    # ensure at least one masked position per row
    none_masked = ~mask.any(dim=1)
    if none_masked.any():
        mask[none_masked, 0] = True
    corrupted = torch.where(mask, mask_id, input_ids)

    logits = model.predict_token_logits(corrupted, t)  # [B, L, V]
    V = logits.shape[-1]
    ce = F.cross_entropy(logits.reshape(-1, V), input_ids.reshape(-1),
                         reduction="none").view(B, L)
    m = mask.float()
    ce_masked = (ce * m).sum(dim=1) / m.sum(dim=1)      # per-sample mean over masked
    loss = (ce_masked / t).mean()                        # LLaDA 1/t weighting
    return loss, {"ce_masked": ce_masked.mean(), "t_mean": t.mean()}


@torch.no_grad()
def maskgit_generate(model, prompt_ids: torch.Tensor, gen_len: int, mask_id: int,
                     steps: int = 32, temperature: float = 0.7, top_k: int = 50):
    """Iterative unmasking with a cosine schedule; prompt stays fixed."""
    device = prompt_ids.device
    B, P = prompt_ids.shape
    ids = torch.cat([prompt_ids,
                     torch.full((B, gen_len), mask_id, dtype=torch.long,
                                device=device)], dim=1)
    still_masked = torch.zeros(B, P + gen_len, dtype=torch.bool, device=device)
    still_masked[:, P:] = True
    for s in range(steps):
        frac_masked = still_masked.float().mean().item()
        logits = model.predict_token_logits(ids, max(frac_masked, T_MIN))
        logits = logits.float() / temperature
        if top_k:
            kth = logits.topk(top_k, dim=-1).values[..., -1:]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        probs = F.softmax(logits, dim=-1)
        sampled = torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(B, -1)
        conf = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
        conf = conf.masked_fill(~still_masked, float("inf"))  # keep decided tokens
        # cosine schedule: fraction still masked after this step
        keep_masked = math.cos(math.pi / 2 * (s + 1) / steps)
        n_keep = int(gen_len * keep_masked)
        ids = torch.where(still_masked, sampled, ids)
        if n_keep > 0:
            # re-mask the n_keep lowest-confidence generated positions
            remask = torch.zeros_like(still_masked)
            idx = conf.argsort(dim=1)[:, :n_keep]
            remask.scatter_(1, idx, True)
            remask &= still_masked
            ids = torch.where(remask, mask_id, ids)
            still_masked = remask
        else:
            still_masked = torch.zeros_like(still_masked)
            break
    return ids


@torch.no_grad()
def eval_step(model, tokenizer, step: int, mask_id: int):
    model.eval()
    for p in EVAL_PROMPTS:
        ids = torch.tensor([tokenizer.encode(p, add_special_tokens=False)],
                           dtype=torch.long, device=DEVICE)
        g = maskgit_generate(model, ids, gen_len=64, mask_id=mask_id)
        txt = tokenizer.decode([x for x in g[0].tolist() if x < mask_id])
        print(f"[step {step}] {p!r} -> {txt!r}", flush=True)
    # infill-recovery gate: mask 50% of a held-out sentence, measure recovery
    gt = torch.tensor([tokenizer.encode(GATE_TEXT, add_special_tokens=False)],
                      dtype=torch.long, device=DEVICE)
    torch.manual_seed(0)
    m = torch.rand_like(gt, dtype=torch.float) < 0.5
    corrupted = torch.where(m, mask_id, gt)
    logits = model.predict_token_logits(corrupted, 0.5)
    pred = logits.argmax(dim=-1)
    acc = (pred[m] == gt[m]).float().mean().item()
    print(f"[step {step}] infill@50% recovery: {acc:.3f}", flush=True)
    model.train()
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--docs", type=int, default=200_000)
    ap.add_argument("--skip-docs", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.steps, args.batch, args.docs = 20, 8, 300

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")

    ck = torch.load(CKPT, map_location="cpu")
    cfg = dict(ck["config"])
    sd = ck["model_state_dict"]
    emb_key = next(k for k in sd if k.endswith("token_embed.embedding.weight"))
    if "mask_id" in ck:
        # resuming a masked-diffusion checkpoint: [MASK] row already present
        mask_id = ck["mask_id"]
    else:
        old_vocab = sd[emb_key].shape[0]
        mask_id = old_vocab
        # one new row for [MASK], init = mean embedding (head is tied, so this is all)
        sd[emb_key] = torch.cat([sd[emb_key],
                                 sd[emb_key].mean(dim=0, keepdim=True)], dim=0)
        cfg["vocab_size"] = old_vocab + 1
    sig = set(inspect.signature(DIMBA.__init__).parameters) - {"self"}
    model = DIMBA(**{k: v for k, v in cfg.items() if k in sig})
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f"loaded {CKPT}: vocab={cfg['vocab_size']} mask_id={mask_id} "
          f"missing={len(miss)} unexpected={len(unexp)}", flush=True)
    assert not miss, f"missing keys: {miss[:5]}"
    model = model.to(DEVICE).to(torch.bfloat16)
    del ck, sd

    stream = build_stream(tokenizer, args.docs, args.skip_docs)
    loader = DataLoader(PackedChunks(stream, SEQ_LEN), batch_size=args.batch,
                        shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / WARMUP) *
        (0.5 * (1 + math.cos(min(s, args.steps) / args.steps * math.pi)) * 0.9 + 0.1))

    if not args.smoke:
        import os
        os.makedirs(OUT_DIR, exist_ok=True)
        print("baseline (repaired_final backbone, before masked-diffusion training):",
              flush=True)
        eval_step(model, tokenizer, 0, mask_id)

    model.train()
    step, t0, it = 0, time.time(), iter(loader)
    while step < args.steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        batch = batch.to(DEVICE, non_blocking=True)
        loss, parts = masked_diffusion_loss(model, batch, mask_id)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1
        if step % 100 == 0 or step == 1:
            ps = {k: f"{v.item():.4f}" for k, v in parts.items()}
            tok_s = step * args.batch * SEQ_LEN / (time.time() - t0)
            print(f"step {step}/{args.steps} loss={loss.item():.4f} {ps} "
                  f"lr={sched.get_last_lr()[0]:.2e} {tok_s / 1e3:.0f}k tok/s "
                  f"({time.time() - t0:.0f}s)", flush=True)
        if not args.smoke and step % EVAL_EVERY == 0:
            torch.save({"model_state_dict": model.state_dict(), "config": cfg,
                        "mask_id": mask_id},
                       f"{OUT_DIR}/mdm_latest.pt")
            eval_step(model, tokenizer, step, mask_id)

    if args.smoke:
        eval_step(model, tokenizer, step, mask_id)
    else:
        torch.save({"model_state_dict": model.state_dict(), "config": cfg,
                    "mask_id": mask_id},
                   f"{OUT_DIR}/mdm_final.pt")
    print("MDM_DONE", flush=True)


if __name__ == "__main__":
    main()
