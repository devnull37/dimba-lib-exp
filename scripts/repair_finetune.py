"""Repair finetune: fix the t~1.0 high-noise dead zone in the budget28 model.

Resumes checkpoints/distill_budget28_collapsed/final.pt and trains with the
FIXED objective (snr_floor=0.5 + ce_time_fade=True, see
docs/ROOT_CAUSE_BUDGET28.md), oversampling high noise levels: 50% of timesteps
uniform over t in [0.7, 1.0] (the dead zone and its approach), 50% logit-normal
(the production distribution, protects the intact t<=0.9 skill).

Every EVAL_EVERY steps: saves a checkpoint and prints 2 free generations from
pure noise so coherence progress is visible directly in the log.

Usage: repair_finetune.py [--steps N] [--batch B] [--smoke]
"""
import argparse
import inspect
import time

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from dimba import DIMBA
from dimba.training import compute_dimba_losses
from dimba.diffusion.sampling import sample_from_model_flow

CKPT = "checkpoints/distill_budget28_collapsed/final.pt"
OUT_DIR = "checkpoints/repair"
SEQ_LEN = 512
LR = 1e-4
WARMUP = 100
EVAL_EVERY = 1000
DEVICE = "cuda"

EVAL_PROMPTS = [
    "The capital of France is",
    "Once upon a time there",
]


class PackedChunks(Dataset):
    def __init__(self, stream: torch.Tensor, seq_len: int):
        self.stream, self.seq_len = stream, seq_len
        self.n = max(0, stream.numel() // seq_len)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.stream[i * self.seq_len:(i + 1) * self.seq_len]


def build_stream(tokenizer, n_docs: int) -> torch.Tensor:
    from datasets import load_dataset
    print(f"streaming FineWeb sample-10BT, caching {n_docs} docs ...", flush=True)
    ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT",
                      split="train", streaming=True)
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


def sample_repair_timesteps(model, batch_size: int, device: str) -> torch.Tensor:
    """50% uniform over t in [0.7, 1.0] (dead zone), 50% logit-normal (production)."""
    T = model.num_diffusion_steps
    n_hi = batch_size // 2
    hi = torch.randint(int(0.7 * (T - 1)), T, (n_hi,), device=device)
    ln = model.noise_schedule.sample_timesteps(batch_size - n_hi, device,
                                               mode="logit_normal")
    t = torch.cat([hi, ln.to(device)])
    return t[torch.randperm(batch_size, device=device)]


@torch.no_grad()
def eval_generations(model, tokenizer, step: int):
    model.eval()
    for p in EVAL_PROMPTS:
        ids = torch.tensor([tokenizer.encode(p, add_special_tokens=False)],
                           dtype=torch.long, device=DEVICE)
        g = sample_from_model_flow(model, ids, seq_len=80, num_steps=50,
                                   sampler="heun", temperature=0.3, top_k=50,
                                   device=DEVICE)
        txt = tokenizer.decode([x for x in g[0].tolist() if x < 49152])
        print(f"[step {step}] {p!r} -> {txt!r}", flush=True)
    model.train()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--docs", type=int, default=200_000)
    ap.add_argument("--smoke", action="store_true",
                    help="20 steps, tiny cache, no saves")
    args = ap.parse_args()
    if args.smoke:
        args.steps, args.batch, args.docs = 20, 8, 300

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")

    ck = torch.load(CKPT, map_location="cpu")
    cfg = dict(ck["config"])
    sd = ck["model_state_dict"]
    # Adopt the checkpoint's actual vocab rows (may include block-CoT extras).
    emb_key = next(k for k in sd if k.endswith("token_embed.embedding.weight"))
    cfg["vocab_size"] = sd[emb_key].shape[0]
    sig = set(inspect.signature(DIMBA.__init__).parameters) - {"self"}
    model = DIMBA(**{k: v for k, v in cfg.items() if k in sig})
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f"loaded {CKPT}: vocab={cfg['vocab_size']} "
          f"missing={len(miss)} unexpected={len(unexp)}", flush=True)
    assert not miss, f"missing keys: {miss[:5]}"
    model = model.to(DEVICE).to(torch.bfloat16)
    del ck, sd

    stream = build_stream(tokenizer, args.docs)
    loader = DataLoader(PackedChunks(stream, SEQ_LEN), batch_size=args.batch,
                        shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / WARMUP) *
        (0.5 * (1 + torch.cos(torch.tensor(min(s, args.steps) / args.steps * 3.14159)).item())
         * 0.9 + 0.1))

    if not args.smoke:
        import os
        os.makedirs(OUT_DIR, exist_ok=True)
        print("baseline generations (before repair):", flush=True)
        eval_generations(model, tokenizer, 0)

    model.train()
    step, t0, it = 0, time.time(), iter(loader)
    while step < args.steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        batch = batch.to(DEVICE, non_blocking=True)
        t = sample_repair_timesteps(model, batch.shape[0], DEVICE)
        loss, parts = compute_dimba_losses(
            model, batch, t,
            ce_loss_weight=1.0, min_snr_gamma=5.0,
            snr_floor=0.5, ce_time_fade=True,   # THE FIX
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1
        if step % 100 == 0 or step == 1:
            ps = {k: f"{v.item():.4f}" for k, v in parts.items() if hasattr(v, "item")}
            tok_s = step * args.batch * SEQ_LEN / (time.time() - t0)
            print(f"step {step}/{args.steps} loss={loss.item():.4f} {ps} "
                  f"lr={sched.get_last_lr()[0]:.2e} {tok_s / 1e3:.0f}k tok/s "
                  f"({time.time() - t0:.0f}s)", flush=True)
        if not args.smoke and step % EVAL_EVERY == 0:
            torch.save({"model_state_dict": model.state_dict(), "config": cfg},
                       f"{OUT_DIR}/repaired_latest.pt")
            eval_generations(model, tokenizer, step)

    if args.smoke:
        eval_generations(model, tokenizer, step)
    else:
        torch.save({"model_state_dict": model.state_dict(), "config": cfg},
                   f"{OUT_DIR}/repaired_final.pt")
    print("REPAIR_DONE", flush=True)


if __name__ == "__main__":
    main()
