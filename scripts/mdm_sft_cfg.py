"""SFT for the masked-diffusion model (LLaDA-style instruction tuning).

Resumes checkpoints/masked_diffusion/mdm_latest.pt. Each example is a fixed
SEQ_LEN row: [prompt | response | EOS padding]. The prompt is NEVER masked
(always clean conditioning); the response region — including the EOS padding,
which is how the model learns to end answers — is masked at ratio t ~ U[T_MIN,1]
and trained with CE on masked positions, 1/t weighted. Run GRPO only after the
SFT checkpoint passes the parseability and held-out quality gates.

Data: tatsu-lab/alpaca (52k, no trust_remote_code), template
"Question: {instruction}\n{input}\nAnswer: {output}".

Usage: mdm_sft.py [--steps N] [--batch B] [--smoke]
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

CKPT = "checkpoints/masked_diffusion2/mdm_latest.pt"
OUT_DIR = "checkpoints/mdm_sft_cfg"
P_DROP = 0.1          # fraction of rows trained with a null (fully masked) prompt
GUIDANCE = 2.5        # eval-time CFG scale: uncond + s * (cond - uncond)
SEQ_LEN = 256
LR = 5e-5
WARMUP = 100
EVAL_EVERY = 500
T_MIN = 0.03
DEVICE = "cuda"

EVAL_PROMPTS = [
    "Question: What is the capital of France?\nAnswer:",
    "Question: Write one sentence about the ocean.\nAnswer:",
    "Question: What is 2 + 3?\nAnswer:",
]


class SFTRows(Dataset):
    """Fixed-length rows + prompt/response lengths, prebuilt on CPU."""

    def __init__(self, rows, plens, rlens):
        self.rows, self.plens, self.rlens = rows, plens, rlens

    def __len__(self):
        return self.rows.shape[0]

    def __getitem__(self, i):
        return self.rows[i], self.plens[i], self.rlens[i]


def build_sft_rows(tokenizer):
    from datasets import load_dataset
    print("loading tatsu-lab/alpaca ...", flush=True)
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    rows, plens, rlens = [], [], []
    for ex in ds:
        q = ex["instruction"]
        if ex.get("input"):
            q += "\n" + ex["input"]
        p = tokenizer.encode(f"Question: {q}\nAnswer:", add_special_tokens=False)
        r = tokenizer.encode(" " + ex["output"], add_special_tokens=False)
        if len(p) >= SEQ_LEN - 8:  # need room for at least a stub answer
            continue
        row = (p + r)[:SEQ_LEN - 1] + [eos]
        rlen = len(row)  # content + exactly one EOS; padding beyond is NOT trained
        row = row + [eos] * (SEQ_LEN - len(row))
        rows.append(row)
        plens.append(len(p))
        rlens.append(rlen)
    rows = torch.tensor(rows, dtype=torch.long)
    plens = torch.tensor(plens, dtype=torch.long)
    rlens = torch.tensor(rlens, dtype=torch.long)
    print(f"SFT rows: {rows.shape[0]} x {SEQ_LEN}", flush=True)
    return SFTRows(rows, plens, rlens)


def sft_loss(model, rows, plens, rlens, mask_id: int):
    B, L = rows.shape
    t = torch.rand(B, device=rows.device) * (1.0 - T_MIN) + T_MIN
    pos = torch.arange(L, device=rows.device)[None, :]
    # response + first EOS only; v1 trained the whole EOS pad tail and the
    # model collapsed to predicting EOS everywhere (empty answers)
    resp = (pos >= plens[:, None]) & (pos < rlens[:, None])
    mask = (torch.rand(B, L, device=rows.device) < t[:, None]) & resp
    none_masked = ~mask.any(dim=1)
    if none_masked.any():
        first_resp = plens.clamp(max=L - 1)
        mask[none_masked, first_resp[none_masked]] = True
    corrupted = torch.where(mask, mask_id, rows)
    # CFG prompt dropout: null-condition P_DROP of rows by masking their prompt
    drop = torch.rand(B, device=rows.device) < P_DROP
    corrupted = torch.where(drop[:, None] & (pos < plens[:, None]), mask_id,
                            corrupted)

    logits = model.predict_token_logits(corrupted, t)
    V = logits.shape[-1]
    ce = F.cross_entropy(logits.reshape(-1, V), rows.reshape(-1),
                         reduction="none").view(B, L)
    m = mask.float()
    ce_masked = (ce * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
    loss = (ce_masked / t).mean()
    return loss, {"ce_masked": ce_masked.mean(), "t_mean": t.mean()}


@torch.no_grad()
def maskgit_generate(model, prompt_ids, gen_len, mask_id,
                     steps=32, temperature=0.7, top_k=50, guidance=0.0):
    device = prompt_ids.device
    B, P = prompt_ids.shape
    ids = torch.cat([prompt_ids,
                     torch.full((B, gen_len), mask_id, dtype=torch.long,
                                device=device)], dim=1)
    still_masked = torch.zeros(B, P + gen_len, dtype=torch.bool, device=device)
    still_masked[:, P:] = True
    for s in range(steps):
        frac = still_masked.float().mean().item()
        logits = model.predict_token_logits(ids, max(frac, T_MIN)).float()
        if guidance:
            null_ids = ids.clone()
            null_ids[:, :P] = mask_id
            lu = model.predict_token_logits(null_ids, max(frac, T_MIN)).float()
            logits = lu + guidance * (logits - lu)
        logits = logits / temperature
        if top_k:
            kth = logits.topk(top_k, dim=-1).values[..., -1:]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        probs = F.softmax(logits, dim=-1)
        sampled = torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(B, -1)
        conf = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
        conf = conf.masked_fill(~still_masked, float("inf"))
        n_keep = int(gen_len * math.cos(math.pi / 2 * (s + 1) / steps))
        ids = torch.where(still_masked, sampled, ids)
        if n_keep > 0:
            remask = torch.zeros_like(still_masked)
            remask.scatter_(1, conf.argsort(dim=1)[:, :n_keep], True)
            remask &= still_masked
            ids = torch.where(remask, mask_id, ids)
            still_masked = remask
        else:
            break
    return ids


@torch.no_grad()
def eval_step(model, tokenizer, step: int, mask_id: int):
    model.eval()
    eos = tokenizer.eos_token_id
    for p in EVAL_PROMPTS:
        ids = torch.tensor([tokenizer.encode(p, add_special_tokens=False)],
                           dtype=torch.long, device=DEVICE)
        for gtag, gs in (("g0", 0.0), (f"g{GUIDANCE}", GUIDANCE)):
            g = maskgit_generate(model, ids, gen_len=48, mask_id=mask_id,
                                 guidance=gs)
            toks = [x for x in g[0, ids.shape[1]:].tolist() if x < mask_id]
            if eos in toks:
                toks = toks[:toks.index(eos)]
            print(f"[step {step}][{gtag}] {p!r} -> {tokenizer.decode(toks)!r}",
                  flush=True)
    model.train()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.steps, args.batch = 20, 8

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")

    ck = torch.load(CKPT, map_location="cpu")
    cfg, sd, mask_id = dict(ck["config"]), ck["model_state_dict"], ck["mask_id"]
    sig = set(inspect.signature(DIMBA.__init__).parameters) - {"self"}
    model = DIMBA(**{k: v for k, v in cfg.items() if k in sig})
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f"loaded {CKPT}: vocab={cfg['vocab_size']} mask_id={mask_id} "
          f"missing={len(miss)} unexpected={len(unexp)}", flush=True)
    assert not miss, f"missing keys: {miss[:5]}"
    model = model.to(DEVICE).to(torch.bfloat16)
    del ck, sd

    data = build_sft_rows(tokenizer)
    loader = DataLoader(data, batch_size=args.batch, shuffle=True,
                        num_workers=2, pin_memory=True, drop_last=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / WARMUP) *
        (0.5 * (1 + math.cos(min(s, args.steps) / args.steps * math.pi)) * 0.9 + 0.1))

    if not args.smoke:
        import os
        os.makedirs(OUT_DIR, exist_ok=True)
        print("baseline (mdm pretrained, before SFT):", flush=True)
        eval_step(model, tokenizer, 0, mask_id)

    model.train()
    step, t0, it = 0, time.time(), iter(loader)
    while step < args.steps:
        try:
            rows, plens, rlens = next(it)
        except StopIteration:
            it = iter(loader)
            rows, plens, rlens = next(it)
        rows = rows.to(DEVICE, non_blocking=True)
        plens = plens.to(DEVICE, non_blocking=True)
        rlens = rlens.to(DEVICE, non_blocking=True)
        loss, parts = sft_loss(model, rows, plens, rlens, mask_id)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1
        if step % 100 == 0 or step == 1:
            ps = {k: f"{v.item():.4f}" for k, v in parts.items()}
            print(f"step {step}/{args.steps} loss={loss.item():.4f} {ps} "
                  f"lr={sched.get_last_lr()[0]:.2e} ({time.time() - t0:.0f}s)",
                  flush=True)
        if not args.smoke and step % EVAL_EVERY == 0:
            torch.save({"model_state_dict": model.state_dict(), "config": cfg,
                        "mask_id": mask_id},
                       f"{OUT_DIR}/mdm_sft_latest.pt")
            eval_step(model, tokenizer, step, mask_id)

    if args.smoke:
        eval_step(model, tokenizer, step, mask_id)
    else:
        torch.save({"model_state_dict": model.state_dict(), "config": cfg,
                    "mask_id": mask_id},
                   f"{OUT_DIR}/mdm_sft_final.pt")
    print("MDM_SFT_DONE", flush=True)


if __name__ == "__main__":
    main()
