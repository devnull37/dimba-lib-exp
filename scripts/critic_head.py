"""Token-critic head: a small frozen-base discriminator that scores each visible
token as wrong/right. Sidesteps the echo-chamber failure of repair-v2 (the head
never generates, only discriminates) and the calibration wall (it is trained on
labeled planted errors, not on the model's self-confidence).

Base: checkpoints/mdm_sft_cfg2/mdm_sft_final.pt (frozen, no_grad).
Critic input: x_dec (decoded latent) from the base's masked-diffusion pipeline.
Corruptions: half random tokens, half the base model's own samples. Labels:
planted=1, clean visible response tokens=0. BCE on visible response positions.

Usage: critic_head.py [--steps N] [--batch B] [--smoke]
"""
import argparse
import inspect
import math
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dimba import DIMBA

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mdm_sft_cfg2 as m2

CKPT = "checkpoints/mdm_sft_cfg2/mdm_sft_final.pt"
OUT = "checkpoints/critic_head.pt"
SEQ_LEN = 256
T_MIN = 0.03
WRONG_FRAC = 0.2
LR = 1e-3
DEVICE = "cuda"


class Critic(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d, 256), torch.nn.GELU(), torch.nn.Linear(256, 1))

    def forward(self, h):
        return self.net(h).squeeze(-1)


@torch.no_grad()
def x_dec_features(model, ids, t):
    """Replicate predict_token_logits up to the decoded latent (critic input)."""
    B = ids.shape[0]
    z = model.encode_latent(model.token_embed(ids))
    cond = model._build_conditioning(None, B, ids.device)
    t_idx = model._to_timestep_index(t, B, ids.device)
    raw = model._denoiser_raw(z, t_idx, cond, None)
    z0 = model._to_x0_latent(z, raw, t_idx)
    return model.decode_latent(z0)


def corrupt(model, rows, plens, rlens, mask_id):
    B, L = rows.shape
    t = torch.rand(B, device=rows.device) * (1.0 - T_MIN) + T_MIN
    pos = torch.arange(L, device=rows.device)[None, :]
    resp = (pos >= plens[:, None]) & (pos < rlens[:, None])
    mask = (torch.rand(B, L, device=rows.device) < t[:, None]) & resp
    wrong = (torch.rand(B, L, device=rows.device) < WRONG_FRAC * t[:, None]) & resp & ~mask
    corrupted = torch.where(mask, mask_id, rows)
    with torch.no_grad():
        # half of planted errors: model's own samples (hard negatives)
        probe = torch.where(wrong | mask, mask_id, rows)
        pl = model.predict_token_logits(probe, t).float()
        smp = torch.multinomial(F.softmax(pl.view(-1, pl.shape[-1]), -1), 1).view(B, L)
    rand_toks = torch.randint(0, mask_id, (B, L), device=rows.device)
    use_rand = torch.rand(B, L, device=rows.device) < 0.5
    planted = torch.where(use_rand, rand_toks, smp)
    # a plant that equals the gold token is not an error — drop it from labels
    wrong &= planted != rows
    corrupted = torch.where(wrong, planted, corrupted)
    visible = resp & ~mask
    return corrupted, t, wrong.float(), visible


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.steps, args.batch = 20, 8

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
    ck = torch.load(CKPT, map_location="cpu")
    cfg, mask_id = dict(ck["config"]), ck["mask_id"]
    sig = set(inspect.signature(DIMBA.__init__).parameters) - {"self"}
    model = DIMBA(**{k: v for k, v in cfg.items() if k in sig})
    model.load_state_dict(ck["model_state_dict"], strict=False)
    model = model.to(DEVICE).to(torch.bfloat16).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"base loaded (frozen): {CKPT}", flush=True)

    with torch.no_grad():
        d = x_dec_features(model, torch.zeros(1, 8, dtype=torch.long, device=DEVICE), 0.5).shape[-1]
    critic = Critic(d).to(DEVICE).to(torch.bfloat16)
    print(f"critic input dim: {d}", flush=True)

    data = m2.build_sft_rows(tokenizer)
    n_hold = 2000
    hold = torch.utils.data.Subset(data, range(len(data) - n_hold, len(data)))
    train = torch.utils.data.Subset(data, range(len(data) - n_hold))
    loader = DataLoader(train, batch_size=args.batch, shuffle=True,
                        num_workers=2, pin_memory=True, drop_last=True)

    opt = torch.optim.AdamW(critic.parameters(), lr=LR, weight_decay=0.01)
    step, t0, it = 0, time.time(), iter(loader)
    while step < args.steps:
        try:
            rows, plens, rlens = next(it)
        except StopIteration:
            it = iter(loader)
            rows, plens, rlens = next(it)
        rows = rows.to(DEVICE); plens = plens.to(DEVICE); rlens = rlens.to(DEVICE)
        corrupted, t, y, visible = corrupt(model, rows, plens, rlens, mask_id)
        with torch.no_grad():
            h = x_dec_features(model, corrupted, t)
        logit = critic(h).float()
        m = visible.float()
        bce = F.binary_cross_entropy_with_logits(logit, y, reduction="none")
        # planted errors are ~10% of visible tokens — upweight them
        w = m * (1.0 + 9.0 * y)
        loss = (bce * w).sum() / w.sum().clamp(min=1.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        if step % 100 == 0 or step == 1:
            print(f"step {step}/{args.steps} bce={loss.item():.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    # ---- eval on held-out rows: precision@k (k = #planted per row) ----
    critic.eval()
    hl = DataLoader(hold, batch_size=64)
    hits, tot, auc_n, auc_ok = 0, 0, 0, 0
    torch.manual_seed(123)
    with torch.no_grad():
        for rows, plens, rlens in hl:
            rows = rows.to(DEVICE); plens = plens.to(DEVICE); rlens = rlens.to(DEVICE)
            corrupted, t, y, visible = corrupt(model, rows, plens, rlens, mask_id)
            s = critic(x_dec_features(model, corrupted, t)).float()
            s = s.masked_fill(~visible, float("-inf"))
            for b in range(rows.shape[0]):
                k = int(y[b].sum())
                if k == 0:
                    continue
                top = s[b].topk(k).indices
                hits += y[b, top].sum().item(); tot += k
                # pairwise AUC sample: planted vs random clean visible
                pl_idx = y[b].nonzero().squeeze(-1)
                cl = (visible[b] & (y[b] == 0)).nonzero().squeeze(-1)
                if cl.numel():
                    n = min(k, cl.numel())
                    auc_ok += (s[b, pl_idx[:n]] > s[b, cl[torch.randperm(cl.numel())[:n]]]).sum().item()
                    auc_n += n
    print(f"CRITIC EVAL: precision@k = {hits/max(tot,1)*100:.1f}%  "
          f"pairwise-AUC ~ {auc_ok/max(auc_n,1)*100:.1f}%  (chance: ~10% / 50%)", flush=True)
    torch.save({"critic_state_dict": critic.state_dict(), "dim": d}, OUT)
    print("CRITIC_DONE", flush=True)


if __name__ == "__main__":
    main()
