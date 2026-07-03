"""KD-on finetune: test whether teacher supervision fixes generation commitment.

Resumes checkpoints/repair/repaired_final.pt (the fixed-objective denoiser) and
trains with the fixed objective (snr_floor=0.5, ce_time_fade) PLUS soft-label KD
from the SmolLM-135M teacher applied at the *sampled noisy timestep* — unlike
the stage-3 KD in dimba.distillation.trainer, which runs the student at t=0 and
therefore never trains the denoiser at noise. Teacher (causal) logits at
position i predict token i+1; student x0-logits at position i reconstruct token
i, so KD aligns student[:, 1:] with teacher[:, :-1]. KD and CE both fade with
(1 - t): at pure noise a clean-text target is unmatchable and would re-teach
the posterior-mean blend (docs/ROOT_CAUSE_BUDGET28.md).

kd_weight anneals 1.0 -> 0.3 (cosine). Timesteps: 50% uniform + 50%
logit-normal so high t stays covered. Generation probes + checkpoint every
EVAL_EVERY steps.

Usage: kd_finetune.py [--steps N] [--batch B] [--smoke]
"""
import argparse
import inspect
import math
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from dimba import DIMBA
from dimba.diffusion.sampling import sample_from_model_flow

CKPT = "checkpoints/repair/repaired_final.pt"
OUT_DIR = "checkpoints/kd_finetune"
TEACHER = "HuggingFaceTB/SmolLM-135M"
SEQ_LEN = 512
LR = 1e-4
WARMUP = 100
EVAL_EVERY = 1000
KD_TEMP = 2.0
KD_CHUNK = 16  # KL in fp32 over the full vocab is ~6 GB at B=64; chunk it
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


def sample_timesteps(model, batch_size: int, device: str) -> torch.Tensor:
    """50% uniform (covers t~1) + 50% logit-normal (production distribution)."""
    T = model.num_diffusion_steps
    n_u = batch_size // 2
    u = torch.randint(0, T, (n_u,), device=device)
    ln = model.noise_schedule.sample_timesteps(batch_size - n_u, device,
                                               mode="logit_normal")
    t = torch.cat([u, ln.to(device)])
    return t[torch.randperm(batch_size, device=device)]


def kd_dimba_loss(model, teacher, input_ids, t, kd_weight: float):
    """Fixed diffusion loss + faded CE anchor + faded noisy-t KD."""
    x_pred, _noise, info = model(input_ids, t)
    target, pred = info["z_0"], info["z0_hat"]
    dm = info.get("diffuse_mask")

    per_pos = ((pred - target) ** 2).mean(dim=-1)
    if dm is not None:
        eff = dm.to(torch.float32)
        per_sample = (per_pos * eff).sum(dim=1) / eff.sum(dim=1).clamp(min=1.0)
    else:
        per_sample = per_pos.mean(dim=1)
    t_cont = (t.float() / (model.num_diffusion_steps - 1)).clamp(1e-5, 1.0 - 1e-5)
    snr_flow = ((1.0 - t_cont) / t_cont) ** 2
    weight = snr_flow.clamp(max=5.0).clamp(min=0.5)  # THE FIX
    diff_loss = (per_sample * weight).mean()

    logits = model.output_head(x_pred)
    B, L, V = logits.shape
    ce_per = F.cross_entropy(logits.reshape(-1, V), input_ids.reshape(-1),
                             reduction="none").view(B, L).mean(dim=1)
    fade = 1.0 - t_cont
    ce_loss = (ce_per * fade).mean()

    with torch.no_grad():
        t_logits = teacher(input_ids).logits  # [B, L, 49152]
    Vt = t_logits.shape[-1]
    s_kd = logits[:, 1:, :Vt]        # student reconstructs tokens 1..L-1
    t_kd = t_logits[:, :-1]          # teacher predicts tokens 1..L-1
    kl_samples = []
    for i in range(0, B, KD_CHUNK):
        s = (s_kd[i:i + KD_CHUNK].float() / KD_TEMP)
        tt = (t_kd[i:i + KD_CHUNK].float() / KD_TEMP)
        soft = F.softmax(tt, dim=-1)
        kl = (soft * (soft.clamp_min(1e-9).log() - F.log_softmax(s, dim=-1))
              ).sum(dim=-1).mean(dim=1)  # [chunk]
        kl_samples.append(kl)
    kd_loss = (torch.cat(kl_samples) * fade).mean() * (KD_TEMP ** 2)

    loss = model.recon_loss_weight * diff_loss + ce_loss + kd_weight * kd_loss
    parts = {"diff": diff_loss, "ce": ce_loss, "kd": kd_loss}
    if model.latent_diffusion:
        x_0 = model.token_embed(input_ids)
        ae_loss = F.mse_loss(model.decode_latent(info["z_0"]), x_0)
        loss = loss + model.latent_loss_weight * ae_loss
        parts["ae"] = ae_loss
    return loss, parts


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
    ap.add_argument("--steps", type=int, default=16000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--docs", type=int, default=200_000)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.steps, args.batch, args.docs = 20, 8, 300

    tokenizer = AutoTokenizer.from_pretrained(TEACHER)
    teacher = AutoModelForCausalLM.from_pretrained(
        TEACHER, torch_dtype=torch.bfloat16).to(DEVICE).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    ck = torch.load(CKPT, map_location="cpu")
    cfg = dict(ck["config"])
    sd = ck["model_state_dict"]
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
        (0.5 * (1 + math.cos(min(s, args.steps) / args.steps * math.pi)) * 0.9 + 0.1))

    if not args.smoke:
        import os
        os.makedirs(OUT_DIR, exist_ok=True)
        print("baseline generations (repaired_final, before KD):", flush=True)
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
        t = sample_timesteps(model, batch.shape[0], DEVICE)
        # anneal kd_weight 1.0 -> 0.3 over the run
        kd_w = 0.3 + 0.7 * 0.5 * (1 + math.cos(step / args.steps * math.pi))
        loss, parts = kd_dimba_loss(model, teacher, batch, t, kd_w)
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
                  f"kd_w={kd_w:.2f} lr={sched.get_last_lr()[0]:.2e} "
                  f"{tok_s / 1e3:.0f}k tok/s ({time.time() - t0:.0f}s)", flush=True)
        if not args.smoke and step % EVAL_EVERY == 0:
            torch.save({"model_state_dict": model.state_dict(), "config": cfg},
                       f"{OUT_DIR}/kd_latest.pt")
            eval_generations(model, tokenizer, step)

    if args.smoke:
        eval_generations(model, tokenizer, step)
    else:
        torch.save({"model_state_dict": model.state_dict(), "config": cfg},
                   f"{OUT_DIR}/kd_final.pt")
    print("KD_FINETUNE_DONE", flush=True)


if __name__ == "__main__":
    main()
