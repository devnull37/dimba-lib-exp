"""Muon vs AdamW A/B: 2k steps of continued MDM pretraining each, same data/seed.
Muon: Newton-Schulz orthogonalized nesterov momentum on hidden 2D matrices,
Moonlight-style 0.2*sqrt(max(m,n)) update scaling; embeddings/1D stay on AdamW."""
import inspect, json, math, os, sys, time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, "/workspace/dimba-lib-exp/src")
sys.path.insert(0, "/workspace/dimba-lib-exp/scripts")
from transformers import AutoTokenizer
from dimba import DIMBA
import masked_diffusion_finetune as mdf

SCRATCH = "/tmp/claude-0/-workspace/e7af4c75-0211-4993-889c-4db42f486fc9/scratchpad"
CKPT = "/workspace/dimba-lib-exp/checkpoints/masked_diffusion2/mdm_final.pt"
STEPS, BATCH, LR, WARMUP = 2000, 64, 1e-4, 100
DEV = "cuda"

tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")

cache = f"{SCRATCH}/muon_ab_stream.pt"
if os.path.exists(cache):
    stream = torch.load(cache)
else:
    stream = mdf.build_stream(tok, 60000)
    torch.save(stream, cache)
print(f"stream: {stream.numel()/1e6:.1f}M tokens", flush=True)


def ns5(G, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.to(torch.bfloat16)
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum=0.95):
        super().__init__(params, dict(lr=lr, momentum=momentum))

    @torch.no_grad()
    def step(self):
        for grp in self.param_groups:
            for p in grp["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "m" not in st:
                    st["m"] = torch.zeros_like(p.grad)
                buf = st["m"]
                buf.mul_(grp["momentum"]).add_(p.grad)
                g = p.grad.add(buf, alpha=grp["momentum"])  # nesterov
                o = ns5(g)
                scale = 0.2 * math.sqrt(max(p.shape[-2], p.shape[-1]))
                p.add_(o, alpha=-grp["lr"] * scale)


def load_model():
    ck = torch.load(CKPT, map_location="cpu")
    cfg, mid = dict(ck["config"]), ck["mask_id"]
    sig = set(inspect.signature(DIMBA.__init__).parameters) - {"self"}
    m = DIMBA(**{k: v for k, v in cfg.items() if k in sig})
    m.load_state_dict(ck["model_state_dict"], strict=False)
    return m.to(DEV).to(torch.bfloat16), mid


def lr_factor(s):
    w = min(1.0, (s + 1) / WARMUP)
    return w * (0.5 * (1 + math.cos(min(s, STEPS) / STEPS * math.pi)) * 0.9 + 0.1)


def run(arm):
    torch.manual_seed(0)
    model, mid = load_model()
    vocab = model.predict_token_logits(
        torch.zeros(1, 8, dtype=torch.long, device=DEV), 0.5).shape[-1]
    if arm == "adamw":
        opts = [torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)]
    else:
        hidden, rest = [], []
        for n, p in model.named_parameters():
            if p.ndim == 2 and "embed" not in n.lower() and vocab not in p.shape:
                hidden.append(p)
            else:
                rest.append(p)
        print(f"[{arm}] muon params: {len(hidden)}, adamw params: {len(rest)}", flush=True)
        opts = [Muon(hidden, lr=LR), torch.optim.AdamW(rest, lr=LR, weight_decay=0.0)]

    torch.manual_seed(0)
    loader = DataLoader(mdf.PackedChunks(stream, mdf.SEQ_LEN), batch_size=BATCH,
                        shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    curve, step, t0, it = [], 0, time.time(), iter(loader)
    model.train()
    while step < STEPS:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        batch = batch.to(DEV, non_blocking=True)
        loss, parts = mdf.masked_diffusion_loss(model, batch, mid)
        for o in opts:
            o.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        f = lr_factor(step)
        for o in opts:
            for grp in o.param_groups:
                grp["lr"] = LR * f
        for o in opts:
            o.step()
        step += 1
        if step % 50 == 0:
            ce = parts["ce_masked"].item()
            curve.append([step, ce])
            if step % 200 == 0:
                print(f"[{arm}] step {step}/{STEPS} ce_masked={ce:.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
    acc = mdf.eval_step(model, tok, STEPS, mid)
    tail = sum(c for _, c in curve[-8:]) / 8
    del model, opts
    torch.cuda.empty_cache()
    return {"curve": curve, "tail_ce": tail, "infill": acc}


res = {}
for arm in ("adamw", "muon"):
    print(f"===== ARM: {arm} =====", flush=True)
    res[arm] = run(arm)
    print(f"[{arm}] tail ce_masked (last 400 steps avg): {res[arm]['tail_ce']:.4f} | "
          f"infill@50%: {res[arm]['infill']:.3f}", flush=True)
json.dump(res, open(f"{SCRATCH}/muon_ab_results.json", "w"))
print("MUON_AB_DONE", flush=True)
