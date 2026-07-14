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
from dimba.training.optimizers import HybridMuon, build_optimizer
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
    optimizer = build_optimizer(model, name=arm, lr=LR, weight_decay=0.0)
    if isinstance(optimizer, HybridMuon):
        print(
            f"[{arm}] muon params: {len(optimizer.muon_parameter_names)}, "
            f"adamw params: {len(optimizer.adamw_parameter_names)}",
            flush=True,
        )
    opts = [optimizer]

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
