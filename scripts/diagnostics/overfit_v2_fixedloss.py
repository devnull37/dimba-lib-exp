"""Overfit test v2 — FIXED objective. Hypothesis from partial_noise_test:
model is a perfect denoiser for t<=0.9 but produces a constant unigram blend at
t=1.0 because (a) min-SNR weight ~0 kills the diffusion gradient at high t and
(b) the unweighted CE anchor actively teaches unigram output at high t.

Fix under test (training-side, two lines):
  1. diffusion weight = clamp(snr, min=0.5, max=5)   (was min=1e-3 -> dead at high t)
  2. CE weight scaled per-sample by (1 - t_cont)     (anchor fades at high noise)

Success = free generation from PURE noise (t=1.0) recovers memorized sentences.
"""
import inspect, time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from dimba import DIMBA
from dimba.diffusion.sampling import sample_from_model_flow

SCRATCH = "/tmp/claude-0/-workspace/e7af4c75-0211-4993-889c-4db42f486fc9/scratchpad"
STEPS, SEQ_LEN, LR, DEVICE = 4000, 128, 3e-4, "cuda"
tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")

TEXTS = [
    "The sun rises in the east and sets in the west. Every morning the sky turns orange and pink before the day begins.",
    "Water is made of two hydrogen atoms and one oxygen atom. When water freezes it becomes ice, and when it boils it becomes steam.",
    "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the English alphabet at least once.",
    "Paris is the capital of France. The city is famous for the Eiffel Tower, its museums, and its long history of art and culture.",
    "To bake bread you need flour, water, yeast, and salt. Mix the ingredients, let the dough rise, and bake it in a hot oven.",
    "The ocean covers more than seventy percent of the surface of the Earth. It is home to millions of species of plants and animals.",
    "A computer program is a list of instructions that tells the machine exactly what to do, one small step at a time.",
    "In winter the days grow short and cold. Snow falls quietly on the fields and the rivers freeze under a grey sky.",
]

def make_batch():
    rows = []
    for t_ in TEXTS:
        ids = tok.encode(t_, add_special_tokens=False)
        while len(ids) < SEQ_LEN:
            ids = ids + tok.encode(" " + t_, add_special_tokens=False)
        rows.append(ids[:SEQ_LEN])
    return torch.tensor(rows, dtype=torch.long, device=DEVICE)

def fixed_losses(model, input_ids, t):
    """compute_dimba_losses with the two-line objective fix."""
    x_pred, _noise, info = model(input_ids, t)
    target, pred = info["z_0"], info["z0_hat"]
    per_pos = ((pred - target) ** 2).mean(dim=-1)
    per_sample = per_pos.mean(dim=1)
    t_cont = (t.float() / (model.num_diffusion_steps - 1)).clamp(1e-5, 1 - 1e-5)
    snr_flow = ((1.0 - t_cont) / t_cont) ** 2
    weight = torch.clamp(snr_flow, min=0.5, max=5.0)            # FIX 1: floor 0.5, was 1e-3
    diff_loss = (per_sample * weight).mean()
    logits = model.output_head(x_pred)
    B, L, V = logits.shape
    ce_per = F.cross_entropy(logits.reshape(-1, V), input_ids.reshape(-1),
                             reduction="none").view(B, L).mean(dim=1)
    ce_loss = (ce_per * (1.0 - t_cont)).mean()                  # FIX 2: CE fades at high t
    return model.recon_loss_weight * diff_loss + ce_loss, \
           {"diff_loss": diff_loss.detach(), "ce_loss": ce_loss.detach()}

ck = torch.load("checkpoints/distill_budget28_collapsed/final.pt", map_location="cpu")
cfg = dict(ck["config"]); del ck
cfg["vocab_size"] = 49152
sig = set(inspect.signature(DIMBA.__init__).parameters) - {"self"}
model = DIMBA(**{k: v for k, v in cfg.items() if k in sig}).to(DEVICE).to(torch.bfloat16)
model.train()

batch = make_batch()
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
_fm_ln = bool(getattr(getattr(model, "flow_schedule", None), "logit_normal_sampling", False))
t0 = time.time()
for step in range(1, STEPS + 1):
    # 50/50 logit-normal (production) and uniform (covers high t the former misses)
    if _fm_ln and step % 2 == 0:
        t = model.noise_schedule.sample_timesteps(batch.shape[0], DEVICE, mode="logit_normal")
    else:
        t = torch.randint(0, model.num_diffusion_steps, (batch.shape[0],), device=DEVICE)
    loss, parts = fixed_losses(model, batch, t)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if step % 400 == 0 or step == 1:
        ps = {k: f"{v.item():.4f}" for k, v in parts.items()}
        print(f"step {step}/{STEPS} loss={loss.item():.4f} {ps} ({time.time()-t0:.0f}s)", flush=True)

torch.save({"model_state_dict": model.state_dict(), "config": cfg}, f"{SCRATCH}/overfit_v2_model.pt")
print("saved overfit_v2_model.pt", flush=True)

model.eval()
with torch.no_grad():
    tz = torch.zeros(batch.shape[0], dtype=torch.long, device=DEVICE)
    acc = (model.predict_token_logits(batch, tz).argmax(-1) == batch).float().mean().item()
print(f"\nA) t=0 reconstruction acc: {acc:.3f}", flush=True)

print("\nB) FREE GENERATION from pure noise (the real test):", flush=True)
for i in (0, 3, 6):
    prompt = batch[i:i+1, :8]
    for samp, n in (("euler", 20), ("heun", 50)):
        with torch.no_grad():
            g = sample_from_model_flow(model, prompt, seq_len=120, num_steps=n,
                                       sampler=samp, temperature=0.3, top_k=50, device=DEVICE)
        out = g[0].tolist()
        target = batch[i, 8:].tolist()
        match = sum(a == b for a, b in zip(out, target)) / len(target)
        print(f"  row{i} [{samp} n={n}] match-vs-own-row={match:.2f}", flush=True)
        print(f"    {tok.decode([x for x in out if x < 49152])!r}", flush=True)

print("\nOVERFIT_V2_DONE", flush=True)
