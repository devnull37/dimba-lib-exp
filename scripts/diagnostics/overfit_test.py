"""Overfit contract test: can DIMBA's train->sample pipeline produce coherent text AT ALL?

Trains the EXACT production architecture (config read from a real checkpoint, weights
random) on 8 fixed sentences until memorized, using the IDENTICAL loss path as
DistillationTrainer._stage3_step (logit-normal t + compute_dimba_losses, bf16).
Then: (a) t=0 clean-pass reconstruction accuracy, (b) flow sampling with memorized
prompts. A model that memorized its data but still samples soup => the training->
sampling contract is broken (bug), and no token budget / KD / money fixes it.
"""
import inspect, sys, time
import torch
from transformers import AutoTokenizer

from dimba import DIMBA
from dimba.training import compute_dimba_losses
from dimba.diffusion.sampling import sample_from_model_flow

CKPT_FOR_CONFIG = "checkpoints/distill_budget28_collapsed/final.pt"
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
SEQ_LEN = 128
LR = 3e-4
LOG_EVERY = 200
DEVICE = "cuda"

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
        while len(ids) < SEQ_LEN:                      # tile to fill the row
            ids = ids + tok.encode(" " + t_, add_special_tokens=False)
        rows.append(ids[:SEQ_LEN])
    return torch.tensor(rows, dtype=torch.long, device=DEVICE)

# ---- model: exact production architecture, random weights ----
ck = torch.load(CKPT_FOR_CONFIG, map_location="cpu")
cfg = dict(ck["config"]); del ck
sig = set(inspect.signature(DIMBA.__init__).parameters) - {"self"}
cfg["vocab_size"] = 49152
model = DIMBA(**{k: v for k, v in cfg.items() if k in sig}).to(DEVICE).to(torch.bfloat16)
model.train()
print(f"model: flow={getattr(model,'use_flow_matching',None)} "
      f"params={sum(p.numel() for p in model.parameters())/1e6:.0f}M", flush=True)

batch = make_batch()
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)

_fm_ln = bool(getattr(getattr(model, "flow_schedule", None), "logit_normal_sampling", False))
t0 = time.time()
for step in range(1, STEPS + 1):
    if _fm_ln:
        t = model.noise_schedule.sample_timesteps(batch.shape[0], DEVICE, mode="logit_normal")
    else:
        t = torch.randint(0, model.num_diffusion_steps, (batch.shape[0],), device=DEVICE)
    loss, parts = compute_dimba_losses(model, batch, t, ce_loss_weight=1.0, min_snr_gamma=5.0)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if step % LOG_EVERY == 0 or step == 1:
        ps = {k: f"{v.item():.4f}" for k, v in parts.items() if hasattr(v, "item")}
        print(f"step {step}/{STEPS} loss={loss.item():.4f} {ps} ({time.time()-t0:.0f}s)", flush=True)

torch.save({"model_state_dict": model.state_dict(), "config": cfg},
           "/tmp/claude-0/-workspace/e7af4c75-0211-4993-889c-4db42f486fc9/scratchpad/overfit_model.pt")
print("saved overfit_model.pt", flush=True)

# ---- eval A: t=0 clean-pass reconstruction (tests decode head path) ----
model.eval()
with torch.no_grad():
    tz = torch.zeros(batch.shape[0], dtype=torch.long, device=DEVICE)
    logits = model.predict_token_logits(batch, tz)
    acc = (logits.argmax(-1) == batch).float().mean().item()
print(f"\nA) t=0 clean-pass reconstruction accuracy: {acc:.3f}  (memorized => ~1.0)", flush=True)

# ---- eval B: flow sampling from memorized prompts (THE contract test) ----
print("\nB) flow samples (prompt = first 8 tokens of memorized rows):", flush=True)
for i in (0, 3, 6):
    prompt = batch[i:i+1, :8]
    for steps_n, samp in ((50, "heun"), (20, "euler")):
        with torch.no_grad():
            g = sample_from_model_flow(model, prompt, seq_len=SEQ_LEN - 8, num_steps=steps_n,
                                       sampler=samp, temperature=0.3, top_k=50, device=DEVICE)
        out = g[0].tolist()
        target = batch[i, 8:].tolist()
        match = sum(a == b for a, b in zip(out, target)) / len(target)
        print(f"  row{i} [{samp} n={steps_n}] token-match={match:.2f}", flush=True)
        print(f"    {tok.decode([x for x in out if x < 49152])!r}", flush=True)

print("\nOVERFIT_TEST_DONE  (memorized+coherent => contract OK; memorized+soup => contract BROKEN)", flush=True)
