"""Sampler-side: does concentrating ODE steps near t=1 (the mode-decision zone)
improve global mode commitment on the v2 fixed-loss model?"""
import inspect, sys
import torch
from transformers import AutoTokenizer
from dimba import DIMBA

S = "/tmp/claude-0/-workspace/e7af4c75-0211-4993-889c-4db42f486fc9/scratchpad"
DEVICE = "cuda"
tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
TEXTS = [
    "The sun rises in the east and sets in the west. Every morning the sky turns orange and pink before the day begins.",
    "Paris is the capital of France. The city is famous for the Eiffel Tower, its museums, and its long history of art and culture.",
    "A computer program is a list of instructions that tells the machine exactly what to do, one small step at a time.",
]
def row(text):
    ids = tok.encode(text, add_special_tokens=False)
    while len(ids) < 128: ids = ids + tok.encode(" " + text, add_special_tokens=False)
    return torch.tensor([ids[:128]], dtype=torch.long, device=DEVICE)

ck = torch.load(f"{S}/overfit_v2_model.pt", map_location="cpu")
cfg = dict(ck["config"]); sd = ck["model_state_dict"]
sig = set(inspect.signature(DIMBA.__init__).parameters) - {"self"}
model = DIMBA(**{k: v for k, v in cfg.items() if k in sig})
model.load_state_dict(sd, strict=False)
model = model.to(DEVICE).to(torch.bfloat16).eval()

@torch.no_grad()
def gen(prompt_ids, schedule):
    B = prompt_ids.shape[0]
    pl = model.encode_latent(model.token_embed(prompt_ids)); P = pl.shape[1]
    cond = model.conditioning_from_prompt(prompt_ids, B, DEVICE)
    x_t = torch.cat([pl, torch.randn(B, 120, model.d_latent, device=DEVICE, dtype=pl.dtype)], 1)
    x_sc = None
    for i in range(len(schedule) - 1):
        t_cur, t_nxt = float(schedule[i]), float(schedule[i + 1])
        tb = torch.full((B,), t_cur, device=DEVICE)
        x0_hat = model.denoise_flow(x_t, tb, cond, x_sc); x_sc = x0_hat
        x_t = x_t + (x_t - x0_hat) / max(t_cur, 1e-5) * (t_nxt - t_cur)
        x_t[:, :P] = pl
    return model.output_head(model.decode_latent(x_t[:, P:])).argmax(-1)

lin = torch.linspace(1.0, 0.0, 51)
dense_hi = 1.0 - torch.linspace(0.0, 1.0, 51) ** 2          # many steps near t=1
dense_hi2 = 1.0 - torch.linspace(0.0, 1.0, 101) ** 3        # extreme: 100 steps, cubic near 1
for name, sched in (("linear-50", lin), ("dense-hi-50", dense_hi), ("dense-hi3-100", dense_hi2)):
    print(f"\n--- schedule {name} ---")
    for text in TEXTS:
        ids = row(text)
        out = gen(ids[:, :8], sched)[0].tolist()
        target = ids[0, 8:].tolist()
        m = sum(a == b for a, b in zip(out, target)) / len(target)
        print(f"  match={m:.2f} :: {tok.decode([x for x in out if x < 49152])[:170]!r}", flush=True)
print("\nDENSE_TEST_DONE")
