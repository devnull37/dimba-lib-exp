"""Test the Diffusion-LM clamping trick on the flow sampler.

Hypothesis: sample_from_model_flow fails to commit to a mode because x0_hat is a
posterior MEAN (blend of texts). Snapping x0_hat to the nearest token embedding
each ODE step (clamp trick, Li et al. 2022) forces commitment. If clamped sampling
recovers memorized text from the overfit model, the train->sample contract is
FIXABLE at the sampler level (no retraining needed).

Usage: clamp_sample_test.py [ckpt] [--rows] ; default ckpt = overfit_model.pt
"""
import inspect, sys
import torch
from transformers import AutoTokenizer
from dimba import DIMBA

SCRATCH = "/tmp/claude-0/-workspace/e7af4c75-0211-4993-889c-4db42f486fc9/scratchpad"
CKPT = sys.argv[1] if len(sys.argv) > 1 else f"{SCRATCH}/overfit_model.pt"
SEQ_LEN = 120           # response length; rows are 128 with an 8-token prompt
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

def make_rows():
    rows = []
    for t_ in TEXTS:
        ids = tok.encode(t_, add_special_tokens=False)
        while len(ids) < 128:
            ids = ids + tok.encode(" " + t_, add_special_tokens=False)
        rows.append(ids[:128])
    return torch.tensor(rows, dtype=torch.long, device=DEVICE)

def load(path):
    ck = torch.load(path, map_location="cpu")
    cfg = dict(ck["config"]); sd = ck["model_state_dict"]
    sig = set(inspect.signature(DIMBA.__init__).parameters) - {"self"}
    emb = next((k for k in sd if k.endswith("token_embed.embedding.weight")), None)
    if emb is not None and sd[emb].shape[0] != cfg.get("vocab_size"):
        cfg["vocab_size"] = sd[emb].shape[0]
    m = DIMBA(**{k: v for k, v in cfg.items() if k in sig})
    miss, unexp = m.load_state_dict(sd, strict=False)
    print(f"[{path}] miss={len(miss)} unexp={len(unexp)}", flush=True)
    return m.to(DEVICE).to(torch.bfloat16).eval()

@torch.no_grad()
def flow_sample(model, prompt_ids, seq_len, num_steps=50, sampler="euler",
                clamp_mode="none", clamp_below_t=0.5, temperature=0.0, top_k=None):
    """Replicates sample_from_model_flow with an optional per-step clamp of x0_hat
    to the nearest token embedding (via head argmax -> re-embed)."""
    device = DEVICE
    d_latent = model.d_latent
    prompt_ids = prompt_ids.to(device)
    B = prompt_ids.shape[0]
    prompt_latent = model.encode_latent(model.token_embed(prompt_ids))
    P = prompt_latent.shape[1]
    cond = model.conditioning_from_prompt(prompt_ids, B, device)

    x_t = torch.cat([prompt_latent,
                     torch.randn(B, seq_len, d_latent, device=device, dtype=prompt_latent.dtype)], dim=1)
    ts = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

    def clamp_x0(x0_hat):
        dec = model.decode_latent(x0_hat)
        ids = model.output_head(dec).argmax(-1)
        return model.encode_latent(model.token_embed(ids))

    def velocity(xt, t_val, x_sc):
        t_batch = torch.full((B,), t_val, device=device)
        x0_hat = model.denoise_flow(xt, t_batch, cond, x_sc)
        raw = x0_hat
        if clamp_mode == "all" or (clamp_mode == "late" and t_val < clamp_below_t):
            x0_hat = clamp_x0(x0_hat)
        v = (xt - x0_hat) / max(t_val, 1e-5)
        return v, raw

    x_sc = None
    for i in range(num_steps):
        t_cur, t_nxt = float(ts[i]), float(ts[i + 1])
        dt = t_nxt - t_cur
        v_cur, x0_raw = velocity(x_t, t_cur, x_sc)
        x_sc = x0_raw
        if sampler == "heun" and i < num_steps - 1:
            x_mid = x_t + v_cur * dt
            x_mid[:, :P, :] = prompt_latent
            v_nxt, _ = velocity(x_mid, t_nxt, x_sc)
            x_t = x_t + 0.5 * (v_cur + v_nxt) * dt
        else:
            x_t = x_t + v_cur * dt
        x_t[:, :P, :] = prompt_latent

    logits = model.output_head(model.decode_latent(x_t[:, P:, :]))
    if temperature and temperature > 0:
        logits = logits / temperature
        probs = torch.softmax(logits.float(), dim=-1)
        return torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(B, seq_len)
    return logits.argmax(-1)

model = load(CKPT)
rows = make_rows()

CONFIGS = [
    ("none", "euler", 20), ("none", "heun", 50),
    ("all",  "euler", 20), ("all",  "heun", 50),
    ("late", "euler", 20), ("late", "heun", 50),
]
for i in (0, 3, 6):
    prompt = rows[i:i+1, :8]
    target = rows[i, 8:8+SEQ_LEN].tolist()
    print(f"\n=== row {i}: {tok.decode(rows[i,:20].tolist())!r} ... ===", flush=True)
    for cm, samp, n in CONFIGS:
        g = flow_sample(model, prompt, SEQ_LEN, num_steps=n, sampler=samp, clamp_mode=cm)
        out = g[0].tolist()
        match = sum(a == b for a, b in zip(out, target)) / len(target)
        txt = tok.decode([x for x in out if x < 49152])
        print(f"  [clamp={cm:5s} {samp} n={n}] match={match:.2f} :: {txt[:150]!r}", flush=True)

print("\nCLAMP_TEST_DONE", flush=True)
