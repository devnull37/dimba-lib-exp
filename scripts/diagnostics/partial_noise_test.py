"""Where does denoising die? Noise a MEMORIZED row to level t_start, then integrate
the flow ODE down to 0 and measure recovery. If recovery is good for small t_start
but collapses at high t_start, the high-noise regime is the dead zone (min-SNR
downweighting + unweighted CE anchor -> unigram behavior at high t)."""
import inspect, sys
import torch
from transformers import AutoTokenizer
from dimba import DIMBA

SCRATCH = "/tmp/claude-0/-workspace/e7af4c75-0211-4993-889c-4db42f486fc9/scratchpad"
CKPT = sys.argv[1] if len(sys.argv) > 1 else f"{SCRATCH}/overfit_model.pt"
DEVICE = "cuda"
tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")

TEXTS = [
    "The sun rises in the east and sets in the west. Every morning the sky turns orange and pink before the day begins.",
    "Paris is the capital of France. The city is famous for the Eiffel Tower, its museums, and its long history of art and culture.",
    "A computer program is a list of instructions that tells the machine exactly what to do, one small step at a time.",
]

def row(text):
    ids = tok.encode(text, add_special_tokens=False)
    while len(ids) < 128:
        ids = ids + tok.encode(" " + text, add_special_tokens=False)
    return torch.tensor([ids[:128]], dtype=torch.long, device=DEVICE)

def load(path):
    ck = torch.load(path, map_location="cpu")
    cfg = dict(ck["config"]); sd = ck["model_state_dict"]
    sig = set(inspect.signature(DIMBA.__init__).parameters) - {"self"}
    emb = next((k for k in sd if k.endswith("token_embed.embedding.weight")), None)
    if emb is not None and sd[emb].shape[0] != cfg.get("vocab_size"):
        cfg["vocab_size"] = sd[emb].shape[0]
    m = DIMBA(**{k: v for k, v in cfg.items() if k in sig})
    m.load_state_dict(sd, strict=False)
    return m.to(DEVICE).to(torch.bfloat16).eval()

@torch.no_grad()
def recover(model, ids, t_start, num_steps=25):
    """Noise the true row to t_start along the flow path, then Euler-integrate to 0."""
    z0 = model.encode_latent(model.token_embed(ids))
    eps = torch.randn_like(z0)
    x_t = (1.0 - t_start) * z0 + t_start * eps
    cond = model.conditioning_from_prompt(ids[:, :8], 1, DEVICE)
    ts = torch.linspace(t_start, 0.0, num_steps + 1, device=DEVICE)
    x_sc = None
    for i in range(num_steps):
        t_cur, t_nxt = float(ts[i]), float(ts[i + 1])
        tb = torch.full((1,), t_cur, device=DEVICE)
        x0_hat = model.denoise_flow(x_t, tb, cond, x_sc)
        x_sc = x0_hat
        v = (x_t - x0_hat) / max(t_cur, 1e-5)
        x_t = x_t + v * (t_nxt - t_cur)
    out = model.output_head(model.decode_latent(x_t)).argmax(-1)
    acc = (out == ids).float().mean().item()
    return acc, out[0].tolist()

model = load(CKPT)
print(f"ckpt={CKPT}")
for text in TEXTS:
    ids = row(text)
    print(f"\n=== {text[:60]!r} ===")
    for t_start in (0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0):
        acc, out = recover(model, ids, t_start)
        preview = tok.decode([x for x in out[:30] if x < 49152])
        print(f"  t_start={t_start:.1f} recovery={acc:.2f} :: {preview!r}", flush=True)
print("\nPARTIAL_NOISE_TEST_DONE")
