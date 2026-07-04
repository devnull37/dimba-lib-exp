"""Repair-training A/B: (A) planted-error detection, (B) self-correcting generation."""
import inspect, math, sys, torch
import torch.nn.functional as F
sys.path.insert(0, "/workspace/dimba-lib-exp/src")
from transformers import AutoTokenizer
from dimba import DIMBA

T_MIN = 0.03
DEV = "cuda"
tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
eos = tok.eos_token_id

CKPTS = {
    "cfg2  (pre-repair)": "/workspace/dimba-lib-exp/checkpoints/mdm_sft_cfg2/mdm_sft_final.pt",
    "repair(post-repair)": "/workspace/dimba-lib-exp/checkpoints/mdm_repair/mdm_sft_final.pt",
}

def load(path):
    ck = torch.load(path, map_location="cpu")
    cfg, mid = dict(ck["config"]), ck["mask_id"]
    sig = set(inspect.signature(DIMBA.__init__).parameters) - {"self"}
    m = DIMBA(**{k: v for k, v in cfg.items() if k in sig})
    m.load_state_dict(ck["model_state_dict"], strict=False)
    return m.to(DEV).to(torch.bfloat16).eval(), mid

SENTS = [
    "The capital of France is Paris, a city famous for its museums and food.",
    "Dogs are loyal animals that love to play and run in the park.",
    "The ocean covers most of the surface of the Earth.",
    "The sky is blue during the day and dark at night.",
    "Water is made of hydrogen and oxygen.",
    "A healthy diet includes fruits, vegetables and whole grains.",
    "The sun rises in the east and sets in the west.",
    "Books are a great way to learn about the world.",
    "Exercise is important for both the body and the mind.",
    "The moon orbits the Earth once every month.",
    "Computers can store and process large amounts of information.",
    "Rain falls from clouds and helps plants to grow.",
]

@torch.no_grad()
def detect_test(model, mid, frac=0.15, seed=0):
    g = torch.Generator().manual_seed(seed)
    fix, tot, keep, ktot = 0, 0, 0, 0
    for s in SENTS:
        ids = torch.tensor(tok.encode(" " + s, add_special_tokens=False))
        L = len(ids)
        k = max(1, int(L * frac))
        pos = torch.randperm(L, generator=g)[:k]
        cor = ids.clone()
        cor[pos] = torch.randint(0, mid, (k,), generator=g)
        same = cor[pos] == ids[pos]
        if same.any():
            cor[pos[same]] = (ids[pos[same]] + 7) % mid
        logits = model.predict_token_logits(cor[None].to(DEV), 0.15)[0].float()
        pred = logits.argmax(-1).cpu()
        planted = torch.zeros(L, dtype=torch.bool)
        planted[pos] = True
        fix += (pred[planted] == ids[planted]).sum().item(); tot += k
        keep += (pred[~planted] == ids[~planted]).sum().item(); ktot += L - k
    return fix / tot, keep / ktot

@torch.no_grad()
def guided_logits(model, mid, ids, P, t):
    lc = model.predict_token_logits(ids, t).float()
    u = ids.clone(); u[:, :P] = mid
    lu = model.predict_token_logits(u, t).float()
    return lu + 2.0 * (lc - lu)

@torch.no_grad()
def generate(model, mid, prompt_ids, gen_len=40, steps=128, temperature=0.7,
             top_k=20, freq_pen=0.7):
    B, P = prompt_ids.shape
    ids = torch.cat([prompt_ids, torch.full((B, gen_len), mid, dtype=torch.long,
                                            device=DEV)], dim=1)
    still = torch.zeros(B, P + gen_len, dtype=torch.bool, device=DEV)
    still[:, P:] = True
    for s in range(steps):
        frac = still.float().mean().item()
        logits = guided_logits(model, mid, ids, P, max(frac, T_MIN))
        for b in range(B):
            comm = ids[b, P:][~still[b, P:]]
            if comm.numel():
                uniq, cnt = comm.unique(return_counts=True)
                logits[b, :, uniq] -= freq_pen * (cnt - 1).clamp(min=0).float()
        logits = logits / temperature
        kth = logits.topk(top_k, dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
        probs = F.softmax(logits, dim=-1)
        sampled = torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(B, -1)
        conf = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
        conf = conf.masked_fill(~still, float("inf"))
        n_keep = int(gen_len * math.cos(math.pi / 2 * (s + 1) / steps))
        ids = torch.where(still, sampled, ids)
        if n_keep > 0:
            remask = torch.zeros_like(still)
            remask.scatter_(1, conf.argsort(dim=1)[:, :n_keep], True)
            remask &= still
            ids = torch.where(remask, mid, ids)
            still = remask
        else:
            break
    return ids

@torch.no_grad()
def refine(model, mid, ids, P, rounds=2, frac=0.25):
    """Self-correction: re-mask the least-confident committed tokens, refill."""
    B, L = ids.shape
    for _ in range(rounds):
        lg = guided_logits(model, mid, ids, P, 0.2)
        conf = F.softmax(lg, -1).gather(-1, ids.unsqueeze(-1)).squeeze(-1)
        gen = torch.zeros_like(ids, dtype=torch.bool)
        gen[:, P:] = (ids[:, P:] != eos) & (ids[:, P:] != mid)
        k = max(1, int((L - P) * frac))
        worst = conf.masked_fill(~gen, float("inf")).argsort(1)[:, :k]
        m = torch.zeros_like(gen)
        m.scatter_(1, worst, True)
        m &= gen
        ids = ids.masked_fill(m, mid)
        lg2 = guided_logits(model, mid, ids, P, 0.2)
        ids = torch.where(m, lg2.argmax(-1), ids)
    return ids

def show(ids, P):
    t = [x for x in ids[0, P:].tolist() if x < 49152]
    if eos in t:
        t = t[:t.index(eos)]
    return tok.decode(t)

PROMPTS = ["What is the capital of France?", "What color is the sky?",
           "Write one sentence about the ocean.", "Tell me about dogs."]

for name, path in CKPTS.items():
    model, mid = load(path)
    fr, kr = detect_test(model, mid)
    print(f"[A] {name}: fixes planted errors {fr*100:.1f}% | keeps clean tokens {kr*100:.1f}%", flush=True)
    torch.manual_seed(11)
    for q in PROMPTS:
        p = f"Question: {q}\nAnswer:"
        pi = torch.tensor([tok.encode(p, add_special_tokens=False)], device=DEV)
        out = generate(model, mid, pi)
        ref = refine(model, mid, out.clone(), pi.shape[1])
        print(f"[B] {name} | {q}", flush=True)
        print(f"    draft  :{show(out, pi.shape[1])}", flush=True)
        print(f"    refined:{show(ref, pi.shape[1])}", flush=True)
    del model
    torch.cuda.empty_cache()
print("SELFCORRECT_TEST_DONE", flush=True)
