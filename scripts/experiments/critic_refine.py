"""Critic-guided refinement: critic picks remask positions (not self-confidence)."""
import torch, torch.nn.functional as F
src = open("critic_bon.py").read()
exec(src.split("N = 8")[0])  # loads model, mid, critic, xdec, critic_score, gap_score

@torch.no_grad()
def tok_wrongness(ids):
    return torch.sigmoid(critic(xdec(ids, 0.15)).squeeze(-1).float())

@torch.no_grad()
def critic_refine(ids, P, rounds=2, frac=0.25, temperature=0.7, top_k=20):
    B, L = ids.shape
    for _ in range(rounds):
        w = tok_wrongness(ids)
        gen = torch.zeros_like(ids, dtype=torch.bool)
        gen[:, P:] = (ids[:, P:] != mid) & (ids[:, P:] != eos)
        k = max(1, int((L - P) * frac))
        worst = w.masked_fill(~gen, float("-inf")).argsort(1, descending=True)[:, :k]
        m = torch.zeros_like(gen); m.scatter_(1, worst, True); m &= gen
        ids = ids.masked_fill(m, mid)
        lg = guided_logits(model, mid, ids, P, 0.25) / temperature
        kth = lg.topk(top_k, -1).values[..., -1:]
        lg = lg.masked_fill(lg < kth, float("-inf"))
        smp = torch.multinomial(F.softmax(lg, -1).view(-1, lg.shape[-1]), 1).view(B, L)
        ids = torch.where(m, smp, ids)
    return ids

KEY = {"What is the capital of France?": "paris", "What is the capital of Japan?": "tokyo",
       "What color is the sky?": "blue", "Write one sentence about the ocean.": "ocean",
       "Tell me about dogs.": "dog"}
hit_d, hit_r = 0, 0
for q, key in KEY.items():
    p = f"Question: {q}\nAnswer:"
    pi = torch.tensor([tok.encode(p, add_special_tokens=False)], device=DEV)
    P = pi.shape[1]
    torch.manual_seed(11)
    draft = generate(model, mid, pi)
    ref = critic_refine(draft.clone(), P)
    wd = tok_wrongness(draft); wr = tok_wrongness(ref)
    gen = torch.zeros_like(draft, dtype=torch.bool); gen[:, P:] = draft[:, P:] != eos
    sd, sr = show(draft, P), show(ref, P)
    hit_d += key in sd.lower(); hit_r += key in sr.lower()
    print(f"== {q}  critic-wrongness {(wd*gen).sum()/gen.sum():.3f} -> {(wr*gen).sum()/gen.sum():.3f}", flush=True)
    print(f"  draft  :{sd}", flush=True)
    print(f"  refined:{sr}", flush=True)
print(f"keyword hits: draft {hit_d}/5, critic-refined {hit_r}/5", flush=True)
print("CRITIC_REFINE_DONE", flush=True)
