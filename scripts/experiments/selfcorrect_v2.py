import torch, torch.nn.functional as F
src = open("/tmp/claude-0/-workspace/e7af4c75-0211-4993-889c-4db42f486fc9/scratchpad/selfcorrect_test.py").read()
src = src.split("PROMPTS = [")[0]
exec(src)

@torch.no_grad()
def refine2(model, mid, ids, P, rounds=2, frac=0.25, temp=0.7, top_k=20, freq_pen=0.9):
    B, L = ids.shape
    for _ in range(rounds):
        lg = guided_logits(model, mid, ids, P, 0.2)
        conf = F.softmax(lg, -1).gather(-1, ids.unsqueeze(-1)).squeeze(-1)
        gen = torch.zeros_like(ids, dtype=torch.bool)
        gen[:, P:] = (ids[:, P:] != eos) & (ids[:, P:] != mid)
        k = max(1, int((L - P) * frac))
        worst = conf.masked_fill(~gen, float("inf")).argsort(1)[:, :k]
        m = torch.zeros_like(gen); m.scatter_(1, worst, True); m &= gen
        ids = ids.masked_fill(m, mid)
        lg2 = guided_logits(model, mid, ids, P, 0.2)
        for b in range(B):  # anti-repeat penalty on the still-committed tokens
            comm = ids[b, P:][ids[b, P:] != mid]
            if comm.numel():
                uniq, cnt = comm.unique(return_counts=True)
                lg2[b, :, uniq] -= freq_pen * (cnt - 1).clamp(min=0).float()
        lg2 = lg2 / temp
        kth = lg2.topk(top_k, dim=-1).values[..., -1:]
        lg2 = lg2.masked_fill(lg2 < kth, float("-inf"))
        smp = torch.multinomial(F.softmax(lg2, -1).view(-1, lg2.shape[-1]), 1).view(B, -1)
        ids = torch.where(m, smp, ids)
    return ids

PROMPTS = ["What is the capital of France?", "What color is the sky?",
           "Write one sentence about the ocean.", "Tell me about dogs."]
model, mid = load(CKPTS["repair(post-repair)"])
torch.manual_seed(11)
for q in PROMPTS:
    p = f"Question: {q}\nAnswer:"
    pi = torch.tensor([tok.encode(p, add_special_tokens=False)], device=DEV)
    out = generate(model, mid, pi)
    ref = refine2(model, mid, out.clone(), pi.shape[1])
    print(f"{q}", flush=True)
    print(f"  draft  :{show(out, pi.shape[1])}", flush=True)
    print(f"  refined:{show(ref, pi.shape[1])}", flush=True)
print("V2_DONE", flush=True)
