"""Best-of-N v2: gap scored at MASKED positions (leave-k-out), cond vs uncond."""
import torch, torch.nn.functional as F
src = open("/tmp/claude-0/-workspace/e7af4c75-0211-4993-889c-4db42f486fc9/scratchpad/selfcorrect_test.py").read()
src = src.split("PROMPTS = [")[0]
exec(src)

@torch.no_grad()
def gap_score(model, mid, ids, P, K=4):
    B, L = ids.shape
    gen = torch.zeros_like(ids, dtype=torch.bool)
    gen[:, P:] = (ids[:, P:] != eos) & (ids[:, P:] != mid)
    pos = torch.arange(L, device=ids.device)[None, :].expand(B, L)
    tot = torch.zeros(B, device=ids.device); n = torch.zeros(B, device=ids.device)
    for r in range(K):
        m = gen & (pos % K == r)
        if not m.any(): continue
        x = ids.masked_fill(m, mid)
        lc = F.log_softmax(model.predict_token_logits(x, 0.25).float(), -1)
        u = x.clone(); u[:, :P] = mid
        lu = F.log_softmax(model.predict_token_logits(u, 0.25).float(), -1)
        tokp = ids.unsqueeze(-1)
        gap = (lc.gather(-1, tokp) - lu.gather(-1, tokp)).squeeze(-1)
        tot += (gap * m).sum(1); n += m.sum(1)
    return tot / n.clamp(min=1)

model, mid = load(CKPTS["cfg2  (pre-repair)"])
N = 8
PROMPTS = ["What is the capital of France?", "What color is the sky?",
           "Write one sentence about the ocean.", "What is 2 + 3?"]
torch.manual_seed(11)
for q in PROMPTS:
    p = f"Question: {q}\nAnswer:"
    pi = torch.tensor([tok.encode(p, add_special_tokens=False)] * N, device=DEV)
    out = generate(model, mid, pi)
    sc = gap_score(model, mid, out, pi.shape[1]).tolist()
    order = sorted(range(N), key=lambda i: -sc[i])
    print(f"== {q}", flush=True)
    for tag, i in [("BEST ", order[0]), ("2nd  ", order[1]), ("WORST", order[-1])]:
        print(f"  {tag} (gap {sc[i]:.3f}):{show(out[i:i+1], pi.shape[1])}", flush=True)
print("BESTOFN2_DONE", flush=True)
