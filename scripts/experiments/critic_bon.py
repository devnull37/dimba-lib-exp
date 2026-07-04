"""Best-of-N ranked by the trained critic head vs guidance-gap ranker."""
import torch, torch.nn.functional as F
src = open("/tmp/claude-0/-workspace/e7af4c75-0211-4993-889c-4db42f486fc9/scratchpad/selfcorrect_test.py").read()
exec(src.split("PROMPTS = [")[0])

model, mid = load(CKPTS["cfg2  (pre-repair)"])
ck = torch.load("/workspace/dimba-lib-exp/checkpoints/critic_head.pt", map_location="cpu")
critic = torch.nn.Sequential(torch.nn.Linear(ck["dim"], 256), torch.nn.GELU(),
                             torch.nn.Linear(256, 1)).to(DEV).to(torch.bfloat16)
critic.load_state_dict({k.replace("net.", ""): v for k, v in ck["critic_state_dict"].items()})
critic = critic.to(DEV).to(torch.bfloat16)
critic.eval()

@torch.no_grad()
def xdec(ids, t):
    B = ids.shape[0]
    z = model.encode_latent(model.token_embed(ids))
    cond = model._build_conditioning(None, B, ids.device)
    ti = model._to_timestep_index(t, B, ids.device)
    raw = model._denoiser_raw(z, ti, cond, None)
    return model.decode_latent(model._to_x0_latent(z, raw, ti))

@torch.no_grad()
def critic_score(ids, P):
    """Mean predicted wrongness over generated tokens (lower = better)."""
    gen = torch.zeros_like(ids, dtype=torch.bool); gen[:, P:] = (ids[:, P:] != mid) & (ids[:, P:] != eos)
    s = torch.sigmoid(critic(xdec(ids, 0.15)).squeeze(-1).float())
    return (s * gen).sum(1) / gen.sum(1).clamp(min=1)

@torch.no_grad()
def gap_score(ids, P, K=4):
    gen = torch.zeros_like(ids, dtype=torch.bool); gen[:, P:] = (ids[:, P:] != mid) & (ids[:, P:] != eos)
    pos = torch.arange(ids.shape[1], device=DEV)[None, :]
    tokp = ids.unsqueeze(-1); tot = torch.zeros(ids.shape[0], device=DEV); n = torch.zeros_like(tot)
    for r in range(K):
        m = gen & (pos % K == r)
        x = ids.masked_fill(m, mid)
        lc = F.log_softmax(model.predict_token_logits(x, 0.25).float(), -1)
        u = x.clone(); u[:, :P] = mid
        lu = F.log_softmax(model.predict_token_logits(u, 0.25).float(), -1)
        gap = (lc.gather(-1, tokp) - lu.gather(-1, tokp)).squeeze(-1)
        tot += (gap * m).sum(1); n += m.sum(1)
    return tot / n.clamp(min=1)

N = 8
for q in ["What is the capital of France?", "What color is the sky?",
          "Write one sentence about the ocean.", "Tell me about dogs."]:
    p = f"Question: {q}\nAnswer:"
    pi = torch.tensor([tok.encode(p, add_special_tokens=False)], device=DEV)
    P = pi.shape[1]
    cands = []
    for i in range(N):
        torch.manual_seed(100 + i)
        cands.append(generate(model, mid, pi))
    ids = torch.cat(cands, 0)
    cs = critic_score(ids, P); gs = gap_score(ids, P)
    print(f"== {q}", flush=True)
    print(f"  critic pick (wrongness {cs.min():.3f}):{show(ids[cs.argmin()][None], P)}", flush=True)
    print(f"  gap    pick (gap {gs.max():.3f})      :{show(ids[gs.argmax()][None], P)}", flush=True)
    print(f"  critic worst(wrongness {cs.max():.3f}):{show(ids[cs.argmax()][None], P)}", flush=True)
print("CRITIC_BON_DONE", flush=True)
