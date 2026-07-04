"""Free sampler upgrades A/B: RCR (falling-confidence remask) + lookahead verify.
Base sampler = CFG g2.0 + exempt-first freq penalty, 128 steps."""
import torch, torch.nn.functional as F
src = open("/tmp/claude-0/-workspace/e7af4c75-0211-4993-889c-4db42f486fc9/scratchpad/selfcorrect_test.py").read()
src = src.split("PROMPTS = [")[0]
exec(src)

@torch.no_grad()
def gen3(model, mid, prompt_ids, gen_len=40, steps=128, temperature=0.7,
         top_k=20, guidance=2.0, freq_pen=0.7, rcr=False, look=False,
         rcr_ratio=0.85, look_ratio=0.85, rcr_cap=4):
    import math
    B, P = prompt_ids.shape
    L = P + gen_len
    ids = torch.cat([prompt_ids, torch.full((B, gen_len), mid, dtype=torch.long,
                                            device=DEV)], dim=1)
    still = torch.zeros(B, L, dtype=torch.bool, device=DEV)
    still[:, P:] = True
    commit_p = torch.zeros(B, L, device=DEV)  # peak untempered prob while committed
    fires = {'rcr': 0, 'look': 0}

    def guided(x, t):
        lc = model.predict_token_logits(x, t).float()
        u = x.clone(); u[:, :P] = mid
        lu = model.predict_token_logits(u, t).float()
        lg = lu + guidance * (lc - lu)
        for b in range(B):
            comm = x[b, P:][x[b, P:] != mid]
            if comm.numel():
                uniq, cnt = comm.unique(return_counts=True)
                lg[b, :, uniq] -= freq_pen * (cnt - 1).clamp(min=0).float()
        return lg

    for s in range(steps):
        frac = still.float().mean().item()
        t = max(frac, T_MIN)
        lg = guided(ids, t)
        pu = F.softmax(lg, dim=-1)  # untempered, for confidence tracking

        # RCR: remask committed tokens whose confidence FELL since commit
        if rcr and 2 < s < int(steps * 0.85):
            comm = ~still & (ids != mid)
            comm[:, :P] = False
            p_now = pu.gather(-1, ids.unsqueeze(-1)).squeeze(-1)
            ratio = p_now / commit_p.clamp(min=1e-6)
            drop = comm & (ratio < rcr_ratio)
            for b in range(B):
                idxs = drop[b].nonzero().squeeze(-1)
                if idxs.numel() > rcr_cap:
                    keep = ratio[b, idxs].argsort()[:rcr_cap]
                    sel = idxs[keep]
                else:
                    sel = idxs
                if sel.numel():
                    fires['rcr'] += sel.numel()
                    ids[b, sel] = mid
                    still[b, sel] = True
                    commit_p[b, sel] = 0.0
            if drop.any():
                lg = guided(ids, max(still.float().mean().item(), T_MIN))
                pu = F.softmax(lg, dim=-1)

        lt = lg / temperature
        kth = lt.topk(top_k, dim=-1).values[..., -1:]
        lt = lt.masked_fill(lt < kth, float("-inf"))
        probs = F.softmax(lt, dim=-1)
        sampled = torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(B, -1)
        conf = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
        conf = conf.masked_fill(~still, float("inf"))
        p_smp = pu.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)

        n_keep = int(still[:, P:].sum(1).float().mean().item()
                     * math.cos(math.pi / 2 * (s + 1) / steps))
        prev_still = still.clone()
        ids = torch.where(still, sampled, ids)
        if n_keep > 0:
            remask = torch.zeros_like(still)
            remask.scatter_(1, conf.argsort(dim=1)[:, :n_keep], True)
            remask &= still
            ids = torch.where(remask, mid, ids)
            still = remask
        else:
            still = torch.zeros_like(still)

        new = prev_still & ~still  # committed this step
        commit_p = torch.where(new, p_smp, commit_p)
        held = ~still & (ids != mid)
        held[:, :P] = False
        p_cur = pu.gather(-1, ids.unsqueeze(-1)).squeeze(-1)
        commit_p = torch.where(held, torch.maximum(commit_p, p_cur), commit_p)

        # lookahead verify: re-score with commits in place; drop ones that
        # got WORSE once their neighbors could see them
        if look and new.any() and s < steps - 1:
            lg2 = guided(ids, max(still.float().mean().item(), T_MIN))
            p2 = F.softmax(lg2, -1).gather(-1, ids.unsqueeze(-1)).squeeze(-1)
            bad = new & (p2 < look_ratio * commit_p.clamp(min=1e-6))
            if bad.any():
                fires['look'] += int(bad.sum())
                ids = ids.masked_fill(bad, mid)
                still = still | bad
                commit_p = torch.where(bad, torch.zeros_like(commit_p), commit_p)
        if not still.any():
            break
    return ids, fires

model, mid = load(CKPTS["cfg2  (pre-repair)"])
PROMPTS = ["What is the capital of France?", "What is the capital of Japan?",
           "What color is the sky?", "Write one sentence about the ocean.",
           "Tell me about dogs."]
CONFIGS = [("aggressive +RCR+look", dict(rcr=True, look=True))]
for name, kw in CONFIGS:
    torch.manual_seed(11)
    print(f"===== {name} =====", flush=True)
    for q in PROMPTS:
        p = f"Question: {q}\nAnswer:"
        pi = torch.tensor([tok.encode(p, add_special_tokens=False)], device=DEV)
        out, fires = gen3(model, mid, pi, **kw)
        print(f"  {q} [rcr:{fires['rcr']} look:{fires['look']}] ->{show(out, pi.shape[1])}", flush=True)
print("RCR_LOOK_DONE", flush=True)
