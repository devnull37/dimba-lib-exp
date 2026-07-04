"""Bidirectional-backbone A/B for DIMBA's masked-diffusion denoiser.

Three from-scratch arms, identical data/seed/optimizer, isolating ONLY how the
two scan directions get their weights:

  A "double"      (control): the current architecture. Each Mamba2Block owns two
                  independent directional mixers (mamba_fwd + mamba_bwd).
  B "shared":     ONE mixer per block, its Parameter objects reused for both
                  directions. We literally point block.mamba_bwd at block.mamba_fwd,
                  so the existing _mix() runs the shared stack on the sequence and
                  on the reversed sequence and sums them, exactly as the control
                  merges fwd/bwd -- just with tied weights.
  C "shared_lora": arm B, plus tiny per-direction LoRA adapters (rank 16, alpha 16)
                  on the mixer's principal projections (in_proj / out_proj). The
                  shared base is used both ways; a separate adapter set specializes
                  the forward pass and another the backward pass.

Correctness note: in the stock code (src/dimba/models/denoiser.py, Mamba2Block._mix)
the backward direction processes the REVERSED sequence:
    y_bwd = flip( mamba_bwd( flip(h) ) )
and merges by sum (bidir_merge="sum" in this checkpoint's config). Arms B and C
preserve that flip-in / flip-out behaviour precisely; only the weights change.

No library files are modified. All surgery is done here after construction.
"""
import argparse
import inspect
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, "/workspace/dimba-lib-exp/src")
sys.path.insert(0, "/workspace/dimba-lib-exp/scripts")
from transformers import AutoTokenizer  # noqa: E402
from dimba import DIMBA  # noqa: E402
import masked_diffusion_finetune as mdf  # noqa: E402

SCRATCH = "/tmp/claude-0/-workspace/e7af4c75-0211-4993-889c-4db42f486fc9/scratchpad"
CKPT = "/workspace/dimba-lib-exp/checkpoints/mdm_sft_cfg2/mdm_sft_final.pt"
RESULTS = "/workspace/dimba-lib-exp/scripts/experiments/bidir_ab_results.json"
DEV = "cuda"

# Training protocol (mirrors muon_ab.py / masked_diffusion_finetune.py).
BATCH = 64
LR = 1e-4
WARMUP = 100
LOG_EVERY = 100
TAIL_STEPS = 200          # tail CE = mean CE over the last 200 steps
N_DOCS = 60000            # same FineWeb cache size as muon_ab

LORA_R = 16
LORA_ALPHA = 16


# --------------------------------------------------------------------------- #
# LoRA on a wrapped nn.Linear, with per-direction adapters selected by a flag. #
# --------------------------------------------------------------------------- #
class _DirState:
    """Shared mutable direction selector: 0 = forward pass, 1 = backward pass."""

    def __init__(self):
        self.dir = 0


class LoRALinear(nn.Module):
    """Wrap a frozen? (no -- trainable base) nn.Linear and add TWO LoRA adapters,
    one per scan direction, selected at call time by a shared _DirState.

    Standard LoRA: y = base(x) + (alpha/r) * (x @ A^T) @ B^T, with A ~ N(0, 0.02^2)
    and B initialised to zero (so the adapter is a no-op at init and the arm starts
    identical to arm B). in_features/out_features are read from the wrapped Linear.
    """

    def __init__(self, base: nn.Linear, state: _DirState, r: int, alpha: float):
        super().__init__()
        self.base = base
        self.state = state
        self.r = r
        self.scaling = alpha / r
        in_f, out_f = base.in_features, base.out_features
        # Two adapters: index 0 -> forward, index 1 -> backward.
        self.lora_A = nn.ParameterList(
            [nn.Parameter(torch.empty(r, in_f)) for _ in range(2)]
        )
        self.lora_B = nn.ParameterList(
            [nn.Parameter(torch.zeros(out_f, r)) for _ in range(2)]
        )
        for a in self.lora_A:
            nn.init.normal_(a, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        d = self.state.dir
        A, B = self.lora_A[d], self.lora_B[d]
        lora = F.linear(F.linear(x, A.to(x.dtype)), B.to(x.dtype))
        return out + self.scaling * lora


# --------------------------------------------------------------------------- #
# Arm construction.                                                           #
# --------------------------------------------------------------------------- #
def _build_base_model():
    """Fresh RANDOM-init DIMBA from the checkpoint's config (weights discarded)."""
    ck = torch.load(CKPT, map_location="cpu")
    cfg, mid = dict(ck["config"]), ck["mask_id"]
    sig = set(inspect.signature(DIMBA.__init__).parameters) - {"self"}
    model = DIMBA(**{k: v for k, v in cfg.items() if k in sig})
    return model, mid


def _bind_shared_mix(block):
    """Return a bound _mix that runs block.mamba_fwd on the sequence AND on the
    reversed sequence, merging the same way stock code does (sum, this cfg).

    This mirrors Mamba2Block._mix exactly but calls the SAME mixer twice, so the
    forward and backward passes share weights. When LoRA is present the mixer's
    wrapped projections read block._dir_state to pick the per-direction adapter.
    """
    state = getattr(block, "_dir_state", None)

    def _mix(h):
        if state is not None:
            state.dir = 0
        y = block.mamba_fwd(h)
        if state is not None:
            state.dir = 1
        y_bwd = torch.flip(block.mamba_fwd(torch.flip(h, dims=[1])), dims=[1])
        if state is not None:
            state.dir = 0
        # this checkpoint uses bidir_merge="sum" (block.merge is None) -> add.
        if block.merge is not None:
            y = block.merge(torch.cat([y, y_bwd], dim=-1))
        else:
            y = y + y_bwd
        return y

    return _mix


def build_arm(arm: str):
    """Build one arm's model (random init). torch.manual_seed(0) is set by caller
    BEFORE this so init is comparable across arms."""
    model, mid = _build_base_model()
    blocks = model.denoiser.blocks

    if arm == "double":
        pass  # control: unmodified.

    elif arm in ("shared", "shared_lora"):
        for blk in blocks:
            assert blk.bidirectional and blk.mamba_bwd is not None
            # Tie backward -> forward: same Parameter objects. mamba_bwd is now
            # dead weight (never called after we override _mix), but we also drop
            # its module so its params don't count / don't get an optimizer entry.
            blk.mamba_bwd = None
            if arm == "shared_lora":
                state = _DirState()
                blk._dir_state = state
                # Per-direction LoRA on the mixer's PRINCIPAL projection, in_proj
                # (shape 2450x576 here: it produces z, x, B, C and dt -- the block's
                # single largest and most consequential matrix).
                #
                # Why in_proj only (not out_proj): the mamba_ssm.Mamba2 CUDA kernel's
                # fused mem-eff path is the only working path in this environment (the
                # non-fused fallback trips a causal_conv1d channel-last stride
                # assertion even with NO wrapper). That fused path calls self.in_proj(u)
                # as a real module -- so a LoRA wrapper on in_proj is applied correctly
                # -- but it reads out_proj.weight DIRECTLY (bypassing the module's
                # __call__), so a wrapper on out_proj would be silently skipped and the
                # non-fused route that would honour it is unavailable. in_proj alone is
                # the "LoRA on the block's principal matrix" arm; separate adapters
                # still specialise each direction.
                mixer = blk.mamba_fwd
                mixer.in_proj = LoRALinear(mixer.in_proj, state, LORA_R, LORA_ALPHA)
            # Override _mix to run the (now single) shared stack both directions.
            blk._mix = _bind_shared_mix(blk)
    else:
        raise ValueError(arm)

    return model, mid


def unique_trainable_params(model):
    """id()-deduplicated count of trainable parameters (tied params counted once)."""
    seen, tot = set(), 0
    for p in model.parameters():
        if p.requires_grad and id(p) not in seen:
            seen.add(id(p))
            tot += p.numel()
    return tot


# --------------------------------------------------------------------------- #
# Training (mirrors muon_ab.py adamw arm).                                    #
# --------------------------------------------------------------------------- #
def lr_factor(s, steps):
    w = min(1.0, (s + 1) / WARMUP)
    return w * (0.5 * (1 + math.cos(min(s, steps) / steps * math.pi)) * 0.9 + 0.1)


def run_arm(arm, stream, steps):
    torch.manual_seed(0)                      # comparable init across arms
    model, mid = build_arm(arm)
    model = model.to(DEV).to(torch.bfloat16)
    uniq = unique_trainable_params(model)
    # AdamW over unique trainable params only (avoid double-registering tied params).
    seen, params = set(), []
    for p in model.parameters():
        if p.requires_grad and id(p) not in seen:
            seen.add(id(p))
            params.append(p)
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)

    torch.manual_seed(0)                      # identical data shuffle across arms
    loader = DataLoader(mdf.PackedChunks(stream, mdf.SEQ_LEN), batch_size=BATCH,
                        shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

    curve, all_ce, step, t0, it = [], [], 0, time.time(), iter(loader)
    model.train()
    nan_hit = False
    while step < steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        batch = batch.to(DEV, non_blocking=True)
        loss, parts = mdf.masked_diffusion_loss(model, batch, mid)
        if not torch.isfinite(loss):
            print(f"[{arm}] NON-FINITE loss at step {step}; recording and stopping arm.",
                  flush=True)
            nan_hit = True
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        f = lr_factor(step, steps)
        for grp in opt.param_groups:
            grp["lr"] = LR * f
        opt.step()
        step += 1
        ce = parts["ce_masked"].item()
        all_ce.append(ce)
        if step % LOG_EVERY == 0:
            curve.append([step, ce])
            print(f"[{arm}] step {step}/{steps} ce_masked={ce:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    wall_min = (time.time() - t0) / 60.0
    tail = sum(all_ce[-TAIL_STEPS:]) / max(1, len(all_ce[-TAIL_STEPS:]))
    out = {
        "unique_trainable_params": uniq,
        "curve": curve,
        "tail_ce": tail,
        "wall_minutes": round(wall_min, 2),
        "steps_completed": step,
        "nan": nan_hit,
    }
    del model, opt
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    steps = args.steps
    if args.smoke:
        steps = 30

    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
    cache = f"{SCRATCH}/bidir_ab_stream.pt"
    if os.path.exists(cache):
        stream = torch.load(cache)
    else:
        alt = f"{SCRATCH}/muon_ab_stream.pt"
        if os.path.exists(alt):
            stream = torch.load(alt)          # reuse muon_ab's identical FineWeb cache
        else:
            stream = mdf.build_stream(tok, N_DOCS)
            torch.save(stream, cache)
    print(f"stream: {stream.numel() / 1e6:.1f}M tokens | steps={steps} smoke={args.smoke}",
          flush=True)

    res = {}
    for arm in ("double", "shared", "shared_lora"):
        print(f"===== ARM: {arm} =====", flush=True)
        # Print unique param count up front for verification.
        torch.manual_seed(0)
        _m, _ = build_arm(arm)
        print(f"[{arm}] unique trainable params: "
              f"{unique_trainable_params(_m) / 1e6:.4f}M", flush=True)
        del _m
        torch.cuda.empty_cache()
        res[arm] = run_arm(arm, stream, steps)
        r = res[arm]
        print(f"[{arm}] DONE uniq={r['unique_trainable_params']/1e6:.3f}M "
              f"tail_ce={r['tail_ce']:.4f} steps={r['steps_completed']} "
              f"min={r['wall_minutes']:.1f} nan={r['nan']}", flush=True)
        if not args.smoke:
            json.dump(res, open(RESULTS, "w"), indent=2)  # checkpoint after each arm

    if not args.smoke:
        json.dump(res, open(RESULTS, "w"), indent=2)
    print("BIDIR_AB_DONE", flush=True)


if __name__ == "__main__":
    main()
