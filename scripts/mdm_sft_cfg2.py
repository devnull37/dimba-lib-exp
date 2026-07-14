"""SFT for the masked-diffusion model (LLaDA-style instruction tuning).

Starts from checkpoints/mdm_sft_cfg/mdm_sft_final.pt. Each example is a fixed
SEQ_LEN row: [prompt | response | EOS padding]. The prompt is NEVER masked
(always clean conditioning); the response plus its first EOS (not the pad tail)
is masked at ratio t ~ U[T_MIN,1]
and trained with CE on masked positions, 1/t weighted. Run the documented
quality gate before preference optimization; no RL method can bootstrap useful
rewards from an incoherent base.

Data: tatsu-lab/alpaca (52k, no trust_remote_code), template
"Question: {instruction}\n{input}\nAnswer: {output}".

Multi-GPU: ``torchrun --standalone --nproc-per-node=8 scripts/mdm_sft_cfg2.py``.
``--batch`` is per GPU; effective batch is batch * processes * accumulation.
"""
import argparse
import inspect
import math
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from dimba import DIMBA
from dimba.training.distributed import (
    backward_sync_context,
    cycling_batches,
    init_distributed,
    make_sampler,
    rank_zero_torch_cache,
    reduce_metrics,
    restore_training_state,
    save_training_checkpoint,
    seed_process,
    wrap_ddp,
)
from dimba.training.masked import (
    MaskedDiffusionObjective,
    freeze_masked_only_unused_parameters,
)
from dimba.training.optimizers import HybridMuon, build_optimizer
from dimba.utils.backends import configure_cuda_training, require_fast_cuda_mamba2

CKPT = "checkpoints/mdm_sft_cfg/mdm_sft_final.pt"
OUT_DIR = "checkpoints/mdm_sft_cfg2"
P_DROP = 0.1          # fraction of rows trained with a null (fully masked) prompt
GUIDANCE = 2.5        # eval-time CFG scale: uncond + s * (cond - uncond)
UL_WEIGHT = 0.5       # unlikelihood penalty on visible-neighbor repeats
SEQ_LEN = 256
LR = 5e-5
WARMUP = 100
EVAL_EVERY = 500
T_MIN = 0.03
DEVICE = "cuda"

EVAL_PROMPTS = [
    "Question: What is the capital of France?\nAnswer:",
    "Question: Write one sentence about the ocean.\nAnswer:",
    "Question: What is 2 + 3?\nAnswer:",
]


class SFTRows(Dataset):
    """Fixed-length rows + prompt/response lengths, prebuilt on CPU."""

    def __init__(self, rows, plens, rlens):
        self.rows, self.plens, self.rlens = rows, plens, rlens

    def __len__(self):
        return self.rows.shape[0]

    def __getitem__(self, i):
        return self.rows[i], self.plens[i], self.rlens[i]


def build_sft_rows(tokenizer):
    from datasets import load_dataset
    print("loading tatsu-lab/alpaca ...", flush=True)
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    pairs = []
    for ex in ds:
        q = ex["instruction"]
        if ex.get("input"):
            q += "\n" + ex["input"]
        pairs.append((q, ex["output"]))
    # synthetic arithmetic: Alpaca has almost none, and the tokenizer is already
    # digit-split — the math failure is a data gap, not a tokenization problem
    import random
    rng = random.Random(0)
    for _ in range(20000):
        a, b = rng.randint(0, 99), rng.randint(0, 99)
        op = rng.choice(["+", "-", "*"])
        if op == "*":
            a, b = rng.randint(0, 12), rng.randint(0, 12)
        ans = a + b if op == "+" else (a - b if op == "-" else a * b)
        q = rng.choice([f"What is {a} {op} {b}?", f"Calculate {a} {op} {b}.",
                        f"{a} {op} {b} = ?"])
        pairs.append((q, f"{a} {op} {b} = {ans}"))
    # SmolTalk: HF's SFT mix built for SmolLM-135M/360M — single-turn pairs only,
    # capped so tokenization stays fast; full-fit rows only (no truncated answers)
    ds2 = load_dataset("HuggingFaceTB/smoltalk", "all", split="train")
    st = []
    for ex in ds2:
        m = ex["messages"]
        if len(m) >= 2 and m[0]["role"] == "user" and m[1]["role"] == "assistant":
            st.append((m[0]["content"], m[1]["content"]))
    rng.shuffle(st)
    pairs.extend(st[:350000])
    rng.shuffle(pairs)
    print(f"pairs: {len(pairs)} (alpaca + 20k math + {min(len(st),350000)} smoltalk)", flush=True)
    rows, plens, rlens = [], [], []
    for q, out in pairs:
        p = tokenizer.encode(f"Question: {q}\nAnswer:", add_special_tokens=False)
        r = tokenizer.encode(" " + out, add_special_tokens=False)
        if len(p) >= SEQ_LEN - 8:  # need room for at least a stub answer
            continue
        if len(p) + len(r) > SEQ_LEN - 1:
            continue
        row = p + r + [eos]
        rlen = len(row)  # content + exactly one EOS; padding beyond is NOT trained
        row = row + [eos] * (SEQ_LEN - len(row))
        rows.append(row)
        plens.append(len(p))
        rlens.append(rlen)
    rows = torch.tensor(rows, dtype=torch.long)
    plens = torch.tensor(plens, dtype=torch.long)
    rlens = torch.tensor(rlens, dtype=torch.long)
    print(f"SFT rows: {rows.shape[0]} x {SEQ_LEN}", flush=True)
    return SFTRows(rows, plens, rlens)


@torch.inference_mode()
def maskgit_generate(model, prompt_ids, gen_len, mask_id,
                     steps=32, temperature=0.7, top_k=50, guidance=0.0):
    device = prompt_ids.device
    B, P = prompt_ids.shape
    ids = torch.cat([prompt_ids,
                     torch.full((B, gen_len), mask_id, dtype=torch.long,
                                device=device)], dim=1)
    positions = torch.arange(P, P + gen_len, device=device).unsqueeze(0).expand(B, -1)
    n_active = gen_len
    for s in range(steps):
        t = max(n_active / (P + gen_len), T_MIN)
        if guidance not in (0.0, 1.0):
            null_ids = ids.clone()
            null_ids[:, :P] = mask_id
            features = model.predict_token_features(torch.cat((ids, null_ids), dim=0), t)
            conditional, unconditional = features.chunk(2, dim=0)
            features = unconditional + guidance * (conditional - unconditional)
        else:
            # Historical g0 means conditional generation with CFG disabled.
            features = model.predict_token_features(ids, t)
        logits = model.project_token_features(features, positions=positions).float()
        logits = logits / temperature
        if top_k:
            top_values, top_indices = logits.topk(top_k, dim=-1)
        else:
            top_values = logits
            top_indices = torch.arange(logits.shape[-1], device=device).view(1, 1, -1)
        probs = F.softmax(top_values, dim=-1)
        choice = torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1).view(B, -1, 1)
        sampled = top_indices.expand(B, n_active, -1).gather(-1, choice).squeeze(-1)
        confidence = probs.gather(-1, choice).squeeze(-1)
        ids = ids.scatter(1, positions, sampled)
        n_keep = int(gen_len * math.cos(math.pi / 2 * (s + 1) / steps))
        if n_keep > 0:
            keep_local = confidence.topk(n_keep, dim=1, largest=False).indices
            positions = positions.gather(1, keep_local)
            ids = ids.scatter(1, positions, torch.full_like(positions, mask_id))
        else:
            break
        n_active = n_keep
    return ids


@torch.inference_mode()
def eval_step(model, tokenizer, step: int, mask_id: int):
    model.eval()
    device = next(model.parameters()).device
    eos = tokenizer.eos_token_id
    for p in EVAL_PROMPTS:
        ids = torch.tensor([tokenizer.encode(p, add_special_tokens=False)],
                           dtype=torch.long, device=device)
        for gtag, gs in (("g0", 0.0), (f"g{GUIDANCE}", GUIDANCE)):
            g = maskgit_generate(model, ids, gen_len=48, mask_id=mask_id,
                                 guidance=gs)
            toks = [x for x in g[0, ids.shape[1]:].tolist() if x < mask_id]
            if eos in toks:
                toks = toks[:toks.index(eos)]
            print(f"[step {step}][{gtag}] {p!r} -> {tokenizer.decode(toks)!r}",
                  flush=True)
    model.train()


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--batch", type=int, default=64, help="microbatch per GPU")
    ap.add_argument("--accumulate", type=int, default=1, help="gradient accumulation steps")
    ap.add_argument("--optimizer", choices=["adamw", "muon"], default="adamw")
    ap.add_argument("--output-dir", default=OUT_DIR, help="checkpoint/cache directory")
    ap.add_argument("--resume", help="exact training checkpoint to resume")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default=DEVICE)
    ap.add_argument("--workers", type=int, default=2, help="data-loader workers per process")
    ap.add_argument("--local-rank", "--local_rank", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    if args.batch < 1 or args.accumulate < 1 or args.workers < 0:
        ap.error("--batch and --accumulate must be positive; --workers cannot be negative")
    return args


def _run(args, context):
    if args.smoke:
        args.steps, args.batch, args.accumulate = 20, 8, 1

    seed_process(args.seed, context)
    configure_cuda_training(context.device)
    if context.is_main:
        global_batch = args.batch * args.accumulate * context.world_size
        print(
            f"distributed={context.enabled} world_size={context.world_size} "
            f"device={context.device} per_gpu_batch={args.batch} "
            f"accumulate={args.accumulate} global_batch={global_batch}",
            flush=True,
        )

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")

    source_checkpoint = args.resume or CKPT
    ck = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    cfg, sd, mask_id = dict(ck["config"]), ck["model_state_dict"], ck["mask_id"]
    sig = set(inspect.signature(DIMBA.__init__).parameters) - {"self"}
    model = DIMBA(**{k: v for k, v in cfg.items() if k in sig})
    miss, unexp = model.load_state_dict(sd, strict=True)
    if context.is_main:
        print(f"loaded {source_checkpoint}: vocab={cfg['vocab_size']} mask_id={mask_id} "
              f"missing={len(miss)} unexpected={len(unexp)}", flush=True)
    assert not miss and not unexp
    require_fast_cuda_mamba2(model, context.device)
    frozen = freeze_masked_only_unused_parameters(model)
    if context.is_main:
        print(f"masked-only freeze: {len(frozen)} unused prompt-encoder tensors", flush=True)
    dtype = torch.bfloat16 if context.device.type == "cuda" else torch.float32
    model = model.to(device=context.device, dtype=dtype)
    del sd

    def build_cached_rows():
        rows = build_sft_rows(tokenizer)
        return rows.rows, rows.plens, rows.rlens

    data_id = f"sft_rows_v2_{SEQ_LEN}"
    cached_rows = rank_zero_torch_cache(
        f"{args.output_dir}/data/{data_id}.pt",
        context,
        build_cached_rows,
    )
    data = SFTRows(*cached_rows)
    sampler = make_sampler(data, context, seed=args.seed)
    loader = DataLoader(
        data,
        batch_size=args.batch,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=context.device.type == "cuda",
        drop_last=True,
        persistent_workers=args.workers > 0,
        generator=torch.Generator().manual_seed(args.seed + context.rank),
    )
    if len(loader) == 0:
        raise ValueError("Dataset shard is smaller than one per-GPU batch")

    objective = MaskedDiffusionObjective(
        model,
        mask_id,
        t_min=T_MIN,
        prompt_dropout=P_DROP,
        neighbor_unlikelihood_weight=UL_WEIGHT,
    )
    train_objective = wrap_ddp(objective, context)

    opt = build_optimizer(
        model,
        name=args.optimizer,
        lr=LR,
        weight_decay=0.0,
        fused=context.device.type == "cuda",
    )
    if context.is_main and isinstance(opt, HybridMuon):
        print(
            f"Muon split: {len(opt.muon_parameter_names)} hidden matrices; "
            f"{len(opt.adamw_parameter_names)} tensors on AdamW",
            flush=True,
        )
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / WARMUP) *
        (0.5 * (1 + math.cos(min(s, args.steps) / args.steps * math.pi)) * 0.9 + 0.1))

    step = epoch = batch_in_epoch = 0
    run_signature = {
        "data_id": data_id,
        "sequence_length": SEQ_LEN,
        "learning_rate": LR,
        "warmup": WARMUP,
        "t_min": T_MIN,
        "prompt_dropout": P_DROP,
        "neighbor_unlikelihood_weight": UL_WEIGHT,
        "masked_prompt_encoder_frozen": True,
        "device_type": context.device.type,
    }
    if args.resume:
        expected = {
            "optimizer_name": args.optimizer,
            "per_gpu_batch": args.batch,
            "grad_accum": args.accumulate,
            "planned_steps": args.steps,
            "world_size": context.world_size,
            "seed": args.seed,
            "run_signature": run_signature,
        }
        mismatches = {
            key: (ck.get(key), value) for key, value in expected.items() if ck.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Exact resume configuration mismatch: {mismatches}")
        step, epoch, batch_in_epoch = restore_training_state(
            ck,
            context=context,
            optimizer=opt,
            scheduler=sched,
        )
        if context.is_main:
            print(
                f"resumed step={step} epoch={epoch} batch_in_epoch={batch_in_epoch}",
                flush=True,
            )
    del ck

    if not args.smoke and not args.resume:
        context.barrier()
        if context.is_main:
            print("baseline (mdm pretrained, before SFT):", flush=True)
            eval_step(model, tokenizer, 0, mask_id)
        context.barrier()

    train_objective.train()
    start_step, t0 = step, time.time()
    batches = cycling_batches(
        loader,
        sampler,
        epoch=epoch,
        batch_in_epoch=batch_in_epoch,
    )
    checkpoint_extra = {
        "config": cfg,
        "mask_id": mask_id,
        "optimizer_name": args.optimizer,
        "per_gpu_batch": args.batch,
        "grad_accum": args.accumulate,
        "planned_steps": args.steps,
        "seed": args.seed,
        "run_signature": run_signature,
    }
    while step < args.steps:
        opt.zero_grad(set_to_none=True)
        metric_sums = {}
        for microstep in range(args.accumulate):
            batch, epoch, batch_in_epoch = next(batches)
            rows, plens, rlens = (
                tensor.to(context.device, non_blocking=True) for tensor in batch
            )
            with backward_sync_context(
                train_objective, sync=microstep == args.accumulate - 1
            ):
                loss, parts = train_objective(rows, plens, rlens)
                (loss / args.accumulate).backward()
            current = {"loss": loss.detach(), **parts}
            for key, value in current.items():
                metric_sums[key] = metric_sums.get(key, 0) + value.detach()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1
        if step % 100 == 0 or step == 1:
            metrics = reduce_metrics(
                {key: value / args.accumulate for key, value in metric_sums.items()},
                context,
            )
            if context.is_main:
                ps = {
                    key: f"{value.item():.4f}"
                    for key, value in metrics.items()
                    if key != "loss"
                }
                tokens = (
                    (step - start_step)
                    * args.batch
                    * args.accumulate
                    * context.world_size
                    * SEQ_LEN
                )
                tok_s = tokens / max(time.time() - t0, 1e-6)
                print(f"step {step}/{args.steps} loss={metrics['loss'].item():.4f} {ps} "
                      f"lr={sched.get_last_lr()[0]:.2e} {tok_s / 1e3:.0f}k tok/s "
                      f"({time.time() - t0:.0f}s)", flush=True)
        if not args.smoke and step % EVAL_EVERY == 0:
            context.barrier()
            if context.is_main:
                eval_step(model, tokenizer, step, mask_id)
            context.barrier()
            save_training_checkpoint(
                f"{args.output_dir}/mdm_sft_latest.pt",
                context=context,
                model=model,
                optimizer=opt,
                scheduler=sched,
                step=step,
                epoch=epoch,
                batch_in_epoch=batch_in_epoch,
                extra=checkpoint_extra,
            )

    if args.smoke:
        context.barrier()
        if context.is_main:
            eval_step(model, tokenizer, step, mask_id)
        context.barrier()
    else:
        save_training_checkpoint(
            f"{args.output_dir}/mdm_sft_final.pt",
            context=context,
            model=model,
            optimizer=opt,
            scheduler=sched,
            step=step,
            epoch=epoch,
            batch_in_epoch=batch_in_epoch,
            extra=checkpoint_extra,
        )
    if context.is_main:
        print("MDM_SFT_DONE", flush=True)


def main(argv=None):
    args = parse_args(argv)
    context = init_distributed(args.device)
    try:
        _run(args, context)
    finally:
        context.close()


if __name__ == "__main__":
    main()
