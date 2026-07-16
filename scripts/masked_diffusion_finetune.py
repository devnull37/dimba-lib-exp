"""Masked discrete diffusion (LLaDA/MDLM-style) training for the DIMBA backbone.

This is the discrete alternative to the continuous high-noise repair described in
docs/ROOT_CAUSE_BUDGET28.md; both tracks remain supported until measured quality
gates choose between them. The masked track switches the *objective*:
corrupt by replacing tokens with a new [MASK] token, predict the originals
with plain cross-entropy on masked positions (LLaDA 1/t weighting). No SNR
weighting, no latent blending — both proven pathologies vanish by
construction. A continuous checkpoint's bidirectional Mamba backbone can be
reused; a fresh conversion adds one [MASK] embedding row initialized to the
mean embedding. The repo's
predict_token_logits() is the intended entry point for exactly this track
(mask-ratio in (0,1] is mapped onto the timestep conditioning).

Generation: MaskGIT-style iterative unmasking (confidence-based, cosine
schedule). Probes + infill-recovery gate + checkpoint every EVAL_EVERY steps.

Multi-GPU: ``torchrun --standalone --nproc-per-node=8 scripts/masked_diffusion_finetune.py``.
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

CKPT = "checkpoints/masked_diffusion/mdm_final.pt"
OUT_DIR = "checkpoints/masked_diffusion2"
SEQ_LEN = 512
LR = 1e-4
WARMUP = 200
EVAL_EVERY = 500
T_MIN = 0.03          # avoid the 1/t blowup; LLaDA clamps similarly
DEVICE = "cuda"

EVAL_PROMPTS = [
    "The capital of France is",
    "Once upon a time there",
]
GATE_TEXT = ("The quick brown fox jumps over the lazy dog while the sun sets "
             "behind the mountains and the river flows quietly through the valley.")


class PackedChunks(Dataset):
    def __init__(self, stream: torch.Tensor, seq_len: int):
        self.stream, self.seq_len = stream, seq_len
        self.n = max(0, stream.numel() // seq_len)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.stream[i * self.seq_len:(i + 1) * self.seq_len]


def build_stream(tokenizer, n_docs: int, skip_docs: int = 0) -> torch.Tensor:
    from datasets import load_dataset
    print(f"streaming FineWeb sample-10BT, skipping {skip_docs}, "
          f"caching {n_docs} docs ...", flush=True)
    ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT",
                      split="train", streaming=True)
    if skip_docs:
        ds = ds.skip(skip_docs)
    eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    chunks = []
    for i, row in enumerate(ds):
        if i >= n_docs:
            break
        enc = tokenizer.encode(row["text"], add_special_tokens=False)
        enc.append(eos)
        chunks.append(torch.tensor(enc, dtype=torch.long))
        if (i + 1) % 20000 == 0:
            print(f"  {i + 1}/{n_docs} docs", flush=True)
    stream = torch.cat(chunks)
    print(f"cache ready: {stream.numel() / 1e6:.1f}M tokens", flush=True)
    return stream


@torch.inference_mode()
def maskgit_generate(model, prompt_ids: torch.Tensor, gen_len: int, mask_id: int,
                     steps: int = 32, temperature: float = 0.7, top_k: int = 50):
    """Device-resident iterative unmasking; project only unresolved positions."""
    device = prompt_ids.device
    B, P = prompt_ids.shape
    ids = torch.cat([prompt_ids,
                     torch.full((B, gen_len), mask_id, dtype=torch.long,
                                device=device)], dim=1)
    positions = torch.arange(P, P + gen_len, device=device).unsqueeze(0).expand(B, -1)
    n_active = gen_len
    for s in range(steps):
        t = max(n_active / (P + gen_len), T_MIN)
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
    for p in EVAL_PROMPTS:
        ids = torch.tensor([tokenizer.encode(p, add_special_tokens=False)],
                           dtype=torch.long, device=device)
        g = maskgit_generate(model, ids, gen_len=64, mask_id=mask_id)
        txt = tokenizer.decode([x for x in g[0].tolist() if x < mask_id])
        print(f"[step {step}] {p!r} -> {txt!r}", flush=True)
    # infill-recovery gate: mask 50% of a held-out sentence, measure recovery
    gt = torch.tensor([tokenizer.encode(GATE_TEXT, add_special_tokens=False)],
                      dtype=torch.long, device=device)
    g_rng = torch.Generator(device=gt.device).manual_seed(0)
    m = torch.rand(gt.shape, generator=g_rng, device=gt.device) < 0.5
    corrupted = torch.where(m, mask_id, gt)
    logits = model.predict_token_logits(corrupted, 0.5)
    pred = logits.argmax(dim=-1)
    acc = (pred[m] == gt[m]).float().mean().item()
    print(f"[step {step}] infill@50% recovery: {acc:.3f}", flush=True)
    model.train()
    return acc


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=64, help="microbatch per GPU")
    ap.add_argument("--accumulate", type=int, default=1, help="gradient accumulation steps")
    ap.add_argument("--docs", type=int, default=200_000)
    ap.add_argument("--skip-docs", type=int, default=0)
    ap.add_argument("--optimizer", choices=["adamw", "muon"], default="adamw")
    ap.add_argument("--output-dir", default=OUT_DIR, help="checkpoint/cache directory")
    ap.add_argument("--resume", help="exact training checkpoint to resume")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default=DEVICE)
    ap.add_argument("--workers", type=int, default=2, help="data-loader workers per process")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the denoiser forward (CUDA only): fuses "
                         "the norm/AdaLN/FFN/flip glue between the fused "
                         "mamba_ssm kernels; the data-dependent masked loss "
                         "stays eager")
    ap.add_argument("--local-rank", "--local_rank", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    if args.batch < 1 or args.accumulate < 1 or args.workers < 0:
        ap.error("--batch and --accumulate must be positive; --workers cannot be negative")
    return args


def _run(args, context):
    if args.smoke:
        args.steps, args.batch, args.docs, args.accumulate = 20, 8, 300, 1

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
    cfg = dict(ck["config"])
    sd = ck["model_state_dict"]
    emb_key = next(k for k in sd if k.endswith("token_embed.embedding.weight"))
    if "mask_id" in ck:
        # resuming a masked-diffusion checkpoint: [MASK] row already present
        mask_id = ck["mask_id"]
    else:
        old_vocab = sd[emb_key].shape[0]
        mask_id = old_vocab
        # one new row for [MASK], init = mean embedding (head is tied, so this is all)
        sd[emb_key] = torch.cat([sd[emb_key],
                                 sd[emb_key].mean(dim=0, keepdim=True)], dim=0)
        cfg["vocab_size"] = old_vocab + 1
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

    if args.compile and context.device.type == "cuda":
        # Compile only the backbone forward: `features[target_mask]` in the
        # loss is data-dependent and would recompile every step, so the loss
        # math stays eager. dynamic=False: training shapes are fixed
        # ([batch, SEQ_LEN]), and static shapes give inductor the best kernels.
        model.predict_token_features = torch.compile(
            model.predict_token_features, dynamic=False
        )
        if context.is_main:
            print("torch.compile: denoiser forward compiled (dynamic=False)",
                  flush=True)

    data_id = f"fineweb_{args.skip_docs}_{args.docs}_{SEQ_LEN}"
    cache = f"{args.output_dir}/data/{data_id}.pt"
    stream = rank_zero_torch_cache(
        cache,
        context,
        lambda: build_stream(tokenizer, args.docs, args.skip_docs),
    )
    data = PackedChunks(stream, SEQ_LEN)
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

    objective = MaskedDiffusionObjective(model, mask_id, t_min=T_MIN)
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
            print("baseline (repaired_final backbone, before masked-diffusion training):",
                  flush=True)
            eval_step(model, tokenizer, step, mask_id)
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
            batch = batch.to(context.device, non_blocking=True)
            with backward_sync_context(
                train_objective, sync=microstep == args.accumulate - 1
            ):
                loss, parts = train_objective(batch)
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
                f"{args.output_dir}/mdm_latest.pt",
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
            f"{args.output_dir}/mdm_final.pt",
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
        print("MDM_DONE", flush=True)


def main(argv=None):
    args = parse_args(argv)
    context = init_distributed(args.device)
    try:
        _run(args, context)
    finally:
        context.close()


if __name__ == "__main__":
    main()
