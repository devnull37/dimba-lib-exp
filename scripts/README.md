# DIMBA scripts

Run commands from the repository root with `PYTHONPATH=src`. See
`docs/PERFORMANCE_AND_SCALING.md` for the production backend and benchmark policy.

## Canonical training launchers

### Continuous Stage 3 on one H100

Validate the resolved recipe without CUDA:

```bash
PYTHONPATH=src python3 scripts/train_h100.py \
  --preset repair1b \
  --phase distill \
  --stage3-optimizer adamw \
  --save-dir checkpoints/repair1b \
  --dry-run
```

Remove `--dry-run` on the H100. This launcher requires the fused
`mamba_ssm.Mamba2` plus `causal-conv1d` stack and rejects `--phase all`. Continuous Stage 3
supports single-node `torchrun`; alignment runs once on rank 0 and exact resume requires the same
world size. SFT/GRPO remain single-process. AdamW is the default; Muon is an opt-in
pilot with `--stage3-optimizer muon`. Run each phase separately and pass
`--quality-gate-passed` before SFT or GRPO. Use a distinct `--save-dir` for
every AdamW/Muon arm.

Stage-3 exact resume requires the same preset, optimizer, batch/backend,
topology, and data configuration:

```bash
PYTHONPATH=src python3 scripts/train_h100.py \
  --preset repair1b \
  --phase distill \
  --save-dir checkpoints/repair1b \
  --resume \
  --checkpoint checkpoints/repair1b/distill_latest.pt
```

### Masked base training with DDP

```bash
PYTHONPATH=src torchrun --standalone --nproc-per-node=8 \
  scripts/masked_diffusion_finetune.py \
  --steps 40000 \
  --batch 8 \
  --accumulate 2 \
  --optimizer adamw \
  --output-dir checkpoints/masked-base-adamw
```

### Masked SFT with DDP

```bash
PYTHONPATH=src torchrun --standalone --nproc-per-node=8 \
  scripts/mdm_sft_cfg2.py \
  --steps 15000 \
  --batch 8 \
  --accumulate 2 \
  --optimizer adamw \
  --output-dir checkpoints/masked-sft-adamw
```

For both masked launchers, `--batch` is the microbatch per GPU and global batch
is `batch × world size × accumulate`. `--resume PATH` is exact only with the
same world size and run signature. Give every AdamW/Muon arm a distinct
`--output-dir`.

### Apple Silicon development training

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=src \
  python3 scripts/train_interactive.py
```

Select `mps-small`. It is the tested fp32 PyTorch-MPS latent recipe. MLX is an
inference backend, not a training backend.

## H100 performance gate

Run the complete parity and promotion suite on a real H100:

```bash
PYTHONPATH=src python3 scripts/benchmark_h100.py \
  --checkpoint checkpoints/model.pt \
  --output artifacts/h100-benchmark.json \
  --fail-on-gate
```

The default suite runs masked inference, continuous DDIM, continuous DPM++,
continuous fused CE, and masked training. A subset is diagnostic only and is
not promotion evidence. `--dry-run` validates the benchmark plan on a non-CUDA
machine.

## Inference and evaluation

```bash
PYTHONPATH=src python3 scripts/generate.py \
  "What is the capital of France?" \
  --checkpoint checkpoints/model.pt \
  --quality 0.5 \
  --backend auto

PYTHONPATH=src python3 scripts/evaluate.py --help
```

`generate.py` loads the architecture embedded in the checkpoint with a strict
state-dict check. `--backend auto` selects MLX on supported Apple Silicon and
Torch otherwise.

## Other maintained entry points

- `train.py` — generic YAML-driven training.
- `train_vae.py` — TokenVAE experiments.
- `train_interactive.py` — local preset wizard.
- `finetuning/finetune_sft.py` — direct SFT.
- `finetuning/finetune_dpo.py` — DPO, IPO, and SimPO.
- `finetuning/finetune_grpo.py` — reward-driven GRPO.
- `upload_to_hf.py` — Hugging Face upload utility.

Inspect an entry point before launching it:

```bash
PYTHONPATH=src python3 scripts/<script>.py --help
```

Older experiment-specific scripts remain for reproducibility; they are not the
canonical next-run launchers.
