# DIMBA post-training

These scripts preserve clean-prompt conditioning and score only response
positions. Run them from the repository root with `PYTHONPATH=src`.

## Choose the objective

- `finetune_sft.py` — supervised fine-tuning from instruction/response data.
- `finetune_dpo.py` — DPO, IPO, or reference-free SimPO on preference pairs,
  using a diffusion-ELBO/VRPO log-probability surrogate.
- `finetune_grpo.py` — group-relative policy optimization with a pluggable
  verifiable reward. `numeric` is the default; `token_overlap` is deprecated
  because it rewards copying.
- `finetune_interactive.py` — convenience wizard for SFT or GRPO. DPO/IPO/SimPO
  are launched directly.

## Examples

SFT:

```bash
PYTHONPATH=src python3 scripts/finetuning/finetune_sft.py \
  --base-checkpoint checkpoints/base.pt \
  --dataset HuggingFaceH4/ultrachat_200k \
  --dataset-split train_sft \
  --batch-size 4 \
  --grad-accumulation-steps 16 \
  --num-epochs 3 \
  --output-dir checkpoints/sft
```

DPO with antithetic Monte Carlo timesteps:

```bash
PYTHONPATH=src python3 scripts/finetuning/finetune_dpo.py \
  --base-checkpoint checkpoints/base.pt \
  --dataset m-a-p/CodeFeedback-Filtered-Preference \
  --loss-type dpo \
  --mc-samples 2 \
  --antithetic \
  --batch-size 2 \
  --output-dir checkpoints/dpo
```

Change `--loss-type` to `ipo` or `simpo`; SimPO is reference-free and also
accepts `--gamma`.

GRPO with the numeric reward:

```bash
PYTHONPATH=src python3 scripts/finetuning/finetune_grpo.py \
  --base-checkpoint checkpoints/base.pt \
  --dataset openai/gsm8k \
  --reward numeric \
  --num-generations 4 \
  --sampling-steps 32 \
  --batch-size 2 \
  --output-dir checkpoints/grpo
```

All three direct trainers support LoRA through `--use-lora`; add `--use-qlora`
only on a supported low-memory backend. Check each script's `--help` before a
run because their batch, sequence-length, and tokenizer flag names differ.

## Checkpoints and resume

SFT, DPO/IPO/SimPO, and GRPO save atomic portable weight checkpoints, but these
direct post-training scripts do **not** restore optimizer state, RNG state, or a
data cursor. They therefore do not provide exact mid-run resume and expose no
`--resume` flag. Restart a selected phase from a saved weight checkpoint; do
not describe that as an exact continuation.

The exact-resume paths are continuous Stage 3 in `scripts/train_h100.py` and
same-topology masked base/SFT training in `scripts/masked_diffusion_finetune.py`
and `scripts/mdm_sft_cfg2.py`.
