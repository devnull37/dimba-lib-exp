# DIMBA Fine-Tuning

This subsystem provides the interactive fine-tuning wizard and dedicated SFT/GRPO training scripts.

## Wizard Usage

Launch:

```bash
python scripts/finetuning/finetune_interactive.py
```

Wizard flow (6 steps):
1. Select mode: `sft` or `grpo`
2. Choose base checkpoint path
3. Choose dataset (HF preset/custom or local file)
4. Select VRAM budget and tuning method
5. Set hyperparameters
6. Pick trainer script and launch

Trainer script auto-detection order is mode-aware:

For `sft` mode:
1. `scripts/finetuning/finetune_sft.py`
2. `scripts/finetuning/finetune_grpo.py`
3. `scripts/finetuning/train.py`
4. `scripts/train.py`

For `grpo` mode:
1. `scripts/finetuning/finetune_grpo.py`
2. `scripts/finetuning/finetune_sft.py`
3. `scripts/finetuning/train.py`
4. `scripts/train.py`

The wizard runs `<trainer> --help`, detects supported flags, and only passes flags found there.

## Script Interfaces

Dedicated scripts:
- SFT: `scripts/finetuning/finetune_sft.py`
- GRPO: `scripts/finetuning/finetune_grpo.py`

SFT core flags:
- Required: `--base-checkpoint`, `--dataset`, `--output-dir`
- Common: `--dataset-split`, `--dataset-config`, `--batch-size`, `--grad-accumulation-steps`, `--num-epochs`, `--max-steps`, `--learning-rate`, `--use-lora`, `--use-qlora`, `--lora-r`, `--lora-alpha`, `--device`

GRPO core flags:
- Required: `--base-checkpoint`, `--dataset`
- Common: `--dataset-split`, `--batch-size`, `--epochs`, `--max-steps`, `--learning-rate`, `--output-dir`, `--use-lora`, `--use-qlora`, `--lora-r`, `--lora-alpha`, `--device`

The wizard also supports generic/multi-mode trainer aliases when present:
- Mode aliases: `--mode`, `--train-mode`, `--training-mode`
- Base model aliases: `--base-model`, `--base-model-path`, `--base-checkpoint`, `--model-path`, `--model`, `--checkpoint`, `--checkpoint-path`
- Dataset aliases: `--dataset`, `--dataset-name`, `--dataset_name`, `--hf-dataset`, `--dataset-config`, `--dataset_config`, `--hf-dataset-config`, `--dataset-split`, `--dataset_split`, `--split`
- Local dataset aliases: `--dataset-path`, `--data-path`, `--train-file`, `--dataset-file`, `--dataset-format`, `--data-format`
- Method aliases: `--use-lora`, `--lora`, `--enable-lora`, `--use-qlora`, `--qlora`, `--full-finetune`, `--full-ft`
- Optimization aliases: `--batch-size`, `--per-device-train-batch-size`, `--train-batch-size`, `--micro-batch-size`, `--gradient-accumulation-steps`, `--grad-accumulation-steps`, `--grad-accum-steps`, `--accumulation-steps`, `--epochs`, `--num-train-epochs`, `--num-epochs`, `--max-steps`, `--max_steps`, `--learning-rate`, `--lr`, `--output-dir`, `--output_dir`
- LoRA aliases: `--lora-r`, `--lora_r`, `--rank`, `--lora-alpha`, `--lora_alpha`
- Hardware aliases: `--gpus`, `--num-gpus`, `--num_gpus`, `--device`, `--accelerator`

## LoRA vs Q-LoRA vs Full Fine-Tune

- `Q-LoRA`: best for low memory (`<12GB`) or CPU-only; lowest memory footprint.
- `LoRA`: strong default for mid-range memory (`12GB` to `<40GB`).
- `Full fine-tune`: typically only practical at high memory (`>=40GB`).

Wizard defaults follow that policy automatically.

## Dataset Options

Wizard menu options and preset defaults:
1. `ultrachat_200k` -> `HuggingFaceH4/ultrachat_200k` (default split `train_sft`)
2. `Code-Feedback` -> `m-a-p/CodeFeedback-Filtered-Instruction` for SFT, `m-a-p/CodeFeedback-Filtered-Preference` for GRPO (default split `train`)
3. `OpenHermes-2.5` -> `teknium/OpenHermes-2.5` (default split `train`)
4. `feedback-collection` -> `argilla/feedback-collection` (default split `train`)
5. Custom HF dataset
6. Local file

Recommended HF dataset IDs:
- `HuggingFaceH4/ultrachat_200k` for SFT chat tuning
- `m-a-p/CodeFeedback-Filtered-Instruction` (SFT)
- `m-a-p/CodeFeedback-Filtered-Preference` (preference/GRPO)
- `teknium/OpenHermes-2.5` for chat SFT
- `argilla/feedback-collection` for preference/GRPO

Local file notes:
- Use `.json` or `.jsonl` where possible.
- SFT data should provide instruction/chat supervision.
- GRPO data should provide preference pairs (`prompt`, `chosen`, `rejected`) or equivalent fields.

## Key CLI Examples

Wizard:

```bash
python scripts/finetuning/finetune_interactive.py
```

SFT with LoRA (HF dataset):

```bash
python scripts/finetuning/finetune_sft.py \
  --base-checkpoint checkpoints/base \
  --dataset HuggingFaceH4/ultrachat_200k \
  --dataset-split train_sft \
  --use-lora \
  --lora-r 8 \
  --lora-alpha 16 \
  --batch-size 4 \
  --grad-accumulation-steps 16 \
  --num-epochs 3 \
  --learning-rate 2e-4 \
  --output-dir checkpoints/finetune_sft_lora
```

GRPO with Q-LoRA (preference dataset):

```bash
python scripts/finetuning/finetune_grpo.py \
  --base-checkpoint checkpoints/base \
  --dataset argilla/feedback-collection \
  --dataset-split train \
  --use-lora \
  --use-qlora \
  --lora-r 4 \
  --lora-alpha 8 \
  --batch-size 1 \
  --epochs 1 \
  --max-steps 500 \
  --learning-rate 2e-4 \
  --output-dir checkpoints/finetune_grpo_qlora
```

SFT full fine-tune with local JSONL:

```bash
python scripts/finetuning/finetune_sft.py \
  --base-checkpoint checkpoints/base \
  --dataset data/my_sft.jsonl \
  --batch-size 2 \
  --grad-accumulation-steps 16 \
  --num-epochs 3 \
  --learning-rate 2e-5 \
  --output-dir checkpoints/finetune_sft_full
```

Inspect trainer flags before running direct CLI:

```bash
python scripts/finetuning/finetune_sft.py --help
python scripts/finetuning/finetune_grpo.py --help
```
