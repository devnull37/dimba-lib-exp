#!/usr/bin/env python3
"""Direct Preference Optimization (DPO) for DIMBA via an ELBO surrogate.

This script aligns a DIMBA diffusion language model on ``{prompt, chosen,
rejected}`` preference triplets using DPO (Rafailov et al., 2023,
arXiv:2305.18290). It mirrors the CLI and checkpoint-loading structure of
``finetune_sft.py`` and reuses the repo's LoRA / Q-LoRA helpers when importable.

Why an ELBO surrogate (diffusion-DPO):
    Standard DPO needs the sequence log-likelihood ``log pi(y | x)`` of the
    policy and a frozen reference. For an autoregressive model that is a cheap
    sum of token log-probs, but DIMBA is a **non-autoregressive masked diffusion
    LM** whose exact marginal likelihood requires integrating over the diffusion
    trajectory and is intractable. Following Diffusion-DPO (Wallace et al., 2023,
    arXiv:2311.12908) and VRPO / LLaDA 1.5 (Zhu et al., 2025, arXiv:2505.19223),
    we replace each ``log pi(y | x)`` with a Monte-Carlo **ELBO surrogate**: a
    denoising forward at sampled diffusion timestep(s) yields per-position token
    logits, and the masked summed log-prob of the realized response tokens is a
    one-sample estimate of the ELBO term (see
    ``dimba.training.preference.elbo_sequence_logprob``). Optional
    *antithetic timestep sampling* (VRPO) reduces the variance of this estimate
    and hence of the preference gradient.

The four required log-probs per pair (policy/reference x chosen/rejected) are
plugged into the Bradley-Terry ``dpo_loss`` (or ``ipo_loss``); a reference-free
``simpo_loss`` is also selectable, in which case the reference model is skipped.

NOTE: This script is correct and runnable *in principle* but is intended to be
launched by the user on real hardware/data. Do not run heavy training here.
"""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import random
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Add local src/ to import path (mirrors finetune_sft.py).
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = (SCRIPT_DIR / ".." / ".." / "src").resolve()
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Reuse the SFT script's robust checkpoint/tokenizer/LoRA utilities to avoid
# duplicating the inference logic and to stay consistent with the SFT path.
import finetune_sft as sft  # noqa: E402  (path set above)

from dimba.models.diffusion import DIMBA  # noqa: E402
from dimba.training.preference import (  # noqa: E402
    dpo_loss,
    elbo_sequence_logprob,
    ipo_loss,
    simpo_loss,
)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_preference_row(row: Dict[str, Any]) -> Optional[Tuple[str, str, str]]:
    """Extract a ``(prompt, chosen, rejected)`` triplet from a raw record.

    Accepts a range of common column names used by preference datasets and, when
    no explicit prompt is present, derives it from the longest common prefix of
    the chosen/rejected texts.

    Args:
        row: Raw dataset record.

    Returns:
        ``(prompt, chosen, rejected)`` or ``None`` when chosen/rejected missing.
    """
    p_keys = ("prompt", "input", "instruction", "question", "query")
    c_keys = ("chosen", "preferred", "chosen_response", "accepted", "winner")
    r_keys = ("rejected", "rejected_response", "other_response", "discarded", "loser")

    def pick(keys: Sequence[str]) -> Optional[str]:
        for k in keys:
            if k in row and row[k] is not None:
                return str(row[k])
        return None

    prompt = pick(p_keys)
    chosen = pick(c_keys)
    rejected = pick(r_keys)
    if chosen is None or rejected is None:
        return None
    if prompt is None:
        # Derive a shared prompt prefix from the two responses.
        i = 0
        n = min(len(chosen), len(rejected))
        while i < n and chosen[i] == rejected[i]:
            i += 1
        prompt = chosen[:i]
        chosen, rejected = chosen[i:], rejected[i:]
    return prompt, chosen, rejected


def load_preference_rows(args: argparse.Namespace) -> List[Tuple[str, str, str]]:
    """Load preference triplets from the repo helper, a local file, or HF.

    Tries ``dimba.data.finetuning.load_and_format_finetuning_records`` first
    (handles suggested datasets / formatters), then falls back to JSON/JSONL and
    finally the ``datasets`` library, mirroring ``finetune_grpo.py``.

    Args:
        args: Parsed CLI arguments.

    Returns:
        List of ``(prompt, chosen, rejected)`` triplets.

    Raises:
        ValueError: When no valid preference rows can be parsed.
    """
    raw_rows: List[Dict[str, Any]] = []

    helper = sft.optional_import("dimba.data.finetuning")
    if helper is not None and hasattr(helper, "load_and_format_finetuning_records"):
        try:
            records, _ = helper.load_and_format_finetuning_records(
                source=args.dataset,
                split=args.dataset_split,
                max_examples=(args.max_train_samples if args.max_train_samples > 0 else None),
                strict=False,
            )
            raw_rows = [r for r in records if isinstance(r, dict)]
        except Exception:
            raw_rows = []

    if not raw_rows:
        ds_path = Path(args.dataset)
        if ds_path.exists() and ds_path.suffix.lower() == ".jsonl":
            raw_rows = sft.read_jsonl(ds_path)
        elif ds_path.exists() and ds_path.suffix.lower() == ".json":
            with ds_path.open("r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, list):
                raw_rows = obj
            elif isinstance(obj, dict):
                raw_rows = obj.get(args.dataset_split, obj.get("train", []))
        else:
            try:
                from datasets import load_dataset
            except ImportError as exc:
                raise ImportError(
                    "The 'datasets' package is required for this dataset format."
                ) from exc
            ds = load_dataset(args.dataset, split=args.dataset_split)
            raw_rows = [dict(x) for x in ds]

    out: List[Tuple[str, str, str]] = []
    for row in raw_rows:
        item = normalize_preference_row(row)
        if item is None:
            continue
        prompt, chosen, rejected = item
        if chosen.strip() and rejected.strip():
            out.append((prompt, chosen, rejected))
        if args.max_train_samples > 0 and len(out) >= args.max_train_samples:
            break

    if not out:
        raise ValueError("No valid preference triplets found in dataset.")
    return out


def build_pair_tensors(
    tokenizer: Any,
    prompt: str,
    response: str,
    max_seq_length: int,
    pad_token_id: int,
    ignore_index: int,
) -> Dict[str, torch.Tensor]:
    """Tokenize a (prompt, response) pair into full ids + a response mask.

    Reuses the SFT template machinery so prompt conditioning matches the SFT/GRPO
    forward. Only response positions are marked in ``response_mask`` (used to
    restrict the ELBO log-prob to the completion).

    Args:
        tokenizer: HF or DIMBA tokenizer.
        prompt: Prompt text.
        response: Response text to score.
        max_seq_length: Max tokenized length.
        pad_token_id: Padding id.
        ignore_index: Label ignore index (for parity with SFT labels).

    Returns:
        Dict with ``input_ids``, ``attention_mask``, ``response_mask``, ``labels``.
    """
    full_text, prompt_prefix = sft.parse_template(
        template="{instruction}\n\n{input}\n\n{response}",
        instruction=prompt,
        input_text="",
        response=response,
    )
    input_ids, attention_mask = sft.encode_text(
        tokenizer=tokenizer,
        text=full_text,
        max_length=max_seq_length,
        pad_token_id=pad_token_id,
        pad_to_max_length=True,
    )
    prompt_ids, _ = sft.encode_text(
        tokenizer=tokenizer,
        text=prompt_prefix,
        max_length=max_seq_length,
        pad_token_id=pad_token_id,
        pad_to_max_length=False,
    )
    prompt_len = int(min(prompt_ids.shape[0], max_seq_length))

    response_mask = attention_mask.clone().float()
    response_mask[:prompt_len] = 0.0
    response_mask[attention_mask == 0] = 0.0

    labels = input_ids.clone()
    labels[response_mask == 0] = ignore_index

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "response_mask": response_mask,
        "labels": labels,
    }


class PreferenceTripletDataset(Dataset):
    """Tokenized ``{prompt, chosen, rejected}`` triplets for DPO."""

    def __init__(
        self,
        rows: Sequence[Tuple[str, str, str]],
        tokenizer: Any,
        max_seq_length: int,
        pad_token_id: int,
        ignore_index: int,
    ) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        prompt, chosen, rejected = self.rows[idx]
        chosen_t = build_pair_tensors(
            self.tokenizer, prompt, chosen, self.max_seq_length, self.pad_token_id, self.ignore_index
        )
        rejected_t = build_pair_tensors(
            self.tokenizer, prompt, rejected, self.max_seq_length, self.pad_token_id, self.ignore_index
        )
        return {
            "chosen_input_ids": chosen_t["input_ids"],
            "chosen_response_mask": chosen_t["response_mask"],
            "chosen_labels": chosen_t["labels"],
            "rejected_input_ids": rejected_t["input_ids"],
            "rejected_response_mask": rejected_t["response_mask"],
            "rejected_labels": rejected_t["labels"],
        }


def collate_triplets(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Stack triplet tensors along the batch dimension."""
    out: Dict[str, torch.Tensor] = {}
    for key in batch[0].keys():
        out[key] = torch.stack([item[key] for item in batch], dim=0)
    return out


def policy_logprob(
    model: DIMBA,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    response_mask: torch.Tensor,
    num_mc_samples: int,
    antithetic: bool,
) -> torch.Tensor:
    """ELBO-surrogate summed response log-prob under ``model``.

    Wraps :func:`dimba.training.preference.elbo_sequence_logprob` with DIMBA's
    default diffusion-conditioned forward.

    Args:
        model: Policy or reference DIMBA model.
        input_ids: Full sequence ids ``[batch, seq]``.
        labels: Realized response token ids ``[batch, seq]``.
        response_mask: Response mask ``[batch, seq]``.
        num_mc_samples: Timestep MC samples for the ELBO estimate.
        antithetic: Use antithetic timestep pairing (VRPO).

    Returns:
        Per-sequence ELBO log-prob ``[batch]``.
    """
    safe_labels = labels.clone()
    safe_labels[response_mask == 0] = 0  # Indices ignored by the mask anyway.
    return elbo_sequence_logprob(
        model,
        input_ids=input_ids,
        labels=safe_labels,
        mask=response_mask,
        num_mc_samples=num_mc_samples,
        antithetic=antithetic,
    )


def compute_dpo_batch_loss(
    policy: DIMBA,
    reference: Optional[DIMBA],
    batch: Dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute the selected preference loss for one batch of triplets.

    Args:
        policy: Trainable DIMBA policy.
        reference: Frozen reference DIMBA (``None`` for reference-free SimPO).
        batch: Collated triplet batch.
        args: Parsed CLI arguments (``loss_type``, ``beta``, ``gamma``, etc.).

    Returns:
        Tuple ``(loss, metrics)`` where ``metrics`` holds scalar logging values.
    """
    c_ids = batch["chosen_input_ids"]
    c_mask = batch["chosen_response_mask"]
    c_labels = batch["chosen_labels"]
    r_ids = batch["rejected_input_ids"]
    r_mask = batch["rejected_response_mask"]
    r_labels = batch["rejected_labels"]

    pi_chosen = policy_logprob(policy, c_ids, c_labels, c_mask, args.mc_samples, args.antithetic)
    pi_rejected = policy_logprob(policy, r_ids, r_labels, r_mask, args.mc_samples, args.antithetic)

    if args.loss_type == "simpo":
        chosen_len = c_mask.sum(dim=-1)
        rejected_len = r_mask.sum(dim=-1)
        loss, chosen_reward, rejected_reward = simpo_loss(
            pi_chosen, pi_rejected, chosen_len, rejected_len, beta=args.beta, gamma=args.gamma
        )
    else:
        if reference is None:
            raise RuntimeError("Reference model required for dpo/ipo loss.")
        with torch.no_grad():
            ref_chosen = policy_logprob(
                reference, c_ids, c_labels, c_mask, args.mc_samples, args.antithetic
            )
            ref_rejected = policy_logprob(
                reference, r_ids, r_labels, r_mask, args.mc_samples, args.antithetic
            )
        if args.loss_type == "ipo":
            loss, chosen_reward, rejected_reward = ipo_loss(
                pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta=args.beta
            )
        else:  # standard DPO
            loss, chosen_reward, rejected_reward = dpo_loss(
                pi_chosen,
                pi_rejected,
                ref_chosen,
                ref_rejected,
                beta=args.beta,
                label_smoothing=args.label_smoothing,
            )

    accuracy = (chosen_reward > rejected_reward).float().mean()
    margin = (chosen_reward - rejected_reward).mean()
    metrics = {
        "loss": float(loss.item()),
        "reward_acc": float(accuracy.item()),
        "reward_margin": float(margin.item()),
        "pi_chosen_lp": float(pi_chosen.mean().item()),
        "pi_rejected_lp": float(pi_rejected.mean().item()),
    }
    return loss, metrics


def maybe_apply_lora(model: DIMBA, args: argparse.Namespace) -> Tuple[DIMBA, bool, Optional[ModuleType]]:
    """Apply repo LoRA/Q-LoRA helper when available, else built-in LoRA fallback.

    Reuses ``finetune_sft`` helpers so behavior matches the SFT path.

    Args:
        model: Policy model.
        args: Parsed CLI arguments.

    Returns:
        Tuple ``(model, used_repo_lora, lora_helper_module)``.
    """
    if args.use_qlora:
        model, _, _ = sft.maybe_apply_repo_quantization_helper(model, args)

    lora_targets = sft.parse_target_modules(args.lora_target_modules)
    model, used_repo_lora, lora_module = sft.maybe_apply_repo_lora_helper(
        model=model, args=args, target_modules=lora_targets
    )
    if not used_repo_lora:
        fallback_targets = lora_targets if lora_targets is not None else ["denoiser"]
        sft.apply_builtin_lora(
            model=model,
            target_modules=fallback_targets,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
    return model, used_repo_lora, lora_module


def parse_args() -> argparse.Namespace:
    """CLI arguments (mirrors finetune_sft.py where applicable)."""
    parser = argparse.ArgumentParser(description="DPO fine-tuning for DIMBA (ELBO surrogate)")

    parser.add_argument("--base-checkpoint", type=str, required=True, help="Path to DIMBA checkpoint")
    parser.add_argument("--dataset", type=str, required=True, help="Preference dataset path or HF name")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save outputs")

    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--dataset-split", type=str, default="train")
    parser.add_argument("--max-train-samples", type=int, default=-1)

    parser.add_argument(
        "--loss-type",
        type=str,
        default="dpo",
        choices=["dpo", "ipo", "simpo"],
        help="Preference objective. 'simpo' is reference-free.",
    )
    parser.add_argument("--beta", type=float, default=0.1, help="DPO/IPO KL strength (SimPO: 2.0 typical)")
    parser.add_argument("--gamma", type=float, default=1.0, help="SimPO target reward margin")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="cDPO label smoothing")
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=1,
        help="Monte-Carlo timestep samples for the ELBO log-prob surrogate.",
    )
    parser.add_argument(
        "--antithetic",
        action="store_true",
        help="Use VRPO antithetic timestep sampling (requires even --mc-samples).",
    )

    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--use-qlora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", type=str, default=None)

    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--ignore-index", type=int, default=-100)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)

    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")

    return parser.parse_args()


def main() -> None:
    """Main DPO entrypoint."""
    args = parse_args()
    if args.use_qlora:
        args.use_lora = True
    if args.antithetic and args.mc_samples % 2 != 0:
        raise ValueError("--antithetic requires an even --mc-samples.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = sft.choose_device(args.device)

    print("=" * 80)
    print(f"DIMBA DPO ({args.loss_type.upper()}, ELBO surrogate)")
    print("=" * 80)
    print(f"Base checkpoint: {args.base_checkpoint}")
    print(f"Dataset: {args.dataset} (split={args.dataset_split})")
    print(f"Loss: {args.loss_type} beta={args.beta} mc_samples={args.mc_samples} antithetic={args.antithetic}")
    print(f"Device: {device}")

    policy, load_info = sft.load_dimba_checkpoint(args.base_checkpoint, map_location="cpu")
    print(f"Loaded policy with vocab_size={load_info['vocab_size']}")

    tokenizer, tokenizer_vocab_size = sft.load_tokenizer(args, vocab_size_hint=policy.vocab_size)
    pad_token_id = sft.get_pad_token_id(tokenizer)

    # Reference model: a frozen copy of the *base* policy (DPO/IPO). SimPO is
    # reference-free, so we skip the (expensive) reference forward entirely.
    reference: Optional[DIMBA] = None
    if args.loss_type in {"dpo", "ipo"}:
        reference = copy.deepcopy(policy).to(device)
        reference.eval()
        for p in reference.parameters():
            p.requires_grad = False

    if args.use_lora:
        policy, used_repo_lora, _ = maybe_apply_lora(policy, args)
        print(f"LoRA enabled (repo_helper={used_repo_lora}).")
    else:
        for p in policy.parameters():
            p.requires_grad = True

    policy.to(device)
    policy.train()

    trainable, total = sft.count_parameters(policy)
    if trainable == 0:
        raise RuntimeError("No trainable parameters found.")
    print(f"Trainable params: {trainable:,} / {total:,}")

    rows = load_preference_rows(args)
    dataset = PreferenceTripletDataset(
        rows=rows,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        pad_token_id=pad_token_id,
        ignore_index=args.ignore_index,
    )
    print(f"Preference triplets: {len(dataset):,}")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_triplets,
        pin_memory=(device.type == "cuda"),
    )

    optimizer = AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    stop_training = False

    for epoch in range(max(1, args.num_epochs)):
        if stop_training:
            break
        iterator = tqdm(dataloader, desc=f"Epoch {epoch + 1}", leave=False)
        for batch_idx, batch in enumerate(iterator):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            with nullcontext():
                loss, metrics = compute_dpo_batch_loss(policy, reference, batch, args)
                loss = loss / args.grad_accumulation_steps
            loss.backward()

            is_update_step = ((batch_idx + 1) % args.grad_accumulation_steps == 0) or (
                (batch_idx + 1) == len(dataloader)
            )
            if is_update_step:
                if args.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), args.gradient_clip_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % max(1, args.log_every) == 0:
                    print(
                        f"step={global_step} loss={metrics['loss']:.6f} "
                        f"reward_acc={metrics['reward_acc']:.3f} "
                        f"margin={metrics['reward_margin']:.4f} "
                        f"pi_c={metrics['pi_chosen_lp']:.2f} pi_r={metrics['pi_rejected_lp']:.2f}"
                    )
                if args.max_steps > 0 and global_step >= args.max_steps:
                    stop_training = True
                    break

    final_ckpt_path = output_dir / "dpo_model.pt"
    torch.save(
        {
            "state_dict": policy.state_dict(),
            "global_step": global_step,
            "args": vars(args),
            "vocab_size": policy.vocab_size,
            "tokenizer_vocab_size": tokenizer_vocab_size,
            "model_config": sft.filter_kwargs_for_callable(
                DIMBA.__init__,
                {
                    "d_model": policy.d_model,
                    "d_prompt": policy.d_prompt,
                    "num_diffusion_steps": policy.num_diffusion_steps,
                    "latent_diffusion": policy.latent_diffusion,
                    "d_latent": getattr(policy, "d_latent", None),
                    "use_weight_tying": policy.use_weight_tying,
                    "use_vae_latent": policy.use_vae_latent,
                },
            ),
        },
        final_ckpt_path,
    )
    print(f"Saved DPO model checkpoint: {final_ckpt_path}")

    if args.use_lora:
        lora_state = sft.extract_lora_state_dict(policy)
        adapter_dir = output_dir / "lora_adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": lora_state}, adapter_dir / "adapter_model.pt")
        print(f"Saved LoRA adapter weights: {adapter_dir / 'adapter_model.pt'}")

    tokenizer_path = sft.save_tokenizer(tokenizer, output_dir)
    if tokenizer_path is not None:
        print(f"Saved tokenizer: {tokenizer_path}")

    print("=" * 80)
    print("DPO complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
