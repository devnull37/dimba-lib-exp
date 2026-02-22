#!/usr/bin/env python3
"""Interactive fine-tuning wizard for SFT and GRPO.

Flow:
1) Mode selection: SFT vs GRPO
2) Base model checkpoint path validation
3) Dataset selection
4) Hardware detection and VRAM budget selection
5) Hyperparameter setup with VRAM-aware defaults
6) Command build, confirmation, and launch
"""

from __future__ import annotations

import math
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


DATASET_MENU = {
    "1": {
        "label": "ultrachat_200k",
        "hf_dataset": "HuggingFaceH4/ultrachat_200k",
        "default_split": "train_sft",
    },
    "2": {
        "label": "Code-Feedback",
        "hf_dataset_by_mode": {
            "sft": "m-a-p/CodeFeedback-Filtered-Instruction",
            "grpo": "m-a-p/CodeFeedback-Filtered-Preference",
        },
        "default_split": "train",
    },
    "3": {
        "label": "OpenHermes-2.5",
        "hf_dataset": "teknium/OpenHermes-2.5",
        "default_split": "train",
    },
    "4": {
        "label": "feedback-collection",
        "hf_dataset": "argilla/feedback-collection",
        "default_split": "train",
    },
    "5": {
        "label": "custom HF dataset",
    },
    "6": {
        "label": "local file",
    },
}


VRAM_BUDGETS = {
    "1": 4,
    "2": 8,
    "3": 16,
    "4": 24,
    "5": 40,
}


MODE_TRAINING_SCRIPT_CANDIDATES = {
    "sft": (
        "scripts/finetuning/finetune_sft.py",
        "scripts/finetuning/finetune_grpo.py",
        "scripts/finetuning/train.py",
        "scripts/train.py",
    ),
    "grpo": (
        "scripts/finetuning/finetune_grpo.py",
        "scripts/finetuning/finetune_sft.py",
        "scripts/finetuning/train.py",
        "scripts/train.py",
    ),
}


def print_header(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def prompt_text(prompt: str, default: str | None = None, allow_empty: bool = False) -> str:
    while True:
        suffix = f" [{default}]" if default is not None and default != "" else ""
        raw = input(f"{prompt}{suffix}: ").strip()
        if raw:
            return raw
        if default is not None:
            return default
        if allow_empty:
            return ""
        print("Input is required.")


def prompt_int(prompt: str, default: int, min_value: int = 0) -> int:
    while True:
        raw = prompt_text(prompt, str(default))
        try:
            value = int(raw)
        except ValueError:
            print("Please enter a valid integer.")
            continue
        if value < min_value:
            print(f"Please enter a value >= {min_value}.")
            continue
        return value


def prompt_float(prompt: str, default: float, min_value: float = 0.0) -> float:
    while True:
        raw = prompt_text(prompt, str(default))
        try:
            value = float(raw)
        except ValueError:
            print("Please enter a valid number.")
            continue
        if value < min_value:
            print(f"Please enter a value >= {min_value}.")
            continue
        return value


def prompt_menu_choice(title: str, options: dict[str, str], default: str) -> str:
    print_header(title)
    for key, label in options.items():
        print(f"[{key}] {label}")
    while True:
        choice = prompt_text("Select option", default=default).strip()
        if choice in options:
            return choice
        print(f"Invalid option: {choice}. Choose one of: {', '.join(options.keys())}")


def resolve_path(raw_path: str, project_root: Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.exists():
        alt = (project_root / raw_path).resolve()
        if alt.exists():
            return alt
    return path


def select_mode() -> str:
    choice = prompt_menu_choice(
        "Step 1/6 - Training Mode",
        {
            "1": "SFT (Supervised Fine-Tuning)",
            "2": "GRPO",
        },
        default="1",
    )
    return "sft" if choice == "1" else "grpo"


def prompt_base_model_path(project_root: Path) -> str:
    print_header("Step 2/6 - Base Model Checkpoint")
    while True:
        raw = prompt_text("Enter base model checkpoint path")
        path = resolve_path(raw, project_root)
        if path.exists():
            if path.is_file() or path.is_dir():
                return str(path)
            print(f"Path exists but is not file/dir: {path}")
            continue
        print(f"Checkpoint path not found: {path}")


def default_hf_dataset_for_mode(preset: dict[str, Any], mode: str) -> str:
    by_mode = preset.get("hf_dataset_by_mode")
    if isinstance(by_mode, dict):
        candidate = by_mode.get(mode)
        if candidate:
            return str(candidate)
    return str(preset.get("hf_dataset", ""))


def select_dataset(project_root: Path, mode: str) -> dict[str, Any]:
    print_header("Step 3/6 - Dataset Selection")
    for key in ("1", "2", "3", "4", "5", "6"):
        print(f"[{key}] {DATASET_MENU[key]['label']}")

    while True:
        choice = prompt_text("Select dataset", default="1")
        if choice not in DATASET_MENU:
            print("Invalid selection.")
            continue

        if choice in {"1", "2", "3", "4"}:
            preset = DATASET_MENU[choice]
            dataset_default = default_hf_dataset_for_mode(preset, mode)
            dataset_name = prompt_text(
                "HF dataset id",
                default=dataset_default,
            )
            dataset_config = prompt_text(
                "Dataset config (optional)",
                default="",
                allow_empty=True,
            )
            split = prompt_text("Dataset split", default=str(preset["default_split"]))
            return {
                "kind": "hf",
                "label": preset["label"],
                "dataset_name": dataset_name,
                "dataset_config": dataset_config,
                "split": split,
            }

        if choice == "5":
            dataset_name = prompt_text("HF dataset id (owner/name)")
            dataset_config = prompt_text(
                "Dataset config (optional)",
                default="",
                allow_empty=True,
            )
            split = prompt_text("Dataset split", default="train")
            return {
                "kind": "hf",
                "label": "custom HF dataset",
                "dataset_name": dataset_name,
                "dataset_config": dataset_config,
                "split": split,
            }

        local_raw = prompt_text("Local dataset file path")
        local_path = resolve_path(local_raw, project_root)
        if not local_path.exists():
            print(f"File not found: {local_path}")
            continue
        if not local_path.is_file():
            print(f"Path is not a file: {local_path}")
            continue
        return {
            "kind": "local",
            "label": "local file",
            "path": str(local_path),
            "format": local_path.suffix.lower().lstrip("."),
        }


def detect_hardware() -> dict[str, Any]:
    detected: dict[str, Any] = {"gpus": [], "cpu_only": True}
    try:
        import torch
    except Exception:
        return detected

    if torch.cuda.is_available():
        detected["cpu_only"] = False
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            memory_gb = props.total_memory / (1024**3)
            detected["gpus"].append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "memory_gb": memory_gb,
                    "backend": "cuda",
                }
            )
    return detected


def default_vram_bucket(detected: dict[str, Any]) -> str:
    if not detected["gpus"]:
        return "1"
    max_mem = max(gpu["memory_gb"] for gpu in detected["gpus"])
    if max_mem >= 40:
        return "5"
    if max_mem >= 24:
        return "4"
    if max_mem >= 16:
        return "3"
    if max_mem >= 8:
        return "2"
    return "1"


def suggest_strategy(vram_budget_gb: int, has_gpu: bool) -> tuple[str, str]:
    if not has_gpu:
        return "qlora", "CPU-only environment detected. Q-LoRA is the safest default."
    if vram_budget_gb < 12:
        return "qlora", "VRAM is below 12GB. Q-LoRA is recommended."
    if vram_budget_gb >= 40:
        return "full", "40GB+ VRAM can often support full fine-tuning."
    return "lora", "LoRA is a strong default for this VRAM range."


def select_vram_and_strategy(detected: dict[str, Any]) -> tuple[int, str]:
    print_header("Step 4/6 - Hardware and VRAM")
    if detected["gpus"]:
        print("Detected GPUs:")
        for gpu in detected["gpus"]:
            print(
                f"- GPU {gpu['index']}: {gpu['name']} "
                f"({gpu['memory_gb']:.1f} GB, {gpu['backend']})"
            )
    else:
        print("No CUDA GPU detected. Training will run on CPU unless your trainer handles other backends.")

    default_bucket = default_vram_bucket(detected)
    print("\nSelect VRAM budget:")
    print("[1] 4GB")
    print("[2] 8GB")
    print("[3] 16GB")
    print("[4] 24GB")
    print("[5] 40GB+")
    while True:
        bucket_choice = prompt_text("VRAM budget", default=default_bucket)
        if bucket_choice in VRAM_BUDGETS:
            break
        print("Invalid VRAM option.")

    vram_budget = VRAM_BUDGETS[bucket_choice]
    recommended_strategy, reason = suggest_strategy(vram_budget, has_gpu=bool(detected["gpus"]))
    strategy_default = {"lora": "1", "qlora": "2", "full": "3"}[recommended_strategy]

    print(f"\nRecommendation: {reason}")
    print("Select tuning method:")
    print("[1] LoRA")
    print("[2] Q-LoRA")
    print("[3] Full fine-tuning")

    while True:
        strategy_choice = prompt_text("Method", default=strategy_default)
        if strategy_choice in {"1", "2", "3"}:
            break
        print("Invalid method.")

    strategy = {"1": "lora", "2": "qlora", "3": "full"}[strategy_choice]
    if vram_budget < 12:
        print("Tip: VRAM < 12GB detected. `--use-qlora` is strongly recommended.")
    return vram_budget, strategy


def default_lora_rank(vram_budget_gb: int) -> int:
    if vram_budget_gb >= 24:
        return 16
    if vram_budget_gb >= 16:
        return 8
    return 4


def default_batch_size(vram_budget_gb: int, strategy: str) -> int:
    if vram_budget_gb >= 40:
        base = 16
    elif vram_budget_gb >= 24:
        base = 8
    elif vram_budget_gb >= 16:
        base = 4
    elif vram_budget_gb >= 8:
        base = 2
    else:
        base = 1

    if strategy == "full":
        base = max(1, base // 2)
    elif strategy == "qlora":
        base = min(32, base * 2)
    return base


def default_grad_accum(batch_size: int, mode: str, strategy: str) -> int:
    if strategy == "full":
        target_effective_batch = 16 if mode == "grpo" else 32
    else:
        target_effective_batch = 32 if mode == "grpo" else 64
    return max(1, int(math.ceil(target_effective_batch / max(1, batch_size))))


def default_max_steps(vram_budget_gb: int, mode: str) -> int:
    if mode == "grpo":
        if vram_budget_gb >= 24:
            return 1000
        if vram_budget_gb >= 16:
            return 800
        return 500
    if vram_budget_gb >= 24:
        return 3000
    if vram_budget_gb >= 16:
        return 2000
    return 1000


def select_hyperparameters(mode: str, strategy: str, vram_budget: int) -> dict[str, Any]:
    print_header("Step 5/6 - Hyperparameters")

    lr_default = 2e-5 if strategy == "full" else 2e-4
    lora_r_default = default_lora_rank(vram_budget)
    batch_size_default = default_batch_size(vram_budget, strategy)
    grad_accum_default = default_grad_accum(batch_size_default, mode, strategy)
    epochs_default = 1 if mode == "grpo" else 3
    max_steps_default = default_max_steps(vram_budget, mode)
    output_dir_default = f"checkpoints/finetune_{mode}_{strategy}"

    params: dict[str, Any] = {}
    params["batch_size"] = prompt_int("Per-device batch size", batch_size_default, min_value=1)
    params["gradient_accumulation_steps"] = prompt_int(
        "Gradient accumulation steps",
        grad_accum_default,
        min_value=1,
    )
    params["epochs"] = prompt_int("Epochs", epochs_default, min_value=1)
    params["max_steps"] = prompt_int(
        "Max steps (0 to disable)",
        max_steps_default,
        min_value=0,
    )
    params["learning_rate"] = prompt_float("Learning rate", lr_default, min_value=0.0)
    params["output_dir"] = prompt_text("Output directory", output_dir_default)

    if strategy in {"lora", "qlora"}:
        params["lora_r"] = prompt_int("LoRA rank (r)", lora_r_default, min_value=1)
        params["lora_alpha"] = prompt_int(
            "LoRA alpha",
            params["lora_r"] * 2,
            min_value=1,
        )
    else:
        params["lora_r"] = None
        params["lora_alpha"] = None

    if vram_budget < 12 and strategy != "qlora":
        print("Suggestion: switch to Q-LoRA or add `--use-qlora` for lower memory usage.")

    return params


def choose_training_script(project_root: Path, mode: str) -> Path:
    print_header("Step 6/6 - Training Script")
    available: list[Path] = []
    seen: set[Path] = set()
    candidates = MODE_TRAINING_SCRIPT_CANDIDATES.get(mode, MODE_TRAINING_SCRIPT_CANDIDATES["sft"])
    for rel in candidates:
        candidate = (project_root / rel).resolve()
        if candidate.exists() and candidate.is_file() and candidate not in seen:
            available.append(candidate)
            seen.add(candidate)

    if available:
        print("Detected training scripts:")
        for idx, path in enumerate(available, start=1):
            rel = path.relative_to(project_root)
            print(f"[{idx}] {rel}")
        print("[c] Custom path")

        default_choice = "1"
        while True:
            choice = prompt_text("Select script", default=default_choice).lower()
            if choice == "c":
                break
            if choice.isdigit() and 1 <= int(choice) <= len(available):
                return available[int(choice) - 1]
            print("Invalid selection.")

    while True:
        raw = prompt_text("Enter training script path (.py)")
        candidate = resolve_path(raw, project_root)
        if candidate.exists() and candidate.is_file() and candidate.suffix == ".py":
            return candidate
        print(f"Invalid script path: {candidate}")


def extract_supported_flags(script_path: Path) -> set[str] | None:
    pattern = re.compile(r"--([a-zA-Z0-9][a-zA-Z0-9_-]*)")

    help_text = ""
    try:
        completed = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        help_text = (completed.stdout or "") + "\n" + (completed.stderr or "")
    except Exception:
        help_text = ""

    if help_text.strip():
        matches = set(pattern.findall(help_text))
        if matches:
            return matches

    try:
        source_text = script_path.read_text(encoding="utf-8")
    except Exception:
        return None
    matches = set(pattern.findall(source_text))
    return matches or None


def pick_flag(aliases: list[str], supported_flags: set[str] | None) -> str | None:
    if supported_flags is None:
        return aliases[0]
    for alias in aliases:
        if alias in supported_flags:
            return alias
    return None


def add_option(
    cmd: list[str],
    supported_flags: set[str] | None,
    aliases: list[str],
    value: Any,
    omitted: list[str],
) -> None:
    if value is None:
        return
    if isinstance(value, str) and value == "":
        return
    flag = pick_flag(aliases, supported_flags)
    if flag is None:
        omitted.append(f"--{aliases[0]} {value}")
        return
    cmd.extend([f"--{flag}", str(value)])


def add_switch(
    cmd: list[str],
    supported_flags: set[str] | None,
    aliases: list[str],
    enabled: bool,
    omitted: list[str],
) -> None:
    if not enabled:
        return
    flag = pick_flag(aliases, supported_flags)
    if flag is None:
        omitted.append(f"--{aliases[0]}")
        return
    cmd.append(f"--{flag}")


def build_training_command(
    script_path: Path,
    mode: str,
    base_model_path: str,
    dataset: dict[str, Any],
    strategy: str,
    hyperparams: dict[str, Any],
    hardware: dict[str, Any],
) -> tuple[list[str], list[str]]:
    supported_flags = extract_supported_flags(script_path)
    omitted: list[str] = []
    cmd: list[str] = [sys.executable, str(script_path)]

    mode_flag = pick_flag(["mode", "train-mode", "training-mode"], supported_flags)
    if mode_flag is not None:
        cmd.extend([f"--{mode_flag}", mode])
    add_option(
        cmd,
        supported_flags,
        [
            "base-model",
            "base-model-path",
            "base-checkpoint",
            "model-path",
            "model",
            "checkpoint",
            "checkpoint-path",
        ],
        base_model_path,
        omitted,
    )

    add_option(cmd, supported_flags, ["dataset-type", "data-type"], dataset["kind"], omitted)
    if dataset["kind"] == "hf":
        add_option(
            cmd,
            supported_flags,
            ["dataset", "dataset-name", "dataset_name", "hf-dataset"],
            dataset["dataset_name"],
            omitted,
        )
        add_option(
            cmd,
            supported_flags,
            ["dataset-config", "dataset_config", "hf-dataset-config"],
            dataset.get("dataset_config"),
            omitted,
        )
        add_option(
            cmd,
            supported_flags,
            ["dataset-split", "dataset_split", "split"],
            dataset.get("split"),
            omitted,
        )
    else:
        add_option(
            cmd,
            supported_flags,
            ["dataset-path", "data-path", "train-file", "dataset-file"],
            dataset["path"],
            omitted,
        )
        add_option(
            cmd,
            supported_flags,
            ["dataset-format", "data-format"],
            dataset.get("format"),
            omitted,
        )

    if strategy == "qlora":
        add_switch(cmd, supported_flags, ["use-qlora", "qlora"], True, omitted)
        add_switch(cmd, supported_flags, ["use-lora", "lora", "enable-lora"], True, omitted)
    elif strategy == "lora":
        add_switch(cmd, supported_flags, ["use-lora", "lora", "enable-lora"], True, omitted)
    else:
        add_switch(cmd, supported_flags, ["full-finetune", "full-ft"], True, omitted)

    add_option(
        cmd,
        supported_flags,
        ["batch-size", "per-device-train-batch-size", "train-batch-size", "micro-batch-size"],
        hyperparams["batch_size"],
        omitted,
    )
    add_option(
        cmd,
        supported_flags,
        [
            "gradient-accumulation-steps",
            "grad-accumulation-steps",
            "grad-accum-steps",
            "accumulation-steps",
        ],
        hyperparams["gradient_accumulation_steps"],
        omitted,
    )
    add_option(
        cmd,
        supported_flags,
        ["epochs", "num-train-epochs", "num-epochs"],
        hyperparams["epochs"],
        omitted,
    )
    if hyperparams["max_steps"] > 0:
        add_option(
            cmd,
            supported_flags,
            ["max-steps", "max_steps"],
            hyperparams["max_steps"],
            omitted,
        )
    add_option(
        cmd,
        supported_flags,
        ["learning-rate", "lr"],
        hyperparams["learning_rate"],
        omitted,
    )
    add_option(
        cmd,
        supported_flags,
        ["output-dir", "output_dir"],
        hyperparams["output_dir"],
        omitted,
    )

    if strategy in {"lora", "qlora"}:
        add_option(cmd, supported_flags, ["lora-r", "lora_r", "rank"], hyperparams["lora_r"], omitted)
        add_option(
            cmd,
            supported_flags,
            ["lora-alpha", "lora_alpha"],
            hyperparams["lora_alpha"],
            omitted,
        )

    if hardware["gpus"]:
        add_option(
            cmd,
            supported_flags,
            ["gpus", "num-gpus", "num_gpus"],
            len(hardware["gpus"]),
            omitted,
        )
    else:
        add_option(
            cmd,
            supported_flags,
            ["gpus", "num-gpus", "num_gpus"],
            0,
            omitted,
        )
        add_option(
            cmd,
            supported_flags,
            ["device", "accelerator"],
            "cpu",
            omitted,
        )

    return cmd, omitted


def format_command(command: list[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(command)
    return shlex.join(command)


def run_command(command: list[str], project_root: Path) -> int:
    print_header("Launching Training")
    print(format_command(command))
    print()
    process = subprocess.run(command, cwd=str(project_root), check=False)
    return process.returncode


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    print_header("DIMBA Fine-Tuning Interactive Wizard")

    mode = select_mode()
    base_model_path = prompt_base_model_path(project_root)
    dataset = select_dataset(project_root, mode=mode)
    hardware = detect_hardware()
    vram_budget, strategy = select_vram_and_strategy(hardware)
    hyperparams = select_hyperparameters(mode, strategy, vram_budget)
    training_script = choose_training_script(project_root, mode=mode)

    command, omitted = build_training_command(
        script_path=training_script,
        mode=mode,
        base_model_path=base_model_path,
        dataset=dataset,
        strategy=strategy,
        hyperparams=hyperparams,
        hardware=hardware,
    )

    print_header("Command Preview")
    print(f"Mode: {mode.upper()}")
    print(f"Base model: {base_model_path}")
    print(f"Dataset: {dataset['label']}")
    print(f"Method: {strategy.upper()}")
    print(f"VRAM budget: {vram_budget}GB{'+' if vram_budget == 40 else ''}")
    print(f"Training script: {training_script}")
    print("\nCommand:")
    print(format_command(command))

    if omitted:
        print("\nNote: these options were not added (flag not detected in target script):")
        for item in omitted:
            print(f"- {item}")

    confirm = prompt_text("\nStart training now? (y/N)", default="n").lower()
    if confirm not in {"y", "yes"}:
        print("Cancelled.")
        return

    return_code = run_command(command, project_root=project_root)
    if return_code == 0:
        print("Training finished successfully.")
    else:
        print(f"Training exited with non-zero status: {return_code}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
