#!/usr/bin/env python3
"""Distillation script: distill a HuggingFace Transformer teacher into DIMBA.

This script performs cross-architecture knowledge distillation, transferring
knowledge from a pretrained HuggingFace Transformer into DIMBA's bidirectional
Mamba-2 diffusion model. It supports three distillation stages:

- Stage 1: Align Mamba-2 mixing matrices to teacher attention maps.
- Stage 2: Align student hidden states to teacher layer outputs.
- Stage 3: Standard diffusion objective plus optional soft-label KD loss.

Usage:
    python distill.py --config distill_config.yaml [--output-dir ./checkpoints/distill]

Example YAML (``distillation`` section):
    distillation:
      teacher_model: gpt2
      teacher_type: causal
      mode: convert
      block_ffn: true
      principled_init: false
      layer_map_mode: uniform
      num_student_layers: null
      share_vocab: false
      kd_weight: 1.0
      kd_temp: 2.0
      device: cpu
      stages:
        - name: stage2
          steps: 100
          lr: 1.0e-4
"""

import logging
import os
import sys
import argparse
from typing import Any, Dict, Optional

import torch
import yaml

# ---------------------------------------------------------------------------
# Ensure src/ is on the Python path regardless of cwd.
# ---------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, "..", "src")
sys.path.insert(0, os.path.abspath(src_dir))

from dimba import DIMBA  # noqa: E402
from dimba.data import DummyDataset, collate_fn  # noqa: E402
from dimba.tokenizers import BPETokenizer, SimpleCharacterTokenizer  # noqa: E402
from dimba.distillation import (  # noqa: E402
    TeacherWrapper,
    build_student_from_teacher,
    principled_init_from_teacher,
    DistillationTrainer,
    DistillationConfig,
    LayerMap,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Parsed configuration dictionary.
    """
    with open(config_path, "r") as fh:
        return yaml.safe_load(fh)


def create_tokenizer(config: Dict[str, Any]):
    """Instantiate a tokenizer from the ``tokenizer`` config section.

    Falls back to a BPETokenizer with vocab_size=10000 when the section is absent.

    Args:
        config: Top-level configuration dictionary.

    Returns:
        A tokenizer instance (BPETokenizer or SimpleCharacterTokenizer).
    """
    tok_cfg = config.get("tokenizer", {"type": "bpe", "vocab_size": 10000})
    if tok_cfg.get("type", "bpe") == "bpe":
        return BPETokenizer(vocab_size=tok_cfg.get("vocab_size", 10000))
    return SimpleCharacterTokenizer(vocab_size=tok_cfg.get("vocab_size", 256))


def create_dataloader(config: Dict[str, Any], vocab_size: int) -> torch.utils.data.DataLoader:
    """Build a DataLoader from the ``data`` config section.

    When ``data.type`` is ``'dummy'`` (or absent), a :class:`DummyDataset` is
    used so the script works without a real dataset.  For HuggingFace datasets,
    :class:`~dimba.data.HuggingFaceDataset` is imported lazily to avoid a hard
    dependency when running in dummy mode.

    Args:
        config: Top-level configuration dictionary.
        vocab_size: Vocabulary size, used to construct the dummy dataset.

    Returns:
        A configured :class:`torch.utils.data.DataLoader`.
    """
    data_cfg = config.get("data", {})
    dataset_type = data_cfg.get("type", "dummy")
    batch_size = data_cfg.get("batch_size", 8)
    num_workers = data_cfg.get("num_workers", 0)
    seq_length = data_cfg.get("max_length", 128)
    num_examples = data_cfg.get("num_examples", 256)

    if dataset_type == "huggingface":
        from dimba.data import HuggingFaceDataset  # noqa: PLC0415

        dataset = HuggingFaceDataset(
            dataset_name=data_cfg.get("dataset_name", "wikitext"),
            dataset_config=data_cfg.get("dataset_config", "wikitext-2-raw-v1"),
            split=data_cfg.get("split", "train"),
            tokenizer=None,  # raw ids expected from HF dataset
            max_length=seq_length,
            streaming=data_cfg.get("streaming", False),
        )
    else:
        # Dummy dataset: random token ids, no network required.
        dataset = DummyDataset(
            size=num_examples,
            vocab_size=vocab_size,
            seq_length=seq_length,
        )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    return loader


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for the distillation script."""
    parser = argparse.ArgumentParser(
        description="Distill a HuggingFace Transformer teacher into a DIMBA student."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="distill_config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./checkpoints/distill",
        help="Directory in which to save the distilled checkpoint.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="dimba_distilled",
        help="Basename for the saved checkpoint file (without extension).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override the device setting from config (e.g. 'cpu', 'cuda').",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    )

    print("=" * 70)
    print("DIMBA Cross-Architecture Knowledge Distillation")
    print("=" * 70)

    # ---- load config -------------------------------------------------------
    print(f"\nLoading configuration from: {args.config}")
    config = load_config(args.config)

    distil_raw = config.get("distillation", {})
    cfg: DistillationConfig = DistillationConfig.from_dict(distil_raw)

    # CLI device override
    if args.device is not None:
        cfg.device = args.device

    print(f"  Teacher model : {cfg.teacher_model}")
    print(f"  Teacher type  : {cfg.teacher_type}")
    print(f"  Mode          : {cfg.mode}")
    print(f"  Device        : {cfg.device}")
    print(f"  Stages        : {[s.get('name', '?') for s in cfg.stages]}")

    # ---- build teacher -----------------------------------------------------
    print("\nLoading teacher model...")
    teacher = TeacherWrapper(
        model_id_or_path=cfg.teacher_model,
        teacher_type=cfg.teacher_type,
        device=cfg.device,
        dtype=torch.float32,
    )
    print(f"  Layers    : {teacher.num_layers}")
    print(f"  Heads     : {teacher.num_heads}")
    print(f"  d_model   : {teacher.d_model}")
    print(f"  vocab_size: {teacher.vocab_size}")
    print(f"  FFN type  : {teacher.ffn_type}  (hidden={teacher.ffn_hidden})")

    # ---- build student -----------------------------------------------------
    layer_map: Optional[LayerMap] = None

    if cfg.mode == "convert":
        print("\nBuilding DIMBA student from teacher (weight surgery)...")
        # Pass any model-level overrides from the config's 'model' section,
        # skipping keys that are already set by build_student_from_teacher.
        model_overrides = {
            k: v
            for k, v in config.get("model", {}).items()
            if k
            not in (
                "vocab_size",
                "d_model",
                "num_denoiser_layers",
                "block_ffn",
                "ffn_type",
                "ffn_hidden",
                "latent_diffusion",
                "padding_idx",
                "use_simple_mamba",
            )
        }
        model, layer_map = build_student_from_teacher(
            teacher,
            num_student_layers=cfg.num_student_layers,
            block_ffn=cfg.block_ffn,
            inherit_embeddings=True,
            inherit_ffn=True,
            inherit_head=True,
            layer_map_mode=cfg.layer_map_mode,
            **model_overrides,
        )
        print(f"  Student layers : {len(model.denoiser.blocks)}")
    else:
        # Build a plain DIMBA from the 'model' config section.
        print("\nBuilding DIMBA student from config (no surgery)...")
        model_cfg = config.get("model", {})
        if not model_cfg:
            raise ValueError(
                "Config must contain a 'model' section when mode != 'convert'."
            )
        model = DIMBA(**model_cfg)
        # Build a layer map so the trainer has one if stages need it.
        n_student = len(model.denoiser.blocks)
        from dimba.distillation import LayerMap as _LayerMap  # noqa: PLC0415

        layer_map = _LayerMap(
            teacher.num_layers,
            n_student,
            mode=cfg.layer_map_mode,
        )
        print(f"  Student layers : {n_student}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # ---- principled init ---------------------------------------------------
    if cfg.principled_init and layer_map is not None:
        print("\nApplying principled initialization from teacher attention weights...")
        principled_init_from_teacher(model, teacher, layer_map)
        print("  Principled init complete.")

    # ---- data --------------------------------------------------------------
    print("\nBuilding data loader...")
    loader = create_dataloader(config, vocab_size=teacher.vocab_size)
    print(f"  Batch size   : {loader.batch_size}")
    print(f"  Dataset size : {len(loader.dataset)}")

    # ---- distillation trainer ----------------------------------------------
    print("\nCreating DistillationTrainer...")
    trainer = DistillationTrainer(model, teacher, cfg, layer_map=layer_map)

    # ---- run stages --------------------------------------------------------
    if not cfg.stages:
        print("\nNo stages defined in config; skipping distillation training.")
    else:
        print(f"\nRunning {len(cfg.stages)} distillation stage(s)...")
        print("=" * 70)
        model = trainer.run(loader)
        print("=" * 70)
        print("Distillation complete.")

    # ---- save checkpoint ---------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, f"{args.name}.pt")
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": model.config,
    }
    torch.save(checkpoint, ckpt_path)
    print(f"\nCheckpoint saved to: {ckpt_path}")

    # Also save a copy of the run config alongside the checkpoint.
    config_save_path = os.path.join(args.output_dir, f"{args.name}_config.yaml")
    with open(config_save_path, "w") as fh:
        yaml.dump(config, fh)
    print(f"Config saved to   : {config_save_path}")

    print("\n" + "=" * 70)
    print("Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
