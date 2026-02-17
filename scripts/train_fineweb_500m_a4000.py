#!/usr/bin/env python3
"""Train DIMBA (~500M) on FineWeb using an RTX A4000 16GB profile.

Usage:
    # Train with default config
    python scripts/train_fineweb_500m_a4000.py

    # Train with custom config
    python scripts/train_fineweb_500m_a4000.py --config configs/my_config.yaml

    # Auto-upload to HuggingFace after training
    python scripts/train_fineweb_500m_a4000.py --repo-id username/dimba-500m --upload-token $HF_TOKEN
"""

import argparse
import os
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# Add src to path
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = (SCRIPT_DIR / ".." / "src").resolve()
sys.path.insert(0, str(SRC_DIR))

from dimba.data import HuggingFaceDataset, HuggingFaceIterableDataset, collate_fn
from dimba.tokenizers import BPETokenizer
from dimba.training import DIMBALightningModule


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def upload_artifacts_to_hf(save_dir: str, repo_id: str, token: str, private: bool = False) -> None:
    """Upload training artifacts to a Hugging Face model repo."""
    try:
        from huggingface_hub import HfApi, upload_folder
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for upload. Install with: pip install huggingface_hub"
        ) from exc

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    upload_folder(
        repo_id=repo_id,
        folder_path=save_dir,
        repo_type="model",
        token=token,
        commit_message="Upload DIMBA 500M A4000 FineWeb checkpoint and tokenizer",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DIMBA 500M profile on FineWeb (A4000 16GB)")
    parser.add_argument("--config", default="configs/fineweb_500m_a4000.yaml", help="YAML config path")
    parser.add_argument("--hf-token", default=None, help="Optional HF token for dataset access")
    parser.add_argument("--repo-id", default=None, help="Optional HF repo id for auto-upload after training")
    parser.add_argument("--upload-private", action="store_true", help="Create private HF model repo on auto-upload")
    parser.add_argument("--upload-token", default=None, help="HF token for upload (defaults to HF_TOKEN env var)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    print("=" * 64)
    print("DIMBA FineWeb training profile: RTX A4000 16GB / ~500M params")
    print("=" * 64)

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {props.total_memory / 1e9:.1f} GB")
    else:
        print("WARNING: CUDA not available; this profile is intended for GPU use.")

    tokenizer = BPETokenizer(vocab_size=cfg["tokenizer"]["vocab_size"])

    data_cfg = cfg["data"]
    train_split = data_cfg.get("train_split", "train")
    val_split = data_cfg.get("val_split", "validation")
    val_fallback_split = data_cfg.get("val_fallback_split", "train")

    streaming = data_cfg.get("streaming", False)
    dataset_cls = HuggingFaceIterableDataset if streaming else HuggingFaceDataset

    train_dataset = dataset_cls(
        dataset_name=data_cfg["dataset_name"],
        dataset_config=data_cfg.get("dataset_config"),
        split=train_split,
        tokenizer=tokenizer,
        max_length=data_cfg["max_length"],
    )

    try:
        val_dataset = dataset_cls(
            dataset_name=data_cfg["dataset_name"],
            dataset_config=data_cfg.get("dataset_config"),
            split=val_split,
            tokenizer=tokenizer,
            max_length=data_cfg["max_length"],
        )
    except ValueError as exc:
        # FineWeb and some other datasets expose only a train split.
        if val_split != val_fallback_split:
            print(
                f"WARNING: Could not load validation split '{val_split}' ({exc}). "
                f"Falling back to split '{val_fallback_split}' for validation."
            )
            val_dataset = dataset_cls(
                dataset_name=data_cfg["dataset_name"],
                dataset_config=data_cfg.get("dataset_config"),
                split=val_fallback_split,
                tokenizer=tokenizer,
                max_length=data_cfg["max_length"],
            )
        else:
            raise

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=not streaming,
        num_workers=data_cfg["num_workers"],
        collate_fn=collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        collate_fn=collate_fn,
    )

    train_cfg = cfg["training"]
    lightning_module = DIMBALightningModule(
        vocab_size=cfg["tokenizer"]["vocab_size"],
        model_config=cfg["model"],
        learning_rate=float(train_cfg["learning_rate"]),
        warmup_steps=int(train_cfg["warmup_steps"]),
        ema_decay=float(train_cfg["ema_decay"]),
        use_ema=bool(train_cfg["use_ema"]),
        ema_device=str(train_cfg.get("ema_device", "cpu")),
        ema_update_interval=int(train_cfg.get("ema_update_interval", 1)),
    )

    total_params = sum(p.numel() for p in lightning_module.model.parameters())
    print(f"Model parameters: {total_params:,} (~{total_params / 1e6:.1f}M)")

    ckpt_cfg = cfg["checkpoint"]
    save_dir = ckpt_cfg["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    tokenizer.save(os.path.join(save_dir, "tokenizer.json"))
    with open(os.path.join(save_dir, "train_config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,
        filename="dimba-500m-a4000-{step:07d}-{val/loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=int(ckpt_cfg.get("save_top_k", 3)),
        save_last=True,
    )

    trainer = pl.Trainer(
        max_steps=int(train_cfg["max_steps"]),
        devices=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        accumulate_grad_batches=int(train_cfg.get("accumulate_grad_batches", 1)),
        gradient_clip_val=float(train_cfg.get("gradient_clip", 1.0)),
        callbacks=[checkpoint_callback],
        logger=TensorBoardLogger("./logs", name="dimba_fineweb_500m_a4000"),
        log_every_n_steps=int(train_cfg.get("log_interval", 50)),
        val_check_interval=int(train_cfg.get("val_interval", 500)),
        benchmark=True,
    )

    trainer.fit(lightning_module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    print("Training complete.")
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Artifacts saved under: {save_dir}")

    if args.repo_id:
        upload_token = args.upload_token or os.getenv("HF_TOKEN")
        if not upload_token:
            raise ValueError("--repo-id was provided but no upload token found. Use --upload-token or set HF_TOKEN.")

        print(f"Uploading artifacts to https://huggingface.co/{args.repo_id} ...")
        upload_artifacts_to_hf(
            save_dir=save_dir,
            repo_id=args.repo_id,
            token=upload_token,
            private=args.upload_private,
        )
        print(f"Upload complete: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
