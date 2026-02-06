#!/usr/bin/env python3
"""Train DIMBA (~500M) on FineWeb using an RTX A4000 16GB profile."""

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

from dimba.data import HuggingFaceDataset, collate_fn
from dimba.tokenizers import BPETokenizer
from dimba.training import DIMBALightningModule


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DIMBA 500M profile on FineWeb (A4000 16GB)")
    parser.add_argument("--config", default="configs/fineweb_500m_a4000.yaml", help="YAML config path")
    parser.add_argument("--hf-token", default=None, help="Optional HF token for dataset access")
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
    train_dataset = HuggingFaceDataset(
        dataset_name=data_cfg["dataset_name"],
        split="train",
        tokenizer=tokenizer,
        max_length=data_cfg["max_length"],
        streaming=data_cfg.get("streaming", False),
    )
    val_dataset = HuggingFaceDataset(
        dataset_name=data_cfg["dataset_name"],
        split="validation",
        tokenizer=tokenizer,
        max_length=data_cfg["max_length"],
        streaming=data_cfg.get("streaming", False),
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=not data_cfg.get("streaming", False),
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


if __name__ == "__main__":
    main()
