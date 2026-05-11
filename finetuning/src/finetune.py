"""
Unified finetuning entry point for Tasks 1-5.

Usage examples:
  # Task 1 — Infarct classification (v3 backbone, freeze encoder)
  python finetune.py --task 1 --model v3 --checkpoint /path/to/last.ckpt \
      --data_dir /data/preprocessed/task1 --output_dir /models/task1_v3

  # Task 2 — Meningioma segmentation (v2 backbone, full finetune)
  python finetune.py --task 2 --model v2 --checkpoint /path/to/last.ckpt \
      --data_dir /data/preprocessed/task2 --output_dir /models/task2_v2 \
      --freeze_encoder

  # Task 3 — Brain age regression (v1 backbone)
  python finetune.py --task 3 --model v1 --checkpoint /path/to/last.ckpt \
      --data_dir /data/preprocessed/task3 --output_dir /models/task3_v1
"""

import argparse
import sys
from pathlib import Path

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).parent))

import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

from backbone.vit_wrapper import ViTWrapper, CONFIGS
from data.task_configs import get_task
from data.dataset import make_loaders
from models.seg_decoder import build_seg_model
from models.lit_cls import LitClassification
from models.lit_reg import LitRegression
from models.lit_seg import LitSegmentation


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task",          type=int, required=True, choices=[1, 2, 3, 4, 5])
    p.add_argument("--model",         type=str, required=True, choices=["v1", "v2", "v3"])
    p.add_argument("--checkpoint",    type=str, required=True, help="Pretraining last.ckpt")
    p.add_argument("--data_dir",      type=str, required=True)
    p.add_argument("--output_dir",    type=str, required=True)
    p.add_argument("--freeze_encoder",action="store_true",
                   help="Freeze ViT encoder weights (recommended for very few-shot)")
    p.add_argument("--epochs",        type=int, default=200)
    p.add_argument("--batch_size",    type=int, default=2)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--weight_decay",  type=float, default=1e-5)
    p.add_argument("--num_workers",   type=int, default=4)
    p.add_argument("--precision",     type=str, default="bf16-mixed")
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--no_wandb",      action="store_true")
    return p.parse_args()


def build_model(args, task_cfg):
    mv   = args.model
    cfg  = CONFIGS[mv]
    D    = cfg["embed_dim"]
    task = task_cfg.task_type

    if task == "segmentation":
        seg_model = build_seg_model(mv, task_cfg.num_classes, args.checkpoint)
        return LitSegmentation(
            seg_model=seg_model,
            img_size=cfg["img_size"],
            num_classes=task_cfg.num_classes,
            freeze_encoder=args.freeze_encoder,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    backbone = ViTWrapper(mv, args.checkpoint, freeze=False)

    if task == "classification":
        return LitClassification(
            backbone=backbone,
            embed_dim=D,
            num_classes=task_cfg.num_classes,
            freeze_backbone=args.freeze_encoder,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    if task == "regression":
        return LitRegression(
            backbone=backbone,
            embed_dim=D,
            freeze_backbone=args.freeze_encoder,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    raise ValueError(f"Unsupported task type: {task}")


def main():
    args     = parse_args()
    task_cfg = get_task(args.task)
    L.seed_everything(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f" FOMO26 Finetuning | Task {args.task}: {task_cfg.task_name}")
    print(f" Backbone: {args.model} | Freeze encoder: {args.freeze_encoder}")
    print(f"{'='*60}\n")

    tr_dl, vl_dl = make_loaders(
        args.data_dir, task_cfg, args.model,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )

    model = build_model(args, task_cfg)

    callbacks = [
        ModelCheckpoint(
            dirpath=out_dir,
            filename=f"task{args.task}_{args.model}_{{epoch:03d}}",
            monitor=f"val/{task_cfg.metric_primary}",
            mode="max" if task_cfg.metric_primary != "mae" else "min",
            save_top_k=3,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        EarlyStopping(
            monitor=f"val/{task_cfg.metric_primary}",
            patience=30,
            mode="max" if task_cfg.metric_primary != "mae" else "min",
        ),
    ]

    logger = False
    if not args.no_wandb:
        from lightning.pytorch.loggers import WandbLogger
        import wandb
        wandb.init(project="fomo26-finetune",
                   name=f"task{args.task}_{args.model}",
                   dir=str(out_dir))
        logger = WandbLogger(project="fomo26-finetune",
                             name=f"task{args.task}_{args.model}",
                             save_dir=str(out_dir), log_model=False)

    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=1,
        precision=args.precision,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=5,
        num_sanity_val_steps=0,
        enable_progress_bar=True,
    )

    trainer.fit(model, tr_dl, vl_dl, ckpt_path="last")

    if not args.no_wandb:
        import wandb
        wandb.finish()

    print(f"\nDone. Best checkpoint: {callbacks[0].best_model_path}")


if __name__ == "__main__":
    main()
