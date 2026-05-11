"""
Linear probing entry point for Tasks 6 & 7.

Step 1 (extract): Run frozen backbone over all data and save embeddings.
Step 2 (train):   Train a single linear layer on the cached embeddings.
Step 3 (eval):    Compute OvR AUROC + F1; Task 7 also breaks down by group.

Usage:
  # Step 1 — extract embeddings (run once per backbone)
  python linear_probe.py extract \
      --model v3 --checkpoint /path/to/last.ckpt \
      --data_dir /data/preprocessed/task6 \
      --embed_dir /embeddings/task6_v3

  # Step 2+3 — train probe and evaluate
  python linear_probe.py train \
      --task 6 --embed_dir /embeddings/task6_v3 \
      --output_dir /models/task6_v3

  # Task 7 — add group CSV
  python linear_probe.py train \
      --task 7 --embed_dir /embeddings/task6_v3 \
      --group_csv /data/task7_groups.csv \
      --output_dir /models/task7_v3
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping

from backbone.vit_wrapper import ViTWrapper, CONFIGS
from data.dataset import FOMODataset, EmbeddingDataset
from data.transforms import get_val_transforms
from data.task_configs import get_task
from models.lit_linear_probe import LitLinearProbe, extract_embeddings
from metrics import task_scores


# ---------------------------------------------------------------------------
# Step 1: extract
# ---------------------------------------------------------------------------

def cmd_extract(args):
    mv  = args.model
    cfg = CONFIGS[mv]
    out = Path(args.embed_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[extract] Loading {mv} backbone from {args.checkpoint}")
    backbone = ViTWrapper(mv, args.checkpoint, freeze=True)
    backbone.eval()

    tfm = get_val_transforms(["image"], None, cfg["img_size"], is_seg=False)
    ds  = FOMODataset(args.data_dir, task_type="classification", transform=tfm)
    dl  = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embs, lbls = extract_embeddings(backbone, dl, device=device,
                                     save_path=str(out / "embeddings.npz"))
    print(f"[extract] Saved {embs.shape} embeddings to {out}/embeddings.npz")


# ---------------------------------------------------------------------------
# Step 2+3: train & eval
# ---------------------------------------------------------------------------

def cmd_train(args):
    task_cfg = get_task(args.task)
    embed_dir = Path(args.embed_dir)
    out_dir   = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = EmbeddingDataset(str(embed_dir / "embeddings.npz"),
                          group_csv=getattr(args, "group_csv", None))

    embed_dim  = ds.embeddings.shape[1]
    n_classes  = int(ds.labels.max().item()) + 1
    print(f"[train] embed_dim={embed_dim}, n_classes={n_classes}, N={len(ds)}")

    # 80/20 split
    n_val  = max(1, int(0.2 * len(ds)))
    n_tr   = len(ds) - n_val
    g      = torch.Generator().manual_seed(args.seed)
    tr_ds, vl_ds = random_split(ds, [n_tr, n_val], generator=g)

    tr_dl = DataLoader(tr_ds, batch_size=256, shuffle=True)
    vl_dl = DataLoader(vl_ds, batch_size=256, shuffle=False)

    model = LitLinearProbe(embed_dim=embed_dim, num_classes=n_classes, lr=args.lr)

    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="cpu",   # linear probe fits on CPU easily
        callbacks=[EarlyStopping("val/auroc", patience=20, mode="max")],
        logger=False,
        enable_progress_bar=True,
    )
    trainer.fit(model, tr_dl, vl_dl)

    # Final evaluation on val set
    model.eval()
    all_logits, all_labels, all_groups = [], [], []
    for batch in vl_dl:
        with torch.no_grad():
            logits = model(batch["embed"])
        all_logits.append(logits.cpu().numpy())
        all_labels.append(batch["label"].numpy())
        if "group" in batch:
            all_groups.append(np.array(batch["group"]))

    preds = {
        "y_true": np.concatenate(all_labels),
        "y_prob": torch.sigmoid(torch.from_numpy(np.concatenate(all_logits))).numpy(),
    }
    if all_groups:
        preds["groups"] = np.concatenate(all_groups)

    scores = task_scores(args.task, preds)
    print(f"\n[eval] Scores: {json.dumps(scores, indent=2)}")
    with open(out_dir / "scores.json", "w") as f:
        json.dump(scores, f, indent=2)

    torch.save(model.state_dict(), out_dir / "linear_probe.pt")
    print(f"[train] Saved probe → {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")

    # extract subcommand
    ex = sub.add_parser("extract")
    ex.add_argument("--model",      required=True, choices=["v1", "v2", "v3"])
    ex.add_argument("--checkpoint", required=True)
    ex.add_argument("--data_dir",   required=True)
    ex.add_argument("--embed_dir",  required=True)

    # train subcommand
    tr = sub.add_parser("train")
    tr.add_argument("--task",       type=int, required=True, choices=[6, 7])
    tr.add_argument("--embed_dir",  required=True)
    tr.add_argument("--output_dir", required=True)
    tr.add_argument("--group_csv",  default=None)
    tr.add_argument("--epochs",     type=int, default=500)
    tr.add_argument("--lr",         type=float, default=1e-3)
    tr.add_argument("--seed",       type=int, default=42)

    args = p.parse_args()
    if args.cmd == "extract":
        cmd_extract(args)
    elif args.cmd == "train":
        cmd_train(args)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
