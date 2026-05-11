"""
Lightning module for linear probing (Tasks 6 & 7).

The backbone is completely frozen. A single Linear layer is trained on top
of the mean-pooled CLS token. This is the only allowed setup for Tasks 6-7
(no fine-tuning of the backbone).

Task 6: OvR AUROC + F1 across all classes.
Task 7: Same metrics but evaluated per demographic/acquisition group.

Embeddings are pre-extracted (offline) into a tensor cache for speed.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics.classification import MultilabelAUROC, MultilabelF1Score


class LinearProbeHead(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class LitLinearProbe(L.LightningModule):
    """
    Args:
        embed_dim   : backbone output dimension
        num_classes : number of OvR classes
        lr          : learning rate (only for the linear head)
    """

    def __init__(self, embed_dim: int, num_classes: int, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.head = LinearProbeHead(embed_dim, num_classes)
        self.loss_fn  = nn.BCEWithLogitsLoss()
        self.val_auroc = MultilabelAUROC(num_labels=num_classes, average="macro")
        self.val_f1    = MultilabelF1Score(num_labels=num_classes, average="macro")
        self._val_logits, self._val_targets = [], []

    # ------------------------------------------------------------------
    # Expects pre-extracted embeddings: batch = {"embed": (B,D), "label": (B,C)}
    def forward(self, embed):
        return self.head(embed)

    def training_step(self, batch, _):
        embed, y = batch["embed"], batch["label"].float()
        logits = self(embed)
        loss = self.loss_fn(logits, y)
        self.log("train/loss", loss, prog_bar=True, batch_size=embed.shape[0])
        return loss

    def validation_step(self, batch, _):
        embed, y = batch["embed"], batch["label"].float()
        logits = self(embed)
        loss = self.loss_fn(logits, y)
        self.log("val/loss", loss, batch_size=embed.shape[0])
        self._val_logits.append(logits.detach().cpu())
        self._val_targets.append(y.detach().cpu().long())

    def on_validation_epoch_end(self):
        if not self._val_logits:
            return
        logits  = torch.cat(self._val_logits)
        targets = torch.cat(self._val_targets)
        probs   = torch.sigmoid(logits)
        auroc = self.val_auroc(probs, targets)
        f1    = self.val_f1(probs, targets)
        self.log("val/auroc", auroc, prog_bar=True)
        self.log("val/f1",    f1,    prog_bar=True)
        self._val_logits.clear()
        self._val_targets.clear()

    # ------------------------------------------------------------------
    def configure_optimizers(self):
        return torch.optim.LBFGS(
            self.head.parameters(), lr=self.hparams.lr, max_iter=20
        )


# ---------------------------------------------------------------------------
# Offline embedding extraction helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_embeddings(
    backbone: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: str = "cuda",
    save_path: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Runs the frozen backbone over all samples and collects CLS embeddings.

    Returns:
        embeddings : (N, D) float32 numpy array
        labels     : (N, ...) numpy array
    """
    backbone.eval().to(device)
    all_emb, all_lbl = [], []

    for batch in loader:
        x = batch["image"].to(device)
        if x.shape[1] == 1:
            emb = backbone.encode(x)["cls"]
        else:
            emb = backbone.encode_multi_sequence(x)["cls"]
        all_emb.append(emb.cpu().float().numpy())
        all_lbl.append(batch["label"].numpy())

    embeddings = np.concatenate(all_emb, axis=0)
    labels     = np.concatenate(all_lbl, axis=0)

    if save_path:
        np.savez(save_path, embeddings=embeddings, labels=labels)
        print(f"Embeddings saved → {save_path}  shape={embeddings.shape}")

    return embeddings, labels


class EmbeddingDataset(torch.utils.data.Dataset):
    """Thin dataset that wraps a pre-extracted .npz embedding file."""

    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        self.embeddings = torch.from_numpy(data["embeddings"]).float()
        self.labels     = torch.from_numpy(data["labels"])

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return {"embed": self.embeddings[idx], "label": self.labels[idx]}
