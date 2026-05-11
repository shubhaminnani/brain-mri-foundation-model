"""
Lightning module for few-shot binary / multiclass classification.
Tasks 1 (Infarct, AUROC), 5 (Polymicrogyria, AUROC).

Architecture: ViTWrapper → mean-pool CLS (multi-seq) → Dropout → MLP → logits
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics.classification import AUROC, BinaryAUROC


class _MLPHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, in_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim // 2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class LitClassification(L.LightningModule):
    """
    Args:
        backbone     : ViTWrapper instance (already loaded)
        embed_dim    : backbone embed_dim (768 / 1024 / 1152)
        num_classes  : 2 for binary tasks (Tasks 1, 5)
        freeze_backbone: freeze ViT weights (recommended for very few-shot)
        lr / weight_decay: optimiser hyper-params
    """

    def __init__(
        self,
        backbone: nn.Module,
        embed_dim: int,
        num_classes: int = 2,
        freeze_backbone: bool = True,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        label_smoothing: float = 0.05,
        class_weights: list[float] | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"])
        self.backbone = backbone

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        self.head = _MLPHead(embed_dim, num_classes)

        weight_t = torch.tensor(class_weights) if class_weights else None
        self.loss_fn = nn.CrossEntropyLoss(
            weight=weight_t, label_smoothing=label_smoothing
        )

        if num_classes == 2:
            self.val_auroc = BinaryAUROC()
        else:
            self.val_auroc = AUROC(task="multiclass", num_classes=num_classes,
                                   average="macro")
        self._val_preds, self._val_targets = [], []

    # ------------------------------------------------------------------
    def _encode(self, x):
        if x.shape[1] == 1:
            return self.backbone.encode(x)["cls"]
        return self.backbone.encode_multi_sequence(x)["cls"]

    def forward(self, x):
        return self.head(self._encode(x))

    # ------------------------------------------------------------------
    def training_step(self, batch, _):
        x, y = batch["image"], batch["label"].long()
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train/loss", loss, prog_bar=True, batch_size=x.shape[0])
        return loss

    def validation_step(self, batch, _):
        x, y = batch["image"], batch["label"].long()
        logits = self(x)
        loss = self.loss_fn(logits, y)
        probs = F.softmax(logits, dim=1)
        self._val_preds.append(probs.detach().cpu())
        self._val_targets.append(y.detach().cpu())
        self.log("val/loss", loss, prog_bar=True, batch_size=x.shape[0])

    def on_validation_epoch_end(self):
        if not self._val_preds:
            return
        preds   = torch.cat(self._val_preds)
        targets = torch.cat(self._val_targets)
        if preds.shape[1] == 2:
            auroc = self.val_auroc(preds[:, 1], targets)
        else:
            auroc = self.val_auroc(preds, targets)
        self.log("val/auroc", auroc, prog_bar=True)
        self._val_preds.clear()
        self._val_targets.clear()

    # ------------------------------------------------------------------
    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)
        return {"optimizer": opt, "lr_scheduler": sched, "monitor": "val/auroc"}
