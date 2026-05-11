"""
Lightning module for brain age regression.
Task 3: predict chronological age from T1w+T2w MRI.
Metrics: MAE + Pearson correlation coefficient.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import lightning as L
from torchmetrics.regression import MeanAbsoluteError, PearsonCorrCoef


class LitRegression(L.LightningModule):
    """
    Args:
        backbone     : ViTWrapper (multi-seq encode if C>1)
        embed_dim    : backbone embed_dim
        freeze_backbone: freeze ViT weights
        lr / weight_decay / dropout: training hyper-params
    """

    def __init__(
        self,
        backbone: nn.Module,
        embed_dim: int,
        freeze_backbone: bool = False,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"])
        self.backbone = backbone

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
        )

        self.loss_fn = nn.MSELoss()
        self.val_mae  = MeanAbsoluteError()
        self.val_corr = PearsonCorrCoef()
        self._val_preds, self._val_targets = [], []

    # ------------------------------------------------------------------
    def _encode(self, x):
        if x.shape[1] == 1:
            return self.backbone.encode(x)["cls"]
        return self.backbone.encode_multi_sequence(x)["cls"]

    def forward(self, x):
        return self.head(self._encode(x)).squeeze(-1)   # (B,)

    # ------------------------------------------------------------------
    def training_step(self, batch, _):
        x, y = batch["image"], batch["label"].float()
        pred = self(x)
        loss = self.loss_fn(pred, y)
        self.log("train/mse", loss, prog_bar=True, batch_size=x.shape[0])
        return loss

    def validation_step(self, batch, _):
        x, y = batch["image"], batch["label"].float()
        pred = self(x)
        loss = self.loss_fn(pred, y)
        self.log("val/mse", loss, batch_size=x.shape[0])
        self._val_preds.append(pred.detach().cpu())
        self._val_targets.append(y.detach().cpu())

    def on_validation_epoch_end(self):
        if not self._val_preds:
            return
        preds   = torch.cat(self._val_preds)
        targets = torch.cat(self._val_targets)
        mae  = self.val_mae(preds, targets)
        corr = self.val_corr(preds, targets)
        self.log("val/mae",  mae,  prog_bar=True)
        self.log("val/corr", corr, prog_bar=True)
        self._val_preds.clear()
        self._val_targets.clear()

    # ------------------------------------------------------------------
    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=50, T_mult=2
        )
        return {"optimizer": opt, "lr_scheduler": sched, "monitor": "val/mae"}
