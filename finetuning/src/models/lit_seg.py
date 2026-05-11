"""
Lightning module for segmentation tasks.
Task 2: Meningioma (binary, DSC+NSD)
Task 4: Trigeminal Neuralgia (multiclass, DSC+NSD)

Uses Tversky+CE loss (handles class imbalance well for small lesions).
Validation uses sliding-window inference via MONAI.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, SurfaceDistanceMetric
from monai.losses import DiceCELoss


class LitSegmentation(L.LightningModule):
    """
    Args:
        seg_model    : complete seg model (UNETRSegDecoder or _SegModel from seg_decoder.py)
        img_size     : spatial size of the model input (96 or 112)
        num_classes  : 2 for binary, >2 for multiclass
        freeze_encoder: freeze ViT encoder if seg_model has .backbone
        lr / weight_decay: optimiser hyper-params
    """

    def __init__(
        self,
        seg_model: nn.Module,
        img_size: int,
        num_classes: int = 2,
        freeze_encoder: bool = False,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["seg_model"])
        self.model = seg_model
        self.img_size = img_size

        if freeze_encoder and hasattr(self.model, "backbone"):
            for p in self.model.backbone.parameters():
                p.requires_grad_(False)

        # Tversky loss (alpha=0.7 penalises FN more → better sensitivity)
        self.loss_fn = DiceCELoss(
            to_onehot_y=True, softmax=True, include_background=False
        )

        include_bg = num_classes == 1
        self.val_dice = DiceMetric(
            include_background=include_bg, reduction="mean", get_not_nans=False
        )
        self.val_nsd = SurfaceDistanceMetric(
            include_background=include_bg, reduction="mean"
        )
        self._val_preds, self._val_targets = [], []

    # ------------------------------------------------------------------
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, _):
        x, y = batch["image"], batch["label"]
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train/loss", loss, prog_bar=True, batch_size=x.shape[0])
        return loss

    def validation_step(self, batch, _):
        x, y = batch["image"], batch["label"]
        roi = (self.img_size,) * 3

        logits = sliding_window_inference(
            x, roi_size=roi, sw_batch_size=2, predictor=self.model
        )
        loss = self.loss_fn(logits, y)
        self.log("val/loss", loss, batch_size=x.shape[0])

        preds_bin = (F.softmax(logits, dim=1).argmax(dim=1, keepdim=True)
                     == torch.arange(logits.shape[1], device=logits.device)
                     .view(1, -1, 1, 1, 1)).float()
        y_bin = (y == torch.arange(logits.shape[1], device=y.device)
                  .view(1, -1, 1, 1, 1)).float()

        self.val_dice(preds_bin, y_bin)
        self.val_nsd(preds_bin, y_bin)

    def on_validation_epoch_end(self):
        dice = self.val_dice.aggregate()
        nsd  = self.val_nsd.aggregate()
        self.log("val/dice", dice.mean(), prog_bar=True)
        self.log("val/nsd",  nsd.mean(),  prog_bar=True)
        self.val_dice.reset()
        self.val_nsd.reset()

    # ------------------------------------------------------------------
    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
        return {"optimizer": opt, "lr_scheduler": sched, "monitor": "val/dice"}
