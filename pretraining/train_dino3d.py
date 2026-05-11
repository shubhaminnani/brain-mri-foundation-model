"""
3D DINOv3 pretraining on FOMO300K using BrainIAC's ViT-B backbone.

Losses imported from the cloned dinov3 repo (dinov3.loss):
  - DINO:  CLS-token self-distillation via Sinkhorn-Knopp centering
  - iBOT:  masked patch-token prediction (local objective)
  - KoLeo: entropy regularisation to spread representations

Architecture:
  - Backbone: MONAI ViT-B, 96³ input, 16³ patches → 216 patch tokens
  - Student + EMA teacher (no FSDP needed for ViT-B scale)
  - DINO head + iBOT head (separate prototypes)
"""

import argparse
import copy
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from monai.networks.nets import ViT
from torch.utils.data import DataLoader, Dataset
import wandb

# DINOv3 loss modules (cloned repo must be on PYTHONPATH)
from dinov3.loss import DINOLoss, iBOTPatchLoss, KoLeoLoss
from dinov3.layers.dino_head import DINOHead


# ── Backbone wrapper ───────────────────────────────────────────────────────────

class BrainIACViT(nn.Module):
    """MONAI ViT-B wrapped to return DINO-compatible dict."""

    def __init__(self):
        super().__init__()
        self.vit = ViT(
            in_channels=1,
            img_size=(96, 96, 96),
            patch_size=(16, 16, 16),
            hidden_size=768,
            mlp_dim=3072,
            num_layers=12,
            num_heads=12,
            save_attn=False,
        )
        self.norm = nn.LayerNorm(768)

    def forward(self, x, masks=None):
        # x: (B, 1, 96, 96, 96)
        # MONAI 3D ViT returns (B, 216, 768) — no dedicated CLS token
        out, _ = self.vit(x)           # (B, 216, 768)
        out = self.norm(out)
        return {
            "x_norm_clstoken":    out.mean(dim=1),  # (B, 768) mean-pooled global rep
            "x_norm_patchtokens": out,               # (B, 216, 768) all patch tokens
        }


# ── Lightning module ───────────────────────────────────────────────────────────

class DINO3DModel(L.LightningModule):

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        self.args = args

        # Student
        self.student_backbone = BrainIACViT()
        self.student_dino_head = DINOHead(
            in_dim=768,
            out_dim=args.dino_prototypes,
            hidden_dim=2048,
            bottleneck_dim=256,
            nlayers=3,
        )
        self.student_ibot_head = DINOHead(
            in_dim=768,
            out_dim=args.ibot_prototypes,
            hidden_dim=2048,
            bottleneck_dim=256,
            nlayers=3,
        )

        # Teacher (EMA copy, no grad)
        self.teacher_backbone = copy.deepcopy(self.student_backbone)
        self.teacher_dino_head = copy.deepcopy(self.student_dino_head)
        self.teacher_ibot_head = copy.deepcopy(self.student_ibot_head)
        for p in (*self.teacher_backbone.parameters(),
                  *self.teacher_dino_head.parameters(),
                  *self.teacher_ibot_head.parameters()):
            p.requires_grad_(False)

        # Losses
        self.dino_loss = DINOLoss(args.dino_prototypes)
        self.ibot_loss = iBOTPatchLoss(args.ibot_prototypes)
        self.koleo_loss = KoLeoLoss()
        # Initialize EMA centers to zero (they start as NaN by default)
        self.dino_loss.init_weights()
        self.ibot_loss.init_weights()

        self._step = 0  # global iteration counter for EMA momentum schedule

    # ── warm-start from BrainIAC SimCLR checkpoint ───────────────────────────
    def load_brainiac_weights(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        backbone_state = {k[9:]: v for k, v in state.items()
                          if k.startswith("backbone.")}
        missing, unexpected = self.student_backbone.vit.load_state_dict(
            backbone_state, strict=False)
        print(f"BrainIAC init — missing: {len(missing)}, unexpected: {len(unexpected)}")
        self.teacher_backbone.vit.load_state_dict(
            self.student_backbone.vit.state_dict())
        print("BrainIAC weights loaded into student + teacher.")

    # ── masking helpers ───────────────────────────────────────────────────────
    def _make_patch_mask(self, B: int, n_patches: int = 216) -> torch.Tensor:
        """Random block mask over patch tokens for iBOT. Returns bool (B, n_patches)."""
        ratio = self.args.ibot_mask_ratio
        n_mask = int(ratio * n_patches)
        mask = torch.zeros(B, n_patches, dtype=torch.bool)
        for i in range(B):
            idx = torch.randperm(n_patches)[:n_mask]
            mask[i, idx] = True
        return mask

    # ── EMA teacher update ────────────────────────────────────────────────────
    def _ema_momentum(self) -> float:
        """Cosine schedule from momentum_teacher → 1.0 over all training steps."""
        total = self.trainer.estimated_stepping_batches
        m_start = self.args.teacher_momentum
        return 1 - (1 - m_start) * (math.cos(math.pi * self._step / total) + 1) / 2

    @torch.no_grad()
    def _update_teacher(self):
        m = self._ema_momentum()
        for ps, pt in zip(self.student_backbone.parameters(),
                          self.teacher_backbone.parameters()):
            pt.data.mul_(m).add_(ps.data, alpha=1 - m)
        for ps, pt in zip(self.student_dino_head.parameters(),
                          self.teacher_dino_head.parameters()):
            pt.data.mul_(m).add_(ps.data, alpha=1 - m)
        for ps, pt in zip(self.student_ibot_head.parameters(),
                          self.teacher_ibot_head.parameters()):
            pt.data.mul_(m).add_(ps.data, alpha=1 - m)

    # ── teacher temperature schedule ──────────────────────────────────────────
    def _teacher_temp(self) -> float:
        epoch = self.current_epoch
        if epoch < self.args.warmup_temp_epochs:
            return self.args.warmup_temp + \
                   (self.args.teacher_temp - self.args.warmup_temp) * \
                   epoch / self.args.warmup_temp_epochs
        return self.args.teacher_temp

    # ── forward ───────────────────────────────────────────────────────────────
    def training_step(self, batch, batch_idx):
        global_crops = batch["global_crops"]   # (2, B, 1, 96, 96, 96)
        local_crops  = batch["local_crops"]    # (n_local, B, 1, 48, 48, 48)

        n_global, B = global_crops.shape[:2]
        n_local      = local_crops.shape[0]
        N_patches    = 216  # 6³ patch tokens per 96³ volume

        # Mask for iBOT on first global crop
        mask = self._make_patch_mask(B, N_patches).to(self.device)  # (B, 216) bool

        teacher_temp = self._teacher_temp()

        # ── Teacher forward (no grad) ─────────────────────────────────────────
        with torch.no_grad():
            t_outs  = [self.teacher_backbone(global_crops[i]) for i in range(n_global)]
            t_cls   = torch.stack([o["x_norm_clstoken"]    for o in t_outs])  # (2,B,768)
            t_patch = torch.stack([o["x_norm_patchtokens"] for o in t_outs])  # (2,B,216,768)

            # DINO: CLS tokens → head → Sinkhorn-Knopp soft assignment
            t_cls_head = self.teacher_dino_head(t_cls.flatten(0, 1))          # (2B, K_dino)
            t_dino_soft = self.dino_loss.sinkhorn_knopp_teacher(
                t_cls_head, teacher_temp=teacher_temp)                         # (2B, K_dino)
            t_dino_soft = t_dino_soft.unflatten(0, (n_global, B))             # (2, B, K_dino)

            # iBOT: all patch tokens of first global crop → iBOT head → softmax+centering
            t_ibot_logits = self.teacher_ibot_head(
                t_patch[0].reshape(B * N_patches, 768))                       # (B*216, K_ibot)
            t_ibot_logits = t_ibot_logits.view(B, N_patches, -1)              # (B, 216, K_ibot)

            # Apply pending center update, then compute soft targets
            self.ibot_loss.apply_center_update()
            t_ibot_soft = F.softmax(
                (t_ibot_logits - self.ibot_loss.center) / teacher_temp, dim=-1)  # (B,216,K_ibot)

            # Schedule async EMA center update for next step
            self.ibot_loss.reduce_center_update(t_ibot_logits)

        # ── Student forward ───────────────────────────────────────────────────
        # First global crop: zero masked patches in input before ViT
        s0 = self.student_backbone(self._apply_patch_mask(global_crops[0], mask))
        s1 = self.student_backbone(global_crops[1])

        s_global_cls = [s0["x_norm_clstoken"], s1["x_norm_clstoken"]]         # each (B,768)

        # Local crops are 48³ but ViT expects 96³ (fixed position embeddings).
        # Upsample 2× so each local crop is a zoomed-in 96³ view — the local-global
        # multi-scale signal is preserved; the ViT just sees a higher-magnification patch.
        s_local_cls = [
            self.student_backbone(
                F.interpolate(local_crops[i], size=(96, 96, 96),
                              mode="trilinear", align_corners=False)
            )["x_norm_clstoken"]
            for i in range(n_local)
        ]

        # DINO head on all CLS tokens
        s_cls_all  = torch.stack(s_global_cls + s_local_cls)                  # (2+n,B,768)
        s_dino_head = self.student_dino_head(s_cls_all.flatten(0, 1))         # ((2+n)*B,K)
        s_dino_head = s_dino_head.unflatten(0, (n_global + n_local, B))       # (2+n,B,K)

        # iBOT head on ALL patch positions of masked first global crop
        s_ibot_logits = self.student_ibot_head(
            s0["x_norm_patchtokens"].reshape(B * N_patches, 768))             # (B*216,K_ibot)
        s_ibot_logits = s_ibot_logits.view(B, N_patches, -1)                  # (B,216,K_ibot)

        # ── DINO loss ─────────────────────────────────────────────────────────
        dino_loss = self.dino_loss(
            student_logits=s_dino_head,   # (2+n_local, B, K_dino)
            teacher_probs=t_dino_soft,    # (2,        B, K_dino)
        )

        # ── iBOT loss (cross-entropy only on masked positions) ────────────────
        ibot_loss = self.ibot_loss.forward(
            student_patch_tokens=s_ibot_logits,   # (B, 216, K_ibot) raw logits
            teacher_patch_tokens=t_ibot_soft,     # (B, 216, K_ibot) soft probs
            student_masks_flat=mask,              # (B, 216) bool
        )

        # ── KoLeo loss ────────────────────────────────────────────────────────
        s_global_cls_stack = torch.stack(s_global_cls)                        # (2,B,768)
        koleo_loss = sum(
            self.koleo_loss(s_global_cls_stack[i]) for i in range(n_global)
        ) / n_global

        loss = dino_loss + ibot_loss + 0.1 * koleo_loss

        # Freeze prototype weights (last layer of heads) for first N epochs
        if self.current_epoch < self.args.freeze_last_layer:
            for head in (self.student_dino_head, self.student_ibot_head):
                for name, p in head.named_parameters():
                    if "last_layer" in name:
                        p.grad = None

        opt = self.optimizers()
        current_lr = opt.param_groups[0]["lr"]
        self.log_dict({
            "train/loss":   loss,
            "train/dino":   dino_loss,
            "train/ibot":   ibot_loss,
            "train/koleo":  koleo_loss,
            "train/ema_m":  self._ema_momentum(),
            "train/t_temp": teacher_temp,
            "train/lr":     current_lr,
        }, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)

        # Step-level print so tee-buffered log files show progress (rank 0 only)
        if self.global_step % 50 == 0 and self.trainer.is_global_zero:
            print(
                f"  step={self.global_step:6d}  "
                f"loss={loss.item():.4f}  "
                f"dino={dino_loss.item():.4f}  "
                f"ibot={ibot_loss.item():.4f}  "
                f"koleo={koleo_loss.item():.4f}",
                flush=True,
            )

        return loss

    def on_train_epoch_end(self):
        if not self.trainer.is_global_zero:
            return
        m = self.trainer.logged_metrics
        # Lightning appends _epoch suffix when on_step=True, on_epoch=True
        print(
            f"\n{'='*70}\n"
            f"  EPOCH {self.current_epoch:3d} COMPLETE\n"
            f"  loss={m.get('train/loss_epoch', float('nan')):.4f}  "
            f"dino={m.get('train/dino_epoch', float('nan')):.4f}  "
            f"ibot={m.get('train/ibot_epoch', float('nan')):.4f}  "
            f"koleo={m.get('train/koleo_epoch', float('nan')):.4f}  "
            f"lr={m.get('train/lr_epoch', self.args.lr):.2e}  "
            f"mom={m.get('train/ema_m_epoch', float('nan')):.5f}\n"
            f"{'='*70}",
            flush=True,
        )

    def on_after_backward(self):
        self._step += 1
        self._update_teacher()
        total_norm = sum(
            p.grad.data.norm(2).item() ** 2
            for p in self.student_backbone.parameters()
            if p.grad is not None
        ) ** 0.5
        self.log("train/backbone_grad_norm", total_norm,
                 on_step=True, on_epoch=False, sync_dist=True, prog_bar=False)

    # ── patch masking helper ──────────────────────────────────────────────────
    @torch.no_grad()
    def _apply_patch_mask(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Zero-out 16³ patch regions in the input volume where mask=True."""
        B, C, D, H, W = x.shape
        P = 6   # patches per dim (96 / 16)
        # (B, 216) → (B, 6, 6, 6) → (B, 96, 96, 96) via repeat_interleave
        vol_mask = mask.view(B, P, P, P) \
                       .repeat_interleave(16, dim=1) \
                       .repeat_interleave(16, dim=2) \
                       .repeat_interleave(16, dim=3)  # (B, 96, 96, 96)
        x = x.clone()
        x[vol_mask.unsqueeze(1).expand_as(x)] = 0.0
        return x

    # ── optimiser ─────────────────────────────────────────────────────────────
    def configure_optimizers(self):
        params = [
            {"params": self.student_backbone.parameters()},
            {"params": self.student_dino_head.parameters()},
            {"params": self.student_ibot_head.parameters()},
        ]
        opt = torch.optim.AdamW(params, lr=self.args.lr,
                                weight_decay=self.args.weight_decay,
                                betas=(0.9, 0.999))

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(self.args.warmup_epochs *
                           (total_steps / self.args.epochs))

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            cosine = 0.5 * (1 + math.cos(math.pi * progress))
            min_ratio = self.args.min_lr / self.args.lr
            return max(min_ratio, cosine)

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return [opt], [{"scheduler": sched, "interval": "step"}]


# ── Dataset ────────────────────────────────────────────────────────────────────

class FOMO300KDataset(Dataset):
    """
    Expects FOMO300K preprocessed as flat .npy files (z-normalised, 1mm iso, cropped).
    Same preprocessing as FOMO2JOMO baseline (fomo25/src/data/fomo-60k/preprocess.py).
    Returns multi-view crops for DINO training.
    """

    def __init__(self, data_dir: str, global_size=96, local_size=48, n_local=6):
        self.files = sorted(Path(data_dir).rglob("*.npy"))
        assert len(self.files) > 0, f"No .npy files found in {data_dir}"
        self.global_size = global_size
        self.local_size = local_size
        self.n_local = n_local
        print(f"Dataset: {len(self.files)} scans in {data_dir}")

    def __len__(self):
        return len(self.files)

    def _random_crop_3d(self, vol: np.ndarray, size: int) -> np.ndarray:
        """Random 3D crop of shape (size, size, size) from vol."""
        d, h, w = vol.shape
        if d < size or h < size or w < size:
            # Pad if too small
            pad = [(max(0, size - s + 1) // 2, max(0, size - s + 1) - max(0, size - s + 1) // 2)
                   for s in (d, h, w)]
            vol = np.pad(vol, pad, mode="reflect")
            d, h, w = vol.shape
        sd = np.random.randint(0, d - size + 1)
        sh = np.random.randint(0, h - size + 1)
        sw = np.random.randint(0, w - size + 1)
        return vol[sd:sd+size, sh:sh+size, sw:sw+size]

    def _augment(self, crop: np.ndarray) -> torch.Tensor:
        """Minimal augmentation: random flip + intensity jitter."""
        for ax in range(3):
            if np.random.rand() < 0.5:
                crop = np.flip(crop, axis=ax).copy()
        # Gaussian noise
        crop = crop + np.random.randn(*crop.shape).astype(np.float32) * 0.05
        # Clip to reasonable range and rescale to [0,1]
        crop = np.clip(crop, -3.5, 3.5)
        crop = (crop + 3.5) / 7.0
        return torch.from_numpy(crop).float().unsqueeze(0)   # (1, size, size, size)

    def __getitem__(self, idx):
        vol = np.load(self.files[idx]).astype(np.float32)

        global_crops = torch.stack([
            self._augment(self._random_crop_3d(vol, self.global_size))
            for _ in range(2)
        ])  # (2, 1, 96, 96, 96)

        local_crops = torch.stack([
            self._augment(self._random_crop_3d(vol, self.local_size))
            for _ in range(self.n_local)
        ])  # (n_local, 1, 48, 48, 48)

        return {"global_crops": global_crops, "local_crops": local_crops}


def collate_fn(batch):
    global_crops = torch.stack([b["global_crops"] for b in batch], dim=1)  # (2,B,1,96,96,96)
    local_crops  = torch.stack([b["local_crops"]  for b in batch], dim=1)  # (n,B,1,48,48,48)
    return {"global_crops": global_crops, "local_crops": local_crops}


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",       required=True)
    p.add_argument("--output_dir",     required=True)
    p.add_argument("--brainiac_ckpt",  default="")

    p.add_argument("--epochs",          type=int,   default=100)
    p.add_argument("--warmup_epochs",   type=int,   default=10)
    p.add_argument("--batch_size",      type=int,   default=8)
    p.add_argument("--grad_accum",      type=int,   default=2)
    p.add_argument("--num_workers",     type=int,   default=8)

    p.add_argument("--global_crop_size", type=int,  default=96)
    p.add_argument("--local_crop_size",  type=int,  default=48)
    p.add_argument("--local_crops_n",    type=int,  default=6)
    p.add_argument("--ibot_mask_ratio",  type=float, default=0.3)

    p.add_argument("--lr",              type=float, default=1e-3)
    p.add_argument("--min_lr",          type=float, default=1e-6)
    p.add_argument("--weight_decay",    type=float, default=0.04)
    p.add_argument("--weight_decay_end", type=float, default=0.4)
    p.add_argument("--clip_grad",       type=float, default=3.0)
    p.add_argument("--freeze_last_layer", type=int, default=1)

    p.add_argument("--dino_prototypes", type=int,   default=65536)
    p.add_argument("--ibot_prototypes", type=int,   default=65536)
    p.add_argument("--teacher_momentum", type=float, default=0.992)
    p.add_argument("--teacher_temp",    type=float, default=0.07)
    p.add_argument("--warmup_temp",     type=float, default=0.04)
    p.add_argument("--warmup_temp_epochs", type=int, default=30)

    p.add_argument("--save_every",      type=int,   default=10)
    p.add_argument("--precision",       type=str,   default="bf16-mixed")
    p.add_argument("--experiment",      type=str,   default="dino3d")
    p.add_argument("--max_steps",       type=int,   default=-1,
                   help="Stop after N optimizer steps (default=-1 = full training). "
                        "Set to ~50-200 for a smoke test.")
    p.add_argument("--no_wandb",        action="store_true",
                   help="Disable W&B logging (useful for smoke tests).")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    torch.set_float32_matmul_precision("high")
    L.seed_everything(42)

    # Dataset
    dataset = FOMO300KDataset(
        args.data_dir,
        global_size=args.global_crop_size,
        local_size=args.local_crop_size,
        n_local=args.local_crops_n,
    )
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              collate_fn=collate_fn, drop_last=True)

    # Model
    model = DINO3DModel(args)
    if args.brainiac_ckpt and os.path.isfile(args.brainiac_ckpt):
        model.load_brainiac_weights(args.brainiac_ckpt)

    # Callbacks
    ckpt_cb = ModelCheckpoint(
        dirpath=args.output_dir,
        filename="epoch{epoch:03d}",
        every_n_epochs=args.save_every,
        save_last=True,
        save_top_k=-1,
    )
    if args.no_wandb:
        logger = False
        callbacks = [ckpt_cb]
    else:
        wandb.init(project="fomo26-dino3d", name=args.experiment, dir=args.output_dir)
        logger = WandbLogger(project="fomo26-dino3d", name=args.experiment,
                             save_dir=args.output_dir, log_model=False)
        callbacks = [ckpt_cb, LearningRateMonitor(logging_interval="step")]

    trainer = L.Trainer(
        max_epochs=args.epochs,
        max_steps=args.max_steps,
        accelerator="gpu",
        devices="auto",
        strategy="ddp_find_unused_parameters_false",
        precision=args.precision,
        accumulate_grad_batches=args.grad_accum,
        gradient_clip_val=args.clip_grad,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=20,
        enable_progress_bar=True,
        num_sanity_val_steps=0,
    )

    trainer.fit(model, train_loader, ckpt_path="last")
    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
