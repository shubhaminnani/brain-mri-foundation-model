"""
train_dino3d_v2.py — 3D DINOv2 pretraining on FOMO300K

Improvements over v1:
  - True CLS token via DinoVisionTransformer3d (3DINO ViT-B backbone)
  - mask_token-based iBOT masking (proper learned token, not input zeroing)
  - Modality embeddings (T1w/T2w/T2starw/FLAIR/PDw parsed from filename)
  - 3DINO augmentation pipeline (RandGibbsNoise, RandHistogramShift,
    RandAdjustContrast, anisotropy-aware RandomResizedCrop3d)
  - Local crops processed at native 48^3 via trilinear PE interpolation
    (no upsampling workaround needed)
"""

import argparse
import copy
import math
import os
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader, Dataset
import wandb

# ── 3DINO backbone + augmentations ───────────────────────────────────────────
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE / "3DINO"))
from dinov2.models.vision_transformer import DinoVisionTransformer3d
from dinov2.layers import MemEffAttention
from dinov2.layers.block import NestedTensorBlock as Block
from dinov2.data.augmentations import DataAugmentationDINO3d

# ── DINOv3 losses ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(_HERE / "dinov3"))
from dinov3.loss import DINOLoss, iBOTPatchLoss, KoLeoLoss
from dinov3.layers.dino_head import DINOHead


# ── Modality parsing ──────────────────────────────────────────────────────────
# Groups based on FOMO300K actual distribution (from mri_info.tsv):
#   0 T1w        T1w, MP2RAGE, UNIT1               ~83K scans
#   1 T2w        T2w, T2starw                       ~33K scans
#   2 FLAIR      FLAIR                              ~22K scans
#   3 T1c        T1c (contrast-enhanced T1)         ~13K scans
#   4 DWI        dwi, bval*, trace, adc             ~80K scans
#   5 PDw        PDw                                ~3.5K scans
#   6 GRE/SWI    gre, swi, angio                    ~12K scans
#   7 ASL        asl, cbf, m0scan                   ~7K scans
#   8 unknown    anything else
NUM_MODALITIES = 9

_EXACT_MAP = {
    # T1-like (0)
    "T1w": 0, "MP2RAGE": 0, "UNIT1": 0, "T1map": 0, "R1map": 0,
    # T2-like (1)
    "T2w": 1, "T2starw": 1, "MESE": 1,
    # FLAIR (2)
    "FLAIR": 2,
    # Contrast-enhanced T1 (3)
    "T1c": 3,
    # Diffusion (4)
    "dwi": 4, "trace": 4, "adc": 4,
    # Proton density (5)
    "PDw": 5,
    # GRE / susceptibility (6)
    "gre": 6, "swi": 6, "angio": 6, "FLASH": 6, "UTE": 6, "R2starmap": 6,
    # ASL / perfusion (7)
    "asl": 7, "cbf": 7, "m0scan": 7, "att": 7,
    # "scan" and "MTRmap" left as unknown (8) — too ambiguous / too few
}


def parse_modality(path: Path) -> int:
    stem = path.stem
    suffix = stem.split("_")[-1]   # last BIDS entity, e.g. "T1w", "bval1000"

    # DWI b-value volumes (bval0, bval1000, bval2000, …)
    if suffix.startswith("bval"):
        return 4

    if suffix in _EXACT_MAP:
        return _EXACT_MAP[suffix]

    # Legacy / fallback exact-match on full stem entries
    for key, idx in _EXACT_MAP.items():
        if stem.endswith(f"_{key}") or stem == key:
            return idx
    return NUM_MODALITIES - 1


# ── Backbone ──────────────────────────────────────────────────────────────────

class DINO3dBackbone(DinoVisionTransformer3d):
    """DinoVisionTransformer3d with an optional per-modality embedding on patch tokens."""

    def __init__(self, use_modality_embed: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.use_modality_embed = use_modality_embed
        if use_modality_embed:
            self.modality_embed = nn.Embedding(NUM_MODALITIES, self.embed_dim)
            nn.init.normal_(self.modality_embed.weight, std=0.02)

    def prepare_tokens_with_masks(self, x, masks=None, modality_idx=None):
        B, nc, w, h, d = x.shape
        x = self.patch_embed(x)                           # (B, N, C)
        if masks is not None:
            x = torch.where(
                masks.unsqueeze(-1),
                self.mask_token.to(x.dtype).unsqueeze(0),
                x,
            )
        if self.use_modality_embed and modality_idx is not None:
            x = x + self.modality_embed(modality_idx).unsqueeze(1)  # (B,1,C) broadcast
        x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h, d)
        return x

    def forward_features(self, x, masks=None, modality_idx=None):
        if isinstance(x, list):
            return self.forward_features_list(x, masks)
        x = self.prepare_tokens_with_masks(x, masks, modality_idx)
        for blk in self.blocks:
            x = blk(x)
        x_norm = self.norm(x)
        return {
            "x_norm_clstoken":    x_norm[:, 0],
            "x_norm_patchtokens": x_norm[:, 1:],
            "x_prenorm": x,
            "masks": masks,
        }

    def forward(self, x, masks=None, modality_idx=None, is_training=False):
        ret = self.forward_features(x, masks=masks, modality_idx=modality_idx)
        if is_training:
            return ret
        return self.head(ret["x_norm_clstoken"])


def load_pretrained_3dino(backbone: DINO3dBackbone, ckpt_path: str, log=print) -> None:
    """Load 3DINO-ViT teacher weights into a DINO3dBackbone.

    The 3DINO checkpoint stores weights under 'teacher' → 'backbone.*'.
    With global_crop_size=112 (7×7×7 patches × 16), all weights including
    pos_embed [1,344,1024] match exactly — nothing needs to be skipped.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    src = ckpt.get("teacher", ckpt)

    SKIP: set[str] = set()  # all weights compatible when global_crop_size=112
    model_sd = backbone.state_dict()

    to_load, skip_incompat, skip_missing = {}, [], []

    for src_key, param in src.items():
        if not src_key.startswith("backbone."):
            continue
        key = src_key[len("backbone."):]

        if key in SKIP:
            skip_incompat.append(key)
            continue
        if key not in model_sd:
            skip_missing.append(key)
            continue
        if param.shape != model_sd[key].shape:
            skip_incompat.append(f"{key}: ckpt{list(param.shape)} vs model{list(model_sd[key].shape)}")
            continue
        to_load[key] = param

    backbone.load_state_dict(to_load, strict=False)
    log(f"[pretrained] Loaded {len(to_load)}/{len(model_sd)} tensors from {Path(ckpt_path).name}")
    if skip_incompat:
        log(f"[pretrained] Skipped (shape mismatch / explicit skip): {skip_incompat}")
    if skip_missing:
        log(f"[pretrained] Skipped (not in model): {skip_missing[:5]}")


EMBED_DIM = 1024  # ViT-L — matches 3DINO-ViT pretrained weights


def _build_backbone(args) -> DINO3dBackbone:
    return DINO3dBackbone(
        img_size=args.global_crop_size,
        patch_size=16,
        in_chans=1,
        embed_dim=EMBED_DIM,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        block_fn=partial(Block, attn_class=MemEffAttention),
        block_chunks=4,        # matches 3DINO checkpoint's nested block structure
        init_values=1.0,       # enables LayerScale (ls1.gamma / ls2.gamma in checkpoint)
        use_modality_embed=args.modality_embed,
    )


# ── Lightning module ──────────────────────────────────────────────────────────

class DINO3DModel(L.LightningModule):

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        self.args = args

        self.student_backbone  = _build_backbone(args)
        if args.pretrained:
            load_pretrained_3dino(self.student_backbone, args.pretrained)

        self.student_dino_head = DINOHead(in_dim=EMBED_DIM, out_dim=args.dino_prototypes,
                                          hidden_dim=2048, bottleneck_dim=256, nlayers=3)
        self.student_ibot_head = DINOHead(in_dim=EMBED_DIM, out_dim=args.ibot_prototypes,
                                          hidden_dim=2048, bottleneck_dim=256, nlayers=3)

        self.teacher_backbone  = copy.deepcopy(self.student_backbone)
        self.teacher_dino_head = copy.deepcopy(self.student_dino_head)
        self.teacher_ibot_head = copy.deepcopy(self.student_ibot_head)
        for p in (*self.teacher_backbone.parameters(),
                  *self.teacher_dino_head.parameters(),
                  *self.teacher_ibot_head.parameters()):
            p.requires_grad_(False)

        self.dino_loss  = DINOLoss(args.dino_prototypes)
        self.ibot_loss  = iBOTPatchLoss(args.ibot_prototypes)
        self.koleo_loss = KoLeoLoss()
        self.dino_loss.init_weights()
        self.ibot_loss.init_weights()

        self._step = 0

    # ── helpers ───────────────────────────────────────────────────────────────

    def _make_patch_mask(self, B: int, n_patches: int) -> torch.Tensor:
        n_mask = int(self.args.ibot_mask_ratio * n_patches)
        mask = torch.zeros(B, n_patches, dtype=torch.bool)
        for i in range(B):
            mask[i, torch.randperm(n_patches)[:n_mask]] = True
        return mask

    def _ema_momentum(self) -> float:
        total = self.trainer.estimated_stepping_batches
        m = self.args.teacher_momentum
        return 1 - (1 - m) * (math.cos(math.pi * self._step / total) + 1) / 2

    @torch.no_grad()
    def _update_teacher(self):
        m = self._ema_momentum()
        pairs = [
            (self.student_backbone,  self.teacher_backbone),
            (self.student_dino_head, self.teacher_dino_head),
            (self.student_ibot_head, self.teacher_ibot_head),
        ]
        for student_mod, teacher_mod in pairs:
            for ps, pt in zip(student_mod.parameters(), teacher_mod.parameters()):
                pt.data.mul_(m).add_(ps.data, alpha=1 - m)

    def _teacher_temp(self) -> float:
        e = self.current_epoch
        if e < self.args.warmup_temp_epochs:
            return self.args.warmup_temp + \
                   (self.args.teacher_temp - self.args.warmup_temp) * \
                   e / self.args.warmup_temp_epochs
        return self.args.teacher_temp

    # ── training step ─────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        global_crops  = batch["global_crops"]              # (2, B, 1, 96, 96, 96)
        local_crops   = batch["local_crops"]               # (n, B, 1, 48, 48, 48)
        modality_idx  = batch.get("modality_idx")          # (B,) long or None

        n_global, B = global_crops.shape[:2]
        n_local      = local_crops.shape[0]
        N_global     = (self.args.global_crop_size // 16) ** 3   # 343 (7³ at 112³ crop)
        teacher_temp = self._teacher_temp()

        # Random block mask for iBOT on first global crop only
        mask = self._make_patch_mask(B, N_global).to(self.device)   # (B, N_global)

        # ── Teacher: clean global crops, no masking ────────────────────────
        with torch.no_grad():
            t_outs = [
                self.teacher_backbone(
                    global_crops[i], masks=None,
                    modality_idx=modality_idx, is_training=True,
                )
                for i in range(n_global)
            ]
            t_cls   = torch.stack([o["x_norm_clstoken"]    for o in t_outs])  # (2,B,C)
            t_patch = torch.stack([o["x_norm_patchtokens"] for o in t_outs])  # (2,B,N,C)

            t_cls_head  = self.teacher_dino_head(t_cls.flatten(0, 1))
            t_dino_soft = self.dino_loss.sinkhorn_knopp_teacher(
                t_cls_head, teacher_temp=teacher_temp)
            t_dino_soft = t_dino_soft.unflatten(0, (n_global, B))

            t_ibot_logits = self.teacher_ibot_head(
                t_patch[0].reshape(B * N_global, EMBED_DIM)).view(B, N_global, -1)
            self.ibot_loss.apply_center_update()
            t_ibot_soft = F.softmax(
                (t_ibot_logits - self.ibot_loss.center) / teacher_temp, dim=-1)
            self.ibot_loss.reduce_center_update(t_ibot_logits)

        # ── Student: first global masked, rest clean ───────────────────────
        # mask_token substitution happens inside prepare_tokens_with_masks
        s0 = self.student_backbone(
            global_crops[0], masks=mask, modality_idx=modality_idx, is_training=True)
        s1 = self.student_backbone(
            global_crops[1], masks=None, modality_idx=modality_idx, is_training=True)

        s_global_cls = [s0["x_norm_clstoken"], s1["x_norm_clstoken"]]

        # Local crops at native 48^3; DinoVisionTransformer3d interpolates PE automatically
        s_local_cls = [
            self.student_backbone(
                local_crops[i], masks=None, modality_idx=modality_idx, is_training=True
            )["x_norm_clstoken"]
            for i in range(n_local)
        ]

        s_cls_all   = torch.stack(s_global_cls + s_local_cls)          # (2+n, B, C)
        s_dino_head = self.student_dino_head(s_cls_all.flatten(0, 1))
        s_dino_head = s_dino_head.unflatten(0, (n_global + n_local, B))

        s_ibot_logits = self.student_ibot_head(
            s0["x_norm_patchtokens"].reshape(B * N_global, EMBED_DIM)).view(B, N_global, -1)

        # ── Losses ────────────────────────────────────────────────────────
        dino_loss  = self.dino_loss(
            student_logits=s_dino_head, teacher_probs=t_dino_soft)
        ibot_loss  = self.ibot_loss.forward(
            student_patch_tokens=s_ibot_logits,
            teacher_patch_tokens=t_ibot_soft,
            student_masks_flat=mask,
        )
        koleo_loss = sum(
            self.koleo_loss(cls_i) for cls_i in s_global_cls
        ) / n_global
        loss = dino_loss + ibot_loss + 0.1 * koleo_loss

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
        }, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True, batch_size=B)

        if self.global_step % 50 == 0 and self.trainer.is_global_zero:
            print(
                f"  step={self.global_step:6d}  "
                f"loss={loss.item():.4f}  "
                f"dino={dino_loss.item():.4f}  "
                f"ibot={ibot_loss.item():.4f}  "
                f"koleo={koleo_loss.item():.4f}  "
                f"lr={current_lr:.2e}",
                flush=True,
            )

        return loss

    def on_train_epoch_end(self):
        if not self.trainer.is_global_zero:
            return
        m = self.trainer.logged_metrics
        print(
            f"\n{'='*70}\n"
            f"  EPOCH {self.current_epoch:3d} COMPLETE\n"
            f"  loss={m.get('train/loss_epoch', float('nan')):.4f}  "
            f"dino={m.get('train/dino_epoch', float('nan')):.4f}  "
            f"ibot={m.get('train/ibot_epoch', float('nan')):.4f}  "
            f"koleo={m.get('train/koleo_epoch', float('nan')):.4f}\n"
            f"  lr={m.get('train/lr_epoch', self.args.lr):.2e}  "
            f"mom={m.get('train/ema_m_epoch', float('nan')):.5f}  "
            f"t_temp={m.get('train/t_temp_epoch', float('nan')):.4f}\n"
            f"{'='*70}",
            flush=True,
        )

    def on_after_backward(self):
        self._step += 1
        self._update_teacher()

    # ── optimiser ─────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            [
                {"params": self.student_backbone.parameters()},
                {"params": self.student_dino_head.parameters()},
                {"params": self.student_ibot_head.parameters()},
            ],
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
            betas=(0.9, 0.999),
        )
        total   = self.trainer.estimated_stepping_batches
        warmup  = int(self.args.warmup_epochs * total / self.args.epochs)
        min_ratio = self.args.min_lr / self.args.lr

        def lr_lambda(step):
            if step < warmup:
                return step / max(1, warmup)
            progress = (step - warmup) / max(1, total - warmup)
            return max(min_ratio, 0.5 * (1 + math.cos(math.pi * progress)))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return [opt], [{"scheduler": sched, "interval": "step"}]


# ── Dataset ───────────────────────────────────────────────────────────────────

class FOMO300KDatasetV2(Dataset):
    """
    Loads preprocessed .npy volumes (z-normalised), rescales to [-1, 1],
    and applies 3DINO augmentations to produce multi-view DINO crops.
    """

    def __init__(
        self,
        data_dir: str,
        global_crops_in_slice_scale=(0.48, 1.0),
        global_crops_cross_slice_scale=(0.48, 1.0),
        local_crops_in_slice_scale=(0.08, 0.48),
        local_crops_cross_slice_scale=(0.1, 0.5),
        n_local: int = 6,
        global_crops_size: int = 96,
        local_crops_size: int = 48,
    ):
        self.files = sorted(Path(data_dir).rglob("*.npy"))
        assert len(self.files) > 0, f"No .npy files found in {data_dir}"
        self.modality_indices = [parse_modality(f) for f in self.files]
        print(f"Dataset: {len(self.files)} scans in {data_dir}")

        # MONAI 1.3.x bug: Compose seeds sub-transforms with randint(MAX_SEED, dtype='uint32')
        # where MAX_SEED=2^32 overflows uint32. Reset MONAI seed to None to skip this path.
        from monai.utils import set_determinism
        set_determinism(None)

        self.augmentation = DataAugmentationDINO3d(
            global_crops_in_slice_scale=global_crops_in_slice_scale,
            global_crops_cross_slice_scale=global_crops_cross_slice_scale,
            local_crops_in_slice_scale=local_crops_in_slice_scale,
            local_crops_cross_slice_scale=local_crops_cross_slice_scale,
            local_crops_number=n_local,
            global_crops_size=global_crops_size,
            local_crops_size=local_crops_size,
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        vol = np.load(self.files[idx]).astype(np.float32)
        # z-score clipped to ±3.5σ → rescale to [-1, 1] expected by 3DINO augmentations
        vol = np.clip(vol, -3.5, 3.5) / 3.5
        vol_t = torch.from_numpy(vol).unsqueeze(0)          # (1, H, W, D)

        output, _ = self.augmentation(vol_t)

        return {
            "global_crops": torch.stack(output["global_crops"]),  # (2, 1, 96, 96, 96)
            "local_crops":  torch.stack(output["local_crops"]),   # (n, 1, 48, 48, 48)
            "modality_idx": self.modality_indices[idx],
        }


def collate_fn(batch):
    return {
        "global_crops": torch.stack([b["global_crops"] for b in batch], dim=1),  # (2,B,...)
        "local_crops":  torch.stack([b["local_crops"]  for b in batch], dim=1),  # (n,B,...)
        "modality_idx": torch.tensor([b["modality_idx"] for b in batch], dtype=torch.long),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   required=True)
    p.add_argument("--output_dir", required=True)

    p.add_argument("--modality_embed", action=argparse.BooleanOptionalAction, default=True,
                   help="Add learned modality embedding to patch tokens.")

    p.add_argument("--epochs",             type=int,   default=100)
    p.add_argument("--warmup_epochs",      type=int,   default=10)
    p.add_argument("--batch_size",         type=int,   default=8)
    p.add_argument("--grad_accum",         type=int,   default=2)
    p.add_argument("--num_workers",        type=int,   default=8)

    p.add_argument("--global_crop_size",   type=int,   default=112)
    p.add_argument("--local_crop_size",    type=int,   default=48)
    p.add_argument("--local_crops_n",      type=int,   default=6)
    p.add_argument("--ibot_mask_ratio",    type=float, default=0.3)

    p.add_argument("--global_crops_in_slice_scale",    type=float, nargs=2, default=[0.48, 1.0])
    p.add_argument("--global_crops_cross_slice_scale", type=float, nargs=2, default=[0.48, 1.0])
    p.add_argument("--local_crops_in_slice_scale",     type=float, nargs=2, default=[0.08, 0.48])
    p.add_argument("--local_crops_cross_slice_scale",  type=float, nargs=2, default=[0.10, 0.50])

    p.add_argument("--lr",                 type=float, default=1e-3)
    p.add_argument("--min_lr",             type=float, default=1e-6)
    p.add_argument("--weight_decay",       type=float, default=0.04)
    p.add_argument("--clip_grad",          type=float, default=3.0)
    p.add_argument("--freeze_last_layer",  type=int,   default=1)

    p.add_argument("--dino_prototypes",    type=int,   default=65536)
    p.add_argument("--ibot_prototypes",    type=int,   default=65536)
    p.add_argument("--teacher_momentum",   type=float, default=0.992)
    p.add_argument("--teacher_temp",       type=float, default=0.07)
    p.add_argument("--warmup_temp",        type=float, default=0.04)
    p.add_argument("--warmup_temp_epochs", type=int,   default=30)

    p.add_argument("--save_every",  type=int,   default=10)
    p.add_argument("--precision",   type=str,   default="bf16-mixed")
    p.add_argument("--experiment",  type=str,   default="dino3d_v2")
    p.add_argument("--max_steps",   type=int,   default=-1,
                   help="Cap steps for smoke-test (default -1 = full run).")
    p.add_argument("--no_wandb",    action="store_true")
    p.add_argument("--pretrained",  type=str,
                   default="/mnt/f/challenge/3DINO/3dino_vit_weights.pth",
                   help="Path to 3DINO-ViT teacher checkpoint for weight init. "
                        "Pass empty string to train from scratch.")
    p.add_argument("--ddp_backend", type=str, default="nccl",
                   choices=["nccl", "gloo"],
                   help="DDP process group backend. Use 'gloo' for WSL2.")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    torch.set_float32_matmul_precision("high")
    L.seed_everything(42)

    dataset = FOMO300KDatasetV2(
        args.data_dir,
        global_crops_in_slice_scale=tuple(args.global_crops_in_slice_scale),
        global_crops_cross_slice_scale=tuple(args.global_crops_cross_slice_scale),
        local_crops_in_slice_scale=tuple(args.local_crops_in_slice_scale),
        local_crops_cross_slice_scale=tuple(args.local_crops_cross_slice_scale),
        n_local=args.local_crops_n,
        global_crops_size=args.global_crop_size,
        local_crops_size=args.local_crop_size,
    )
    train_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        collate_fn=collate_fn, drop_last=True,
    )

    model = DINO3DModel(args)

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
        strategy=DDPStrategy(process_group_backend=args.ddp_backend,
                             find_unused_parameters=False),
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
