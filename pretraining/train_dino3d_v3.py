"""
train_dino3d_v3.py — 3D DINOv3-style pretraining on FOMO300K

Changes from v2:
  - 3D RoPE via rope-nd math (no learned pos_embed, generalises to any crop size)
  - Storage / register tokens (4 × DINOv3 style)
  - embed_dim=1152, head_dim=72 (72 % 6 = 0 — required for 3D RoPE)
  - LayerScale (init=1e-5) + stochastic depth on every block
  - Trains from scratch (RoPE architecture is incompatible with 3DINO checkpoint)
"""

import argparse
import copy
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from timm.layers import DropPath
from torch.utils.data import DataLoader, Dataset
import wandb

_HERE = Path(__file__).parent

# ── DINOv3 losses + heads + 3D augmentations ─────────────────────────────────
sys.path.insert(0, str(_HERE / "dinov3"))
from dinov3.loss import DINOLoss, iBOTPatchLoss, KoLeoLoss
from dinov3.layers.dino_head import DINOHead
from dinov3.layers import LayerScale, Mlp
from dinov3.data.augmentations import DataAugmentationDINO3d


# ── Modality parsing (same as v2) ─────────────────────────────────────────────
NUM_MODALITIES = 9
_EXACT_MAP = {
    "T1w": 0, "MP2RAGE": 0, "UNIT1": 0, "T1map": 0, "R1map": 0,
    "T2w": 1, "T2starw": 1, "MESE": 1,
    "FLAIR": 2,
    "T1c": 3,
    "dwi": 4, "trace": 4, "adc": 4,
    "PDw": 5,
    "gre": 6, "swi": 6, "angio": 6, "FLASH": 6, "UTE": 6, "R2starmap": 6,
    "asl": 7, "cbf": 7, "m0scan": 7, "att": 7,
}


def parse_modality(path: Path) -> int:
    stem = path.stem
    suffix = stem.split("_")[-1]
    if suffix.startswith("bval"):
        return 4
    if suffix in _EXACT_MAP:
        return _EXACT_MAP[suffix]
    for key, idx in _EXACT_MAP.items():
        if stem.endswith(f"_{key}") or stem == key:
            return idx
    return NUM_MODALITIES - 1


# ── 3D RoPE (rope-nd math, dynamic spatial size) ──────────────────────────────

class RoPE3D(nn.Module):
    """
    3D Rotary Position Embedding using rope-nd's formula, computed on-the-fly
    for any (H, W, D) crop size.

    Constraint: head_dim % 6 == 0  (3 spatial axes × 2 for complex rotation)
    The frequency tensor `theta` is stored as a persistent buffer so it is
    saved/restored with the model checkpoint.
    """

    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        assert head_dim % 6 == 0, (
            f"head_dim={head_dim} must be divisible by 6 for 3D RoPE "
            f"(3 axes × 2 for complex rotation)"
        )
        k_max = head_dim // 6   # frequencies per spatial axis
        theta = 1.0 / (base ** (torch.arange(k_max).float() / k_max))
        self.register_buffer("theta", theta)   # [k_max]

    def _rotations(self, H: int, W: int, D: int) -> torch.Tensor:
        """Compute [HWD, head_dim//2] complex rotation tensor for given grid."""
        t = self.theta                                          # [k_max]
        gh = torch.arange(H, device=t.device, dtype=torch.float32)
        gw = torch.arange(W, device=t.device, dtype=torch.float32)
        gd = torch.arange(D, device=t.device, dtype=torch.float32)
        # outer product → [axis, k_max], then broadcast to [H, W, D, k_max]
        ah = (gh[:, None, None, None] * t).expand(H, W, D, -1)
        aw = (gw[None, :, None, None] * t).expand(H, W, D, -1)
        ad = (gd[None, None, :, None] * t).expand(H, W, D, -1)
        angles = torch.cat([ah, aw, ad], dim=-1)               # [H,W,D, head_dim//2]
        angles = angles.reshape(H * W * D, -1)                 # [HWD, head_dim//2]
        return torch.polar(torch.ones_like(angles), angles)    # complex

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        H: int, W: int, D: int,
        prefix: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply 3D RoPE to the spatial tokens of Q and K.

        q, k : [B, heads, N, head_dim]  where N = prefix + H*W*D
        prefix: number of non-spatial prefix tokens (CLS + storage) to skip
        """
        rot = self._rotations(H, W, D)   # [HWD, head_dim//2] complex

        def apply(x: torch.Tensor) -> torch.Tensor:
            # x: [B, heads, HWD, head_dim]
            xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
            return torch.view_as_real(xc * rot).flatten(-2).to(x.dtype)

        q = torch.cat([q[:, :, :prefix], apply(q[:, :, prefix:])], dim=2)
        k = torch.cat([k[:, :, :prefix], apply(k[:, :, prefix:])], dim=2)
        return q, k


# ── 3D Patch Embedding ────────────────────────────────────────────────────────

class PatchEmbed3D(nn.Module):
    def __init__(self, patch_size: int = 16, in_chans: int = 1, embed_dim: int = 1152):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def reset_parameters(self):
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor):
        # x: [B, C, W, H, D]
        x = self.proj(x)                          # [B, embed_dim, Pw, Ph, Pd]
        Pw, Ph, Pd = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)          # [B, Pw*Ph*Pd, embed_dim]
        return x, (Pw, Ph, Pd)


# ── 3D Self-Attention ──────────────────────────────────────────────────────────

class SelfAttention3D(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.rope = RoPE3D(self.head_dim, base=rope_base)

    def forward(
        self,
        x: torch.Tensor,
        H: int, W: int, D: int,
        prefix: int,
    ) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2)   # [B, heads, N, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q, k = self.rope(q, k, H, W, D, prefix)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


# ── Transformer Block ──────────────────────────────────────────────────────────

class Block3D(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop_path: float = 0.0,
        layerscale_init: float = 1e-5,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = SelfAttention3D(dim, num_heads, qkv_bias=qkv_bias,
                                    proj_bias=proj_bias, rope_base=rope_base)
        self.ls1 = LayerScale(dim, init_values=layerscale_init)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=int(dim * ffn_ratio),
                       bias=ffn_bias)
        self.ls2 = LayerScale(dim, init_values=layerscale_init)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, H: int, W: int, D: int, prefix: int) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), H, W, D, prefix)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


# ── Backbone ───────────────────────────────────────────────────────────────────

EMBED_DIM = 1152   # head_dim = 1152 // 16 = 72,  72 % 6 = 0  ✓ for 3D RoPE
N_STORAGE  = 4     # DINOv3-style register / storage tokens


class DINO3dBackboneV3(nn.Module):
    """
    DINOv3-style ViT for 3D volumes.

    No learned pos_embed — position is encoded via 3D RoPE inside every
    attention layer (rope-nd math, dynamic crop size).
    Prepends 1 CLS token + N_STORAGE storage tokens; patch tokens follow.
    """

    def __init__(
        self,
        patch_size: int = 16,
        in_chans: int = 1,
        embed_dim: int = EMBED_DIM,
        depth: int = 24,
        num_heads: int = 16,
        ffn_ratio: float = 4.0,
        n_storage_tokens: int = N_STORAGE,
        drop_path_rate: float = 0.1,
        layerscale_init: float = 1e-5,
        rope_base: float = 10000.0,
        use_modality_embed: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_storage_tokens = n_storage_tokens
        self.prefix = 1 + n_storage_tokens          # CLS + storage

        self.patch_embed = PatchEmbed3D(patch_size, in_chans, embed_dim)
        self.cls_token   = nn.Parameter(torch.empty(1, 1, embed_dim))
        if n_storage_tokens > 0:
            self.storage_tokens = nn.Parameter(torch.empty(1, n_storage_tokens, embed_dim))
        self.mask_token  = nn.Parameter(torch.empty(1, embed_dim))

        # stochastic depth — linearly increasing schedule
        dpr = [drop_path_rate * i / (depth - 1) for i in range(depth)]
        self.blocks = nn.ModuleList([
            Block3D(
                dim=embed_dim,
                num_heads=num_heads,
                ffn_ratio=ffn_ratio,
                drop_path=dpr[i],
                layerscale_init=layerscale_init,
                rope_base=rope_base,
            )
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        self.use_modality_embed = use_modality_embed
        if use_modality_embed:
            self.modality_embed = nn.Embedding(NUM_MODALITIES, embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.zeros_(self.mask_token)
        if self.n_storage_tokens > 0:
            nn.init.trunc_normal_(self.storage_tokens, std=0.02)
        if self.use_modality_embed:
            nn.init.normal_(self.modality_embed.weight, std=0.02)
        self.patch_embed.reset_parameters()
        for blk in self.blocks:
            nn.init.trunc_normal_(blk.attn.qkv.weight, std=0.02)
            nn.init.trunc_normal_(blk.attn.proj.weight, std=0.02)
            nn.init.trunc_normal_(blk.mlp.fc1.weight, std=0.02)
            nn.init.trunc_normal_(blk.mlp.fc2.weight, std=0.02)

    def prepare_tokens_with_masks(
        self,
        x: torch.Tensor,
        masks: torch.Tensor | None = None,
        modality_idx: torch.Tensor | None = None,
    ):
        B = x.shape[0]
        x, (Hw, Hh, Hd) = self.patch_embed(x)      # [B, HWD, C]

        # mask token substitution (before prepending CLS/storage)
        if masks is not None:
            x = torch.where(
                masks.unsqueeze(-1),
                self.mask_token.to(x.dtype).unsqueeze(0).expand(B, -1, -1),
                x,
            )

        # modality embedding on patch tokens
        if self.use_modality_embed and modality_idx is not None:
            x = x + self.modality_embed(modality_idx).unsqueeze(1)

        # prepend CLS + storage tokens
        cls = self.cls_token.expand(B, -1, -1)
        if self.n_storage_tokens > 0:
            storage = self.storage_tokens.expand(B, -1, -1)
            x = torch.cat([cls, storage, x], dim=1)
        else:
            x = torch.cat([cls, x], dim=1)

        return x, (Hw, Hh, Hd)

    def forward_features(
        self,
        x: torch.Tensor,
        masks: torch.Tensor | None = None,
        modality_idx: torch.Tensor | None = None,
    ) -> dict:
        x, (Hw, Hh, Hd) = self.prepare_tokens_with_masks(x, masks, modality_idx)
        for blk in self.blocks:
            x = blk(x, Hw, Hh, Hd, self.prefix)
        x_norm = self.norm(x)
        return {
            "x_norm_clstoken":    x_norm[:, 0],
            "x_storage_tokens":   x_norm[:, 1:self.prefix],
            "x_norm_patchtokens": x_norm[:, self.prefix:],
            "x_prenorm": x,
            "masks": masks,
        }

    def forward(self, x, masks=None, modality_idx=None, is_training=False):
        ret = self.forward_features(x, masks=masks, modality_idx=modality_idx)
        if is_training:
            return ret
        return ret["x_norm_clstoken"]


def _build_backbone(args) -> DINO3dBackboneV3:
    return DINO3dBackboneV3(
        patch_size=16,
        in_chans=1,
        embed_dim=EMBED_DIM,
        depth=24,
        num_heads=16,
        ffn_ratio=4.0,
        n_storage_tokens=N_STORAGE,
        drop_path_rate=args.drop_path_rate,
        layerscale_init=1e-5,
        rope_base=args.rope_base,
        use_modality_embed=args.modality_embed,
    )


# ── Lightning Module ───────────────────────────────────────────────────────────

class DINO3DModelV3(L.LightningModule):

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        self.args = args

        self.student_backbone  = _build_backbone(args)
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

        self.dino_loss = DINOLoss(args.dino_prototypes)
        self.ibot_loss = iBOTPatchLoss(args.ibot_prototypes)
        self.koleo_loss = KoLeoLoss()
        self.dino_loss.init_weights()
        self.ibot_loss.init_weights()

        self._step = 0

    # ── EMA ───────────────────────────────────────────────────────────────────

    def _ema_momentum(self) -> float:
        total = max(1, self.trainer.estimated_stepping_batches)
        return self.args.teacher_momentum + (1 - self.args.teacher_momentum) * (
            math.cos(math.pi * self._step / total) + 1
        ) / 2

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

    # ── temperature schedule ──────────────────────────────────────────────────

    def _teacher_temp(self) -> float:
        total = max(1, self.trainer.estimated_stepping_batches)
        warmup = int(self.args.warmup_temp_epochs * total / self.args.epochs)
        if self._step < warmup:
            return self.args.warmup_temp + (self.args.teacher_temp - self.args.warmup_temp) * (
                self._step / max(1, warmup)
            )
        return self.args.teacher_temp

    # ── patch mask ────────────────────────────────────────────────────────────

    def _make_patch_mask(self, B: int, N: int) -> torch.Tensor:
        n_mask = max(1, int(N * self.args.ibot_mask_ratio))
        mask = torch.zeros(B, N, dtype=torch.bool)
        for i in range(B):
            mask[i, torch.randperm(N)[:n_mask]] = True
        return mask

    # ── training step ─────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        global_crops  = batch["global_crops"]        # (2, B, 1, g, g, g)
        local_crops   = batch["local_crops"]         # (n, B, 1, l, l, l)
        modality_idx  = batch.get("modality_idx")

        n_global, B = global_crops.shape[:2]
        n_local      = local_crops.shape[0]
        N_global     = (self.args.global_crop_size // 16) ** 3   # 343 at 112³
        teacher_temp = self._teacher_temp()

        mask = self._make_patch_mask(B, N_global).to(self.device)   # (B, N_global)

        # ── Teacher ──────────────────────────────────────────────────────────
        with torch.no_grad():
            t_outs = [
                self.teacher_backbone(
                    global_crops[i], masks=None,
                    modality_idx=modality_idx, is_training=True,
                )
                for i in range(n_global)
            ]
            t_cls   = torch.stack([o["x_norm_clstoken"]    for o in t_outs])   # (2,B,C)
            t_patch = torch.stack([o["x_norm_patchtokens"] for o in t_outs])   # (2,B,N,C)

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

        # ── Student ───────────────────────────────────────────────────────────
        s0 = self.student_backbone(
            global_crops[0], masks=mask, modality_idx=modality_idx, is_training=True)
        s1 = self.student_backbone(
            global_crops[1], masks=None, modality_idx=modality_idx, is_training=True)

        s_global_cls = [s0["x_norm_clstoken"], s1["x_norm_clstoken"]]
        s_local_cls  = [
            self.student_backbone(
                local_crops[i], masks=None,
                modality_idx=modality_idx, is_training=True,
            )["x_norm_clstoken"]
            for i in range(n_local)
        ]

        s_cls_all   = torch.stack(s_global_cls + s_local_cls)          # (2+n, B, C)
        s_dino_head = self.student_dino_head(s_cls_all.flatten(0, 1))
        s_dino_head = s_dino_head.unflatten(0, (n_global + n_local, B))

        s_ibot_logits = self.student_ibot_head(
            s0["x_norm_patchtokens"].reshape(B * N_global, EMBED_DIM)).view(B, N_global, -1)

        # ── Losses ────────────────────────────────────────────────────────────
        dino_loss = self.dino_loss(
            student_logits=s_dino_head, teacher_probs=t_dino_soft)
        ibot_loss = self.ibot_loss.forward(
            student_patch_tokens=s_ibot_logits,
            teacher_patch_tokens=t_ibot_soft,
            student_masks_flat=mask,
        )
        koleo_loss = sum(
            self.koleo_loss(s_global_cls[i]) for i in range(n_global)
        ) / n_global

        loss = dino_loss + ibot_loss + 0.1 * koleo_loss

        # ── Freeze last-layer heads early in training ─────────────────────────
        if self.current_epoch < self.args.freeze_last_layer:
            for h in (self.student_dino_head, self.student_ibot_head):
                if hasattr(h, "last_layer"):
                    for p in h.last_layer.parameters():
                        p.grad = None

        current_lr = self.optimizers().param_groups[0]["lr"]
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
            f"mom={m.get('train/ema_m_epoch', float('nan')):.5f}\n"
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


# ── Dataset (reused from v2) ──────────────────────────────────────────────────

class FOMO300KDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        global_crops_in_slice_scale=(0.48, 1.0),
        global_crops_cross_slice_scale=(0.48, 1.0),
        local_crops_in_slice_scale=(0.08, 0.48),
        local_crops_cross_slice_scale=(0.1, 0.5),
        n_local: int = 6,
        global_crops_size: int = 112,
        local_crops_size: int = 48,
    ):
        self.files = sorted(Path(data_dir).rglob("*.npy"))
        assert len(self.files) > 0, f"No .npy files in {data_dir}"
        self.modality_indices = [parse_modality(f) for f in self.files]
        print(f"Dataset: {len(self.files)} scans in {data_dir}")

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
        vol = np.clip(vol, -3.5, 3.5) / 3.5
        vol_t = torch.from_numpy(vol).unsqueeze(0)
        output, _ = self.augmentation(vol_t)
        return {
            "global_crops": torch.stack(output["global_crops"]),
            "local_crops":  torch.stack(output["local_crops"]),
            "modality_idx": self.modality_indices[idx],
        }


def collate_fn(batch):
    return {
        "global_crops": torch.stack([b["global_crops"] for b in batch], dim=1),
        "local_crops":  torch.stack([b["local_crops"]  for b in batch], dim=1),
        "modality_idx": torch.tensor([b["modality_idx"] for b in batch], dtype=torch.long),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   required=True)
    p.add_argument("--output_dir", required=True)

    p.add_argument("--modality_embed", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--epochs",             type=int,   default=100)
    p.add_argument("--warmup_epochs",      type=int,   default=10)
    p.add_argument("--batch_size",         type=int,   default=8)
    p.add_argument("--grad_accum",         type=int,   default=1)
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
    p.add_argument("--drop_path_rate",     type=float, default=0.1)
    p.add_argument("--rope_base",          type=float, default=10000.0)

    p.add_argument("--dino_prototypes",    type=int,   default=65536)
    p.add_argument("--ibot_prototypes",    type=int,   default=65536)
    p.add_argument("--teacher_momentum",   type=float, default=0.992)
    p.add_argument("--teacher_temp",       type=float, default=0.07)
    p.add_argument("--warmup_temp",        type=float, default=0.04)
    p.add_argument("--warmup_temp_epochs", type=int,   default=30)

    p.add_argument("--save_every",  type=int,   default=10)
    p.add_argument("--precision",   type=str,   default="bf16-mixed")
    p.add_argument("--experiment",  type=str,   default="dino3d_v3")
    p.add_argument("--max_steps",   type=int,   default=-1)
    p.add_argument("--no_wandb",    action="store_true")
    p.add_argument("--ddp_backend", type=str,   default="nccl",
                   choices=["nccl", "gloo"])
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    torch.set_float32_matmul_precision("high")
    L.seed_everything(42)

    dataset = FOMO300KDataset(
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

    model = DINO3DModelV3(args)

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
