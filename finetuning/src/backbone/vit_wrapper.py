"""
Unified ViT backbone wrapper for FOMO26.

Wraps all three pretrained backbones behind a single interface:
  v1 — MONAI ViT-B  (96³,  embed_dim=768,  train_dino3d.py)
  v2 — 3DINO ViT-L  (112³, embed_dim=1024, train_dino3d_v2.py)
  v3 — RoPE ViT     (112³, embed_dim=1152, train_dino3d_v3.py)

All three expose:
  encode(x)                → {"cls": (B,D), "patches": (B,N,D)}
  encode_multi_sequence(x) → {"cls": (B,D), "patches": (B,N,D)}  multi-channel input
  forward(x)               → (B,D)  CLS token (for classifier heads)
"""

from __future__ import annotations
import sys
from functools import partial
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Backbone metadata
# ---------------------------------------------------------------------------
CONFIGS: Dict[str, dict] = {
    "v1": {"embed_dim": 768,  "img_size": 96,  "patch_size": 16, "n_patches": 216},
    "v2": {"embed_dim": 1024, "img_size": 112, "patch_size": 16, "n_patches": 343},
    "v3": {"embed_dim": 1152, "img_size": 112, "patch_size": 16, "n_patches": 343},
}

# Root of the challenge repo (two levels above this file: finetuning/src/backbone/)
_CHALLENGE = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Per-version loaders
# ---------------------------------------------------------------------------

def _load_v1(state: dict) -> nn.Module:
    """MONAI ViT-B wrapped as BrainIACViT."""
    sys.path.insert(0, str(_CHALLENGE / "pretraining"))
    from train_dino3d import BrainIACViT  # noqa: PLC0415
    model = BrainIACViT()
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  [v1] missing={len(missing)}, unexpected={len(unexpected)}")
    return model


def _load_v2(state: dict) -> nn.Module:
    """3DINO DinoVisionTransformer3d (ViT-L, block_chunks=4)."""
    sys.path.insert(0, str(_CHALLENGE / "3DINO"))
    from dinov2.models.vision_transformer import DinoVisionTransformer3d  # noqa: PLC0415
    from dinov2.layers import MemEffAttention                              # noqa: PLC0415
    from dinov2.layers.block import NestedTensorBlock as Block             # noqa: PLC0415

    model = DinoVisionTransformer3d(
        img_size=112, patch_size=16, in_chans=1,
        embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4.0,
        block_fn=partial(Block, attn_class=MemEffAttention),
        block_chunks=4, init_values=1.0,
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  [v2] missing={len(missing)}, unexpected={len(unexpected)}")
    return model


def _load_v3(state: dict) -> nn.Module:
    """Custom RoPE ViT (DINO3dBackboneV3)."""
    sys.path.insert(0, str(_CHALLENGE / "pretraining"))
    from train_dino3d_v3 import DINO3dBackboneV3  # noqa: PLC0415

    model = DINO3dBackboneV3()
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  [v3] missing={len(missing)}, unexpected={len(unexpected)}")
    return model


_LOADERS = {"v1": _load_v1, "v2": _load_v2, "v3": _load_v3}


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

class ViTWrapper(nn.Module):
    """
    Loads the student_backbone from a DINO Lightning checkpoint and exposes
    a unified encode() interface for downstream tasks.

    Args:
        model_version : "v1" | "v2" | "v3"
        checkpoint_path: Path to last.ckpt produced by the pretraining scripts.
        freeze: If True, all backbone parameters are frozen (for linear probing).
    """

    def __init__(self, model_version: str, checkpoint_path: str, freeze: bool = False):
        super().__init__()
        assert model_version in CONFIGS, f"Unknown version '{model_version}'. Choose v1/v2/v3."

        cfg = CONFIGS[model_version]
        self.model_version = model_version
        self.embed_dim  = cfg["embed_dim"]
        self.img_size   = cfg["img_size"]
        self.patch_size = cfg["patch_size"]
        self.n_patches  = cfg["n_patches"]

        # Extract student_backbone weights from Lightning checkpoint
        ckpt  = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        prefix = "student_backbone."
        backbone_state = {k[len(prefix):]: v for k, v in state.items()
                          if k.startswith(prefix)}

        if not backbone_state:
            # Fallback: checkpoint is raw backbone state (not Lightning)
            backbone_state = state

        print(f"[ViTWrapper] Loading {model_version} from {checkpoint_path}")
        self.backbone = _LOADERS[model_version](backbone_state)

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            print(f"  Backbone frozen.")

        print(f"  embed_dim={self.embed_dim}, img_size={self.img_size}³, "
              f"n_patches={self.n_patches}")

    # ------------------------------------------------------------------
    def _forward_raw(self, x: torch.Tensor) -> dict:
        """Run backbone and normalise output to {"x_norm_clstoken", "x_norm_patchtokens"}."""
        out = self.backbone(x)
        if isinstance(out, dict):
            return out
        # v1 BrainIACViT already returns the right dict; v2/v3 do too.
        # Fallback for any unexpected raw tuple (MONAI ViT returns (hidden, attn)).
        if isinstance(out, (tuple, list)):
            tokens = out[0]           # (B, 1+N, D)
            return {
                "x_norm_clstoken":    tokens[:, 0],
                "x_norm_patchtokens": tokens[:, 1:],
            }
        raise ValueError(f"Unexpected backbone output type: {type(out)}")

    # ------------------------------------------------------------------
    def encode(self, x: torch.Tensor) -> dict:
        """
        Single-sequence encode.

        x : (B, 1, H, W, D)
        Returns {"cls": (B, D), "patches": (B, N, D)}
        """
        raw = self._forward_raw(x)
        return {
            "cls":     raw["x_norm_clstoken"],
            "patches": raw["x_norm_patchtokens"],
        }

    def encode_multi_sequence(self, x: torch.Tensor) -> dict:
        """
        Multi-sequence encode (Tasks 1, 2 with DWI/FLAIR/ADC/SWI).

        x : (B, C, H, W, D)  — C independent sequences stacked on channel dim
        Each sequence is encoded independently; CLS tokens are mean-pooled.
        Patch tokens are taken from the first sequence only.

        Returns {"cls": (B, D), "patches": (B, N, D)}
        """
        C = x.shape[1]
        cls_list, patches = [], None
        for i in range(C):
            out = self.encode(x[:, i:i+1])
            cls_list.append(out["cls"])
            if patches is None:
                patches = out["patches"]
        return {
            "cls":     torch.stack(cls_list, dim=1).mean(dim=1),
            "patches": patches,
        }

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Default forward returns CLS token (B, D) for classifier heads."""
        if x.shape[1] == 1:
            return self.encode(x)["cls"]
        return self.encode_multi_sequence(x)["cls"]
