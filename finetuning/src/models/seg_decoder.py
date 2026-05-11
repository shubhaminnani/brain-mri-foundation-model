"""
Segmentation decoders for all backbone versions.

v1 (96³, D=768)  → MONAI UNETR with BrainIAC-initialised ViT encoder
v2 (112³, D=1024) → PatchTokenDecoder (conv upsampler on final patch tokens)
v3 (112³, D=1152) → PatchTokenDecoder

PatchTokenDecoder works by reshaping (B, N, D) patch tokens into a 3-D grid
and progressively upsampling via transposed convolutions + skip-free blocks.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Patch-token decoder (v2 / v3)
# ---------------------------------------------------------------------------

class _ConvBlock(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.GELU(),
        )


class PatchTokenDecoder(nn.Module):
    """
    Decodes ViT patch tokens → voxel-space segmentation logits.

    The patch grid (e.g. 7³ at 112/16) is progressively doubled via
    transposed convolutions until the spatial resolution matches img_size.
    Channel count halves at each stage (clamped to min_ch).

    Args:
        embed_dim   : D from the ViT backbone (1024 or 1152)
        img_size    : target output spatial size (same as ViT input)
        patch_size  : patch size used in the backbone (default 16)
        out_channels: number of segmentation classes
        min_ch      : minimum feature channel count
    """

    def __init__(
        self,
        embed_dim: int,
        img_size: int,
        patch_size: int = 16,
        out_channels: int = 2,
        min_ch: int = 32,
    ):
        super().__init__()
        self.grid = img_size // patch_size           # e.g. 7
        self.img_size = img_size
        n_up = int(math.log2(patch_size))            # 4 stages for patch_size=16

        dims = [embed_dim]
        for _ in range(n_up):
            dims.append(max(dims[-1] // 2, min_ch))

        self.up_blocks = nn.ModuleList()
        for i in range(n_up):
            self.up_blocks.append(nn.Sequential(
                nn.ConvTranspose3d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                nn.InstanceNorm3d(dims[i + 1]),
                nn.GELU(),
                _ConvBlock(dims[i + 1], dims[i + 1]),
            ))

        self.head = nn.Conv3d(dims[-1], out_channels, kernel_size=1)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        patches : (B, N, D)
        returns : (B, out_channels, img_size, img_size, img_size)
        """
        B, N, D = patches.shape
        g = self.grid
        x = patches.transpose(1, 2).reshape(B, D, g, g, g)

        for blk in self.up_blocks:
            x = blk(x)

        if x.shape[-1] != self.img_size:
            x = F.interpolate(x, size=(self.img_size,) * 3,
                               mode="trilinear", align_corners=False)
        return self.head(x)


# ---------------------------------------------------------------------------
# UNETR wrapper for v1 (uses MONAI's built-in decoder with skip connections)
# ---------------------------------------------------------------------------

class UNETRSegDecoder(nn.Module):
    """
    MONAI UNETR with the ViT encoder pre-initialised from a BrainIAC
    Lightning checkpoint (student_backbone.vit.* keys).

    The full UNETR forward pass handles both encoding and decoding.
    """

    def __init__(
        self,
        out_channels: int = 2,
        img_size: int = 96,
        checkpoint_path: str | None = None,
    ):
        super().__init__()
        from monai.networks.nets import UNETR  # lazy import

        self.model = UNETR(
            in_channels=1,
            out_channels=out_channels,
            img_size=(img_size,) * 3,
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            norm_name="instance",
            res_block=True,
            dropout_rate=0.0,
        )

        if checkpoint_path:
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            state = ckpt.get("state_dict", ckpt)
            vit_state = {k[len("student_backbone.vit."):]: v
                         for k, v in state.items()
                         if k.startswith("student_backbone.vit.")}
            miss, unex = self.model.vit.load_state_dict(vit_state, strict=False)
            print(f"[UNETRSegDecoder] ViT init: missing={len(miss)}, unexpected={len(unex)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_seg_model(
    model_version: str,
    out_channels: int,
    checkpoint_path: str,
) -> nn.Module:
    """
    Returns a complete segmentation model for the given backbone version.

    v1 → UNETRSegDecoder  (end-to-end, includes its own ViT encoder)
    v2 → (ViTWrapper, PatchTokenDecoder)  returned as nn.ModuleDict
    v3 → (ViTWrapper, PatchTokenDecoder)  returned as nn.ModuleDict
    """
    from backbone.vit_wrapper import ViTWrapper, CONFIGS  # noqa: PLC0415

    cfg = CONFIGS[model_version]

    if model_version == "v1":
        return UNETRSegDecoder(
            out_channels=out_channels,
            img_size=cfg["img_size"],
            checkpoint_path=checkpoint_path,
        )

    backbone = ViTWrapper(model_version, checkpoint_path, freeze=False)
    decoder  = PatchTokenDecoder(
        embed_dim=cfg["embed_dim"],
        img_size=cfg["img_size"],
        out_channels=out_channels,
    )

    class _SegModel(nn.Module):
        def __init__(self, bb, dec):
            super().__init__()
            self.backbone = bb
            self.decoder  = dec

        def forward(self, x):
            out = self.backbone.encode(x) if x.shape[1] == 1 \
                  else self.backbone.encode_multi_sequence(x)
            return self.decoder(out["patches"])

    return _SegModel(backbone, decoder)
