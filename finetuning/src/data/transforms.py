"""
MONAI transform pipelines for finetuning.

Two spatial sizes are supported:
  96  — v1 backbone (MONAI ViT-B)
  112 — v2 / v3 backbones

Train transforms include spatial + intensity augmentation.
Val/Test transforms are deterministic (load → resize → normalize).
"""

from __future__ import annotations
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    NormalizeIntensityd,
    CropForegroundd,
    Resized,
    RandAffined,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandAdjustContrastd,
    ToTensord,
    EnsureTyped,
)

_INTERP = "trilinear"


def _base_transforms(image_keys: list[str], img_size: int) -> list:
    """Shared deterministic pre-processing (load → RAS → 1mm → crop → resize → znorm)."""
    return [
        LoadImaged(keys=image_keys, ensure_channel_first=True),
        Orientationd(keys=image_keys, axcodes="RAS"),
        Spacingd(keys=image_keys, pixdim=(1.0, 1.0, 1.0), mode=_INTERP),
        CropForegroundd(keys=image_keys, source_key=image_keys[0]),
        Resized(keys=image_keys, spatial_size=(img_size,) * 3, mode=_INTERP),
        NormalizeIntensityd(keys=image_keys, nonzero=True, channel_wise=True),
    ]


def get_train_transforms(
    image_keys: list[str],
    label_key: str | None,
    img_size: int,
    is_seg: bool = False,
) -> Compose:
    """
    Returns augmented training transforms.
    For segmentation tasks (is_seg=True), the label volume is included in
    spatial transforms.
    """
    spatial_keys = image_keys + ([label_key] if is_seg and label_key else [])
    seg_interp   = ["bilinear"] * len(image_keys) + (["nearest"] if is_seg and label_key else [])

    base = _base_transforms(image_keys, img_size)

    aug = [
        RandAffined(
            keys=spatial_keys,
            mode=seg_interp if is_seg else [_INTERP] * len(image_keys),
            prob=0.5,
            rotate_range=(0.15, 0.15, 0.15),
            translate_range=(8, 8, 8),
            scale_range=(0.1, 0.1, 0.1),
            padding_mode="border",
        ),
        RandFlipd(keys=spatial_keys, spatial_axis=[0], prob=0.5),
        RandFlipd(keys=spatial_keys, spatial_axis=[2], prob=0.5),
        RandGaussianNoised(keys=image_keys, prob=0.2, std=0.05),
        RandGaussianSmoothd(keys=image_keys, prob=0.2,
                            sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5), sigma_z=(0.5, 1.5)),
        RandAdjustContrastd(keys=image_keys, prob=0.3, gamma=(0.7, 1.5)),
        EnsureTyped(keys=spatial_keys),
        ToTensord(keys=spatial_keys),
    ]

    return Compose(base + aug)


def get_val_transforms(
    image_keys: list[str],
    label_key: str | None,
    img_size: int,
    is_seg: bool = False,
) -> Compose:
    """Deterministic validation / test transforms (no augmentation)."""
    seg_keys = image_keys + ([label_key] if is_seg and label_key else [])
    base = _base_transforms(image_keys, img_size)
    tail = [
        EnsureTyped(keys=seg_keys),
        ToTensord(keys=seg_keys),
    ]
    return Compose(base + tail)


def get_transforms_for_task(
    task_cfg,
    model_version: str,
    split: str = "train",
) -> Compose:
    """
    Convenience wrapper: builds transforms from a TaskConfig.

    Args:
        task_cfg     : TaskConfig from task_configs.py
        model_version: "v1" | "v2" | "v3"  (determines img_size)
        split        : "train" | "val" | "test"
    """
    from backbone.vit_wrapper import CONFIGS  # noqa: PLC0415

    img_size = CONFIGS[model_version]["img_size"]
    is_seg   = task_cfg.task_type == "segmentation"
    n_mod    = len(task_cfg.modalities)
    img_keys = [f"image_{i}" for i in range(n_mod)] if n_mod > 1 else ["image"]
    lbl_key  = "label" if is_seg else None

    if split == "train":
        return get_train_transforms(img_keys, lbl_key, img_size, is_seg)
    return get_val_transforms(img_keys, lbl_key, img_size, is_seg)
