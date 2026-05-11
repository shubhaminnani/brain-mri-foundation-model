"""
Inference for Task 1: Infarct Classification (binary, AUROC).
Input:  subject folders with DWI / FLAIR / ADC / SWI NIfTIs
Output: <output_dir>/<subject_id>.txt  containing probability of class 1
"""

from __future__ import annotations
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import nibabel as nib
import numpy as np

from backbone.vit_wrapper import ViTWrapper, CONFIGS
from models.lit_cls import LitClassification
from data.preprocess.fomo1 import _znorm, _resample_to_1mm
from monai.transforms import Resized, ToTensord, Compose, EnsureChannelFirstd


_MODALITY_KEYS = ["dwi", "flair", "adc", "swi"]


def _find_modalities(subject_dir: Path) -> dict:
    files = {}
    for f in sorted(subject_dir.glob("**/*.nii.gz")):
        name = f.name.lower()
        for key in _MODALITY_KEYS:
            if key in name and key not in files:
                files[key] = f
    return files


def _load_and_preprocess(path: Path, img_size: int) -> torch.Tensor:
    arr = _znorm(_resample_to_1mm(nib.load(path)).get_fdata(dtype=np.float32))
    t   = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1,1,H,W,D)
    t   = F.interpolate(t, size=(img_size,) * 3, mode="trilinear", align_corners=False)
    return t.squeeze(0)  # (1,H,W,D)


def predict(args):
    device   = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg      = CONFIGS[args.model]
    img_size = cfg["img_size"]
    D        = cfg["embed_dim"]

    # Load backbone
    backbone = ViTWrapper(args.model, args.checkpoint, freeze=True).to(device).eval()

    # Load finetuned classifier head
    lit_model = LitClassification.load_from_checkpoint(
        args.finetune_ckpt,
        backbone=backbone,
        embed_dim=D,
        num_classes=2,
        map_location=device,
    )
    lit_model.eval().to(device)

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    subjects   = [d for d in input_dir.iterdir() if d.is_dir()]

    print(f"[Task1] Predicting {len(subjects)} subjects...")
    with torch.no_grad():
        for subj in subjects:
            mods = _find_modalities(subj)
            tensors = []
            for key in _MODALITY_KEYS:
                if key in mods:
                    tensors.append(_load_and_preprocess(mods[key], img_size))
            if not tensors:
                print(f"  SKIP {subj.name}: no modalities found")
                continue

            x    = torch.stack(tensors, dim=0).unsqueeze(0).to(device)  # (1,C,H,W,D)
            logits = lit_model(x)
            prob   = F.softmax(logits, dim=1)[0, 1].item()

            (output_dir / f"{subj.name}.txt").write_text(f"{prob:.6f}\n")

    print(f"[Task1] Done → {output_dir}")
