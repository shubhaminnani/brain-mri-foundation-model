"""
Inference for Task 3: Brain Age Regression (MAE + Pearson correlation).
Output: <output_dir>/<subject_id>.txt  — predicted age (float)
"""

from __future__ import annotations
from pathlib import Path

import torch
import torch.nn.functional as F
import nibabel as nib
import numpy as np

from backbone.vit_wrapper import ViTWrapper, CONFIGS
from models.lit_reg import LitRegression
from data.preprocess.fomo1 import _znorm, _resample_to_1mm


def _load_vol(path: Path, img_size: int) -> torch.Tensor:
    arr = _znorm(_resample_to_1mm(nib.load(path)).get_fdata(dtype=np.float32))
    t   = torch.from_numpy(arr)[None, None]
    return F.interpolate(t, size=(img_size,) * 3, mode="trilinear", align_corners=False).squeeze(0)


def predict(args):
    device   = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg      = CONFIGS[args.model]
    img_size = cfg["img_size"]
    D        = cfg["embed_dim"]

    backbone = ViTWrapper(args.model, args.checkpoint, freeze=True).to(device).eval()
    lit_model = LitRegression.load_from_checkpoint(
        args.finetune_ckpt,
        backbone=backbone,
        embed_dim=D,
        map_location=device,
    )
    lit_model.eval().to(device)

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    subjects   = [d for d in input_dir.iterdir() if d.is_dir()]

    print(f"[Task3] Predicting {len(subjects)} subjects...")
    with torch.no_grad():
        for subj in subjects:
            t1_files = list(subj.glob("*T1*.nii.gz")) + list(subj.glob("*t1*.nii.gz"))
            t2_files = list(subj.glob("*T2*.nii.gz")) + list(subj.glob("*t2*.nii.gz"))
            if not t1_files or not t2_files:
                print(f"  SKIP {subj.name}")
                continue

            t1 = _load_vol(t1_files[0], img_size)
            t2 = _load_vol(t2_files[0], img_size)
            x  = torch.stack([t1, t2], dim=0).unsqueeze(0).to(device)  # (1,2,H,W,D)

            age_pred = lit_model(x).item()
            (output_dir / f"{subj.name}.txt").write_text(f"{age_pred:.4f}\n")

    print(f"[Task3] Done → {output_dir}")
