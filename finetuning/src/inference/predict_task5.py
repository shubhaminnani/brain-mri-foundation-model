"""
Inference for Task 5: Polymicrogyria Classification (few-shot binary, AUROC).
Output: <output_dir>/<subject_id>.txt  — probability of class 1 (polymicrogyria)
"""

from __future__ import annotations
from pathlib import Path

import torch
import torch.nn.functional as F
import nibabel as nib
import numpy as np

from backbone.vit_wrapper import ViTWrapper, CONFIGS
from models.lit_cls import LitClassification
from data.preprocess.fomo1 import _znorm, _resample_to_1mm


def predict(args):
    device   = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg      = CONFIGS[args.model]
    img_size = cfg["img_size"]
    D        = cfg["embed_dim"]

    backbone  = ViTWrapper(args.model, args.checkpoint, freeze=True).to(device).eval()
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

    subjects = [d for d in input_dir.iterdir() if d.is_dir()]
    print(f"[Task5] Predicting {len(subjects)} subjects...")

    with torch.no_grad():
        for subj in subjects:
            t1_files = list(subj.glob("*T1*.nii.gz")) + list(subj.glob("*t1*.nii.gz"))
            if not t1_files:
                print(f"  SKIP {subj.name}")
                continue

            arr = _znorm(_resample_to_1mm(nib.load(t1_files[0])).get_fdata(dtype=np.float32))
            t   = torch.from_numpy(arr)[None, None]
            x   = F.interpolate(t, size=(img_size,)*3, mode="trilinear",
                                  align_corners=False).to(device)

            logits = lit_model(x)
            prob   = F.softmax(logits, dim=1)[0, 1].item()
            (output_dir / f"{subj.name}.txt").write_text(f"{prob:.6f}\n")

    print(f"[Task5] Done → {output_dir}")
