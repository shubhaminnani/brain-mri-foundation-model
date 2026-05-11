"""
Inference for Task 4: Trigeminal Neuralgia Segmentation (multiclass, DSC+NSD).
Output: <output_dir>/<subject_id>_pred.nii.gz  — 0/1/2 label map
"""

from __future__ import annotations
from pathlib import Path

import torch
import torch.nn.functional as F
import nibabel as nib
import numpy as np
from monai.inferers import sliding_window_inference

from models.seg_decoder import build_seg_model
from models.lit_seg import LitSegmentation
from backbone.vit_wrapper import CONFIGS
from data.preprocess.fomo1 import _znorm
from data.preprocess.fomo4 import _resample


def predict(args):
    device   = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg      = CONFIGS[args.model]
    img_size = cfg["img_size"]

    seg_model = build_seg_model(args.model, out_channels=3, checkpoint_path=args.checkpoint)
    lit_model = LitSegmentation.load_from_checkpoint(
        args.finetune_ckpt,
        seg_model=seg_model,
        img_size=img_size,
        num_classes=3,
        map_location=device,
    )
    lit_model.eval().to(device)

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    subjects   = [d for d in input_dir.iterdir() if d.is_dir()]

    print(f"[Task4] Predicting {len(subjects)} subjects...")
    with torch.no_grad():
        for subj in subjects:
            t2_files = list(subj.glob("*T2*.nii.gz")) + list(subj.glob("*t2*.nii.gz"))
            if not t2_files:
                continue
            orig = nib.load(t2_files[0])
            arr  = _znorm(_resample(orig, pixdim=(0.5, 0.5, 0.5)))
            t    = torch.from_numpy(arr)[None, None]
            x    = F.interpolate(t, size=(img_size,)*3, mode="trilinear",
                                  align_corners=False).to(device)

            logits = sliding_window_inference(
                x, roi_size=(img_size,)*3, sw_batch_size=1,
                predictor=lit_model.model
            )
            pred = logits.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)
            nib.save(nib.Nifti1Image(pred, orig.affine, orig.header),
                     output_dir / f"{subj.name}_pred.nii.gz")

    print(f"[Task4] Done → {output_dir}")
