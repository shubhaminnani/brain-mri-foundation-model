"""
Inference for Task 2: Meningioma Segmentation (binary, DSC+NSD).
Output: <output_dir>/<subject_id>_pred.nii.gz  — binary mask in original space
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
from data.preprocess.fomo1 import _znorm, _resample_to_1mm


_MODALITY_KEYS = ["dwi", "flair", "swi"]


def _find_modalities(subject_dir: Path) -> dict:
    files = {}
    for f in sorted(subject_dir.glob("**/*.nii.gz")):
        name = f.name.lower()
        for key in _MODALITY_KEYS:
            if key in name and key not in files:
                files[key] = f
    return files


def _load_vol(path: Path, img_size: int) -> torch.Tensor:
    arr = _znorm(_resample_to_1mm(nib.load(path)).get_fdata(dtype=np.float32))
    t   = torch.from_numpy(arr)[None, None]
    return F.interpolate(t, size=(img_size,) * 3, mode="trilinear", align_corners=False).squeeze(0)


def predict(args):
    device   = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg      = CONFIGS[args.model]
    img_size = cfg["img_size"]

    seg_model = build_seg_model(args.model, out_channels=2, checkpoint_path=args.checkpoint)
    lit_model = LitSegmentation.load_from_checkpoint(
        args.finetune_ckpt,
        seg_model=seg_model,
        img_size=img_size,
        num_classes=2,
        map_location=device,
    )
    lit_model.eval().to(device)

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    subjects   = [d for d in input_dir.iterdir() if d.is_dir()]

    print(f"[Task2] Predicting {len(subjects)} subjects...")
    with torch.no_grad():
        for subj in subjects:
            orig_img = nib.load(next(subj.glob("**/*.nii.gz")))
            mods = _find_modalities(subj)
            tensors = [_load_vol(mods[k], img_size) for k in _MODALITY_KEYS if k in mods]
            if not tensors:
                continue
            x = torch.stack(tensors, dim=0).unsqueeze(0).to(device)

            logits = sliding_window_inference(
                x, roi_size=(img_size,) * 3, sw_batch_size=1,
                predictor=lit_model.model
            )
            pred = logits.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)

            # Resample back to original space
            pred_img = nib.Nifti1Image(pred, orig_img.affine, orig_img.header)
            nib.save(pred_img, output_dir / f"{subj.name}_pred.nii.gz")

    print(f"[Task2] Done → {output_dir}")
