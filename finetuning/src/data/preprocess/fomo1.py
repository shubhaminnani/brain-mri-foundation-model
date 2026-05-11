"""
Preprocessing for Task 1: Infarct Classification.
Modalities: DWI, T2FLAIR, ADC, SWI_OR_T2STAR
Label: binary .txt (0=negative, 1=positive)

Pipeline per subject:
  1. Locate all 4 modality NIfTIs in ses_1/
  2. Yucca: RAS reorientation → 1mm isotropic → z-norm per volume → crop-to-nonzero
  3. Save each modality as <sid>_image_<i>.nii.gz
  4. Copy label.txt as <sid>_label.txt
"""

from __future__ import annotations
import os
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import nibabel as nib
import numpy as np


_MODALITY_MAP = {
    "dwi":  0,
    "flair": 1,
    "adc":  2,
    "swi":  3,
    "t2s":  3,
}


def _znorm(arr: np.ndarray) -> np.ndarray:
    mask = arr != 0
    if mask.sum() == 0:
        return arr
    mu, sigma = arr[mask].mean(), arr[mask].std()
    arr = (arr - mu) / (sigma + 1e-8)
    arr[~mask] = 0.0
    return arr.astype(np.float32)


def _crop_nonzero(arr: np.ndarray):
    nonzero = np.argwhere(arr != 0)
    if len(nonzero) == 0:
        return arr, (slice(None),) * arr.ndim
    mins = nonzero.min(axis=0)
    maxs = nonzero.max(axis=0) + 1
    slices = tuple(slice(mn, mx) for mn, mx in zip(mins, maxs))
    return arr[slices], slices


def _resample_to_1mm(img: nib.Nifti1Image) -> nib.Nifti1Image:
    """Resample to 1mm isotropic using MONAI Spacing."""
    from monai.transforms import Spacingd, Orientationd, EnsureChannelFirstd, Compose
    import torch
    data = img.get_fdata(dtype=np.float32)[None]  # (1, H, W, D)
    tfm = Compose([
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
    ])
    out = tfm({"image": torch.from_numpy(data), "image_meta_dict": {"affine": img.affine}})
    arr = out["image"].numpy().squeeze(0)
    return nib.Nifti1Image(arr, np.eye(4))


def _process_subject(args):
    sid, raw_dir, out_dir, labels_dir = args
    raw_dir, out_dir, labels_dir = Path(raw_dir), Path(out_dir), Path(labels_dir)
    ses = raw_dir / sid / "ses_1"
    if not ses.is_dir():
        return f"SKIP {sid}: no ses_1"

    lbl_file = labels_dir / sid / "ses_1" / "label.txt"
    if not lbl_file.exists():
        return f"SKIP {sid}: no label.txt"

    modality_imgs = {}
    for f in ses.glob("*.nii.gz"):
        name = f.stem.lower()
        for key, idx in _MODALITY_MAP.items():
            if key in name and idx not in modality_imgs:
                modality_imgs[idx] = f
                break

    if len(modality_imgs) < 4:
        return f"SKIP {sid}: only {len(modality_imgs)}/4 modalities"

    try:
        for i in range(4):
            img = nib.load(modality_imgs[i])
            img_r = _resample_to_1mm(img)
            arr = _znorm(img_r.get_fdata(dtype=np.float32))
            arr_c, _ = _crop_nonzero(arr)
            out_img = nib.Nifti1Image(arr_c, np.eye(4))
            safe = sid.replace(".", "_")
            nib.save(out_img, out_dir / f"{safe}_image_{i}.nii.gz")

        safe = sid.replace(".", "_")
        shutil.copy(lbl_file, out_dir / f"{safe}_label.txt")
        return f"OK {sid}"
    except Exception as e:
        return f"ERROR {sid}: {e}"


def preprocess_task1(raw_dir: str, out_dir: str, n_jobs: int = 8):
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels_dir = raw_dir / "labels"
    images_dir = raw_dir / "preprocessed"

    subjects = [d.name for d in images_dir.iterdir() if d.is_dir()]
    print(f"Task 1: {len(subjects)} subjects → {out_dir}")

    args = [(s, images_dir, out_dir, labels_dir) for s in subjects]
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        for result in ex.map(_process_subject, args):
            print(result)
