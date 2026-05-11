"""
Preprocessing for Task 4: Trigeminal Neuralgia Segmentation (multiclass).
Modality: T2w (single sequence, high-resolution)
Label: multiclass mask {0: bg, 1: trigeminal nerve, 2: surrounding vessel}
Target spacing: 0.5mm isotropic (preserve fine anatomical detail)
"""

from __future__ import annotations
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import nibabel as nib
import numpy as np

from data.preprocess.fomo1 import _znorm


def _resample(img: nib.Nifti1Image, pixdim=(0.5, 0.5, 0.5), mode="bilinear"):
    from monai.transforms import Spacingd, Orientationd, Compose
    import torch
    arr = img.get_fdata(dtype=np.float32)[None]
    tfm = Compose([
        Orientationd(keys=["x"], axcodes="RAS"),
        Spacingd(keys=["x"], pixdim=pixdim, mode=mode),
    ])
    out = tfm({"x": torch.from_numpy(arr), "x_meta_dict": {"affine": img.affine}})
    return out["x"].numpy().squeeze(0)


def _process_subject(args):
    sid, images_dir, out_dir, labels_dir = args
    images_dir, out_dir, labels_dir = Path(images_dir), Path(out_dir), Path(labels_dir)

    t2_files = list((images_dir / sid).glob("*T2*.nii.gz")) + \
               list((images_dir / sid).glob("*t2*.nii.gz"))
    mask_files = list((labels_dir / sid).glob("*.nii.gz"))

    if not t2_files or not mask_files:
        return f"SKIP {sid}"

    try:
        safe = sid.replace(".", "_")
        arr = _znorm(_resample(nib.load(t2_files[0])))
        nib.save(nib.Nifti1Image(arr, np.eye(4)), out_dir / f"{safe}_image.nii.gz")

        msk = _resample(nib.load(mask_files[0]), mode="nearest").astype(np.uint8)
        nib.save(nib.Nifti1Image(msk, np.eye(4)), out_dir / f"{safe}_label.nii.gz")
        return f"OK {sid}"
    except Exception as e:
        return f"ERROR {sid}: {e}"


def preprocess_task4(raw_dir: str, out_dir: str, n_jobs: int = 8):
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images_dir = raw_dir / "images"
    labels_dir = raw_dir / "labels"
    subjects   = [d.name for d in images_dir.iterdir() if d.is_dir()]
    print(f"Task 4: {len(subjects)} subjects → {out_dir}")

    args = [(s, images_dir, out_dir, labels_dir) for s in subjects]
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        for r in ex.map(_process_subject, args):
            print(r)
