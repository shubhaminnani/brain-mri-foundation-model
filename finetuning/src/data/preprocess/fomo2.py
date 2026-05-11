"""
Preprocessing for Task 2: Meningioma Segmentation.
Modalities: DWI, T2FLAIR, SWI_OR_T2STAR
Label: binary segmentation mask .nii.gz

Pipeline: RAS → 1mm → znorm per volume → save modalities + mask
"""

from __future__ import annotations
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import nibabel as nib
import numpy as np

from data.preprocess.fomo1 import _znorm, _resample_to_1mm  # shared utils


_MODALITY_MAP = {"dwi": 0, "flair": 1, "swi": 2, "t2s": 2}


def _process_subject(args):
    sid, images_dir, out_dir, labels_dir = args
    images_dir = Path(images_dir)
    out_dir = Path(out_dir)
    labels_dir = Path(labels_dir)

    ses = images_dir / sid / "ses_1"
    if not ses.is_dir():
        return f"SKIP {sid}"

    mask_file = labels_dir / sid / "ses_1" / "mask.nii.gz"
    if not mask_file.exists():
        # Try alternate name
        candidates = list((labels_dir / sid / "ses_1").glob("*.nii.gz"))
        mask_file = candidates[0] if candidates else None
    if not mask_file:
        return f"SKIP {sid}: no mask"

    modality_imgs = {}
    for f in ses.glob("*.nii.gz"):
        name = f.stem.lower()
        for key, idx in _MODALITY_MAP.items():
            if key in name and idx not in modality_imgs:
                modality_imgs[idx] = f
                break

    if len(modality_imgs) < 3:
        return f"SKIP {sid}: {len(modality_imgs)}/3 modalities"

    try:
        safe = sid.replace(".", "_")
        for i in range(3):
            img_r = _resample_to_1mm(nib.load(modality_imgs[i]))
            arr = _znorm(img_r.get_fdata(dtype=np.float32))
            nib.save(nib.Nifti1Image(arr, np.eye(4)), out_dir / f"{safe}_image_{i}.nii.gz")

        # Resample mask with nearest-neighbour
        from monai.transforms import Spacingd, Orientationd, Compose
        import torch
        msk = nib.load(mask_file)
        msk_arr = msk.get_fdata(dtype=np.float32)[None]
        tfm = Compose([
            Orientationd(keys=["mask"], axcodes="RAS"),
            Spacingd(keys=["mask"], pixdim=(1.0, 1.0, 1.0), mode="nearest"),
        ])
        out_msk = tfm({"mask": torch.from_numpy(msk_arr),
                       "mask_meta_dict": {"affine": msk.affine}})
        msk_out = out_msk["mask"].numpy().squeeze(0)
        nib.save(nib.Nifti1Image(msk_out.astype(np.uint8), np.eye(4)),
                 out_dir / f"{safe}_label.nii.gz")
        return f"OK {sid}"
    except Exception as e:
        return f"ERROR {sid}: {e}"


def preprocess_task2(raw_dir: str, out_dir: str, n_jobs: int = 8):
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images_dir  = raw_dir / "preprocessed"
    labels_dir  = raw_dir / "labels"
    subjects = [d.name for d in images_dir.iterdir() if d.is_dir()]
    print(f"Task 2: {len(subjects)} subjects → {out_dir}")

    args = [(s, images_dir, out_dir, labels_dir) for s in subjects]
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        for r in ex.map(_process_subject, args):
            print(r)
