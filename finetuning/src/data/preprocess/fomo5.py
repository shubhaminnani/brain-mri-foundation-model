"""
Preprocessing for Task 5: Polymicrogyria Classification (few-shot binary).
Modality: T1w (single sequence)
Label: binary .txt (0=negative, 1=polymicrogyria)

Same pipeline as Task 1 but single modality.
"""

from __future__ import annotations
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import nibabel as nib
import numpy as np

from data.preprocess.fomo1 import _znorm, _resample_to_1mm


def _process_subject(args):
    sid, images_dir, out_dir, labels_dir = args
    images_dir, out_dir, labels_dir = Path(images_dir), Path(out_dir), Path(labels_dir)

    t1_files = (list((images_dir / sid).glob("*T1*.nii.gz")) +
                list((images_dir / sid).glob("*t1*.nii.gz")))
    lbl = labels_dir / sid / "label.txt"

    if not t1_files or not lbl.exists():
        return f"SKIP {sid}"

    try:
        safe = sid.replace(".", "_")
        arr = _znorm(_resample_to_1mm(nib.load(t1_files[0])).get_fdata(dtype=np.float32))
        nib.save(nib.Nifti1Image(arr, np.eye(4)), out_dir / f"{safe}_image.nii.gz")
        shutil.copy(lbl, out_dir / f"{safe}_label.txt")
        return f"OK {sid}"
    except Exception as e:
        return f"ERROR {sid}: {e}"


def preprocess_task5(raw_dir: str, out_dir: str, n_jobs: int = 8):
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images_dir = raw_dir / "images"
    labels_dir = raw_dir / "labels"
    subjects   = [d.name for d in images_dir.iterdir() if d.is_dir()]
    print(f"Task 5: {len(subjects)} subjects → {out_dir}")

    args = [(s, images_dir, out_dir, labels_dir) for s in subjects]
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        for r in ex.map(_process_subject, args):
            print(r)
