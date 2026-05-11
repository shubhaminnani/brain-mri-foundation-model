"""
Preprocessing for Task 3: Brain Age Regression.
Modalities: T1, T2
Label: age as float in .txt

Pipeline: RAS → 1mm → znorm → save T1+T2 pair + age label
"""

from __future__ import annotations
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import nibabel as nib
import numpy as np

from data.preprocess.fomo1 import _znorm, _resample_to_1mm


def _process_subject(args):
    sid, images_dir, out_dir, labels_dir = args
    images_dir, out_dir, labels_dir = Path(images_dir), Path(out_dir), Path(labels_dir)

    ses = images_dir / sid / "ses_1"
    lbl = labels_dir / sid / "ses_1" / "label.txt"
    if not ses.is_dir() or not lbl.exists():
        return f"SKIP {sid}"

    t1_files = list(ses.glob("*t1*.nii.gz")) + list(ses.glob("*T1*.nii.gz"))
    t2_files = list(ses.glob("*t2*.nii.gz")) + list(ses.glob("*T2*.nii.gz"))
    if not t1_files or not t2_files:
        return f"SKIP {sid}: missing T1 or T2"

    try:
        safe = sid.replace(".", "_")
        for i, f in enumerate([t1_files[0], t2_files[0]]):
            arr = _znorm(_resample_to_1mm(nib.load(f)).get_fdata(dtype=np.float32))
            nib.save(nib.Nifti1Image(arr, np.eye(4)), out_dir / f"{safe}_image_{i}.nii.gz")
        import shutil
        shutil.copy(lbl, out_dir / f"{safe}_label.txt")
        return f"OK {sid}"
    except Exception as e:
        return f"ERROR {sid}: {e}"


def preprocess_task3(raw_dir: str, out_dir: str, n_jobs: int = 8):
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images_dir = raw_dir / "preprocessed"
    labels_dir = raw_dir / "labels"
    subjects   = [d.name for d in images_dir.iterdir() if d.is_dir()]
    print(f"Task 3: {len(subjects)} subjects → {out_dir}")

    args = [(s, images_dir, out_dir, labels_dir) for s in subjects]
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        for r in ex.map(_process_subject, args):
            print(r)
