"""
Dataset classes for all 7 FOMO26 downstream tasks.

FOMODataset        — generic NIfTI dataset (classification, regression, seg)
FOMOMultiSeqDataset— stacks multiple modality channels (Tasks 1, 2, 3)
EmbeddingDataset   — pre-extracted embeddings for linear probing (Tasks 6, 7)

Data directory structure (after preprocessing):
  <data_dir>/<split>/
      <subject_id>_image.nii.gz        (single modality)
      <subject_id>_image_0.nii.gz      (modality 0)
      <subject_id>_image_1.nii.gz      (modality 1)
      ...
      <subject_id>_label.nii.gz        (segmentation mask)
      <subject_id>_label.txt           (classification / regression value)
"""

from __future__ import annotations
import csv
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_label_txt(path: str) -> float:
    with open(path) as f:
        return float(f.read().strip())


def _load_nifti(path: str) -> np.ndarray:
    img = nib.load(path)
    return img.get_fdata(dtype=np.float32)


# ---------------------------------------------------------------------------
# Single-modality dataset (Tasks 3, 4, 5)
# ---------------------------------------------------------------------------

class FOMODataset(Dataset):
    """
    Expects:
      data_dir/<subject_id>_image.nii.gz
      data_dir/<subject_id>_label.nii.gz   (segmentation)
      data_dir/<subject_id>_label.txt      (classification / regression)

    Args:
        data_dir   : directory containing preprocessed .npy or NIfTI files
        task_type  : "classification" | "segmentation" | "regression"
        transform  : MONAI Compose transform applied to {"image": path, "label": path}
        subjects   : optional list of subject IDs to restrict dataset
    """

    def __init__(
        self,
        data_dir: str,
        task_type: str,
        transform: Optional[Callable] = None,
        subjects: Optional[List[str]] = None,
    ):
        self.data_dir  = Path(data_dir)
        self.task_type = task_type
        self.transform = transform
        self.is_seg    = task_type == "segmentation"

        if subjects is not None:
            self.subjects = subjects
        else:
            # Auto-discover subject IDs from _image.nii.gz files
            self.subjects = sorted(
                p.name.replace("_image.nii.gz", "")
                for p in self.data_dir.glob("*_image.nii.gz")
            )

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx: int) -> dict:
        sid = self.subjects[idx]
        img_path = str(self.data_dir / f"{sid}_image.nii.gz")

        sample: Dict = {"image": img_path, "subject_id": sid}

        if self.is_seg:
            sample["label"] = str(self.data_dir / f"{sid}_label.nii.gz")
        else:
            lbl_path = self.data_dir / f"{sid}_label.txt"
            sample["label"] = _read_label_txt(str(lbl_path))

        if self.transform:
            sample = self.transform(sample)

        if not self.is_seg:
            sample["label"] = torch.tensor(sample["label"], dtype=torch.float32)

        return sample


# ---------------------------------------------------------------------------
# Multi-modality dataset (Tasks 1, 2, 3 with multi-sequence MRI)
# ---------------------------------------------------------------------------

class FOMOMultiSeqDataset(Dataset):
    """
    Stacks N modalities into a single tensor of shape (N, H, W, D).

    Expects files:
      data_dir/<subject_id>_image_0.nii.gz
      data_dir/<subject_id>_image_1.nii.gz
      ...
      data_dir/<subject_id>_label.txt  OR  _label.nii.gz
    """

    def __init__(
        self,
        data_dir: str,
        n_modalities: int,
        task_type: str,
        transform: Optional[Callable] = None,
        subjects: Optional[List[str]] = None,
    ):
        self.data_dir    = Path(data_dir)
        self.n_mod       = n_modalities
        self.task_type   = task_type
        self.transform   = transform
        self.is_seg      = task_type == "segmentation"

        if subjects is not None:
            self.subjects = subjects
        else:
            self.subjects = sorted(
                p.name.replace("_image_0.nii.gz", "")
                for p in self.data_dir.glob("*_image_0.nii.gz")
            )

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx: int) -> dict:
        sid = self.subjects[idx]
        img_paths = [
            str(self.data_dir / f"{sid}_image_{i}.nii.gz")
            for i in range(self.n_mod)
        ]

        # Build MONAI-style dict with per-modality keys
        sample: Dict = {f"image_{i}": img_paths[i] for i in range(self.n_mod)}
        sample["subject_id"] = sid

        if self.is_seg:
            sample["label"] = str(self.data_dir / f"{sid}_label.nii.gz")
        else:
            lbl_val = _read_label_txt(
                str(self.data_dir / f"{sid}_label.txt")
            )
            sample["label"] = lbl_val

        if self.transform:
            sample = self.transform(sample)

        # Stack modality tensors → (N, H, W, D)
        imgs = torch.stack([sample[f"image_{i}"].squeeze(0) for i in range(self.n_mod)], dim=0)
        out  = {"image": imgs, "subject_id": sid}

        if self.is_seg:
            out["label"] = sample["label"]
        else:
            out["label"] = torch.tensor(sample["label"], dtype=torch.float32)

        return out


# ---------------------------------------------------------------------------
# Embedding dataset for linear probing
# ---------------------------------------------------------------------------

class EmbeddingDataset(Dataset):
    """
    Loads pre-extracted embeddings from a .npz file produced by
    models/lit_linear_probe.py::extract_embeddings().

    Each item: {"embed": (D,), "label": scalar or (C,), "subject_id": str}
    """

    def __init__(self, npz_path: str, group_csv: Optional[str] = None):
        data = np.load(npz_path, allow_pickle=True)
        self.embeddings  = torch.from_numpy(data["embeddings"]).float()
        self.labels      = torch.from_numpy(data["labels"])
        self.subject_ids = data.get("subject_ids", np.array([""] * len(self.embeddings)))
        self.groups      = None

        if group_csv:
            self.groups = self._load_groups(group_csv)

    @staticmethod
    def _load_groups(csv_path: str) -> Dict[str, str]:
        groups = {}
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                groups[row["subject_id"]] = row["group"]
        return groups

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx: int) -> dict:
        item = {
            "embed":      self.embeddings[idx],
            "label":      self.labels[idx],
            "subject_id": str(self.subject_ids[idx]),
        }
        if self.groups is not None:
            item["group"] = self.groups.get(str(self.subject_ids[idx]), "unknown")
        return item


# ---------------------------------------------------------------------------
# Utility: build train/val DataLoaders from a task config
# ---------------------------------------------------------------------------

def make_loaders(
    data_dir: str,
    task_cfg,
    model_version: str,
    batch_size: int = 2,
    num_workers: int = 4,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple:
    """
    Returns (train_loader, val_loader) for a given task config.
    Splits subjects into train/val using val_fraction.
    """
    from data.transforms import get_transforms_for_task  # noqa: PLC0415
    from torch.utils.data import DataLoader, random_split  # noqa: PLC0415

    n_mod   = len(task_cfg.modalities)
    is_seg  = task_cfg.task_type == "segmentation"
    is_lp   = task_cfg.task_type == "linear_probe"

    tr_tfm = get_transforms_for_task(task_cfg, model_version, "train")
    vl_tfm = get_transforms_for_task(task_cfg, model_version, "val")

    ds_cls = FOMOMultiSeqDataset if n_mod > 1 else FOMODataset
    kw     = {"n_modalities": n_mod} if n_mod > 1 else {}

    full_ds = ds_cls(
        data_dir=data_dir,
        task_type=task_cfg.task_type,
        transform=tr_tfm,
        **kw,
    )

    n_val  = max(1, int(len(full_ds) * val_fraction))
    n_tr   = len(full_ds) - n_val
    g      = torch.Generator().manual_seed(seed)
    tr_ds, vl_ds = random_split(full_ds, [n_tr, n_val], generator=g)

    # Apply deterministic transforms to val split
    vl_ds.dataset.transform = vl_tfm  # type: ignore[attr-defined]

    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                       num_workers=num_workers, pin_memory=True, drop_last=True)
    vl_dl = DataLoader(vl_ds, batch_size=1, shuffle=False,
                       num_workers=num_workers, pin_memory=True)

    return tr_dl, vl_dl
