"""
Standalone metric computation for final evaluation outputs.
Used by inference scripts to produce per-task score dicts.
"""

from __future__ import annotations
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    mean_absolute_error,
)
from scipy.stats import pearsonr


def compute_auroc(y_true: np.ndarray, y_prob: np.ndarray, multi_class: bool = False) -> float:
    """Binary or OvR multi-class AUROC."""
    if multi_class:
        return float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
    return float(roc_auc_score(y_true, y_prob))


def compute_f1(y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro") -> float:
    return float(f1_score(y_true, y_pred, average=average, zero_division=0))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def compute_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    r, _ = pearsonr(y_true.flatten(), y_pred.flatten())
    return float(r)


def compute_dice(pred_bin: np.ndarray, gt_bin: np.ndarray, smooth: float = 1e-6) -> float:
    """Volumetric DSC between two binary arrays."""
    inter = (pred_bin * gt_bin).sum()
    return float((2.0 * inter + smooth) / (pred_bin.sum() + gt_bin.sum() + smooth))


def compute_nsd(
    pred_bin: np.ndarray,
    gt_bin: np.ndarray,
    spacing_mm: tuple = (1.0, 1.0, 1.0),
    tolerance_mm: float = 2.0,
) -> float:
    """
    Normalized Surface Distance (NSD) at tolerance_mm.
    Requires skimage or SimpleITK for surface extraction.
    """
    try:
        from monai.metrics import SurfaceDistanceMetric
        import torch
        metric = SurfaceDistanceMetric(include_background=False, reduction="mean")
        p = torch.from_numpy(pred_bin[None, None].astype(np.float32))
        g = torch.from_numpy(gt_bin[None, None].astype(np.float32))
        metric(p, g)
        return float(metric.aggregate())
    except Exception:
        return float("nan")


def task_scores(task_id: int, predictions: dict) -> dict:
    """
    Compute all relevant metrics for a task given a predictions dict.

    predictions must contain:
      "y_true": np.ndarray
      "y_prob" or "y_pred": np.ndarray  (depending on task)
    """
    y_true = predictions["y_true"]
    scores = {}

    if task_id in (1, 5):
        y_prob = predictions["y_prob"]
        scores["auroc"] = compute_auroc(y_true, y_prob[:, 1] if y_prob.ndim == 2 else y_prob)

    elif task_id in (2, 4):
        y_pred = predictions["y_pred"]
        y_gt   = predictions["y_gt"]
        for c in range(1, y_pred.shape[1] if y_pred.ndim > 1 else 2):
            p = (y_pred == c).astype(np.float32) if y_pred.ndim == 1 else y_pred[:, c]
            g = (y_gt   == c).astype(np.float32) if y_gt.ndim == 1   else y_gt[:, c]
            scores[f"dice_class{c}"] = compute_dice(p, g)
        scores["dice_mean"] = float(np.mean([v for k, v in scores.items() if "dice" in k]))

    elif task_id == 3:
        y_pred = predictions["y_pred"]
        scores["mae"]       = compute_mae(y_true, y_pred)
        scores["pearson_r"] = compute_pearson(y_true, y_pred)

    elif task_id in (6, 7):
        y_prob = predictions["y_prob"]
        y_pred = y_prob.argmax(axis=-1) if y_prob.ndim > 1 else (y_prob > 0.5).astype(int)
        scores["auroc_ovr"] = compute_auroc(y_true, y_prob, multi_class=True)
        scores["f1"]        = compute_f1(y_true, y_pred)

        if task_id == 7 and "groups" in predictions:
            for grp in np.unique(predictions["groups"]):
                mask = predictions["groups"] == grp
                if mask.sum() < 2:
                    continue
                scores[f"auroc_{grp}"] = compute_auroc(
                    y_true[mask], y_prob[mask], multi_class=(y_prob.shape[-1] > 2)
                )
                scores[f"f1_{grp}"] = compute_f1(y_true[mask], y_pred[mask])

    return scores
