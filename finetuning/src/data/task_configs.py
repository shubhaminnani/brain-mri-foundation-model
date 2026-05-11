"""
Task configurations for all 7 FOMO26 downstream tasks.
Each config drives preprocessing, data loading, model selection, and metrics.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class TaskConfig:
    task_id        : int
    task_name      : str
    task_type      : str          # "classification" | "segmentation" | "regression" | "linear_probe"
    modalities     : Tuple[str, ...]
    num_classes    : int
    norm_op        : str          # "volume_wise_znorm" | "channel_wise_znorm"
    target_spacing : Tuple[float, float, float]
    crop_to_nonzero: bool
    metric_primary : str          # metric used for model selection / ranking
    metrics        : List[str]
    label_extension: str          # ".txt" | ".nii.gz"
    labels         : Dict
    few_shot       : bool = False
    n_support      : int  = 0     # k in k-shot (0 = use all training data)
    multiclass_seg : bool = False


# ---------------------------------------------------------------------------
# Task 1 — Infarct Classification (few-shot binary, AUROC)
# ---------------------------------------------------------------------------
TASK1 = TaskConfig(
    task_id         = 1,
    task_name       = "Task001_InfarctClassification",
    task_type       = "classification",
    modalities      = ("DWI", "T2FLAIR", "ADC", "SWI_OR_T2STAR"),
    num_classes     = 2,
    norm_op         = "volume_wise_znorm",
    target_spacing  = (1.0, 1.0, 1.0),
    crop_to_nonzero = True,
    metric_primary  = "auroc",
    metrics         = ["auroc"],
    label_extension = ".txt",
    labels          = {0: "negative", 1: "positive"},
    few_shot        = True,
    n_support       = 21,   # 21 labelled finetuning cases
)

# ---------------------------------------------------------------------------
# Task 2 — Meningioma Segmentation (few-shot binary, DSC+NSD)
# ---------------------------------------------------------------------------
TASK2 = TaskConfig(
    task_id         = 2,
    task_name       = "Task002_MeningiomaSegmentation",
    task_type       = "segmentation",
    modalities      = ("DWI", "T2FLAIR", "SWI_OR_T2STAR"),
    num_classes     = 2,
    norm_op         = "volume_wise_znorm",
    target_spacing  = (1.0, 1.0, 1.0),
    crop_to_nonzero = True,
    metric_primary  = "dice",
    metrics         = ["dice", "nsd"],
    label_extension = ".nii.gz",
    labels          = {0: "background", 1: "meningioma"},
    few_shot        = True,
    n_support       = 23,
)

# ---------------------------------------------------------------------------
# Task 3 — Brain Age Regression (MAE + Pearson correlation)
# ---------------------------------------------------------------------------
TASK3 = TaskConfig(
    task_id         = 3,
    task_name       = "Task003_BrainAgeRegression",
    task_type       = "regression",
    modalities      = ("T1", "T2"),
    num_classes     = 1,
    norm_op         = "volume_wise_znorm",
    target_spacing  = (1.0, 1.0, 1.0),
    crop_to_nonzero = True,
    metric_primary  = "mae",
    metrics         = ["mae", "pearson_r"],
    label_extension = ".txt",
    labels          = {"regression": "age"},
    few_shot        = False,
    n_support       = 200,
)

# ---------------------------------------------------------------------------
# Task 4 — Trigeminal Neuralgia Segmentation (multiclass, DSC+NSD)
# ---------------------------------------------------------------------------
TASK4 = TaskConfig(
    task_id         = 4,
    task_name       = "Task004_TrigeminalNeuralgiaSegmentation",
    task_type       = "segmentation",
    modalities      = ("T2",),
    num_classes     = 3,   # 0=background, 1=trigeminal nerve, 2=surrounding vessel
    norm_op         = "volume_wise_znorm",
    target_spacing  = (0.5, 0.5, 0.5),  # high-resolution T2w
    crop_to_nonzero = True,
    metric_primary  = "dice",
    metrics         = ["dice", "nsd"],
    label_extension = ".nii.gz",
    labels          = {0: "background", 1: "trigeminal_nerve", 2: "vessel"},
    few_shot        = False,
    multiclass_seg  = True,
)

# ---------------------------------------------------------------------------
# Task 5 — Polymicrogyria Classification (few-shot binary, AUROC)
# ---------------------------------------------------------------------------
TASK5 = TaskConfig(
    task_id         = 5,
    task_name       = "Task005_PolymicrogyriaClassification",
    task_type       = "classification",
    modalities      = ("T1",),
    num_classes     = 2,
    norm_op         = "volume_wise_znorm",
    target_spacing  = (1.0, 1.0, 1.0),
    crop_to_nonzero = True,
    metric_primary  = "auroc",
    metrics         = ["auroc"],
    label_extension = ".txt",
    labels          = {0: "negative", 1: "polymicrogyria"},
    few_shot        = True,
    n_support       = -1,  # TBD when data released
)

# ---------------------------------------------------------------------------
# Task 6 — Linear Probing (no finetuning, OvR AUROC + F1)
# ---------------------------------------------------------------------------
TASK6 = TaskConfig(
    task_id         = 6,
    task_name       = "Task006_LinearProbing",
    task_type       = "linear_probe",
    modalities      = ("T1",),
    num_classes     = -1,  # set at runtime from label set
    norm_op         = "volume_wise_znorm",
    target_spacing  = (1.0, 1.0, 1.0),
    crop_to_nonzero = True,
    metric_primary  = "auroc_ovr",
    metrics         = ["auroc_ovr", "f1"],
    label_extension = ".txt",
    labels          = {},  # populated from data at runtime
    few_shot        = False,
)

# ---------------------------------------------------------------------------
# Task 7 — Bias & Fairness (linear probe, same setup as Task 6)
# ---------------------------------------------------------------------------
TASK7 = TaskConfig(
    task_id         = 7,
    task_name       = "Task007_BiasAndFairness",
    task_type       = "linear_probe",
    modalities      = ("T1",),
    num_classes     = -1,
    norm_op         = "volume_wise_znorm",
    target_spacing  = (1.0, 1.0, 1.0),
    crop_to_nonzero = True,
    metric_primary  = "auroc_ovr",
    metrics         = ["auroc_ovr", "f1"],
    label_extension = ".txt",
    labels          = {},
    few_shot        = False,
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
ALL_TASKS: Dict[int, TaskConfig] = {
    1: TASK1, 2: TASK2, 3: TASK3,
    4: TASK4, 5: TASK5, 6: TASK6, 7: TASK7,
}


def get_task(task_id: int) -> TaskConfig:
    if task_id not in ALL_TASKS:
        raise ValueError(f"Unknown task_id={task_id}. Valid: {list(ALL_TASKS)}")
    return ALL_TASKS[task_id]
