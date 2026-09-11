"""Task-aware metric computation utilities for BC/MCC/MLC/SEG tasks."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from .TaskSpec import TaskSpec, TaskType
from ._utils import to_numpy


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _preds_for_mcc(y_pred_np: np.ndarray, threshold: float) -> np.ndarray:
    if y_pred_np.ndim == 1:
        scores = y_pred_np.astype(float)
        if not np.all((scores >= 0.0) & (scores <= 1.0)):
            scores = _sigmoid(scores)
        return (scores >= threshold).astype(int)
    if y_pred_np.ndim == 2 and y_pred_np.shape[1] == 1:
        scores = y_pred_np[:, 0].astype(float)
        if not np.all((scores >= 0.0) & (scores <= 1.0)):
            scores = _sigmoid(scores)
        return (scores >= threshold).astype(int)
    return np.argmax(y_pred_np, axis=1)


def _preds_for_mlc(y_pred_np: np.ndarray, threshold: float) -> np.ndarray:
    if y_pred_np.ndim == 1:
        y_pred_np = y_pred_np[:, None]
    scores = y_pred_np.astype(float)
    if not np.all((scores >= 0.0) & (scores <= 1.0)):
        scores = _sigmoid(scores)
    return (scores >= threshold).astype(int)


def _seg_logits_to_mask(y_pred_np: np.ndarray, spec: TaskSpec) -> np.ndarray:
    # Expected shapes:
    # binary: (N,H,W) or (N,1,H,W) or (N,H,W,1)
    # multiclass: (N,C,H,W) or (N,H,W,C)
    if y_pred_np.ndim == 3:
        scores = y_pred_np.astype(float)
        if not np.all((scores >= 0.0) & (scores <= 1.0)):
            scores = _sigmoid(scores)
        return (scores >= spec.segmentation_threshold).astype(int)

    if y_pred_np.ndim != 4:
        raise ValueError(f"Unsupported segmentation prediction shape: {y_pred_np.shape}")

    if y_pred_np.shape[1] == 1:
        scores = y_pred_np[:, 0].astype(float)
        if not np.all((scores >= 0.0) & (scores <= 1.0)):
            scores = _sigmoid(scores)
        return (scores >= spec.segmentation_threshold).astype(int)

    if y_pred_np.shape[-1] == 1:
        scores = y_pred_np[..., 0].astype(float)
        if not np.all((scores >= 0.0) & (scores <= 1.0)):
            scores = _sigmoid(scores)
        return (scores >= spec.segmentation_threshold).astype(int)

    # Multi-class segmentation logits/proba
    if spec.class_axis == 1:
        return np.argmax(y_pred_np, axis=1)
    return np.argmax(y_pred_np, axis=-1)


def _dice_binary(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    y_true_f = y_true.reshape(-1).astype(np.float32)
    y_pred_f = y_pred.reshape(-1).astype(np.float32)
    inter = float(np.sum(y_true_f * y_pred_f))
    return float((2.0 * inter + eps) / (np.sum(y_true_f) + np.sum(y_pred_f) + eps))


def _iou_binary(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    y_true_f = y_true.reshape(-1).astype(np.float32)
    y_pred_f = y_pred.reshape(-1).astype(np.float32)
    inter = float(np.sum(y_true_f * y_pred_f))
    union = float(np.sum(y_true_f) + np.sum(y_pred_f) - inter)
    return float((inter + eps) / (union + eps))


def _mean_per_class_metric(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int, fn) -> float:
    vals = []
    for cls in range(num_classes):
        vals.append(fn((y_true == cls).astype(np.int32), (y_pred == cls).astype(np.int32)))
    return float(np.mean(vals))


def compute_metrics(y_true: Any, y_pred: Any, task_spec: TaskSpec) -> Dict[str, float]:
    """Compute metrics from predictions according to a :class:`TaskSpec`.

    Args:
        y_true: Ground-truth labels/tensors/arrays.
        y_pred: Predicted logits/probabilities/tensors.
        task_spec: Task configuration describing metric behavior.

    Returns:
        Dictionary of scalar metric values.
    """
    y_true_np = to_numpy(y_true)
    y_pred_np = to_numpy(y_pred)

    if task_spec.task_type in {TaskType.bc, TaskType.mcc}:
        if y_true_np.ndim >= 2 and y_true_np.shape[-1] > 1:
            y_true_1d = np.argmax(y_true_np, axis=-1).reshape(-1)
        else:
            y_true_1d = y_true_np.reshape(-1)
        y_hat = _preds_for_mcc(y_pred_np, task_spec.classification_threshold)
        return {
            "accuracy": float(accuracy_score(y_true_1d, y_hat)),
            "precision": float(precision_score(y_true_1d, y_hat, average=task_spec.average, zero_division=0)),
            "recall": float(recall_score(y_true_1d, y_hat, average=task_spec.average, zero_division=0)),
            "f1": float(f1_score(y_true_1d, y_hat, average=task_spec.average, zero_division=0)),
        }

    if task_spec.task_type == TaskType.mlc:
        y_hat = _preds_for_mlc(y_pred_np, task_spec.multilabel_threshold)
        y_true_2d = y_true_np if y_true_np.ndim == 2 else y_true_np[:, None]
        y_true_2d = y_true_2d.astype(int)

        subset_acc = float(np.mean(np.all(y_true_2d == y_hat, axis=1)))
        return {
            "subset_accuracy": subset_acc,
            "precision_micro": float(precision_score(y_true_2d, y_hat, average="micro", zero_division=0)),
            "recall_micro": float(recall_score(y_true_2d, y_hat, average="micro", zero_division=0)),
            "f1_micro": float(f1_score(y_true_2d, y_hat, average="micro", zero_division=0)),
            "f1_macro": float(f1_score(y_true_2d, y_hat, average="macro", zero_division=0)),
        }

    if task_spec.task_type == TaskType.seg:
        y_true_mask = y_true_np
        y_pred_mask = _seg_logits_to_mask(y_pred_np, task_spec)

        if y_true_mask.shape != y_pred_mask.shape:
            raise ValueError(
                f"Segmentation shape mismatch: y_true {y_true_mask.shape} vs y_pred {y_pred_mask.shape}"
            )

        pixel_acc = float(np.mean(y_true_mask.reshape(-1) == y_pred_mask.reshape(-1)))

        if task_spec.num_classes <= 2:
            return {
                "pixel_accuracy": pixel_acc,
                "dice": _dice_binary(y_true_mask, y_pred_mask),
                "iou": _iou_binary(y_true_mask, y_pred_mask),
            }

        return {
            "pixel_accuracy": pixel_acc,
            "dice_mean": _mean_per_class_metric(y_true_mask, y_pred_mask, task_spec.num_classes, _dice_binary),
            "iou_mean": _mean_per_class_metric(y_true_mask, y_pred_mask, task_spec.num_classes, _iou_binary),
        }

    raise ValueError(f"Unsupported task type: {task_spec.task_type}")
