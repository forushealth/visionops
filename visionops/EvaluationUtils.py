"""Unified evaluation helpers for keras and torch backends."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .MetricRegistry import compute_metrics
from .TaskSpec import TaskSpec


def evaluate_classification(
    model: Any,
    backend: str,
    x: Optional[Any] = None,
    y: Optional[Any] = None,
    dataloader: Optional[Any] = None,
    dataset: Optional[Any] = None,
    threshold: float = 0.5,
    average: str = "weighted",
    task_type: str = "mcc",
    num_classes: int = 2,
    multilabel_threshold: float = 0.5,
) -> Dict[str, float]:
    """Evaluate a classification model with task-aware defaults.

    This is a convenience wrapper around :func:`evaluate_task` that builds a
    :class:`TaskSpec` for BC/MCC/MLC use cases.
    """
    task = str(task_type).lower().strip()
    task_spec = TaskSpec(
        task_type=task,
        num_classes=int(num_classes),
        classification_threshold=threshold,
        multilabel_threshold=multilabel_threshold,
        average=average,
    )
    return evaluate_task(
        model=model,
        backend=backend,
        task_spec=task_spec,
        x=x,
        y=y,
        dataloader=dataloader,
        dataset=dataset,
    )


def evaluate_task(
    model: Any,
    backend: str,
    task_spec: TaskSpec,
    x: Optional[Any] = None,
    y: Optional[Any] = None,
    dataloader: Optional[Any] = None,
    dataset: Optional[Any] = None,
) -> Dict[str, float]:
    """Evaluate a model for keras or torch and compute task-specific metrics.

    Args:
        model: Loaded keras model or torch module.
        backend: Backend name, either ``keras`` or ``torch``.
        task_spec: Task metadata describing BC/MCC/MLC/SEG settings.
        x: Keras features array/tensor when not using ``dataset``.
        y: Keras labels array/tensor when not using ``dataset``.
        dataloader: Torch dataloader yielding ``(inputs, labels)``.
        dataset: Keras dataset yielding ``(inputs, labels)``.

    Returns:
        Metric dictionary from :func:`compute_metrics`.
    """
    backend = backend.lower().strip()

    if backend == "keras":
        if dataset is not None:
            y_pred_raw = model.predict(dataset, verbose=0)
            y_chunks = []
            for batch in dataset:
                if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                    y_chunks.append(np.array(batch[1]))
                else:
                    raise ValueError("Keras dataset must yield (inputs, labels)")
            y_true = np.concatenate(y_chunks, axis=0)
        else:
            if x is None or y is None:
                raise ValueError("For keras evaluation, provide x and y (or dataset)")
            y_pred_raw = model.predict(x, verbose=0)
            y_true = y

    elif backend == "torch":
        if dataloader is None:
            raise ValueError("For torch evaluation, provide dataloader")
        import torch

        device = next(model.parameters()).device if hasattr(model, "parameters") else torch.device("cpu")
        model.eval()

        logits_all = []
        labels_all = []

        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                    inputs, labels = batch[0], batch[1]
                else:
                    raise ValueError("Torch dataloader must yield (inputs, labels)")

                inputs = inputs.to(device)
                outputs = model(inputs)

                logits_all.append(outputs.detach().cpu().numpy())
                labels_all.append(labels.detach().cpu().numpy())

        y_pred_raw = np.concatenate(logits_all, axis=0)
        y_true = np.concatenate(labels_all, axis=0)

    else:
        raise ValueError("backend must be 'keras' or 'torch'")

    return compute_metrics(y_true=y_true, y_pred=y_pred_raw, task_spec=task_spec)
