"""Task specification models shared across training and evaluation."""

from __future__ import annotations

from enum import Enum
from typing import Dict

from pydantic import BaseModel, Field, field_validator


class TaskType(str, Enum):
    """Supported task families for unified pipelines."""

    bc = "bc"  # binary classification
    mcc = "mcc"  # multi-class classification (includes binary)
    mlc = "mlc"  # multi-label classification
    seg = "seg"  # segmentation

    def __str__(self) -> str:
        """Return the enum value so `str(TaskType.bc)` is `bc`."""
        return self.value


class TaskSpec(BaseModel):
    """Task metadata controlling metrics, thresholds, and class semantics."""

    task_type: TaskType = TaskType.mcc
    num_classes: int = 2
    average: str = "weighted"
    class_axis: int = 1

    classification_threshold: float = 0.5
    multilabel_threshold: float = 0.5
    segmentation_threshold: float = 0.5

    id2label: Dict[int, str] = Field(default_factory=dict)

    @field_validator("task_type", mode="before")
    @classmethod
    def _normalize_task_type(cls, value):
        """Accept enum-like strings such as `TaskType.bc` in addition to raw aliases."""
        if hasattr(value, "value"):
            value = value.value
        if isinstance(value, str):
            text = value.strip().lower()
            if "." in text:
                text = text.rsplit(".", 1)[-1]
            return text
        return value

    @property
    def is_classification(self) -> bool:
        """Return ``True`` when the task is BC/MCC/MLC classification."""
        return self.task_type in {TaskType.bc, TaskType.mcc, TaskType.mlc}

    @property
    def is_segmentation(self) -> bool:
        """Return ``True`` when the task is segmentation."""
        return self.task_type == TaskType.seg
