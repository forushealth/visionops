"""Task-aware PyTorch Lightning utilities for BC, MCC, and MLC workflows."""

import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
    MultilabelAccuracy,
    MultilabelF1Score,
    MultilabelPrecision,
    MultilabelRecall,
)
import pytorch_lightning as L
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import numpy as np
import os
from typing import Any, Callable, Dict, List, Optional
import torchvision
from torchvision.models import (
    efficientnet_b0,
    efficientnet_b1,
    efficientnet_b2,
    efficientnet_b3,
    efficientnet_b4,
    efficientnet_b5,
    efficientnet_b6,
    efficientnet_b7,
    efficientnet_v2_l,
    efficientnet_v2_m,
    efficientnet_v2_s,
    mobilenet_v2,
    mobilenet_v3_large,
    mobilenet_v3_small,
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
)
from .ImageProcessingOptions import build_image_processing_pipelines


def _cfg_value(config_dict: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Return the first present key from config dictionary."""
    for key in keys:
        if key in config_dict:
            return config_dict[key]
    return default


_MODEL_BUILDERS: Dict[str, Callable[..., nn.Module]] = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "resnet101": resnet101,
    "resnet152": resnet152,
    "efficientnet_b0": efficientnet_b0,
    "efficientnet_b1": efficientnet_b1,
    "efficientnet_b2": efficientnet_b2,
    "efficientnet_b3": efficientnet_b3,
    "efficientnet_b4": efficientnet_b4,
    "efficientnet_b5": efficientnet_b5,
    "efficientnet_b6": efficientnet_b6,
    "efficientnet_b7": efficientnet_b7,
    "efficientnet_v2_l": efficientnet_v2_l,
    "efficientnet_v2_m": efficientnet_v2_m,
    "efficientnet_v2_s": efficientnet_v2_s,
    "mobilenet_v2": mobilenet_v2,
    "mobilenet_v3_large": mobilenet_v3_large,
    "mobilenet_v3_small": mobilenet_v3_small,
}


def _build_torchvision_backbone(model_name: str, pretrained: bool = True) -> nn.Module:
    """Build one explicitly supported torchvision classification backbone."""
    normalized_name = str(model_name).strip().lower()
    try:
        builder = _MODEL_BUILDERS[normalized_name]
    except KeyError:
        supported = ", ".join(sorted(_MODEL_BUILDERS))
        raise ValueError(f"Unsupported model_name {model_name!r}. Choose one of: {supported}") from None

    weights = "IMAGENET1K_V1" if pretrained else None
    return builder(weights=weights)


def _normalize_task_type(task_type: str) -> str:
    """Normalize task aliases and enum-like inputs for shared classifiers."""
    # TaskSpec.task_type can be an Enum (TaskType.bc) or string-like alias.
    raw = task_type.value if hasattr(task_type, "value") else task_type
    task = str(raw).lower().strip()
    if "." in task:
        task = task.rsplit(".", 1)[-1]
    if task == "bc":
        return "mcc"
    return task


def _split_mlc_labels(value: Any) -> List[str]:
    """Split a multi-label cell into normalized label tokens."""
    if isinstance(value, (list, tuple, np.ndarray)):
        return [str(x).strip() for x in value if str(x).strip()]
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    for sep in ["|", ";"]:
        text = text.replace(sep, ",")
    return [token.strip() for token in text.split(",") if token.strip()]


class TaskDatasetBuilder(Dataset):
    """Task-aware image dataset for BC/MCC/MLC torch workflows."""

    def __init__(
        self,
        dataset_dataframe: pd.DataFrame,
        data_directory_path: str,
        task_type: str = "mcc",
        num_classes: int = 2,
        image_path_column: str = "path",
        image_label_column: str = "labels",
        image_size: tuple = (224, 224, 3),
        label_to_idx: Optional[Dict[Any, int]] = None,
        processing_pipeline: Optional[Any] = None,
        post_has_normalization: bool = False,
    ) -> None:
        """Create a dataset wrapper that emits tensors and task-encoded labels."""
        self.dataframe = dataset_dataframe.reset_index(drop=True)
        self.data_directory_path = data_directory_path
        self.task_type = _normalize_task_type(task_type)
        self.num_classes = int(num_classes)
        self.image_path_column = image_path_column
        self.image_label_column = image_label_column
        self.image_height = int(image_size[0])
        self.image_width = int(image_size[1])
        self.label_to_idx = label_to_idx or {}
        self.processing_pipeline = processing_pipeline
        self.post_has_normalization = bool(post_has_normalization)

    def __len__(self) -> int:
        """Return the number of rows in the dataset."""
        return len(self.dataframe)

    def _encode_label(self, raw_value: Any):
        """Encode one label cell based on task type and known label mapping."""
        if self.task_type == "mlc":
            y = torch.zeros(self.num_classes, dtype=torch.float32)
            for tag in _split_mlc_labels(raw_value):
                idx = self.label_to_idx.get(tag)
                if idx is not None and 0 <= int(idx) < self.num_classes:
                    y[int(idx)] = 1.0
            return y

        if raw_value in self.label_to_idx:
            return torch.tensor(int(self.label_to_idx[raw_value]), dtype=torch.long)

        raw_str = str(raw_value)
        if raw_str in self.label_to_idx:
            return torch.tensor(int(self.label_to_idx[raw_str]), dtype=torch.long)

        if isinstance(raw_value, str):
            try:
                return torch.tensor(int(raw_value), dtype=torch.long)
            except ValueError as exc:
                raise ValueError(f"Unknown class label: {raw_value}") from exc

        return torch.tensor(int(raw_value), dtype=torch.long)

    def __getitem__(self, idx):
        """Load one image tensor and encoded label tensor."""
        if torch.is_tensor(idx):
            idx = idx.tolist()
        row = self.dataframe.iloc[int(idx)]
        image_path = os.path.join(self.data_directory_path, str(row[self.image_path_column]))
        try:
            with Image.open(image_path) as source:
                img = source.convert("RGB")
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Unable to load image {image_path}") from exc

        img = img.resize((self.image_width, self.image_height))
        img_np = np.asarray(img, dtype=np.uint8)
        if self.processing_pipeline is not None:
            out = self.processing_pipeline(image=img_np)
            img_np = out["image"]
            if not isinstance(img_np, np.ndarray):
                img_np = np.asarray(img_np)
            if not np.issubdtype(img_np.dtype, np.integer):
                img_np = img_np.astype(np.float32)
                if (not self.post_has_normalization) and img_np.size > 0 and float(np.max(img_np)) > 1.0:
                    img_np = img_np / 255.0
        x = torchvision.transforms.functional.to_tensor(img_np)
        y = self._encode_label(row[self.image_label_column])
        return x, y


class TaskDataLoaders(L.LightningDataModule):
    """Task-aware datamodule for BC/MCC/MLC torch workflows."""

    def __init__(
        self,
        config_dict: dict,
        batch_size: int = 32,
        task_type: str = "mcc",
        num_classes: Optional[int] = None,
        num_workers: int = 2,
    ) -> None:
        """Initialize task-aware train/validation dataloaders from config."""
        super().__init__()

        train_csv = _cfg_value(config_dict, "Train_data_csv", "train_data_csv")
        val_csv = _cfg_value(config_dict, "Val_data_csv", "val_data_csv")
        if train_csv is None or val_csv is None:
            raise KeyError("Missing train/val csv keys. Use Train_data_csv/Val_data_csv or train_data_csv/val_data_csv.")

        self.train_df = pd.read_csv(str(train_csv))
        self.val_df = pd.read_csv(str(val_csv))

        self.data_directory_path = _cfg_value(config_dict, "Data_directory_path", "data_directory_path")
        if self.data_directory_path is None:
            raise KeyError("Missing data directory key. Use Data_directory_path or data_directory_path.")

        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.task_type = _normalize_task_type(task_type)
        self.image_path_column = _cfg_value(config_dict, "Image_path_column", "image_path_column")
        self.image_label_column = _cfg_value(
            config_dict,
            "Image_categorical_label_column",
            "image_categorical_label_column",
            "image_label_column",
            "Image_label_column",
        )
        self.image_size = tuple(_cfg_value(config_dict, "Image_size", "image_size", default=(224, 224, 3)))
        data_cfg = config_dict.get("data", {}) if isinstance(config_dict.get("data", {}), dict) else {}
        tfms_cfg = data_cfg.get("tfms", data_cfg.get("transforms", {}))
        if not tfms_cfg and isinstance(config_dict.get("transforms"), dict):
            tfms_cfg = config_dict.get("transforms")
        self.image_processing = build_image_processing_pipelines(
            tfms_cfg if isinstance(tfms_cfg, dict) else {},
            image_size=self.image_size,
        )
        if self.image_processing.get("enabled", False):
            names = self.image_processing.get("section_names", {})
            print(
                "[TaskDataLoaders] Using config-driven image processing "
                f"pre={names.get('pre', [])}, aug={names.get('aug', [])}, post={names.get('post', [])}"
            )

        self.image_path_column = self._resolve_column(
            self.image_path_column,
            ["path", "image_path", "filepath", "file_path"],
            "image path",
        )
        self.image_label_column = self._resolve_column(
            self.image_label_column,
            ["labels", "categorical_label", "label", "target"],
            "label",
        )

        self.label_to_idx: Dict[Any, int] = {}
        self.num_classes = int(num_classes) if num_classes is not None else None
        self._prepare_label_space()

        config_dict["Batch_size"] = self.batch_size
        config_dict["batch_size"] = self.batch_size
        config_dict["task_type"] = self.task_type
        config_dict["num_classes"] = int(self.num_classes)
        if self.label_to_idx:
            config_dict["label_to_idx"] = dict(self.label_to_idx)
        self.prepare_data_per_node = True

    def _resolve_column(self, configured: Optional[str], fallbacks: List[str], column_kind: str) -> str:
        """Resolve a shared column name present in both train and validation CSVs."""
        if configured and configured in self.train_df.columns and configured in self.val_df.columns:
            return configured
        for col in fallbacks:
            if col in self.train_df.columns and col in self.val_df.columns:
                return col
        if configured:
            raise KeyError(
                f"Configured {column_kind} column '{configured}' was not found in both train/val CSV files."
            )
        raise KeyError(
            f"Could not infer {column_kind} column from CSV files. Tried: {', '.join(fallbacks)}."
        )

    def _prepare_label_space(self) -> None:
        """Infer label vocabulary/mapping and validate ``num_classes``."""
        train_labels = self.train_df[self.image_label_column]
        val_labels = self.val_df[self.image_label_column]

        if self.task_type == "mlc":
            vocab = sorted({token for v in pd.concat([train_labels, val_labels], ignore_index=True) for token in _split_mlc_labels(v)})
            self.label_to_idx = {token: i for i, token in enumerate(vocab)}
            inferred = len(vocab)
            if self.num_classes is None:
                self.num_classes = inferred
            if self.num_classes < inferred:
                raise ValueError(f"num_classes={self.num_classes} is smaller than inferred mlc labels={inferred}")
            return

        label_series = pd.concat([train_labels, val_labels], ignore_index=True)
        if pd.api.types.is_numeric_dtype(label_series):
            numeric = pd.to_numeric(label_series, errors="raise").astype(int)
            classes = sorted(numeric.unique().tolist())
            self.label_to_idx = {int(cls): i for i, cls in enumerate(classes)}
            self.label_to_idx.update({str(cls): i for i, cls in enumerate(classes)})
        else:
            classes = sorted(label_series.astype(str).unique().tolist())
            self.label_to_idx = {cls: i for i, cls in enumerate(classes)}

        inferred = len(classes)
        if self.num_classes is None:
            self.num_classes = inferred
        elif self.num_classes < inferred:
            raise ValueError(f"num_classes={self.num_classes} is smaller than inferred labels={inferred}")
        else:
            # Keep classifier head aligned to observed labels for MCC/BC tasks.
            self.num_classes = inferred

    def prepare_data(self):
        """Lightning hook (no-op because CSVs are already prepared)."""
        return

    def setup(self, stage: Optional[str] = None):
        """Build train/validation dataset instances for the given stage."""
        if stage == "fit" or stage is None:
            self.train = TaskDatasetBuilder(
                dataset_dataframe=self.train_df,
                data_directory_path=self.data_directory_path,
                task_type=self.task_type,
                num_classes=int(self.num_classes),
                image_path_column=self.image_path_column,
                image_label_column=self.image_label_column,
                image_size=self.image_size,
                label_to_idx=self.label_to_idx,
                processing_pipeline=self.image_processing.get("train") if self.image_processing.get("enabled", False) else None,
                post_has_normalization=bool(self.image_processing.get("post_has_normalization", False)),
            )
            self.val = TaskDatasetBuilder(
                dataset_dataframe=self.val_df,
                data_directory_path=self.data_directory_path,
                task_type=self.task_type,
                num_classes=int(self.num_classes),
                image_path_column=self.image_path_column,
                image_label_column=self.image_label_column,
                image_size=self.image_size,
                label_to_idx=self.label_to_idx,
                processing_pipeline=self.image_processing.get("eval") if self.image_processing.get("enabled", False) else None,
                post_has_normalization=bool(self.image_processing.get("post_has_normalization", False)),
            )

    def train_dataloader(self):
        """Return shuffled training dataloader."""
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        """Return validation dataloader."""
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)


class TaskClassifier(L.LightningModule):
    """Task-aware classifier for BC/MCC/MLC torch workflows."""

    def __init__(
        self,
        num_classes: int,
        task_type: str = "mcc",
        model_name: str = "resnet18",
        pretrained: bool = True,
        learning_rate: float = 0.001,
    ) -> None:
        """Initialize a task-aware classifier with a torchvision backbone."""
        super().__init__()
        self.task_type = _normalize_task_type(task_type)
        self.num_classes = int(num_classes)
        self.learning_rate = float(learning_rate)

        if self.task_type not in {"mcc", "mlc"}:
            raise ValueError("task_type must resolve to one of {'mcc','mlc'} (alias 'bc' is supported)")
        if self.task_type == "mlc" and self.num_classes < 2:
            raise ValueError("For mlc task, num_classes must be >= 2")

        self.model = _build_torchvision_backbone(model_name=model_name, pretrained=pretrained)
        out_dim = self._output_dim()
        self._replace_head(out_dim)
        self.criterion = self._build_loss()
        self._build_metrics()

    def _output_dim(self) -> int:
        if self.task_type == "mlc":
            return self.num_classes
        if self.num_classes > 2:
            return self.num_classes
        return 1

    def _replace_head(self, out_dim: int) -> None:
        if hasattr(self.model, "fc") and isinstance(self.model.fc, nn.Linear):
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, out_dim)
            return

        if hasattr(self.model, "classifier"):
            classifier = self.model.classifier
            if isinstance(classifier, nn.Linear):
                self.model.classifier = nn.Linear(classifier.in_features, out_dim)
                return
            if isinstance(classifier, nn.Sequential):
                for i in range(len(classifier) - 1, -1, -1):
                    if isinstance(classifier[i], nn.Linear):
                        classifier[i] = nn.Linear(classifier[i].in_features, out_dim)
                        self.model.classifier = classifier
                        return

        raise ValueError("Unsupported backbone head. Expected `fc` or `classifier` with Linear layer.")

    def _build_loss(self):
        """Create task-specific loss function."""
        if self.task_type == "mlc":
            return nn.BCEWithLogitsLoss()
        if self.num_classes > 2:
            return nn.CrossEntropyLoss()
        return nn.BCEWithLogitsLoss()

    def _build_metrics(self) -> None:
        if self.task_type == "mlc":
            self.accuracy = MultilabelAccuracy(num_labels=self.num_classes)
            self.precision = MultilabelPrecision(num_labels=self.num_classes, average="macro")
            self.recall = MultilabelRecall(num_labels=self.num_classes, average="macro")
            self.f1score = MultilabelF1Score(num_labels=self.num_classes, average="macro")
            return
        if self.num_classes > 2:
            self.accuracy = MulticlassAccuracy(num_classes=self.num_classes)
            self.precision = MulticlassPrecision(num_classes=self.num_classes, average="macro")
            self.recall = MulticlassRecall(num_classes=self.num_classes, average="macro")
            self.f1score = MulticlassF1Score(num_classes=self.num_classes, average="macro")
            return

        self.accuracy = BinaryAccuracy()
        self.precision = BinaryPrecision()
        self.recall = BinaryRecall()
        self.f1score = BinaryF1Score()

    def forward(self, x):
        """Run forward pass through the backbone model."""
        return self.model(x.float())

    def _loss_and_metrics(self, logits, labels):
        if self.task_type == "mlc":
            y = labels.float()
            loss = self.criterion(logits, y)
            acc = self.accuracy(logits, y.int())
            pr = self.precision(logits, y.int())
            rc = self.recall(logits, y.int())
            f1 = self.f1score(logits, y.int())
            return loss, acc, pr, rc, f1

        if self.num_classes > 2:
            y = labels.long()
            loss = self.criterion(logits, y)
            acc = self.accuracy(logits, y)
            pr = self.precision(logits, y)
            rc = self.recall(logits, y)
            f1 = self.f1score(logits, y)
            return loss, acc, pr, rc, f1

        y = labels.float().unsqueeze(1) if labels.ndim == 1 else labels.float()
        loss = self.criterion(logits, y)
        y_int = y.int()
        acc = self.accuracy(logits, y_int)
        pr = self.precision(logits, y_int)
        rc = self.recall(logits, y_int)
        f1 = self.f1score(logits, y_int)
        return loss, acc, pr, rc, f1

    def training_step(self, batch, batch_idx):
        """Execute one training step and log metrics."""
        images, labels = batch
        logits = self(images)
        loss, acc, pr, rc, f1 = self._loss_and_metrics(logits, labels)
        self.log_dict(
            {
                "train_loss": loss,
                "train_acc": acc,
                "train_precision": pr,
                "train_recall": rc,
                "train_f1score": f1,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        """Execute one validation step and log metrics."""
        images, labels = batch
        logits = self(images)
        loss, acc, pr, rc, f1 = self._loss_and_metrics(logits, labels)
        self.log_dict(
            {
                "val_loss": loss,
                "val_acc": acc,
                "val_precision": pr,
                "val_recall": rc,
                "val_f1score": f1,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def configure_optimizers(self):
        """Build optimizer for Lightning trainer."""
        return optim.Adam(self.parameters(), lr=self.learning_rate)
