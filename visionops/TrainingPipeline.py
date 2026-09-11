"""Training-only orchestration pipeline for keras and torch workflows."""

from __future__ import annotations

import json
import os
import re
import shutil
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .HpUtils import resolve_stage_from_yaml
from .HpUtils import freeze_preprocessing_manifest
from .TaskSpec import TaskSpec
from .UnifiedTrainer import UnifiedTrainer
from ._dfperf import DfPerfLifecycleMixin
from ._pipeline_facade import PipelineFacadeMixin
from ._utils import normalize_path_value, run_cleanup_steps


class TrainingPipeline(PipelineFacadeMixin, DfPerfLifecycleMixin):
    """Training-only pipeline.

    Responsibilities:
    - pre-training checks and runtime perf monitoring
    - model training (keras/torch)
    - export prediction artifacts for post-training model-free evaluation

    This pipeline is intentionally separate from evaluation.
    """

    PATH_COL_CANDIDATES = ("image_path", "full_path", "path", "file_path", "filepath", "image")
    LABEL_COL_CANDIDATES = ("label", "labels", "target", "class", "categorical_label", "y_true")
    LOGITS_COL_CANDIDATES = ("logits", "logit", "y_pred", "prediction", "predictions")
    ARTIFACT_SECTION_ORDER = (
        "top_artifacts",
        "data_related",
        "pre_training_related",
        "training_related",
        "evaluation_related",
    )
    ARTIFACT_SECTION_MLFLOW_DIRS = {
        "top_artifacts": "01_run_metadata",
        "data_related": "02_data_preprocessing",
        "pre_training_related": "03_runtime_environment",
        "training_related": "04_model_training",
        "evaluation_related": "05_model_evaluation",
    }
    ARTIFACT_LAYOUT_DIRNAME = "artifacts"
    ARTIFACT_LAYOUT_MANIFEST_NAME = "artifact_layout_manifest.json"
    TOP_ARTIFACT_KEYS = ("config_path", "mlflow_run_id", "artifact_catalog_json")
    DATA_ARTIFACT_KEYS = ("preprocessing_manifest",)
    PRETRAINING_ARTIFACT_KEYS = (
        "dfperf_preflight_json",
        "dfperf_gpu_samples_csv",
        "dfperf_gpu_summary_json",
        "dfperf_metrics_json",
        "dfperf_report_json",
        "dfperf_report_md",
    )
    TRAINING_ARTIFACT_KEYS = ("keras_model_path", "keras_train_history_csv")
    EVALUATION_ARTIFACT_KEYS = ("val_logits_csv", "keras_val_logits_csv", "torch_val_logits_csv", "torch_eval_outputs_npz")

    def __init__(
        self,
        backend: str,
        config_path: Optional[str] = None,
        use_mlflow: bool = False,
        task_spec: Optional[TaskSpec] = None,
        task_type: str = "mcc",
        num_classes: int = 2,
        run_dfperf_preflight: bool = True,
        dfperf_outdir: str = "artifacts/03_runtime_environment",
        dfperf_strict: bool = False,
        dfperf_sample_gpu: bool = False,
        dfperf_sample_seconds: float = 5.0,
        dfperf_sample_interval: float = 1.0,
        dfperf_monitor_during_training: bool = True,
        dfperf_generate_report: bool = True,
        dfperf_generate_plots: bool = True,
        dfperf_tag: Optional[str] = None,
        export_eval_outputs: bool = True,
        eval_outputs_outdir: Optional[str] = None,
    ) -> None:
        """Initialize the training pipeline and runtime monitoring options.

        Example:
            >>> pipe = TrainingPipeline(backend="keras", config_path="config.yaml", use_mlflow=False)
            >>> pipe.backend
            'keras'
        """
        backend = backend.lower().strip()
        if backend not in {"keras", "torch"}:
            raise ValueError("backend must be 'keras' or 'torch'")

        self.backend = backend
        self.config_path = config_path
        self.use_mlflow = use_mlflow
        self.task_spec = task_spec or TaskSpec(task_type=task_type, num_classes=num_classes)

        self.run_dfperf_preflight = run_dfperf_preflight
        self.dfperf_outdir = dfperf_outdir
        self.dfperf_strict = dfperf_strict
        self.dfperf_sample_gpu = dfperf_sample_gpu
        self.dfperf_sample_seconds = dfperf_sample_seconds
        self.dfperf_sample_interval = dfperf_sample_interval
        self.dfperf_monitor_during_training = dfperf_monitor_during_training
        self.dfperf_generate_report = dfperf_generate_report
        self.dfperf_generate_plots = dfperf_generate_plots
        self.dfperf_tag = dfperf_tag

        self.export_eval_outputs = export_eval_outputs
        self.eval_outputs_outdir = eval_outputs_outdir or os.path.join(
            os.path.dirname(self.dfperf_outdir),
            "05_model_evaluation",
        )

        self._trainer = UnifiedTrainer(backend=backend, config_path=config_path)
        self._fit_runner = None
        self._torch_datamodule = None
        self.model = None
        self.mlflow_logger = None
        self.dfperf_report = None
        self.dfperf_runtime_report = None
        self.eval_artifacts: Dict[str, str] = {}
        self.artifact_catalog: Dict[str, Dict[str, Optional[str]]] = {}
        self.artifact_catalog_path: Optional[str] = None
        self.ordered_artifacts_root: Optional[str] = None
        self.artifact_layout_manifest_path: Optional[str] = None
        self.preprocessing_manifest_path: Optional[str] = None

    @staticmethod
    def _find_first_existing(columns, candidates) -> Optional[str]:
        lookup = {str(c).strip().lower(): str(c) for c in columns}
        for cand in candidates:
            key = str(cand).strip().lower()
            if key in lookup:
                return lookup[key]
        return None

    @staticmethod
    def _serialize_logits_value(value: Any) -> str:
        if isinstance(value, str):
            return value
        arr = np.array(value)
        if arr.ndim == 0:
            return str([float(arr.item())])
        return str(arr.reshape(-1).tolist())

    @staticmethod
    def _serialize_label_value(value: Any) -> Any:
        if isinstance(value, (list, tuple, np.ndarray)):
            return str(np.array(value).reshape(-1).tolist())
        if isinstance(value, np.generic):
            return value.item()
        return value

    @staticmethod
    def _as_existing_file(path: Any) -> Optional[str]:
        if isinstance(path, str) and path and os.path.isfile(path):
            return path
        return None

    @staticmethod
    def _ordered_section(items: Dict[str, Optional[str]], preferred_keys: tuple[str, ...]) -> Dict[str, Optional[str]]:
        out: Dict[str, Optional[str]] = {}
        for key in preferred_keys:
            out[key] = items.get(key)
        return out

    @staticmethod
    def _safe_artifact_key(key: str) -> str:
        text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(key)).strip("._-")
        return text or "artifact"

    def _build_artifact_catalog(self) -> Dict[str, Dict[str, Optional[str]]]:
        top_items: Dict[str, Optional[str]] = {}
        if isinstance(self.config_path, str) and self.config_path:
            top_items["config_path"] = os.path.abspath(self.config_path)
        else:
            top_items["config_path"] = None
        run_id = None
        if self.mlflow_logger is not None and getattr(self.mlflow_logger, "run_id", None):
            run_id = str(self.mlflow_logger.run_id)
        elif isinstance(self.eval_artifacts.get("mlflow_run_id"), str):
            run_id = str(self.eval_artifacts.get("mlflow_run_id"))
        top_items["mlflow_run_id"] = run_id
        top_items["artifact_catalog_json"] = self.artifact_catalog_path

        data_items: Dict[str, Optional[str]] = {
            "preprocessing_manifest": self._as_existing_file(self.preprocessing_manifest_path),
        }

        pre_items: Dict[str, Optional[str]] = {
            "dfperf_preflight_json": self._as_existing_file(
                self.dfperf_report.get("preflight_json")
                if isinstance(self.dfperf_report, dict)
                else None
            )
        }
        if isinstance(self.dfperf_runtime_report, dict):
            runtime_map = self.dfperf_runtime_report.get("artifacts", {})
            if isinstance(runtime_map, dict):
                for key in sorted(runtime_map.keys()):
                    pre_items[f"dfperf_{key}"] = self._as_existing_file(runtime_map.get(key))

        training_items: Dict[str, Optional[str]] = {}
        evaluation_items: Dict[str, Optional[str]] = {}
        for key, value in self.eval_artifacts.items():
            if key in {"mlflow_run_id", "artifact_catalog_json"}:
                continue
            if key in self.DATA_ARTIFACT_KEYS:
                data_items[key] = self._as_existing_file(value)
                continue
            if key in self.TRAINING_ARTIFACT_KEYS or any(tok in key for tok in ("model", "history", "checkpoint")):
                training_items[key] = self._as_existing_file(value)
                continue
            if key in self.EVALUATION_ARTIFACT_KEYS or any(tok in key for tok in ("logits", "eval_outputs", "evaluator")):
                evaluation_items[key] = self._as_existing_file(value)
                continue
            evaluation_items[key] = self._as_existing_file(value)

        catalog = {
            "top_artifacts": self._ordered_section(top_items, self.TOP_ARTIFACT_KEYS),
            "data_related": self._ordered_section(data_items, self.DATA_ARTIFACT_KEYS),
            "pre_training_related": self._ordered_section(pre_items, self.PRETRAINING_ARTIFACT_KEYS),
            "training_related": self._ordered_section(training_items, self.TRAINING_ARTIFACT_KEYS),
            "evaluation_related": self._ordered_section(evaluation_items, self.EVALUATION_ARTIFACT_KEYS),
        }
        return catalog

    def _save_artifact_catalog(self) -> Optional[str]:
        try:
            os.makedirs(self.dfperf_outdir, exist_ok=True)
            self.artifact_catalog_path = os.path.join(self.dfperf_outdir, "artifact_catalog.json")
            self.eval_artifacts["artifact_catalog_json"] = self.artifact_catalog_path
            self.artifact_catalog = self._build_artifact_catalog()
            with open(self.artifact_catalog_path, "w", encoding="utf-8") as f:
                json.dump(self.artifact_catalog, f, indent=2)
            return self.artifact_catalog_path
        except Exception:
            return None

    def _materialize_ordered_artifacts(self) -> Optional[str]:
        """Create a stable on-disk 5-folder artifact layout for each run."""
        try:
            if not self.artifact_catalog:
                self.artifact_catalog = self._build_artifact_catalog()

            layout_root = os.path.dirname(os.path.abspath(self.dfperf_outdir))
            os.makedirs(layout_root, exist_ok=True)

            layout_manifest: Dict[str, Any] = {}
            for section in self.ARTIFACT_SECTION_ORDER:
                section_map = self.artifact_catalog.get(section, {})
                section_subdir = self.ARTIFACT_SECTION_MLFLOW_DIRS.get(section, section)
                section_dir = os.path.join(layout_root, section_subdir)
                os.makedirs(section_dir, exist_ok=True)

                ordered_entries = []
                if isinstance(section_map, dict):
                    for idx, (key, path) in enumerate(section_map.items(), start=1):
                        safe_key = self._safe_artifact_key(key)
                        prefix = f"{idx:02d}_{safe_key}"
                        if isinstance(path, str) and path and os.path.isfile(path):
                            ext = os.path.splitext(path)[1]
                            dst = os.path.join(section_dir, f"{prefix}{ext}")
                            shutil.copy2(path, dst)
                            entry = {
                                "key": key,
                                "status": "present",
                                "source_path": path,
                                "materialized_path": dst,
                            }
                        else:
                            dst = os.path.join(section_dir, f"{prefix}.missing.txt")
                            with open(dst, "w", encoding="utf-8") as fp:
                                fp.write(f"{key}: missing\n")
                                fp.write(f"source_path={path}\n")
                            entry = {
                                "key": key,
                                "status": "missing",
                                "source_path": path,
                                "materialized_path": dst,
                            }
                        ordered_entries.append(entry)

                section_manifest_path = os.path.join(section_dir, "section_manifest.json")
                with open(section_manifest_path, "w", encoding="utf-8") as fp:
                    json.dump(ordered_entries, fp, indent=2)

                layout_manifest[section] = {
                    "section_dir": section_dir,
                    "entries": ordered_entries,
                    "count": len(ordered_entries),
                }

            manifest_path = os.path.join(layout_root, self.ARTIFACT_LAYOUT_MANIFEST_NAME)
            with open(manifest_path, "w", encoding="utf-8") as fp:
                json.dump(layout_manifest, fp, indent=2)

            self.ordered_artifacts_root = layout_root
            self.artifact_layout_manifest_path = manifest_path
            self.eval_artifacts["ordered_artifacts_root"] = layout_root
            self.eval_artifacts["artifact_layout_manifest_json"] = manifest_path
            return layout_root
        except Exception:
            return None

    def _standardize_logits_csv(self, src_csv: str, dst_csv: str) -> str:
        """Normalize logits CSV schema to image_path/label/logits."""
        df = pd.read_csv(src_csv)
        if df.empty:
            pd.DataFrame(columns=["image_path", "label", "logits"]).to_csv(dst_csv, index=False)
            return dst_csv

        path_col = self._find_first_existing(df.columns, self.PATH_COL_CANDIDATES)
        label_col = self._find_first_existing(df.columns, self.LABEL_COL_CANDIDATES)
        logits_col = self._find_first_existing(df.columns, self.LOGITS_COL_CANDIDATES)

        if label_col is None:
            raise ValueError(f"Could not infer label column from logits CSV: {list(df.columns)}")
        if logits_col is None:
            raise ValueError(f"Could not infer logits column from logits CSV: {list(df.columns)}")

        image_vals = df[path_col].map(normalize_path_value).tolist() if path_col else [""] * len(df)
        out_df = pd.DataFrame(
            {
                "image_path": image_vals,
                "label": df[label_col].map(self._serialize_label_value),
                "logits": df[logits_col].map(self._serialize_logits_value),
            }
        )
        out_df.to_csv(dst_csv, index=False)
        return dst_csv

    def _resolve_torch_eval_image_paths(self, dataloader: Any, expected_len: int) -> list[str]:
        """Best-effort path reconstruction from dataloader/datamodule metadata."""
        expected_len = max(0, int(expected_len))
        if expected_len == 0:
            return []

        def _from_dataframe(df: Any, path_col: Any, data_dir: Any) -> list[str]:
            if not isinstance(df, pd.DataFrame):
                return []
            if not path_col or str(path_col) not in df.columns:
                return []
            values = df[str(path_col)].tolist()[:expected_len]
            out: list[str] = []
            base_dir = str(data_dir) if data_dir else ""
            for raw in values:
                p = normalize_path_value(raw)
                if base_dir and not os.path.isabs(p):
                    p = os.path.join(base_dir, p)
                out.append(p)
            return out

        dataset = getattr(dataloader, "dataset", None)
        if dataset is not None:
            ds_df = getattr(dataset, "dataframe", None)
            ds_path_col = getattr(dataset, "image_path_column", None)
            ds_data_dir = getattr(dataset, "data_directory_path", None)
            paths = _from_dataframe(ds_df, ds_path_col, ds_data_dir)
            if paths:
                return paths

        datamodule = self._torch_datamodule
        if datamodule is not None:
            dm_df = getattr(datamodule, "val_df", None)
            dm_path_col = getattr(datamodule, "image_path_column", None)
            dm_data_dir = getattr(datamodule, "data_directory_path", None)
            paths = _from_dataframe(dm_df, dm_path_col, dm_data_dir)
            if paths:
                return paths

        return []

    def fit(self, **kwargs: Any) -> Any:
        """Run full training flow including preflight and artifact exports.

        Example:
            >>> # pipe = TrainingPipeline(backend="keras", config_path="config.yaml")
            >>> # pipe.fit()  # doctest: +SKIP
            >>> True
            True
        """
        self._run_pre_training_checks()
        runtime_session = self._start_dfperf_runtime_monitor()

        result = None
        training_error = None
        try:
            if self.backend == "keras":
                result = self._fit_keras(**kwargs)
            else:
                result = self._fit_torch(**kwargs)

            if self.export_eval_outputs:
                self._export_eval_outputs(**kwargs)
        except BaseException as exc:
            training_error = exc
            raise
        finally:
            cleanup_steps = [
                (
                    "dfperf runtime finalization",
                    lambda: self._finalize_dfperf_runtime_monitor(
                        session=runtime_session,
                        training_ok=training_error is None,
                        error_message=str(training_error) if training_error is not None else None,
                    ),
                ),
                ("artifact catalog save", self._save_artifact_catalog),
                ("ordered artifact materialization", self._materialize_ordered_artifacts),
                ("MLflow artifact logging", self._log_artifacts_to_mlflow),
            ]
            if self.use_mlflow and self.mlflow_logger is not None:
                cleanup_steps.append(("MLflow run finalization", self.mlflow_logger.end_run))
            run_cleanup_steps(cleanup_steps, primary_error=training_error)

        return result

    def visualise_training_images(
        self,
        dataset_type: str = "train",
        num_images: int = 9,
        **kwargs: Any,
    ) -> Any:
        """Visualize preprocessed training/validation images from pipeline input.

        Example:
            >>> # pipe.visualise_training_images(dataset_type="train", num_images=9)  # doctest: +SKIP
            >>> True
            True
        """
        split = str(dataset_type).lower().strip()
        if split not in {"train", "val"}:
            raise ValueError("dataset_type must be 'train' or 'val'.")
        count = int(num_images)
        if count < 1:
            raise ValueError("num_images must be >= 1.")

        if self.backend == "keras":
            runner = kwargs.get("runner") or self._fit_runner
            if runner is None:
                if not self.config_path:
                    raise ValueError("config_path is required to build Keras data visualizer.")
                from .KerasTrainer import KerasFitTrainer

                effective_cfg = resolve_stage_from_yaml(
                    self.config_path,
                    stage="train",
                    validate=False,
                    snapshot_dir=os.path.join(self.dfperf_outdir, "config"),
                )
                effective_cfg["task_type"] = str(self.task_spec.task_type)
                effective_cfg["num_classes"] = int(self.task_spec.num_classes)
                runner = KerasFitTrainer(self.config_path, config=effective_cfg)
            return runner.data_visualization(dataset_type=split, num_images=count)

        datamodule = kwargs.get("datamodule") or self._torch_datamodule
        if datamodule is None and self.config_path:
            from .PLTrainingUtils import TaskDataLoaders

            effective_cfg: Dict[str, Any] = {}
            try:
                effective_cfg = resolve_stage_from_yaml(
                    self.config_path,
                    stage="train",
                    validate=False,
                    snapshot_dir=os.path.join(self.dfperf_outdir, "config"),
                )
            except Exception:
                effective_cfg = {}

            if effective_cfg:
                datamodule = TaskDataLoaders(
                    config_dict=effective_cfg,
                    batch_size=int(effective_cfg.get("batch_size", effective_cfg.get("Batch_size", 16))),
                    task_type=str(self.task_spec.task_type),
                    num_classes=int(self.task_spec.num_classes),
                )

        if datamodule is None:
            raise ValueError("Torch visualization requires datamodule or config_path with train/val CSV settings.")

        if hasattr(datamodule, "setup"):
            datamodule.setup("fit")

        loader = datamodule.train_dataloader() if split == "train" else datamodule.val_dataloader()
        batch = next(iter(loader))
        if not isinstance(batch, (tuple, list)) or len(batch) < 1:
            raise ValueError("Could not read image batch from dataloader.")

        images = batch[0]
        labels = batch[1] if len(batch) > 1 else None
        try:
            import torch

            if isinstance(images, torch.Tensor):
                images_np = images.detach().cpu().numpy()
            else:
                images_np = np.asarray(images)
        except Exception:
            images_np = np.asarray(images)

        labels_np = None
        if labels is not None:
            try:
                import torch

                if isinstance(labels, torch.Tensor):
                    labels_np = labels.detach().cpu().numpy()
                else:
                    labels_np = np.asarray(labels)
            except Exception:
                try:
                    labels_np = np.asarray(labels)
                except Exception:
                    labels_np = None

        class_names = None
        class_mapping = getattr(datamodule, "class_to_idx", None)
        if isinstance(class_mapping, dict) and class_mapping:
            class_names = {int(v): k for k, v in class_mapping.items()}
        if class_names is None:
            class_names = getattr(datamodule, "classes", None)
        if class_names is None:
            split_dataset = getattr(datamodule, "train_dataset" if split == "train" else "val_dataset", None)
            if split_dataset is not None:
                ds_class_mapping = getattr(split_dataset, "class_to_idx", None)
                if isinstance(ds_class_mapping, dict) and ds_class_mapping:
                    class_names = {int(v): k for k, v in ds_class_mapping.items()}
                elif hasattr(split_dataset, "classes"):
                    class_names = split_dataset.classes

        if images_np.ndim != 4:
            raise ValueError(f"Expected image batch with 4 dims, got shape={images_np.shape}.")

        # Convert NCHW to NHWC for plotting when needed.
        if images_np.shape[1] in {1, 3} and images_np.shape[-1] not in {1, 3}:
            images_np = np.transpose(images_np, (0, 2, 3, 1))

        images_np = images_np.astype(np.float32)
        if np.nanmin(images_np) < 0.0 or np.nanmax(images_np) > 1.0:
            mins = images_np.min(axis=(1, 2, 3), keepdims=True)
            maxs = images_np.max(axis=(1, 2, 3), keepdims=True)
            denom = np.where((maxs - mins) < 1e-6, 1.0, maxs - mins)
            images_np = (images_np - mins) / denom

        from .DataUtils import visualise_dataset

        visualise_dataset(images_np, image_number=count, labels=labels_np, class_names=class_names)
        return images_np[: min(count, images_np.shape[0])]

    def visualize_training_images(
        self,
        dataset_type: str = "train",
        num_images: int = 9,
        **kwargs: Any,
    ) -> Any:
        """Alias of `visualise_training_images` using US spelling.

        Example:
            >>> # pipe.visualize_training_images(dataset_type="train", num_images=9)  # doctest: +SKIP
            >>> True
            True
        """
        return self.visualise_training_images(dataset_type=dataset_type, num_images=num_images, **kwargs)

    def _fit_keras(self, **kwargs: Any) -> Any:
        """Train using Keras trainer and persist preprocessing manifest.

        Example:
            >>> # pipe._fit_keras()  # doctest: +SKIP
            >>> True
            True
        """
        if not self.config_path:
            raise ValueError("config_path is required for keras backend")
        if self.task_spec.is_segmentation:
            raise ValueError("KerasFitTrainer currently supports classification tasks only (bc/mcc/mlc).")

        from .KerasTrainer import KerasFitTrainer

        effective_cfg = resolve_stage_from_yaml(
            self.config_path,
            stage="train",
            validate=False,
            snapshot_dir=os.path.join(self.dfperf_outdir, "config"),
        )
        effective_cfg["task_type"] = str(self.task_spec.task_type)
        effective_cfg["num_classes"] = int(self.task_spec.num_classes)
        runner = KerasFitTrainer(self.config_path, config=effective_cfg)
        # Pipeline-level switch must override YAML logging defaults.
        runner.use_mlflow = bool(self.use_mlflow)

        if self.use_mlflow:
            from .MLflowLogger import create_keras_callback
            if self.mlflow_logger is None:
                self.setup_mlflow()
            runner.use_mlflow = False
            runner.mlflow_logger = self.mlflow_logger
            runner.auto_end_mlflow_run = False
            runner.callbacks.append(create_keras_callback(self.mlflow_logger))

        history = runner.fit()
        try:
            self.preprocessing_manifest_path = freeze_preprocessing_manifest(
                effective_cfg,
                outdir=os.path.join(self.dfperf_outdir, "config"),
            )
            self.eval_artifacts["preprocessing_manifest"] = self.preprocessing_manifest_path
        except Exception:
            self.preprocessing_manifest_path = None
        self._fit_runner = runner
        self.model = runner.model
        return history

    def _fit_torch(self, **kwargs: Any) -> Any:
        """Train with torch backend and auto-build components when omitted.

        Example:
            >>> # pipe = TrainingPipeline(backend="torch", config_path="config.yaml")
            >>> # pipe._fit_torch()  # doctest: +SKIP
            >>> True
            True
        """
        effective_cfg: Dict[str, Any] = {}
        if self.config_path:
            try:
                effective_cfg = resolve_stage_from_yaml(
                    self.config_path,
                    stage="train",
                    validate=False,
                    snapshot_dir=os.path.join(self.dfperf_outdir, "config"),
                )
            except Exception:
                effective_cfg = {}

        if kwargs.get("model") is None:
            if not effective_cfg:
                raise ValueError(
                    "Torch fit requires either `model` + dataloaders or a valid config_path with train/val CSV settings."
                )
            from .PLTrainingUtils import TaskClassifier, TaskDataLoaders

            batch_size = int(
                kwargs.get(
                    "batch_size",
                    effective_cfg.get("batch_size", effective_cfg.get("Batch_size", 16)),
                )
            )
            datamodule = kwargs.get("datamodule")
            if datamodule is None:
                num_workers = int(
                    kwargs.get(
                        "num_workers",
                        effective_cfg.get("num_workers", 2),
                    )
                )
                datamodule = TaskDataLoaders(
                    config_dict=effective_cfg if effective_cfg else {},
                    batch_size=batch_size,
                    task_type=str(self.task_spec.task_type),
                    num_classes=int(self.task_spec.num_classes),
                    num_workers=num_workers,
                )
                kwargs["datamodule"] = datamodule

            kwargs["model"] = TaskClassifier(
                task_type=str(self.task_spec.task_type),
                num_classes=int(self.task_spec.num_classes),
                model_name=str(effective_cfg.get("model_name", "resnet18")),
                learning_rate=float(effective_cfg.get("learning_rate", 1e-3)),
            )

        if kwargs.get("max_epochs") is None:
            kwargs["max_epochs"] = int(effective_cfg.get("epochs", 10))

        callbacks = kwargs.get("callbacks")
        if callbacks is None:
            callbacks = []

        if self.use_mlflow:
            from .MLflowLogger import create_lightning_callback

            if self.mlflow_logger is None:
                self.setup_mlflow()
            callbacks = list(callbacks)
            callbacks.append(create_lightning_callback(self.mlflow_logger))

        kwargs["callbacks"] = callbacks
        result = self._trainer.fit(**kwargs)
        self.model = kwargs.get("model")
        self._torch_datamodule = kwargs.get("datamodule")
        return result

    def _export_eval_outputs(self, **kwargs: Any) -> None:
        """Export backend-specific prediction artifacts for evaluation.

        Example:
            >>> # pipe._export_eval_outputs()  # doctest: +SKIP
            >>> True
            True
        """
        os.makedirs(self.eval_outputs_outdir, exist_ok=True)

        if self.backend == "keras":
            runner = self._fit_runner
            if runner is None:
                return
            if getattr(runner, "val_logits_csv_path", None):
                src_logits = str(runner.val_logits_csv_path)
                dst_logits = os.path.join(self.eval_outputs_outdir, "val_logits.csv")
                if not os.path.isfile(src_logits):
                    raise FileNotFoundError(f"Keras validation logits artifact not found: {src_logits}")
                self._standardize_logits_csv(src_logits, dst_logits)
                self.eval_artifacts["keras_val_logits_csv"] = dst_logits
                self.eval_artifacts["val_logits_csv"] = dst_logits
            if getattr(runner, "train_history_csv_path", None):
                src_hist = str(runner.train_history_csv_path)
                dst_hist = os.path.join(self.eval_outputs_outdir, "train_history.csv")
                if not os.path.isfile(src_hist):
                    raise FileNotFoundError(f"Keras training history artifact not found: {src_hist}")
                shutil.copy2(src_hist, dst_hist)
                self.eval_artifacts["keras_train_history_csv"] = dst_hist
            final_model_path = runner.config.get("final_model_path") if hasattr(runner, "config") else None
            if isinstance(final_model_path, str) and final_model_path and os.path.isfile(final_model_path):
                self.eval_artifacts["keras_model_path"] = final_model_path
            elif getattr(runner, "model", None) is not None:
                exported_model_path = os.path.join(self.eval_outputs_outdir, "keras_model.keras")
                runner.model.save(exported_model_path)
                self.eval_artifacts["keras_model_path"] = exported_model_path
            if self.mlflow_logger is not None and getattr(self.mlflow_logger, "run_id", None):
                self.eval_artifacts["mlflow_run_id"] = self.mlflow_logger.run_id
            return

        # Torch export
        model = self.model
        if model is None:
            raise RuntimeError(
                "Torch eval export failed: trained model is unavailable. "
                "Ensure torch fit completed successfully before evaluation export."
            )

        dataloader = kwargs.get("eval_dataloader") or kwargs.get("val_dataloader")
        if dataloader is None:
            datamodule = kwargs.get("datamodule") or self._torch_datamodule
            if datamodule is not None and hasattr(datamodule, "val_dataloader"):
                try:
                    if hasattr(datamodule, "setup"):
                        datamodule.setup("fit")
                    dataloader = datamodule.val_dataloader()
                except Exception:
                    dataloader = None

        if dataloader is None:
            raise RuntimeError(
                "Torch eval export failed: validation dataloader is unavailable. "
                "Provide `datamodule` or `eval_dataloader` to TrainingPipeline.fit(...)."
            )

        try:
            import torch
        except Exception:
            return

        device = next(model.parameters()).device if hasattr(model, "parameters") else torch.device("cpu")
        model.eval()

        y_true_all = []
        y_pred_all = []

        with torch.no_grad():
            for batch in dataloader:
                if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                    continue
                x, y = batch[0], batch[1]
                x = x.to(device)
                y_pred = model(x)

                y_pred_all.append(y_pred.detach().cpu().numpy())
                if hasattr(y, "detach"):
                    y_true_all.append(y.detach().cpu().numpy())
                else:
                    y_true_all.append(np.array(y))

        if not y_true_all or not y_pred_all:
            raise RuntimeError(
                "Torch eval export failed: validation loop produced no predictions. "
                "Check validation CSV/data loader and label columns."
            )

        y_true_np = np.concatenate(y_true_all, axis=0)
        y_pred_np = np.concatenate(y_pred_all, axis=0)

        n = min(len(y_true_np), len(y_pred_np))
        image_paths = self._resolve_torch_eval_image_paths(dataloader, expected_len=n)
        if len(image_paths) < n:
            image_paths = image_paths + ([""] * (n - len(image_paths)))

        if y_true_np.ndim == 1:
            label_vals = y_true_np.reshape(-1)[:n].tolist()
        elif y_true_np.ndim == 2 and y_true_np.shape[1] == 1:
            label_vals = y_true_np.reshape(-1)[:n].tolist()
        else:
            label_vals = [np.array(row).reshape(-1).tolist() for row in y_true_np[:n]]

        if y_pred_np.ndim == 1:
            logits_vals = [[float(v)] for v in y_pred_np.reshape(-1)[:n].tolist()]
        else:
            logits_vals = [np.array(row).reshape(-1).tolist() for row in y_pred_np[:n]]

        torch_logits_csv = os.path.join(self.eval_outputs_outdir, "val_logits.csv")
        torch_logits_df = pd.DataFrame(
            {
                "image_path": image_paths[:n],
                "label": label_vals,
                "logits": logits_vals,
            }
        )
        torch_logits_df.to_csv(torch_logits_csv, index=False)
        self.eval_artifacts["torch_val_logits_csv"] = torch_logits_csv
        self.eval_artifacts["val_logits_csv"] = torch_logits_csv

        npz_path = os.path.join(self.eval_outputs_outdir, "eval_outputs.npz")
        np.savez(
            npz_path,
            y_true=y_true_np,
            y_pred=y_pred_np,
            image_path=np.asarray(image_paths[:n], dtype=str),
        )
        self.eval_artifacts["torch_eval_outputs_npz"] = npz_path
        if self.mlflow_logger is not None and getattr(self.mlflow_logger, "run_id", None):
            self.eval_artifacts["mlflow_run_id"] = self.mlflow_logger.run_id

    def _log_artifacts_to_mlflow(self) -> None:
        """Log pipeline artifacts to MLflow when logging is enabled.

        Example:
            >>> # pipe._log_artifacts_to_mlflow()  # doctest: +SKIP
            >>> True
            True
        """
        if not self.use_mlflow or self.mlflow_logger is None:
            return

        if not self.artifact_catalog:
            self.artifact_catalog = self._build_artifact_catalog()

        seen: set[str] = set()
        for section in self.ARTIFACT_SECTION_ORDER:
            section_map = self.artifact_catalog.get(section, {})
            artifact_subdir = self.ARTIFACT_SECTION_MLFLOW_DIRS.get(section, section)
            if not isinstance(section_map, dict):
                continue
            for _, path in section_map.items():
                if not isinstance(path, str) or not os.path.isfile(path):
                    continue
                rp = os.path.abspath(path)
                if rp in seen:
                    continue
                seen.add(rp)
                try:
                    self.mlflow_logger.log_artifact(path, artifact_path=artifact_subdir)
                except Exception:
                    pass
