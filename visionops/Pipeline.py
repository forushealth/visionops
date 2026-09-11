"""Unified train/evaluate/explain pipeline for keras and torch backends."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from .HpUtils import resolve_stage_from_yaml
from .HpUtils import freeze_preprocessing_manifest
from .EvaluationUtils import evaluate_task
from .TaskSpec import TaskSpec
from .UnifiedTrainer import UnifiedTrainer
from ._dfperf import DfPerfLifecycleMixin
from ._pipeline_facade import PipelineFacadeMixin
from ._utils import run_cleanup_steps


class Pipeline(PipelineFacadeMixin, DfPerfLifecycleMixin):
    """High-level pipeline for unified train/evaluate/explain workflows.

    Supports both `keras` and `torch` backends with optional shared MLflow logging.
    """

    def __init__(
        self,
        backend: str,
        config_path: Optional[str] = None,
        use_mlflow: bool = False,
        task_spec: Optional[TaskSpec] = None,
        task_type: str = "mcc",
        num_classes: int = 2,
        run_dfperf_preflight: bool = True,
        dfperf_outdir: str = "artifacts/dfperf",
        dfperf_strict: bool = False,
        dfperf_sample_gpu: bool = False,
        dfperf_sample_seconds: float = 5.0,
        dfperf_sample_interval: float = 1.0,
        dfperf_monitor_during_training: bool = True,
        dfperf_generate_report: bool = True,
        dfperf_generate_plots: bool = True,
        dfperf_tag: Optional[str] = None,
    ) -> None:
        """Initialize a full pipeline with training, evaluation, and explainers.

        Example:
            >>> pipe = Pipeline(backend="torch", config_path="config.yaml", use_mlflow=False)
            >>> pipe.backend
            'torch'
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

        self._trainer = UnifiedTrainer(backend=backend, config_path=config_path)
        self._fit_runner = None
        self.model = None
        self.mlflow_logger = None
        self.dfperf_report = None
        self.dfperf_runtime_report = None
        self.preprocessing_manifest_path: Optional[str] = None

    def fit(self, **kwargs: Any) -> Any:
        """Train model for the configured backend.

        Keras:
            Uses `KerasFitTrainer` and stores trained model in `self.model`.

        Torch:
            Builds model/datamodule from config when omitted, then forwards to
            `UnifiedTrainer.fit(...)`.

        Example:
            >>> # pipe.fit(max_epochs=1)  # doctest: +SKIP
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
                ("dfperf artifact logging", self._log_dfperf_artifacts_to_mlflow),
            ]
            if self.use_mlflow and self.mlflow_logger is not None:
                cleanup_steps.append(("MLflow run finalization", self.mlflow_logger.end_run))
            run_cleanup_steps(cleanup_steps, primary_error=training_error)

        return result

    def _log_dfperf_artifacts_to_mlflow(self) -> None:
        """Log preflight/runtime artifacts to MLflow.

        Example:
            >>> # pipe._log_dfperf_artifacts_to_mlflow()  # doctest: +SKIP
            >>> True
            True
        """
        if not self.use_mlflow or self.mlflow_logger is None:
            return

        # Preflight artifact
        if self.dfperf_report and isinstance(self.dfperf_report, dict):
            preflight_path = self.dfperf_report.get("preflight_json")
            if preflight_path:
                try:
                    self.mlflow_logger.log_artifact(preflight_path)
                except Exception:
                    pass

        # Runtime artifacts
        if self.dfperf_runtime_report and isinstance(self.dfperf_runtime_report, dict):
            artifact_map = self.dfperf_runtime_report.get("artifacts", {})
            if isinstance(artifact_map, dict):
                for path in artifact_map.values():
                    if isinstance(path, str) and path:
                        try:
                            self.mlflow_logger.log_artifact(path)
                        except Exception:
                            pass

    def _fit_keras(self, **kwargs: Any) -> Any:
        """Train using Keras backend with stage-resolved config.

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
        from .MLflowLogger import create_keras_callback

        effective_cfg = resolve_stage_from_yaml(
            self.config_path,
            stage="train",
            validate=False,
            snapshot_dir=os.path.join(self.dfperf_outdir, "config"),
        )
        effective_cfg["task_type"] = str(self.task_spec.task_type)
        effective_cfg["num_classes"] = int(self.task_spec.num_classes)
        runner = KerasFitTrainer(self.config_path, config=effective_cfg)

        if self.use_mlflow:
            if self.mlflow_logger is None:
                self.setup_mlflow()

            # Avoid duplicate run creation inside KerasFitTrainer.
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
            if self.use_mlflow and self.mlflow_logger is not None:
                self.mlflow_logger.log_artifact(self.preprocessing_manifest_path)
        except Exception:
            self.preprocessing_manifest_path = None
        self._fit_runner = runner
        self.model = runner.model
        return history

    def _fit_torch(self, **kwargs: Any) -> Any:
        """Train with torch backend and auto-build components when omitted.

        Example:
            >>> # pipe = Pipeline(backend="torch", config_path="config.yaml")
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
                datamodule = TaskDataLoaders(
                    config_dict=effective_cfg if effective_cfg else {},
                    batch_size=batch_size,
                    task_type=str(self.task_spec.task_type),
                    num_classes=int(self.task_spec.num_classes),
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

        return result

    def evaluate(self, **kwargs: Any) -> Dict[str, float]:
        """Run unified evaluation for configured backend.

        Example:
            >>> # pipe.evaluate(x=x_val, y=y_val)  # doctest: +SKIP
            >>> True
            True
        """
        model = kwargs.pop("model", None) or self.model
        if model is None:
            raise ValueError("No model available. Train first or pass model=... to evaluate().")

        task_spec = kwargs.pop("task_spec", None) or self.task_spec
        return evaluate_task(model=model, backend=self.backend, task_spec=task_spec, **kwargs)
