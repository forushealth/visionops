"""MLflow logging helpers with lazy backend integration."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

LOGGER = logging.getLogger(__name__)


class ConfigLoader:
    """Load YAML configuration and expand path templates."""

    def __init__(self, config_path: str = "config.yaml") -> None:
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        try:
            with open(config_path, encoding="utf-8") as file:
                config = yaml.safe_load(file) or {}
        except yaml.YAMLError as exc:
            raise ValueError(f"Error parsing YAML config file: {exc}") from exc
        if not isinstance(config, dict):
            raise ValueError("MLflow configuration must be a YAML mapping")

        self.config = config
        self.format_paths()

    def format_paths(self) -> None:
        """Expand path values using keys from the experiment section."""
        paths = self.config.get("paths", {})
        experiment = self.config.get("experiment", {})
        if not isinstance(paths, dict) or not isinstance(experiment, dict):
            return
        for key, value in paths.items():
            if not isinstance(value, str):
                continue
            try:
                paths[key] = value.format(**experiment)
            except KeyError:
                continue

    def get(self, section: str, key: str) -> Any:
        """Return a required configuration value."""
        values = self.config.get(section)
        if not isinstance(values, dict) or key not in values:
            raise KeyError(f"Missing configuration value: {section}.{key}")
        return values[key]


class MLflowLogger:
    """Own one MLflow client run without changing global active-run state."""

    def __init__(self, config_path: str = "config.yaml") -> None:
        try:
            from mlflow.tracking import MlflowClient
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "MLflow logging requires the 'tracking' extra: pip install '.[tracking]'"
            ) from exc

        config_path = os.path.abspath(config_path)
        self.config_loader = ConfigLoader(config_path)
        experiment = self.config_loader.config.setdefault("experiment", {})
        if not isinstance(experiment, dict):
            raise ValueError("The experiment configuration must be a mapping")

        experiment_name = str(experiment.setdefault("name", "Default"))
        run_name = str(experiment.setdefault("run_name", "run"))
        tracking_uri = str(
            experiment.setdefault(
                "tracking_uri",
                Path(config_path).parent.joinpath("mlruns").as_uri(),
            )
        )

        self.client = MlflowClient(tracking_uri=tracking_uri)
        existing = self.client.get_experiment_by_name(experiment_name)
        experiment_id = (
            existing.experiment_id
            if existing is not None
            else self.client.create_experiment(experiment_name)
        )
        self.run = self.client.create_run(
            experiment_id,
            tags={"mlflow.runName": run_name},
        )
        self.run_id = self.run.info.run_id
        self._ended = False
        LOGGER.info("MLflow run started: %s", self.run_id)

    def _logging_enabled(self, key: str) -> bool:
        logging_config = self.config_loader.config.get("logging", {})
        if not isinstance(logging_config, dict):
            return True
        return bool(logging_config.get(key, True))

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters to this logger's run."""
        if not self._logging_enabled("log_hyperparams"):
            return
        for key, value in params.items():
            self.client.log_param(self.run_id, str(key), value)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log scalar metrics to this logger's run."""
        if not self._logging_enabled("log_metrics"):
            return
        metric_step = 0 if step is None else int(step)
        for key, value in metrics.items():
            self.client.log_metric(self.run_id, str(key), float(value), step=metric_step)

    def log_artifact(self, filepath: str, artifact_path: Optional[str] = None) -> None:
        """Log one local file to this logger's run."""
        if self._logging_enabled("log_artifacts"):
            self.client.log_artifact(self.run_id, filepath, artifact_path)

    def log_model(self, model: Any, framework: str = "tensorflow") -> None:
        """Save a backend model locally and log the resulting artifact.

        Backend imports occur only for the selected framework. This method does
        not activate or terminate process-global MLflow runs.
        """
        framework = framework.lower().strip()
        if framework == "pytorch":
            import torch

            path = self.config_loader.config.get("paths", {}).get("model_pytorch", "model.pt")
            torch.save(model.state_dict(), path)
        elif framework in {"tensorflow", "keras"}:
            path = self.config_loader.config.get("paths", {}).get("model_tensorflow", "model.keras")
            model.save(path)
        else:
            raise ValueError("framework must be 'pytorch', 'tensorflow', or 'keras'")
        self.log_artifact(str(path))

    def end_run(self, status: str = "FINISHED") -> None:
        """Terminate only the run created by this logger."""
        if not self._ended:
            self.client.set_terminated(self.run_id, status=status)
            self._ended = True


def create_keras_callback(mlflow_logger: MLflowLogger):
    """Create a Keras callback only after TensorFlow is installed."""
    try:
        from tensorflow.keras.callbacks import Callback
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Keras logging requires the 'keras' extra: pip install '.[keras,tracking]'"
        ) from exc

    class _MLflowKerasCallback(Callback):
        def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, float]] = None) -> None:
            if logs:
                mlflow_logger.log_metrics(logs, step=epoch)

        def on_train_end(self, logs: Optional[Dict[str, float]] = None) -> None:
            if self.model is None:
                return
            try:
                path = mlflow_logger.config_loader.config.get("paths", {}).get(
                    "model_keras",
                    "model.keras",
                )
                self.model.save(path)
                mlflow_logger.log_artifact(str(path))
            except (OSError, ValueError) as exc:
                LOGGER.warning("Could not save Keras model: %s", exc)

    return _MLflowKerasCallback()


def create_lightning_callback(
    mlflow_logger: MLflowLogger,
    *,
    log_model_on_train_end: bool = True,
):
    """Create a Lightning callback only after Lightning is installed."""
    try:
        from pytorch_lightning.callbacks import Callback
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Lightning logging requires the 'torch' extra: pip install '.[torch,tracking]'"
        ) from exc

    class _MLflowLightningCallback(Callback):
        def on_train_epoch_end(self, trainer, pl_module) -> None:
            metrics = {}
            for key, value in trainer.callback_metrics.items():
                try:
                    metrics[key] = float(value.item() if hasattr(value, "item") else value)
                except (TypeError, ValueError):
                    continue
            if metrics:
                mlflow_logger.log_metrics(metrics, step=int(trainer.current_epoch))

        def on_train_end(self, trainer, pl_module) -> None:
            if not log_model_on_train_end:
                return
            try:
                mlflow_logger.log_model(pl_module, framework="pytorch")
            except (OSError, ValueError) as exc:
                LOGGER.warning("Could not log Lightning model: %s", exc)

    return _MLflowLightningCallback()


__all__ = [
    "ConfigLoader",
    "MLflowLogger",
    "create_keras_callback",
    "create_lightning_callback",
]
