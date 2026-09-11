"""Evaluation-only orchestration for model-free metrics and optional CAMs."""

from __future__ import annotations

import ast
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd

from .CamUtils import compare_cams, generate_cam, visualise_CAM
from .HpUtils import load_frozen_preprocessing_manifest
from .MetricRegistry import compute_metrics
from .RunEvaluator import Evaluator
from .TaskSpec import TaskSpec
from ._utils import csv_has_column, first_existing, normalize_path_value


class EvaluationPipeline:
    """Evaluation-only pipeline.

    Designed to work without a model by consuming saved predictions + labels.
    Model is only required for CAM generation.
    """

    MANIFEST_CANDIDATE_NAMES = (
        "preprocessing_manifest.yaml",
        "preprocessing_manifest.yml",
    )
    CAM_PATH_COL_CANDIDATES = ("full_path", "path", "image_path", "file_path", "image")
    CONFIG_CANDIDATE_NAMES = (
        "config_effective_train.yaml",
        "config.yaml",
    )
    RUN_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")

    def _save_cam_figure(self, fig: Any, prefix: str) -> Optional[str]:
        """Persist CAM comparison figures for downstream reporting."""
        if fig is None:
            return None
        try:
            outdir = Path(self.outdir) / "cam_images"
            outdir.mkdir(parents=True, exist_ok=True)
            stamp = int(time.time() * 1000)
            out = outdir / f"{prefix}_{stamp}.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            return str(out)
        except Exception:
            return None

    def __init__(
        self,
        task_spec: Optional[TaskSpec] = None,
        task_type: str = "mcc",
        num_classes: int = 2,
        outdir: str = "artifacts/evaluation",
        generate_report: bool = True,
        preprocessing_manifest_path: Optional[str] = None,
    ) -> None:
        """Initialize evaluation pipeline configuration and artifact cache."""
        self.task_spec = task_spec or TaskSpec(task_type=task_type, num_classes=num_classes)
        self.outdir = outdir
        self.generate_report = generate_report
        self.last_report: Optional[Dict[str, Any]] = None
        self._model_cache: Dict[str, Any] = {}
        self.preprocessing_manifest_path = preprocessing_manifest_path
        self.frozen_preprocessing: Optional[Dict[str, Any]] = None
        if preprocessing_manifest_path:
            self._load_preprocessing_manifest(preprocessing_manifest_path)

    def _load_preprocessing_manifest(self, manifest_path: str) -> None:
        """Load and cache frozen preprocessing manifest in evaluation mode."""
        self.preprocessing_manifest_path = manifest_path
        self.frozen_preprocessing = load_frozen_preprocessing_manifest(
            manifest_path,
            evaluation_mode=True,
        )

    @staticmethod
    def _normalize_image_size(image_size: Any) -> tuple[int, int, int]:
        default = (64, 64, 3)
        if isinstance(image_size, (list, tuple)) and len(image_size) >= 2:
            h = int(image_size[0])
            w = int(image_size[1])
            c = int(image_size[2]) if len(image_size) >= 3 else 3
            return (h, w, c)
        return default

    @staticmethod
    def _resolve_existing_path(raw_path: str, base_dirs) -> Optional[str]:
        p = Path(str(raw_path)).expanduser()
        if p.is_file():
            return str(p)
        for base in base_dirs:
            if not base:
                continue
            try:
                cand = (Path(base).expanduser() / str(raw_path)).resolve()
            except Exception:
                cand = Path(base).expanduser() / str(raw_path)
            if cand.is_file():
                return str(cand)
        return None

    @staticmethod
    def _read_yaml(path: Path) -> Dict[str, Any]:
        try:
            import yaml
        except Exception:
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            if isinstance(data, dict):
                return data
        except Exception:
            return {}
        return {}

    @staticmethod
    def _extract_run_id(value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        text = str(value).strip()
        if EvaluationPipeline.RUN_ID_PATTERN.match(text):
            return text
        p = Path(text)
        for part in p.parts:
            if EvaluationPipeline.RUN_ID_PATTERN.match(part):
                return part
        return None

    def _report_metadata(self) -> Dict[str, Any]:
        if isinstance(self.last_report, dict) and isinstance(self.last_report.get("metadata"), dict):
            return dict(self.last_report.get("metadata") or {})
        return {}

    def _resolve_artifact_context(
        self,
        *,
        source: Optional[str] = None,
        mlflow_run_id: Optional[str] = None,
        mlflow_tracking_uri: Optional[str] = None,
        mlflow_artifact_path: Optional[str] = None,
        prefer_last_source: bool = False,
    ) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
        report_meta = self._report_metadata()
        resolved_source = source
        if resolved_source is None and prefer_last_source and isinstance(self.last_report, dict):
            resolved_source = self.last_report.get("source")

        resolved_run_id = mlflow_run_id or report_meta.get("mlflow_run_id")
        resolved_tracking_uri = mlflow_tracking_uri or report_meta.get("mlflow_tracking_uri")
        resolved_artifact_path = mlflow_artifact_path or report_meta.get("mlflow_artifact_path")

        # Allow passing run-id via `source` in notebook cells.
        if resolved_run_id is None and resolved_source and not Path(str(resolved_source)).exists():
            resolved_run_id = str(resolved_source)

        return (
            resolved_source,
            resolved_run_id,
            resolved_tracking_uri,
            resolved_artifact_path,
        )

    def _discover_config_near_source(self, source_path: str) -> Optional[str]:
        p = Path(source_path)
        if not p.exists():
            return None
        start_dir = p.parent if p.is_file() else p
        candidate_rel = [
            "config/config_effective_train.yaml",
            "dfperf_train_run/config/config_effective_train.yaml",
            "config_effective_train.yaml",
            "config.yaml",
        ]
        for cur in [start_dir, *list(start_dir.parents)]:
            for rel in candidate_rel:
                c = cur / rel
                if c.is_file():
                    return str(c)
        return None

    def _discover_config_from_mlflow(
        self,
        run_id: str,
        *,
        tracking_uri: Optional[str] = None,
        artifact_path: Optional[str] = None,
    ) -> Optional[str]:
        try:
            root = self._download_mlflow_artifacts(
                run_id,
                tracking_uri=tracking_uri,
                artifact_path=artifact_path,
            )
        except Exception:
            return None
        if root.is_file() and root.suffix.lower() in {".yaml", ".yml"}:
            return str(root)

        candidate_rel = [
            "config/config_effective_train.yaml",
            "dfperf_train_run/config/config_effective_train.yaml",
            "config_effective_train.yaml",
            "config.yaml",
        ]
        for rel in candidate_rel:
            c = root / rel
            if c.is_file():
                return str(c)

        for name in self.CONFIG_CANDIDATE_NAMES:
            found = sorted(root.rglob(name))
            if found:
                return str(found[0])
        return None

    def _load_cam_config(
        self,
        *,
        source: Optional[str] = None,
        mlflow_run_id: Optional[str] = None,
        mlflow_tracking_uri: Optional[str] = None,
        mlflow_artifact_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        source, mlflow_run_id, mlflow_tracking_uri, mlflow_artifact_path = self._resolve_artifact_context(
            source=source,
            mlflow_run_id=mlflow_run_id,
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_artifact_path=mlflow_artifact_path,
            prefer_last_source=True,
        )
        config_path = None
        if source:
            config_path = self._discover_config_near_source(source)
        if config_path is None and mlflow_run_id:
            config_path = self._discover_config_from_mlflow(
                mlflow_run_id,
                tracking_uri=mlflow_tracking_uri,
                artifact_path=mlflow_artifact_path,
            )
        if config_path is None:
            return {}
        cfg = self._read_yaml(Path(config_path))
        if isinstance(cfg, dict):
            cfg["_config_path"] = str(config_path)
        return cfg

    def _sync_task_spec_from_artifacts(
        self,
        *,
        source: Optional[str] = None,
        mlflow_run_id: Optional[str] = None,
        mlflow_tracking_uri: Optional[str] = None,
        mlflow_artifact_path: Optional[str] = None,
    ) -> None:
        """Best-effort update of task_type/num_classes from saved config artifacts."""
        cfg = self._load_cam_config(
            source=source,
            mlflow_run_id=mlflow_run_id,
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_artifact_path=mlflow_artifact_path,
        )
        if not isinstance(cfg, dict) or not cfg:
            return

        raw_task_type = cfg.get("task_type", getattr(self.task_spec, "task_type", "mcc"))
        raw_num_classes = cfg.get("num_classes", getattr(self.task_spec, "num_classes", 2))
        try:
            self.task_spec = TaskSpec(task_type=raw_task_type, num_classes=int(raw_num_classes))
        except Exception:
            # Keep existing task_spec when artifact config is incomplete/invalid.
            return

    def _discover_manifest_near_source(self, source_path: str) -> Optional[str]:
        """Discover preprocessing manifest from local saved artifacts."""
        p = Path(source_path)
        if not p.exists():
            return None

        start_dir = p.parent if p.is_file() else p
        for cur in [start_dir, *list(start_dir.parents)]:
            for name in self.MANIFEST_CANDIDATE_NAMES:
                direct = cur / name
                if direct.is_file():
                    return str(direct)
                under_config = cur / "config" / name
                if under_config.is_file():
                    return str(under_config)
        return None

    def _discover_manifest_from_mlflow(
        self,
        run_id: str,
        *,
        tracking_uri: Optional[str] = None,
        artifact_path: Optional[str] = None,
    ) -> Optional[str]:
        """Discover preprocessing manifest in MLflow run artifacts."""
        try:
            from mlflow.tracking import MlflowClient
        except Exception:
            return None

        client = MlflowClient(tracking_uri=tracking_uri)

        # Fast-path: known artifact locations without downloading entire run tree.
        candidate_roots = []
        ap = (artifact_path or "").strip("/")
        if ap:
            candidate_roots.extend([ap, f"{ap}/config"])
        else:
            candidate_roots.extend(["", "config"])

        for root in candidate_roots:
            for name in self.MANIFEST_CANDIDATE_NAMES:
                subpath = f"{root}/{name}".strip("/")
                try:
                    local = client.download_artifacts(run_id, subpath)
                except Exception:
                    continue
                lp = Path(local)
                if lp.is_file():
                    return str(lp)

        # Fallback: download root and scan once.
        try:
            local_root = Path(client.download_artifacts(run_id, ap))
        except Exception:
            return None
        for name in self.MANIFEST_CANDIDATE_NAMES:
            found = sorted(local_root.rglob(name))
            if found:
                return str(found[0])
        return None

    def _autoload_preprocessing_from_artifacts(
        self,
        *,
        source: Optional[str] = None,
        mlflow_run_id: Optional[str] = None,
        mlflow_tracking_uri: Optional[str] = None,
        mlflow_artifact_path: Optional[str] = None,
    ) -> None:
        """Auto-load frozen preprocessing from saved artifacts when available."""
        if self.frozen_preprocessing is not None:
            return

        manifest_path: Optional[str] = None
        if source:
            manifest_path = self._discover_manifest_near_source(source)
            if manifest_path is None and not Path(source).exists():
                manifest_path = self._discover_manifest_from_mlflow(
                    source,
                    tracking_uri=mlflow_tracking_uri,
                    artifact_path=mlflow_artifact_path,
                )
        if manifest_path is None and mlflow_run_id:
            manifest_path = self._discover_manifest_from_mlflow(
                mlflow_run_id,
                tracking_uri=mlflow_tracking_uri,
                artifact_path=mlflow_artifact_path,
            )
        if manifest_path:
            self._load_preprocessing_manifest(manifest_path)

    @staticmethod
    def _download_mlflow_artifacts(
        run_id: str,
        *,
        tracking_uri: Optional[str] = None,
        artifact_path: Optional[str] = None,
    ) -> Path:
        """Download MLflow artifacts for a run and return local root path.

        Example:
            >>> # EvaluationPipeline._download_mlflow_artifacts("run_id")  # doctest: +SKIP
            >>> True
            True
        """
        try:
            from mlflow.tracking import MlflowClient
        except Exception as exc:
            raise ValueError("MLflow is required when mlflow_run_id is provided.") from exc

        client = MlflowClient(tracking_uri=tracking_uri)
        artifact_subpath = (artifact_path or "").strip("/")
        try:
            local = client.download_artifacts(run_id, artifact_subpath)
        except Exception:
            # If a specific artifact path was provided but is wrong/missing,
            # fall back to run root and let downstream resolvers discover files.
            if artifact_subpath:
                try:
                    local = client.download_artifacts(run_id, "")
                except Exception:
                    local = None
            else:
                local = None
            if local is None and tracking_uri and str(tracking_uri).startswith("file://"):
                # Some historical runs can have stale `artifact_uri` metadata.
                # Recover by locating `<tracking_uri>/<exp_id>/<run_id>/artifacts` directly.
                root = Path(str(tracking_uri)[7:]).expanduser()
                cands = sorted(root.glob(f"*/{run_id}/artifacts"))
                if cands:
                    fallback_root = cands[0]
                    fallback = fallback_root / artifact_subpath if artifact_subpath else fallback_root
                    if fallback.exists():
                        local = str(fallback)
            if local is None:
                # Tracking URI is not always passed from notebooks.
                # Best-effort fallback: discover local file-store runs under the current workspace.
                cands = sorted(Path.cwd().glob(f"**/mlruns/*/{run_id}/artifacts"))
                if cands:
                    fallback_root = cands[0]
                    fallback = fallback_root / artifact_subpath if artifact_subpath else fallback_root
                    if fallback.exists():
                        local = str(fallback)
            if local is None:
                raise
        return Path(local)

    def _resolve_keras_logits_csv_from_mlflow(
        self,
        run_id: str,
        *,
        tracking_uri: Optional[str] = None,
        artifact_path: Optional[str] = None,
        logits_csv_name: Optional[str] = None,
    ) -> str:
        """Resolve local logits CSV path from MLflow run artifacts.

        Example:
            >>> # pipe._resolve_keras_logits_csv_from_mlflow("run_id")  # doctest: +SKIP
            >>> True
            True
        """
        root = self._download_mlflow_artifacts(
            run_id,
            tracking_uri=tracking_uri,
            artifact_path=artifact_path,
        )
        if root.is_file():
            if root.suffix.lower() == ".csv" and csv_has_column(root, "logits"):
                return str(root)
            raise FileNotFoundError(f"MLflow artifact is not a logits CSV: {root}")

        if logits_csv_name:
            cand = root / logits_csv_name
            if cand.is_file():
                return str(cand)
            raise FileNotFoundError(f"Could not find logits csv in artifacts: {cand}")

        cands = sorted(root.rglob("*val*logit*.csv"))
        if not cands:
            cands = [p for p in root.rglob("*.csv") if csv_has_column(p, "logits")]
        if not cands:
            raise FileNotFoundError("Could not locate validation logits CSV in MLflow run artifacts.")
        return str(cands[0])

    def _collect_metric_history_local(self, run_id: str) -> Dict[str, Dict[int, float]]:
        histories: Dict[str, Dict[int, float]] = {}
        metric_dirs = sorted(Path.cwd().glob(f"**/mlruns/*/{run_id}/metrics"))
        for mdir in metric_dirs:
            if not mdir.is_dir():
                continue
            for metric_file in mdir.iterdir():
                if not metric_file.is_file():
                    continue
                key = metric_file.name
                step_map: Dict[int, float] = histories.setdefault(key, {})
                try:
                    lines = metric_file.read_text(encoding="utf-8").splitlines()
                except Exception:
                    continue
                for line in lines:
                    parts = str(line).strip().split()
                    if len(parts) < 3:
                        continue
                    try:
                        val = float(parts[1])
                        step = int(parts[2])
                    except Exception:
                        continue
                    step_map[step] = val
        return histories

    def _collect_metric_history_mlflow(
        self,
        run_id: str,
        *,
        tracking_uri: Optional[str] = None,
    ) -> Dict[str, Dict[int, float]]:
        try:
            from mlflow.tracking import MlflowClient
        except Exception:
            return {}

        histories: Dict[str, Dict[int, float]] = {}
        try:
            client = MlflowClient(tracking_uri=tracking_uri)
            run = client.get_run(run_id)
            metric_keys = list((run.data.metrics or {}).keys())
        except Exception:
            return {}

        fallback_keys = [
            "train_loss",
            "val_loss",
            "loss",
            "accuracy",
            "acc",
            "train_acc",
            "val_acc",
            "train_accuracy",
            "val_accuracy",
            "train_acc_epoch",
            "val_acc_epoch",
            "train_accuracy_epoch",
            "val_accuracy_epoch",
        ]
        for key in fallback_keys:
            if key not in metric_keys:
                metric_keys.append(key)

        for key in metric_keys:
            try:
                hist = client.get_metric_history(run_id, key)
            except Exception:
                continue
            if not hist:
                continue
            step_map: Dict[int, float] = {}
            for m in hist:
                try:
                    step_map[int(m.step)] = float(m.value)
                except Exception:
                    continue
            if step_map:
                histories[key] = step_map
        return histories

    def _materialize_history_csv_from_run(
        self,
        run_id: str,
        *,
        outdir: Optional[str] = None,
        tracking_uri: Optional[str] = None,
    ) -> Optional[str]:
        histories = self._collect_metric_history_local(run_id)
        if not histories:
            histories = self._collect_metric_history_mlflow(run_id, tracking_uri=tracking_uri)
        if not histories:
            return None

        def _pick_series(keys: list[str]) -> Optional[Dict[int, float]]:
            for k in keys:
                if k in histories and histories[k]:
                    return histories[k]
            return None

        selected = {
            "loss": _pick_series(["loss", "train_loss", "training_loss"]),
            "val_loss": _pick_series(["val_loss", "validation_loss", "valid_loss"]),
            "accuracy": _pick_series(["accuracy", "acc", "train_accuracy", "train_acc", "train_accuracy_epoch", "train_acc_epoch"]),
            "val_accuracy": _pick_series(["val_accuracy", "val_acc", "validation_accuracy", "valid_accuracy", "val_accuracy_epoch", "val_acc_epoch"]),
        }

        available = [v for v in selected.values() if v]
        if not available:
            return None

        all_steps = sorted({int(step) for series in available for step in series.keys()})
        if not all_steps:
            return None

        df = pd.DataFrame(index=all_steps)
        for col, series in selected.items():
            if not series:
                continue
            df[col] = [series.get(step, np.nan) for step in all_steps]

        if df.empty:
            return None

        dest_dir = Path(outdir or self.outdir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        out_csv = dest_dir / "train_history.csv"
        df.reset_index(drop=True).to_csv(out_csv, index=False)
        return str(out_csv)

    def _resolve_npz_from_mlflow(
        self,
        run_id: str,
        *,
        tracking_uri: Optional[str] = None,
        artifact_path: Optional[str] = None,
        npz_name: Optional[str] = None,
    ) -> str:
        """Resolve local NPZ evaluation artifact path from MLflow run artifacts.

        Example:
            >>> # pipe._resolve_npz_from_mlflow("run_id")  # doctest: +SKIP
            >>> True
            True
        """
        root = self._download_mlflow_artifacts(
            run_id,
            tracking_uri=tracking_uri,
            artifact_path=artifact_path,
        )
        if root.is_file():
            if root.suffix.lower() == ".npz":
                return str(root)
            raise FileNotFoundError(f"MLflow artifact is not an npz file: {root}")

        if npz_name:
            cand = root / npz_name
            if cand.is_file():
                return str(cand)
            raise FileNotFoundError(f"Could not find npz in artifacts: {cand}")

        preferred = sorted(root.rglob("*eval_outputs*.npz"))
        if preferred:
            return str(preferred[0])

        cands = sorted(root.rglob("*.npz"))
        if not cands:
            raise FileNotFoundError("Could not locate eval .npz artifact in MLflow run artifacts.")
        return str(cands[0])

    def _resolve_keras_logits_csv_from_local(
        self,
        path_or_dir: str,
        *,
        logits_csv_name: Optional[str] = None,
    ) -> Optional[str]:
        """Resolve a local logits CSV from a file path or artifact root directory."""
        root = Path(path_or_dir).expanduser()
        if not root.exists():
            return None
        if root.is_file():
            return str(root) if root.suffix.lower() == ".csv" else None

        if logits_csv_name:
            direct = root / logits_csv_name
            if direct.is_file():
                return str(direct)
            named = sorted(root.rglob(logits_csv_name))
            if named:
                return str(named[0])

        preferred = sorted(root.rglob("*val*logit*.csv"))
        if preferred:
            return str(preferred[0])

        cands = [p for p in root.rglob("*.csv") if csv_has_column(p, "logits")]
        if cands:
            return str(cands[0])
        return None

    @staticmethod
    def _resolve_npz_from_local(
        path_or_dir: str,
        *,
        npz_name: Optional[str] = None,
    ) -> Optional[str]:
        """Resolve a local evaluation NPZ from a file path or artifact root directory."""
        root = Path(path_or_dir).expanduser()
        if not root.exists():
            return None
        if root.is_file():
            return str(root) if root.suffix.lower() == ".npz" else None

        if npz_name:
            direct = root / npz_name
            if direct.is_file():
                return str(direct)
            named = sorted(root.rglob(npz_name))
            if named:
                return str(named[0])

        preferred = sorted(root.rglob("*eval_outputs*.npz"))
        if preferred:
            return str(preferred[0])
        cands = sorted(root.rglob("*.npz"))
        if cands:
            return str(cands[0])
        return None

    @staticmethod
    def _load_keras_model(model_path: str, custom_objects: Optional[Dict[str, Any]] = None):
        import tensorflow as tf

        return tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)

    @staticmethod
    def _load_torch_model(
        model_path: str,
        *,
        task_type: str = "mcc",
        num_classes: int = 2,
    ):
        import torch

        obj = torch.load(
            model_path,
            map_location=torch.device("cpu"),
            weights_only=True,
        )
        state_dict = None
        if isinstance(obj, dict):
            if isinstance(obj.get("state_dict"), dict):
                state_dict = obj.get("state_dict")
            elif isinstance(obj.get("model_state_dict"), dict):
                state_dict = obj.get("model_state_dict")
            elif obj and all(hasattr(v, "shape") for v in obj.values()):
                state_dict = obj

        if isinstance(state_dict, dict):
            from .PLTrainingUtils import TaskClassifier

            model = TaskClassifier(
                task_type=str(task_type),
                num_classes=int(num_classes),
                pretrained=False,
            )
            try:
                model.load_state_dict(state_dict, strict=False)
            except RuntimeError:
                # Common checkpoint layouts: `module.*` or backbone-only `model.*`.
                cleaned = {}
                for key, value in state_dict.items():
                    normalized_key = str(key)
                    if normalized_key.startswith("module."):
                        normalized_key = normalized_key[len("module.") :]
                    if normalized_key.startswith("model."):
                        normalized_key = normalized_key[len("model.") :]
                    cleaned[normalized_key] = value
                model.model.load_state_dict(cleaned, strict=False)
            model.eval()
            return model
        raise ValueError(
            "torch model file does not contain a loadable state_dict. "
            "Pass model=... directly, or use model_loader=... for custom checkpoint formats."
        )

    @staticmethod
    def _find_keras_model_path(root: Path) -> Optional[str]:
        if root.is_file():
            if root.suffix.lower() in {".keras", ".h5", ".hdf5"}:
                return str(root)
            return None

        if (root / "saved_model.pb").is_file():
            return str(root)

        for pat in ("*.keras", "*.h5", "*.hdf5"):
            cands = sorted(root.rglob(pat))
            if cands:
                return str(cands[0])

        for pb in sorted(root.rglob("saved_model.pb")):
            return str(pb.parent)

        return None

    @staticmethod
    def _find_torch_model_path(root: Path) -> Optional[str]:
        if root.is_file():
            if root.suffix.lower() in {".pt", ".pth", ".ckpt"} or root.name.lower() == "model":
                return str(root)
            return None

        for pat in ("*.pt", "*.pth", "*.ckpt"):
            cands = sorted(root.rglob(pat))
            if cands:
                return str(cands[0])
        cands = sorted(p for p in root.rglob("model") if p.is_file())
        if cands:
            return str(cands[0])
        return None

    def _load_model_from_mlflow(
        self,
        *,
        framework: str,
        mlflow_run_id: str,
        mlflow_tracking_uri: Optional[str] = None,
        mlflow_artifact_path: Optional[str] = None,
        model_loader: Optional[Callable[[str], Any]] = None,
        custom_objects: Optional[Dict[str, Any]] = None,
    ) -> Any:
        try:
            import mlflow
        except Exception as exc:
            raise ValueError("MLflow is required when mlflow_run_id is provided.") from exc

        framework = framework.lower().strip()
        artifact_path = (mlflow_artifact_path or "").strip("/")
        preferred_artifact_path = artifact_path or "model"
        local_root = self._download_mlflow_artifacts(
            mlflow_run_id,
            tracking_uri=mlflow_tracking_uri,
            artifact_path=preferred_artifact_path,
        )

        if framework == "keras":
            if (local_root / "MLmodel").is_file():
                try:
                    return mlflow.tensorflow.load_model(str(local_root))
                except Exception:
                    pass

            model_path = self._find_keras_model_path(local_root)
            if model_path:
                return self._load_keras_model(model_path, custom_objects=custom_objects)
            raise FileNotFoundError(
                f"Could not locate a keras model artifact in run_id={mlflow_run_id} under '{artifact_path or '/'}'."
            )

        if framework == "torch":
            if (local_root / "MLmodel").is_file():
                try:
                    model = mlflow.pytorch.load_model(str(local_root))
                    model.eval()
                    return model
                except Exception:
                    pass
            model_path = self._find_torch_model_path(local_root)
            if model_path:
                if model_loader is not None:
                    return model_loader(model_path)
                return self._load_torch_model(
                    model_path,
                    task_type=str(self.task_spec.task_type),
                    num_classes=int(self.task_spec.num_classes),
                )
            raise FileNotFoundError(
                f"Could not locate a torch model artifact in run_id={mlflow_run_id} under '{artifact_path or '/'}'."
            )

        raise ValueError("framework must be 'torch' or 'keras'")

    def _resolve_model(
        self,
        *,
        framework: str,
        model: Any = None,
        model_path: Optional[str] = None,
        mlflow_run_id: Optional[str] = None,
        mlflow_tracking_uri: Optional[str] = None,
        mlflow_artifact_path: Optional[str] = None,
        model_loader: Optional[Callable[[str], Any]] = None,
        custom_objects: Optional[Dict[str, Any]] = None,
    ) -> Any:
        if model is not None:
            return model

        key = "|".join(
            [
                framework.lower().strip(),
                model_path or "",
                mlflow_run_id or "",
                mlflow_tracking_uri or "",
                mlflow_artifact_path or "",
            ]
        )
        can_cache = model_loader is None and not custom_objects
        if can_cache and key in self._model_cache:
            return self._model_cache[key]

        resolved = None
        if model_path:
            if model_loader is not None:
                resolved = model_loader(model_path)
            elif framework.lower().strip() == "keras":
                resolved = self._load_keras_model(model_path, custom_objects=custom_objects)
            elif framework.lower().strip() == "torch":
                resolved = self._load_torch_model(
                    model_path,
                    task_type=str(self.task_spec.task_type),
                    num_classes=int(self.task_spec.num_classes),
                )
            else:
                raise ValueError("framework must be 'torch' or 'keras'")
        elif mlflow_run_id:
            resolved = self._load_model_from_mlflow(
                framework=framework,
                mlflow_run_id=mlflow_run_id,
                mlflow_tracking_uri=mlflow_tracking_uri,
                mlflow_artifact_path=mlflow_artifact_path,
                model_loader=model_loader,
                custom_objects=custom_objects,
            )

        if resolved is None:
            raise ValueError(
                "CAM generation requires a model. Provide one of: "
                "model=..., model_path='...', or mlflow_run_id='...'."
            )

        if can_cache:
            self._model_cache[key] = resolved
        return resolved

    def evaluate_arrays(
        self,
        y_true: Any,
        y_pred: Any,
        *,
        tag: str = "evaluation",
        source: str = "arrays",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Evaluate directly from in-memory truth/prediction arrays."""
        meta = dict(metadata or {})
        if self.preprocessing_manifest_path:
            meta.setdefault("preprocessing_manifest_path", self.preprocessing_manifest_path)
            meta.setdefault("preprocessing_loaded", self.frozen_preprocessing is not None)

        metrics = compute_metrics(y_true=y_true, y_pred=y_pred, task_spec=self.task_spec)
        report = {
            "tag": tag,
            "task_type": str(self.task_spec.task_type),
            "num_classes": self.task_spec.num_classes,
            "source": source,
            "metrics": metrics,
            "metadata": meta,
            "status": "ok",
        }

        artifacts = self._write_report_artifacts(report) if self.generate_report else {}
        report["artifacts"] = artifacts
        self.last_report = report
        return report

    def evaluate_npz(
        self,
        npz_path: str,
        *,
        y_true_key: str = "y_true",
        y_pred_key: str = "y_pred",
        tag: str = "evaluation_npz",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Evaluate from an ``.npz`` file containing ``y_true`` and ``y_pred`` keys."""
        self._autoload_preprocessing_from_artifacts(source=npz_path)
        with np.load(npz_path, allow_pickle=False) as payload:
            missing = [key for key in (y_true_key, y_pred_key) if key not in payload]
            if missing:
                raise ValueError(f"NPZ is missing required keys: {', '.join(missing)}")
            y_true = np.asarray(payload[y_true_key])
            y_pred = np.asarray(payload[y_pred_key])
        return self.evaluate_arrays(
            y_true=y_true,
            y_pred=y_pred,
            tag=tag,
            source=npz_path,
            metadata=metadata,
        )

    def evaluate_keras_logits_csv(
        self,
        csv_path: str,
        *,
        y_true_col: str = "label",
        logits_col: str = "logits",
        tag: str = "evaluation_keras_csv",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Evaluate from Keras logits CSV emitted by training pipeline."""
        self._autoload_preprocessing_from_artifacts(source=csv_path)
        df = pd.read_csv(csv_path)
        if y_true_col not in df.columns or logits_col not in df.columns:
            raise ValueError(f"CSV must contain '{y_true_col}' and '{logits_col}' columns")

        def _parse_cell(v):
            if isinstance(v, str):
                try:
                    parsed = ast.literal_eval(v)
                except Exception:
                    try:
                        return float(v)
                    except Exception:
                        return v
                return parsed
            return v

        y_true_parsed = [_parse_cell(v) for v in df[y_true_col].tolist()]
        y_true_np_obj = np.array(y_true_parsed, dtype=object)
        try:
            if y_true_np_obj.ndim == 1 and len(y_true_np_obj) > 0 and isinstance(
                y_true_np_obj[0], (list, tuple, np.ndarray)
            ):
                y_true = np.array([np.array(x).reshape(-1) for x in y_true_parsed], dtype=int)
            else:
                y_true = np.array(y_true_parsed, dtype=int)
        except Exception:
            y_true = np.array(y_true_parsed)

        parsed = [_parse_cell(v) for v in df[logits_col].tolist()]
        parsed_np = np.array(parsed, dtype=object)

        # Convert object list to numeric tensor when possible
        try:
            if parsed_np.ndim == 1 and len(parsed_np) > 0 and isinstance(parsed_np[0], (list, tuple, np.ndarray)):
                y_pred = np.array([np.array(x).reshape(-1) for x in parsed], dtype=float)
            else:
                y_pred = np.array(parsed, dtype=float)
        except Exception:
            y_pred = np.array(parsed)

        return self.evaluate_arrays(
            y_true=y_true,
            y_pred=y_pred,
            tag=tag,
            source=csv_path,
            metadata=metadata,
        )

    def _materialize_logits_csv_from_npz(
        self,
        npz_path: str,
        *,
        outdir: Optional[str] = None,
        y_true_key: str = "y_true",
        y_pred_key: str = "y_pred",
    ) -> str:
        """Create a temporary logits-like CSV from an eval NPZ artifact.

        This enables `Evaluator` advanced plots/stats to run for torch outputs.

        Example:
            >>> # pipe._materialize_logits_csv_from_npz("eval_outputs.npz")  # doctest: +SKIP
            >>> True
            True
        """
        with np.load(npz_path, allow_pickle=False) as payload:
            if y_true_key not in payload or y_pred_key not in payload:
                raise ValueError(f"NPZ must contain keys '{y_true_key}' and '{y_pred_key}'.")

            y_true = np.asarray(payload[y_true_key])
            y_pred = np.asarray(payload[y_pred_key])
            path_values: Optional[list[str]] = None
            for path_key in ("image_path", "full_path", "path", "paths", "file_path"):
                if path_key not in payload:
                    continue
                raw = np.asarray(payload[path_key], dtype=str).reshape(-1).tolist()
                path_values = [normalize_path_value(v) for v in raw]
                break

        if y_true.ndim > 1:
            if y_true.shape[-1] == 1:
                y_true = y_true.reshape(-1)
            else:
                y_true = np.argmax(y_true, axis=-1)
        y_true = y_true.astype(int).reshape(-1)

        if y_pred.ndim == 1:
            logits_col = [[float(v)] for v in y_pred.tolist()]
        else:
            logits_col = [np.array(row, dtype=float).reshape(-1).tolist() for row in y_pred]

        n = min(len(y_true), len(logits_col))
        if path_values is not None:
            n = min(n, len(path_values))
        if path_values is None:
            path_values = [""] * n
        elif len(path_values) < n:
            path_values = path_values + ([""] * (n - len(path_values)))

        df = pd.DataFrame(
            {
                "image_path": path_values[:n],
                "label": y_true[:n].tolist(),
                "logits": logits_col[:n],
            }
        )

        dest_dir = Path(outdir or self.outdir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        out_csv = dest_dir / "_tmp_evaluator_logits_from_npz.csv"
        df.to_csv(out_csv, index=False)
        return str(out_csv)

    def evaluate_from_artifacts(
        self,
        *,
        backend: str,
        logits_csv_path: Optional[str] = None,
        npz_path: Optional[str] = None,
        mlflow_run_id: Optional[str] = None,
        mlflow_tracking_uri: Optional[str] = None,
        mlflow_artifact_path: Optional[str] = None,
        logits_csv_name: Optional[str] = None,
        npz_name: Optional[str] = None,
        y_true_col: str = "label",
        logits_col: str = "logits",
        y_true_key: str = "y_true",
        y_pred_key: str = "y_pred",
        tag: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Evaluate from local artifacts or MLflow run artifacts, without train pipeline objects.

        Example:
            >>> # pipe.evaluate_from_artifacts(backend="keras", logits_csv_path="val_logits.csv")  # doctest: +SKIP
            >>> True
            True
        """
        mode = str(backend).lower().strip()
        if mode not in {"keras", "torch"}:
            raise ValueError("backend must be 'keras' or 'torch'")

        meta = dict(metadata or {})
        if mlflow_run_id:
            meta.setdefault("mlflow_run_id", mlflow_run_id)
        if mlflow_tracking_uri:
            meta.setdefault("mlflow_tracking_uri", mlflow_tracking_uri)
        if mlflow_artifact_path:
            meta.setdefault("mlflow_artifact_path", mlflow_artifact_path)
        if mlflow_run_id:
            # Prefer task metadata recorded with the run when available.
            self._sync_task_spec_from_artifacts(
                source=logits_csv_path or npz_path,
                mlflow_run_id=mlflow_run_id,
                mlflow_tracking_uri=mlflow_tracking_uri,
                mlflow_artifact_path=mlflow_artifact_path,
            )

        if mode == "keras":
            csv_path = logits_csv_path
            if csv_path:
                resolved_csv = self._resolve_keras_logits_csv_from_local(
                    str(csv_path),
                    logits_csv_name=logits_csv_name,
                )
                if resolved_csv:
                    csv_path = resolved_csv
                elif Path(str(csv_path)).expanduser().is_dir():
                    csv_path = None
            if not csv_path and mlflow_run_id:
                csv_path = self._resolve_keras_logits_csv_from_mlflow(
                    mlflow_run_id,
                    tracking_uri=mlflow_tracking_uri,
                    artifact_path=mlflow_artifact_path,
                    logits_csv_name=logits_csv_name,
                )
            if not csv_path:
                raise ValueError(
                    "For backend='keras', provide logits_csv_path or mlflow_run_id."
                )
            self._autoload_preprocessing_from_artifacts(
                source=csv_path,
                mlflow_run_id=mlflow_run_id,
                mlflow_tracking_uri=mlflow_tracking_uri,
                mlflow_artifact_path=mlflow_artifact_path,
            )
            return self.evaluate_keras_logits_csv(
                str(csv_path),
                y_true_col=y_true_col,
                logits_col=logits_col,
                tag=tag or "evaluation_keras_artifacts",
                metadata=meta,
            )

        resolved_npz = npz_path
        if resolved_npz:
            resolved_local_npz = self._resolve_npz_from_local(
                str(resolved_npz),
                npz_name=npz_name,
            )
            if resolved_local_npz:
                resolved_npz = resolved_local_npz
            elif Path(str(resolved_npz)).expanduser().is_dir():
                resolved_npz = None
        torch_csv_from_mlflow = None
        if not resolved_npz and mlflow_run_id:
            try:
                resolved_npz = self._resolve_npz_from_mlflow(
                    mlflow_run_id,
                    tracking_uri=mlflow_tracking_uri,
                    artifact_path=mlflow_artifact_path,
                    npz_name=npz_name,
                )
            except Exception:
                resolved_npz = None
                try:
                    torch_csv_from_mlflow = self._resolve_keras_logits_csv_from_mlflow(
                        mlflow_run_id,
                        tracking_uri=mlflow_tracking_uri,
                        artifact_path=mlflow_artifact_path,
                        logits_csv_name=logits_csv_name,
                    )
                except Exception:
                    torch_csv_from_mlflow = None
        if not resolved_npz and torch_csv_from_mlflow:
            self._autoload_preprocessing_from_artifacts(
                source=torch_csv_from_mlflow,
                mlflow_run_id=mlflow_run_id,
                mlflow_tracking_uri=mlflow_tracking_uri,
                mlflow_artifact_path=mlflow_artifact_path,
            )
            meta.setdefault("source_backend", "torch")
            return self.evaluate_keras_logits_csv(
                str(torch_csv_from_mlflow),
                y_true_col=y_true_col,
                logits_col=logits_col,
                tag=tag or "evaluation_torch_artifacts",
                metadata=meta,
            )
        if not resolved_npz and npz_path:
            # Fallback for artifact roots that only contain evaluator CSV outputs.
            torch_csv = self._resolve_keras_logits_csv_from_local(
                str(npz_path),
                logits_csv_name="_tmp_evaluator_logits_from_npz.csv",
            )
            if torch_csv:
                self._autoload_preprocessing_from_artifacts(
                    source=torch_csv,
                    mlflow_run_id=mlflow_run_id,
                    mlflow_tracking_uri=mlflow_tracking_uri,
                    mlflow_artifact_path=mlflow_artifact_path,
                )
                meta.setdefault("source_backend", "torch")
                return self.evaluate_keras_logits_csv(
                    str(torch_csv),
                    y_true_col=y_true_col,
                    logits_col=logits_col,
                    tag=tag or "evaluation_torch_artifacts",
                    metadata=meta,
                )
        if not resolved_npz:
            raise ValueError(
                "For backend='torch', provide npz_path, mlflow_run_id, "
                "or an artifact root containing eval_outputs*.npz."
            )
        self._autoload_preprocessing_from_artifacts(
            source=resolved_npz,
            mlflow_run_id=mlflow_run_id,
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_artifact_path=mlflow_artifact_path,
        )
        return self.evaluate_npz(
            str(resolved_npz),
            y_true_key=y_true_key,
            y_pred_key=y_pred_key,
            tag=tag or "evaluation_torch_artifacts",
            metadata=meta,
        )

    def load_cam_images(
        self,
        *,
        source: Optional[str] = None,
        mlflow_run_id: Optional[str] = None,
        mlflow_tracking_uri: Optional[str] = None,
        mlflow_artifact_path: Optional[str] = None,
        image_number: int = 8,
        image_size: Optional[tuple[int, int, int]] = None,
    ) -> Dict[str, Any]:
        """Load CAM-ready images from saved artifacts/config instead of notebook dataframes.

        Example:
            >>> # out = pipe.load_cam_images(source='artifacts/eval_artifacts/val_logits.csv')  # doctest: +SKIP
            >>> True
            True
        """
        if image_number <= 0:
            raise ValueError("image_number must be >= 1")

        src, mlflow_run_id, mlflow_tracking_uri, mlflow_artifact_path = self._resolve_artifact_context(
            source=source,
            mlflow_run_id=mlflow_run_id,
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_artifact_path=mlflow_artifact_path,
            prefer_last_source=True,
        )

        cfg = self._load_cam_config(
            source=src,
            mlflow_run_id=mlflow_run_id,
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_artifact_path=mlflow_artifact_path,
        )
        cfg_path = cfg.get("_config_path")
        cfg_dir = Path(cfg_path).parent if cfg_path else None

        inferred_size = self._normalize_image_size(image_size if image_size is not None else cfg.get("image_size"))
        h, w, _ = inferred_size

        base_dirs = [Path.cwd()]
        data_dir = cfg.get("data_directory_path")
        if data_dir:
            base_dirs.append(Path(str(data_dir)))
        if cfg_dir is not None:
            base_dirs.append(cfg_dir)
            base_dirs.extend(list(cfg_dir.parents))

        path_values = []

        # Primary source: logits CSV (contains full_path/path).
        if src and Path(str(src)).is_file() and str(src).lower().endswith(".csv"):
            try:
                df_src = pd.read_csv(str(src))
                col = first_existing(df_src.columns, self.CAM_PATH_COL_CANDIDATES)
                if col:
                    path_values = df_src[col].astype(str).tolist()
                    base_dirs.append(Path(str(src)).parent)
            except Exception:
                path_values = []

        # Fallback source: validation CSV from saved config.
        if not path_values:
            val_csv = cfg.get("val_data_csv")
            if val_csv:
                val_csv_path = self._resolve_existing_path(str(val_csv), base_dirs)
                if val_csv_path:
                    try:
                        df_val = pd.read_csv(val_csv_path)
                        col = cfg.get("image_path_column") or first_existing(
                            df_val.columns, self.CAM_PATH_COL_CANDIDATES
                        )
                        if col and col in df_val.columns:
                            path_values = df_val[col].astype(str).tolist()
                            base_dirs.append(Path(val_csv_path).parent)
                    except Exception:
                        path_values = []

        # MLflow-only fallback: scan downloaded run artifacts for validation CSV.
        if not path_values and mlflow_run_id:
            try:
                root = self._download_mlflow_artifacts(
                    mlflow_run_id,
                    tracking_uri=mlflow_tracking_uri,
                    artifact_path=mlflow_artifact_path,
                )
                csv_cands = sorted(root.rglob("*val*.csv"))
                if not csv_cands:
                    csv_cands = sorted(root.rglob("*.csv"))
                for cand in csv_cands:
                    try:
                        df_c = pd.read_csv(cand)
                    except Exception:
                        continue
                    col = first_existing(df_c.columns, self.CAM_PATH_COL_CANDIDATES)
                    if col:
                        path_values = df_c[col].astype(str).tolist()
                        base_dirs.append(cand.parent)
                        break
            except Exception:
                path_values = []

        if not path_values:
            raise ValueError(
                "Could not find image paths for CAM visualisation from source/config/artifacts. "
                "Provide `source=...` pointing to a CSV with one of "
                f"{self.CAM_PATH_COL_CANDIDATES}, or log validation CSV/config artifacts in MLflow."
            )

        from PIL import Image

        loaded_images = []
        loaded_paths = []
        skipped = 0
        for raw in path_values:
            if len(loaded_images) >= int(image_number):
                break
            p = self._resolve_existing_path(normalize_path_value(raw), base_dirs)
            if not p:
                skipped += 1
                continue
            try:
                with Image.open(p) as im:
                    arr = np.array(im.convert("RGB").resize((int(w), int(h))))
                loaded_images.append(arr)
                loaded_paths.append(p)
            except Exception:
                skipped += 1

        if not loaded_images:
            raise ValueError("No CAM-ready images could be loaded from resolved image paths.")

        cam_batch = np.stack(loaded_images, axis=0)
        return {
            "cam_image": cam_batch[0],
            "cam_batch": cam_batch,
            "loaded_paths": loaded_paths,
            "num_loaded": int(len(loaded_paths)),
            "num_skipped": int(skipped),
            "image_size": inferred_size,
            "config_path": cfg_path,
        }

    def compare_cams_from_artifacts(
        self,
        *,
        framework: str,
        model: Any = None,
        model_path: Optional[str] = None,
        mlflow_run_id: Optional[str] = None,
        mlflow_tracking_uri: Optional[str] = None,
        mlflow_artifact_path: Optional[str] = None,
        model_loader: Optional[Callable[[str], Any]] = None,
        custom_objects: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None,
        target_layer: Any = None,
        cam_methods=("gradcam", "gradcamplusplus", "xgradcam", "layercam"),
        class_index: Optional[int] = None,
        figsize=(12, 8),
        overlay: bool = True,
        alpha: float = 0.45,
        show_plot: bool = True,
    ):
        """Compare CAMs by internally loading one image from saved artifacts/config.

        Example:
            >>> # pipe.compare_cams_from_artifacts(framework='keras', model_path='model.h5')  # doctest: +SKIP
            >>> True
            True
        """
        source, mlflow_run_id, mlflow_tracking_uri, mlflow_artifact_path = self._resolve_artifact_context(
            source=source,
            mlflow_run_id=mlflow_run_id,
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_artifact_path=mlflow_artifact_path,
            prefer_last_source=True,
        )
        bundle = self.load_cam_images(
            source=source,
            mlflow_run_id=mlflow_run_id,
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_artifact_path=mlflow_artifact_path,
            image_number=1,
        )
        return self.compare_cams(
            model=model,
            model_path=model_path,
            mlflow_run_id=mlflow_run_id,
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_artifact_path=mlflow_artifact_path,
            model_loader=model_loader,
            custom_objects=custom_objects,
            image=bundle["cam_image"],
            framework=framework,
            target_layer=target_layer,
            cam_methods=cam_methods,
            class_index=class_index,
            figsize=figsize,
            overlay=overlay,
            alpha=alpha,
            show_plot=show_plot,
        )

    def visualise_cams_from_artifacts(
        self,
        *,
        framework: str,
        model: Any = None,
        model_path: Optional[str] = None,
        mlflow_run_id: Optional[str] = None,
        mlflow_tracking_uri: Optional[str] = None,
        mlflow_artifact_path: Optional[str] = None,
        model_loader: Optional[Callable[[str], Any]] = None,
        custom_objects: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None,
        target_layer: Any = None,
        cam_names=("GradCAM", "EigenCAM"),
        class_index: Optional[int] = None,
        image_number: int = 8,
        overlay: bool = True,
        alpha: float = 0.45,
        figsize=None,
        show_plot: bool = True,
    ):
        """Visualise CAM grids by internally loading images from saved artifacts/config.

        Example:
            >>> # pipe.visualise_cams_from_artifacts(framework='keras', model_path='model.h5')  # doctest: +SKIP
            >>> True
            True
        """
        source, mlflow_run_id, mlflow_tracking_uri, mlflow_artifact_path = self._resolve_artifact_context(
            source=source,
            mlflow_run_id=mlflow_run_id,
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_artifact_path=mlflow_artifact_path,
            prefer_last_source=True,
        )
        bundle = self.load_cam_images(
            source=source,
            mlflow_run_id=mlflow_run_id,
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_artifact_path=mlflow_artifact_path,
            image_number=image_number,
        )
        return self.visualise_cams(
            model=model,
            model_path=model_path,
            mlflow_run_id=mlflow_run_id,
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_artifact_path=mlflow_artifact_path,
            model_loader=model_loader,
            custom_objects=custom_objects,
            dataset_array=bundle["cam_batch"],
            framework=framework,
            target_layer=target_layer,
            cam_names=cam_names,
            class_index=class_index,
            image_number=min(int(image_number), int(bundle["cam_batch"].shape[0])),
            overlay=overlay,
            alpha=alpha,
            figsize=figsize,
            show_plot=show_plot,
        )

    def explain(
        self,
        *,
        model: Any = None,
        model_path: Optional[str] = None,
        mlflow_run_id: Optional[str] = None,
        mlflow_tracking_uri: Optional[str] = None,
        mlflow_artifact_path: Optional[str] = None,
        model_loader: Optional[Callable[[str], Any]] = None,
        custom_objects: Optional[Dict[str, Any]] = None,
        image: Any,
        framework: str,
        target_layer: Any = None,
        class_index: Optional[int] = None,
        cam_method: str = "gradcam",
    ):
        """Generate a CAM heatmap using a provided or resolved model."""
        _, mlflow_run_id, mlflow_tracking_uri, mlflow_artifact_path = self._resolve_artifact_context(
            mlflow_run_id=mlflow_run_id,
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_artifact_path=mlflow_artifact_path,
            prefer_last_source=True,
        )
        self._autoload_preprocessing_from_artifacts(
            source=model_path,
            mlflow_run_id=mlflow_run_id,
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_artifact_path=mlflow_artifact_path,
        )
        model_obj = self._resolve_model(
            framework=framework,
            model=model,
            model_path=model_path,
            mlflow_run_id=mlflow_run_id,
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_artifact_path=mlflow_artifact_path,
            model_loader=model_loader,
            custom_objects=custom_objects,
        )

        return generate_cam(
            model=model_obj,
            image=image,
            framework=framework,
            target_layer=target_layer,
            class_index=class_index,
            cam_method=cam_method,
        )

    def build_evaluator(
        self,
        source: Optional[str] = None,
        *,
        output_dir: Optional[str] = None,
        history_csv_path: Optional[str] = None,
        backend: Optional[str] = None,
        y_true_key: str = "y_true",
        y_pred_key: str = "y_pred",
        **kwargs: Any,
    ) -> Evaluator:
        """Create advanced evaluator (plots/threshold/stats/report) for keras/torch artifacts.

        For torch `.npz` outputs, this method auto-materializes a logits-style CSV
        so `Evaluator` can run without backend-specific notebook branches.
        """
        evaluator_kwargs = dict(kwargs)
        report_meta = (
            self.last_report.get("metadata", {})
            if isinstance(self.last_report, dict) and isinstance(self.last_report.get("metadata"), dict)
            else {}
        )

        resolved_source = source
        if not resolved_source and isinstance(self.last_report, dict):
            resolved_source = self.last_report.get("source") or report_meta.get("mlflow_run_id")
        if not resolved_source:
            raise ValueError(
                "No evaluator source found. Pass `source=...` or run `evaluate_from_artifacts(...)` first."
            )

        if evaluator_kwargs.get("tracking_uri") is None and report_meta.get("mlflow_tracking_uri"):
            evaluator_kwargs["tracking_uri"] = report_meta.get("mlflow_tracking_uri")

        artifact_path = evaluator_kwargs.pop("artifact_path", None) or report_meta.get("mlflow_artifact_path")
        source = str(resolved_source)
        resolved_history_csv = history_csv_path

        self._autoload_preprocessing_from_artifacts(
            source=source,
            mlflow_run_id=source if not Path(source).exists() else None,
            mlflow_tracking_uri=evaluator_kwargs.get("tracking_uri"),
            mlflow_artifact_path=artifact_path,
        )

        evaluator_source = source
        source_path = Path(source)
        mode = str(backend or "").lower().strip()

        if source_path.exists() and source_path.is_file() and source_path.suffix.lower() == ".npz":
            evaluator_source = self._materialize_logits_csv_from_npz(
                str(source_path),
                outdir=output_dir or self.outdir,
                y_true_key=y_true_key,
                y_pred_key=y_pred_key,
            )
        elif (not source_path.exists()) and mode == "torch":
            # Treat `source` as MLflow run-id for torch: resolve npz, then materialize CSV.
            try:
                npz_path = self._resolve_npz_from_mlflow(
                    source,
                    tracking_uri=evaluator_kwargs.get("tracking_uri"),
                    artifact_path=artifact_path,
                )
                evaluator_source = self._materialize_logits_csv_from_npz(
                    npz_path,
                    outdir=output_dir or self.outdir,
                    y_true_key=y_true_key,
                    y_pred_key=y_pred_key,
                )
            except Exception:
                # Fallback when run has logits CSV but no NPZ.
                evaluator_source = self._resolve_keras_logits_csv_from_mlflow(
                    source,
                    tracking_uri=evaluator_kwargs.get("tracking_uri"),
                    artifact_path=artifact_path,
                )

        if not resolved_history_csv:
            run_id_for_history = (
                self._extract_run_id(report_meta.get("mlflow_run_id"))
                or self._extract_run_id(source if not source_path.exists() else str(source_path))
            )
            if run_id_for_history:
                resolved_history_csv = self._materialize_history_csv_from_run(
                    run_id_for_history,
                    outdir=output_dir or self.outdir,
                    tracking_uri=evaluator_kwargs.get("tracking_uri"),
                )

        return Evaluator(
            evaluator_source,
            output_dir=output_dir or self.outdir,
            history_csv_path=resolved_history_csv,
            **evaluator_kwargs,
        )

    def compare_cams(
        self,
        *,
        model: Any = None,
        model_path: Optional[str] = None,
        mlflow_run_id: Optional[str] = None,
        mlflow_tracking_uri: Optional[str] = None,
        mlflow_artifact_path: Optional[str] = None,
        model_loader: Optional[Callable[[str], Any]] = None,
        custom_objects: Optional[Dict[str, Any]] = None,
        image: Any,
        framework: str,
        target_layer: Any = None,
        cam_methods=("gradcam", "gradcamplusplus", "xgradcam", "layercam"),
        class_index: Optional[int] = None,
        figsize=(12, 8),
        overlay: bool = True,
        alpha: float = 0.45,
        show_plot: bool = True,
    ):
        """Compare multiple CAM methods for one image."""
        _, mlflow_run_id, mlflow_tracking_uri, mlflow_artifact_path = self._resolve_artifact_context(
            mlflow_run_id=mlflow_run_id,
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_artifact_path=mlflow_artifact_path,
            prefer_last_source=True,
        )
        self._autoload_preprocessing_from_artifacts(
            source=model_path,
            mlflow_run_id=mlflow_run_id,
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_artifact_path=mlflow_artifact_path,
        )
        model_obj = self._resolve_model(
            framework=framework,
            model=model,
            model_path=model_path,
            mlflow_run_id=mlflow_run_id,
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_artifact_path=mlflow_artifact_path,
            model_loader=model_loader,
            custom_objects=custom_objects,
        )

        cams, fig = compare_cams(
            model=model_obj,
            image=image,
            framework=framework,
            target_layer=target_layer,
            cam_methods=cam_methods,
            class_index=class_index,
            figsize=figsize,
            overlay=overlay,
            alpha=alpha,
            show_plot=show_plot,
        )
        self._save_cam_figure(fig, prefix="compare_cams")
        return cams, fig

    def visualise_cams(
        self,
        *,
        model: Any = None,
        model_path: Optional[str] = None,
        mlflow_run_id: Optional[str] = None,
        mlflow_tracking_uri: Optional[str] = None,
        mlflow_artifact_path: Optional[str] = None,
        model_loader: Optional[Callable[[str], Any]] = None,
        custom_objects: Optional[Dict[str, Any]] = None,
        dataset_array: Any,
        framework: str,
        target_layer: Any = None,
        cam_names=("GradCAM", "EigenCAM"),
        class_index: Optional[int] = None,
        image_number: int = 8,
        overlay: bool = True,
        alpha: float = 0.45,
        figsize=None,
        show_plot: bool = True,
    ):
        """Visualize CAMs over a grid of images."""
        _, mlflow_run_id, mlflow_tracking_uri, mlflow_artifact_path = self._resolve_artifact_context(
            mlflow_run_id=mlflow_run_id,
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_artifact_path=mlflow_artifact_path,
            prefer_last_source=True,
        )
        self._autoload_preprocessing_from_artifacts(
            source=model_path,
            mlflow_run_id=mlflow_run_id,
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_artifact_path=mlflow_artifact_path,
        )
        model_obj = self._resolve_model(
            framework=framework,
            model=model,
            model_path=model_path,
            mlflow_run_id=mlflow_run_id,
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_artifact_path=mlflow_artifact_path,
            model_loader=model_loader,
            custom_objects=custom_objects,
        )

        cams, fig = visualise_CAM(
            dataset_array=dataset_array,
            model=model_obj,
            framework=framework,
            target_layer=target_layer,
            cam_names=cam_names,
            class_index=class_index,
            image_number=image_number,
            overlay=overlay,
            alpha=alpha,
            figsize=figsize,
            show_plot=show_plot,
        )
        self._save_cam_figure(fig, prefix="visualise_cams")
        return cams, fig

    def _write_report_artifacts(self, report: Dict[str, Any]) -> Dict[str, str]:
        os.makedirs(self.outdir, exist_ok=True)

        metrics_json = os.path.join(self.outdir, "metrics.json")
        report_json = os.path.join(self.outdir, "report.json")
        report_md = os.path.join(self.outdir, "report.md")

        with open(metrics_json, "w", encoding="utf-8") as f:
            json.dump(report.get("metrics", {}), f, indent=2)

        with open(report_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        with open(report_md, "w", encoding="utf-8") as f:
            f.write(self._render_md(report))

        return {
            "metrics_json": metrics_json,
            "report_json": report_json,
            "report_md": report_md,
        }

    @staticmethod
    def _render_md(report: Dict[str, Any]) -> str:
        lines = [
            "# evaluation report",
            "",
            "## Summary",
            f"- tag: {report.get('tag')}",
            f"- task_type: {report.get('task_type')}",
            f"- num_classes: {report.get('num_classes')}",
            f"- source: {report.get('source')}",
            "",
            "## Metrics",
        ]

        metrics = report.get("metrics", {})
        for k, v in metrics.items():
            lines.append(f"- {k}: {v}")

        if report.get("metadata"):
            lines.extend(["", "## Metadata"])
            for k, v in report["metadata"].items():
                lines.append(f"- {k}: {v}")

        return "\n".join(lines)
