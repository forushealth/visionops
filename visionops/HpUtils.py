"""Pydantic configuration models and YAML helpers for experiments."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ._utils import as_list
from pydantic import BaseModel, Field


class DataSplit(BaseModel):
    """Train/validation/test split settings."""

    train: float = 0.8
    val: float = 0.2
    test: float = 0.0
    seed: Optional[int] = None
    stratify: bool = True


class TransformConfig(BaseModel):
    """Pre/augmentation/post transform configuration."""

    pre: List[Any] = Field(default_factory=list)
    aug: List[Any] = Field(default_factory=list)
    post: List[Any] = Field(default_factory=list)


class DataConfig(BaseModel):
    """Data loading and transformation configuration block."""

    split: DataSplit = Field(default_factory=DataSplit)
    bs: int = 32
    num_workers: int = 2
    shuffle_train: bool = True
    oversample_train: bool = False
    tfms: TransformConfig = Field(default_factory=TransformConfig)


class TrainConfig(BaseModel):
    """Core optimizer/training schedule settings."""

    epochs: int = 10
    learning_rate: float = 1e-3
    lr_scheduler: str = "ReduceLROnPlateau"


class LoggingConfig(BaseModel):
    """MLflow/logging toggles for runs."""

    use_mlflow: bool = False
    log_hyperparams: bool = True
    log_metrics: bool = True
    log_artifacts: bool = True


class ExperimentConfig(BaseModel):
    """Top-level experiment configuration schema."""

    model_config = {"protected_namespaces": ()}

    yaml_path: str = "config.yaml"
    experiment_name: str
    experiment_version: str
    date: str = Field(default_factory=lambda: str(date.today()))

    dataset_name: str = ""
    data_directory_path: str = ""
    image_size: tuple[int, int, int] = (224, 224, 3)
    image_path_column: str = "path"
    image_label_column: str = "label"
    image_categorical_label_column: str = "categorical_label"

    train_data_csv: str = ""
    val_data_csv: str = ""
    test_data_csv: str = ""

    data: DataConfig = Field(default_factory=DataConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    model_save_path: str = ""
    model_checkpoints_path: str = ""
    tag: List[str] = Field(default_factory=list)


def _to_dict(model: BaseModel) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def create_experiment_config(
    experiment_name: str,
    version: str,
    yaml_path: str = "config.yaml",
    overrides: Optional[Dict[str, Any]] = None,
) -> ExperimentConfig:
    """Create an experiment config, apply overrides, and persist YAML."""
    cfg = ExperimentConfig(
        yaml_path=yaml_path,
        experiment_name=experiment_name,
        experiment_version=str(version),
        model_save_path=f"../models/model_{experiment_name}_v{version}.h5",
        model_checkpoints_path=f"../models/checkpoints_{experiment_name}_v{version}",
        tag=[experiment_name, "BC"],
    )
    if overrides:
        payload = _deep_update(_to_dict(cfg), overrides)
        cfg = ExperimentConfig(**payload)
    save_config_yaml(cfg, yaml_path)
    return cfg


def save_config_yaml(config: ExperimentConfig, yaml_path: Optional[str] = None) -> str:
    """Serialize an :class:`ExperimentConfig` to YAML and return path."""
    target = yaml_path or config.yaml_path
    with open(target, "w", encoding="utf-8") as file:
        yaml.safe_dump(_to_dict(config), file, sort_keys=False)
    return target


def load_config_yaml(yaml_path: str = "config.yaml") -> ExperimentConfig:
    """Load experiment YAML and return parsed :class:`ExperimentConfig`."""
    with open(yaml_path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    data.setdefault("yaml_path", yaml_path)
    return ExperimentConfig(**data)


def load_raw_config_yaml(yaml_path: str = "config.yaml") -> Dict[str, Any]:
    """Load YAML config as a raw dictionary without schema validation."""
    with open(yaml_path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config at '{yaml_path}' must be a YAML mapping/object.")
    data.setdefault("yaml_path", yaml_path)
    return data


def resolve_stage_config(config: Dict[str, Any], stage: str) -> Dict[str, Any]:
    """Return effective config using a common YAML + per-stage overrides.

    Supported stage blocks (optional):
    - ``stages: {train: {...}, eval: {...}, explain: {...}}``
    - ``stage_overrides: {train: {...}, eval: {...}, explain: {...}}``
    """
    if not isinstance(config, dict):
        raise TypeError("config must be a dictionary")

    effective = {k: v for k, v in config.items() if k not in {"stages", "stage_overrides"}}
    stage_key = str(stage).strip().lower()

    stages = config.get("stages", {})
    if isinstance(stages, dict) and isinstance(stages.get(stage_key), dict):
        effective = _deep_update(effective, stages[stage_key])

    stage_overrides = config.get("stage_overrides", {})
    if isinstance(stage_overrides, dict) and isinstance(stage_overrides.get(stage_key), dict):
        effective = _deep_update(effective, stage_overrides[stage_key])

    return effective


def resolve_stage_from_yaml(
    yaml_path: str = "config.yaml",
    stage: str = "train",
    *,
    validate: bool = False,
    snapshot_dir: Optional[str] = None,
    snapshot_filename: Optional[str] = None,
) -> Dict[str, Any]:
    """Resolve effective stage config from a common YAML file.

    Args:
        yaml_path: Path to the common YAML config.
        stage: Stage name such as ``train``, ``eval``, or ``explain``.
        validate: If True, validates base YAML against ``ExperimentConfig``.
        snapshot_dir: Optional directory to persist resolved effective config YAML.
        snapshot_filename: Optional snapshot filename. Defaults to
            ``config_effective_<stage>.yaml``.
    """
    if validate:
        # Schema validation for the common/base contract.
        load_config_yaml(yaml_path)

    raw = load_raw_config_yaml(yaml_path)
    effective = resolve_stage_config(raw, stage=stage)

    if snapshot_dir:
        out_dir = Path(snapshot_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        name = snapshot_filename or f"config_effective_{str(stage).strip().lower()}.yaml"
        out_path = out_dir / name
        with open(out_path, "w", encoding="utf-8") as file:
            yaml.safe_dump(effective, file, sort_keys=False)
        effective["_effective_config_path"] = str(out_path)

    return effective


_PROBABILITY_KEYS = ("p", "prob", "probability")
_POSTPROCESS_KEYWORDS = (
    "normalize",
    "normalise",
    "standardize",
    "standardise",
    "zscore",
    "minmax",
    "rescale",
)


def _transform_name(step: Any) -> str:
    if isinstance(step, str):
        return step.strip()
    if isinstance(step, dict):
        for key in ("name", "type", "op", "transform"):
            val = step.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        if len(step) == 1:
            first_key = next(iter(step.keys()))
            if isinstance(first_key, str):
                return first_key.strip()
    return ""


def _transform_probability(step: Any) -> float:
    if isinstance(step, dict):
        for key in _PROBABILITY_KEYS:
            if key in step:
                try:
                    return float(step[key])
                except Exception:
                    return 1.0
        if len(step) == 1:
            first_val = next(iter(step.values()))
            if isinstance(first_val, dict):
                for key in _PROBABILITY_KEYS:
                    if key in first_val:
                        try:
                            return float(first_val[key])
                        except Exception:
                            return 1.0
    return 1.0


def split_image_processing_transforms(transforms: Any) -> Dict[str, List[Any]]:
    """Split transforms into pre/aug/post.

    Rules:
    - ``post``: transform names containing normalization/standardization keywords.
    - ``aug``: probability < 1.0 (i.e., less than 100%).
    - ``pre``: everything else (typically deterministic operations).
    """
    pre: List[Any] = []
    aug: List[Any] = []
    post: List[Any] = []

    for step in as_list(transforms):
        name = _transform_name(step).lower()
        prob = _transform_probability(step)
        if any(k in name for k in _POSTPROCESS_KEYWORDS):
            post.append(step)
        elif prob < 1.0:
            aug.append(step)
        else:
            pre.append(step)
    return {"pre": pre, "aug": aug, "post": post}


def normalize_image_processing_sections(tfms: Any) -> Dict[str, List[Any]]:
    """Normalize transform spec into ``pre/aug/post`` sections."""
    if isinstance(tfms, dict):
        has_sections = any(k in tfms for k in ("pre", "aug", "post"))
        if has_sections:
            pre = as_list(tfms.get("pre"))
            aug = as_list(tfms.get("aug"))
            post = as_list(tfms.get("post"))
            return {"pre": pre, "aug": aug, "post": post}

        for key in ("all", "transforms", "pipeline", "steps"):
            if key in tfms:
                return split_image_processing_transforms(tfms.get(key))
        return {"pre": [], "aug": [], "post": []}

    return split_image_processing_transforms(tfms)


def build_stage_image_processing(transforms: Dict[str, List[Any]], stage: str) -> List[Any]:
    """Build ordered transform list for a stage.

    - train: ``pre + aug + post``
    - eval/infer/test: ``pre + post``
    """
    stage_key = str(stage).strip().lower()
    pre = list(transforms.get("pre", []))
    aug = list(transforms.get("aug", []))
    post = list(transforms.get("post", []))
    if stage_key == "train":
        return pre + aug + post
    return pre + post


def freeze_preprocessing_manifest(
    effective_config: Dict[str, Any],
    outdir: str,
    *,
    filename: str = "preprocessing_manifest.yaml",
) -> str:
    """Persist a frozen preprocessing/augmentation manifest for reproducibility."""
    cfg = dict(effective_config or {})
    data_cfg = cfg.get("data", {}) if isinstance(cfg.get("data", {}), dict) else {}
    tfms = data_cfg.get("tfms", data_cfg.get("transforms", cfg.get("transforms", {})))
    sections = normalize_image_processing_sections(tfms)

    manifest = {
        "image_size": cfg.get("image_size"),
        "image_path_column": cfg.get("image_path_column"),
        "image_label_column": cfg.get("image_label_column"),
        "transforms": sections,
        "pipelines": {
            "train": build_stage_image_processing(sections, stage="train"),
            "eval": build_stage_image_processing(sections, stage="eval"),
        },
        "order": {
            "train": "pre+aug+post",
            "eval": "pre+post",
        },
        "runtime": {
            "use_albumentations": bool(cfg.get("use_albumentations", False)),
        },
    }

    out_path = Path(outdir) / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)
    return str(out_path)


def load_frozen_preprocessing_manifest(
    manifest_path: str,
    *,
    evaluation_mode: bool = True,
) -> Dict[str, Any]:
    """Load frozen manifest and select stage-appropriate processing sequence."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f) or {}
    if not isinstance(manifest, dict):
        raise ValueError(f"Invalid preprocessing manifest at '{manifest_path}'.")

    sections = normalize_image_processing_sections(manifest.get("transforms", {}))
    manifest["transforms"] = sections
    manifest["pipelines"] = {
        "train": build_stage_image_processing(sections, stage="train"),
        "eval": build_stage_image_processing(sections, stage="eval"),
    }
    manifest["order"] = {"train": "pre+aug+post", "eval": "pre+post"}

    if evaluation_mode:
        manifest["transforms"]["aug"] = []
        manifest["active_stage"] = "eval"
        manifest["active_pipeline"] = build_stage_image_processing(manifest["transforms"], stage="eval")
    else:
        manifest["active_stage"] = "train"
        manifest["active_pipeline"] = build_stage_image_processing(manifest["transforms"], stage="train")
    return manifest
