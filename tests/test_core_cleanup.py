"""Focused regressions for shared helpers and training cleanup behavior."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from visionops.DataUtils import ConfigDict
from visionops.Pipeline import Pipeline
from visionops.TrainingPipeline import TrainingPipeline
from visionops._utils import csv_has_column


def test_csv_has_column_uses_shared_header_probe(tmp_path: Path) -> None:
    csv_path = tmp_path / "predictions.csv"
    csv_path.write_text("\ufefflabel,logits\n0,0.25\n", encoding="utf-8")

    assert csv_has_column(csv_path, "logits")
    assert not csv_has_column(csv_path, "missing")
    assert not csv_has_column(tmp_path / "does-not-exist.csv", "logits")


def test_config_dict_does_not_mutate_global_safe_dumper(tmp_path: Path) -> None:
    tuple_representer = yaml.SafeDumper.yaml_representers.get(tuple)
    yaml_path = tmp_path / "config.yaml"

    ConfigDict({"image_size": (32, 32, 3)}, yaml_path=str(yaml_path))

    assert yaml.SafeDumper.yaml_representers.get(tuple) is tuple_representer
    assert yaml.safe_load(yaml_path.read_text(encoding="utf-8")) == {"image_size": [32, 32, 3]}


def test_pipeline_cleanup_cannot_mask_training_error(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = Pipeline(backend="keras", use_mlflow=False)
    later_cleanup_ran = []

    monkeypatch.setattr(pipeline, "_run_pre_training_checks", lambda: None)
    monkeypatch.setattr(pipeline, "_start_dfperf_runtime_monitor", lambda: object())

    def fail_training(**_kwargs):
        raise ValueError("training exploded")

    def fail_finalizer(**_kwargs):
        raise RuntimeError("finalizer exploded")

    monkeypatch.setattr(pipeline, "_fit_keras", fail_training)
    monkeypatch.setattr(pipeline, "_finalize_dfperf_runtime_monitor", fail_finalizer)
    monkeypatch.setattr(pipeline, "_log_dfperf_artifacts_to_mlflow", lambda: later_cleanup_ran.append(True))

    with pytest.raises(ValueError, match="training exploded") as exc_info:
        pipeline.fit()

    assert later_cleanup_ran == [True]
    if hasattr(exc_info.value, "__notes__"):
        assert any("finalizer exploded" in note for note in exc_info.value.__notes__)


def test_training_pipeline_cleanup_cannot_mask_training_error(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = TrainingPipeline(
        backend="keras",
        use_mlflow=False,
        export_eval_outputs=False,
    )
    later_cleanup_steps = []

    monkeypatch.setattr(pipeline, "_run_pre_training_checks", lambda: None)
    monkeypatch.setattr(pipeline, "_start_dfperf_runtime_monitor", lambda: object())

    def fail_training(**_kwargs):
        raise ValueError("training exploded")

    def fail_finalizer(**_kwargs):
        raise RuntimeError("finalizer exploded")

    monkeypatch.setattr(pipeline, "_fit_keras", fail_training)
    monkeypatch.setattr(pipeline, "_finalize_dfperf_runtime_monitor", fail_finalizer)
    monkeypatch.setattr(pipeline, "_save_artifact_catalog", lambda: later_cleanup_steps.append("catalog"))
    monkeypatch.setattr(
        pipeline,
        "_materialize_ordered_artifacts",
        lambda: later_cleanup_steps.append("materialize"),
    )
    monkeypatch.setattr(pipeline, "_log_artifacts_to_mlflow", lambda: later_cleanup_steps.append("log"))

    with pytest.raises(ValueError, match="training exploded") as exc_info:
        pipeline.fit()

    assert later_cleanup_steps == ["catalog", "materialize", "log"]
    if hasattr(exc_info.value, "__notes__"):
        assert any("finalizer exploded" in note for note in exc_info.value.__notes__)


def test_cleanup_failure_propagates_after_successful_training(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = Pipeline(backend="keras", use_mlflow=False)
    later_cleanup_ran = []

    monkeypatch.setattr(pipeline, "_run_pre_training_checks", lambda: None)
    monkeypatch.setattr(pipeline, "_start_dfperf_runtime_monitor", lambda: object())
    monkeypatch.setattr(pipeline, "_fit_keras", lambda **_kwargs: "trained")
    monkeypatch.setattr(
        pipeline,
        "_finalize_dfperf_runtime_monitor",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("finalizer exploded")),
    )
    monkeypatch.setattr(pipeline, "_log_dfperf_artifacts_to_mlflow", lambda: later_cleanup_ran.append(True))

    with pytest.raises(RuntimeError, match="finalizer exploded"):
        pipeline.fit()

    assert later_cleanup_ran == [True]
