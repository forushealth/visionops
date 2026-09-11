"""Smoke tests for task metrics and evaluation pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from visionops.EvaluationPipeline import EvaluationPipeline
from visionops.MetricRegistry import compute_metrics
from visionops.TaskSpec import TaskSpec, TaskType


def test_compute_metrics_bc_mcc_mlc_smoke() -> None:
    """Validate core metric paths for BC, MCC, and MLC tasks."""
    # BC
    y_true_bc = np.array([0, 1, 1, 0], dtype=int)
    y_pred_bc_logits = np.array([-3.0, 2.2, 1.7, -1.5], dtype=float)
    bc_metrics = compute_metrics(
        y_true=y_true_bc,
        y_pred=y_pred_bc_logits,
        task_spec=TaskSpec(task_type=TaskType.bc, num_classes=2),
    )
    assert set(bc_metrics.keys()) == {"accuracy", "precision", "recall", "f1"}
    assert bc_metrics["accuracy"] == 1.0

    # MCC
    y_true_mcc = np.array([0, 1, 2, 1], dtype=int)
    y_pred_mcc_logits = np.array(
        [
            [9.0, 1.0, 0.0],
            [0.3, 3.2, 0.2],
            [0.1, 0.4, 5.0],
            [0.2, 4.1, 0.5],
        ],
        dtype=float,
    )
    mcc_metrics = compute_metrics(
        y_true=y_true_mcc,
        y_pred=y_pred_mcc_logits,
        task_spec=TaskSpec(task_type=TaskType.mcc, num_classes=3),
    )
    assert set(mcc_metrics.keys()) == {"accuracy", "precision", "recall", "f1"}
    assert mcc_metrics["accuracy"] == 1.0

    # MLC
    y_true_mlc = np.array(
        [
            [1, 0, 1],
            [0, 1, 0],
            [1, 1, 0],
        ],
        dtype=int,
    )
    y_pred_mlc_logits = np.array(
        [
            [0.8, 0.1, 0.9],
            [0.1, 0.7, 0.2],
            [0.6, 0.8, 0.1],
        ],
        dtype=float,
    )
    mlc_metrics = compute_metrics(
        y_true=y_true_mlc,
        y_pred=y_pred_mlc_logits,
        task_spec=TaskSpec(task_type=TaskType.mlc, num_classes=3),
    )
    assert set(mlc_metrics.keys()) == {
        "subset_accuracy",
        "precision_micro",
        "recall_micro",
        "f1_micro",
        "f1_macro",
    }
    assert mlc_metrics["subset_accuracy"] == 1.0


def test_evaluation_pipeline_evaluate_arrays_npz_csv_smoke(tmp_path: Path) -> None:
    """Run evaluation pipeline over arrays, npz, and keras-style logits CSV."""
    outdir = tmp_path / "eval"
    pipeline = EvaluationPipeline(
        task_spec=TaskSpec(task_type=TaskType.mcc, num_classes=3),
        outdir=str(outdir),
        generate_report=True,
    )

    y_true = np.array([0, 1, 2, 1], dtype=int)
    y_pred = np.array(
        [
            [4.0, 1.0, 0.0],
            [0.1, 3.0, 0.2],
            [0.2, 0.3, 5.0],
            [0.5, 2.1, 0.3],
        ],
        dtype=float,
    )

    report_arrays = pipeline.evaluate_arrays(y_true=y_true, y_pred=y_pred, tag="smoke_arrays")
    assert report_arrays["status"] == "ok"
    assert report_arrays["metrics"]["accuracy"] == 1.0
    for artifact_path in report_arrays["artifacts"].values():
        assert Path(artifact_path).is_file()

    npz_path = tmp_path / "eval_outputs.npz"
    np.savez(npz_path, y_true=y_true, y_pred=y_pred)
    report_npz = pipeline.evaluate_npz(str(npz_path), tag="smoke_npz")
    assert report_npz["status"] == "ok"
    assert report_npz["source"] == str(npz_path)

    logits_csv = tmp_path / "val_logits.csv"
    csv_df = pd.DataFrame(
        {
            "label": y_true.tolist(),
            "logits": [str(v.tolist()) for v in y_pred],
        }
    )
    csv_df.to_csv(logits_csv, index=False)

    report_csv = pipeline.evaluate_keras_logits_csv(str(logits_csv), tag="smoke_csv")
    assert report_csv["status"] == "ok"
    assert report_csv["source"] == str(logits_csv)
    assert report_csv["metrics"]["accuracy"] == 1.0

    npz_with_paths = tmp_path / "eval_outputs_with_paths.npz"
    image_paths = np.array(
        [
            "images/a.png",
            "images/b.png",
            "images/c.png",
            "images/d.png",
        ],
        dtype=str,
    )
    np.savez(npz_with_paths, y_true=y_true, y_pred=y_pred, image_path=image_paths)
    materialized_csv = pipeline._materialize_logits_csv_from_npz(str(npz_with_paths), outdir=str(tmp_path))
    mat_df = pd.read_csv(materialized_csv)
    assert list(mat_df.columns) == ["image_path", "label", "logits"]
    assert mat_df["image_path"].astype(str).tolist() == image_paths.astype(str).tolist()


def test_evaluation_pipeline_rejects_pickle_backed_npz(tmp_path: Path) -> None:
    npz_path = tmp_path / "unsafe.npz"
    np.savez(
        npz_path,
        y_true=np.array([{"label": 1}], dtype=object),
        y_pred=np.array([[0.1, 0.9]]),
    )
    pipeline = EvaluationPipeline(
        task_spec=TaskSpec(task_type="mcc", num_classes=2),
        outdir=str(tmp_path / "evaluation"),
    )

    with pytest.raises(ValueError, match="Object arrays cannot be loaded"):
        pipeline.evaluate_npz(str(npz_path))


def test_build_evaluator_recovers_history_from_local_mlflow_metrics(tmp_path: Path, monkeypatch) -> None:
    """When history CSV is missing, build_evaluator should materialize one from MLflow metric files."""
    run_id = "37b195fdc8964aacb7995e628ad6abfc"
    monkeypatch.chdir(tmp_path)

    run_metrics_dir = tmp_path / "mlruns" / "0" / run_id / "metrics"
    run_metrics_dir.mkdir(parents=True, exist_ok=True)
    (run_metrics_dir / "train_loss").write_text(
        "1700000000000 0.90 0\n1700000001000 0.70 1\n1700000002000 0.50 2\n",
        encoding="utf-8",
    )
    (run_metrics_dir / "val_loss").write_text(
        "1700000000000 0.95 0\n1700000001000 0.75 1\n1700000002000 0.60 2\n",
        encoding="utf-8",
    )
    (run_metrics_dir / "train_acc").write_text(
        "1700000000000 0.60 0\n1700000001000 0.72 1\n1700000002000 0.83 2\n",
        encoding="utf-8",
    )
    (run_metrics_dir / "val_acc").write_text(
        "1700000000000 0.58 0\n1700000001000 0.70 1\n1700000002000 0.80 2\n",
        encoding="utf-8",
    )

    npz_source = tmp_path / "mlruns" / "0" / run_id / "artifacts" / "eval_outputs.npz"
    npz_source.parent.mkdir(parents=True, exist_ok=True)
    y_true = np.array([0, 1, 1, 0], dtype=int)
    y_pred = np.array([[-1.2], [1.7], [1.5], [-0.6]], dtype=float)
    np.savez(npz_source, y_true=y_true, y_pred=y_pred)

    pipeline = EvaluationPipeline(outdir=str(tmp_path / "eval_out"))
    pipeline.last_report = {"metadata": {"mlflow_run_id": run_id}}
    evaluator = pipeline.build_evaluator(source=str(npz_source), output_dir=str(tmp_path / "eval_out"))
    summary = evaluator.history_analysis(show=False)

    assert "epochs_logged" in summary
    assert "final_val_accuracy" in summary
