"""Smoke tests for Evaluator report and plotting flow."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from visionops.RunEvaluator import Evaluator
from visionops.RunExperimentDescription import run_experiment_description


def _make_dummy_images(img_dir: Path, count: int) -> list[str]:
    """Create tiny synthetic RGB images and return file paths."""
    img_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for i in range(count):
        img = np.zeros((16, 16, 3), dtype=np.uint8)
        img[:, :, 0] = (i * 30) % 255
        img[:, :, 1] = (255 - i * 20) % 255
        img[:, :, 2] = (i * 10) % 255
        p = img_dir / f"img_{i}.png"
        Image.fromarray(img).save(p)
        paths.append(str(p))
    return paths


def test_run_evaluator_full_smoke(tmp_path: Path, monkeypatch) -> None:
    """Exercise evaluator stats/history/report flow on synthetic artifacts."""
    img_paths = _make_dummy_images(tmp_path / "images", count=8)

    # Binary logits as 1-value vectors represented as strings.
    y_true = [0, 1, 1, 0, 1, 0, 1, 0]
    logits = [[-2.0], [1.6], [2.0], [-1.0], [1.3], [-2.1], [2.4], [-0.8]]

    logits_csv = tmp_path / "val_logits.csv"
    pd.DataFrame(
        {
            "label": y_true,
            "logits": [str(v) for v in logits],
            "full_path": img_paths,
        }
    ).to_csv(logits_csv, index=False)

    history_csv = tmp_path / "train_history.csv"
    pd.DataFrame(
        {
            "loss": [0.8, 0.5, 0.3],
            "val_loss": [0.7, 0.45, 0.28],
            "accuracy": [0.62, 0.75, 0.88],
            "val_accuracy": [0.60, 0.73, 0.86],
        }
    ).to_csv(history_csv, index=False)

    evaluator = Evaluator(
        source=str(logits_csv),
        output_dir=tmp_path / "evaluator_out",
        history_csv_path=str(history_csv),
    )

    stats = evaluator.statistical_tests()
    assert "accuracy_argmax" in stats

    def _unexpected_llm_call(*args, **kwargs):
        raise AssertionError("local LLM must be opt-in")

    monkeypatch.setattr(evaluator, "_run_local_llm", _unexpected_llm_call)
    history_summary = evaluator.history_analysis()
    assert "epochs_logged" in history_summary

    # Pre-seed accuracy to keep generate_report() in non-plot path.
    evaluator.metrics["accuracy"] = stats["accuracy_argmax"]

    report_paths = evaluator.generate_report()
    assert Path(report_paths["report_json"]).is_file()
    assert Path(report_paths["report_md"]).is_file()
    assert Path(report_paths["report_pdf"]).is_file()

    report_payload = json.loads(Path(report_paths["report_json"]).read_text(encoding="utf-8"))
    for key in ("data_overview", "starting_hyperparameters", "pre_training_details", "training_details"):
        assert key in report_payload

    md_text = Path(report_paths["report_md"]).read_text(encoding="utf-8")
    for heading in (
        "## Data Overview",
        "## Starting Hyperparameters",
        "## Pre-Training Details",
        "## Training Details",
        "## Evaluation Metrics",
    ):
        assert heading in md_text


def test_prediction_example_without_image_paths(tmp_path: Path) -> None:
    """prediction_example should still render placeholder panels without image path columns."""
    y_true = [0, 1, 1, 0, 1, 0]
    logits = [[-1.1], [1.7], [1.2], [0.4], [1.9], [-0.8]]

    logits_csv = tmp_path / "val_logits_no_paths.csv"
    pd.DataFrame(
        {
            "label": y_true,
            "logits": [str(v) for v in logits],
        }
    ).to_csv(logits_csv, index=False)

    evaluator = Evaluator(
        source=str(logits_csv),
        output_dir=tmp_path / "evaluator_out_no_paths",
    )

    example_paths = evaluator.prediction_example(k=3, show=False)
    assert "correct_predictions" in example_paths
    assert "incorrect_predictions" in example_paths
    for _, p in example_paths.items():
        assert Path(p).is_file()


def test_run_experiment_description_helper(tmp_path: Path) -> None:
    """run_experiment_description should generate report artifacts via Evaluator."""
    logits_csv = tmp_path / "val_logits_wrapper.csv"
    pd.DataFrame(
        {
            "label": [0, 1, 0, 1],
            "logits": [str([-1.2]), str([1.6]), str([-0.9]), str([1.3])],
        }
    ).to_csv(logits_csv, index=False)

    history_csv = tmp_path / "train_history_wrapper.csv"
    pd.DataFrame(
        {
            "loss": [0.9, 0.6, 0.4],
            "val_loss": [0.8, 0.58, 0.45],
            "accuracy": [0.55, 0.71, 0.85],
            "val_accuracy": [0.52, 0.69, 0.82],
        }
    ).to_csv(history_csv, index=False)

    report_paths = run_experiment_description(
        source=str(logits_csv),
        output_dir=tmp_path / "experiment_description_out",
        history_csv_path=str(history_csv),
        show=False,
        markdown_only=True,
    )
    assert Path(report_paths["report_json"]).is_file()
    assert Path(report_paths["report_md"]).is_file()
