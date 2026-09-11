"""Smoke tests for training-time eval artifact exports."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from visionops.TrainingPipeline import TrainingPipeline


def test_standardize_logits_csv_schema(tmp_path: Path) -> None:
    """Keras-style source CSV should normalize to image_path/label/logits."""
    src_csv = tmp_path / "raw_val_logits.csv"
    pd.DataFrame(
        {
            "full_path": ["img/a.png", "img/b.png"],
            "labels": [0, 1],
            "logits": ["[-1.2]", "[1.6]"],
        }
    ).to_csv(src_csv, index=False)

    pipe = TrainingPipeline(backend="keras", use_mlflow=False, export_eval_outputs=False)
    dst_csv = tmp_path / "val_logits.csv"
    out_csv = pipe._standardize_logits_csv(str(src_csv), str(dst_csv))

    out_df = pd.read_csv(out_csv)
    assert list(out_df.columns) == ["image_path", "label", "logits"]
    assert out_df["image_path"].tolist() == ["img/a.png", "img/b.png"]
    assert out_df["label"].tolist() == [0, 1]


def test_keras_export_uses_validated_artifacts_and_native_model_format(tmp_path: Path) -> None:
    logits = tmp_path / "raw_logits.csv"
    pd.DataFrame(
        {"path": ["a.png"], "label": [0], "logits": ["[0.25]"]}
    ).to_csv(logits, index=False)
    history = tmp_path / "raw_history.csv"
    pd.DataFrame({"loss": [1.0]}).to_csv(history, index=False)

    class FakeModel:
        def save(self, path: str) -> None:
            Path(path).write_text("model", encoding="utf-8")

    pipe = TrainingPipeline(
        backend="keras",
        use_mlflow=False,
        export_eval_outputs=True,
        eval_outputs_outdir=str(tmp_path / "export"),
    )
    pipe._fit_runner = SimpleNamespace(
        val_logits_csv_path=str(logits),
        train_history_csv_path=str(history),
        config={},
        model=FakeModel(),
    )

    pipe._export_eval_outputs()

    assert Path(pipe.eval_artifacts["keras_model_path"]).suffix == ".keras"
    assert Path(pipe.eval_artifacts["keras_model_path"]).is_file()
    assert Path(pipe.eval_artifacts["keras_val_logits_csv"]).is_file()
    assert Path(pipe.eval_artifacts["keras_train_history_csv"]).is_file()


def test_keras_export_rejects_missing_declared_artifact(tmp_path: Path) -> None:
    pipe = TrainingPipeline(
        backend="keras",
        use_mlflow=False,
        export_eval_outputs=True,
        eval_outputs_outdir=str(tmp_path / "export"),
    )
    pipe._fit_runner = SimpleNamespace(
        val_logits_csv_path=str(tmp_path / "missing.csv"),
        train_history_csv_path=None,
        config={},
        model=None,
    )

    with pytest.raises(FileNotFoundError, match="validation logits"):
        pipe._export_eval_outputs()


def test_torch_export_writes_standardized_logits_csv(tmp_path: Path) -> None:
    """Torch export should persist val_logits.csv with image_path/label/logits."""
    torch = pytest.importorskip("torch")

    from torch.utils.data import DataLoader, Dataset

    class _ToyDataset(Dataset):
        def __init__(self) -> None:
            self.dataframe = pd.DataFrame({"image_path": ["a.png", "b.png", "c.png"]})
            self.image_path_column = "image_path"
            self.data_directory_path = str(tmp_path)

        def __len__(self) -> int:
            return len(self.dataframe)

        def __getitem__(self, idx: int):
            x = torch.ones(3, 4, 4, dtype=torch.float32) * float(idx + 1)
            y = torch.tensor(idx % 2, dtype=torch.long)
            return x, y

    class _ToyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.head = torch.nn.Linear(3 * 4 * 4, 1)

        def forward(self, x):
            return self.head(x.reshape(x.shape[0], -1))

    pipe = TrainingPipeline(
        backend="torch",
        use_mlflow=False,
        export_eval_outputs=True,
        eval_outputs_outdir=str(tmp_path / "eval_artifacts"),
    )
    pipe.model = _ToyModel()

    loader = DataLoader(_ToyDataset(), batch_size=2, shuffle=False)
    pipe._export_eval_outputs(eval_dataloader=loader)

    csv_path = Path(pipe.eval_artifacts["torch_val_logits_csv"])
    assert csv_path.is_file()
    logits_df = pd.read_csv(csv_path)
    assert list(logits_df.columns) == ["image_path", "label", "logits"]
    assert len(logits_df) == 3
    assert all(str(tmp_path) in str(p) for p in logits_df["image_path"].tolist())

    npz_path = Path(pipe.eval_artifacts["torch_eval_outputs_npz"])
    assert npz_path.is_file()
    with np.load(npz_path, allow_pickle=False) as payload:
        assert "image_path" in payload
        assert len(np.array(payload["image_path"]).reshape(-1)) == 3
        assert payload["image_path"].dtype.kind in {"U", "S"}


def test_artifact_catalog_has_stable_sections_and_order(tmp_path: Path) -> None:
    """Artifact catalog should keep fixed section order and preferred key order."""
    pipe = TrainingPipeline(
        backend="keras",
        use_mlflow=False,
        export_eval_outputs=False,
        dfperf_outdir=str(tmp_path / "dfperf"),
    )

    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    preflight = tmp_path / "preflight.json"
    preflight.write_text("{}", encoding="utf-8")
    metrics = tmp_path / "metrics.json"
    metrics.write_text("{}", encoding="utf-8")
    model = tmp_path / "keras_model.h5"
    model.write_text("model", encoding="utf-8")
    logits = tmp_path / "val_logits.csv"
    pd.DataFrame({"image_path": ["a.png"], "label": [0], "logits": ["[0.1]"]}).to_csv(logits, index=False)

    pipe.preprocessing_manifest_path = str(manifest)
    pipe.dfperf_report = {"preflight_json": str(preflight)}
    pipe.dfperf_runtime_report = {"artifacts": {"metrics_json": str(metrics)}}
    pipe.eval_artifacts = {
        "keras_model_path": str(model),
        "val_logits_csv": str(logits),
        "mlflow_run_id": "run_123",
    }

    catalog_path = pipe._save_artifact_catalog()
    assert catalog_path is not None
    loaded = json.loads(Path(catalog_path).read_text(encoding="utf-8"))

    assert list(loaded.keys()) == [
        "top_artifacts",
        "data_related",
        "pre_training_related",
        "training_related",
        "evaluation_related",
    ]
    assert list(loaded["training_related"].keys())[:2] == ["keras_model_path", "keras_train_history_csv"]
    assert list(loaded["evaluation_related"].keys())[:4] == [
        "val_logits_csv",
        "keras_val_logits_csv",
        "torch_val_logits_csv",
        "torch_eval_outputs_npz",
    ]
    assert len(loaded["top_artifacts"]) == len(pipe.TOP_ARTIFACT_KEYS)
    assert len(loaded["data_related"]) == len(pipe.DATA_ARTIFACT_KEYS)
    assert len(loaded["pre_training_related"]) == len(pipe.PRETRAINING_ARTIFACT_KEYS)
    assert len(loaded["training_related"]) == len(pipe.TRAINING_ARTIFACT_KEYS)
    assert len(loaded["evaluation_related"]) == len(pipe.EVALUATION_ARTIFACT_KEYS)


def test_ordered_artifact_layout_materializes_five_sections(tmp_path: Path) -> None:
    """Materialized layout should always create fixed 5 ordered section folders."""
    pipe = TrainingPipeline(
        backend="keras",
        use_mlflow=False,
        export_eval_outputs=False,
        dfperf_outdir=str(tmp_path / "dfperf"),
    )

    preflight = tmp_path / "preflight.json"
    preflight.write_text("{}", encoding="utf-8")
    metrics = tmp_path / "metrics.json"
    metrics.write_text("{}", encoding="utf-8")
    model = tmp_path / "keras_model.h5"
    model.write_text("model", encoding="utf-8")
    logits = tmp_path / "val_logits.csv"
    pd.DataFrame({"image_path": ["a.png"], "label": [0], "logits": ["[0.1]"]}).to_csv(logits, index=False)

    pipe.dfperf_report = {"preflight_json": str(preflight)}
    pipe.dfperf_runtime_report = {"artifacts": {"metrics_json": str(metrics)}}
    pipe.eval_artifacts = {
        "keras_model_path": str(model),
        "val_logits_csv": str(logits),
    }

    pipe._save_artifact_catalog()
    layout_root = pipe._materialize_ordered_artifacts()
    assert layout_root is not None

    layout_root_path = Path(layout_root)
    for section in pipe.ARTIFACT_SECTION_ORDER:
        section_dir = layout_root_path / pipe.ARTIFACT_SECTION_MLFLOW_DIRS[section]
        assert section_dir.is_dir()
        section_manifest = section_dir / "section_manifest.json"
        assert section_manifest.is_file()

        entries = json.loads(section_manifest.read_text(encoding="utf-8"))
        expected_keys = list(pipe.artifact_catalog.get(section, {}).keys())
        assert [row.get("key") for row in entries] == expected_keys

    training_dir = layout_root_path / pipe.ARTIFACT_SECTION_MLFLOW_DIRS["training_related"]
    assert (training_dir / "02_keras_train_history_csv.missing.txt").is_file()
