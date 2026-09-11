from __future__ import annotations

import builtins
import importlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from visionops._utils import to_numpy


def _block_backends(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".", 1)[0] in {"tensorflow", "torch", "pytorch_lightning"}:
            raise AssertionError(f"unexpected optional backend import: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def test_numpy_conversion_does_not_import_training_backends(monkeypatch) -> None:
    _block_backends(monkeypatch)
    value = np.array([1.0, 2.0])

    assert to_numpy(value) is value


def test_tracking_helpers_import_without_training_backends(monkeypatch) -> None:
    _block_backends(monkeypatch)
    sys.modules.pop("visionops.MLflowLogger", None)

    module = importlib.import_module("visionops.MLflowLogger")

    assert callable(module.create_keras_callback)
    assert callable(module.create_lightning_callback)


def test_local_torch_loader_uses_weights_only(monkeypatch) -> None:
    from visionops.EvaluationPipeline import EvaluationPipeline

    calls = {}

    def fake_load(path, **kwargs):
        calls["path"] = path
        calls.update(kwargs)
        raise RuntimeError("stop after checking loader options")

    fake_torch = SimpleNamespace(load=fake_load, device=lambda value: value)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    with pytest.raises(RuntimeError, match="stop after checking loader options"):
        EvaluationPipeline._load_torch_model("model.pt")

    assert calls["weights_only"] is True
    assert calls["map_location"] == "cpu"
