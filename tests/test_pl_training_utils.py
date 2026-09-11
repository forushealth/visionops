"""Focused tests for the canonical PyTorch Lightning model utilities."""

from __future__ import annotations

import importlib

import pytest


SUPPORTED_MODEL_NAMES = {
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "efficientnet_b0",
    "efficientnet_b1",
    "efficientnet_b2",
    "efficientnet_b3",
    "efficientnet_b4",
    "efficientnet_b5",
    "efficientnet_b6",
    "efficientnet_b7",
    "efficientnet_v2_l",
    "efficientnet_v2_m",
    "efficientnet_v2_s",
    "mobilenet_v2",
    "mobilenet_v3_large",
    "mobilenet_v3_small",
}


def _load_pl_training_utils():
    pytest.importorskip("torchmetrics")
    pytest.importorskip("pytorch_lightning")
    return importlib.import_module("visionops.PLTrainingUtils")


def test_backbone_builder_has_an_explicit_supported_name_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_pl_training_utils()
    assert set(module._MODEL_BUILDERS) == SUPPORTED_MODEL_NAMES

    calls = []
    sentinel = object()

    def fake_builder(*, weights):
        calls.append(weights)
        return sentinel

    monkeypatch.setitem(module._MODEL_BUILDERS, "resnet18", fake_builder)

    assert module._build_torchvision_backbone(" ResNet18 ", pretrained=True) is sentinel
    assert module._build_torchvision_backbone("resnet18", pretrained=False) is sentinel
    assert calls == ["IMAGENET1K_V1", None]

    with pytest.raises(ValueError, match="Unsupported model_name") as exc_info:
        module._build_torchvision_backbone("not-a-model")
    assert "mobilenet_v3_small" in str(exc_info.value)
    assert "resnet152" in str(exc_info.value)


def test_only_task_aware_training_classes_remain() -> None:
    module = _load_pl_training_utils()

    for legacy_name in ("DatasetBuilder", "DataLoaders", "BinaryClassifier", "Trainer", "Model_finder"):
        assert not hasattr(module, legacy_name)

    assert module.TaskDatasetBuilder is not None
    assert module.TaskDataLoaders is not None
    assert module.TaskClassifier is not None
