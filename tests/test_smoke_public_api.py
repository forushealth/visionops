"""Smoke tests for the canonical VisionOps package namespace."""

from __future__ import annotations

import importlib


def test_public_namespace_exports_core_api() -> None:
    import visionops

    assert isinstance(visionops.__version__, str)
    assert visionops.__version__
    assert len(visionops.__all__) == len(set(visionops.__all__))
    assert all(hasattr(visionops, name) for name in visionops.__all__)


def test_submodules_use_the_canonical_namespace() -> None:
    import visionops

    task_spec_module = importlib.import_module("visionops.TaskSpec")
    data_module = importlib.import_module("visionops.DataUtils")

    assert task_spec_module.TaskSpec is visionops.TaskSpec
    assert data_module.__name__ == "visionops.DataUtils"
