"""Smoke tests for finder utility recommendation logic."""

from __future__ import annotations

import pytest

from visionops import FinderUtils


def test_keras_batch_finder_applies_safety_downshift(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keras finder should downshift from max stable batch to safer recommended batch."""

    def _fake_probe(config_path: str, batch_size: int, quiet: bool = True) -> float:
        if int(batch_size) >= 48:
            raise RuntimeError("CUDA out of memory")
        return float(1.0 / max(1, int(batch_size)))

    monkeypatch.setattr(FinderUtils, "_keras_probe_one_step", _fake_probe)

    result = FinderUtils.run_batch_finder(
        backend="keras",
        config_dict={"batch_size": 16},
        config_path="dummy.yaml",
        batch_candidates=[8, 16, 24, 32, 48],
        stop_on_oom=True,
        keras_safety_downshift=1,
        quiet=True,
    )

    assert result["max_stable_batch_size"] == 32
    assert result["best_batch_size"] == 24
    assert result["safety_downshift_applied"] == 1
    assert len(result["history"]) == 5


def test_keras_batch_finder_no_success_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """When all probes fail, default configured batch size should be preserved."""

    def _always_fail(*args, **kwargs):
        raise RuntimeError("ResourceExhaustedError: OOM")

    monkeypatch.setattr(FinderUtils, "_keras_probe_one_step", _always_fail)

    result = FinderUtils.run_batch_finder(
        backend="keras",
        config_dict={"batch_size": 12},
        config_path="dummy.yaml",
        batch_candidates=[8, 16],
        stop_on_oom=False,
        keras_safety_downshift=1,
        quiet=True,
    )

    assert result["max_stable_batch_size"] == 12
    assert result["best_batch_size"] == 12
    assert result["safety_downshift_applied"] == 1
    assert all(item["status"] == "error" for item in result["history"])
