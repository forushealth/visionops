from __future__ import annotations

from pathlib import Path

import pytest

from visionops.SysUtils import is_image, search


def test_search_is_case_insensitive_and_deduplicated(tmp_path: Path) -> None:
    (tmp_path / "sample.JPG").write_bytes(b"image")
    (tmp_path / "notes.txt").write_text("notes", encoding="utf-8")

    result = search(tmp_path, patterns=["*.jpg", "*.JPG"], path_type="base")

    assert result == ["sample.JPG"]


def test_search_can_ignore_hidden_paths(tmp_path: Path) -> None:
    hidden_dir = tmp_path / ".cache"
    hidden_dir.mkdir()
    (hidden_dir / "hidden.png").write_bytes(b"image")
    (tmp_path / "visible.png").write_bytes(b"image")

    result = search(tmp_path, patterns="*.png", ignore_hidden=True, path_type="base")

    assert result == ["visible.png"]


def test_search_rejects_invalid_options(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="path_type"):
        search(tmp_path, path_type="unknown")


@pytest.mark.parametrize("path", ["photo.jpg", "scan.TIFF", Path("plot.png")])
def test_is_image(path: str | Path) -> None:
    assert is_image(path)
