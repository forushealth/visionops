"""Shared internal helpers with no mandatory backend dependencies."""

from __future__ import annotations

import ast
import csv
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

T = TypeVar("T")


def csv_has_column(path: str | Path, column: str) -> bool:
    """Return whether a CSV header contains ``column`` without loading its rows."""
    try:
        with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
            header = next(csv.reader(stream), ())
    except (OSError, TypeError, ValueError, UnicodeError, csv.Error):
        return False
    return column in header


def run_cleanup_steps(
    steps: Iterable[tuple[str, Callable[[], Any]]],
    *,
    primary_error: BaseException | None = None,
) -> None:
    """Run cleanup callbacks without replacing an existing primary exception."""
    cleanup_errors: list[tuple[str, Exception]] = []
    for name, step in steps:
        try:
            step()
        except Exception as exc:
            cleanup_errors.append((name, exc))

    if not cleanup_errors:
        return

    if primary_error is not None:
        add_note = getattr(primary_error, "add_note", None)
        if callable(add_note):
            for name, exc in cleanup_errors:
                add_note(f"VisionOps cleanup step {name!r} failed: {type(exc).__name__}: {exc}")
        return

    _, error = cleanup_errors[0]
    add_note = getattr(error, "add_note", None)
    if callable(add_note):
        for extra_name, extra_error in cleanup_errors[1:]:
            add_note(
                f"Additional VisionOps cleanup step {extra_name!r} failed: "
                f"{type(extra_error).__name__}: {extra_error}"
            )
    raise error


def as_list(value: T | Sequence[T] | None) -> list[T]:
    """Normalize an optional scalar or sequence to a list."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def first_existing(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    """Return the first candidate present in the supplied column names."""
    column_set = set(columns)
    return next((candidate for candidate in candidates if candidate in column_set), None)


def normalize_path_value(value: Any) -> str:
    """Normalize stringified byte paths emitted by dataframe/NPZ workflows."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")

    text = str(value)
    if not text.startswith(("b'", 'b"')):
        return text

    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return text
    if isinstance(parsed, (bytes, bytearray)):
        return bytes(parsed).decode("utf-8", errors="ignore")
    return text


def to_numpy(value: Any) -> np.ndarray:
    """Convert NumPy, Torch, TensorFlow, and array-like values to NumPy."""
    if isinstance(value, np.ndarray):
        return value

    module_root = type(value).__module__.split(".", 1)[0]
    if module_root == "torch":
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()

    if module_root in {"tensorflow", "keras"}:
        import tensorflow as tf

        if tf.is_tensor(value):
            return value.numpy()

    return np.asarray(value)
