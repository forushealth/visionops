"""Regression tests for DataUtils runtime input validation."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from visionops.DataUtils import DataAdapterImgClf, _df_make_folds, df_get_splits


def test_fold_column_name_rejects_non_string() -> None:
    with pytest.raises(TypeError, match="'fold_col_name' must be a string"):
        _df_make_folds(pd.DataFrame({"value": [1, 2]}), 2, fold_col_name=1)


def test_stratify_column_validation_has_specific_exception_types() -> None:
    df = pd.DataFrame({"label": ["a", "b"]})

    with pytest.raises(TypeError, match="Can't find column named '1'"):
        df_get_splits(df, 2, stratify_col_name=1)
    with pytest.raises(ValueError, match="Can't find column named 'missing'"):
        df_get_splits(df, 2, stratify_col_name="missing")


def test_missing_dataset_root_raises_file_not_found(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing"

    with pytest.raises(FileNotFoundError, match="Can't find any directory"):
        DataAdapterImgClf(missing_path)


def test_framework_validation_raises_value_error_before_using_adapter_state() -> None:
    adapter = object.__new__(DataAdapterImgClf)

    with pytest.raises(ValueError, match="'framework' must be"):
        adapter.to_framework("unsupported")


def test_validation_still_runs_in_optimized_mode() -> None:
    project_root = Path(__file__).resolve().parents[1]
    code = """
from visionops.DataUtils import _df_make_folds
import pandas as pd

try:
    _df_make_folds(pd.DataFrame({'value': [1, 2]}), 2, fold_col_name=1)
except TypeError as error:
    if "'fold_col_name' must be a string" not in str(error):
        raise
else:
    raise RuntimeError('validation did not run')
"""

    result = subprocess.run(
        [sys.executable, "-O", "-c", code],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
