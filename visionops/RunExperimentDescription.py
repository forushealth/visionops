"""Functional helper for generating reports from evaluator artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

from .RunEvaluator import Evaluator


def run_experiment_description(
    *,
    source: str,
    output_dir: Union[str, Path] = "artifacts/evaluation/evaluator",
    tracking_uri: Optional[str] = None,
    logits_csv_name: Optional[str] = None,
    history_csv_path: Optional[str] = None,
    label_col: Optional[str] = None,
    path_col: Optional[str] = None,
    show: bool = True,
    markdown_only: bool = False,
    include_history: bool = True,
    use_llm: bool = False,
    llm_model: str = "llama2",
) -> Dict[str, str]:
    """Create an evaluator and generate its report artifacts."""
    evaluator = Evaluator(
        source=source,
        output_dir=output_dir,
        tracking_uri=tracking_uri,
        logits_csv_name=logits_csv_name,
        history_csv_path=history_csv_path,
        label_col=label_col,
        path_col=path_col,
    )
    return evaluator.generate_report(
        show=show,
        markdown_only=markdown_only,
        include_history=include_history,
        use_llm=use_llm,
        llm_model=llm_model,
    )


__all__ = ["run_experiment_description"]
