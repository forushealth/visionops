"""Run-level evaluation utilities for logits/history artifacts."""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import tempfile
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from ._utils import csv_has_column, first_existing

from .MetricRegistry import _sigmoid


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    den = np.sum(ex, axis=axis, keepdims=True) + 1e-12
    return ex / den


def _ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


class Evaluator:
    """Post-training evaluator with notebook-friendly API.

    Designed to mirror the original Keras evaluation flow:
    - plots_metrics()
    - prediction_example()
    - optimal_threshold(method='auc'|'roc'|'pr')
    - statistical_tests()
    - history_analysis()
    - generate_report()
    """

    LABEL_COL_CANDIDATES = ("label", "labels", "target", "class")
    PATH_COL_CANDIDATES = ("full_path", "path", "image_path", "file_path", "image")

    def __init__(
        self,
        source: str,
        *,
        output_dir: Union[str, Path] = "artifacts/evaluation/evaluator",
        tracking_uri: Optional[str] = None,
        logits_csv_name: Optional[str] = None,
        history_csv_path: Optional[str] = None,
        label_col: Optional[str] = None,
        path_col: Optional[str] = None,
    ) -> None:
        """Load evaluation artifacts from CSV/run source and prepare tensors."""
        self.source = source
        self.output_dir = _ensure_dir(output_dir)
        self.figures: Dict[str, str] = {}
        self.metrics: Dict[str, Any] = {}
        self._history_summary: Optional[str] = None
        self._stats_cache: Optional[Dict[str, Any]] = None
        self._thresholds: Dict[Union[str, int], float] = {}

        self.logits_csv_path, self.history_csv_path = self._resolve_paths(
            source=source,
            tracking_uri=tracking_uri,
            logits_csv_name=logits_csv_name,
            history_csv_path=history_csv_path,
        )

        self.df_val = pd.read_csv(self.logits_csv_path)
        self.label_col = label_col or first_existing(self.df_val.columns, self.LABEL_COL_CANDIDATES)
        self.path_col = path_col or first_existing(self.df_val.columns, self.PATH_COL_CANDIDATES)
        if self.label_col is None:
            raise ValueError(f"Could not infer label column. Available: {list(self.df_val.columns)}")
        if "logits" not in self.df_val.columns:
            raise ValueError("Validation CSV must contain 'logits' column.")

        # Normalize labels to integer ids for metric computation
        if pd.api.types.is_numeric_dtype(self.df_val[self.label_col]):
            self.y_true = self.df_val[self.label_col].to_numpy().astype(int)
            uniq = sorted(np.unique(self.y_true).tolist())
            self.class_to_id = {int(c): int(c) for c in uniq}
            self.id_to_class = {int(c): int(c) for c in uniq}
        else:
            cat = pd.Categorical(self.df_val[self.label_col])
            self.y_true = cat.codes.astype(int)
            self.class_to_id = {str(c): int(i) for i, c in enumerate(cat.categories)}
            self.id_to_class = {int(i): str(c) for i, c in enumerate(cat.categories)}

        self.logits = self._extract_logits(self.df_val["logits"])
        self.y_prob = self._to_probabilities(self.logits)
        self.num_classes = int(self.y_prob.shape[1])
        self.y_pred = np.argmax(self.y_prob, axis=1).astype(int)

        self.history_df = None
        if self.history_csv_path and os.path.isfile(self.history_csv_path):
            try:
                self.history_df = pd.read_csv(self.history_csv_path)
            except Exception:
                self.history_df = None

    def _resolve_paths(
        self,
        *,
        source: str,
        tracking_uri: Optional[str],
        logits_csv_name: Optional[str],
        history_csv_path: Optional[str],
    ) -> Tuple[str, Optional[str]]:
        # Source as direct CSV path
        if os.path.isfile(source):
            hist = history_csv_path if history_csv_path and os.path.isfile(history_csv_path) else None
            return source, hist

        # Source as run-id (optional mlflow path)
        try:
            from mlflow.tracking import MlflowClient
        except Exception as e:
            raise ValueError(
                f"`source` is not a file path and MLflow is unavailable for run-id lookup: {source}"
            ) from e

        client = MlflowClient(tracking_uri=tracking_uri)
        artifact_dir = Path(client.download_artifacts(source, ""))

        if logits_csv_name:
            logits_csv = artifact_dir / logits_csv_name
            if not logits_csv.exists():
                raise FileNotFoundError(f"Could not find logits csv: {logits_csv}")
        else:
            candidates = list(artifact_dir.rglob("*val*logit*.csv"))
            if not candidates:
                # fallback: any CSV with a logits column
                candidates = list(artifact_dir.rglob("*.csv"))
                candidates = [p for p in candidates if csv_has_column(p, "logits")]
            if not candidates:
                raise FileNotFoundError("Could not locate validation logits CSV in run artifacts.")
            logits_csv = candidates[0]

        hist = None
        if history_csv_path and os.path.isfile(history_csv_path):
            hist = history_csv_path
        else:
            hist_cands = list(artifact_dir.rglob("*history*.csv"))
            if hist_cands:
                hist = str(hist_cands[0])

        return str(logits_csv), hist

    @staticmethod
    def _extract_logits(series: pd.Series) -> np.ndarray:
        rows = []
        for val in series.tolist():
            if isinstance(val, str):
                try:
                    parsed = ast.literal_eval(val)
                except Exception:
                    parsed = float(val)
            else:
                parsed = val
            arr = np.array(parsed, dtype=float).reshape(-1)
            rows.append(arr)

        max_len = max((len(x) for x in rows), default=1)
        logits = np.zeros((len(rows), max_len), dtype=float)
        for i, arr in enumerate(rows):
            logits[i, : len(arr)] = arr
        return logits

    @staticmethod
    def _to_probabilities(logits: np.ndarray) -> np.ndarray:
        if logits.ndim == 1:
            logits = logits.reshape(-1, 1)

        if logits.shape[1] == 1:
            v = logits[:, 0]
            if np.all((v >= 0.0) & (v <= 1.0)):
                pos = v
            else:
                pos = _sigmoid(v)
            return np.vstack([1.0 - pos, pos]).T

        # Multi-class: accept either probabilities or logits.
        row_sums = logits.sum(axis=1)
        if np.all((logits >= 0.0) & (logits <= 1.0)) and np.allclose(row_sums, 1.0, atol=1e-3):
            probs = logits
        else:
            probs = _softmax(logits, axis=1)
        return probs

    def _predict_with_thresholds(self) -> np.ndarray:
        if not self._thresholds:
            return self.y_pred.copy()

        if self.num_classes == 2:
            thr = float(self._thresholds.get("overall", 0.5))
            return (self.y_prob[:, 1] >= thr).astype(int)

        # Multi-class thresholding (one-vs-rest)
        pred = np.full(len(self.y_true), -1, dtype=int)
        for i in range(len(self.y_true)):
            cls_scores = []
            for c in range(self.num_classes):
                thr = float(self._thresholds.get(c, 0.5))
                score = self.y_prob[i, c] - thr
                cls_scores.append(score)
            cls_scores = np.array(cls_scores, dtype=float)
            if np.max(cls_scores) < 0:
                pred[i] = int(np.argmax(self.y_prob[i]))
            else:
                pred[i] = int(np.argmax(cls_scores))
        return pred

    def optimal_threshold(self, *, method: str = "auc", show: bool = True) -> Dict[Union[str, int], float]:
        """Find class decision thresholds from ROC/PR criteria."""
        method = method.lower().strip()
        if method == "auc":
            method = "roc"
        if method not in {"roc", "pr"}:
            raise ValueError("method must be 'auc', 'roc', or 'pr'")

        import matplotlib.pyplot as plt

        outdir = _ensure_dir(self.output_dir / "thresholds")
        thresholds: Dict[Union[str, int], float] = {}

        if self.num_classes == 2:
            y = self.y_true
            s = self.y_prob[:, 1]
            if method == "pr":
                p, r, th = precision_recall_curve(y, s)
                f1 = 2 * p[1:] * r[1:] / (p[1:] + r[1:] + 1e-8)
                idx = int(np.argmax(f1))
                thr = float(th[idx])
                fig, ax = plt.subplots()
                ax.plot(r, p, label="PR")
                ax.scatter(r[idx + 1], p[idx + 1], c="red", s=40, label=f"thr={thr:.3f}")
                ax.set_xlabel("Recall")
                ax.set_ylabel("Precision")
                ax.set_title("PR Threshold")
                ax.legend()
                pth = outdir / "pr_threshold.png"
            else:
                fpr, tpr, th = roc_curve(y, s)
                j = tpr - fpr
                idx = int(np.argmax(j))
                thr = float(th[idx])
                fig, ax = plt.subplots()
                ax.plot(fpr, tpr, label="ROC")
                ax.scatter(fpr[idx], tpr[idx], c="red", s=40, label=f"thr={thr:.3f}")
                ax.plot([0, 1], [0, 1], "--")
                ax.set_xlabel("FPR")
                ax.set_ylabel("TPR")
                ax.set_title("ROC Threshold")
                ax.legend()
                pth = outdir / "roc_threshold.png"

            fig.tight_layout()
            fig.savefig(pth, dpi=150)
            if show:
                plt.show()
            plt.close(fig)

            self.figures[pth.stem] = str(pth)
            thresholds["overall"] = thr

        else:
            # one-vs-rest thresholds per class
            for c in range(self.num_classes):
                y = (self.y_true == c).astype(int)
                s = self.y_prob[:, c]
                if method == "pr":
                    p, r, th = precision_recall_curve(y, s)
                    f1 = 2 * p[1:] * r[1:] / (p[1:] + r[1:] + 1e-8)
                    idx = int(np.argmax(f1))
                    thr = float(th[idx]) if len(th) > 0 else 0.5
                else:
                    fpr, tpr, th = roc_curve(y, s)
                    j = tpr - fpr
                    idx = int(np.argmax(j))
                    thr = float(th[idx]) if len(th) > 0 else 0.5
                thresholds[c] = thr

        self._thresholds = thresholds
        self.metrics["optimal_thresholds"] = thresholds
        return thresholds

    def plots_metrics(self, show: bool = True, use_optimal_threshold: bool = False) -> Dict[str, str]:
        """Generate evaluation plots and compute summary classification metrics."""
        import matplotlib.pyplot as plt

        outdir = _ensure_dir(self.output_dir / "plots")
        paths: Dict[str, str] = {}

        y_pred = self._predict_with_thresholds() if use_optimal_threshold else self.y_pred

        # History curves (if available)
        if self.history_df is not None and not self.history_df.empty:
            epoch = self.history_df.index.to_numpy()
            loss_col = next((c for c in self.history_df.columns if c.lower() == "loss"), None)
            val_loss_col = next((c for c in self.history_df.columns if c.lower() == "val_loss"), None)
            acc_col = next((c for c in self.history_df.columns if "accuracy" in c.lower() and "val_" not in c.lower()), None)
            val_acc_col = next((c for c in self.history_df.columns if "val_accuracy" in c.lower()), None)

            if loss_col:
                fig, ax = plt.subplots()
                ax.plot(epoch, self.history_df[loss_col], label=loss_col)
                if val_loss_col:
                    ax.plot(epoch, self.history_df[val_loss_col], label=val_loss_col)
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")
                ax.set_title("Training Loss")
                ax.legend()
                p = outdir / "history_loss.png"
                fig.tight_layout()
                fig.savefig(p, dpi=150)
                if show:
                    plt.show()
                plt.close(fig)
                paths["history_loss"] = str(p)

            if acc_col:
                fig, ax = plt.subplots()
                ax.plot(epoch, self.history_df[acc_col], label=acc_col)
                if val_acc_col:
                    ax.plot(epoch, self.history_df[val_acc_col], label=val_acc_col)
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Accuracy")
                ax.set_title("Training Accuracy")
                ax.legend()
                p = outdir / "history_accuracy.png"
                fig.tight_layout()
                fig.savefig(p, dpi=150)
                if show:
                    plt.show()
                plt.close(fig)
                paths["history_accuracy"] = str(p)

        cm = confusion_matrix(self.y_true, y_pred)
        fig, ax = plt.subplots()
        im = ax.imshow(cm, cmap="Blues")
        fig.colorbar(im, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        ax.set_xticks(np.arange(cm.shape[1]))
        ax.set_yticks(np.arange(cm.shape[0]))
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center")
        p_cm = outdir / "confusion_matrix.png"
        fig.tight_layout()
        fig.savefig(p_cm, dpi=150)
        if show:
            plt.show()
        plt.close(fig)
        paths["confusion_matrix"] = str(p_cm)

        cls_ids = sorted(np.unique(self.y_true).tolist())
        cls_acc = []
        for c in cls_ids:
            mask = self.y_true == c
            cls_acc.append(float(np.mean(y_pred[mask] == c)) if np.any(mask) else 0.0)
        fig, ax = plt.subplots()
        ax.bar([str(self.id_to_class.get(c, c)) for c in cls_ids], cls_acc)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Class")
        ax.set_ylabel("Accuracy")
        ax.set_title("Per-class Accuracy")
        p_pc = outdir / "per_class_accuracy.png"
        fig.tight_layout()
        fig.savefig(p_pc, dpi=150)
        if show:
            plt.show()
        plt.close(fig)
        paths["per_class_accuracy"] = str(p_pc)

        # ROC/PR for binary; micro-average for multi-class.
        if self.num_classes == 2:
            y_bin = self.y_true
            scores = self.y_prob[:, 1]

            fpr, tpr, _ = roc_curve(y_bin, scores)
            aucv = float(roc_auc_score(y_bin, scores))
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f"AUC={aucv:.4f}")
            ax.plot([0, 1], [0, 1], "--")
            ax.set_xlabel("FPR")
            ax.set_ylabel("TPR")
            ax.set_title("ROC Curve")
            ax.legend()
            p_roc = outdir / "roc_curve.png"
            fig.tight_layout()
            fig.savefig(p_roc, dpi=150)
            if show:
                plt.show()
            plt.close(fig)
            paths["roc_curve"] = str(p_roc)

            precision, recall, _ = precision_recall_curve(y_bin, scores)
            fig, ax = plt.subplots()
            ax.plot(recall, precision)
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title("Precision-Recall Curve")
            p_pr = outdir / "pr_curve.png"
            fig.tight_layout()
            fig.savefig(p_pr, dpi=150)
            if show:
                plt.show()
            plt.close(fig)
            paths["pr_curve"] = str(p_pr)

        self.figures.update(paths)

        self.metrics.update(
            {
                "accuracy": float(accuracy_score(self.y_true, y_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(self.y_true, y_pred)),
                "precision_weighted": float(precision_score(self.y_true, y_pred, average="weighted", zero_division=0)),
                "recall_weighted": float(recall_score(self.y_true, y_pred, average="weighted", zero_division=0)),
                "f1_weighted": float(f1_score(self.y_true, y_pred, average="weighted", zero_division=0)),
            }
        )

        print("\n=== Classification report ===")
        print(classification_report(self.y_true, y_pred, digits=4, zero_division=0))
        return paths

    def prediction_example(self, *, k: int = 5, show: bool = True) -> Dict[str, str]:
        """Render example panels for correct/incorrect/confidence-based predictions."""
        import matplotlib.pyplot as plt
        try:
            from IPython.display import Image as _IPyImage, display as _ip_display
        except Exception:
            _IPyImage = None
            _ip_display = None

        has_path_col = self.path_col is not None and self.path_col in self.df_val.columns
        if show and not has_path_col:
            print("No image path column found; rendering metadata-only example panels.")

        outdir = _ensure_dir(self.output_dir / "examples")
        conf = np.max(self.y_prob, axis=1)
        idx = np.arange(len(self.y_true))

        correct = idx[self.y_pred == self.y_true]
        incorrect = idx[self.y_pred != self.y_true]
        high_conf = idx[np.argsort(conf)[::-1]]
        low_conf = idx[np.argsort(conf)]
        high_conf_wrong = incorrect[np.argsort(conf[incorrect])[::-1]] if len(incorrect) else np.array([], dtype=int)

        groups = {
            "correct_predictions": correct[:k],
            "incorrect_predictions": incorrect[:k],
            "highest_confidence": high_conf[:k],
            "lowest_confidence": low_conf[:k],
            "high_confidence_wrong": high_conf_wrong[:k],
        }

        paths: Dict[str, str] = {}
        for name, sel in groups.items():
            sel = np.array(sel, dtype=int)
            if len(sel) == 0:
                continue

            n_cols = min(3, len(sel))
            n_rows = int(np.ceil(len(sel) / max(1, n_cols)))
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
            axes_arr = np.array(axes, dtype=object).reshape(-1)

            for ax, i in zip(axes_arr, sel, strict=True):
                status_note = ""
                if has_path_col:
                    path_val = self.df_val.iloc[int(i)][self.path_col]
                    if isinstance(path_val, bytes):
                        path_str = path_val.decode("utf-8", errors="ignore")
                    else:
                        path_str = str(path_val)
                    if path_str.startswith(("b'", 'b"')):
                        try:
                            path_str = ast.literal_eval(path_str).decode("utf-8", errors="ignore")
                        except Exception:
                            pass

                    if os.path.isfile(path_str):
                        try:
                            ax.imshow(plt.imread(path_str))
                        except Exception:
                            status_note = "Unreadable image"
                    else:
                        status_note = "Missing image path"
                else:
                    status_note = "No image path column"

                if status_note:
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    ax.text(0.5, 0.55, status_note, ha="center", va="center", fontsize=8)
                    ax.text(0.5, 0.40, f"sample_index={int(i)}", ha="center", va="center", fontsize=7)

                gt = self.id_to_class.get(int(self.y_true[i]), int(self.y_true[i]))
                pd_ = self.id_to_class.get(int(self.y_pred[i]), int(self.y_pred[i]))
                ax.set_title(f"GT: {gt}\\nPred: {pd_}\\nConf: {conf[i]:.2f}", fontsize=8)
                ax.axis("off")

            for ax in axes_arr[len(sel) :]:
                ax.axis("off")

            fig.suptitle(name.replace("_", " ").title(), fontsize=12)
            p = outdir / f"{name}.png"
            fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
            fig.savefig(p, dpi=150)
            if show:
                plt.show()
                if _IPyImage is not None and _ip_display is not None:
                    try:
                        _ip_display(_IPyImage(filename=str(p)))
                    except Exception:
                        pass
            plt.close(fig)
            paths[name] = str(p)

        self.figures.update(paths)
        return paths

    def statistical_tests(self, show: bool = True) -> Dict[str, Any]:
        """Compute statistical diagnostics for predictions and thresholds.

        Args:
            show: Print computed statistics to stdout.
        """
        stats: Dict[str, Any] = {}

        stats["accuracy_argmax"] = float(accuracy_score(self.y_true, self.y_pred))
        stats["balanced_accuracy_argmax"] = float(balanced_accuracy_score(self.y_true, self.y_pred))

        if self.num_classes == 2:
            y = self.y_true
            s = self.y_prob[:, 1]
            try:
                stats["roc_auc"] = float(roc_auc_score(y, s))
            except Exception:
                stats["roc_auc"] = None
            try:
                stats["average_precision"] = float(average_precision_score(y, s))
            except Exception:
                stats["average_precision"] = None

        # Threshold-vs-argmax McNemar-style comparison when thresholds exist.
        if self._thresholds:
            y_thr = self._predict_with_thresholds()
            stats["accuracy_threshold"] = float(accuracy_score(self.y_true, y_thr))

            argmax_correct = self.y_pred == self.y_true
            thr_correct = y_thr == self.y_true
            b = int(np.sum(argmax_correct & ~thr_correct))
            c = int(np.sum(~argmax_correct & thr_correct))
            if (b + c) > 0:
                chi2 = ((abs(b - c) - 1) ** 2) / (b + c)
            else:
                chi2 = 0.0
            stats["mcnemar_b"] = b
            stats["mcnemar_c"] = c
            stats["mcnemar_chi2_approx"] = float(chi2)

            try:
                from scipy.stats import chi2 as chi2_dist

                stats["mcnemar_pvalue_approx"] = float(chi2_dist.sf(chi2, 1))
            except Exception:
                stats["mcnemar_pvalue_approx"] = None

        # Distribution drift: predicted vs true class frequencies.
        try:
            from scipy.stats import chisquare

            obs = np.bincount(self.y_pred, minlength=self.num_classes).astype(float)
            exp = np.bincount(self.y_true, minlength=self.num_classes).astype(float)
            if exp.sum() > 0:
                exp = exp * (obs.sum() / exp.sum())
            ch = chisquare(obs, f_exp=exp)
            stats["pred_dist_chi2"] = float(ch.statistic)
            stats["pred_dist_pvalue"] = float(ch.pvalue)
        except Exception:
            stats["pred_dist_chi2"] = None
            stats["pred_dist_pvalue"] = None

        self._stats_cache = stats
        if show:
            print("\n=== Statistical tests ===")
            try:
                print(json.dumps(stats, indent=2))
            except Exception:
                print(stats)
        return stats

    def history_analysis(
        self,
        show: bool = True,
        *,
        use_llm: bool = False,
        llm_model: str = "llama2",
        llm_timeout_s: int = 120,
    ) -> str:
        """Summarize training history curves from optional history CSV.

        Args:
            show: Print history summary to stdout.
            use_llm: Ask an already-running local Ollama installation to add an
                interpretation. Disabled by default and never starts a service
                or downloads a model.
            llm_model: Preferred name of an already-installed Ollama model.
            llm_timeout_s: Maximum time for the local inference request.
        """
        if self.history_df is None or self.history_df.empty:
            summary = "No training history CSV was found; history analysis skipped."
            self._history_summary = summary
            if show:
                print("\n=== History analysis ===")
                print(summary)
            return summary

        h = self.history_df
        loss_col = next((c for c in h.columns if c.lower() == "loss"), None)
        val_loss_col = next((c for c in h.columns if c.lower() == "val_loss"), None)
        acc_col = next((c for c in h.columns if "accuracy" in c.lower() and "val_" not in c.lower()), None)
        val_acc_col = next((c for c in h.columns if "val_accuracy" in c.lower()), None)

        lines: List[str] = []
        lines.append("Training history analysis:")
        lines.append(f"- epochs_logged: {len(h)}")

        if loss_col:
            lines.append(f"- final_train_loss: {float(h[loss_col].iloc[-1]):.6f}")
            lines.append(f"- best_train_loss: {float(h[loss_col].min()):.6f}")
        if val_loss_col:
            lines.append(f"- final_val_loss: {float(h[val_loss_col].iloc[-1]):.6f}")
            lines.append(f"- best_val_loss: {float(h[val_loss_col].min()):.6f}")
        if acc_col:
            lines.append(f"- final_train_accuracy: {float(h[acc_col].iloc[-1]):.6f}")
            lines.append(f"- best_train_accuracy: {float(h[acc_col].max()):.6f}")
        if val_acc_col:
            best_idx = int(np.argmax(h[val_acc_col].to_numpy()))
            lines.append(f"- final_val_accuracy: {float(h[val_acc_col].iloc[-1]):.6f}")
            lines.append(f"- best_val_accuracy: {float(h[val_acc_col].max()):.6f} (epoch_index={best_idx})")

        fallback_text = self._rule_based_history_insights(
            history_df=h,
            loss_col=loss_col,
            val_loss_col=val_loss_col,
            acc_col=acc_col,
            val_acc_col=val_acc_col,
        )
        lines.append("")
        lines.append("Interpretation summary:")
        lines.append(fallback_text)

        if use_llm:
            llm_prompt = self._build_history_llm_prompt(
                history_df=h,
                loss_col=loss_col,
                val_loss_col=val_loss_col,
                acc_col=acc_col,
                val_acc_col=val_acc_col,
            )
            llm_text, llm_error, used_model = self._run_local_llm(
                llm_prompt,
                model=llm_model,
                timeout_s=llm_timeout_s,
            )
            if llm_text:
                lines.extend(("", f"Local LLM interpretation ({used_model or llm_model}):", llm_text))
                self.metrics["history_llm_status"] = "ok"
            else:
                self.metrics["history_llm_status"] = llm_error or "unavailable"

        summary = "\n".join(lines)
        self._history_summary = summary
        if show:
            print("\n=== History analysis ===")
            print(summary)
        return summary

    @staticmethod
    def _rule_based_history_insights(
        *,
        history_df: pd.DataFrame,
        loss_col: Optional[str],
        val_loss_col: Optional[str],
        acc_col: Optional[str],
        val_acc_col: Optional[str],
    ) -> str:
        def _series(col: Optional[str]) -> List[float]:
            if col is None or col not in history_df.columns:
                return []
            vals = pd.to_numeric(history_df[col], errors="coerce").dropna().tolist()
            return [float(v) for v in vals]

        train_loss = _series(loss_col)
        val_loss = _series(val_loss_col)
        train_acc = _series(acc_col)
        val_acc = _series(val_acc_col)

        notes: List[str] = []

        if train_loss and val_loss:
            val_improved = val_loss[-1] <= val_loss[0]
            train_improved = train_loss[-1] <= train_loss[0]
            if train_improved and val_improved:
                notes.append("Loss curves show stable convergence on both train and validation splits.")
            elif train_improved and not val_improved:
                notes.append("Training loss improved while validation loss did not, indicating possible overfitting.")

        if train_acc and val_acc:
            gap = float(train_acc[-1] - val_acc[-1])
            if gap > 0.08:
                notes.append("Train-validation accuracy gap is notable; generalization can likely be improved.")
            elif gap < 0.03:
                notes.append("Train-validation accuracy gap is small, suggesting good generalization.")
            if val_acc[-1] >= 0.90:
                notes.append("Validation accuracy is strong; prioritize calibration/threshold tuning over major retraining.")
            elif val_acc[-1] >= 0.80:
                notes.append("Validation accuracy is solid; moderate gains may come from data quality and tuning.")
            else:
                notes.append("Validation accuracy is still moderate; additional training/data curation may help.")

        if len(history_df) <= 3:
            notes.append("Only a few epochs are logged; consider running more epochs with early stopping.")

        if not notes:
            notes.append("Training history is available but limited; continue monitoring loss/accuracy trends per epoch.")

        suggestions = (
            "Next steps: review hard false positives/negatives, tune decision threshold on validation data, "
            "and apply mild regularization or augmentation if overfitting signs persist."
        )
        return " ".join(notes + [suggestions])

    def _build_history_llm_prompt(
        self,
        *,
        history_df: pd.DataFrame,
        loss_col: Optional[str],
        val_loss_col: Optional[str],
        acc_col: Optional[str],
        val_acc_col: Optional[str],
    ) -> str:
        def _series_text(col: Optional[str], max_points: int = 30) -> str:
            if col is None or col not in history_df.columns:
                return "[]"
            vals = pd.to_numeric(history_df[col], errors="coerce").dropna().tolist()
            if len(vals) <= max_points:
                return str([round(float(v), 6) for v in vals])
            head_n = max_points // 2
            tail_n = max_points - head_n
            head = [round(float(v), 6) for v in vals[:head_n]]
            tail = [round(float(v), 6) for v in vals[-tail_n:]]
            return f"{head} ... {tail}"

        if self._stats_cache is None:
            try:
                self.statistical_tests(show=False)
            except Exception:
                self._stats_cache = {}

        stats_text = "{}"
        try:
            stats_text = json.dumps(self._stats_cache or {}, indent=2)
        except Exception:
            stats_text = str(self._stats_cache or {})

        try:
            clf_report = classification_report(self.y_true, self.y_pred, digits=4, zero_division=0)
        except Exception:
            clf_report = "classification_report unavailable"

        prompt = f"""
You are a concise ML training analyst.
Interpret the training dynamics and evaluation behavior from the metrics below.
Do not repeat all raw numbers. Focus on diagnosis and actionable next steps.

Epochs: {len(history_df)}
Train loss: {_series_text(loss_col)}
Val loss: {_series_text(val_loss_col)}
Train accuracy: {_series_text(acc_col)}
Val accuracy: {_series_text(val_acc_col)}

Classification report:
{clf_report}

Statistical tests:
{stats_text}

Provide:
1) Short diagnosis (convergence / overfitting / underfitting / stability).
2) Confidence in model quality.
3) Concrete retraining suggestions (data, epochs, thresholding, regularization).
Keep it practical and brief.
""".strip()
        return prompt

    @staticmethod
    def _run_local_llm(
        prompt: str,
        *,
        model: str = "llama2",
        timeout_s: int = 120,
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        def _extract_model_names(raw: str) -> List[str]:
            lines = [ln.strip() for ln in str(raw).splitlines() if ln.strip()]
            if not lines:
                return []
            out: List[str] = []
            for ln in lines[1:]:
                parts = ln.split()
                if parts:
                    out.append(parts[0].strip())
            return out

        def _pick_model(installed: List[str], preferred: str) -> Optional[str]:
            if not installed:
                return None
            prefer = [
                preferred,
                f"{preferred}:latest",
                "llama3.2:1b",
                "qwen2.5:1.5b",
                "phi3:mini",
                "gemma2:2b",
                "llama2:latest",
            ]
            for want in prefer:
                for got in installed:
                    if got == want or got.startswith(f"{want}:") or got.startswith(f"{want.split(':')[0]}:"):
                        return got
            return installed[0]

        # CLI path (preferred). The service and model must already exist.
        if shutil.which("ollama"):
            try:
                list_proc = subprocess.run(
                    ["ollama", "list"],
                    capture_output=True,
                    text=True,
                    timeout=15,
                    check=False,
                )
                if list_proc.returncode != 0:
                    error = (list_proc.stderr or list_proc.stdout or "ollama service is unavailable").strip()
                    return None, error, None

                installed = _extract_model_names(list_proc.stdout or "")
                chosen = _pick_model(installed, model)
                if not chosen:
                    return None, "no local ollama models installed", None

                run_proc = subprocess.run(
                    ["ollama", "run", chosen],
                    input=prompt,
                    capture_output=True,
                    text=True,
                    timeout=max(int(timeout_s), 30),
                    check=False,
                )
                out = (run_proc.stdout or "").strip()
                if run_proc.returncode == 0 and out:
                    return out, None, chosen
                err = (run_proc.stderr or "").strip() or "empty response from ollama CLI"
                return None, err, chosen
            except subprocess.TimeoutExpired:
                return None, "local LLM timed out", None
            except Exception as exc:
                return None, f"ollama CLI failed: {exc}", None

        # Python client fallback (local ollama server must already be running)
        try:
            import ollama

            resp = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            msg = (
                resp.get("message", {}).get("content")
                if isinstance(resp, dict)
                else None
            )
            if msg and str(msg).strip():
                return str(msg).strip(), None, model
            return None, "ollama python client returned empty output", model
        except Exception as exc:
            return None, f"ollama is not available locally: {exc}", None

    def _collect_cam_image_paths(self) -> List[Path]:
        """Collect CAM figure files from common artifact locations."""
        roots = [
            self.output_dir / "cam_images",
            self.output_dir.parent / "cam_images",
            self.output_dir.parent / "evaluation_report" / "cam_images",
            self.output_dir.parent / "evaluation_report_pkg" / "cam_images",
        ]
        out: List[Path] = []
        seen: set[str] = set()
        for root in roots:
            if not root.exists() or not root.is_dir():
                continue
            for p in sorted(root.glob("*.png")):
                rp = str(p.resolve())
                if rp in seen:
                    continue
                seen.add(rp)
                out.append(p)
        return out

    def _collect_report_figures(self) -> Dict[str, str]:
        """Merge evaluator figures with discovered CAM figures."""
        merged: Dict[str, str] = dict(self.figures)
        known = {str(Path(v).resolve()) for v in merged.values()}
        for i, p in enumerate(self._collect_cam_image_paths(), start=1):
            rp = str(p.resolve())
            if rp in known:
                continue
            key = f"cam_example_{i}"
            while key in merged:
                i += 1
                key = f"cam_example_{i}"
            merged[key] = str(p)
            known.add(rp)
        return merged

    @staticmethod
    def _order_figure_map(figure_map: Dict[str, str]) -> Dict[str, str]:
        """Return figures in a stable section-aware order for reports."""
        if not figure_map:
            return {}

        prediction_keys = {
            "correct_predictions",
            "incorrect_predictions",
            "highest_confidence",
            "lowest_confidence",
            "high_confidence_wrong",
        }
        eval_keys = {"confusion_matrix", "per_class_accuracy", "roc_curve", "pr_curve"}

        def _rank(key: str) -> Tuple[int, str]:
            k = str(key).lower()
            if k.startswith("data_"):
                return (0, k)
            if k.startswith("history_"):
                return (1, k)
            if "threshold" in k:
                return (2, k)
            if k in eval_keys:
                return (3, k)
            if k in prediction_keys:
                return (4, k)
            if k.startswith("cam_"):
                return (5, k)
            return (6, k)

        return {k: figure_map[k] for k in sorted(figure_map.keys(), key=_rank)}

    @staticmethod
    def _safe_json_load(path: Optional[Union[str, Path]]) -> Optional[Any]:
        if not path:
            return None
        p = Path(path)
        if not p.is_file():
            return None
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    @staticmethod
    def _safe_yaml_load(path: Optional[Union[str, Path]]) -> Optional[Dict[str, Any]]:
        if not path:
            return None
        p = Path(path)
        if not p.is_file():
            return None
        try:
            import yaml

            with open(p, "r", encoding="utf-8") as f:
                payload = yaml.safe_load(f)
            if isinstance(payload, dict):
                return payload
            return None
        except Exception:
            return None

    @staticmethod
    def _flatten_dict(data: Dict[str, Any], *, prefix: str = "") -> Dict[str, Any]:
        flat: Dict[str, Any] = {}
        for k, v in (data or {}).items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict):
                flat.update(Evaluator._flatten_dict(v, prefix=key))
            else:
                flat[key] = v
        return flat

    @staticmethod
    def _resolve_path_like(path_value: Any, *, base_dir: Optional[Path] = None) -> Optional[str]:
        if not isinstance(path_value, str) or not path_value.strip():
            return None
        raw = Path(path_value.strip()).expanduser()
        candidates: List[Path] = []
        if raw.is_absolute():
            candidates.append(raw)
        else:
            if base_dir is not None:
                candidates.append((base_dir / raw).resolve())
            candidates.append(raw.resolve())
        for cand in candidates:
            if cand.is_file():
                return str(cand)
        return None

    def _discover_artifact_catalog(self) -> Tuple[Optional[str], Dict[str, Any]]:
        candidate_dirs: List[Path] = []
        logits_parent = Path(self.logits_csv_path).resolve().parent
        candidate_dirs.append(logits_parent)
        candidate_dirs.extend(list(logits_parent.parents)[:4])
        out_dir = Path(self.output_dir).resolve()
        candidate_dirs.append(out_dir)
        candidate_dirs.extend(list(out_dir.parents)[:3])

        seen: set[str] = set()
        for root in candidate_dirs:
            for cand in (
                root / "artifact_catalog.json",
                root / "dfperf" / "artifact_catalog.json",
                root / "artifacts" / "dfperf" / "artifact_catalog.json",
            ):
                key = str(cand)
                if key in seen:
                    continue
                seen.add(key)
                payload = self._safe_json_load(cand)
                if isinstance(payload, dict):
                    return str(cand), payload
        return None, {}

    def _summarize_split_dataframe(
        self,
        *,
        split_name: str,
        csv_path: str,
        outdir: Path,
    ) -> Tuple[Dict[str, Any], Dict[str, str]]:
        summary: Dict[str, Any] = {
            "split": split_name,
            "csv_path": csv_path,
            "status": "missing",
        }
        fig_map: Dict[str, str] = {}

        p = Path(csv_path)
        if not p.is_file():
            return summary, fig_map

        try:
            df = pd.read_csv(p)
        except Exception as exc:
            summary["status"] = f"read_error: {exc}"
            return summary, fig_map

        summary["status"] = "ok"
        summary["rows"] = int(len(df))
        summary["num_columns"] = int(len(df.columns))
        summary["columns"] = [str(c) for c in df.columns.tolist()]
        summary["missing_values_total"] = int(df.isna().sum().sum())

        label_col = first_existing(df.columns, self.LABEL_COL_CANDIDATES)
        summary["label_column"] = label_col

        if label_col is not None and len(df) > 0:
            counts = (
                df[label_col]
                .map(lambda x: str(x))
                .value_counts(dropna=False)
                .sort_values(ascending=False)
            )
            summary["label_distribution"] = {str(k): int(v) for k, v in counts.to_dict().items()}

            try:
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(8, 4))
                labels = [str(x) for x in counts.index.tolist()]
                values = [int(x) for x in counts.values.tolist()]
                ax.bar(labels, values)
                ax.set_xlabel("Label")
                ax.set_ylabel("Count")
                ax.set_title(f"{split_name.title()} Label Distribution")
                if len(labels) > 8:
                    ax.tick_params(axis="x", labelrotation=45)
                fig.tight_layout()
                out_path = outdir / f"data_{split_name}_label_distribution.png"
                fig.savefig(out_path, dpi=150)
                plt.close(fig)
                fig_key = f"data_{split_name}_label_distribution"
                fig_map[fig_key] = str(out_path)
            except Exception:
                pass

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            numeric_summary: Dict[str, Any] = {}
            for col in numeric_cols[:10]:
                ser = pd.to_numeric(df[col], errors="coerce").dropna()
                if ser.empty:
                    continue
                numeric_summary[str(col)] = {
                    "mean": float(ser.mean()),
                    "std": float(ser.std(ddof=0)),
                    "min": float(ser.min()),
                    "max": float(ser.max()),
                }
            if numeric_summary:
                summary["numeric_describe"] = numeric_summary

        return summary, fig_map

    def _build_extended_report_context(self) -> Dict[str, Any]:
        """Collect upstream train/data artifacts and produce report-ready sections."""
        ctx: Dict[str, Any] = {
            "upstream_artifacts": {},
            "data_overview": {"splits": {}, "notes": []},
            "starting_hyperparameters": {},
            "pre_training_details": {},
            "training_details": {},
        }

        artifact_catalog_path, catalog = self._discover_artifact_catalog()
        catalog = catalog if isinstance(catalog, dict) else {}
        top_artifacts = catalog.get("top_artifacts", {}) if isinstance(catalog.get("top_artifacts"), dict) else {}
        data_artifacts = catalog.get("data_related", {}) if isinstance(catalog.get("data_related"), dict) else {}
        pretrain_artifacts = (
            catalog.get("pre_training_related", {})
            if isinstance(catalog.get("pre_training_related"), dict)
            else {}
        )
        training_artifacts = (
            catalog.get("training_related", {})
            if isinstance(catalog.get("training_related"), dict)
            else {}
        )
        evaluation_artifacts = (
            catalog.get("evaluation_related", {})
            if isinstance(catalog.get("evaluation_related"), dict)
            else {}
        )

        ctx["upstream_artifacts"] = {
            "artifact_catalog_path": artifact_catalog_path,
            "top_artifacts": top_artifacts,
            "data_related": data_artifacts,
            "pre_training_related": pretrain_artifacts,
            "training_related": training_artifacts,
            "evaluation_related": evaluation_artifacts,
        }

        config_path = self._resolve_path_like(top_artifacts.get("config_path"))
        if config_path is None:
            # Try common config snapshot locations near logits/eval artifacts.
            logits_parent = Path(self.logits_csv_path).resolve().parent
            for cand in (
                logits_parent.parent / "config" / "config_effective_train.yaml",
                logits_parent.parent / "config" / "config.yaml",
                logits_parent.parent / "config.yaml",
            ):
                if cand.is_file():
                    config_path = str(cand)
                    break

        config_payload = self._safe_yaml_load(config_path)
        if config_path:
            ctx["starting_hyperparameters"]["config_path"] = config_path

        if isinstance(config_payload, dict):
            flat_cfg = self._flatten_dict(config_payload)
            hp_order = [
                "model_name",
                "train.model_name",
                "learning_rate",
                "train.learning_rate",
                "batch_size",
                "train.batch_size",
                "epochs",
                "train.epochs",
                "optimizer",
                "train.optimizer",
                "loss",
                "train.loss",
                "task_type",
                "num_classes",
                "data.image_size",
                "image_size",
                "data.shuffle_train",
                "data.oversample_train",
            ]
            selected: Dict[str, Any] = {}
            for key in hp_order:
                if key in flat_cfg:
                    selected[key] = flat_cfg[key]
            if selected:
                ctx["starting_hyperparameters"]["selected"] = selected

        # Data overview from train/val/test CSVs when available; fallback to logits CSV.
        split_csv_paths: Dict[str, Optional[str]] = {"train": None, "val": None, "test": None}
        if isinstance(config_payload, dict):
            cfg_data = config_payload.get("data", {}) if isinstance(config_payload.get("data"), dict) else {}
            for split_key, cfg_key in (("train", "train_data_csv"), ("val", "val_data_csv"), ("test", "test_data_csv")):
                raw_val = config_payload.get(cfg_key)
                if raw_val is None:
                    raw_val = cfg_data.get(cfg_key)
                split_csv_paths[split_key] = self._resolve_path_like(
                    raw_val,
                    base_dir=Path(config_path).resolve().parent if config_path else None,
                )

        if not split_csv_paths["val"]:
            split_csv_paths["val"] = self.logits_csv_path

        data_fig_dir = _ensure_dir(self.output_dir / "report_data")
        split_summaries: Dict[str, Any] = {}
        split_sizes: Dict[str, int] = {}
        for split, pth in split_csv_paths.items():
            if not pth:
                continue
            summary, figs = self._summarize_split_dataframe(split_name=split, csv_path=pth, outdir=data_fig_dir)
            split_summaries[split] = summary
            if isinstance(summary.get("rows"), int):
                split_sizes[split] = int(summary["rows"])
            if figs:
                self.figures.update(figs)

        if split_sizes:
            try:
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(6, 4))
                keys = list(split_sizes.keys())
                vals = [int(split_sizes[k]) for k in keys]
                ax.bar(keys, vals)
                ax.set_xlabel("Dataset Split")
                ax.set_ylabel("Rows")
                ax.set_title("Dataset Split Sizes")
                fig.tight_layout()
                split_fig = data_fig_dir / "data_split_sizes.png"
                fig.savefig(split_fig, dpi=150)
                plt.close(fig)
                self.figures["data_split_sizes"] = str(split_fig)
            except Exception:
                pass

        ctx["data_overview"]["splits"] = split_summaries
        if not split_summaries:
            ctx["data_overview"]["notes"].append("No train/val/test CSVs were discoverable from upstream artifacts.")

        # Pre-training details from dfperf artifacts.
        pretraining_details: Dict[str, Any] = {"artifacts": dict(pretrain_artifacts)}
        preflight_path = self._resolve_path_like(pretrain_artifacts.get("dfperf_preflight_json"))
        preflight_payload = self._safe_json_load(preflight_path)
        if preflight_path:
            pretraining_details["dfperf_preflight_json"] = preflight_path
        if isinstance(preflight_payload, dict):
            pretraining_details["preflight"] = preflight_payload

        runtime_report_path = self._resolve_path_like(pretrain_artifacts.get("dfperf_report_json"))
        runtime_report_payload = self._safe_json_load(runtime_report_path)
        if runtime_report_path:
            pretraining_details["dfperf_report_json"] = runtime_report_path
        if isinstance(runtime_report_payload, dict):
            pretraining_details["runtime_report"] = runtime_report_payload

        runtime_metrics_path = self._resolve_path_like(pretrain_artifacts.get("dfperf_metrics_json"))
        runtime_metrics_payload = self._safe_json_load(runtime_metrics_path)
        if runtime_metrics_path:
            pretraining_details["dfperf_metrics_json"] = runtime_metrics_path
        if isinstance(runtime_metrics_payload, dict):
            pretraining_details["runtime_metrics"] = runtime_metrics_payload
        ctx["pre_training_details"] = pretraining_details

        # Training details from history/model artifacts.
        training_details: Dict[str, Any] = {
            "artifacts": dict(training_artifacts),
            "history_csv_path": self.history_csv_path,
            "logits_csv_path": self.logits_csv_path,
        }
        if self.history_df is not None and not self.history_df.empty:
            hist_cols = [str(c) for c in self.history_df.columns]
            training_details["history_columns"] = hist_cols
            training_details["epochs_logged"] = int(len(self.history_df))

            def _latest(col_name: str) -> Optional[float]:
                if col_name in self.history_df.columns:
                    try:
                        return float(pd.to_numeric(self.history_df[col_name], errors="coerce").dropna().iloc[-1])
                    except Exception:
                        return None
                return None

            for col in ("loss", "val_loss", "accuracy", "val_accuracy"):
                v = _latest(col)
                if v is not None:
                    training_details[f"final_{col}"] = v

        if self._history_summary:
            training_details["history_analysis"] = self._history_summary

        ctx["training_details"] = training_details
        return ctx

    def _write_pdf_report(self, *, report: Dict[str, Any], figure_map: Dict[str, str], out_pdf: Path) -> None:
        def _ascii_text(v: Any) -> str:
            return str(v).encode("ascii", errors="replace").decode("ascii")

        def _compact_path(v: Any, *, keep_parts: int = 4) -> str:
            s = _ascii_text(v)
            if not s:
                return s
            p = Path(s)
            parts = p.parts
            if len(parts) <= keep_parts:
                return s
            tail = "/".join(parts[-keep_parts:])
            return f".../{tail}"

        def _latex_escape(v: Any) -> str:
            s = _ascii_text(v)
            repl = {
                "\\": r"\textbackslash{}",
                "&": r"\&",
                "%": r"\%",
                "$": r"\$",
                "#": r"\#",
                "_": r"\_",
                "{": r"\{",
                "}": r"\}",
                "~": r"\textasciitilde{}",
                "^": r"\textasciicircum{}",
            }
            return "".join(repl.get(ch, ch) for ch in s)

        def _latex_url(v: Any) -> str:
            s = _ascii_text(v).replace("\\", "/")
            s = s.replace("{", "").replace("}", "")
            return r"\url{" + s + "}"

        def _to_pretty_json(v: Any) -> str:
            try:
                return json.dumps(v if v is not None else {}, indent=2, ensure_ascii=True, default=str)
            except Exception:
                return _ascii_text(v)

        def _add_latex_text_section(tex_lines: List[str], title: str, text_value: Any) -> None:
            tex_lines.extend([rf"\section*{{{_latex_escape(title)}}}", r"\begin{flushleft}"])
            text = _ascii_text(text_value)
            for raw in (text.splitlines() or ["N/A"]):
                wrapped = textwrap.wrap(raw, width=96, break_long_words=True, replace_whitespace=False) or [""]
                for ln in wrapped:
                    tex_lines.append(_latex_escape(ln) + r"\\")
            tex_lines.append(r"\end{flushleft}")

        # Preferred: vector-quality PDF via pdflatex (when available).
        pdflatex = shutil.which("pdflatex")
        if pdflatex:
            try:
                with tempfile.TemporaryDirectory(prefix="visionops_eval_report_tex_") as tmpdir:
                    tmpdir_path = Path(tmpdir)
                    tex_path = tmpdir_path / "report.tex"
                    pdf_path = tmpdir_path / "report.pdf"

                    tex_lines: List[str] = [
                        r"\documentclass[11pt]{article}",
                        r"\usepackage[a4paper,left=1.20in,right=1.00in,top=0.95in,bottom=0.95in]{geometry}",
                        r"\usepackage{graphicx}",
                        r"\usepackage{float}",
                        r"\usepackage{xurl}",
                        r"\usepackage[T1]{fontenc}",
                        r"\usepackage[utf8]{inputenc}",
                        r"\setlength{\parskip}{6pt}",
                        r"\setlength{\parindent}{0pt}",
                        r"\raggedright",
                        r"\begin{document}",
                        r"\begin{center}",
                        r"{\LARGE Evaluator Report}\\[4pt]",
                        r"\end{center}",
                        r"\section*{Summary}",
                        r"\begin{itemize}",
                        rf"\item source: {_latex_url(_compact_path(report.get('source')))}",
                        rf"\item logits\_csv\_path: {_latex_url(_compact_path(report.get('logits_csv_path')))}",
                        rf"\item history\_csv\_path: {_latex_url(_compact_path(report.get('history_csv_path')))}",
                        rf"\item num\_samples: {_latex_escape(report.get('num_samples'))}",
                        rf"\item num\_classes: {_latex_escape(report.get('num_classes'))}",
                        r"\end{itemize}",
                    ]
                    _add_latex_text_section(
                        tex_lines, "Data Overview", _to_pretty_json(report.get("data_overview") or {})
                    )
                    _add_latex_text_section(
                        tex_lines,
                        "Starting Hyperparameters",
                        _to_pretty_json(report.get("starting_hyperparameters") or {}),
                    )
                    _add_latex_text_section(
                        tex_lines,
                        "Pre-Training Details",
                        _to_pretty_json(report.get("pre_training_details") or {}),
                    )
                    _add_latex_text_section(
                        tex_lines, "Training Details", _to_pretty_json(report.get("training_details") or {})
                    )
                    tex_lines.extend(
                        [
                        r"\section*{Evaluation Metrics}",
                        r"\begin{itemize}",
                        ]
                    )
                    for k, v in (report.get("metrics") or {}).items():
                        tex_lines.append(rf"\item {_latex_escape(k)}: {_latex_escape(v)}")
                    tex_lines.extend([r"\end{itemize}", r"\section*{Optimal Thresholds}", r"\begin{itemize}"])
                    thresholds = report.get("optimal_thresholds") or {}
                    if thresholds:
                        for k, v in thresholds.items():
                            tex_lines.append(rf"\item {_latex_escape(k)}: {_latex_escape(v)}")
                    else:
                        tex_lines.append(r"\item (not computed)")
                    tex_lines.extend([r"\end{itemize}", r"\section*{Statistical Tests}", r"\begin{itemize}"])
                    for k, v in (report.get("statistical_tests") or {}).items():
                        tex_lines.append(rf"\item {_latex_escape(k)}: {_latex_escape(v)}")
                    hist_text = _ascii_text(report.get("history_analysis") or "N/A")
                    tex_lines.append(r"\end{itemize}")
                    _add_latex_text_section(tex_lines, "History Analysis", hist_text)

                    for key, p in figure_map.items():
                        pth = Path(p)
                        if not pth.is_file():
                            continue
                        abs_path = str(pth.resolve()).replace("\\", "/")
                        tex_lines.extend(
                            [
                                r"\clearpage",
                                rf"\section*{{{_latex_escape(key)}}}",
                                r"\begin{figure}[H]",
                                r"\centering",
                                rf"\includegraphics[width=0.92\linewidth,height=0.82\textheight,keepaspectratio]{{\detokenize{{{abs_path}}}}}",
                                r"\end{figure}",
                            ]
                        )

                    tex_lines.append(r"\end{document}")
                    tex_path.write_text("\n".join(tex_lines), encoding="utf-8")

                    for _ in range(2):
                        proc = subprocess.run(
                            [pdflatex, "-interaction=nonstopmode", "-halt-on-error", "-output-directory", str(tmpdir_path), str(tex_path)],
                            capture_output=True,
                            text=True,
                            timeout=240,
                            check=False,
                        )
                        if proc.returncode != 0:
                            raise RuntimeError(f"pdflatex failed: {proc.stdout}\n{proc.stderr}")

                    if pdf_path.is_file():
                        out_pdf.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(pdf_path, out_pdf)
                        return
            except Exception:
                # Fall back to rasterized PDF generation below.
                pass

        # Fallback: rasterized, compatibility-focused PDF.
        import matplotlib.pyplot as plt
        from PIL import Image

        with tempfile.TemporaryDirectory(prefix="visionops_eval_report_pages_") as tmpdir:
            tmpdir_path = Path(tmpdir)
            page_paths: List[Path] = []
            page_counter = 0
            page_dpi = 220

            def _new_page(title: str, *, landscape: bool = False):
                # Keep one consistent page orientation so margins look uniform across all pages.
                fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
                fig.text(0.06, 0.975, title, fontsize=14, weight="bold", va="top", ha="left")
                return fig

            def _save_page(fig) -> None:
                nonlocal page_counter
                page_counter += 1
                page_path = tmpdir_path / f"page_{page_counter:04d}.png"
                fig.savefig(page_path, dpi=page_dpi, facecolor="white", pad_inches=0.2)
                plt.close(fig)
                page_paths.append(page_path)

            def _add_text_page(title: str, lines: List[str]) -> None:
                fig = _new_page(title, landscape=False)
                y = 0.94
                step = 0.023
                page_idx = 1
                for raw in lines:
                    wrapped = textwrap.wrap(str(raw), width=104, break_long_words=True, replace_whitespace=False) or [""]
                    for chunk in wrapped:
                        if y < 0.05:
                            _save_page(fig)
                            page_idx += 1
                            fig = _new_page(f"{title} (cont. {page_idx})", landscape=False)
                            y = 0.94
                        fig.text(0.06, y, chunk, fontsize=10, va="top", ha="left")
                        y -= step
                _save_page(fig)

            def _add_image_page(title: str, image_path: Path) -> None:
                try:
                    img = plt.imread(str(image_path))
                except Exception:
                    return

                fig = _new_page(title, landscape=False)
                ax = fig.add_axes([0.08, 0.08, 0.84, 0.82])
                if getattr(img, "ndim", 0) == 2:
                    ax.imshow(img, cmap="gray")
                else:
                    ax.imshow(img)
                ax.axis("off")
                _save_page(fig)

            def _json_lines(payload: Any) -> List[str]:
                return _to_pretty_json(payload).splitlines()

            summary_lines = [
                f"source: {_compact_path(report.get('source'))}",
                f"logits_csv_path: {_compact_path(report.get('logits_csv_path'))}",
                f"history_csv_path: {_compact_path(report.get('history_csv_path'))}",
                f"num_samples: {report.get('num_samples')}",
                f"num_classes: {report.get('num_classes')}",
            ]
            _add_text_page("Evaluator Report Summary", summary_lines)

            _add_text_page("Data Overview", _json_lines(report.get("data_overview") or {}))
            _add_text_page("Starting Hyperparameters", _json_lines(report.get("starting_hyperparameters") or {}))
            _add_text_page("Pre-Training Details", _json_lines(report.get("pre_training_details") or {}))
            _add_text_page("Training Details", _json_lines(report.get("training_details") or {}))

            eval_lines = ["metrics:"]
            for k, v in (report.get("metrics") or {}).items():
                eval_lines.append(f"- {k}: {v}")
            eval_lines.append("")
            eval_lines.append("optimal_thresholds:")
            thresholds = report.get("optimal_thresholds") or {}
            if thresholds:
                for k, v in thresholds.items():
                    eval_lines.append(f"- {k}: {v}")
            else:
                eval_lines.append("- (not computed)")
            _add_text_page("Evaluation Metrics", eval_lines)

            stat_lines = ["statistical_tests:"]
            for k, v in (report.get("statistical_tests") or {}).items():
                stat_lines.append(f"- {k}: {v}")
            _add_text_page("Statistical Tests", stat_lines)

            history_text = str(report.get("history_analysis") or "N/A")
            _add_text_page("History Analysis", history_text.splitlines())

            for key, p in figure_map.items():
                pth = Path(p)
                if not pth.is_file():
                    continue
                _add_image_page(str(key), pth)

            if not page_paths:
                raise RuntimeError("No report pages were generated for PDF output.")

            pil_pages: List[Image.Image] = []
            try:
                for page_path in page_paths:
                    with Image.open(page_path) as im:
                        pil_pages.append(im.convert("RGB"))
                first, *rest = pil_pages
                first.save(str(out_pdf), "PDF", resolution=float(page_dpi), quality=95, save_all=True, append_images=rest)
            finally:
                for im in pil_pages:
                    try:
                        im.close()
                    except Exception:
                        pass

    @staticmethod
    def _pandoc_markdown_to_pdf(md_path: Path, out_pdf: Path) -> Tuple[bool, str]:
        pandoc_bin = shutil.which("pandoc")
        if not pandoc_bin:
            return False, "pandoc not installed"

        engines = [e for e in ("xelatex", "lualatex", "pdflatex") if shutil.which(e)]
        if not engines:
            engines = [""]

        last_err = "pandoc conversion failed"
        for engine in engines:
            cmd = [pandoc_bin, str(md_path), "-o", str(out_pdf)]
            if engine:
                cmd += ["--pdf-engine", engine]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=240, check=False)
            if proc.returncode == 0 and out_pdf.is_file():
                return True, (f"pandoc:{engine}" if engine else "pandoc:default")
            last_err = (proc.stderr or proc.stdout or last_err).strip()
        return False, last_err

    def generate_pdf_report(
        self,
        *,
        markdown_path: Optional[Union[str, Path]] = None,
        output_pdf: Optional[Union[str, Path]] = None,
        report: Optional[Dict[str, Any]] = None,
        figure_map: Optional[Dict[str, str]] = None,
        show: bool = True,
    ) -> str:
        """Generate PDF report from markdown (legacy/original path) with fallback renderer."""
        md_path = Path(markdown_path) if markdown_path is not None else (self.output_dir / "report.md")
        out_pdf = Path(output_pdf) if output_pdf is not None else (self.output_dir / "report.pdf")

        if not md_path.is_file():
            raise FileNotFoundError(f"Markdown report not found: {md_path}")

        ok, backend = self._pandoc_markdown_to_pdf(md_path, out_pdf)
        if ok:
            if show:
                print(f"pdf_backend: {backend}")
            return str(out_pdf)

        if report is None or figure_map is None:
            if "accuracy" not in self.metrics:
                self.plots_metrics(show=False)
            if self._stats_cache is None:
                self.statistical_tests(show=False)
            if self._history_summary is None:
                self.history_analysis(show=False)
            if not any(k in self.figures for k in ("correct_predictions", "incorrect_predictions")):
                try:
                    self.prediction_example(show=False)
                except Exception:
                    pass
            extended_context = self._build_extended_report_context()
            auto_figure_map = self._order_figure_map(self._collect_report_figures())
            auto_report = {
                "source": self.source,
                "logits_csv_path": self.logits_csv_path,
                "history_csv_path": self.history_csv_path,
                "num_samples": int(len(self.y_true)),
                "num_classes": int(self.num_classes),
                "metrics": self.metrics,
                "optimal_thresholds": self._thresholds,
                "statistical_tests": self._stats_cache,
                "history_analysis": self._history_summary,
                "figures": auto_figure_map,
            }
            auto_report.update(extended_context)
            if report is None:
                report = auto_report
            else:
                for k, v in extended_context.items():
                    report.setdefault(k, v)
            if figure_map is None:
                figure_map = auto_figure_map

        if figure_map is None:
            figure_map = self._order_figure_map(self._collect_report_figures())
        else:
            figure_map = self._order_figure_map(figure_map)
        if report is not None:
            report["figures"] = figure_map

        self._write_pdf_report(report=report, figure_map=figure_map, out_pdf=out_pdf)
        if show:
            print("pdf_backend: internal_renderer")
        return str(out_pdf)

    # Alias kept for users expecting an explicit PDF function in the API.
    def generate_report(
        self,
        show: bool = True,
        *,
        markdown_only: bool = False,
        include_history: bool = True,
        use_llm: bool = False,
        llm_model: str = "llama2",
    ) -> Dict[str, str]:
        """Write JSON/Markdown evaluation report files and return their paths.

        Args:
            show: Print generated report artifact paths to stdout.
            markdown_only: When True, skip PDF generation.
            include_history: When True, ensure history analysis is included.
            use_llm: Add analysis from an already-running local Ollama service.
            llm_model: Preferred already-installed Ollama model.
        """
        _ensure_dir(self.output_dir)

        # Ensure summary metrics are present
        if "accuracy" not in self.metrics:
            self.plots_metrics(show=False)
        if self._stats_cache is None:
            self.statistical_tests()
        if include_history and self._history_summary is None:
            self.history_analysis(show=False, use_llm=use_llm, llm_model=llm_model)
        if not any(k in self.figures for k in ("correct_predictions", "incorrect_predictions")):
            try:
                self.prediction_example(show=False)
            except Exception:
                pass

        extended_context = self._build_extended_report_context()
        figure_map = self._order_figure_map(self._collect_report_figures())

        report = {
            "source": self.source,
            "logits_csv_path": self.logits_csv_path,
            "history_csv_path": self.history_csv_path,
            "num_samples": int(len(self.y_true)),
            "num_classes": int(self.num_classes),
            "metrics": self.metrics,
            "optimal_thresholds": self._thresholds,
            "statistical_tests": self._stats_cache,
            "history_analysis": self._history_summary,
            "figures": figure_map,
        }
        report.update(extended_context)

        report_json = self.output_dir / "report.json"
        report_md = self.output_dir / "report.md"
        report_pdf = self.output_dir / "report.pdf"

        with open(report_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)

        def _append_json_section(lines: List[str], title: str, payload: Any) -> None:
            lines.extend(["", f"## {title}", "```json"])
            try:
                lines.append(json.dumps(payload if payload is not None else {}, indent=2, ensure_ascii=False, default=str))
            except Exception:
                lines.append(str(payload))
            lines.append("```")

        def _append_figure(lines: List[str], key: str, path_value: str) -> None:
            pth = Path(path_value)
            try:
                rel = os.path.relpath(str(pth), str(self.output_dir))
            except Exception:
                rel = str(pth)
            lines.append(f"### {key}")
            lines.append(f"![{key}]({rel})")
            lines.append(f"- path: {path_value}")
            lines.append("")

        md_lines = [
            "# evaluator report",
            "",
            "## Summary",
            f"- source: {self.source}",
            f"- logits_csv_path: {self.logits_csv_path}",
            f"- history_csv_path: {self.history_csv_path}",
            f"- num_samples: {len(self.y_true)}",
            f"- num_classes: {self.num_classes}",
        ]
        _append_json_section(md_lines, "Data Overview", report.get("data_overview") or {})

        data_figure_keys = [k for k in figure_map.keys() if str(k).lower().startswith("data_")]
        md_lines.extend(["", "## Data Describe Graphs"])
        if data_figure_keys:
            for key in data_figure_keys:
                _append_figure(md_lines, key, figure_map[key])
        else:
            md_lines.append("- (none generated)")

        _append_json_section(md_lines, "Starting Hyperparameters", report.get("starting_hyperparameters") or {})
        _append_json_section(md_lines, "Pre-Training Details", report.get("pre_training_details") or {})
        _append_json_section(md_lines, "Training Details", report.get("training_details") or {})

        md_lines.extend(["", "## Evaluation Metrics"])
        for k, v in (report.get("metrics") or {}).items():
            md_lines.append(f"- {k}: {v}")

        md_lines.extend(["", "## Optimal Thresholds"])
        if report.get("optimal_thresholds"):
            for k, v in (report.get("optimal_thresholds") or {}).items():
                md_lines.append(f"- {k}: {v}")
        else:
            md_lines.append("- (not computed)")

        md_lines.extend(["", "## Statistical Tests"])
        for k, v in (report.get("statistical_tests") or {}).items():
            md_lines.append(f"- {k}: {v}")

        md_lines.extend(["", "## History Analysis", report.get("history_analysis") or "N/A", "", "## Figures"])
        remaining_figure_keys = [k for k in figure_map.keys() if k not in set(data_figure_keys)]
        if remaining_figure_keys:
            for key in remaining_figure_keys:
                _append_figure(md_lines, key, figure_map[key])
        else:
            md_lines.append("- (none generated)")

        with open(report_md, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))

        pdf_path = ""
        if not markdown_only:
            try:
                pdf_path = self.generate_pdf_report(
                    markdown_path=report_md,
                    output_pdf=report_pdf,
                    report=report,
                    figure_map=figure_map,
                    show=False,
                )
            except Exception:
                pdf_path = ""

        out = {
            "report_json": str(report_json),
            "report_md": str(report_md),
        }
        if pdf_path:
            out["report_pdf"] = pdf_path
        if show:
            print("\n=== Evaluator report ===")
            print(f"report_json: {out['report_json']}")
            print(f"report_md: {out['report_md']}")
            if out.get("report_pdf"):
                print(f"report_pdf: {out['report_pdf']}")
        return out
