"""Unified batch-size and learning-rate finder utilities.

These helpers select backend-specific logic internally while exposing a single
API surface for notebooks and pipelines.
"""

from __future__ import annotations

import contextlib
import gc
import warnings
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .HpUtils import load_raw_config_yaml


def _normalize_backend(backend: str) -> str:
    """Normalize and validate backend string.

    Example:
        >>> _normalize_backend("Keras")
        'keras'
    """
    value = str(backend).lower().strip()
    if value not in {"keras", "torch"}:
        raise ValueError("backend must be 'keras' or 'torch'")
    return value


def _normalize_task_type(task_type: Optional[str]) -> str:
    """Normalize task aliases used by trainers.

    Example:
        >>> _normalize_task_type("bc")
        'mcc'
    """
    value = str(task_type or "mcc").lower().strip()
    if "." in value:
        value = value.split(".")[-1].strip()
    if value == "bc":
        return "mcc"
    return value


def _is_oom_error(message: str) -> bool:
    """Return True when an error message looks like OOM.

    Example:
        >>> _is_oom_error("CUDA out of memory")
        True
    """
    text = str(message).lower()
    return (
        ("out of memory" in text)
        or ("resourceexhausted" in text)
        or ("cuda out of memory" in text)
        or ("cudnn_status_alloc_failed" in text)
        or ("failed to allocate memory" in text)
    )


def _safe_int(value: Any, default: int) -> int:
    """Convert value to int, or return default on failure.

    Example:
        >>> _safe_int("16", 8)
        16
    """
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float) -> float:
    """Convert value to float, or return default on failure.

    Example:
        >>> _safe_float("1e-3", 0.1)
        0.001
    """
    try:
        return float(value)
    except Exception:
        return float(default)


def _config_from_inputs(
    config_path: Optional[str],
    config_dict: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Optional[str]]:
    """Resolve effective config dictionary and path from inputs.

    Example:
        >>> _config_from_inputs(None, {"yaml_path": "config.yaml", "batch_size": 16})[0]["batch_size"]
        16
    """
    if isinstance(config_dict, dict):
        cfg = dict(config_dict)
        if config_path is None and isinstance(cfg.get("yaml_path"), str):
            config_path = cfg.get("yaml_path")
        return cfg, config_path
    if isinstance(config_path, str) and config_path:
        return load_raw_config_yaml(config_path), config_path
    return {}, config_path


def _resolve_task_and_classes(
    cfg: Dict[str, Any],
    task_type: Optional[str],
    num_classes: Optional[int],
) -> Tuple[str, int]:
    """Resolve task type and class count from explicit args or config.

    Example:
        >>> _resolve_task_and_classes({"task_type": "bc", "num_classes": 2}, None, None)
        ('mcc', 2)
    """
    resolved_task = _normalize_task_type(task_type or cfg.get("task_type", "mcc"))
    resolved_classes = _safe_int(num_classes if num_classes is not None else cfg.get("num_classes", 2), 2)
    return resolved_task, max(1, resolved_classes)


def _suggest_lr(lrs: Sequence[float], losses: Sequence[float], start_lr: float) -> float:
    """Suggest LR from loss curve by taking half of best-loss LR.

    Example:
        >>> round(_suggest_lr([1e-4, 1e-3], [0.8, 0.2], 1e-5), 7)
        0.0005
    """
    if not lrs or not losses:
        return float(start_lr)
    arr = np.array(losses, dtype=np.float64)
    finite_mask = np.isfinite(arr)
    if not finite_mask.any():
        return float(start_lr)
    finite_losses = arr[finite_mask]
    finite_lrs = np.array(lrs, dtype=np.float64)[finite_mask]
    idx = int(np.argmin(finite_losses))
    return float(max(float(start_lr), float(finite_lrs[idx]) / 2.0))


@contextlib.contextmanager
def _quiet_finder_runtime(enabled: bool):
    if not enabled:
        yield
        return

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        warnings.filterwarnings(
            "ignore",
            message=r".*Using lambda is incompatible with multiprocessing.*",
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*calling iterator did not fully read the dataset being cached.*",
        )
        yield


def _keras_probe_one_step(config_path: str, batch_size: int, quiet: bool = True) -> float:
    """Run one Keras probe step and return loss.

    Example:
        >>> # _keras_probe_one_step("config.yaml", 16)  # doctest: +SKIP
        >>> True
        True
    """
    import tensorflow as tf
    from .KerasTrainer import KerasFitTrainer

    with _quiet_finder_runtime(bool(quiet)):
        model = None
        trainer = None
        train_ds = None
        metrics = None
        x_batch = None
        y_batch = None
        try:
            trainer = KerasFitTrainer(config_path)
            trainer.config["batch_size"] = int(batch_size)
            probe_samples = max(64, int(batch_size) * 4)
            train_ds, _ = trainer.create_dataset(
                trainer.train_csv_path,
                batch_size=int(batch_size),
                is_training=True,
                max_samples=probe_samples,
                validate_images=False,
                use_cache=False,
                prefetch_buffer=1,
            )

            with trainer.strategy.scope():
                model = trainer._build_model()
                model = trainer._compile_model(model)

            for x_batch, y_batch in train_ds.take(2):
                metrics = model.train_on_batch(x_batch, y_batch, return_dict=True)

            if metrics is None:
                return float("inf")
            return float(metrics.get("loss", float("inf")))
        finally:
            # Explicit teardown is critical for repeated probe loops on GPU.
            try:
                del x_batch, y_batch, metrics, train_ds, model, trainer
            except Exception:
                pass
            gc.collect()
            tf.keras.backend.clear_session()


def _keras_set_learning_rate(optimizer: Any, lr: float) -> None:
    """Set Keras optimizer learning-rate across TF/Keras variants.

    Example:
        >>> # _keras_set_learning_rate(model.optimizer, 1e-3)  # doctest: +SKIP
        >>> True
        True
    """
    lr_value = float(lr)
    target = getattr(optimizer, "learning_rate", None)

    # Modern Keras variable style.
    try:
        if target is not None and hasattr(target, "assign"):
            target.assign(lr_value)
            return
    except Exception:
        pass

    # Legacy Keras backend setter style.
    try:
        import tensorflow as tf

        if target is not None:
            tf.keras.backend.set_value(target, lr_value)
            return
    except Exception:
        pass

    # Property assignment fallback.
    try:
        optimizer.learning_rate = lr_value
        return
    except Exception:
        pass

    # Older optimizer API fallback (`optimizer.lr`).
    old_target = getattr(optimizer, "lr", None)
    try:
        if old_target is not None and hasattr(old_target, "assign"):
            old_target.assign(lr_value)
            return
    except Exception:
        pass
    try:
        optimizer.lr = lr_value
        return
    except Exception:
        pass

    raise ValueError(
        "Could not set optimizer learning_rate. "
        f"optimizer={type(optimizer).__name__}, target_type={type(target).__name__}"
    )


def _torch_build_components(
    cfg: Dict[str, Any],
    task_type: str,
    num_classes: int,
    batch_size: int,
    learning_rate: float,
):
    """Build torch model and datamodule for finder runs.

    Example:
        >>> # _torch_build_components(cfg, "mcc", 2, 16, 1e-3)  # doctest: +SKIP
        >>> True
        True
    """
    from .PLTrainingUtils import TaskClassifier, TaskDataLoaders

    datamodule = TaskDataLoaders(
        config_dict=cfg,
        batch_size=int(batch_size),
        task_type=str(task_type),
        num_classes=int(num_classes),
    )
    datamodule.setup("fit")

    model = TaskClassifier(
        task_type=str(task_type),
        num_classes=int(num_classes),
        model_name=str(cfg.get("model_name", "resnet18")),
        learning_rate=float(learning_rate),
    )
    return model, datamodule


def _torch_probe_one_step(
    cfg: Dict[str, Any],
    task_type: str,
    num_classes: int,
    batch_size: int,
    learning_rate: float,
) -> float:
    """Run one torch probe step and return loss.

    Example:
        >>> # _torch_probe_one_step(cfg, "mcc", 2, 16, 1e-3)  # doctest: +SKIP
        >>> True
        True
    """
    import torch

    model, datamodule = _torch_build_components(
        cfg=cfg,
        task_type=task_type,
        num_classes=int(num_classes),
        batch_size=int(batch_size),
        learning_rate=float(learning_rate),
    )
    loader = datamodule.train_dataloader()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))

    x_batch, y_batch = next(iter(loader))
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)

    optimizer.zero_grad(set_to_none=True)
    logits = model(x_batch)
    loss, *_ = model._loss_and_metrics(logits, y_batch)
    loss.backward()
    optimizer.step()

    loss_value = float(loss.detach().cpu())
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return loss_value


def run_batch_finder(
    backend: str,
    *,
    config_path: Optional[str] = None,
    config_dict: Optional[Dict[str, Any]] = None,
    task_type: Optional[str] = None,
    num_classes: Optional[int] = None,
    batch_candidates: Optional[Iterable[int]] = None,
    learning_rate: Optional[float] = None,
    stop_on_oom: bool = True,
    quiet: bool = True,
    keras_safety_downshift: int = 1,
) -> Dict[str, Any]:
    """Run one-step batch-size probing and return a backend-specific suggestion.

    Example:
        >>> from visionops import run_batch_finder
        >>> out = run_batch_finder("keras", config_path="config.yaml", stop_on_oom=True)  # doctest: +SKIP
        >>> "best_batch_size" in out  # doctest: +SKIP
        True
    """
    selected_backend = _normalize_backend(backend)
    cfg, resolved_path = _config_from_inputs(config_path=config_path, config_dict=config_dict)
    resolved_task, resolved_classes = _resolve_task_and_classes(
        cfg=cfg,
        task_type=task_type,
        num_classes=num_classes,
    )

    default_batch = _safe_int(cfg.get("batch_size", 16), 16)
    default_lr = _safe_float(cfg.get("learning_rate", 1e-3), 1e-3)
    selected_lr = _safe_float(learning_rate, default_lr)
    raw_candidates: Any = batch_candidates
    if raw_candidates is None:
        raw_candidates = cfg.get("finder_batch_candidates")
    if raw_candidates is None:
        raw_candidates = [8, 16, 24, 32, 48, 64]

    parsed_candidates: List[int] = []
    if isinstance(raw_candidates, str):
        tokens = str(raw_candidates).replace(";", ",").split(",")
        for token in tokens:
            token = token.strip()
            if not token:
                continue
            try:
                parsed_candidates.append(int(float(token)))
            except Exception:
                continue
    else:
        try:
            for value in raw_candidates:
                parsed_candidates.append(int(value))
        except Exception:
            parsed_candidates = []

    candidates: List[int] = []
    seen = set()
    for value in parsed_candidates:
        if value <= 0 or value in seen:
            continue
        seen.add(value)
        candidates.append(int(value))
    if not candidates:
        raise ValueError("batch_candidates must contain at least one value.")

    if selected_backend == "keras" and not resolved_path:
        raise ValueError("Keras batch finder requires config_path or config_dict['yaml_path'].")

    history: List[Dict[str, Any]] = []
    best_batch = int(default_batch)

    for batch_size in candidates:
        try:
            if selected_backend == "keras":
                loss_value = _keras_probe_one_step(
                    config_path=str(resolved_path),
                    batch_size=int(batch_size),
                    quiet=bool(quiet),
                )
            else:
                if not cfg:
                    raise ValueError("Torch batch finder requires config_dict or config_path.")
                loss_value = _torch_probe_one_step(
                    cfg=cfg,
                    task_type=resolved_task,
                    num_classes=resolved_classes,
                    batch_size=int(batch_size),
                    learning_rate=float(selected_lr),
                )

            history.append(
                {
                    "batch_size": int(batch_size),
                    "status": "ok",
                    "probe_loss": float(loss_value),
                }
            )
            best_batch = int(batch_size)
        except Exception as exc:
            message = str(exc)
            history.append(
                {
                    "batch_size": int(batch_size),
                    "status": "error",
                    "error": message,
                }
            )
            if stop_on_oom and _is_oom_error(message):
                break

    success_batches = [int(item["batch_size"]) for item in history if item.get("status") == "ok"]
    max_stable_batch = int(success_batches[-1]) if success_batches else int(default_batch)
    if selected_backend == "keras":
        downshift = max(
            0,
            _safe_int(cfg.get("finder_batch_safety_downshift", keras_safety_downshift), keras_safety_downshift),
        )
        if success_batches:
            recommended_idx = max(0, len(success_batches) - 1 - downshift)
            best_batch = int(success_batches[recommended_idx])
        else:
            best_batch = int(default_batch)
    else:
        downshift = 0
        best_batch = int(max_stable_batch)

    return {
        "backend": selected_backend,
        "task_type": resolved_task,
        "num_classes": int(resolved_classes),
        "best_batch_size": int(best_batch),
        "max_stable_batch_size": int(max_stable_batch),
        "safety_downshift_applied": int(downshift),
        "history": history,
    }


def _keras_lr_finder(
    config_path: str,
    start_lr: float,
    end_lr: float,
    num_steps: int,
    batch_size: int,
    quiet: bool = True,
) -> Dict[str, Any]:
    """Run Keras LR range test and return LR/loss trace.

    Example:
        >>> # _keras_lr_finder("config.yaml", 1e-5, 1e-2, 20, 16)  # doctest: +SKIP
        >>> True
        True
    """
    import tensorflow as tf
    from .KerasTrainer import KerasFitTrainer

    with _quiet_finder_runtime(bool(quiet)):
        trainer = None
        model = None
        train_ds = None
        x_batch = None
        y_batch = None
        try:
            trainer = KerasFitTrainer(config_path)
            trainer.config["batch_size"] = int(batch_size)

            steps = max(2, int(num_steps))
            probe_samples = max(int(batch_size) * max(steps, 8), 64)
            train_ds, _ = trainer.create_dataset(
                trainer.train_csv_path,
                batch_size=int(batch_size),
                is_training=True,
                max_samples=probe_samples,
                validate_images=False,
                use_cache=False,
                prefetch_buffer=1,
            )

            with trainer.strategy.scope():
                model = trainer._build_model()
                model = trainer._compile_model(model)

            lr_mult = (float(end_lr) / float(start_lr)) ** (1.0 / float(steps - 1))
            lr = float(start_lr)

            lrs: List[float] = []
            losses: List[float] = []
            best_loss = float("inf")
            stopped_reason: Optional[str] = None

            iterator = iter(train_ds.repeat())
            for _ in range(steps):
                _keras_set_learning_rate(model.optimizer, float(lr))
                x_batch, y_batch = next(iterator)
                try:
                    metrics = model.train_on_batch(x_batch, y_batch, return_dict=True)
                except Exception as exc:
                    message = str(exc)
                    # Some TF stacks intermittently fail graph compilation during finder probing.
                    if ("jit compilation failed" in message.lower()) or _is_oom_error(message):
                        stopped_reason = message
                        break
                    raise
                loss = float(metrics.get("loss", np.nan))

                lrs.append(float(lr))
                losses.append(loss)

                if np.isfinite(loss):
                    best_loss = min(best_loss, loss)
                if (not np.isfinite(loss)) or (loss > 4.0 * max(best_loss, 1e-8)):
                    break
                lr *= lr_mult

            return {
                "backend": "keras",
                "lrs": lrs,
                "losses": losses,
                "steps_run": len(lrs),
                "suggested_lr": _suggest_lr(lrs, losses, start_lr=float(start_lr)),
                "stopped_reason": stopped_reason,
            }
        finally:
            try:
                del x_batch, y_batch, train_ds, model, trainer
            except Exception:
                pass
            gc.collect()
            tf.keras.backend.clear_session()


def _torch_lr_finder(
    cfg: Dict[str, Any],
    task_type: str,
    num_classes: int,
    start_lr: float,
    end_lr: float,
    num_steps: int,
    batch_size: int,
) -> Dict[str, Any]:
    """Run torch LR range test and return LR/loss trace.

    Example:
        >>> # _torch_lr_finder(cfg, "mcc", 2, 1e-5, 1e-2, 20, 16)  # doctest: +SKIP
        >>> True
        True
    """
    import torch

    model, datamodule = _torch_build_components(
        cfg=cfg,
        task_type=task_type,
        num_classes=int(num_classes),
        batch_size=int(batch_size),
        learning_rate=float(start_lr),
    )
    loader = datamodule.train_dataloader()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(start_lr))

    steps = max(2, int(num_steps))
    lr_mult = (float(end_lr) / float(start_lr)) ** (1.0 / float(steps - 1))
    lr = float(start_lr)

    lrs: List[float] = []
    losses: List[float] = []
    best_loss = float("inf")

    iterator = iter(loader)
    for _ in range(steps):
        try:
            x_batch, y_batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x_batch, y_batch = next(iterator)

        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        for group in optimizer.param_groups:
            group["lr"] = float(lr)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x_batch)
        loss, *_ = model._loss_and_metrics(logits, y_batch)
        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().cpu())
        lrs.append(float(lr))
        losses.append(loss_value)

        if np.isfinite(loss_value):
            best_loss = min(best_loss, loss_value)
        if (not np.isfinite(loss_value)) or (loss_value > 4.0 * max(best_loss, 1e-8)):
            break
        lr *= lr_mult

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "backend": "torch",
        "lrs": lrs,
        "losses": losses,
        "steps_run": len(lrs),
        "suggested_lr": _suggest_lr(lrs, losses, start_lr=float(start_lr)),
    }


def run_lr_finder(
    backend: str,
    *,
    config_path: Optional[str] = None,
    config_dict: Optional[Dict[str, Any]] = None,
    task_type: Optional[str] = None,
    num_classes: Optional[int] = None,
    start_lr: float = 1e-5,
    end_lr: float = 5e-2,
    num_steps: int = 40,
    batch_size: Optional[int] = None,
    quiet: bool = True,
) -> Dict[str, Any]:
    """Run backend-specific LR finder and return suggested learning rate.

    Example:
        >>> from visionops import run_lr_finder
        >>> out = run_lr_finder("torch", config_path="config.yaml", start_lr=1e-5, end_lr=1e-2)  # doctest: +SKIP
        >>> "suggested_lr" in out  # doctest: +SKIP
        True
    """
    selected_backend = _normalize_backend(backend)
    cfg, resolved_path = _config_from_inputs(config_path=config_path, config_dict=config_dict)
    resolved_task, resolved_classes = _resolve_task_and_classes(
        cfg=cfg,
        task_type=task_type,
        num_classes=num_classes,
    )
    selected_batch = _safe_int(batch_size if batch_size is not None else cfg.get("batch_size", 16), 16)

    if float(start_lr) <= 0 or float(end_lr) <= 0:
        raise ValueError("start_lr and end_lr must be positive.")
    if float(start_lr) >= float(end_lr):
        raise ValueError("start_lr must be smaller than end_lr.")

    if selected_backend == "keras":
        if not resolved_path:
            raise ValueError("Keras LR finder requires config_path or config_dict['yaml_path'].")
        result = _keras_lr_finder(
            config_path=str(resolved_path),
            start_lr=float(start_lr),
            end_lr=float(end_lr),
            num_steps=int(num_steps),
            batch_size=int(selected_batch),
            quiet=bool(quiet),
        )
    else:
        if not cfg:
            raise ValueError("Torch LR finder requires config_dict or config_path.")
        result = _torch_lr_finder(
            cfg=cfg,
            task_type=resolved_task,
            num_classes=resolved_classes,
            start_lr=float(start_lr),
            end_lr=float(end_lr),
            num_steps=int(num_steps),
            batch_size=int(selected_batch),
        )

    result["task_type"] = resolved_task
    result["num_classes"] = int(resolved_classes)
    result["batch_size"] = int(selected_batch)
    return result


def summarize_finder_result(result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return compact batch/LR summary from finder output.

    Example:
        >>> summarize_finder_result({"backend": "keras", "batch_finder": {"best_batch_size": 32}})
        {'backend': 'keras', 'best_batch_size': 32, 'max_stable_batch_size': None, 'safety_downshift_applied': None, 'suggested_lr': None}
    """
    data = result if isinstance(result, dict) else {}
    batch_block = data.get("batch_finder", {}) if isinstance(data.get("batch_finder", {}), dict) else {}
    lr_block = data.get("lr_finder", {}) if isinstance(data.get("lr_finder", {}), dict) else {}
    return {
        "backend": data.get("backend"),
        "best_batch_size": batch_block.get("best_batch_size"),
        "max_stable_batch_size": batch_block.get("max_stable_batch_size"),
        "safety_downshift_applied": batch_block.get("safety_downshift_applied"),
        "suggested_lr": lr_block.get("suggested_lr"),
    }


def plot_lr_finder_curve(
    lr_result: Optional[Dict[str, Any]],
    figsize: Tuple[int, int] = (7, 4),
    title: Optional[str] = None,
    show_plot: bool = True,
):
    """Plot LR-vs-loss curve from LR finder output.

    Example:
        >>> fig = plot_lr_finder_curve({"lrs": [1e-5, 1e-4], "losses": [1.0, 0.8]}, show_plot=False)
        >>> fig is not None
        True
    """
    import matplotlib.pyplot as plt

    block = lr_result if isinstance(lr_result, dict) else {}
    lrs = block.get("lrs", [])
    losses = block.get("losses", [])
    if not lrs or not losses:
        return None

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(lrs, losses)
    ax.set_xscale("log")
    ax.set_xlabel("learning rate")
    ax.set_ylabel("loss")
    ax.set_title(title or "LR Finder")
    ax.grid(alpha=0.2)

    suggested = block.get("suggested_lr")
    if suggested is not None:
        try:
            suggested_val = float(suggested)
            ax.axvline(suggested_val, color="red", linestyle="--", label=f"suggested={suggested_val:.2e}")
            ax.legend()
        except Exception:
            pass

    plt.tight_layout()
    if show_plot:
        plt.show()
    return fig
