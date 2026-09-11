"""Opt-in runtime noise controls for VisionOps workflows."""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import Any


def _truthy(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}


def is_quiet_mode() -> bool:
    """Return whether quiet mode is enabled (default: enabled)."""
    return _truthy(os.environ.get("VISIONOPS_QUIET", "1"))


def quiet_print(*args: Any, **kwargs: Any) -> None:
    """Print only when quiet mode is disabled."""
    if not is_quiet_mode():
        print(*args, **kwargs)


def configure_runtime_noise() -> None:
    """Apply conservative warning/logging suppression for notebook usage."""
    quiet = is_quiet_mode()
    if not quiet:
        return

    # Keep noisy frameworks quieter by default.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("GLOG_minloglevel", "3")
    os.environ.setdefault("GLOG_logtostderr", "1")
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

    # Prevent matplotlib/fontconfig import spam in restricted environments.
    mpl_dir = Path(os.environ.get("MPLCONFIGDIR", "/tmp/matplotlib-visionops")).expanduser()
    try:
        mpl_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpl_dir)
    except OSError:
        pass

    # Targeted warning suppression for known noisy but low-action messages.
    warnings.filterwarnings("ignore", message=r".*FigureCanvasAgg is non-interactive.*", category=UserWarning)
    warnings.filterwarnings("ignore", message=r".*Corrupt JPEG data:.*")
    warnings.filterwarnings("ignore", message=r".*Unknown image file format.*")
    warnings.filterwarnings("ignore", message=r".*The structure of `inputs` doesn't match the expected structure.*")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"google\.protobuf\..*")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"pytorch_lightning\.utilities\._pytree")

    # Quiet common verbose loggers.
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    logging.getLogger("absl").setLevel(logging.ERROR)
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    logging.getLogger("PIL").setLevel(logging.ERROR)

    try:
        import absl.logging as absl_logging  # type: ignore

        absl_logging._warn_preinit_stderr = False
        absl_logging.set_verbosity(absl_logging.ERROR)
    except (ImportError, AttributeError):
        pass
