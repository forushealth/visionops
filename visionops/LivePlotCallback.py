"""Keras callback for live metric plotting and optional MLflow artifact logging."""

import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import tempfile
from typing import Optional, Sequence


class LivePlotCallback(tf.keras.callbacks.Callback):
    """
    Universal live-plotting callback.

    • Notebook (Jupyter / Colab / VS Code interactive) → inline plot.
    • Local script with GUI backend (Tk, Qt, Mac) → pop-up window.
    • Headless server (no DISPLAY) → skips window but still logs a PNG
      every `log_every_n_epochs` epochs via `mlflow_logger`.

    Parameters
    ----------
    mlflow_logger :  your MLflowLogger instance  |  None
        If provided, PNG snapshots are logged as artifacts.
    metrics : tuple(str)
        Keras history keys to plot (default ``('accuracy', 'val_accuracy')``).
    log_every_n_epochs : int
        Frequency for writing PNG artifacts.
    """

    def __init__(
        self,
        mlflow_logger: Optional[object] = None,
        metrics: Sequence[str] = ('accuracy', 'val_accuracy'),
        log_every_n_epochs: int = 1,
    ):
        """Create a plotting callback for selected Keras metrics.

        Args:
            mlflow_logger: Optional logger implementing ``log_artifact``.
            metrics: Metric names from Keras logs to draw over epochs.
            log_every_n_epochs: Artifact logging frequency in epochs.
        """
        super().__init__()
        self.mlflow_logger   = mlflow_logger
        self.metrics         = list(metrics)
        self.log_every       = log_every_n_epochs
        self.buf             = {m: [] for m in self.metrics}
        self.epochs          = []

        # ---------- backend / environment detection ----------
        self._in_notebook = self._detect_notebook()
        self._interactive = False

        if self._in_notebook:
            # Notebook back-end is already interactive
            from IPython.display import display, clear_output
            self._display = display
            self._clear_output = clear_output
            self._interactive = True
        else:
            # Plain script – try GUI back-ends in order
            for backend in ("TkAgg", "Qt5Agg", "MacOSX"):
                try:
                    matplotlib.use(backend, force=True)
                    self._interactive = True
                    break
                except Exception:
                    continue  # backend not available

        # ---------- figure setup ----------
        self.fig, self.ax = plt.subplots()
        self.lines = {m: self.ax.plot([], [], label=m)[0] for m in self.metrics}
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Value")
        self.ax.set_title("Training progress")
        self.ax.legend()
        if self._interactive and not self._in_notebook:
            plt.show(block=False)  # pop-up non-blocking

    # ---------------------------------------------------------
    def _detect_notebook(self):
        try:
            from IPython import get_ipython
            shell = get_ipython().__class__.__name__
            return shell == "ZMQInteractiveShell"
        except Exception:
            return False

    def _redraw(self):
        if self._in_notebook:
            # Refresh inline output
            self._clear_output(wait=True)
            self._display(self.fig)
        elif self._interactive:
            # Refresh GUI window
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    # ---------- Keras hook ----------
    def on_epoch_end(self, epoch, logs=None):
        """Update the live plot and optionally log a PNG artifact."""
        logs = logs or {}
        self.epochs.append(epoch)
        for m in self.metrics:
            self.buf[m].append(logs.get(m, 0.0))
            self.lines[m].set_data(self.epochs, self.buf[m])

        self.ax.relim(); self.ax.autoscale_view()
        self._redraw()

        if (self.mlflow_logger is not None
                and epoch % self.log_every == 0):
            with tempfile.NamedTemporaryFile(suffix=".png",
                                             delete=False) as tmp:
                self.fig.savefig(tmp.name, bbox_inches="tight")
                self.mlflow_logger.log_artifact(tmp.name)
                os.unlink(tmp.name)
