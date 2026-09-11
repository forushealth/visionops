"""Unified trainer adapter for keras and torch backends."""

from __future__ import annotations

from typing import Any, Optional

from .FinderUtils import run_batch_finder, run_lr_finder


class UnifiedTrainer:
    """Single trainer entrypoint for Keras and Torch workflows.

    Usage:
        trainer = UnifiedTrainer(backend="keras", config_path="config.yaml")
        trainer.fit()

        trainer = UnifiedTrainer(backend="torch")
        trainer.fit(model=lightning_module, datamodule=dm, max_epochs=10)
    """

    def __init__(self, backend: str, config_path: Optional[str] = None) -> None:
        """Create backend-specific trainer facade.

        Example:
            >>> trainer = UnifiedTrainer(backend="keras", config_path="config.yaml")
            >>> trainer.backend
            'keras'
        """
        backend = backend.lower().strip()
        if backend not in {"keras", "torch"}:
            raise ValueError("backend must be 'keras' or 'torch'")
        self.backend = backend
        self.config_path = config_path

    def fit(self, **kwargs: Any) -> Any:
        """Run training for the configured backend.

        Example:
            >>> # UnifiedTrainer("keras", "config.yaml").fit()  # doctest: +SKIP
            >>> True
            True
        """
        if self.backend == "keras":
            return self._fit_keras(**kwargs)
        return self._fit_torch(**kwargs)

    def find_batch_size(self, **kwargs: Any) -> Any:
        """Run backend-aware batch-size finder.

        Example:
            >>> trainer = UnifiedTrainer(backend="keras", config_path="config.yaml")
            >>> # trainer.find_batch_size()  # doctest: +SKIP
            >>> True
            True
        """
        if "config_path" not in kwargs or kwargs.get("config_path") is None:
            kwargs["config_path"] = self.config_path
        return run_batch_finder(self.backend, **kwargs)

    def find_learning_rate(self, **kwargs: Any) -> Any:
        """Run backend-aware learning-rate finder.

        Example:
            >>> trainer = UnifiedTrainer(backend="torch", config_path="config.yaml")
            >>> # trainer.find_learning_rate(start_lr=1e-5, end_lr=1e-2)  # doctest: +SKIP
            >>> True
            True
        """
        if "config_path" not in kwargs or kwargs.get("config_path") is None:
            kwargs["config_path"] = self.config_path
        return run_lr_finder(self.backend, **kwargs)

    def find_hyperparameters(
        self,
        *,
        run_batch_finder: bool = True,
        run_lr_finder: bool = True,
        batch_candidates=None,
        lr_start: float = 1e-5,
        lr_end: float = 5e-2,
        lr_num_steps: int = 40,
        **kwargs: Any,
    ) -> Any:
        """Run configured finder stages and return combined suggestions.

        Example:
            >>> trainer = UnifiedTrainer(backend="keras", config_path="config.yaml")
            >>> trainer.find_hyperparameters(run_batch_finder=False, run_lr_finder=False)
            {'backend': 'keras'}
        """
        result = {"backend": self.backend}

        batch_result = None
        if run_batch_finder:
            batch_kwargs = dict(kwargs)
            if batch_candidates is not None:
                batch_kwargs["batch_candidates"] = batch_candidates
            batch_result = self.find_batch_size(**batch_kwargs)
            result["batch_finder"] = batch_result

        if run_lr_finder:
            lr_kwargs = dict(kwargs)
            if batch_result and isinstance(batch_result, dict):
                suggested_batch = batch_result.get("best_batch_size")
                if suggested_batch is not None and lr_kwargs.get("batch_size") is None:
                    lr_kwargs["batch_size"] = int(suggested_batch)
            lr_kwargs["start_lr"] = float(lr_start)
            lr_kwargs["end_lr"] = float(lr_end)
            lr_kwargs["num_steps"] = int(lr_num_steps)
            result["lr_finder"] = self.find_learning_rate(**lr_kwargs)

        return result

    def _fit_keras(self, **kwargs: Any) -> Any:
        """Train using the Keras backend.

        Example:
            >>> # UnifiedTrainer("keras", "config.yaml")._fit_keras()  # doctest: +SKIP
            >>> True
            True
        """
        if not self.config_path:
            raise ValueError("config_path is required for keras backend")

        from .KerasTrainer import KerasFitTrainer

        trainer = KerasFitTrainer(self.config_path)
        return trainer.fit()

    def _fit_torch(self, **kwargs: Any) -> Any:
        """Torch fit using PyTorch Lightning.

        Required kwargs:
            model: LightningModule
        Optional kwargs:
            datamodule: LightningDataModule
            train_dataloader: DataLoader
            val_dataloader: DataLoader
            max_epochs: int (default 10)
            callbacks: list
            trainer_kwargs: dict

        Example:
            >>> # UnifiedTrainer("torch")._fit_torch(model=module, datamodule=dm)  # doctest: +SKIP
            >>> True
            True
        """
        model = kwargs.get("model")
        if model is None:
            raise ValueError("For torch backend, provide `model` (LightningModule)")

        datamodule = kwargs.get("datamodule")
        train_dl = kwargs.get("train_dataloader")
        val_dl = kwargs.get("val_dataloader")
        max_epochs = int(kwargs.get("max_epochs", 10))
        callbacks = list(kwargs.get("callbacks") or [])
        trainer_kwargs = dict(kwargs.get("trainer_kwargs", {}) or {})

        import pytorch_lightning as L

        # Stabilize notebook/IDE progress rendering (avoids fast redraw flicker).
        progress_refresh_rate = int(trainer_kwargs.pop("progress_refresh_rate", 10))
        if progress_refresh_rate < 1:
            progress_refresh_rate = 10
        trainer_kwargs.setdefault("log_every_n_steps", progress_refresh_rate)

        enable_progress = bool(trainer_kwargs.get("enable_progress_bar", True))
        has_progress_callback = any(
            "progress" in cb.__class__.__name__.lower() and "bar" in cb.__class__.__name__.lower()
            for cb in callbacks
        )
        if enable_progress and not has_progress_callback:
            try:
                from pytorch_lightning.callbacks import TQDMProgressBar

                callbacks.append(TQDMProgressBar(refresh_rate=progress_refresh_rate, leave=True))
            except ModuleNotFoundError:
                pass

        trainer = L.Trainer(max_epochs=max_epochs, callbacks=callbacks, **trainer_kwargs)

        if datamodule is not None:
            return trainer.fit(model, datamodule=datamodule)
        return trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
