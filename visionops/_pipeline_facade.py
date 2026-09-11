"""Shared user-facing helpers for training orchestrators."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .CamUtils import compare_cams, generate_cam, visualise_CAM


class PipelineFacadeMixin:
    """Provide shared tracking, finder, and explainability methods."""

    def setup_mlflow(self, config_path: Optional[str] = None):
        """Create and cache an MLflow logger when tracking is enabled."""
        if not self.use_mlflow:
            return None
        path = config_path or self.config_path
        if not path:
            raise ValueError("config_path is required to initialize MLflowLogger")

        from .MLflowLogger import MLflowLogger

        self.mlflow_logger = MLflowLogger(path)
        return self.mlflow_logger

    def _finder_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        if kwargs.get("task_type") is None:
            kwargs["task_type"] = str(self.task_spec.task_type)
        if kwargs.get("num_classes") is None:
            kwargs["num_classes"] = int(self.task_spec.num_classes)
        if kwargs.get("config_path") is None:
            kwargs["config_path"] = self.config_path
        return kwargs

    def find_batch_size(self, **kwargs: Any) -> Dict[str, Any]:
        """Run the backend-aware batch-size finder."""
        return self._trainer.find_batch_size(**self._finder_kwargs(kwargs))

    def find_learning_rate(self, **kwargs: Any) -> Dict[str, Any]:
        """Run the backend-aware learning-rate finder."""
        return self._trainer.find_learning_rate(**self._finder_kwargs(kwargs))

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
    ) -> Dict[str, Any]:
        """Run configured batch-size and learning-rate finder stages."""
        return self._trainer.find_hyperparameters(
            run_batch_finder=run_batch_finder,
            run_lr_finder=run_lr_finder,
            batch_candidates=batch_candidates,
            lr_start=lr_start,
            lr_end=lr_end,
            lr_num_steps=lr_num_steps,
            **self._finder_kwargs(kwargs),
        )

    def _cam_model(self, model: Any, operation: str) -> Any:
        resolved = model if model is not None else self.model
        if resolved is None:
            raise ValueError(f"No model available. Train first or pass model=... to {operation}().")
        if self.task_spec.is_segmentation:
            raise ValueError("CAM helpers currently support classification tasks only.")
        return resolved

    def explain(self, image: Any, target_layer: Any = None, **kwargs: Any):
        """Generate a CAM heatmap for one image."""
        model = self._cam_model(kwargs.pop("model", None), "explain")
        return generate_cam(
            model=model,
            image=image,
            framework=self.backend,
            target_layer=target_layer,
            class_index=kwargs.get("class_index"),
            cam_method=kwargs.get("cam_method", "gradcam"),
        )

    def compare_cams(
        self,
        image: Any,
        target_layer: Any = None,
        cam_methods=("gradcam", "gradcamplusplus", "xgradcam", "layercam"),
        **kwargs: Any,
    ):
        """Compare multiple CAM methods for one image."""
        model = self._cam_model(kwargs.pop("model", None), "compare_cams")
        return compare_cams(
            model=model,
            image=image,
            framework=self.backend,
            target_layer=target_layer,
            cam_methods=cam_methods,
            class_index=kwargs.get("class_index"),
            figsize=kwargs.get("figsize", (12, 8)),
            overlay=kwargs.get("overlay", True),
            alpha=kwargs.get("alpha", 0.45),
            show_plot=kwargs.get("show_plot", True),
        )

    def visualise_cams(
        self,
        dataset_array: Any,
        target_layer: Any = None,
        cam_names=("GradCAM", "EigenCAM"),
        image_number: int = 8,
        **kwargs: Any,
    ):
        """Render a CAM comparison grid over several images."""
        model = self._cam_model(kwargs.pop("model", None), "visualise_cams")
        return visualise_CAM(
            dataset_array=dataset_array,
            model=model,
            framework=self.backend,
            target_layer=target_layer,
            cam_names=cam_names,
            class_index=kwargs.get("class_index"),
            image_number=image_number,
            overlay=kwargs.get("overlay", True),
            alpha=kwargs.get("alpha", 0.45),
            figsize=kwargs.get("figsize"),
            show_plot=kwargs.get("show_plot", True),
        )
