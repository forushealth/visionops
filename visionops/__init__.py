"""Public API for VisionOps."""

from importlib.metadata import PackageNotFoundError, version as _package_version

try:
    __version__ = _package_version("visionops-toolkit")
except PackageNotFoundError:
    __version__ = "0+unknown"

from .CamUtils import compare_cams, generate_cam, visualise_CAM
from .DfPerfUtils import (
    finish_dfperf_runtime_monitor,
    run_dfperf_preflight,
    start_dfperf_runtime_monitor,
)
from .EvaluationPipeline import EvaluationPipeline
from .EvaluationUtils import evaluate_classification, evaluate_task
from .FinderUtils import (
    plot_lr_finder_curve,
    run_batch_finder,
    run_lr_finder,
    summarize_finder_result,
)
from .GPUUtilisation import (
    GPUSampler,
    generate_gpu_plots,
    gpu_snapshot,
    has_nvidia_smi,
    sample_gpu_utilisation,
    summarize_gpu_samples,
)
from .HpUtils import (
    DataConfig,
    DataSplit,
    ExperimentConfig,
    LoggingConfig,
    TrainConfig,
    TransformConfig,
    build_stage_image_processing,
    create_experiment_config,
    freeze_preprocessing_manifest,
    load_config_yaml,
    load_frozen_preprocessing_manifest,
    load_raw_config_yaml,
    normalize_image_processing_sections,
    resolve_stage_config,
    resolve_stage_from_yaml,
    save_config_yaml,
    split_image_processing_transforms,
)
from .ImageProcessingOptions import (
    build_image_processing_pipelines,
    list_image_processing_options,
)
from .MetricRegistry import compute_metrics
from .Pipeline import Pipeline
from .RunEvaluator import Evaluator
from .RunExperimentDescription import run_experiment_description
from .TaskSpec import TaskSpec, TaskType
from .TrainingPipeline import TrainingPipeline
from .UnifiedTrainer import UnifiedTrainer
from ._quiet import configure_runtime_noise

__all__ = [
    "__version__",
    "DataSplit",
    "TransformConfig",
    "DataConfig",
    "TrainConfig",
    "LoggingConfig",
    "ExperimentConfig",
    "create_experiment_config",
    "load_config_yaml",
    "load_raw_config_yaml",
    "load_frozen_preprocessing_manifest",
    "normalize_image_processing_sections",
    "split_image_processing_transforms",
    "build_stage_image_processing",
    "freeze_preprocessing_manifest",
    "resolve_stage_from_yaml",
    "resolve_stage_config",
    "save_config_yaml",
    "UnifiedTrainer",
    "Pipeline",
    "TrainingPipeline",
    "EvaluationPipeline",
    "Evaluator",
    "run_experiment_description",
    "generate_cam",
    "compare_cams",
    "visualise_CAM",
    "build_image_processing_pipelines",
    "list_image_processing_options",
    "run_batch_finder",
    "run_lr_finder",
    "summarize_finder_result",
    "plot_lr_finder_curve",
    "evaluate_classification",
    "evaluate_task",
    "TaskSpec",
    "TaskType",
    "compute_metrics",
    "run_dfperf_preflight",
    "start_dfperf_runtime_monitor",
    "finish_dfperf_runtime_monitor",
    "GPUSampler",
    "has_nvidia_smi",
    "gpu_snapshot",
    "sample_gpu_utilisation",
    "summarize_gpu_samples",
    "generate_gpu_plots",
    "configure_runtime_noise",
]
