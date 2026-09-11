<div align="center">

<h1>VisionOps</h1>
<h3>MLOps-First Computer Vision</h3>

<p><strong>Train · Evaluate · Explain · Track</strong></p>

<p>
  <a href="#install-from-source">Installation</a> ·
  <a href="#features">Features</a> ·
  <a href="CONTRIBUTING.md">Contributing</a> ·
  <a href="LICENSE">Apache-2.0</a>
</p>

</div>

---

VisionOps is an open-source, MLOps-first toolkit for computer vision that makes
structured training, task-aware evaluation, standardized artifacts,
explainability, and experiment tracking part of the workflow—not afterthoughts.
Built in Python, it brings together tools for computer-vision tasks with
Keras and PyTorch Lightning integrations, alongside runtime diagnostics.

## Train · Evaluate · Explain · Track

VisionOps connects four parts of the model-development workflow through
task-aware interfaces and reusable artifacts:

| Capability | What it brings to your workflow |
| --- | --- |
| **Train** | Structured training workflows with Keras and PyTorch Lightning adapters. |
| **Evaluate** | Task-aware metrics and model-free evaluation from saved predictions, with standardized CSV/NPZ artifacts, plots, and reports. |
| **Explain** | CAM-based utilities for inspecting which image regions influence model predictions. |
| **Track** | MLflow integration brings experiment parameters, metrics, and artifacts together, complemented by runtime and GPU diagnostics. |

The MLOps-first approach makes evaluation outputs, explainability, and experiment
records part of model development from the start, rather than separate scripts
added after training.

## Features

- One task contract for binary (`bc`), multiclass (`mcc`), multilabel (`mlc`),
  and segmentation (`seg`) metrics.
- Separate training and model-free evaluation pipelines.
- CSV and NPZ evaluation artifacts with JSON, Markdown, plots, and PDF
  reports.
- Keras and PyTorch Lightning training adapters.
- CAM generation helpers for post-training explainability.
- GPU sampling and runtime diagnostic reports.
- MLflow integration for experiment parameters, metrics, and artifacts.

## Requirements

- Python 3.10, 3.11, or 3.12
- Keras or PyTorch Lightning for model training
- NVIDIA tooling only when GPU diagnostics are required

## Install from source

Clone the repository and install the core package:

```bash
git clone https://github.com/forushealth/visionops.git
cd visionops
python -m pip install .
```

Install only the capabilities you need:

```bash
python -m pip install ".[image]"
python -m pip install ".[keras,image]"
python -m pip install ".[torch,image]"
python -m pip install ".[tracking]"
python -m pip install ".[all]"
```

For development:

```bash
python -m pip install -e ".[dev]"
pytest -q
```

The installable distribution is named `visionops-toolkit`; the Python import is
`visionops`. This avoids a collision with an unrelated distribution that
already uses the shorter name. Release automation is configured to publish
artifacts to GitHub Releases rather than upload them to a package index.

## Quick start: metrics

```python
import numpy as np

from visionops import TaskSpec, compute_metrics

y_true = np.array([0, 1, 2, 1])
y_logits = np.array(
    [
        [3.0, 0.2, 0.1],
        [0.1, 2.8, 0.3],
        [0.0, 0.2, 2.4],
        [0.2, 2.1, 0.4],
    ]
)

metrics = compute_metrics(
    y_true,
    y_logits,
    TaskSpec(task_type="mcc", num_classes=3),
)
print(metrics)
```

## Training

Use `TrainingPipeline` when you need training plus artifact export:

```python
from visionops import TaskSpec, TrainingPipeline

pipeline = TrainingPipeline(
    backend="torch",
    task_spec=TaskSpec(task_type="mcc", num_classes=3),
    config_path="config.yaml",
    use_mlflow=False,
)

pipeline.fit(model=model, datamodule=datamodule, max_epochs=10)
```

For Keras, provide a YAML configuration and install the `keras` extra. For
PyTorch Lightning, pass a model and datamodule/dataloaders or provide the
dataset settings used by the bundled task-aware data module.

Only load model, checkpoint, and configuration artifacts from sources you
trust. Keras and PyTorch loaders may reconstruct executable Python objects, and
configuration files can select code and filesystem paths used by a run.

## Model-free evaluation

An evaluation pipeline can read arrays or a saved NPZ artifact without a model
object:

```python
from visionops import EvaluationPipeline, TaskSpec

evaluator = EvaluationPipeline(
    task_spec=TaskSpec(task_type="mcc", num_classes=3)
)

metrics = evaluator.evaluate_npz("artifacts/eval_outputs.npz")
```

The NPZ file must contain `y_true` and `y_pred` arrays.

## Main public APIs

| Area | APIs |
|---|---|
| Task definition | `TaskSpec`, `TaskType` |
| Metrics | `compute_metrics`, `evaluate_task` |
| Orchestration | `TrainingPipeline`, `EvaluationPipeline`, `Pipeline` |
| Reports | `Evaluator`, `run_experiment_description` |
| Explainability | `generate_cam`, `compare_cams` |
| Diagnostics | `run_dfperf_preflight`, `GPUSampler` |
| Configuration | `ExperimentConfig`, `load_config_yaml`, `save_config_yaml` |

Backend-specific and lower-level utilities remain available as submodules, for
example `visionops.DataUtils` and `visionops.TorchCAMUtils`.

## Development checks

```bash
python -m compileall -q visionops tests
ruff check visionops tests
pytest -q
python -m build
python -m twine check dist/*
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the contribution workflow.

## License

VisionOps is available under the [Apache License 2.0](LICENSE).
