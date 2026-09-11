# Contributing to VisionOps

Thank you for helping improve VisionOps.

## Development setup

Use Python 3.10, 3.11, or 3.12 in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Install the relevant optional extras when changing backend-specific code, for
example `.[dev,torch,image]` or `.[dev,keras,image]`.

## Before opening a pull request

```bash
python -m compileall -q visionops tests
ruff check visionops tests
pytest -q
```

Add focused regression tests for behavior changes. Keep optional dependencies
lazy so importing the core package does not require a training framework,
tracking service, notebook environment, or GPU.

## Pull requests

- Keep each pull request focused on one concern.
- Explain the user-visible behavior and any compatibility impact.
- Update the README or changelog when the public API changes.
- Do not commit datasets, model weights, credentials, run artifacts, or local
  environment files.
- Prefer explicit errors over silent fallbacks and broad exception handling.

Bug reports should include a minimal reproducer, Python version, operating
system, installed extras, and the complete traceback with secrets removed.
