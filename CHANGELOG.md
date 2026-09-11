# Changelog

All notable changes to VisionOps are documented here. The project follows
[Semantic Versioning](https://semver.org/).

## [0.1.1] - 2026-08-27

### Added

- Canonical `visionops` package for training, evaluation, explainability, and
  runtime diagnostics.
- MIT license, contribution and security policies, and GitHub CI/release
  automation.
- Regression tests for the public namespace and filesystem utilities.

### Changed

- Made backend, image-processing, notebook, and tracking dependencies optional.
- Replaced duplicated filesystem search implementations with one validated API.
- Made runtime noise configuration explicit instead of changing process state
  during package import.
- Separated the installable `visionops-toolkit` distribution from the
  `visionops` import namespace to avoid distribution-name collisions.
- Consolidated duplicated pipeline lifecycle, facade, filesystem, CSV-header,
  array-conversion, image-border, and CAM implementations.
- Made MLflow integration lazy and run-scoped without changing global active
  runs or tracking configuration.
- Made local report LLM analysis opt-in without starting services or
  downloading models.
- Hardened evaluation artifacts with non-pickle NPZ loading, weights-only
  PyTorch checkpoint loading, explicit cleanup sequencing, and trusted-input
  guidance.

### Removed

- Internal platform configuration and publishing automation.
- The former compatibility namespace and runtime import-hook alias.
- Unused legacy utility modules and the duplicated PyTorch training stack.
- Placeholder CAM methods whose implementations did not match their names.
- Undocumented CLI and component-registry claims that had no implementation.
- Process-wide stderr redirection from Keras training.
