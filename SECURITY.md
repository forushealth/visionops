# Security policy

## Supported versions

Until VisionOps reaches 1.0, security fixes are applied to the latest published
minor release only.

| Version | Supported |
|---|---|
| 0.1.x | Yes |
| Earlier versions | No |

## Artifact trust boundary

Treat model, checkpoint, and configuration artifacts as trusted input only.
Keras and PyTorch deserializers may reconstruct executable Python objects, and
configuration files can select code and filesystem paths used by a run. Do not
load artifacts from unknown or unverified sources.

## Reporting a vulnerability

Please use the repository's private GitHub Security Advisory reporting flow.
Do not open a public issue for an unpatched vulnerability.

Include the affected version, impact, reproduction steps, and any suggested
mitigation. Remove credentials, private datasets, model artifacts, and customer
information from the report. Maintainers should acknowledge a complete report
within seven days and coordinate disclosure after a fix is available.
