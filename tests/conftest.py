"""Shared pytest configuration for smoke tests."""

import matplotlib

# Force a non-interactive backend for CI/headless environments.
matplotlib.use("Agg")
