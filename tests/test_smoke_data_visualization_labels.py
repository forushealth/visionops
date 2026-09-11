"""Smoke tests for dataset visualization labels."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from visionops.DataUtils import visualise_dataset


def test_visualise_dataset_renders_label_titles() -> None:
    """Visualization should include readable labels in subplot titles."""
    images = np.random.rand(4, 24, 24, 3).astype(np.float32)
    labels = np.array([0, 1, 0, 1], dtype=np.int64)

    fig = visualise_dataset(
        images,
        image_number=4,
        labels=labels,
        class_names={0: "cat", 1: "dog"},
    )

    titles = [ax.get_title() for ax in fig.axes]
    assert any("Label: cat" in t for t in titles)
    assert any("Label: dog" in t for t in titles)
    plt.close(fig)


def test_visualise_dataset_single_image_keeps_2d_axes() -> None:
    """A single image should still render correctly with labels."""
    images = np.random.rand(1, 16, 16, 3).astype(np.float32)
    labels = np.array([1], dtype=np.int64)

    fig = visualise_dataset(images, image_number=1, labels=labels)
    assert len(fig.axes) == 2
    assert all("Label:" in ax.get_title() for ax in fig.axes)
    plt.close(fig)
