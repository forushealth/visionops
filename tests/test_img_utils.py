from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from visionops.ImgUtils import (
    get_roi_bbox,
    image_grid,
    image_grid_from_pathlist,
    numpy_to_pil,
    remove_border,
)


def test_path_grid_uses_local_rng_and_caps_sample_count(tmp_path: Path) -> None:
    paths = []
    for index in range(2):
        path = tmp_path / f"image-{index}.png"
        Image.new("RGB", (4, 4), color=(index * 20, 0, 0)).save(path)
        paths.append(str(path))

    seed_function = random.seed
    figure = image_grid_from_pathlist(
        paths,
        count=25,
        random_seed=0,
        display_in_notebook=False,
    )

    assert random.seed is seed_function
    assert len(figure.axes) == 2


def test_roi_bbox_uses_one_path_for_dark_and_light_borders() -> None:
    dark_border = np.zeros((7, 7), dtype=np.uint8)
    dark_border[2:5, 2:5] = 255
    light_border = 255 - dark_border

    assert get_roi_bbox(dark_border, threshold=20, border_color="dark") == (1, 1, 5, 5)
    assert get_roi_bbox(light_border, threshold=20, border_color="light") == (1, 1, 5, 5)
    assert np.asarray(remove_border(dark_border, output_as_pil=False)).shape == (5, 5)


def test_image_helpers_reject_ambiguous_inputs() -> None:
    with pytest.raises(ValueError, match="Label length"):
        image_grid([np.zeros((2, 2), dtype=np.uint8)], labels=[])
    with pytest.raises(ValueError, match="3 or 4 channels"):
        numpy_to_pil(np.zeros((2, 2, 2), dtype=np.uint8))
