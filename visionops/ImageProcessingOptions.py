"""Config-driven image preprocessing/augmentation pipelines (Albumentations + OpenCV)."""

from __future__ import annotations

from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ._utils import as_list

try:
    import cv2  # type: ignore
except ModuleNotFoundError:
    cv2 = None

try:
    import albumentations as A  # type: ignore
except ModuleNotFoundError:
    A = None


_DEFAULT_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_DEFAULT_IMAGENET_STD = (0.229, 0.224, 0.225)


def _parse_step(step: Any) -> Tuple[str, Dict[str, Any]]:
    if isinstance(step, str):
        return step.strip().lower(), {}
    if isinstance(step, dict):
        if "name" in step:
            name = str(step["name"]).strip().lower()
            params = {k: v for k, v in step.items() if k != "name"}
            return name, params
        if len(step) == 1:
            key = next(iter(step.keys()))
            val = next(iter(step.values()))
            if isinstance(val, dict):
                return str(key).strip().lower(), dict(val)
            return str(key).strip().lower(), {}
    raise ValueError(f"Invalid transform step format: {step!r}")


def _default_p(section: str) -> float:
    if section in {"pre", "post"}:
        return 1.0
    return 0.5


def _with_default_p(params: Dict[str, Any], section: str) -> Dict[str, Any]:
    out = dict(params)
    if "p" not in out:
        out["p"] = _default_p(section)
    return out


def _ksize_from_params(params: Dict[str, Any], key: str = "ksize", default: int = 3) -> int:
    k = int(params.get(key, default))
    if k < 1:
        k = 1
    if k % 2 == 0:
        k += 1
    return k


def _opencv_gaussian_blur(image: np.ndarray, **kwargs) -> np.ndarray:
    if cv2 is None:
        return image
    k = _ksize_from_params(kwargs, "ksize", default=3)
    sigma = float(kwargs.get("sigma", 0.0))
    return cv2.GaussianBlur(image, (k, k), sigmaX=sigma)


def _opencv_median_blur(image: np.ndarray, **kwargs) -> np.ndarray:
    if cv2 is None:
        return image
    k = _ksize_from_params(kwargs, "ksize", default=3)
    return cv2.medianBlur(image, k)


def _opencv_bilateral_filter(image: np.ndarray, **kwargs) -> np.ndarray:
    if cv2 is None:
        return image
    d = int(kwargs.get("d", 5))
    sigma_color = float(kwargs.get("sigma_color", 75))
    sigma_space = float(kwargs.get("sigma_space", 75))
    return cv2.bilateralFilter(image, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)


def _opencv_equalize_y(image: np.ndarray, **kwargs) -> np.ndarray:
    if cv2 is None:
        return image
    if image.ndim != 3 or image.shape[2] != 3:
        return image
    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


def _per_image_standardize(image: np.ndarray, **kwargs) -> np.ndarray:
    img = image.astype(np.float32)
    mean = float(np.mean(img))
    std = float(np.std(img))
    if std < 1e-6:
        std = 1.0
    return (img - mean) / std


def _build_step(name: str, params: Dict[str, Any], section: str):
    if A is None:
        raise RuntimeError("albumentations is required to build image processing pipelines.")

    p = _with_default_p(params, section)
    n = name.lower().strip()

    # Deterministic preprocessing-friendly operations
    if n == "resize":
        return A.Resize(height=int(p.get("height", 224)), width=int(p.get("width", 224)), p=float(p["p"]))
    if n == "center_crop":
        return A.CenterCrop(height=int(p.get("height", 224)), width=int(p.get("width", 224)), p=float(p["p"]))
    if n == "pad_if_needed":
        return A.PadIfNeeded(min_height=int(p.get("min_height", 224)), min_width=int(p.get("min_width", 224)), p=float(p["p"]))
    if n == "clahe":
        return A.CLAHE(clip_limit=float(p.get("clip_limit", 4.0)), tile_grid_size=tuple(p.get("tile_grid_size", (8, 8))), p=float(p["p"]))
    if n == "equalize":
        return A.Equalize(p=float(p["p"]))
    if n == "sharpen":
        return A.Sharpen(alpha=tuple(p.get("alpha", (0.1, 0.3))), lightness=tuple(p.get("lightness", (0.5, 1.0))), p=float(p["p"]))
    if n == "unsharp_mask":
        return A.UnsharpMask(blur_limit=tuple(p.get("blur_limit", (3, 7))), sigma_limit=tuple(p.get("sigma_limit", (0.0, 0.0))), p=float(p["p"]))
    if n == "to_gray":
        return A.ToGray(p=float(p["p"]))

    # Augmentation
    if n == "horizontal_flip":
        return A.HorizontalFlip(p=float(p["p"]))
    if n == "vertical_flip":
        return A.VerticalFlip(p=float(p["p"]))
    if n == "random_rotate_90":
        return A.RandomRotate90(p=float(p["p"]))
    if n == "rotate":
        return A.Rotate(limit=int(p.get("limit", 30)), p=float(p["p"]))
    if n == "shift_scale_rotate":
        return A.ShiftScaleRotate(
            shift_limit=float(p.get("shift_limit", 0.0625)),
            scale_limit=float(p.get("scale_limit", 0.1)),
            rotate_limit=int(p.get("rotate_limit", 20)),
            p=float(p["p"]),
        )
    if n == "affine":
        return A.Affine(
            scale=p.get("scale", (0.9, 1.1)),
            translate_percent=p.get("translate_percent", (-0.1, 0.1)),
            rotate=p.get("rotate", (-20, 20)),
            p=float(p["p"]),
        )
    if n == "perspective":
        return A.Perspective(scale=tuple(p.get("scale", (0.05, 0.1))), p=float(p["p"]))
    if n == "elastic_transform":
        return A.ElasticTransform(alpha=float(p.get("alpha", 1.0)), sigma=float(p.get("sigma", 50.0)), alpha_affine=float(p.get("alpha_affine", 50.0)), p=float(p["p"]))
    if n == "grid_distortion":
        return A.GridDistortion(num_steps=int(p.get("num_steps", 5)), distort_limit=float(p.get("distort_limit", 0.3)), p=float(p["p"]))
    if n == "coarse_dropout":
        return A.CoarseDropout(
            max_holes=int(p.get("max_holes", 8)),
            max_height=int(p.get("max_height", 16)),
            max_width=int(p.get("max_width", 16)),
            p=float(p["p"]),
        )
    if n == "random_brightness_contrast":
        return A.RandomBrightnessContrast(
            brightness_limit=float(p.get("brightness_limit", 0.2)),
            contrast_limit=float(p.get("contrast_limit", 0.2)),
            p=float(p["p"]),
        )
    if n == "hue_saturation_value":
        return A.HueSaturationValue(
            hue_shift_limit=int(p.get("hue_shift_limit", 20)),
            sat_shift_limit=int(p.get("sat_shift_limit", 30)),
            val_shift_limit=int(p.get("val_shift_limit", 20)),
            p=float(p["p"]),
        )
    if n == "color_jitter":
        return A.ColorJitter(
            brightness=float(p.get("brightness", 0.2)),
            contrast=float(p.get("contrast", 0.2)),
            saturation=float(p.get("saturation", 0.2)),
            hue=float(p.get("hue", 0.1)),
            p=float(p["p"]),
        )
    if n == "random_gamma":
        return A.RandomGamma(gamma_limit=tuple(p.get("gamma_limit", (80, 120))), p=float(p["p"]))
    if n == "gauss_noise":
        return A.GaussNoise(var_limit=tuple(p.get("var_limit", (10.0, 50.0))), p=float(p["p"]))
    if n == "blur":
        return A.Blur(blur_limit=int(p.get("blur_limit", 3)), p=float(p["p"]))
    if n == "median_blur":
        return A.MedianBlur(blur_limit=int(p.get("blur_limit", 3)), p=float(p["p"]))
    if n == "motion_blur":
        return A.MotionBlur(blur_limit=int(p.get("blur_limit", 7)), p=float(p["p"]))
    if n == "channel_shuffle":
        return A.ChannelShuffle(p=float(p["p"]))
    if n == "solarize":
        return A.Solarize(threshold=int(p.get("threshold", 128)), p=float(p["p"]))
    if n == "posterize":
        return A.Posterize(num_bits=int(p.get("num_bits", 4)), p=float(p["p"]))

    # OpenCV-backed options
    if n == "opencv_gaussian_blur":
        return A.Lambda(image=partial(_opencv_gaussian_blur, **p), p=float(p["p"]))
    if n == "opencv_median_blur":
        return A.Lambda(image=partial(_opencv_median_blur, **p), p=float(p["p"]))
    if n == "opencv_bilateral_filter":
        return A.Lambda(image=partial(_opencv_bilateral_filter, **p), p=float(p["p"]))
    if n == "opencv_equalize_y":
        return A.Lambda(image=partial(_opencv_equalize_y, **p), p=float(p["p"]))

    # Post-processing
    if n == "normalize":
        mean = p.get("mean", _DEFAULT_IMAGENET_MEAN)
        std = p.get("std", _DEFAULT_IMAGENET_STD)
        max_pixel_value = float(p.get("max_pixel_value", 255.0))
        return A.Normalize(mean=mean, std=std, max_pixel_value=max_pixel_value, p=float(p["p"]))
    if n == "normalize_imagenet":
        return A.Normalize(mean=_DEFAULT_IMAGENET_MEAN, std=_DEFAULT_IMAGENET_STD, max_pixel_value=255.0, p=float(p["p"]))
    if n == "to_float":
        return A.ToFloat(max_value=float(p.get("max_value", 255.0)), p=float(p["p"]))
    if n == "per_image_standardize":
        return A.Lambda(image=partial(_per_image_standardize, **p), p=float(p["p"]))

    available = list_image_processing_options()
    all_names = sorted(set(available["preprocessing"] + available["augmentation"] + available["postprocessing"]))
    raise ValueError(f"Unknown image processing step '{name}'. Available: {all_names}")


def _build_section(steps: Any, section: str) -> Tuple[List[Any], List[str]]:
    out = []
    names = []
    for raw in as_list(steps):
        name, params = _parse_step(raw)
        out.append(_build_step(name, params, section=section))
        names.append(name)
    return out, names


def list_image_processing_options() -> Dict[str, List[str]]:
    """Return available config names for preprocessing/augmentation/postprocessing."""
    return {
        "preprocessing": sorted(
            [
                "resize",
                "center_crop",
                "pad_if_needed",
                "clahe",
                "equalize",
                "sharpen",
                "unsharp_mask",
                "to_gray",
                "opencv_gaussian_blur",
                "opencv_median_blur",
                "opencv_bilateral_filter",
                "opencv_equalize_y",
            ]
        ),
        "augmentation": sorted(
            [
                "horizontal_flip",
                "vertical_flip",
                "random_rotate_90",
                "rotate",
                "shift_scale_rotate",
                "affine",
                "perspective",
                "elastic_transform",
                "grid_distortion",
                "coarse_dropout",
                "random_brightness_contrast",
                "hue_saturation_value",
                "color_jitter",
                "random_gamma",
                "gauss_noise",
                "blur",
                "median_blur",
                "motion_blur",
                "channel_shuffle",
                "solarize",
                "posterize",
            ]
        ),
        "postprocessing": sorted(
            [
                "normalize",
                "normalize_imagenet",
                "to_float",
                "per_image_standardize",
            ]
        ),
    }


def build_image_processing_pipelines(
    tfms_config: Optional[Dict[str, Any]],
    *,
    image_size: Optional[Tuple[int, int, int]] = None,
) -> Dict[str, Any]:
    """Build train/eval Albumentations pipelines from config sections.

    Expected config:
      {
        "pre": [...],
        "aug": [...],
        "post": [...]
      }
    """
    sections = tfms_config if isinstance(tfms_config, dict) else {}
    pre_cfg = sections.get("pre", [])
    aug_cfg = sections.get("aug", [])
    post_cfg = sections.get("post", [])

    if A is None:
        return {
            "enabled": False,
            "reason": "albumentations_not_available",
            "train": None,
            "eval": None,
            "section_names": {"pre": [], "aug": [], "post": []},
            "post_has_normalization": False,
        }

    pre, pre_names = _build_section(pre_cfg, section="pre")
    aug, aug_names = _build_section(aug_cfg, section="aug")
    post, post_names = _build_section(post_cfg, section="post")

    if image_size and "resize" not in pre_names:
        h, w = int(image_size[0]), int(image_size[1])
        pre = [A.Resize(height=h, width=w, p=1.0)] + pre
        pre_names = ["resize"] + pre_names

    train_pipeline = A.Compose(pre + aug + post)
    eval_pipeline = A.Compose(pre + post)

    post_has_normalization = any(
        n in {"normalize", "normalize_imagenet", "per_image_standardize"} for n in post_names
    )

    return {
        "enabled": bool(pre or aug or post),
        "reason": "ok",
        "train": train_pipeline,
        "eval": eval_pipeline,
        "section_names": {"pre": pre_names, "aug": aug_names, "post": post_names},
        "post_has_normalization": post_has_normalization,
    }
