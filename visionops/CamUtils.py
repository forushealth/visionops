"""Unified CAM utilities for torch and keras backends."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


def _normalize_heatmap(hm: np.ndarray) -> np.ndarray:
    hm = np.nan_to_num(hm.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    hm = np.maximum(hm, 0.0)
    m = float(hm.max()) if hm.size else 0.0
    if m > 0:
        hm = hm / m
    return hm


def _resize_heatmap(hm: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    hm = np.array(hm, dtype=np.float32)
    if hm.ndim != 2:
        raise ValueError(f"CAM heatmap must be 2D, got shape {hm.shape}")
    if hm.shape == (out_h, out_w):
        return _normalize_heatmap(hm)

    try:
        import cv2

        resized = cv2.resize(hm, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        return _normalize_heatmap(resized)
    except Exception:
        # Fallback to simple nearest-neighbor indexing when cv2 is unavailable.
        y_idx = np.linspace(0, hm.shape[0] - 1, out_h).astype(int)
        x_idx = np.linspace(0, hm.shape[1] - 1, out_w).astype(int)
        return _normalize_heatmap(hm[np.ix_(y_idx, x_idx)])


def _overlay_heatmap_on_image(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    import matplotlib.cm as cm

    img = np.array(image, dtype=np.float32)
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    if img.ndim == 3 and img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=2)
    if img.ndim == 3 and img.shape[-1] > 3:
        img = img[:, :, :3]
    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(f"Expected image shape HxWx3 for overlay, got {img.shape}")

    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    img_min = float(np.min(img))
    img_max = float(np.max(img))
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    else:
        img = np.zeros_like(img)

    hm = _normalize_heatmap(heatmap)
    hm_rgb = cm.get_cmap("jet")(hm)[..., :3].astype(np.float32)
    return np.clip((1.0 - alpha) * img + alpha * hm_rgb, 0.0, 1.0)


def _resolve_torch_layer(model: Any, target_layer: Any) -> Any:
    if not isinstance(target_layer, str):
        return target_layer

    layer = model
    for token in target_layer.split("."):
        if token.isdigit():
            layer = layer[int(token)]
        else:
            layer = getattr(layer, token)
    return layer


def _infer_torch_target_layer(model: Any) -> Any:
    """Pick the last convolution layer from a torch model."""
    import torch.nn as nn

    candidates = []
    for _name, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            candidates.append(module)
    if not candidates:
        raise ValueError("Could not infer target_layer for torch model (no Conv layers found).")
    return candidates[-1]


def _infer_keras_target_layer(model: Any) -> str:
    """Pick the last convolution-like layer name from a keras model."""
    import tensorflow as tf

    conv_types = (
        tf.keras.layers.Conv1D,
        tf.keras.layers.Conv2D,
        tf.keras.layers.Conv3D,
        tf.keras.layers.DepthwiseConv2D,
        tf.keras.layers.SeparableConv1D,
        tf.keras.layers.SeparableConv2D,
    )

    candidates = []
    for layer in model.layers:
        if isinstance(layer, conv_types):
            candidates.append(layer.name)
    if not candidates:
        raise ValueError("Could not infer target_layer for keras model (no Conv layers found).")
    return candidates[-1]


def _pca_first_component(feature_map: np.ndarray) -> np.ndarray:
    """feature_map: H x W x C -> returns H x W."""
    h, w, c = feature_map.shape
    x = feature_map.reshape(h * w, c)
    x = x - x.mean(axis=0, keepdims=True)

    # covariance in channel space
    cov = np.dot(x.T, x) / max(1, (x.shape[0] - 1))
    eigvals, eigvecs = np.linalg.eigh(cov)
    pc1 = eigvecs[:, np.argmax(eigvals)]
    proj = np.dot(x, pc1).reshape(h, w)
    return _normalize_heatmap(proj)


def _keras_prepare_image(image: np.ndarray) -> np.ndarray:
    image = np.array(image)
    if image.ndim == 3:
        image = np.expand_dims(image, axis=0)
    if image.ndim != 4:
        raise ValueError(f"Keras CAM expects image rank 3/4, got shape {image.shape}")
    return image.astype(np.float32)


def _keras_select_class_score(preds, class_index: Optional[int] = None):
    import tensorflow as tf

    if len(preds.shape) == 1:
        return preds

    if preds.shape[-1] == 1:
        return preds[:, 0]

    idx = int(tf.argmax(preds[0])) if class_index is None else int(class_index)
    return preds[:, idx]


def _keras_forward_with_grads(model: Any, image: np.ndarray, target_layer: str, class_index: Optional[int]):
    import tensorflow as tf

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(target_layer).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_output, preds = grad_model(image)
        class_score = _keras_select_class_score(preds, class_index)

    grads = tape.gradient(class_score, conv_output)
    return conv_output, grads, preds


def _keras_gradcam(model: Any, image: np.ndarray, target_layer: str, class_index: Optional[int] = None) -> np.ndarray:
    import tensorflow as tf

    image = _keras_prepare_image(image)
    conv_output, grads, _ = _keras_forward_with_grads(model, image, target_layer, class_index)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_map = conv_output[0]
    heatmap = tf.reduce_sum(conv_map * pooled_grads, axis=-1)
    return _normalize_heatmap(heatmap.numpy())


def _keras_gradcam_pp(model: Any, image: np.ndarray, target_layer: str, class_index: Optional[int] = None) -> np.ndarray:
    import tensorflow as tf

    image = _keras_prepare_image(image)
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(target_layer).output, model.output])

    with tf.GradientTape(persistent=True) as tape1:
        with tf.GradientTape(persistent=True) as tape2:
            conv_output, preds = grad_model(image)
            score = _keras_select_class_score(preds, class_index)
        grads = tape2.gradient(score, conv_output)
    grads2 = tape1.gradient(grads, conv_output)
    grads3 = tape1.gradient(grads2, conv_output)
    del tape1
    del tape2

    conv = conv_output[0].numpy()
    g = grads[0].numpy()
    g2 = grads2[0].numpy() if grads2 is not None else np.zeros_like(g)
    g3 = grads3[0].numpy() if grads3 is not None else np.zeros_like(g)

    sum_conv = np.sum(conv, axis=(0, 1), keepdims=True)
    eps = 1e-8
    alpha_num = g2
    alpha_denom = 2.0 * g2 + g3 * sum_conv
    alpha_denom = np.where(np.abs(alpha_denom) < eps, eps, alpha_denom)
    alpha = alpha_num / alpha_denom

    weights = np.sum(np.maximum(g, 0.0) * alpha, axis=(0, 1))
    heatmap = np.sum(conv * weights.reshape(1, 1, -1), axis=-1)
    return _normalize_heatmap(heatmap)


def _keras_xgradcam(model: Any, image: np.ndarray, target_layer: str, class_index: Optional[int] = None) -> np.ndarray:
    image = _keras_prepare_image(image)
    conv_output, grads, _ = _keras_forward_with_grads(model, image, target_layer, class_index)

    conv = conv_output[0].numpy()
    g = grads[0].numpy()

    eps = 1e-8
    num = np.sum(g * conv, axis=(0, 1))
    den = np.sum(conv, axis=(0, 1)) + eps
    weights = num / den

    heatmap = np.sum(conv * weights.reshape(1, 1, -1), axis=-1)
    return _normalize_heatmap(heatmap)


def _keras_layercam(model: Any, image: np.ndarray, target_layer: str, class_index: Optional[int] = None) -> np.ndarray:
    image = _keras_prepare_image(image)
    conv_output, grads, _ = _keras_forward_with_grads(model, image, target_layer, class_index)

    conv = conv_output[0].numpy()
    g = grads[0].numpy()

    heatmap = np.sum(np.maximum(g, 0.0) * conv, axis=-1)
    return _normalize_heatmap(heatmap)


def _keras_hirescam(model: Any, image: np.ndarray, target_layer: str, class_index: Optional[int] = None) -> np.ndarray:
    image = _keras_prepare_image(image)
    conv_output, grads, _ = _keras_forward_with_grads(model, image, target_layer, class_index)

    conv = conv_output[0].numpy()
    g = grads[0].numpy()

    heatmap = np.sum(g * conv, axis=-1)
    return _normalize_heatmap(heatmap)


def _keras_eigencam(model: Any, image: np.ndarray, target_layer: str, class_index: Optional[int] = None) -> np.ndarray:
    import tensorflow as tf

    image = _keras_prepare_image(image)
    feat_model = tf.keras.models.Model([model.inputs], [model.get_layer(target_layer).output])
    conv_output = feat_model(image)[0].numpy()
    return _pca_first_component(conv_output)


def _keras_eigengradcam(model: Any, image: np.ndarray, target_layer: str, class_index: Optional[int] = None) -> np.ndarray:
    image = _keras_prepare_image(image)
    conv_output, grads, _ = _keras_forward_with_grads(model, image, target_layer, class_index)

    conv = conv_output[0].numpy()
    g = grads[0].numpy()
    return _pca_first_component(conv * g)


def _keras_scorecam(
    model: Any,
    image: np.ndarray,
    target_layer: str,
    class_index: Optional[int] = None,
    max_channels: int = 32,
) -> np.ndarray:
    import tensorflow as tf

    image = _keras_prepare_image(image)
    h_in, w_in = int(image.shape[1]), int(image.shape[2])

    feat_model = tf.keras.models.Model([model.inputs], [model.get_layer(target_layer).output, model.output])
    conv_output, preds = feat_model(image)

    conv = conv_output[0].numpy()  # H x W x C
    c = conv.shape[-1]

    # select most active channels for efficiency
    channel_scores = np.mean(np.maximum(conv, 0.0), axis=(0, 1))
    idx = np.argsort(channel_scores)[::-1][: max(1, min(max_channels, c))]

    weights = []
    maps = []

    for ch in idx:
        m = conv[:, :, ch]
        m = _normalize_heatmap(m)
        m_tf = tf.convert_to_tensor(m[None, :, :, None], dtype=tf.float32)
        m_up = tf.image.resize(m_tf, (h_in, w_in), method="bilinear").numpy()[0, :, :, 0]

        masked = image.copy()
        masked[0] = masked[0] * m_up[:, :, None]

        pred = model(masked, training=False)
        if hasattr(pred, "numpy"):
            pred_np = pred.numpy()
        else:
            pred_np = np.array(pred)

        if pred_np.ndim == 1:
            score = float(pred_np[0])
        elif pred_np.shape[-1] == 1:
            score = float(pred_np[0, 0])
        else:
            cls = int(np.argmax(pred_np[0])) if class_index is None else int(class_index)
            score = float(pred_np[0, cls])

        weights.append(score)
        maps.append(m)

    weights = np.array(weights, dtype=np.float32)
    if weights.size == 0:
        return np.zeros(conv.shape[:2], dtype=np.float32)

    # softmax normalize weights
    e = np.exp(weights - np.max(weights))
    w = e / (np.sum(e) + 1e-8)

    heatmap = np.zeros(conv.shape[:2], dtype=np.float32)
    for wi, mi in zip(w, maps, strict=True):
        heatmap += wi * mi

    return _normalize_heatmap(heatmap)


def _keras_ablationcam(
    model: Any,
    image: np.ndarray,
    target_layer: str,
    class_index: Optional[int] = None,
    max_channels: int = 32,
) -> np.ndarray:
    """Approximate AblationCAM for Keras using input-space ablation masks."""
    import tensorflow as tf

    image = _keras_prepare_image(image)
    h_in, w_in = int(image.shape[1]), int(image.shape[2])

    feat_model = tf.keras.models.Model([model.inputs], [model.get_layer(target_layer).output, model.output])
    conv_output, preds = feat_model(image)
    conv = conv_output[0].numpy()  # H x W x C
    c = conv.shape[-1]

    # Baseline score for target class
    pred_np = preds.numpy() if hasattr(preds, "numpy") else np.array(preds)
    if pred_np.ndim == 1:
        baseline = float(pred_np[0])
    elif pred_np.shape[-1] == 1:
        baseline = float(pred_np[0, 0])
    else:
        cls = int(np.argmax(pred_np[0])) if class_index is None else int(class_index)
        baseline = float(pred_np[0, cls])

    # Select most active channels for efficiency
    channel_scores = np.mean(np.maximum(conv, 0.0), axis=(0, 1))
    idx = np.argsort(channel_scores)[::-1][: max(1, min(max_channels, c))]

    weights = []
    maps = []

    for ch in idx:
        m = conv[:, :, ch]
        m = _normalize_heatmap(m)
        m_tf = tf.convert_to_tensor(m[None, :, :, None], dtype=tf.float32)
        m_up = tf.image.resize(m_tf, (h_in, w_in), method="bilinear").numpy()[0, :, :, 0]

        ablated = image.copy()
        # Suppress high-activation regions for this channel.
        ablated[0] = ablated[0] * (1.0 - m_up[:, :, None])

        pred = model(ablated, training=False)
        pred_np2 = pred.numpy() if hasattr(pred, "numpy") else np.array(pred)
        if pred_np2.ndim == 1:
            score = float(pred_np2[0])
        elif pred_np2.shape[-1] == 1:
            score = float(pred_np2[0, 0])
        else:
            cls = int(np.argmax(pred_np2[0])) if class_index is None else int(class_index)
            score = float(pred_np2[0, cls])

        # Higher drop => more important map
        weights.append(max(0.0, baseline - score))
        maps.append(m)

    weights = np.array(weights, dtype=np.float32)
    if weights.size == 0:
        return np.zeros(conv.shape[:2], dtype=np.float32)

    if float(weights.sum()) <= 1e-8:
        weights = np.ones_like(weights) / float(len(weights))
    else:
        weights = weights / float(weights.sum())

    heatmap = np.zeros(conv.shape[:2], dtype=np.float32)
    for wi, mi in zip(weights, maps, strict=True):
        heatmap += wi * mi

    return _normalize_heatmap(heatmap)


def _generate_keras_cam(
    model: Any,
    image: Any,
    target_layer: Optional[str],
    class_index: Optional[int] = None,
    cam_method: str = "gradcam",
) -> np.ndarray:
    if target_layer is None:
        target_layer = _infer_keras_target_layer(model)

    method = cam_method.lower().replace("_", "")

    method_map = {
        "gradcam": _keras_gradcam,
        "gradcamplusplus": _keras_gradcam_pp,
        "xgradcam": _keras_xgradcam,
        "layercam": _keras_layercam,
        "hirescam": _keras_hirescam,
        "eigencam": _keras_eigencam,
        "eigengradcam": _keras_eigengradcam,
        "gradcamelementwise": _keras_layercam,
        "scorecam": _keras_scorecam,
        "ablationcam": _keras_ablationcam,
        # FullGrad has no direct Keras equivalent here; HiResCAM is the closest gradient*activation variant.
        "fullgrad": _keras_hirescam,
    }

    if method not in method_map:
        raise ValueError(
            f"Unknown keras cam_method '{cam_method}'. Supported: {sorted(method_map.keys())}"
        )

    return method_map[method](model=model, image=np.array(image), target_layer=target_layer, class_index=class_index)


def _generate_torch_cam(
    model: Any,
    image: Any,
    target_layer: Any,
    class_index: Optional[int] = None,
    cam_method: str = "gradcam",
) -> np.ndarray:
    import torch
    from .TorchCAMUtils import _preserve_model_state, _resolve_cam_class

    method, cam_class = _resolve_cam_class(cam_method)
    needs_target = method not in {"eigencam", "fullgrad"}
    if needs_target:
        if target_layer is None:
            target = _infer_torch_target_layer(model)
        else:
            target = _resolve_torch_layer(model, target_layer)

    x = image
    if not isinstance(x, torch.Tensor):
        x_np = np.array(image)
        if x_np.ndim == 3 and x_np.shape[-1] in {1, 3}:
            x_np = np.transpose(x_np, (2, 0, 1))
        x = torch.tensor(x_np, dtype=torch.float32)
    else:
        x = x.float()
    if x.ndim == 3:
        x = x.unsqueeze(0)
    device = next(model.parameters()).device
    x = x.to(device)

    with _preserve_model_state(model):
        with torch.no_grad():
            logits = model(x)
        if class_index is None:
            class_index = int(torch.argmax(logits[0]).item())

        if needs_target:
            cam_generator = cam_class(model=model, target_layer=target)
        else:
            cam_generator = cam_class(model=model)
        try:
            return _normalize_heatmap(cam_generator.generate_cam(x, class_index))
        finally:
            cam_generator.remove_hooks()


def generate_cam(
    model: Any,
    image: Any,
    framework: str,
    target_layer: Any = None,
    class_index: Optional[int] = None,
    cam_method: str = "gradcam",
) -> np.ndarray:
    """Generate a single CAM heatmap for torch or keras.

    If `target_layer` is None, the module auto-selects the last convolution layer.
    """
    framework = framework.lower().strip()

    if framework == "keras":
        if target_layer is not None and not isinstance(target_layer, str):
            raise ValueError("For keras, target_layer must be a layer name string or None")
        return _generate_keras_cam(
            model=model,
            image=image,
            target_layer=target_layer,
            class_index=class_index,
            cam_method=cam_method,
        )

    if framework == "torch":
        return _generate_torch_cam(
            model=model,
            image=image,
            target_layer=target_layer,
            class_index=class_index,
            cam_method=cam_method,
        )

    raise ValueError("framework must be 'torch' or 'keras'")


def _to_display_image(image: Any, framework: str) -> np.ndarray:
    img = image
    if framework == "torch":
        import torch

        if isinstance(img, torch.Tensor):
            x = img.detach().cpu().float().numpy()
        else:
            x = np.array(img)

        if x.ndim == 4:
            x = x[0]
        if x.ndim == 3 and x.shape[0] in {1, 3}:
            x = np.transpose(x, (1, 2, 0))
        if x.ndim == 3 and x.shape[2] == 1:
            x = x[:, :, 0]
        return x

    x = np.array(img)
    if x.ndim == 4:
        x = x[0]
    return x


def compare_cams(
    model: Any,
    image: Any,
    framework: str,
    target_layer: Any = None,
    cam_methods: Sequence[str] = ("gradcam", "gradcamplusplus", "xgradcam", "layercam"),
    class_index: Optional[int] = None,
    figsize: Tuple[int, int] = (12, 8),
    overlay: bool = False,
    alpha: float = 0.45,
    show_plot: bool = True,
):
    """Generate and optionally plot multiple CAMs for comparison.

    Returns:
        cams_dict, fig
    """
    import matplotlib.pyplot as plt

    framework = framework.lower().strip()
    cams: Dict[str, np.ndarray] = {}

    for method in cam_methods:
        cams[method] = generate_cam(
            model=model,
            image=image,
            framework=framework,
            target_layer=target_layer,
            class_index=class_index,
            cam_method=method,
        )

    disp_img = _to_display_image(image, framework)

    resized_cams: Dict[str, np.ndarray] = {}
    if disp_img.ndim == 2:
        out_h, out_w = int(disp_img.shape[0]), int(disp_img.shape[1])
    else:
        out_h, out_w = int(disp_img.shape[0]), int(disp_img.shape[1])

    for method, hm in cams.items():
        resized_cams[method] = _resize_heatmap(hm, out_h=out_h, out_w=out_w)

    n = len(cam_methods) + 1
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    fig = plt.figure(figsize=figsize)

    ax = fig.add_subplot(rows, cols, 1)
    if disp_img.ndim == 2:
        ax.imshow(disp_img, cmap="gray")
    else:
        ax.imshow(disp_img)
    ax.set_title("Original")
    ax.axis("off")

    for i, method in enumerate(cam_methods, start=2):
        hm = resized_cams[method]
        ax = fig.add_subplot(rows, cols, i)
        if overlay:
            blended = _overlay_heatmap_on_image(disp_img, hm, alpha=alpha)
            ax.imshow(blended)
        else:
            ax.imshow(hm, cmap="jet")
        ax.set_title(method)
        ax.axis("off")

    plt.tight_layout()
    if show_plot:
        plt.show()

    return resized_cams, fig


def visualise_CAM(
    dataset_array: Any,
    model: Any,
    framework: str,
    target_layer: Any = None,
    cam_names: Sequence[str] = ("GradCAM", "EigenCAM"),
    class_index: Optional[int] = None,
    image_number: int = 8,
    overlay: bool = True,
    alpha: float = 0.45,
    figsize: Optional[Tuple[int, int]] = None,
    show_plot: bool = True,
):
    """Torch-style CAM grid visualisation for both torch and keras backends.

    Args:
        dataset_array: image batch/sequence (N,H,W,C) or sequence of images.
        model: loaded torch/keras model object.
        framework: "torch" or "keras".
        target_layer: layer reference (torch module/name path or keras layer name).
        cam_names: list of CAM methods to compare; each method forms one row.
        class_index: optional target class index.
        image_number: number of samples to visualise from dataset_array.
        overlay: when True, show CAM overlay on input image.
        alpha: overlay blending factor.
    Returns:
        results, fig where results maps cam_name -> list[heatmap]
    """
    import matplotlib.pyplot as plt

    if image_number <= 0:
        raise ValueError("image_number must be >= 1")
    if not cam_names:
        raise ValueError("cam_names must contain at least one method")

    data = dataset_array
    if framework.lower().strip() == "torch":
        import torch

        if isinstance(data, torch.Tensor):
            if data.ndim == 3:
                data = data.unsqueeze(0)
            items = [data[i] for i in range(min(image_number, int(data.shape[0])))]
        else:
            arr = np.array(data)
            if arr.ndim == 3:
                arr = np.expand_dims(arr, axis=0)
            items = [arr[i] for i in range(min(image_number, int(arr.shape[0])))]
    else:
        arr = np.array(data)
        if arr.ndim == 3:
            arr = np.expand_dims(arr, axis=0)
        items = [arr[i] for i in range(min(image_number, int(arr.shape[0])))]

    if not items:
        raise ValueError("dataset_array does not contain any image samples")

    methods = list(cam_names)
    results: Dict[str, list] = {m: [] for m in methods}

    disp_images = []
    for item in items:
        disp_img = _to_display_image(item, framework.lower().strip())
        if disp_img.ndim == 2:
            out_h, out_w = int(disp_img.shape[0]), int(disp_img.shape[1])
        else:
            out_h, out_w = int(disp_img.shape[0]), int(disp_img.shape[1])

        disp_images.append(disp_img)
        for method in methods:
            hm = generate_cam(
                model=model,
                image=item,
                framework=framework,
                target_layer=target_layer,
                class_index=class_index,
                cam_method=method,
            )
            hm = _resize_heatmap(hm, out_h=out_h, out_w=out_w)
            results[method].append(hm)

    rows = 1 + len(methods)
    cols = len(items)
    if figsize is None:
        figsize = (max(2 * cols, 8), max(2 * rows, 6))
    fig = plt.figure(figsize=figsize)

    for c, img in enumerate(disp_images, start=1):
        ax = fig.add_subplot(rows, cols, c)
        if img.ndim == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(img)
        if c == 1:
            ax.set_ylabel("Original")
        ax.axis("off")

    for r, method in enumerate(methods, start=2):
        for c, img in enumerate(disp_images, start=1):
            hm = results[method][c - 1]
            ax = fig.add_subplot(rows, cols, (r - 1) * cols + c)
            if overlay:
                blended = _overlay_heatmap_on_image(img, hm, alpha=alpha)
                ax.imshow(blended)
            else:
                ax.imshow(hm, cmap="jet")
            if c == 1:
                ax.set_ylabel(str(method))
            ax.axis("off")

    plt.tight_layout()
    if show_plot:
        plt.show()
    return results, fig
