"""Torch-based CAM implementations and visualization helpers."""

from contextlib import contextmanager
from functools import wraps
import weakref

import torch
from torch import nn
import numpy as np
import cv2
import matplotlib.pyplot as plt


__all__ = [
    "GradCAM",
    "GradCAMPlusPlus",
    "XGradCAM",
    "ScoreCAM",
    "EigenCAM",
    "EigenGradCAM",
    "LayerCAM",
    "FullGrad",
    "cam_on_image",
    "visualise_CAM",
]


_UNSUPPORTED_CAM_METHODS = {
    "hirescam": "HiResCAM is unavailable because the bundled implementation duplicated GradCAM.",
    "ablationcam": "AblationCAM is unavailable because the bundled implementation did not perform model ablation.",
}


@contextmanager
def _preserve_model_state(model: torch.nn.Module):
    """Run a CAM operation in eval mode and restore exact module state afterward.

    Full backward hooks wrap module outputs in views. Temporarily disabling
    in-place layers prevents common models (for example, ResNet with in-place
    ReLUs) from modifying those views during a CAM backward pass.
    """
    training_states = [(module, bool(module.training)) for module in model.modules()]
    inplace_states = []
    for module in model.modules():
        inplace = getattr(module, "inplace", None)
        if isinstance(inplace, bool) and inplace:
            inplace_states.append((module, inplace))
            module.inplace = False

    model.eval()
    try:
        yield
    finally:
        for module, inplace in inplace_states:
            module.inplace = inplace
        # Assign directly so mixed train/eval state on child modules is retained.
        for module, training in training_states:
            module.training = training


def _preserve_cam_model_state(method):
    """Decorate direct CAM calls so they do not alter caller-owned model state."""
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with _preserve_model_state(self.model):
            # CAM gradients are required even when called from an evaluation
            # loop that has disabled autograd globally.
            with torch.inference_mode(False), torch.enable_grad():
                return method(self, *args, **kwargs)

    return wrapped


class _HookLifecycle:
    """Track removable PyTorch hook handles for one CAM helper."""

    def __init__(self) -> None:
        self._hook_handles = []

    def _track_hook(self, handle):
        self._hook_handles.append(handle)
        return handle

    def remove_hooks(self) -> None:
        """Remove every registered hook; safe to call more than once."""
        while self._hook_handles:
            handle = self._hook_handles.pop()
            try:
                handle.remove()
            except (AttributeError, RuntimeError):
                pass

        activations = getattr(self, "activations", None)
        gradients = getattr(self, "gradients", None)
        if isinstance(activations, dict):
            activations.clear()
        elif hasattr(self, "activations"):
            self.activations = None
        if isinstance(gradients, dict):
            gradients.clear()
        elif hasattr(self, "gradients"):
            self.gradients = None

    close = remove_hooks

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.remove_hooks()
        return False

    def __del__(self):
        try:
            self.remove_hooks()
        except Exception:
            # Destructors must never make interpreter shutdown noisy.
            pass


def _make_activation_gradient_hooks(owner, *, store_by_module: bool = False):
    """Create forward/backward hooks that capture activations and gradients."""
    owner_ref = weakref.ref(owner)

    def forward_hook(module, input, output):
        owner = owner_ref()
        if owner is None:
            return
        output = output.detach()
        if store_by_module:
            owner.activations[module] = output
        else:
            owner.activations = output

    def backward_hook(module, grad_in, grad_out):
        owner = owner_ref()
        if owner is None:
            return
        gradient = grad_out[0] if grad_out else None
        if gradient is not None:
            gradient = gradient.detach()
        if store_by_module:
            owner.gradients[module] = gradient
        else:
            owner.gradients = gradient

    return forward_hook, backward_hook


def _normalize_cam(cam: np.ndarray) -> np.ndarray:
    """Return a finite, non-negative CAM in the 0..1 range."""
    cam = np.nan_to_num(np.asarray(cam, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    cam = np.maximum(cam, 0.0)
    cam -= float(np.min(cam)) if cam.size else 0.0
    maximum = float(np.max(cam)) if cam.size else 0.0
    if maximum > 0.0:
        cam /= maximum
    return cam


def _resize_cam(cam: np.ndarray, input_image: torch.Tensor) -> np.ndarray:
    """Resize a CAM to the input tensor's spatial dimensions and normalize it."""
    height, width = int(input_image.shape[-2]), int(input_image.shape[-1])
    return _normalize_cam(cv2.resize(cam, (width, height)))


class GradCAM(_HookLifecycle):
    """
    Parameters
        *model: torch.nn.Sequential
         trained model
        *target_layer: torch.nn.Conv2d
         last convolution layer
    Return
        * GradCAM heatmap: np.array
    """
    def __init__(self, model: torch.nn.Sequential, target_layer:torch.nn.Conv2d):
        """Initialize GradCAM hooks for a model and target convolution layer."""
        super().__init__()
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        forward_hook, backward_hook = _make_activation_gradient_hooks(self)
        self._track_hook(self.target_layer.register_forward_hook(forward_hook))
        self._track_hook(self.target_layer.register_full_backward_hook(backward_hook))

    def _forward_and_capture_gradient(self, input_image, target_class):
        """Run one target-class gradient pass without changing parameter ``.grad`` values."""
        if not torch.is_floating_point(input_image):
            input_image = input_image.float()
        input_image = input_image.detach().clone().requires_grad_(True)
        self.gradients = None
        self.activations = None

        output = self.model(input_image)
        target = output[0, target_class]
        torch.autograd.grad(target, input_image, retain_graph=False, create_graph=False)

        if self.activations is None or self.gradients is None:
            raise RuntimeError(
                "CAM hooks did not capture activations and gradients. "
                "Ensure target_layer participates in the selected output."
            )
        return input_image

    @_preserve_cam_model_state
    def generate_cam(self, input_image:np.array, target_class:int):
        """
        Parameters
            * input_image: np.array
            * target_class: int
        Return
            * GradCAM heatmap: np.array
        """
        input_image = self._forward_and_capture_gradient(input_image, target_class)

        gradients = self.gradients[0].cpu().numpy()
        activations = self.activations[0].cpu().numpy()
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        return _resize_cam(cam, input_image)

class GradCAMPlusPlus(GradCAM):
    """
    Parameters
        *model: torch.nn.Sequential
         trained model
        *target_layer: torch.nn.Conv2d
         last convolution layer
    Return
        * GradCAM++ heatmap: np.array
    """
    def __init__(self,model: torch.nn.Sequential, target_layer:torch.nn.Conv2d):
        """Initialize GradCAM++ using GradCAM hook registration."""
        super().__init__(model=model, target_layer=target_layer)

    @_preserve_cam_model_state
    def generate_cam(self, input_image:np.array, target_class:int):
        """
        Parameters
            * input_image: np.array
            * target_class: int
        Return
            * GradCAM++ heatmap: np.array
        """
        input_image = self._forward_and_capture_gradient(input_image, target_class)

        gradients = self.gradients[0].cpu().numpy()
        activations = self.activations[0].cpu().numpy()
        weights = self._compute_weights(gradients)
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        return _resize_cam(cam, input_image)

    def _compute_weights(self, gradients):
        alpha = np.sum(gradients, axis=(1, 2))
        positive_gradients = np.maximum(gradients, 0.0)
        alpha_num = positive_gradients
        alpha_denom = np.sum(positive_gradients, axis=(1, 2), keepdims=True) + 1e-10
        alpha = alpha_num / alpha_denom

        weights = np.sum(alpha * np.maximum(gradients, 0.0), axis=(1, 2))
        return weights

class HiResCAM(GradCAM):
    """Legacy placeholder retained only to give existing imports a clear error."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(_UNSUPPORTED_CAM_METHODS["hirescam"])

class XGradCAM(GradCAM):
    """
    Parameters
        *model: torch.nn.Sequential
         trained model
        *target_layer: torch.nn.Conv2d
         last convolution layer
    Return
        * XGradCAM heatmap: np.array
    """
    @_preserve_cam_model_state
    def generate_cam(self, input_image:np.array, target_class:int):
        """
        Parameters
            * input_image: np.array
            * target_class: int
        Return
            * XGradCAM heatmap: np.array
        """
        input_image = self._forward_and_capture_gradient(input_image, target_class)

        gradients = self.gradients[0].cpu().numpy()
        activations = self.activations[0].cpu().numpy()

        # Normalize gradients and compute weights
        gradients_norm = np.linalg.norm(gradients, axis=(1, 2))
        weights = np.maximum(gradients, 0) / (gradients_norm[:, np.newaxis, np.newaxis] + 1e-10)

        cam = np.sum(weights * activations, axis=0)
        return _resize_cam(cam, input_image)

class AblationCAM(GradCAM):
    """Legacy placeholder retained only to give existing imports a clear error."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(_UNSUPPORTED_CAM_METHODS["ablationcam"])

class ScoreCAM(GradCAM):
    """
    Parameters
        *model: torch.nn.Sequential
         trained model
        *target_layer: torch.nn.Conv2d
         last convolution layer
    Return
        * GradCAM heatmap: np.array
    """
    def _register_hooks(self):
        forward_hook, _ = _make_activation_gradient_hooks(self)
        self._track_hook(self.target_layer.register_forward_hook(forward_hook))

    @_preserve_cam_model_state
    def generate_cam(self, input_image:np.array, target_class:int):
        """
        Parameters
            * input_image: np.array
            * target_class: int
        Return
            * ScoreCAM heatmap: np.array
        """
        with torch.no_grad():
            output = self.model(input_image)
            baseline = output[0, target_class].item()

        if self.activations is None:
            raise RuntimeError("CAM hook did not capture target-layer activations.")
        activations = self.activations[0].cpu().numpy()
        weights = []

        for i in range(activations.shape[0]):
            activation_map = activations[i]
            upsampled_map = _resize_cam(activation_map, input_image)
            mask = torch.as_tensor(
                upsampled_map,
                dtype=input_image.dtype,
                device=input_image.device,
            ).unsqueeze(0).unsqueeze(0)
            modified_input = input_image * mask
            with torch.no_grad():
                score = self.model(modified_input)[0, target_class].item()
            weights.append(score - baseline)

        weights = np.array(weights)
        cam = np.sum(weights[:, np.newaxis, np.newaxis] * activations, axis=0)
        return _resize_cam(cam, input_image)

class EigenCAM(_HookLifecycle):
    """
    Parameters
        *model: torch.nn.Sequential
         trained model
    Return
        * EigenCAM heatmap: np.array
    """
    def __init__(self, model:torch.nn.Sequential):
        """Initialize EigenCAM and register convolution hooks."""
        super().__init__()
        self.model = model
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        forward_hook, _ = _make_activation_gradient_hooks(self)
        target_layer = _infer_last_conv_layer(self.model)
        self._track_hook(target_layer.register_forward_hook(forward_hook))

    @_preserve_cam_model_state
    def generate_cam(self, input_image:np.array, target_class:int):
        """
        Parameters
            * input_image: np.array
            * target_class: int
        Return
            * EigenCAM heatmap: np.array
        """
        with torch.no_grad():
            self.model(input_image)
        if self.activations is None:
            raise RuntimeError("CAM hook did not capture convolution activations.")

        activations = self.activations[0].cpu().numpy()
        reshaped_activations = activations.reshape(activations.shape[0], -1)
        if activations.shape[0] == 1:
            principal_eigenvector = np.ones(1, dtype=np.float32)
        else:
            covariance_matrix = np.cov(reshaped_activations)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
            principal_eigenvector = np.real(eigenvectors[:, np.argmax(eigenvalues)])
        principal_eigenvector = principal_eigenvector.reshape(-1, 1, 1)
        cam = np.sum(principal_eigenvector * activations, axis=0)
        return _resize_cam(cam, input_image)

class EigenGradCAM(GradCAM):
    """
    Parameters
        *model: torch.nn.Sequential
         trained model
        *target_layer: torch.nn.Conv2d
         last convolution layer
    Return
        * EigenGradCAM heatmap: np.array
    """
    @_preserve_cam_model_state
    def generate_cam(self, input_image:np.array, target_class:int):
        """
        Parameters
            * input_image: np.array
            * target_class: int
        Return
            * EigenGradCAM heatmap: np.array
        """
        input_image = self._forward_and_capture_gradient(input_image, target_class)

        gradients = self.gradients[0].cpu().numpy()
        activations = self.activations[0].cpu().numpy()

        cam = np.sum(gradients * activations, axis=0)
        return _resize_cam(cam, input_image)

class LayerCAM(GradCAM):
    """
    Parameters
        *model: torch.nn.Sequential
         trained model
        *target_layer: torch.nn.Conv2d
         last convolution layer
    Return
        * LayerCAM heatmap: np.array
    """
    @_preserve_cam_model_state
    def generate_cam(self, input_image:np.array, target_class:int):
        """
        Parameters
            * input_image: np.array
            * target_class: int
        Return
            * LayerCAM heatmap: np.array
        """
        input_image = self._forward_and_capture_gradient(input_image, target_class)

        gradients = self.gradients[0].cpu().numpy()
        activations = self.activations[0].cpu().numpy()

        cam = np.maximum(gradients, 0) * activations
        cam = np.sum(cam, axis=0)
        return _resize_cam(cam, input_image)

class FullGrad(_HookLifecycle):
    """
    Parameters
        *model: torch.nn.Sequential
         trained model
    Return
        * FullGrad heatmap: np.array
    """
    def __init__(self,model:torch.nn.Sequential):
        """Initialize FullGrad state and layer hooks."""
        super().__init__()
        self.model = model
        self.activations = {}
        self.gradients = {}

        # Register hooks to capture activations and gradients
        self._register_hooks()

    def _register_hooks(self):
        forward_hook, backward_hook = _make_activation_gradient_hooks(
            self,
            store_by_module=True,
        )
        for _name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                self._track_hook(module.register_forward_hook(forward_hook))
                self._track_hook(module.register_full_backward_hook(backward_hook))

    @_preserve_cam_model_state
    def generate_cam(self, input_image:np.array, target_class:int):
        """
        Parameters
            * input_image: np.array
            * target_class: int
        Return
            * FullGrad heatmap: np.array
        """
        if not torch.is_floating_point(input_image):
            input_image = input_image.float()
        input_image = input_image.detach().clone().requires_grad_(True)
        self.activations.clear()
        self.gradients.clear()
        output = self.model(input_image)
        target = output[0, target_class]
        torch.autograd.grad(target, input_image, retain_graph=False, create_graph=False)
        cam = np.zeros(input_image.shape[2:], dtype=np.float32)

        # Add contribution from all convolutional and linear layers
        for module, activation_tensor in self.activations.items():
            gradient_tensor = self.gradients.get(module)
            if gradient_tensor is None:
                continue
            if isinstance(module, nn.Conv2d):
                activations = activation_tensor[0].cpu().numpy()
                gradients = gradient_tensor[0].cpu().numpy()

                weights = np.sum(gradients, axis=(1, 2), keepdims=True)
                activations_resized = np.array([cv2.resize(activation, (input_image.shape[3], input_image.shape[2])) for activation in activations])
                cam += np.sum(weights * activations_resized, axis=0)

            elif isinstance(module, nn.Linear):
                gradients = gradient_tensor.cpu().numpy()
                cam += float(np.sum(gradients))

        return _normalize_cam(cam)

def _infer_last_conv_layer(model: torch.nn.Module):
    target = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            target = module
    if target is None:
        raise ValueError("Could not infer target layer: no Conv2d layer found in model.")
    return target


def _resolve_cam_class(cam_name: str):
    cam_key = str(cam_name).replace("_", "").lower()
    if cam_key in _UNSUPPORTED_CAM_METHODS:
        raise ValueError(_UNSUPPORTED_CAM_METHODS[cam_key])

    cam_classes = {
        "gradcam": GradCAM,
        "gradcamplusplus": GradCAMPlusPlus,
        "gradcamelementwise": LayerCAM,
        "xgradcam": XGradCAM,
        "scorecam": ScoreCAM,
        "eigencam": EigenCAM,
        "eigengradcam": EigenGradCAM,
        "layercam": LayerCAM,
        "fullgrad": FullGrad,
    }
    try:
        return cam_key, cam_classes[cam_key]
    except KeyError as exc:
        raise ValueError(
            f"Unknown CAM method '{cam_name}'. Supported methods: "
            f"{', '.join(sorted(cam_classes))}."
        ) from exc


def _load_model_artifact(model_name: str, *, trusted_model: bool) -> nn.Module:
    """Load a serialized model without silently enabling pickle execution."""
    try:
        model = torch.load(str(model_name), map_location="cpu", weights_only=True)
    except TypeError as exc:
        if not trusted_model:
            raise RuntimeError(
                "This PyTorch version cannot perform a weights-only load. "
                "Upgrade PyTorch or set trusted_model=True only for an artifact you trust."
            ) from exc
        model = torch.load(str(model_name), map_location="cpu")
    except Exception as exc:
        if not trusted_model:
            raise RuntimeError(
                "The artifact was rejected by torch.load(weights_only=True). "
                "Set trusted_model=True only when the file and its creator are trusted."
            ) from exc
        model = torch.load(str(model_name), map_location="cpu", weights_only=False)

    if not isinstance(model, nn.Module):
        raise TypeError(
            "cam_on_image requires an artifact containing a torch.nn.Module; "
            "a weights/state-dict artifact needs its model architecture instantiated first."
        )
    return model


def cam_on_image(
    image: np.array,
    model_name: str,
    cam_name: str = "GradCAM",
    target_layer: torch.nn.Module = None,
    target_class: int = 0,
    device: str = None,
    trusted_model: bool = False,
) -> np.array:
    """
    Parameters
        * image: np.array
        * model_name:str
        * cam_name:str
        * trusted_model: bool
          allow pickle-based model loading only for a trusted artifact
    Return
        * cam superimposed image array
    """

    cam_key, cam_class = _resolve_cam_class(cam_name)
    model = _load_model_artifact(model_name, trusted_model=trusted_model)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    needs_target = cam_key not in {"fullgrad", "eigencam"}
    if needs_target and target_layer is None:
        target_layer = _infer_last_conv_layer(model)

    if needs_target:
        grad_cam = cam_class(model, target_layer)
    else:
        grad_cam = cam_class(model)

    img = np.array(image)
    if img.ndim != 3:
        raise ValueError("image must be HxWxC")
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device).float()
    try:
        cam = grad_cam.generate_cam(img_tensor, target_class=target_class)
    finally:
        grad_cam.remove_hooks()

    if cam_key == "fullgrad":
        return cam

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    base = img
    if base.dtype != np.uint8:
        if float(np.max(base)) <= 1.0:
            base = np.uint8(np.clip(base * 255.0, 0, 255))
        else:
            base = np.uint8(np.clip(base, 0, 255))
    cams = cv2.addWeighted(base, 1.0, heatmap, 0.4, 0)
    return cams


def visualise_CAM(
    dataset_array: np.ndarray,
    model_name: str,
    cam_names: list,
    image_number: int = 8,
    target_layer: torch.nn.Module = None,
    target_class: int = 0,
    device: str = None,
    trusted_model: bool = False,
) -> plt.subplot:
    """
    Parameters
        * dataset_array: np.ndarray
          dataset array to visualise the images.
        * image_number: int
          choose the number of the images to display.
    Return
        * a grid of images.
    """
    if image_number <= 0:
        raise ValueError("image_number must be >= 1")
    if not cam_names:
        raise ValueError("cam_names must contain at least one CAM method.")

    n = min(image_number, len(dataset_array))
    rows = 1 + len(cam_names)
    plt.figure(figsize=(n * 2.2, rows * 2.2), dpi=100)

    for i in range(n):
        plt.subplot(rows, n, i + 1)
        plt.imshow(dataset_array[i])
        if i == 0:
            plt.title("Original")
        plt.axis("off")

    for row_idx, cam_name in enumerate(cam_names, start=1):
        cam_imgs = [
            cam_on_image(
                image=dataset_array[i],
                model_name=model_name,
                cam_name=cam_name,
                target_layer=target_layer,
                target_class=target_class,
                device=device,
                trusted_model=trusted_model,
            )
            for i in range(n)
        ]
        for i in range(n):
            plt.subplot(rows, n, row_idx * n + i + 1)
            plt.imshow(cam_imgs[i])
            if i == 0:
                plt.title(str(cam_name))
            plt.axis("off")

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    plt.show()
