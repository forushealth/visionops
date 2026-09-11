"""Focused regression tests for Torch CAM lifecycle and loading safety."""

import gc
import pickle
import weakref

import numpy as np
import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("cv2")

from torch import nn

from visionops.CamUtils import generate_cam
from visionops.TorchCAMUtils import GradCAM, cam_on_image


class _TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(4)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(4, 2)

    def forward(self, inputs):
        features = self.relu(self.norm(self.conv(inputs)))
        return self.classifier(self.pool(features).flatten(1))


def _training_states(model):
    return [module.training for module in model.modules()]


def _hook_counts(module):
    return len(module._forward_hooks), len(module._backward_hooks)


def _model_hook_counts(model):
    return [_hook_counts(module) for module in model.modules()]


def test_generate_cam_restores_model_state_gradients_and_hooks():
    model = _TinyClassifier()
    model.train()
    model.pool.eval()  # Exercise restoration of mixed child-module state.
    image = torch.randn(3, 8, 11)

    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 0.25)

    states_before = _training_states(model)
    grads_before = [parameter.grad.clone() for parameter in model.parameters()]
    running_mean_before = model.norm.running_mean.clone()
    hooks_before = _hook_counts(model.conv)

    cam = generate_cam(
        model=model,
        image=image,
        framework="torch",
        target_layer=model.conv,
        class_index=1,
        cam_method="gradcam",
    )

    assert cam.shape == (8, 11)
    assert np.isfinite(cam).all()
    assert _training_states(model) == states_before
    assert model.relu.inplace is True
    assert torch.equal(model.norm.running_mean, running_mean_before)
    assert _hook_counts(model.conv) == hooks_before
    for parameter, expected_grad in zip(model.parameters(), grads_before, strict=True):
        assert torch.equal(parameter.grad, expected_grad)


def test_direct_gradcam_cleanup_is_idempotent():
    model = _TinyClassifier()
    hooks_before = _hook_counts(model.conv)

    with GradCAM(model, model.conv) as generator:
        forward_hooks, backward_hooks = _hook_counts(model.conv)
        assert forward_hooks == hooks_before[0] + 1
        assert backward_hooks == hooks_before[1] + 1
        cam = generator.generate_cam(torch.randn(1, 3, 6, 9), target_class=0)
        assert cam.shape == (6, 9)

    assert _hook_counts(model.conv) == hooks_before
    generator.remove_hooks()
    assert _hook_counts(model.conv) == hooks_before


def test_unreferenced_gradcam_does_not_remain_attached_to_model():
    model = _TinyClassifier()
    hooks_before = _hook_counts(model.conv)
    generator = GradCAM(model, model.conv)
    generator_ref = weakref.ref(generator)

    del generator
    gc.collect()

    assert generator_ref() is None
    assert _hook_counts(model.conv) == hooks_before


def test_gradcam_enables_required_gradients_inside_inference_mode():
    model = _TinyClassifier()

    with torch.inference_mode():
        image = torch.randn(3, 5, 7)
        cam = generate_cam(
            model=model,
            image=image,
            framework="torch",
            target_layer=model.conv,
            class_index=0,
            cam_method="gradcam",
        )

    assert cam.shape == (5, 7)
    assert np.isfinite(cam).all()


@pytest.mark.parametrize(
    "method",
    [
        "gradcamplusplus",
        "gradcamelementwise",
        "xgradcam",
        "scorecam",
        "eigencam",
        "eigengradcam",
        "layercam",
        "fullgrad",
    ],
)
def test_supported_torch_cam_methods_leave_no_hooks(method):
    model = _TinyClassifier()
    hooks_before = _model_hook_counts(model)

    cam = generate_cam(
        model=model,
        image=torch.randn(3, 5, 7),
        framework="torch",
        target_layer=model.conv,
        class_index=0,
        cam_method=method,
    )

    assert cam.shape == (5, 7)
    assert np.isfinite(cam).all()
    assert _model_hook_counts(model) == hooks_before


@pytest.mark.parametrize("method", ["hirescam", "ablationcam"])
def test_invalid_torch_cam_placeholders_are_rejected(method):
    model = _TinyClassifier()

    with pytest.raises(ValueError, match="unavailable"):
        generate_cam(
            model=model,
            image=torch.randn(3, 5, 7),
            framework="torch",
            target_layer=model.conv,
            class_index=0,
            cam_method=method,
        )


def test_cam_on_image_uses_weights_only_and_cleans_hooks(monkeypatch):
    model = _TinyClassifier()
    calls = []
    hooks_before = _hook_counts(model.conv)

    def fake_load(path, **kwargs):
        calls.append((path, kwargs))
        return model

    monkeypatch.setattr(torch, "load", fake_load)
    output = cam_on_image(
        np.zeros((7, 9, 3), dtype=np.uint8),
        "model.pt",
        cam_name="gradcam",
        target_class=0,
        device="cpu",
    )

    assert output.shape == (7, 9, 3)
    assert calls[0][1]["weights_only"] is True
    assert calls[0][1]["map_location"] == "cpu"
    assert _hook_counts(model.conv) == hooks_before


def test_unsafe_load_fallback_requires_explicit_trust(monkeypatch):
    model = _TinyClassifier()
    weights_only_values = []

    def fake_load(path, **kwargs):
        weights_only = kwargs.get("weights_only")
        weights_only_values.append(weights_only)
        if weights_only is True:
            raise pickle.UnpicklingError("full model pickle")
        return model

    monkeypatch.setattr(torch, "load", fake_load)
    image = np.zeros((5, 7, 3), dtype=np.uint8)

    with pytest.raises(RuntimeError, match="trusted_model=True"):
        cam_on_image(image, "model.pt", device="cpu")
    assert weights_only_values == [True]

    output = cam_on_image(image, "model.pt", device="cpu", trusted_model=True)
    assert output.shape == (5, 7, 3)
    assert weights_only_values == [True, True, False]


def test_rejected_method_does_not_attempt_model_load(monkeypatch):
    def fail_load(*args, **kwargs):
        raise AssertionError("torch.load should not be called")

    monkeypatch.setattr(torch, "load", fail_load)
    with pytest.raises(ValueError, match="unavailable"):
        cam_on_image(
            np.zeros((4, 4, 3), dtype=np.uint8),
            "missing.pt",
            cam_name="ablationcam",
        )
