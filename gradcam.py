"""Grad-CAM and Grad-CAM++ utilities for convolutional feature maps."""

import torch
import torch.nn.functional as F

from utils import (
    find_alexnet_layer,
    find_custom_layer,
    find_densenet_layer,
    find_resnet_layer,
    find_squeezenet_layer,
    find_vgg_layer,
)


class GradCAM:
    """Generate a class-specific Grad-CAM map from a selected layer."""

    def __init__(self, model_dict, model=None, verbose=False):
        self.model_arch = model_dict["arch"]
        self.model_type = model_dict.get("type", model_dict.get("model_type", ""))
        self.layer_name = model_dict["layer_name"]
        self.gradients = None
        self.activations = None

        target = self._resolve_layer()
        target.register_forward_hook(self._save_activation)
        target.register_full_backward_hook(self._save_gradient)

        if verbose and "input_size" in model_dict:
            device = next(self.model_arch.parameters()).device
            with torch.no_grad():
                dummy = torch.zeros(1, 3, *model_dict["input_size"], device=device)
                self.model_arch(dummy)
                print("activation shape:", tuple(self.activations.shape))

    def _resolve_layer(self):
        kind = self.model_type.lower()
        if "vgg" in kind:
            return find_vgg_layer(self.model_arch, self.layer_name)
        if "resnet" in kind:
            return find_resnet_layer(self.model_arch, self.layer_name)
        if "densenet" in kind:
            return find_densenet_layer(self.model_arch, self.layer_name)
        if "alexnet" in kind:
            return find_alexnet_layer(self.model_arch, self.layer_name)
        if "squeezenet" in kind:
            return find_squeezenet_layer(self.model_arch, self.layer_name)
        if "kat_tiny_patch16_224" in kind:
            return self.model_arch.blocks[-1].norm2
        return find_custom_layer(self.model_arch, self.layer_name)

    def _save_activation(self, module, inputs, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def forward(self, input_tensor, class_idx=None, retain_graph=False):
        logits = self.model_arch(input_tensor)
        if class_idx is None:
            class_idx = logits.argmax(dim=1)
        if isinstance(class_idx, int):
            score = logits[:, class_idx].sum()
        else:
            score = logits.gather(1, class_idx.view(-1, 1)).sum()

        self.model_arch.zero_grad(set_to_none=True)
        score.backward(retain_graph=retain_graph)

        gradients = self.gradients
        activations = self.activations
        weights = gradients.flatten(2).mean(dim=2).unsqueeze(-1).unsqueeze(-1)
        cam = F.relu((weights * activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam_min, cam_max = cam.amin(dim=(-2, -1), keepdim=True), cam.amax(dim=(-2, -1), keepdim=True)
        cam = (cam - cam_min) / (cam_max - cam_min).clamp_min(1e-8)
        return cam.detach(), logits.detach()

    __call__ = forward


class GradCAMpp(GradCAM):
    """Grad-CAM++ implementation using higher-order gradient weights."""

    def forward(self, input_tensor, class_idx=None, retain_graph=True):
        logits = self.model_arch(input_tensor)
        if class_idx is None:
            class_idx = logits.argmax(dim=1)
        if isinstance(class_idx, int):
            score = logits[:, class_idx].sum()
        else:
            score = logits.gather(1, class_idx.view(-1, 1)).sum()

        self.model_arch.zero_grad(set_to_none=True)
        score.backward(retain_graph=retain_graph, create_graph=True)

        gradients = self.gradients
        activations = self.activations
        grad_sq = gradients.pow(2)
        grad_cube = gradients.pow(3)
        denominator = 2 * grad_sq + (activations * grad_cube).sum(dim=(2, 3), keepdim=True)
        alpha = grad_sq / denominator.clamp_min(1e-8)
        weights = (alpha * F.relu(gradients)).sum(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam_min, cam_max = cam.amin(dim=(-2, -1), keepdim=True), cam.amax(dim=(-2, -1), keepdim=True)
        cam = (cam - cam_min) / (cam_max - cam_min).clamp_min(1e-8)
        return cam.detach(), logits.detach()
