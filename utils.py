"""General utilities for visualization and Grad-CAM layer selection."""

import cv2
import numpy as np
import torch


def visualize_cam(mask: torch.Tensor, image: torch.Tensor):
    """Convert a normalized CAM mask into a heatmap and overlay."""
    mask_np = mask.detach().squeeze().cpu().numpy()
    heatmap = cv2.applyColorMap(np.uint8(np.clip(mask_np, 0, 1) * 255), cv2.COLORMAP_JET)
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float() / 255.0
    heatmap = heatmap[[2, 1, 0]]

    base = image.detach().cpu().squeeze(0)
    overlay = heatmap + base
    overlay = overlay / overlay.max().clamp_min(torch.finfo(overlay.dtype).eps)
    return heatmap, overlay


def _walk_modules(model, path):
    current = model
    for part in path.split("_"):
        if part.isdigit():
            current = current[int(part)]
        else:
            try:
                current = current._modules[part]
            except (KeyError, AttributeError, IndexError) as exc:
                raise ValueError(f"Layer '{path}' was not found") from exc
    return current


def find_custom_layer(model, target_layer_name):
    return _walk_modules(model, target_layer_name)


def find_resnet_layer(arch, target_layer_name):
    parts = target_layer_name.split("_")
    if not parts[0].startswith("layer"):
        return arch._modules[target_layer_name]
    block_group = arch._modules[parts[0]]
    if len(parts) == 1:
        return block_group
    block_index = int(parts[1].lstrip("bottleneck").lstrip("basicblock"))
    layer = block_group[block_index]
    for name in parts[2:]:
        layer = layer._modules[name]
    return layer


def find_densenet_layer(arch, target_layer_name):
    return _walk_modules(arch, target_layer_name)


def find_vgg_layer(arch, target_layer_name):
    parts = target_layer_name.split("_")
    layer = arch.features if parts[0] == "features" else arch._modules[parts[0]]
    for part in parts[1:]:
        layer = layer[int(part)] if part.isdigit() else layer._modules[part]
    return layer


def find_alexnet_layer(arch, target_layer_name):
    return find_vgg_layer(arch, target_layer_name)


def find_squeezenet_layer(arch, target_layer_name):
    return _walk_modules(arch, target_layer_name)


def denormalize(tensor, mean, std):
    if tensor.ndim != 4:
        raise TypeError("tensor should be 4D")
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device).view(1, 3, 1, 1)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device).view(1, 3, 1, 1)
    return tensor * std + mean


def normalize(tensor, mean, std):
    if tensor.ndim != 4:
        raise TypeError("tensor should be 4D")
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device).view(1, 3, 1, 1)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device).view(1, 3, 1, 1)
    return (tensor - mean) / std


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return normalize(tensor, self.mean, self.std)

    def undo(self, tensor):
        return denormalize(tensor, self.mean, self.std)
