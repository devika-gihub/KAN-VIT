"""Example Grad-CAM visualization entry point."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchinfo import summary
from torchvision.transforms.functional import to_tensor
from torchvision.transforms import Normalize, Resize, Compose
from torchvision.utils import make_grid, save_image

from gradcam import GradCAM, GradCAMpp
from model import kat_tiny_patch16_224
from utils import visualize_cam


# Set these paths for your experiment rather than editing the visualization code.
IMAGE_PATH = Path("/workspace/Acrima_split/train/RG/Im312_g_ACRIMA.jpg")
OUTPUT_DIR = Path("/workspace/outputs")


def main():
    image = Image.open(IMAGE_PATH).convert("RGB")
    tensor = to_tensor(Resize((224, 224))(image)).unsqueeze(0)
    normalizer = Compose([Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    normalized = normalizer(tensor)

    model = kat_tiny_patch16_224().eval().cuda()
    summary(model, (1, 3, 224, 224))

    model_info = {
        "type": "kat_tiny_patch16_224",
        "arch": model,
        "layer_name": "head.weight",
        "input_size": (224, 224),
    }
    gradcam = GradCAM(model_info)
    gradcam_pp = GradCAMpp(model_info)

    with torch.enable_grad():
        mask, _ = gradcam(normalized.cuda())
        mask_pp, _ = gradcam_pp(normalized.cuda())

    heatmap, overlay = visualize_cam(mask, tensor)
    heatmap_pp, overlay_pp = visualize_cam(mask_pp, tensor)
    grid = make_grid(torch.stack([tensor.squeeze(0), heatmap, heatmap_pp, overlay, overlay_pp]))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_image(grid, OUTPUT_DIR / IMAGE_PATH.name)
    print(f"Saved visualization to {OUTPUT_DIR / IMAGE_PATH.name}")


if __name__ == "__main__":
    main()
