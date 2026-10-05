"""Dataset utilities for binary retinal-image classification."""

from pathlib import Path
from typing import Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class RetinalImageDataset(Dataset):
    """Load class-labelled retinal images from a directory tree.

    Expected layout::

        root/
            NRG/
                image_1.jpg
                ...
            RG/
                image_2.jpg
                ...

    The return signature is ``(class_names, image_tensor, class_id)`` so that
    existing training code that expects the original three-item structure can
    be used without changes to the model interface.
    """

    def __init__(self, root_dir: str, classes: Sequence[str], is_train: bool = False,
                 image_size: int = 224) -> None:
        self.root_dir = Path(root_dir)
        self.classes = list(classes)
        self.is_train = is_train
        self.image_size = image_size

        if not self.root_dir.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {self.root_dir}")

        self.class_to_index = {name: idx for idx, name in enumerate(self.classes)}
        self.samples = self._collect_samples()
        if not self.samples:
            raise RuntimeError(f"No supported images found in {self.root_dir}")

        self.train_transform = transforms.Compose([
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=(0, 360)),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])
        self.eval_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

    def _collect_samples(self):
        valid_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        samples = []
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Missing class directory: {class_dir}")
            for path in sorted(class_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in valid_suffixes:
                    samples.append((path, self.class_to_index[class_name]))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        transform = self.train_transform if self.is_train else self.eval_transform
        image = transform(image)
        return self.classes, image, torch.tensor(label, dtype=torch.long)


# Backward-compatible name used by the training script.
Custom_Dataset = RetinalImageDataset
