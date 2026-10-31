"""Image-folder domain dataset: root/domain/class_name/image_file.

The implementation avoids a hard dependency on torchvision so it works in minimal
CPU environments. Images are RGB, resized, converted to float [0, 1], and normalized
with configurable channel statistics.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


class DomainFolderDataset(Dataset):
    def __init__(self, root: str | Path, domain: str, class_to_idx: dict[str, int] | None = None,
                 image_size: int = 224, mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
                 std: tuple[float, float, float] = (0.229, 0.224, 0.225),
                 transform: Callable[[Image.Image], Tensor] | None = None) -> None:
        self.root = Path(root)
        self.domain = domain
        self.domain_root = self.root / domain
        if not self.domain_root.is_dir():
            raise FileNotFoundError(f"Domain folder not found: {self.domain_root}")
        self.classes = sorted(p.name for p in self.domain_root.iterdir() if p.is_dir())
        if not self.classes:
            raise ValueError(f"No class folders found under {self.domain_root}")
        self.class_to_idx = class_to_idx or {name: i for i, name in enumerate(self.classes)}
        unknown = set(self.classes) - set(self.class_to_idx)
        if unknown:
            raise ValueError(f"Classes missing from shared label map: {sorted(unknown)}")
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        self.samples = [(path, self.class_to_idx[class_name])
                        for class_name in self.classes
                        for path in sorted((self.domain_root / class_name).rglob("*"))
                        if path.is_file() and path.suffix.lower() in extensions]
        if not self.samples:
            raise ValueError(f"No supported images found in {self.domain_root}")
        self.image_size = image_size
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                tensor = self.transform(image)
            else:
                image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
                array = np.asarray(image, dtype=np.float32) / 255.0
                tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
                tensor = (tensor - self.mean) / self.std
        return tensor, torch.tensor(label, dtype=torch.long)


def discover_shared_classes(root: str | Path, domains: list[str]) -> dict[str, int]:
    root = Path(root)
    classes: set[str] = set()
    for domain in domains:
        directory = root / domain
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        classes.update(p.name for p in directory.iterdir() if p.is_dir())
    return {name: idx for idx, name in enumerate(sorted(classes))}
