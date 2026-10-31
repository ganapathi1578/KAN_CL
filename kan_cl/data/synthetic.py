"""Deterministic low-dimensional domain-shift datasets for smoke tests and demos."""
from __future__ import annotations

import math
import torch
from torch import Tensor
from torch.utils.data import TensorDataset


def make_synthetic_domains(num_domains: int = 4, samples_per_domain: int = 256,
                           input_dim: int = 2, num_classes: int = 2,
                           seed: int = 17, noise: float = 0.35) -> list[TensorDataset]:
    """Create same-label-task domains with changing rotations/scales/translations.

    Latent class clusters are shared across domains; each domain applies a distinct
    affine transform. The returned datasets are (features, class labels) pairs.
    """
    if num_domains < 1 or samples_per_domain < 2 or input_dim < 2 or num_classes < 2:
        raise ValueError("Need >=1 domains, >=2 samples, input_dim>=2 and >=2 classes")
    datasets: list[TensorDataset] = []
    for domain in range(num_domains):
        generator = torch.Generator().manual_seed(seed + domain)
        labels = torch.randint(num_classes, (samples_per_domain,), generator=generator)
        # Deterministic class centres arranged around a circle in the first two dims.
        angles = torch.arange(num_classes, dtype=torch.float32) * (2.0 * math.pi / num_classes)
        centers = torch.zeros(num_classes, input_dim)
        centers[:, 0] = 1.25 * torch.cos(angles)
        centers[:, 1] = 1.25 * torch.sin(angles)
        latent = centers[labels] + noise * torch.randn(samples_per_domain, input_dim, generator=generator)
        angle = (domain / max(1, num_domains - 1)) * math.pi * 0.75
        rotation = torch.eye(input_dim)
        rotation[:2, :2] = torch.tensor([[math.cos(angle), -math.sin(angle)],
                                         [math.sin(angle), math.cos(angle)]])
        scales = torch.ones(input_dim)
        scales[0] = 0.65 + 0.45 * ((domain % 3) / 2.0)
        scales[1] = 1.35 - 0.45 * ((domain % 3) / 2.0)
        shift = torch.zeros(input_dim)
        shift[0] = 0.35 * math.sin(domain)
        shift[1] = 0.35 * math.cos(domain)
        features = latent @ rotation.T * scales + shift
        datasets.append(TensorDataset(features.float(), labels.long()))
    return datasets
