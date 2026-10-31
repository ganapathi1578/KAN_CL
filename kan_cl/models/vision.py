"""Compact convolutional feature front-end with a selectable KAN continual head."""
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from kan_cl.models.architectures import AdaptiveRouterKAN, HybridBasisMoEKAN, ProtectedSplineKAN, VanillaKAN
from kan_cl.registry import MODELS


class ConvFeatureEncoder(nn.Module):
    """Small trainable image encoder; intentionally dependency-free (no torchvision)."""
    def __init__(self, in_channels: int = 3, feature_dim: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128, feature_dim), nn.LayerNorm(feature_dim), nn.SiLU(),
        )
        self.feature_dim = feature_dim

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected images [B,C,H,W], got shape {tuple(x.shape)}")
        return self.network(x)


class VisionKAN(nn.Module):
    """Image encoder + modular KAN head.

    `head_type` may be `kan`, `protected_spline_kan`, `adaptive_router_kan`, or
    `hybrid_moe_kan`. The wrapper delegates domain growth to its head.
    """
    def __init__(self, output_dim: int, head_type: str = "protected_spline_kan",
                 feature_dim: int = 128, in_channels: int = 3,
                 hidden_dims: Sequence[int] = (64,), basis: str = "bspline",
                 n_basis: int = 8, degree: int = 3, num_domains: int = 2,
                 expert_bases: Sequence[str] = ("bspline", "rbf", "chebyshev"),
                 top_k: int | None = None, freeze_backbone: bool = False) -> None:
        super().__init__()
        self.backbone = ConvFeatureEncoder(in_channels, feature_dim)
        common = {"input_dim": feature_dim, "hidden_dims": hidden_dims,
                  "output_dim": output_dim, "n_basis": n_basis, "degree": degree}
        if head_type == "kan":
            self.head = VanillaKAN(**common, basis=basis)
        elif head_type == "protected_spline_kan":
            self.head = ProtectedSplineKAN(**common, basis="bspline")
        elif head_type == "adaptive_router_kan":
            self.head = AdaptiveRouterKAN(**common, basis=basis, num_domains=num_domains)
        elif head_type == "hybrid_moe_kan":
            self.head = HybridBasisMoEKAN(**common, expert_bases=expert_bases, top_k=top_k)
        else:
            raise KeyError(f"Unknown vision head_type: {head_type}")
        self.head_type = head_type
        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)

    @property
    def num_domains(self) -> int:
        return self.head.num_domains if isinstance(self.head, AdaptiveRouterKAN) else 0

    @property
    def router(self):
        return self.head.router if isinstance(self.head, AdaptiveRouterKAN) else None

    def add_domain(self, copy_from: int | None = None) -> int:
        if not isinstance(self.head, AdaptiveRouterKAN):
            raise RuntimeError("add_domain is available only for an adaptive-router head")
        return self.head.add_domain(copy_from)

    def forward(self, x: Tensor, *args, **kwargs):
        features = self.backbone(x)
        return self.head(features, *args, **kwargs)


@MODELS.register("vision_kan")
def _build_vision_kan(**kwargs):
    kwargs.setdefault("head_type", "kan")
    return VisionKAN(**kwargs)


@MODELS.register("vision_protected_spline_kan")
def _build_vision_protected(**kwargs):
    kwargs.setdefault("head_type", "protected_spline_kan")
    return VisionKAN(**kwargs)


@MODELS.register("vision_adaptive_router_kan")
def _build_vision_router(**kwargs):
    kwargs.setdefault("head_type", "adaptive_router_kan")
    return VisionKAN(**kwargs)


@MODELS.register("vision_hybrid_moe_kan")
def _build_vision_moe(**kwargs):
    kwargs.setdefault("head_type", "hybrid_moe_kan")
    return VisionKAN(**kwargs)


@MODELS.register("cnn")
class CNNClassifier(nn.Module):
    """Conventional CNN classifier baseline for image-folder experiments."""
    def __init__(self, output_dim: int, feature_dim: int = 128, in_channels: int = 3,
                 freeze_backbone: bool = False) -> None:
        super().__init__()
        self.backbone = ConvFeatureEncoder(in_channels, feature_dim)
        self.classifier = nn.Linear(feature_dim, output_dim)
        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.backbone(x))
