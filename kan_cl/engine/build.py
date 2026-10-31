"""Config-driven construction helpers."""
from __future__ import annotations

from typing import Any

from kan_cl.models.architectures import AdaptiveRouterKAN, HybridBasisMoEKAN, ProtectedSplineKAN, VanillaKAN
from kan_cl.registry import MODELS
from kan_cl.engine.components import OPTIMIZERS, LOSSES, DATASETS


def build_model(config: dict[str, Any]):
    model_config = dict(config)
    name = model_config.pop("name", "kan")
    return MODELS.build(name, **model_config)
