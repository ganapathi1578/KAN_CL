"""KAN Continual: modular KAN architectures and domain-incremental learning."""
from kan_cl.models import (
    AdaptiveRouterKAN,
    CNNClassifier,
    FastKAN,
    HybridBasisMoEKAN,
    KACClassifier,
    KANLinear,
    KANNetwork,
    MLPClassifier,
    ProgressiveKAN,
    ProtectedSplineKAN,
    VanillaKAN,
    VisionKAN,
)
from kan_cl.continual import (
    ContinualMetrics,
    MASImportance,
    OnlineEWC,
    ReplayBuffer,
    SynapticIntelligence,
    lwf_distillation_loss,
)

__version__ = "0.1.0"
__all__ = [
    "AdaptiveRouterKAN", "CNNClassifier", "FastKAN", "HybridBasisMoEKAN", "KACClassifier",
    "KANLinear", "KANNetwork", "MLPClassifier", "ProgressiveKAN",
    "ProtectedSplineKAN", "VanillaKAN", "VisionKAN", "ContinualMetrics",
    "MASImportance", "OnlineEWC", "ReplayBuffer", "SynapticIntelligence",
    "lwf_distillation_loss",
]
