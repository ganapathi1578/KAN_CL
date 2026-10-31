"""Matched synthetic-domain comparison across KAN and continual baselines."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from kan_cl.continual.regularization import OnlineEWC
from kan_cl.data import make_synthetic_domains
from kan_cl.engine.build import build_model
from kan_cl.engine.trainer import run_domain_sequence
from kan_cl.engine.experiment import set_seed


MODEL_CONFIGS = {
    "mlp": {"name": "mlp", "hidden_dims": [32, 32]},
    "vanilla_kan": {"name": "kan", "hidden_dims": [24, 24], "basis": "bspline"},
    "fastkan": {"name": "fastkan", "hidden_dims": [24, 24]},
    "kac": {"name": "kac", "hidden_dims": [24, 24]},
    "protected_spline": {"name": "protected_spline_kan", "hidden_dims": [24, 24]},
    "adaptive_router": {"name": "adaptive_router_kan", "num_domains": 3,
                        "hidden_dims": [16, 16]},
    "hybrid_moe": {"name": "hybrid_moe_kan", "expert_bases": ["bspline", "rbf", "chebyshev"],
                    "hidden_dims": [16, 16], "top_k": 2},
    "progressive": {"name": "progressive_kan", "num_tasks": 1, "hidden_dims": [16, 16]},
}


def main(output: str = "outputs/architecture_comparison.json") -> None:
    seed = 71
    set_seed(seed)
    raw_domains = make_synthetic_domains(num_domains=3, samples_per_domain=240,
                                        input_dim=2, num_classes=3, seed=seed)
    train_loaders, test_loaders = [], []
    for index, dataset in enumerate(raw_domains):
        n_train = int(0.8 * len(dataset))
        train_set, test_set = random_split(
            dataset, [n_train, len(dataset) - n_train],
            generator=torch.Generator().manual_seed(seed + index))
        train_loaders.append(DataLoader(train_set, batch_size=48, shuffle=True))
        test_loaders.append(DataLoader(test_set, batch_size=64, shuffle=False))

    report = {}
    for name, model_config in MODEL_CONFIGS.items():
        set_seed(seed)
        config = {**model_config, "input_dim": 2, "output_dim": 3}
        if name != "mlp":
            config.setdefault("n_basis", 7)
        model = build_model(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        regularizer = None
        if name == "mlp":
            regularizer = OnlineEWC(model, strength=0.1, parameter_filter=None)
        elif name == "protected_spline":
            regularizer = OnlineEWC(model, strength=0.1, parameter_filter="coefficients")
        _, metrics = run_domain_sequence(model, train_loaders, test_loaders, optimizer,
                                         epochs=3, ewc_strength=0.0,
                                         regularizer=regularizer, replay_capacity=48,
                                         replay_batch_size=12)
        report[name] = {"parameter_count": sum(p.numel() for p in model.parameters()),
                        **metrics.summary()}
        print(name, report[name])

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=True), encoding="utf-8")
    print(f"Wrote comparison results to {path}")


if __name__ == "__main__":
    main()
