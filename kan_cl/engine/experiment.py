"""Runnable synthetic-domain experiment entry point."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from kan_cl.engine.build import build_model
from kan_cl.engine.components import DATASETS, build_loss, build_optimizer, build_strategy
from kan_cl.engine.trainer import run_domain_sequence


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    content = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(content)
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("YAML config requires PyYAML; install it or use a JSON config") from exc
    loaded = yaml.safe_load(content)
    if not isinstance(loaded, dict):
        raise ValueError("Configuration root must be a mapping")
    return loaded


def run_experiment(config: dict[str, Any], output_dir: str | Path = "outputs/run",
                   device: str | None = None) -> dict[str, Any]:
    seed = int(config.get("seed", 17))
    set_seed(seed)
    selected_device = device or config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if str(selected_device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this PyTorch environment")
    dataset_cfg = config.get("dataset", {})
    domain_count = int(dataset_cfg.get("num_domains", 3))
    batch_size = int(dataset_cfg.get("batch_size", 64))
    dataset_type = dataset_cfg.get("type", "synthetic")
    train_loaders, test_loaders = [], []
    if dataset_type == "synthetic":
        domains = DATASETS.build(
            "synthetic", num_domains=domain_count,
            samples_per_domain=int(dataset_cfg.get("samples_per_domain", 256)),
            input_dim=int(dataset_cfg.get("input_dim", 2)),
            num_classes=int(dataset_cfg.get("num_classes", 3)),
            seed=seed, noise=float(dataset_cfg.get("noise", 0.35)),
        )
        for domain_dataset in domains:
            train_count = max(1, int(0.8 * len(domain_dataset)))
            test_count = len(domain_dataset) - train_count
            if test_count == 0:
                train_count -= 1
                test_count = 1
            train_set, test_set = random_split(
                domain_dataset, [train_count, test_count],
                generator=torch.Generator().manual_seed(seed + len(train_loaders)))
            train_loaders.append(DataLoader(train_set, batch_size=batch_size, shuffle=True))
            test_loaders.append(DataLoader(test_set, batch_size=batch_size, shuffle=False))
    elif dataset_type == "image_folder":
        domain_names = dataset_cfg.get("domains")
        if not domain_names:
            raise ValueError("dataset.domains is required for type=image_folder")
        folder_datasets, shared_classes = DATASETS.build(
            "image_folder", root=dataset_cfg["root"], domains=list(domain_names),
            image_size=int(dataset_cfg.get("image_size", 224)))
        domain_count = len(domain_names)
        image_model_name = str(config.get("model", {}).get("name", ""))
        if not image_model_name.startswith("vision_") and image_model_name != "cnn":
            raise ValueError("image_folder datasets require a vision_* KAN wrapper or the cnn baseline")
        for full_dataset in folder_datasets:
            if len(full_dataset) < 2:
                raise ValueError("Each image domain must contain at least two images for train/test splitting")
            train_count = min(len(full_dataset) - 1, max(1, int(0.8 * len(full_dataset))))
            test_count = len(full_dataset) - train_count
            train_set, test_set = random_split(
                full_dataset, [train_count, test_count],
                generator=torch.Generator().manual_seed(seed + len(train_loaders)))
            train_loaders.append(DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                            num_workers=int(dataset_cfg.get("num_workers", 0))))
            test_loaders.append(DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                           num_workers=int(dataset_cfg.get("num_workers", 0))))
        dataset_cfg["num_classes"] = len(shared_classes)
        dataset_cfg["input_dim"] = None
    else:
        raise KeyError(f"Unknown dataset.type: {dataset_type!r}")

    model_config = dict(config.get("model", {}))
    model_name = model_config.get("name", "kan")
    is_image_model = str(model_name).startswith("vision_") or model_name == "cnn"
    if not is_image_model:
        model_config.setdefault("input_dim", int(dataset_cfg.get("input_dim", 2)))
    model_config.setdefault("output_dim", int(dataset_cfg.get("num_classes", 3)))
    if not is_image_model:
        model_config.setdefault("hidden_dims", [32, 32])
    if model_name in {"adaptive_router_kan", "vision_adaptive_router_kan"}:
        model_config.setdefault("num_domains", domain_count)
    model = build_model(model_config)
    optim_cfg = config.get("optimizer", {})
    optimizer_name = optim_cfg.get("name", "adamw")
    optimizer_kwargs = {k: v for k, v in optim_cfg.items() if k != "name"}
    optimizer_kwargs.setdefault("lr", 1e-3)
    optimizer_kwargs.setdefault("weight_decay", 1e-4)
    optimizer = build_optimizer(optimizer_name, model.parameters(), **optimizer_kwargs)
    loss_cfg = config.get("loss", {})
    loss_fn = build_loss(loss_cfg.get("name", "cross_entropy"), **loss_cfg.get("kwargs", {}))
    continual_cfg = config.get("continual", {})
    strategy_cfg = continual_cfg.get("strategy")
    regularizer = None
    if strategy_cfg:
        strategy_cfg = dict(strategy_cfg)
        strategy_name = strategy_cfg.pop("name")
        regularizer = build_strategy(strategy_name, model=model, **strategy_cfg)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer, metrics = run_domain_sequence(
        model, train_loaders, test_loaders, optimizer,
        epochs=int(config.get("training", {}).get("epochs", 3)),
        device=selected_device,
        ewc_strength=float(continual_cfg.get("ewc_strength", 0.0)),
        regularizer=regularizer,
        replay_capacity=int(continual_cfg.get("replay_capacity", 0)),
        replay_batch_size=int(continual_cfg.get("replay_batch_size", 0)),
        checkpoint_dir=output_dir / "checkpoints",
        loss_fn=loss_fn,
        lwf_weight=float(continual_cfg.get("lwf_weight", 0.0)),
        lwf_temperature=float(continual_cfg.get("lwf_temperature", 2.0)),
    )
    summary = {
        "config": config,
        "history": trainer.history,
        "accuracy_matrix": metrics.matrix.tolist(),
        "metrics": metrics.summary(),
        "device": str(selected_device),
        "model_parameters": sum(p.numel() for p in model.parameters()),
    }
    (output_dir / "results.json").write_text(json.dumps(summary, indent=2, allow_nan=True),
                                               encoding="utf-8")
    torch.save(model.state_dict(), output_dir / "final_model.pt")
    return summary
