"""Registries for optimizers, losses, strategies, and dataset constructors."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from kan_cl.continual.regularization import MASImportance, OnlineEWC, SynapticIntelligence
from kan_cl.data.folder import DomainFolderDataset
from kan_cl.data.synthetic import make_synthetic_domains
from kan_cl.registry import Registry, STRATEGIES

OPTIMIZERS: Registry[Any] = Registry("optimizer")
LOSSES: Registry[Any] = Registry("loss")
DATASETS: Registry[Any] = Registry("dataset")

OPTIMIZERS.register("adam")(torch.optim.Adam)
OPTIMIZERS.register("adamw")(torch.optim.AdamW)
OPTIMIZERS.register("sgd")(torch.optim.SGD)
LOSSES.register("cross_entropy")(nn.CrossEntropyLoss)
LOSSES.register("nll")(nn.NLLLoss)
LOSSES.register("mse")(nn.MSELoss)
LOSSES.register("bce_with_logits")(nn.BCEWithLogitsLoss)
STRATEGIES.register("ewc")(OnlineEWC)
STRATEGIES.register("si")(SynapticIntelligence)
STRATEGIES.register("mas")(MASImportance)


@DATASETS.register("synthetic")
def _synthetic_dataset(**kwargs):
    return make_synthetic_domains(**kwargs)


@DATASETS.register("image_folder")
def _image_folder_dataset(root: str, domains: list[str], image_size: int = 224, **kwargs):
    del kwargs
    from kan_cl.data.folder import discover_shared_classes
    class_map = discover_shared_classes(root, domains)
    datasets = [DomainFolderDataset(root, domain, class_map, image_size=image_size)
                for domain in domains]
    return datasets, class_map


def build_optimizer(name: str, parameters, **kwargs):
    return OPTIMIZERS.build(name, params=parameters, **kwargs)


def build_loss(name: str = "cross_entropy", **kwargs):
    return LOSSES.build(name, **kwargs)


def build_strategy(name: str, **kwargs):
    return STRATEGIES.build(name, **kwargs)
