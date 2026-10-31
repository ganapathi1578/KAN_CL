"""Continual-learning regularizers: per-parameter EWC, SI, MAS, and LwF."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _resize_state_tensor(old: Tensor, target: Tensor, fill_from_target: bool = False) -> Tensor:
    """Resize importance/anchor tensors after expandable model modules change shape."""
    result = target.detach().cpu().clone() if fill_from_target else torch.zeros_like(target, device="cpu")
    old_cpu = old.detach().cpu()
    slices = tuple(slice(0, min(old_dim, new_dim))
                   for old_dim, new_dim in zip(old_cpu.shape, result.shape))
    if len(slices) == old_cpu.ndim and all(s.stop > 0 for s in slices):
        result[slices] = old_cpu[slices]
    return result


def _forward_with_domain(model: nn.Module, x: Tensor, domain_id: int | Tensor | None) -> Tensor:
    if domain_id is not None and hasattr(model, "num_domains") and getattr(model, "router", None) is not None:
        return model(x, domain_labels=torch.as_tensor(domain_id, device=x.device))  # type: ignore[return-value]
    output = model(x)
    if isinstance(output, tuple):
        return output[0]
    return output


class OnlineEWC:
    """Online EWC with coefficient-level importance for KAN spline parameters.

    By default only KAN edge coefficient tensors (including their individual knots)
    are protected. Set `parameter_filter=None` to protect every model parameter.
    Importance is estimated with squared per-batch gradients of the task loss.
    """
    def __init__(self, model: nn.Module, strength: float = 1.0, decay: float = 1.0,
                 parameter_filter: str | None = "coefficients") -> None:
        if strength < 0 or not 0 <= decay <= 1:
            raise ValueError("strength must be >=0 and decay must be in [0, 1]")
        self.strength, self.decay = strength, decay
        self.parameter_filter = parameter_filter
        self.means: dict[str, Tensor] = {}
        self.precision: dict[str, Tensor] = {}

    def _selected(self, model: nn.Module) -> dict[str, nn.Parameter]:
        return {name: p for name, p in model.named_parameters() if p.requires_grad and
                (self.parameter_filter is None or self.parameter_filter in name)}

    def _align_state(self, name: str, parameter: nn.Parameter) -> None:
        current = parameter.detach().cpu()
        if name in self.precision and self.precision[name].shape != current.shape:
            self.precision[name] = _resize_state_tensor(self.precision[name], current)
            self.means[name] = _resize_state_tensor(self.means[name], current, fill_from_target=True)

    def penalty(self, model: nn.Module) -> Tensor:
        selected = self._selected(model)
        if not selected or not self.precision:
            return next(model.parameters()).new_zeros(())
        total = next(model.parameters()).new_zeros(())
        for name, parameter in selected.items():
            if name in self.precision and name in self.means:
                self._align_state(name, parameter)
                total = total + (self.precision[name].to(parameter.device) *
                                 (parameter - self.means[name].to(parameter.device)).square()).sum()
        return self.strength * total

    def estimate_importance(self, model: nn.Module, loader: Iterable,
                            device: torch.device | str = "cpu", domain_id: int | None = None,
                            max_batches: int | None = None) -> dict[str, Tensor]:
        selected = self._selected(model)
        fisher = {name: torch.zeros_like(p, device="cpu") for name, p in selected.items()}
        was_training = model.training
        model.eval()
        batches = 0
        total_examples = 0
        for batch in loader:
            if max_batches is not None and batches >= max_batches:
                break
            x, y = batch[0].to(device), batch[1].to(device)
            model.zero_grad(set_to_none=True)
            logits = _forward_with_domain(model, x, domain_id)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            for name, parameter in selected.items():
                if parameter.grad is not None:
                    fisher[name] += parameter.grad.detach().cpu().square() * x.shape[0]
            total_examples += int(x.shape[0])
            batches += 1
        if batches == 0:
            raise ValueError("Cannot estimate EWC importance from an empty loader")
        # Each per-batch mean loss gradient was weighted by batch size above.
        for name in fisher:
            fisher[name] /= max(1, total_examples)
        if was_training:
            model.train()
        return fisher

    def consolidate(self, model: nn.Module, importance: dict[str, Tensor]) -> None:
        selected = self._selected(model)
        for name, parameter in selected.items():
            if name not in importance:
                continue
            self._align_state(name, parameter)
            new_precision = importance[name].detach().cpu().clamp_min(0)
            old_precision = self.precision.get(name, torch.zeros_like(new_precision))
            old_mean = self.means.get(name, parameter.detach().cpu())
            combined = self.decay * old_precision + new_precision
            weighted_mean = (self.decay * old_precision * old_mean +
                             new_precision * parameter.detach().cpu()) / combined.clamp_min(1e-12)
            self.precision[name] = combined
            self.means[name] = torch.where(combined > 0, weighted_mean, parameter.detach().cpu())

    def end_task(self, model: nn.Module, loader: Iterable, device: torch.device | str = "cpu",
                 domain_id: int | None = None, max_batches: int | None = None) -> None:
        importance = self.estimate_importance(model, loader, device, domain_id, max_batches)
        self.consolidate(model, importance)

    def state_dict(self) -> dict[str, Any]:
        return {"strength": self.strength, "decay": self.decay,
                "parameter_filter": self.parameter_filter,
                "means": self.means, "precision": self.precision}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.strength = float(state["strength"])
        self.decay = float(state["decay"])
        self.parameter_filter = state.get("parameter_filter", "coefficients")
        self.means = state["means"]
        self.precision = state["precision"]


class SynapticIntelligence:
    """A lightweight SI implementation tracking parameter-wise path importance."""
    def __init__(self, model: nn.Module, strength: float = 1.0, damping: float = 0.1) -> None:
        self.strength, self.damping = strength, damping
        self.omega: dict[str, Tensor] = {}
        self.anchor: dict[str, Tensor] = {}
        self.previous: dict[str, Tensor] = {}
        self.path_integral: dict[str, Tensor] = {}
        self._parameter_names: list[str] = []
        self.begin_task(model)

    def begin_task(self, model: nn.Module) -> None:
        self._parameter_names = [n for n, p in model.named_parameters() if p.requires_grad]
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            value = p.detach().cpu().clone()
            self.anchor[name] = value
            self.previous[name] = value.clone()
            self.path_integral[name] = torch.zeros_like(value)
            if name not in self.omega:
                self.omega[name] = torch.zeros_like(value)
            elif self.omega[name].shape != value.shape:
                self.omega[name] = _resize_state_tensor(self.omega[name], value)

    @torch.no_grad()
    def update_after_step(self, model: nn.Module) -> None:
        params = dict(model.named_parameters())
        for name in self._parameter_names:
            p = params[name]
            if p.grad is not None:
                current = p.detach().cpu()
                gradient = p.grad.detach().cpu()
                delta = current - self.previous[name]
                self.path_integral[name] += (-gradient * delta).clamp_min(0)
                self.previous[name].copy_(current)

    @torch.no_grad()
    def end_task(self, model: nn.Module) -> None:
        params = dict(model.named_parameters())
        for name in self._parameter_names:
            current = params[name].detach().cpu()
            delta = current - self.anchor[name]
            contribution = self.path_integral[name] / (delta.square() + self.damping)
            self.omega[name] += contribution.clamp_min(0)
            self.anchor[name] = current.clone()
        self.begin_task(model)

    def penalty(self, model: nn.Module) -> Tensor:
        params = dict(model.named_parameters())
        if not params:
            return torch.tensor(0.0)
        total = next(iter(params.values())).new_zeros(())
        for name, p in params.items():
            if name in self.omega:
                total = total + (self.omega[name].to(p.device) *
                             (p - self.anchor[name].to(p.device)).square()).sum()
        return self.strength * total

    def state_dict(self) -> dict[str, Any]:
        return {"strength": self.strength, "damping": self.damping, "omega": self.omega,
                "anchor": self.anchor, "previous": self.previous,
                "path_integral": self.path_integral, "parameter_names": self._parameter_names}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.strength = float(state["strength"])
        self.damping = float(state["damping"])
        self.omega = state["omega"]
        self.anchor = state["anchor"]
        self.previous = state["previous"]
        self.path_integral = state["path_integral"]
        self._parameter_names = state["parameter_names"]


class MASImportance:
    """Memory Aware Synapses importance from output sensitivity."""
    def __init__(self, model: nn.Module, strength: float = 1.0) -> None:
        self.strength = strength
        self.importance: dict[str, Tensor] = {}
        self.anchor: dict[str, Tensor] = {}
        self._filter = [n for n, p in model.named_parameters() if p.requires_grad]

    def estimate(self, model: nn.Module, loader: Iterable, device: torch.device | str = "cpu",
                 max_batches: int | None = None) -> None:
        current_names = [n for n, p in model.named_parameters() if p.requires_grad]
        self._filter = sorted(set(self._filter).union(current_names))
        accum = {n: torch.zeros_like(p, device="cpu") for n, p in model.named_parameters()
                 if n in self._filter and p.requires_grad}
        was_training = model.training
        model.eval()
        count = 0
        for batch in loader:
            if max_batches is not None and count >= max_batches:
                break
            x = batch[0].to(device)
            model.zero_grad(set_to_none=True)
            output = model(x)
            if isinstance(output, tuple):
                output = output[0]
            objective = output.square().sum(dim=-1).mean()
            objective.backward()
            for name, p in model.named_parameters():
                if name in accum and p.grad is not None:
                    accum[name] += p.grad.detach().abs().cpu() * x.shape[0]
            count += 1
        if count == 0:
            raise ValueError("Cannot estimate MAS importance from an empty loader")
        for name, value in accum.items():
            value /= count
            old_importance = self.importance.get(name, torch.zeros_like(value))
            if old_importance.shape != value.shape:
                old_importance = _resize_state_tensor(old_importance, value)
            self.importance[name] = old_importance + value
            self.anchor[name] = dict(model.named_parameters())[name].detach().cpu().clone()
        if was_training:
            model.train()

    def penalty(self, model: nn.Module) -> Tensor:
        params = dict(model.named_parameters())
        if not params:
            return torch.tensor(0.0)
        total = next(iter(params.values())).new_zeros(())
        for name, p in params.items():
            if name in self.importance:
                total += (self.importance[name].to(p.device) *
                          (p - self.anchor[name].to(p.device)).square()).sum()
        return self.strength * total

    def state_dict(self) -> dict[str, Any]:
        return {"strength": self.strength, "importance": self.importance,
                "anchor": self.anchor, "parameter_names": self._filter}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.strength = float(state["strength"])
        self.importance = state["importance"]
        self.anchor = state["anchor"]
        self._filter = state["parameter_names"]


def lwf_distillation_loss(student_logits: Tensor, teacher_logits: Tensor,
                          temperature: float = 2.0) -> Tensor:
    """Learning-without-Forgetting distillation loss (teacher logits detached)."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return F.kl_div(F.log_softmax(student_logits / temperature, dim=-1),
                    F.softmax(teacher_logits.detach() / temperature, dim=-1),
                    reduction="batchmean") * (temperature ** 2)
