"""Sequential domain training engine with optional EWC and replay."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from kan_cl.continual.metrics import ContinualMetrics
from kan_cl.continual.regularization import (MASImportance, OnlineEWC, SynapticIntelligence,
                                             lwf_distillation_loss)
from kan_cl.continual.replay import ReplayBuffer


def _move_batch(batch: tuple[Tensor, Tensor], device: torch.device) -> tuple[Tensor, Tensor]:
    return batch[0].to(device), batch[1].to(device)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device | str = "cpu",
             task_id: int | None = None) -> float:
    model.eval()
    correct, count = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if hasattr(model, "num_tasks") and task_id is not None:
            logits = model(x, task_id=task_id)
        else:
            logits = model(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        correct += int((logits.argmax(dim=-1) == y).sum())
        count += int(y.numel())
    return correct / max(1, count)


class ContinualTrainer:
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                 device: torch.device | str = "cpu", ewc: OnlineEWC | None = None,
                 replay: ReplayBuffer | None = None, regularizer: Any | None = None,
                 router_loss_weight: float = 0.2,
                 moe_balance_weight: float = 0.01, moe_diversity_weight: float = 0.0,
                 grad_clip_norm: float | None = 1.0, loss_fn: nn.Module | None = None,
                 lwf_weight: float = 0.0, lwf_temperature: float = 2.0) -> None:
        self.model, self.optimizer = model, optimizer
        self.device = torch.device(device)
        self.model.to(self.device)
        self.regularizer = regularizer if regularizer is not None else ewc
        self.ewc = self.regularizer if isinstance(self.regularizer, OnlineEWC) else None
        self.replay = replay
        self.router_loss_weight = router_loss_weight
        self.moe_balance_weight, self.moe_diversity_weight = moe_balance_weight, moe_diversity_weight
        self.grad_clip_norm = grad_clip_norm
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        if lwf_weight < 0 or lwf_temperature <= 0:
            raise ValueError("lwf_weight must be >= 0 and lwf_temperature must be > 0")
        self.lwf_weight, self.lwf_temperature = lwf_weight, lwf_temperature
        self.teacher: nn.Module | None = None
        self.history: list[dict[str, float | int]] = []

    def _classification_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        if isinstance(self.loss_fn, nn.NLLLoss):
            return self.loss_fn(F.log_softmax(logits, dim=-1), targets)
        if isinstance(self.loss_fn, (nn.BCEWithLogitsLoss, nn.MSELoss)):
            one_hot = F.one_hot(targets.long(), num_classes=logits.shape[-1]).to(logits.dtype)
            return self.loss_fn(logits, one_hot)
        return self.loss_fn(logits, targets)

    def _forward_and_loss(self, x: Tensor, y: Tensor, domain_ids: Tensor | None) -> tuple[Tensor, Tensor]:
        is_router = hasattr(self.model, "num_domains") and getattr(self.model, "router", None) is not None
        head = getattr(self.model, "head", self.model)
        if hasattr(self.model, "num_tasks") and domain_ids is not None:
            logits = x.new_empty((x.shape[0], self.model.output_dim))
            for task in torch.unique(domain_ids).tolist():
                mask = domain_ids == int(task)
                logits[mask] = self.model(x[mask], task_id=int(task))
            return self._classification_loss(logits, y), logits
        if is_router:
            logits, aux = self.model(x, domain_labels=domain_ids, return_aux=True)
            task_loss = self._classification_loss(logits, y)
            if domain_ids is not None:
                router_loss = F.cross_entropy(aux["router_logits"], domain_ids)
                task_loss = task_loss + self.router_loss_weight * router_loss
            return task_loss, logits
        if hasattr(head, "load_balance_loss"):
            logits, aux = self.model(x, return_aux=True)
            loss = self._classification_loss(logits, y)
            loss = loss + self.moe_balance_weight * head.load_balance_loss(aux["gates"])
            if self.moe_diversity_weight:
                loss = loss + self.moe_diversity_weight * head.diversity_loss(aux["expert_outputs"])
            return loss, logits
        logits = self.model(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        return self._classification_loss(logits, y), logits

    def train_domain(self, domain_id: int, train_loader: DataLoader, epochs: int = 1,
                     replay_batch_size: int = 0) -> dict[str, float | int]:
        if epochs < 1:
            raise ValueError("epochs must be >= 1")
        self.model.train()
        loss_sum, correct, count, steps = 0.0, 0, 0, 0
        for _ in range(epochs):
            for batch in train_loader:
                x, y = _move_batch(batch, self.device)
                domain_ids = torch.full((y.shape[0],), domain_id, dtype=torch.long, device=self.device)
                if self.replay is not None and replay_batch_size > 0 and len(self.replay) > 0:
                    rx, ry, rd = self.replay.sample(replay_batch_size, self.device)
                    x = torch.cat((x, rx), dim=0)
                    y = torch.cat((y, ry), dim=0)
                    domain_ids = torch.cat((domain_ids, rd), dim=0)
                self.optimizer.zero_grad(set_to_none=True)
                loss, logits = self._forward_and_loss(x, y, domain_ids)
                if self.regularizer is not None:
                    loss = loss + self.regularizer.penalty(self.model)
                if self.teacher is not None and self.lwf_weight > 0:
                    self.teacher.eval()
                    with torch.no_grad():
                        teacher_logits = self.teacher(x)
                        if isinstance(teacher_logits, tuple):
                            teacher_logits = teacher_logits[0]
                    loss = loss + self.lwf_weight * lwf_distillation_loss(
                        logits, teacher_logits, self.lwf_temperature)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss encountered in domain {domain_id}")
                loss.backward()
                if self.grad_clip_norm is not None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.optimizer.step()
                if isinstance(self.regularizer, SynapticIntelligence):
                    self.regularizer.update_after_step(self.model)
                loss_sum += float(loss.detach())
                correct += int((logits.detach().argmax(-1) == y).sum())
                count += int(y.numel())
                steps += 1
                # Add only current-domain data, not replay samples, to the reservoir.
                if self.replay is not None:
                    current_count = batch[1].shape[0]
                    self.replay.add_batch(x[:current_count].detach(), y[:current_count].detach(), domain_id)
        result: dict[str, float | int] = {
            "domain": domain_id, "loss": loss_sum / max(1, steps),
            "train_accuracy": correct / max(1, count), "steps": steps,
        }
        self.history.append(result)
        return result

    def update_teacher(self) -> None:
        """Snapshot the current model for Learning without Forgetting."""
        if self.lwf_weight <= 0:
            return
        self.teacher = copy.deepcopy(self.model).to(self.device).eval()
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)

    def consolidate_task(self, train_loader: DataLoader, domain_id: int,
                         max_batches: int | None = None) -> None:
        if isinstance(self.regularizer, OnlineEWC):
            self.regularizer.end_task(self.model, train_loader, self.device, domain_id, max_batches)
        elif isinstance(self.regularizer, MASImportance):
            self.regularizer.estimate(self.model, train_loader, self.device, max_batches)
        elif isinstance(self.regularizer, SynapticIntelligence):
            self.regularizer.end_task(self.model)

    def save_checkpoint(self, path: str | Path, extra: dict[str, Any] | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {"model": self.model.state_dict(),
                                  "optimizer": self.optimizer.state_dict(),
                                  "history": self.history,
                                  "extra": extra or {}}
        if self.regularizer is not None and hasattr(self.regularizer, "state_dict"):
            payload["regularizer"] = self.regularizer.state_dict()
            payload["regularizer_type"] = self.regularizer.__class__.__name__
        if self.teacher is not None:
            payload["teacher"] = self.teacher.state_dict()
        if self.replay is not None:
            payload["replay"] = self.replay.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path: str | Path, strict: bool = True) -> dict[str, Any]:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["model"], strict=strict)
        self.optimizer.load_state_dict(payload["optimizer"])
        self.history = payload.get("history", [])
        if self.regularizer is not None and "regularizer" in payload:
            if payload.get("regularizer_type") not in (None, self.regularizer.__class__.__name__):
                raise ValueError("Checkpoint regularizer type does not match the configured regularizer")
            self.regularizer.load_state_dict(payload["regularizer"])
        if "teacher" in payload:
            self.teacher = copy.deepcopy(self.model).to(self.device).eval()
            self.teacher.load_state_dict(payload["teacher"])
            for parameter in self.teacher.parameters():
                parameter.requires_grad_(False)
        if self.replay is not None and "replay" in payload:
            self.replay.load_state_dict(payload["replay"])
        return payload.get("extra", {})



def sync_optimizer_with_model(optimizer: torch.optim.Optimizer, model: nn.Module) -> None:
    """Drop removed parameters and add newly allocated model parameters.

    Used when AdaptiveRouterKAN replaces its final router layer while expanding
    the expert bank. New parameters inherit the first optimizer group's settings.
    """
    live = {id(parameter): parameter for parameter in model.parameters()}
    known = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    for group in optimizer.param_groups:
        group["params"] = [parameter for parameter in group["params"] if id(parameter) in live]
    stale_state = [parameter for parameter in optimizer.state if id(parameter) not in live]
    for parameter in stale_state:
        del optimizer.state[parameter]
    new_parameters = [parameter for key, parameter in live.items() if key not in known]
    if new_parameters:
        optimizer.param_groups[0]["params"].extend(new_parameters)


def run_domain_sequence(model: nn.Module, train_loaders: list[DataLoader],
                        test_loaders: list[DataLoader], optimizer: torch.optim.Optimizer,
                        epochs: int = 1, device: torch.device | str = "cpu",
                        ewc_strength: float = 0.0, replay_capacity: int = 0,
                        replay_batch_size: int = 0, checkpoint_dir: str | Path | None = None,
                        loss_fn: nn.Module | None = None, regularizer: Any | None = None,
                        lwf_weight: float = 0.0, lwf_temperature: float = 2.0
                        ) -> tuple[ContinualTrainer, ContinualMetrics]:
    if len(train_loaders) != len(test_loaders) or not train_loaders:
        raise ValueError("train_loaders and test_loaders must have equal non-zero length")
    ewc = OnlineEWC(model, strength=ewc_strength) if ewc_strength > 0 and regularizer is None else None
    replay = ReplayBuffer(replay_capacity) if replay_capacity > 0 else None
    trainer = ContinualTrainer(model, optimizer, device, ewc=ewc, replay=replay,
                               regularizer=regularizer, loss_fn=loss_fn,
                               lwf_weight=lwf_weight, lwf_temperature=lwf_temperature)
    metrics = ContinualMetrics(len(train_loaders))
    for domain_id, train_loader in enumerate(train_loaders):
        if hasattr(model, "add_domain") and getattr(model, "num_domains", 0) and domain_id >= model.num_domains:
            model.add_domain()
            sync_optimizer_with_model(optimizer, model)
        elif hasattr(model, "num_tasks") and domain_id >= model.num_tasks:
            model.add_domain()
            sync_optimizer_with_model(optimizer, model)
        if isinstance(trainer.regularizer, SynapticIntelligence):
            trainer.regularizer.begin_task(model)
        trainer.train_domain(domain_id, train_loader, epochs, replay_batch_size)
        trainer.consolidate_task(train_loader, domain_id)
        for eval_id, test_loader in enumerate(test_loaders[:domain_id + 1]):
            score = evaluate(model, test_loader, device, task_id=eval_id)
            metrics.update(domain_id, eval_id, score)
        trainer.update_teacher()
        if checkpoint_dir is not None:
            trainer.save_checkpoint(Path(checkpoint_dir) / f"after_domain_{domain_id}.pt",
                                    {"trained_through": domain_id})
    return trainer, metrics
