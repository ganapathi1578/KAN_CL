"""Small CPU-side reservoir replay buffer."""
from __future__ import annotations

import random
import torch
from torch import Tensor


class ReplayBuffer:
    """Reservoir sampler storing input, class label, and source domain ID."""
    def __init__(self, capacity: int, seed: int = 0) -> None:
        if capacity < 0:
            raise ValueError("capacity cannot be negative")
        self.capacity = capacity
        self._rng = random.Random(seed)
        self._seen = 0
        self._x: list[Tensor] = []
        self._y: list[int] = []
        self._domain: list[int] = []

    def __len__(self) -> int:
        return len(self._y)

    @torch.no_grad()
    def add_batch(self, x: Tensor, y: Tensor, domain_id: int | Tensor = 0) -> None:
        if self.capacity == 0:
            return
        x_cpu = x.detach().to("cpu")
        y_cpu = y.detach().to("cpu").long().view(-1)
        domains = torch.as_tensor(domain_id).long().view(-1)
        if domains.numel() == 1:
            domains = domains.expand(y_cpu.numel())
        if x_cpu.shape[0] != y_cpu.numel() or domains.numel() != y_cpu.numel():
            raise ValueError("x, y, and domain_id batch sizes must match")
        for sample, label, domain in zip(x_cpu, y_cpu, domains):
            self._seen += 1
            if len(self) < self.capacity:
                index = len(self)
            else:
                index = self._rng.randint(0, self._seen - 1)
                if index >= self.capacity:
                    continue
            if index == len(self):
                self._x.append(sample.clone())
                self._y.append(int(label))
                self._domain.append(int(domain))
            else:
                self._x[index] = sample.clone()
                self._y[index] = int(label)
                self._domain[index] = int(domain)

    def sample(self, batch_size: int, device: torch.device | str = "cpu") -> tuple[Tensor, Tensor, Tensor]:
        if not self._y:
            raise ValueError("Cannot sample from an empty replay buffer")
        count = min(batch_size, len(self))
        indices = self._rng.sample(range(len(self)), count)
        x = torch.stack([self._x[i] for i in indices]).to(device)
        y = torch.tensor([self._y[i] for i in indices], dtype=torch.long, device=device)
        d = torch.tensor([self._domain[i] for i in indices], dtype=torch.long, device=device)
        return x, y, d

    def state_dict(self) -> dict:
        return {"capacity": self.capacity, "seen": self._seen, "x": self._x,
                "y": self._y, "domain": self._domain, "rng_state": self._rng.getstate()}

    def load_state_dict(self, state: dict) -> None:
        self.capacity = int(state["capacity"])
        self._seen = int(state["seen"])
        self._x, self._y, self._domain = state["x"], state["y"], state["domain"]
        self._rng.setstate(state["rng_state"])
