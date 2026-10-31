"""Standard continual-learning accuracy matrix and retention metrics."""
from __future__ import annotations

import torch
from torch import Tensor


class ContinualMetrics:
    def __init__(self, num_domains: int) -> None:
        if num_domains < 1:
            raise ValueError("num_domains must be positive")
        self.matrix = torch.full((num_domains, num_domains), float("nan"), dtype=torch.float64)

    def update(self, trained_through: int, evaluated_domain: int, accuracy: float) -> None:
        if not (0 <= trained_through < self.matrix.shape[0] and
                0 <= evaluated_domain < self.matrix.shape[1]):
            raise IndexError("Domain index outside metric matrix")
        self.matrix[trained_through, evaluated_domain] = float(accuracy)

    def summary(self) -> dict[str, float | list[float]]:
        n = self.matrix.shape[0]
        average_by_stage: list[float] = []
        forgetting: list[float] = []
        for stage in range(n):
            row = self.matrix[stage, :stage + 1]
            average_by_stage.append(float(torch.nanmean(row)) if torch.isfinite(row).any() else float("nan"))
        for domain in range(max(0, n - 1)):
            history = self.matrix[domain:, domain]
            finite = history[torch.isfinite(history)]
            if finite.numel() >= 2:
                forgetting.append(float(finite.max() - finite[-1]))
            else:
                forgetting.append(float("nan"))
        final_row = self.matrix[-1]
        final_values = final_row[torch.isfinite(final_row)]
        average_accuracy = float(final_values.mean()) if final_values.numel() else float("nan")
        first_domain_retention = float(final_row[0]) if torch.isfinite(final_row[0]) else float("nan")
        bwt_values = []
        for domain in range(n - 1):
            initial = self.matrix[domain, domain]
            final = self.matrix[-1, domain]
            if torch.isfinite(initial) and torch.isfinite(final):
                bwt_values.append(float(final - initial))
        return {
            "average_accuracy": average_accuracy,
            "average_accuracy_by_stage": average_by_stage,
            "mean_forgetting": (sum(forgetting) / len(forgetting)
                                if forgetting and all(v == v for v in forgetting) else float("nan")),
            "forgetting_by_domain": forgetting,
            "backward_transfer": (sum(bwt_values) / len(bwt_values) if bwt_values else float("nan")),
            "first_domain_retention": first_domain_retention,
        }

    def state_dict(self) -> dict:
        return {"matrix": self.matrix}

    def load_state_dict(self, state: dict) -> None:
        self.matrix = state["matrix"].clone().to(torch.float64)
