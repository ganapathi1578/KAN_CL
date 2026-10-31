"""Continual-learning KAN architectures proposed in the project brief."""
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from kan_cl.models.layers import KANNetwork
from kan_cl.registry import MODELS


@MODELS.register("kan")
class VanillaKAN(KANNetwork):
    pass


@MODELS.register("protected_spline_kan")
class ProtectedSplineKAN(KANNetwork):
    """B-spline KAN intended to be combined with per-coefficient EWC protection."""
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], output_dim: int,
                 basis: str = "bspline", **kwargs: object) -> None:
        if basis != "bspline":
            raise ValueError("ProtectedSplineKAN requires basis='bspline'")
        super().__init__(input_dim, hidden_dims, output_dim, basis="bspline", **kwargs)


@MODELS.register("adaptive_router_kan")
class AdaptiveRouterKAN(nn.Module):
    """A bank of domain-specialized KANs with a learned input-driven router.

    During sequential training, pass `domain_labels` to route each sample to its
    known source-domain expert while training the router with `router_logits`.
    At test time omit labels; the learned router selects the expert without task IDs.
    `add_domain()` expands both the expert bank and router output head.
    """
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], output_dim: int,
                 num_domains: int = 2, basis: str = "bspline", n_basis: int = 8,
                 degree: int = 3, router_hidden_dim: int = 64, dropout: float = 0.0) -> None:
        super().__init__()
        if num_domains < 1:
            raise ValueError("num_domains must be at least one")
        self.input_dim, self.output_dim = input_dim, output_dim
        self.hidden_dims = tuple(hidden_dims)
        self.experts = nn.ModuleList([
            KANNetwork(input_dim, hidden_dims, output_dim, basis=basis,
                       n_basis=n_basis, degree=degree, dropout=dropout)
            for _ in range(num_domains)
        ])
        self.router = nn.Sequential(nn.Linear(input_dim, router_hidden_dim), nn.Tanh(),
                                    nn.Linear(router_hidden_dim, num_domains))
        self.basis_name = basis
        self.n_basis, self.degree, self.dropout = n_basis, degree, dropout

    @property
    def num_domains(self) -> int:
        return len(self.experts)

    def add_domain(self, copy_from: int | None = None) -> int:
        """Append a new expert and router class; return its new domain index."""
        if copy_from is not None and not 0 <= copy_from < self.num_domains:
            raise IndexError("copy_from is outside the current expert bank")
        new_expert = KANNetwork(self.input_dim, self.hidden_dims, self.output_dim,
                                basis=self.basis_name, n_basis=self.n_basis,
                                degree=self.degree, dropout=self.dropout)
        if copy_from is not None:
            new_expert.load_state_dict(self.experts[copy_from].state_dict())
        reference = next(self.parameters())
        new_expert = new_expert.to(device=reference.device, dtype=reference.dtype)
        self.experts.append(new_expert)
        old_head = self.router[-1]
        assert isinstance(old_head, nn.Linear)
        new_head = nn.Linear(old_head.in_features, old_head.out_features + 1,
                             device=old_head.weight.device, dtype=old_head.weight.dtype)
        with torch.no_grad():
            new_head.weight[:old_head.out_features].copy_(old_head.weight)
            new_head.bias[:old_head.out_features].copy_(old_head.bias)
            nn.init.normal_(new_head.weight[-1:], std=0.02)
            new_head.bias[-1:].zero_()
        self.router[-1] = new_head
        return self.num_domains - 1

    def forward(self, x: Tensor, domain_labels: Tensor | None = None,
                return_aux: bool = False) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        router_logits = self.router(x)
        soft_gates = F.softmax(router_logits, dim=-1)
        if domain_labels is not None:
            labels = domain_labels.to(device=x.device, dtype=torch.long).view(-1)
            if labels.numel() == 1 and x.shape[0] != 1:
                labels = labels.expand(x.shape[0])
            if labels.numel() != x.shape[0]:
                raise ValueError("domain_labels must have one label per sample")
            if labels.min() < 0 or labels.max() >= self.num_domains:
                raise ValueError("domain label is outside the expert bank")
            gates = F.one_hot(labels, num_classes=self.num_domains).to(x.dtype)
        else:
            gates = soft_gates
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        output = torch.sum(expert_outputs * gates.unsqueeze(-1), dim=1)
        if return_aux:
            return output, {"router_logits": router_logits, "gates": gates,
                            "soft_gates": soft_gates, "expert_outputs": expert_outputs}
        return output


@MODELS.register("hybrid_moe_kan")
class HybridBasisMoEKAN(nn.Module):
    """Mixture of KAN experts, each using a different basis family."""
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], output_dim: int,
                 expert_bases: Sequence[str] = ("bspline", "rbf", "chebyshev"),
                 n_basis: int = 8, degree: int = 3, router_hidden_dim: int = 64,
                 top_k: int | None = None, dropout: float = 0.0) -> None:
        super().__init__()
        if len(expert_bases) < 2:
            raise ValueError("HybridBasisMoEKAN requires at least two basis experts")
        if top_k is not None and not 1 <= top_k <= len(expert_bases):
            raise ValueError("top_k must be in [1, number of experts]")
        self.input_dim, self.output_dim = input_dim, output_dim
        self.expert_bases = tuple(expert_bases)
        self.experts = nn.ModuleList([
            KANNetwork(input_dim, hidden_dims, output_dim, basis=b,
                       n_basis=n_basis, degree=degree, dropout=dropout)
            for b in expert_bases
        ])
        self.router = nn.Sequential(nn.Linear(input_dim, router_hidden_dim), nn.GELU(),
                                    nn.Linear(router_hidden_dim, len(expert_bases)))
        self.top_k = top_k

    def forward(self, x: Tensor, return_aux: bool = False) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        logits = self.router(x)
        gates = F.softmax(logits, dim=-1)
        if self.top_k is not None and self.top_k < len(self.experts):
            values, indices = gates.topk(self.top_k, dim=-1)
            sparse = torch.zeros_like(gates).scatter(-1, indices, values)
            gates = sparse / sparse.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        prediction = torch.sum(outputs * gates.unsqueeze(-1), dim=1)
        if return_aux:
            return prediction, {"router_logits": logits, "gates": gates,
                                "expert_outputs": outputs}
        return prediction

    @staticmethod
    def load_balance_loss(gates: Tensor) -> Tensor:
        """Switch-style load balancing term; minimized by non-collapsed routing."""
        mean_usage = gates.mean(dim=0)
        return gates.shape[-1] * (mean_usage.square().sum())

    @staticmethod
    def diversity_loss(expert_outputs: Tensor) -> Tensor:
        """Penalize correlated expert predictions (safe for batch size >= 2)."""
        if expert_outputs.shape[0] < 2:
            return expert_outputs.new_zeros(())
        flattened = expert_outputs.transpose(0, 1).flatten(start_dim=1)
        normalized = F.normalize(flattened, dim=-1)
        gram = normalized @ normalized.T
        identity = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        return ((gram - identity).square().sum() / max(1, gram.numel() - gram.shape[0]))


class _ProgressiveKANColumn(nn.Module):
    """One task column with lateral KAN connections from all older columns."""
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], output_dim: int,
                 previous_columns: int, basis: str, n_basis: int, degree: int) -> None:
        super().__init__()
        from kan_cl.models.layers import KANLinear

        self.hidden_dims = tuple(hidden_dims)
        self.input_layer = KANLinear(input_dim, hidden_dims[0], basis, n_basis, degree)
        self.hidden_layers = nn.ModuleList([
            KANLinear(hidden_dims[i - 1], hidden_dims[i], basis, n_basis, degree)
            for i in range(1, len(hidden_dims))
        ])
        self.lateral_layers = nn.ModuleList([
            nn.ModuleList([KANLinear(hidden_dims[layer], hidden_dims[layer], basis, n_basis, degree)
                           for _ in range(previous_columns)])
            for layer in range(len(hidden_dims))
        ])
        self.output_layer = KANLinear(hidden_dims[-1], output_dim, basis, n_basis, degree)
        self.output_laterals = nn.ModuleList([
            KANLinear(hidden_dims[-1], output_dim, basis, n_basis, degree)
            for _ in range(previous_columns)
        ])

    def forward(self, x: Tensor, previous_features: list[list[Tensor]]) -> tuple[Tensor, list[Tensor]]:
        features: list[Tensor] = []
        hidden = self.input_layer(x)
        for lateral, previous in zip(self.lateral_layers[0], previous_features):
            hidden = hidden + lateral(previous[0])
        hidden = torch.nn.functional.silu(hidden)
        features.append(hidden)
        for layer_index, layer in enumerate(self.hidden_layers, start=1):
            hidden = layer(hidden)
            for lateral, previous in zip(self.lateral_layers[layer_index], previous_features):
                hidden = hidden + lateral(previous[layer_index])
            hidden = torch.nn.functional.silu(hidden)
            features.append(hidden)
        logits = self.output_layer(hidden)
        for lateral, previous in zip(self.output_laterals, previous_features):
            logits = logits + lateral(previous[-1])
        return logits, features


@MODELS.register("progressive_kan")
class ProgressiveKAN(nn.Module):
    """Progressive KAN baseline with frozen old columns and lateral KAN links.

    Like Progressive Neural Networks, this baseline is task/domain-ID aware at
    evaluation: call `model(x, task_id=domain_id)`. If task_id is omitted, the
    newest column is selected. It is included as a task-aware comparison baseline,
    not as a domain-agnostic solution.
    """
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], output_dim: int,
                 num_tasks: int = 1, basis: str = "bspline", n_basis: int = 8,
                 degree: int = 3) -> None:
        super().__init__()
        if not hidden_dims or num_tasks < 1:
            raise ValueError("hidden_dims must be non-empty and num_tasks >= 1")
        self.input_dim, self.output_dim = input_dim, output_dim
        self.hidden_dims = tuple(hidden_dims)
        self.basis, self.n_basis, self.degree = basis, n_basis, degree
        self.columns = nn.ModuleList()
        for _ in range(num_tasks):
            self.add_domain()

    @property
    def num_tasks(self) -> int:
        return len(self.columns)

    def add_domain(self) -> int:
        column = _ProgressiveKANColumn(self.input_dim, self.hidden_dims, self.output_dim,
                                       previous_columns=self.num_tasks, basis=self.basis,
                                       n_basis=self.n_basis, degree=self.degree)
        reference = next(self.parameters(), None)
        if reference is not None:
            column = column.to(device=reference.device, dtype=reference.dtype)
        for old_column in self.columns:
            for parameter in old_column.parameters():
                parameter.requires_grad_(False)
        self.columns.append(column)
        return self.num_tasks - 1

    def forward(self, x: Tensor, task_id: int | None = None) -> Tensor:
        task_id = self.num_tasks - 1 if task_id is None else int(task_id)
        if not 0 <= task_id < self.num_tasks:
            raise ValueError(f"task_id must be in [0, {self.num_tasks - 1}]")
        previous_features: list[list[Tensor]] = []
        logits: Tensor | None = None
        for column_id in range(task_id + 1):
            column = self.columns[column_id]
            if column_id < task_id:
                with torch.no_grad():
                    old_logits, features = column(x, previous_features)
            else:
                old_logits, features = column(x, previous_features)
            previous_features.append(features)
            logits = old_logits
        assert logits is not None
        return logits


@MODELS.register("fastkan")
class FastKAN(KANNetwork):
    """RBF-basis KAN variant (Gaussian expansion)."""
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], output_dim: int,
                 **kwargs: object) -> None:
        kwargs.pop("basis", None)
        super().__init__(input_dim, hidden_dims, output_dim, basis="rbf", **kwargs)


@MODELS.register("kac")
class KACClassifier(KANNetwork):
    """RBF-KAN classification head inspired by the KAC basis substitution."""
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], output_dim: int,
                 **kwargs: object) -> None:
        kwargs.pop("basis", None)
        super().__init__(input_dim, hidden_dims, output_dim, basis="rbf", **kwargs)
