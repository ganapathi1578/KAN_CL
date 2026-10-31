"""KAN layers and composable KAN networks."""
from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn

from kan_cl.registry import MODELS

from kan_cl.models.basis import BASIS_FUNCTIONS


class KANLinear(nn.Module):
    """KAN edge layer: a learnable univariate expansion on every input-output edge.

    Args:
        in_features: Number of input channels.
        out_features: Number of output channels.
        basis: One of bspline/rbf/chebyshev/fourier/wavelet/rational.
        n_basis: Number of basis functions per edge.
        degree: B-spline degree (only used by bspline).
        base_branch: Include a conventional activated linear residual branch.
    """
    def __init__(self, in_features: int, out_features: int, basis: str = "bspline",
                 n_basis: int = 8, degree: int = 3, base_branch: bool = True,
                 bias: bool = True, basis_kwargs: dict | None = None) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        self.in_features, self.out_features = in_features, out_features
        self.basis_name, self.n_basis = basis, n_basis
        self.basis = BASIS_FUNCTIONS.build(basis, n_basis=n_basis, degree=degree,
                                           **(basis_kwargs or {}))
        self.coefficients = nn.Parameter(torch.empty(out_features, in_features, n_basis))
        self.base_branch = base_branch
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features)) if base_branch else None
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.coefficients, mean=0.0,
                        std=1.0 / math.sqrt(self.in_features * self.n_basis))
        if self.base_weight is not None:
            nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(f"Expected last dimension {self.in_features}, got {x.shape[-1]}")
        shape = x.shape[:-1]
        flat = x.reshape(-1, self.in_features)
        basis_values = self.basis(flat)
        # [batch, in, basis] x [out, in, basis] -> [batch, out]
        output = torch.einsum("big,oig->bo", basis_values, self.coefficients)
        if self.base_weight is not None:
            output = output + torch.nn.functional.linear(torch.nn.functional.silu(flat), self.base_weight)
        if self.bias is not None:
            output = output + self.bias
        return output.reshape(*shape, self.out_features)


class KANNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], output_dim: int,
                 basis: str = "bspline", n_basis: int = 8, degree: int = 3,
                 dropout: float = 0.0, base_branch: bool = True) -> None:
        super().__init__()
        dims = [input_dim, *hidden_dims, output_dim]
        if len(dims) < 2:
            raise ValueError("At least input and output dimensions are required")
        layers: list[nn.Module] = []
        for idx, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(KANLinear(in_dim, out_dim, basis=basis, n_basis=n_basis,
                                    degree=degree, base_branch=base_branch))
            if idx < len(dims) - 2:
                layers.append(nn.LayerNorm(out_dim))
                layers.append(nn.SiLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.layers = nn.Sequential(*layers)
        self.input_dim, self.output_dim = input_dim, output_dim
        self.hidden_dims = tuple(hidden_dims)
        self.basis_name = basis

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


@MODELS.register("mlp")
class MLPClassifier(nn.Module):
    """Simple MLP baseline with the same model interface as KANNetwork."""
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], output_dim: int,
                 dropout: float = 0.0) -> None:
        super().__init__()
        dims = [input_dim, *hidden_dims, output_dim]
        layers: list[nn.Module] = []
        for idx, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(a, b))
            if idx < len(dims) - 2:
                layers.extend([nn.LayerNorm(b), nn.GELU()])
                if dropout:
                    layers.append(nn.Dropout(dropout))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)
