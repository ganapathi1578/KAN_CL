"""Univariate basis families used by KAN edges.

Every basis maps a tensor [batch, input_features] to [batch, input_features, n_basis].
The compact-support B-spline implementation uses Cox-de Boor recursion and clamped
uniform knots. Inputs are smoothly bounded with tanh so the basis remains defined
for hidden activations outside the nominal grid interval.
"""
from __future__ import annotations

import math
from typing import Callable

import torch
from torch import Tensor, nn

from kan_cl.registry import BASIS_FUNCTIONS


def _bounded(x: Tensor) -> Tensor:
    return torch.tanh(x)


def bspline_basis(x: Tensor, n_basis: int = 8, degree: int = 3) -> Tensor:
    if n_basis < degree + 1:
        raise ValueError(f"n_basis must be >= degree + 1 (got {n_basis}, {degree})")
    z = (_bounded(x) + 1.0) * 0.5
    # Open/clamped uniform knot vector, length n_basis + degree + 1.
    interior_count = n_basis - degree - 1
    if interior_count > 0:
        interior = torch.linspace(0, 1, interior_count + 2, device=x.device, dtype=x.dtype)[1:-1]
    else:
        interior = torch.empty(0, device=x.device, dtype=x.dtype)
    knots = torch.cat((torch.zeros(degree + 1, device=x.device, dtype=x.dtype), interior,
                       torch.ones(degree + 1, device=x.device, dtype=x.dtype)))
    # Avoid the right-open endpoint issue in degree-zero indicator functions.
    z = z.clamp(0.0, 1.0 - torch.finfo(x.dtype).eps * 4)
    left = knots[:-1].view(*([1] * z.ndim), -1)
    right = knots[1:].view(*([1] * z.ndim), -1)
    values = ((z.unsqueeze(-1) >= left) & (z.unsqueeze(-1) < right)).to(x.dtype)
    for order in range(1, degree + 1):
        count = values.shape[-1] - 1
        t_i = knots[:count]
        t_i_order = knots[order:order + count]
        t_i1 = knots[1:count + 1]
        t_i_order1 = knots[order + 1:order + 1 + count]
        denom_left = t_i_order - t_i
        denom_right = t_i_order1 - t_i1
        dl = torch.where(denom_left.abs() > 0, denom_left, torch.ones_like(denom_left))
        dr = torch.where(denom_right.abs() > 0, denom_right, torch.ones_like(denom_right))
        x_expanded = z.unsqueeze(-1)
        left_term = ((x_expanded - t_i) / dl) * values[..., :count]
        right_term = ((t_i_order1 - x_expanded) / dr) * values[..., 1:count + 1]
        left_term = torch.where(denom_left > 0, left_term, torch.zeros_like(left_term))
        right_term = torch.where(denom_right > 0, right_term, torch.zeros_like(right_term))
        values = left_term + right_term
    return values[..., :n_basis]


def rbf_basis(x: Tensor, n_basis: int = 8, width: float = 0.35) -> Tensor:
    z = _bounded(x).unsqueeze(-1)
    centers = torch.linspace(-1.0, 1.0, n_basis, device=x.device, dtype=x.dtype)
    return torch.exp(-0.5 * ((z - centers) / max(width, 1e-4)).square())


def chebyshev_basis(x: Tensor, n_basis: int = 8) -> Tensor:
    z = _bounded(x)
    values = [torch.ones_like(z)]
    if n_basis > 1:
        values.append(z)
    for _ in range(2, n_basis):
        values.append(2.0 * z * values[-1] - values[-2])
    return torch.stack(values, dim=-1)


def fourier_basis(x: Tensor, n_basis: int = 9) -> Tensor:
    z = _bounded(x)
    values = [torch.ones_like(z)]
    freq = 1
    while len(values) < n_basis:
        values.append(torch.sin(math.pi * freq * z))
        if len(values) < n_basis:
            values.append(torch.cos(math.pi * freq * z))
        freq += 1
    return torch.stack(values, dim=-1)


def wavelet_basis(x: Tensor, n_basis: int = 8, width: float = 0.35) -> Tensor:
    z = _bounded(x).unsqueeze(-1)
    centers = torch.linspace(-1.0, 1.0, n_basis, device=x.device, dtype=x.dtype)
    u = (z - centers) / max(width, 1e-4)
    # Mexican-hat / Ricker wavelet.
    return (1.0 - u.square()) * torch.exp(-0.5 * u.square())


def rational_basis(x: Tensor, n_basis: int = 8) -> Tensor:
    z = _bounded(x)
    values = []
    for power in range(n_basis):
        p = z.pow(power)
        values.append(p / (1.0 + z.square()))
    return torch.stack(values, dim=-1)


@BASIS_FUNCTIONS.register("bspline")
class BSplineBasis(nn.Module):
    def __init__(self, n_basis: int = 8, degree: int = 3, **_: object) -> None:
        super().__init__()
        self.n_basis, self.degree = n_basis, degree

    def forward(self, x: Tensor) -> Tensor:
        return bspline_basis(x, self.n_basis, self.degree)


@BASIS_FUNCTIONS.register("rbf")
class RBFBasis(nn.Module):
    """Gaussian RBFs with trainable centers and positive trainable widths."""
    def __init__(self, n_basis: int = 8, width: float = 0.35,
                 learnable: bool = True, **_: object) -> None:
        super().__init__()
        if width <= 0:
            raise ValueError("RBF width must be positive")
        centers = torch.linspace(-1.0, 1.0, n_basis)
        inverse_softplus = math.log(math.expm1(width))
        self.centers = nn.Parameter(centers, requires_grad=learnable)
        self.log_widths = nn.Parameter(torch.full((n_basis,), inverse_softplus),
                                        requires_grad=learnable)

    def forward(self, x: Tensor) -> Tensor:
        z = _bounded(x).unsqueeze(-1)
        widths = torch.nn.functional.softplus(self.log_widths).clamp_min(1e-4)
        return torch.exp(-0.5 * ((z - self.centers) / widths).square())


@BASIS_FUNCTIONS.register("chebyshev")
class ChebyshevBasis(nn.Module):
    def __init__(self, n_basis: int = 8, **_: object) -> None:
        super().__init__()
        self.n_basis = n_basis

    def forward(self, x: Tensor) -> Tensor:
        return chebyshev_basis(x, self.n_basis)


@BASIS_FUNCTIONS.register("fourier")
class FourierBasis(nn.Module):
    def __init__(self, n_basis: int = 9, **_: object) -> None:
        super().__init__()
        self.n_basis = n_basis

    def forward(self, x: Tensor) -> Tensor:
        return fourier_basis(x, self.n_basis)


@BASIS_FUNCTIONS.register("wavelet")
class WaveletBasis(nn.Module):
    def __init__(self, n_basis: int = 8, width: float = 0.35, **_: object) -> None:
        super().__init__()
        self.n_basis, self.width = n_basis, width

    def forward(self, x: Tensor) -> Tensor:
        return wavelet_basis(x, self.n_basis, self.width)


@BASIS_FUNCTIONS.register("rational")
class RationalBasis(nn.Module):
    def __init__(self, n_basis: int = 8, **_: object) -> None:
        super().__init__()
        self.n_basis = n_basis

    def forward(self, x: Tensor) -> Tensor:
        return rational_basis(x, self.n_basis)


BASIS_BUILDERS: dict[str, Callable[..., nn.Module]] = {
    name: BASIS_FUNCTIONS.get(name) for name in BASIS_FUNCTIONS.names()
}
