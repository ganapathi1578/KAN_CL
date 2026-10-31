import pytest
import torch

from kan_cl.models.basis import (
    bspline_basis, chebyshev_basis, fourier_basis, rational_basis, rbf_basis, wavelet_basis,
)
from kan_cl.models.layers import KANLinear, KANNetwork


@pytest.mark.parametrize("name,fn", [
    ("bspline", lambda x: bspline_basis(x, 8, 3)),
    ("rbf", lambda x: rbf_basis(x, 8)),
    ("chebyshev", lambda x: chebyshev_basis(x, 8)),
    ("fourier", lambda x: fourier_basis(x, 9)),
    ("wavelet", lambda x: wavelet_basis(x, 8)),
    ("rational", lambda x: rational_basis(x, 8)),
])
def test_basis_shapes_and_finite(name, fn):
    del name
    x = torch.randn(5, 4, requires_grad=True)
    y = fn(x)
    assert y.shape == (5, 4, 8) or y.shape == (5, 4, 9)
    assert torch.isfinite(y).all()
    y.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_bspline_local_support_and_partition():
    x = torch.linspace(-1, 1, 100).view(-1, 1)
    values = bspline_basis(x, n_basis=8, degree=3)
    assert values.shape == (100, 1, 8)
    assert (values >= -1e-6).all()
    # Cubic B-spline bases partition unity over the interior of the knot domain.
    sums = values.sum(-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=2e-4)
    # Compact support means a point activates no more than degree+1 basis functions.
    assert (values.squeeze(1).abs() > 1e-7).sum(-1).max() <= 4


def test_kan_linear_gradients():
    layer = KANLinear(4, 3, n_basis=7, degree=3)
    x = torch.randn(6, 4)
    output = layer(x)
    assert output.shape == (6, 3)
    output.square().mean().backward()
    assert layer.coefficients.grad is not None
    assert torch.isfinite(layer.coefficients.grad).all()


def test_kan_network_output_shapes():
    model = KANNetwork(5, [12, 8], 4, basis="rbf", n_basis=6)
    assert model(torch.randn(7, 5)).shape == (7, 4)
    assert model(torch.randn(2, 3, 5)).shape == (2, 3, 4)


def test_invalid_bspline_configuration():
    with pytest.raises(ValueError):
        bspline_basis(torch.randn(2, 1), n_basis=3, degree=3)


def test_rbf_kan_learns_centers_and_widths():
    layer = KANLinear(3, 2, basis="rbf", n_basis=6)
    output = layer(torch.randn(5, 3)).square().mean()
    output.backward()
    assert layer.basis.centers.grad is not None
    assert layer.basis.log_widths.grad is not None
