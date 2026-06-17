import pytest
import torch

from gptff.model.mlp import GatedMLP, MLP


def test_mlp_returns_expected_shape_and_zero_init_output():
    mlp = MLP(4, 2, hidden_dims=(8, 8), zero_init_output=True)
    x = torch.randn(3, 4)

    y = mlp(x)

    assert y.shape == (3, 2)
    assert torch.allclose(y, torch.zeros_like(y))


def test_gated_mlp_returns_finite_features():
    mlp = GatedMLP(4, 6)
    x = torch.randn(3, 4)

    y = mlp(x)

    assert y.shape == (3, 6)
    assert torch.isfinite(y).all()


def test_gated_mlp_zero_init_output():
    mlp = GatedMLP(4, 6, zero_init_output=True)
    x = torch.randn(3, 4)

    y = mlp(x)

    assert y.shape == (3, 6)
    assert torch.allclose(y, torch.zeros_like(y))


def test_mlp_rejects_invalid_dropout():
    with pytest.raises(ValueError, match="dropout"):
        MLP(4, 2, dropout=1.0)

    with pytest.raises(ValueError, match="dropout"):
        GatedMLP(4, 2, dropout=-0.1)
