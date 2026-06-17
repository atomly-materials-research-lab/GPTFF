import torch

from gptff.model.basis import PolynomialCutoff, RadialBesselBasis


def test_radial_bessel_basis_shape():
    basis = RadialBesselBasis(num_radial=8, cutoff=5.0, cutoff_coeff=5)
    distances = torch.tensor([0.5, 1.5, 2.5])

    out = basis(distances)

    assert out.shape == (3, 8)


def test_radial_bessel_basis_is_zero_at_and_beyond_cutoff():
    basis = RadialBesselBasis(num_radial=8, cutoff=5.0, cutoff_coeff=5)
    distances = torch.tensor([5.0, 5.1, 8.0])

    out = basis(distances)

    assert torch.allclose(out, torch.zeros_like(out), atol=1e-7)


def test_radial_bessel_basis_has_no_dead_zero_channel():
    basis = RadialBesselBasis(num_radial=8, cutoff=5.0, cutoff_coeff=5)
    distances = torch.tensor([0.37, 1.13, 2.41, 3.72])

    out = basis(distances)

    assert torch.all(out.abs().sum(dim=0) > 0)


def test_polynomial_cutoff_decays_towards_cutoff():
    cutoff = PolynomialCutoff(cutoff=5.0, cutoff_coeff=5)
    distances = torch.tensor([0.0, 2.5, 4.5, 5.0])

    values = cutoff(distances)

    assert values[0] == 1.0
    assert values[0] > values[1] > values[2] > values[3]
    assert values[3] == 0.0


def test_radial_bessel_basis_has_finite_distance_gradients():
    basis = RadialBesselBasis(num_radial=8, cutoff=5.0, cutoff_coeff=5)
    distances = torch.tensor([0.5, 1.5, 2.5, 4.5], requires_grad=True)

    loss = basis(distances).sum()
    loss.backward()

    assert torch.isfinite(distances.grad).all()
