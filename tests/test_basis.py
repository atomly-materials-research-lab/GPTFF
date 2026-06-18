import torch

from gptff.model.basis import LegendreAngleBasis, PolynomialCutoff, RadialBesselBasis


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


def test_legendre_angle_basis_shape():
    basis = LegendreAngleBasis(num_angular=4)
    cosines = torch.tensor([-0.5, 0.0, 0.5])

    out = basis(cosines)

    assert out.shape == (3, 4)


def test_legendre_angle_basis_matches_normalized_low_orders():
    basis = LegendreAngleBasis(num_angular=3)
    cosines = torch.tensor([-0.5, 0.0, 0.5], dtype=torch.float64)

    out = basis(cosines)
    polynomials = torch.stack(
        [
            torch.ones_like(cosines),
            cosines,
            0.5 * (3 * cosines.square() - 1),
        ],
        dim=1,
    )
    normalization = torch.sqrt(torch.tensor([1.0, 3.0, 5.0], dtype=torch.float64) / 2)

    assert torch.allclose(out, polynomials * normalization)


def test_legendre_angle_basis_has_finite_endpoint_derivatives():
    basis = LegendreAngleBasis(num_angular=9)
    cosines = torch.tensor([-1.0, 1.0], dtype=torch.float64, requires_grad=True)

    loss = basis(cosines).sum()
    first = torch.autograd.grad(loss, cosines, create_graph=True)[0]
    second = torch.autograd.grad(first.sum(), cosines)[0]

    assert torch.isfinite(first).all()
    assert torch.isfinite(second).all()


def test_legendre_angle_basis_has_smooth_collinear_cartesian_curvature():
    basis = LegendreAngleBasis(num_angular=9)

    def second_derivative(displacement):
        transverse = torch.tensor(displacement, dtype=torch.float64, requires_grad=True)
        cosine = -1.0 / torch.sqrt(1.0 + transverse.square())
        response = basis(cosine).sum()
        first = torch.autograd.grad(response, transverse, create_graph=True)[0]
        return torch.autograd.grad(first, transverse)[0]

    at_collinear = second_derivative(0.0)
    near_collinear = second_derivative(1e-6)

    assert torch.isfinite(at_collinear)
    assert torch.isfinite(near_collinear)
    assert torch.allclose(at_collinear, near_collinear, rtol=1e-8, atol=1e-8)
