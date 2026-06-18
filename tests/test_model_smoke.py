import pytest
import torch
from pymatgen.core import Lattice, Structure

from gptff.graph import CrystalGraphBatch, CrystalGraphConverter
from gptff.inference import predict_energy_forces_stress
from gptff.model import GPTFF, GPTFFConfig


def test_model_forward_and_efs_with_smooth_radial_basis():
    cfg = GPTFFConfig(
        atom_feature_dim=8,
        edge_feature_dim=8,
        num_interaction_blocks=1,
        num_radial=8,
        num_angular=4,
        radial_cutoff=3.0,
        angle_cutoff=3.0,
        cutoff_coeff=5,
    )
    structure = Structure(
        Lattice.cubic(3.0),
        ["Na", "Cl"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    graph = CrystalGraphConverter(radial_cutoff=3.0, angle_cutoff=3.0).convert(structure)
    batch = CrystalGraphBatch.from_graphs([graph])

    energy, forces, stress = predict_energy_forces_stress(
        GPTFF(cfg),
        batch,
        create_graph=True,
        compute_stress=True,
    )

    assert energy.shape == (1,)
    assert forces.shape == (2, 3)
    assert stress.shape == (1, 3, 3)
    assert torch.isfinite(energy).all()
    assert torch.isfinite(forces).all()
    assert torch.isfinite(stress).all()


def test_model_forward_and_efs_with_atom_attention():
    cfg = GPTFFConfig(
        atom_feature_dim=8,
        edge_feature_dim=8,
        num_interaction_blocks=1,
        num_radial=8,
        num_angular=4,
        radial_cutoff=3.0,
        angle_cutoff=3.0,
        cutoff_coeff=5,
        atom_attention={
            "enabled": True,
            "num_heads": 2,
            "dropout": 0.0,
            "use_ffn": True,
        },
    )
    structure = Structure(
        Lattice.cubic(3.0),
        ["Na", "Cl"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    graph = CrystalGraphConverter(radial_cutoff=3.0, angle_cutoff=3.0).convert(structure)
    batch = CrystalGraphBatch.from_graphs([graph])

    energy, forces, stress = predict_energy_forces_stress(
        GPTFF(cfg),
        batch,
        create_graph=True,
        compute_stress=True,
    )

    assert energy.shape == (1,)
    assert forces.shape == (2, 3)
    assert stress.shape == (1, 3, 3)
    assert torch.isfinite(energy).all()
    assert torch.isfinite(forces).all()
    assert torch.isfinite(stress).all()


def test_model_rejects_radial_cutoff_mismatch():
    cfg = GPTFFConfig(
        atom_feature_dim=8,
        edge_feature_dim=8,
        num_interaction_blocks=1,
        num_radial=8,
        num_angular=4,
        radial_cutoff=3.0,
        angle_cutoff=3.0,
    )
    structure = Structure(
        Lattice.cubic(3.0),
        ["Na", "Cl"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    graph = CrystalGraphConverter(radial_cutoff=2.9, angle_cutoff=3.0).convert(structure)
    batch = CrystalGraphBatch.from_graphs([graph]).with_geometry()

    with pytest.raises(ValueError, match="Graph radial_cutoff 2.9"):
        GPTFF(cfg)(batch)


def test_model_rejects_angle_cutoff_mismatch():
    cfg = GPTFFConfig(
        atom_feature_dim=8,
        edge_feature_dim=8,
        num_interaction_blocks=1,
        num_radial=8,
        num_angular=4,
        radial_cutoff=3.0,
        angle_cutoff=3.0,
    )
    structure = Structure(
        Lattice.cubic(3.0),
        ["Na", "Cl"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    graph = CrystalGraphConverter(radial_cutoff=3.0, angle_cutoff=2.5).convert(structure)
    batch = CrystalGraphBatch.from_graphs([graph]).with_geometry()

    with pytest.raises(ValueError, match="Graph angle_cutoff 2.5"):
        GPTFF(cfg)(batch)


def test_prediction_uses_model_total_energy():
    structure = Structure(
        Lattice.cubic(3.0),
        ["Na", "Cl"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    graph = CrystalGraphConverter(radial_cutoff=3.0, angle_cutoff=3.0).convert(structure)
    batch = CrystalGraphBatch.from_graphs([graph])

    energy, forces, stress = predict_energy_forces_stress(
        GeometryEnergyModel(),
        batch,
        create_graph=True,
        compute_stress=True,
    )

    assert torch.allclose(energy, torch.tensor([4.5]))
    assert forces.shape == (2, 3)
    assert stress.shape == (1, 3, 3)


def test_prediction_can_skip_stress_derivatives():
    structure = Structure(
        Lattice.cubic(3.0),
        ["Na", "Cl"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    graph = CrystalGraphConverter(radial_cutoff=3.0, angle_cutoff=3.0).convert(structure)
    batch = CrystalGraphBatch.from_graphs([graph])

    energy, forces, stress = predict_energy_forces_stress(
        GeometryEnergyModel(),
        batch,
        create_graph=False,
        compute_stress=False,
    )

    assert torch.allclose(energy, torch.tensor([4.5]))
    assert forces.shape == (2, 3)
    assert stress is None


def test_model_force_is_smooth_across_linear_triplet():
    torch.manual_seed(7)
    model = GPTFF(
        GPTFFConfig(
            atom_feature_dim=8,
            edge_feature_dim=8,
            num_interaction_blocks=1,
            num_radial=8,
            num_angular=9,
            radial_cutoff=2.2,
            angle_cutoff=1.5,
        )
    ).double()
    model.eval()

    force_negative = _linear_triplet_force(model, -1e-3)
    force_collinear = _linear_triplet_force(model, 0.0)
    force_small = _linear_triplet_force(model, 1e-3)
    force_large = _linear_triplet_force(model, 2e-3)

    assert torch.isfinite(
        torch.stack([force_negative, force_collinear, force_small, force_large])
    ).all()
    assert force_collinear.item() == pytest.approx(0.0, abs=1e-10)
    assert force_negative.item() == pytest.approx(-force_small.item(), rel=1e-5, abs=1e-10)
    assert force_small.item() == pytest.approx(0.5 * force_large.item(), rel=1e-3, abs=1e-10)


class GeometryEnergyModel(torch.nn.Module):
    def forward(self, graph):
        site_energy = graph.positions.sum(dim=1, keepdim=True)
        energy = torch.zeros(
            (graph.num_atoms.shape[0], 1),
            dtype=site_energy.dtype,
            device=site_energy.device,
        )
        energy = torch.index_add(energy, 0, graph.atom_batch, site_energy)
        return energy + graph.strain.sum(dim=(1, 2), keepdim=False).unsqueeze(-1)


def _linear_triplet_force(model, transverse_displacement):
    structure = Structure(
        Lattice.cubic(10.0),
        ["Si", "O", "O"],
        [
            [0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [1.0, transverse_displacement, 0.0],
        ],
        coords_are_cartesian=True,
    )
    graph = CrystalGraphConverter(radial_cutoff=2.2, angle_cutoff=1.5).convert(structure)
    batch = CrystalGraphBatch.from_graphs([graph])
    batch = type(batch)(
        **{
            name: value.double()
            if isinstance(value, torch.Tensor) and value.is_floating_point()
            else value
            for name, value in batch.__dict__.items()
        }
    )
    _, forces, _ = predict_energy_forces_stress(
        model,
        batch,
        create_graph=False,
        compute_stress=False,
    )
    return forces[2, 1]
