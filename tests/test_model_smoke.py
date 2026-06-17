import pytest
import torch
from pymatgen.core import Lattice, Structure

from gptff.graph import CrystalGraphBatch, CrystalGraphConverter
from gptff.inference import predict_energy_forces_stress
from gptff.model import GPTFFNet, GPTFFNetConfig


def test_model_forward_and_efs_with_smooth_radial_basis():
    cfg = GPTFFNetConfig(
        node_feature_len=8,
        edge_feature_len=8,
        n_layers=1,
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
    graph = CrystalGraphConverter(r_cut=3.0, a_cut=3.0).convert(structure)
    batch = CrystalGraphBatch.from_graphs([graph])

    energy, forces, stress = predict_energy_forces_stress(
        GPTFFNet(cfg),
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
    cfg = GPTFFNetConfig(
        node_feature_len=8,
        edge_feature_len=8,
        n_layers=1,
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
    graph = CrystalGraphConverter(r_cut=2.9, a_cut=3.0).convert(structure)
    batch = CrystalGraphBatch.from_graphs([graph]).with_geometry()

    with pytest.raises(ValueError, match="Graph radial_cutoff 2.9"):
        GPTFFNet(cfg)(batch)


def test_model_rejects_angle_cutoff_mismatch():
    cfg = GPTFFNetConfig(
        node_feature_len=8,
        edge_feature_len=8,
        n_layers=1,
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
    graph = CrystalGraphConverter(r_cut=3.0, a_cut=2.5).convert(structure)
    batch = CrystalGraphBatch.from_graphs([graph]).with_geometry()

    with pytest.raises(ValueError, match="Graph angle_cutoff 2.5"):
        GPTFFNet(cfg)(batch)


def test_prediction_uses_model_total_energy():
    structure = Structure(
        Lattice.cubic(3.0),
        ["Na", "Cl"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    graph = CrystalGraphConverter(r_cut=3.0, a_cut=3.0).convert(structure)
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
    graph = CrystalGraphConverter(r_cut=3.0, a_cut=3.0).convert(structure)
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
