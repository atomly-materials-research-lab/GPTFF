from dataclasses import replace

import numpy as np
import pytest
import torch
from pymatgen.core import Lattice, Structure

from gptff.graph import CrystalGraphBatch, CrystalGraphConverter
from gptff.inference import predict_energy_forces_stress
from gptff.model import GPTFFNet, GPTFFNetConfig


def test_prediction_returns_zero_unused_geometry_gradients():
    structure = Structure(
        Lattice.cubic(10.0),
        ["Na", "Cl"],
        [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
        coords_are_cartesian=True,
    )
    graph = CrystalGraphConverter(r_cut=1.0, a_cut=1.0).convert(structure)
    batch = CrystalGraphBatch.from_graphs([graph])
    model = GPTFFNet(
        GPTFFNetConfig(
            node_feature_len=8,
            edge_feature_len=8,
            n_layers=1,
            num_radial=4,
            num_angular=2,
            radial_cutoff=1.0,
            angle_cutoff=1.0,
        )
    )

    energy, forces, stress = predict_energy_forces_stress(
        model,
        batch,
        create_graph=True,
        compute_stress=True,
    )

    assert graph.num_edges == 0
    assert torch.isfinite(energy).all()
    assert torch.allclose(forces, torch.zeros_like(forces))
    assert torch.allclose(stress, torch.zeros_like(stress))


def test_force_matches_position_finite_difference():
    graph = _two_atom_graph()
    batch = CrystalGraphBatch.from_graphs([graph])
    model = EdgeLengthEnergyModel()

    _, forces, _ = predict_energy_forces_stress(
        model,
        batch,
        create_graph=False,
        compute_stress=False,
    )

    atom_idx = 0
    coord_idx = 0
    step = 1e-3
    finite_diff_force = -(
        _energy_with_position_delta(model, graph, atom_idx, coord_idx, step)
        - _energy_with_position_delta(model, graph, atom_idx, coord_idx, -step)
    ) / (2 * step)

    assert forces[atom_idx, coord_idx].item() == pytest.approx(
        finite_diff_force,
        abs=5e-3,
    )


def test_inference_outputs_energy_forces_and_stress_by_default():
    graph = _two_atom_graph()
    batch = CrystalGraphBatch.from_graphs([graph])
    model = EdgeLengthEnergyModel()

    energy, forces, stress = predict_energy_forces_stress(model, batch)

    assert energy is not None
    assert forces is not None
    assert stress is not None
    assert forces.requires_grad is False
    assert stress.requires_grad is False


def test_inference_can_explicitly_skip_stress():
    graph = _two_atom_graph()
    batch = CrystalGraphBatch.from_graphs([graph])
    model = EdgeLengthEnergyModel()

    _, forces, stress = predict_energy_forces_stress(
        model,
        batch,
        compute_stress=False,
    )

    assert forces is not None
    assert stress is None


def test_inference_does_not_retain_autograd_graph_by_default(monkeypatch):
    graph = _two_atom_graph()
    batch = CrystalGraphBatch.from_graphs([graph])
    model = EdgeLengthEnergyModel()
    captured_kwargs = {}
    original_grad = torch.autograd.grad

    def recording_grad(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return original_grad(*args, **kwargs)

    monkeypatch.setattr(torch.autograd, "grad", recording_grad)

    predict_energy_forces_stress(model, batch)

    assert captured_kwargs["create_graph"] is False
    assert captured_kwargs["retain_graph"] is False


def test_stress_matches_strain_finite_difference():
    graph = _two_atom_graph()
    batch = CrystalGraphBatch.from_graphs([graph])
    model = EdgeLengthEnergyModel()

    _, _, stress = predict_energy_forces_stress(
        model,
        batch,
        unit_trans=1.0,
        create_graph=False,
        compute_stress=True,
    )

    strain_i = 0
    strain_j = 0
    step = 1e-3
    volume = np.linalg.det(graph.lattice)
    finite_diff_stress = (
        _energy_with_strain_delta(model, graph, strain_i, strain_j, step)
        - _energy_with_strain_delta(model, graph, strain_i, strain_j, -step)
    ) / (2 * step) / volume

    assert stress[0, strain_i, strain_j].item() == pytest.approx(
        finite_diff_stress,
        rel=5e-3,
        abs=5e-4,
    )


class EdgeLengthEnergyModel(torch.nn.Module):
    def forward(self, graph):
        edge_energy = graph.edge_lengths.pow(2).unsqueeze(-1)
        energy = torch.zeros(
            (graph.num_atoms.shape[0], 1),
            dtype=edge_energy.dtype,
            device=edge_energy.device,
        )
        return torch.index_add(energy, 0, graph.edge_batch, edge_energy)


def _two_atom_graph():
    structure = Structure(
        Lattice.cubic(10.0),
        ["Na", "Cl"],
        [[0.0, 0.0, 0.0], [1.2, 0.3, 0.0]],
        coords_are_cartesian=True,
    )
    return CrystalGraphConverter(r_cut=2.0, a_cut=2.0).convert(structure)


def _energy_with_position_delta(model, graph, atom_idx, coord_idx, delta):
    positions = graph.positions.copy()
    positions[atom_idx, coord_idx] += delta
    shifted_graph = replace(graph, positions=positions.astype(np.float32, copy=False))
    return _energy(model, shifted_graph)


def _energy_with_strain_delta(model, graph, strain_i, strain_j, delta):
    transform = np.eye(3, dtype=np.float32)
    transform[strain_i, strain_j] += delta
    strained_graph = replace(
        graph,
        positions=(graph.positions @ transform).astype(np.float32, copy=False),
        lattice=(graph.lattice @ transform).astype(np.float32, copy=False),
    )
    return _energy(model, strained_graph)


def _energy(model, graph):
    batch = CrystalGraphBatch.from_graphs([graph])
    energy, _, _ = predict_energy_forces_stress(
        model,
        batch,
        create_graph=False,
        compute_stress=False,
    )
    return energy.item()
