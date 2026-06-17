import numpy as np
import torch
import pytest

from gptff.graph import CrystalGraph, GraphSample
from gptff.model.element_refs import (
    available_element_ref_presets,
    build_element_ref_tensor,
    fit_element_refs_from_samples,
)
from gptff.model.readout import EnergyHead


def test_energy_head_is_extensive_for_learned_site_energy():
    head = EnergyHead(atom_fea_len=4)
    atom_fea = torch.randn(3, 4)
    atom_types = torch.tensor([1, 1, 1], dtype=torch.long)
    atom_batch = torch.tensor([0, 0, 0], dtype=torch.long)

    single_energy = head(atom_fea[:1], atom_types[:1], atom_batch[:1], num_graphs=1)
    tripled_energy = head(atom_fea[:1].repeat(3, 1), atom_types, atom_batch, num_graphs=1)

    assert torch.allclose(tripled_energy, 3 * single_energy, atol=1e-6, rtol=1e-6)


def test_energy_head_respects_readout_depth():
    head = EnergyHead(atom_fea_len=4, n_readout_layers=4)

    linear_layers = [module for module in head.mlp if isinstance(module, torch.nn.Linear)]

    assert len(linear_layers) == 4
    assert linear_layers[-1].out_features == 1


def test_energy_head_adds_element_reference_energies():
    head = EnergyHead(
        atom_fea_len=4,
        max_atomic_number=3,
        element_refs={"1": -1.5, "3": 2.0},
    )
    for param in head.parameters():
        torch.nn.init.zeros_(param)

    atom_fea = torch.zeros(3, 4)
    atom_types = torch.tensor([1, 3, 1], dtype=torch.long)
    atom_batch = torch.tensor([0, 0, 1], dtype=torch.long)

    energy = head(atom_fea, atom_types, atom_batch, num_graphs=2)

    assert torch.allclose(energy.squeeze(-1), torch.tensor([0.5, -1.5]))


def test_energy_head_accepts_one_indexed_reference_sequence():
    head = EnergyHead(atom_fea_len=4, max_atomic_number=3, element_refs=[-1.0, 0.0, 2.0])

    assert head.element_refs.shape == (4,)
    assert torch.allclose(head.element_refs, torch.tensor([0.0, -1.0, 0.0, 2.0]))


def test_energy_head_accepts_named_element_reference_preset():
    head = EnergyHead(atom_fea_len=4, max_atomic_number=3, element_refs="atomly")
    for param in head.parameters():
        torch.nn.init.zeros_(param)

    atom_fea = torch.zeros(3, 4)
    atom_types = torch.tensor([1, 2, 3], dtype=torch.long)
    atom_batch = torch.tensor([0, 0, 1], dtype=torch.long)

    energy = head(atom_fea, atom_types, atom_batch, num_graphs=2)

    assert "atomly" in available_element_ref_presets()
    assert torch.allclose(
        energy.squeeze(-1),
        torch.tensor([-4.22146044, -3.46224791]),
    )


def test_unknown_named_element_reference_preset_lists_available_presets():
    with pytest.raises(ValueError, match="Available presets: atomly"):
        build_element_ref_tensor("missing", max_atomic_number=94)


def test_fit_element_refs_from_samples_recovers_composition_offsets():
    samples = [
        _sample([1], -1.0),
        _sample([3], 2.0),
        _sample([1, 1, 3], 0.0),
    ]

    refs = fit_element_refs_from_samples(samples, max_atomic_number=3)

    assert refs == pytest.approx({"1": -1.0, "3": 2.0})


def _sample(atom_types, energy):
    atom_types = np.asarray(atom_types, dtype=np.int64)
    graph = CrystalGraph(
        atom_types=atom_types,
        positions=np.zeros((len(atom_types), 3), dtype=np.float32),
        lattice=np.eye(3, dtype=np.float32),
        edge_index=np.empty((2, 0), dtype=np.int64),
        edge_offsets=np.empty((0, 3), dtype=np.float32),
        edge_distances=np.empty((0,), dtype=np.float32),
        triplet_edge_index=np.empty((2, 0), dtype=np.int64),
        triplets_per_atom=np.zeros((len(atom_types),), dtype=np.int64),
        triplets_per_edge=np.empty((0,), dtype=np.int64),
    )
    return GraphSample(
        graph=graph,
        energy=energy,
        forces=np.zeros((len(atom_types), 3), dtype=np.float32),
        stress=np.zeros((3, 3), dtype=np.float32),
    )
