import torch
import torch.nn as nn
import pytest
from pymatgen.core import Lattice, Structure

from gptff.graph import CrystalGraphBatch, CrystalGraphConverter
from gptff.model import GPTFFNet, GPTFFNetConfig
from gptff.model.aggregation import (
    edge_counts_per_center,
    normalize_aggregation,
)
from gptff.model.embedding import EdgeModulation
from gptff.model.interaction import InteractionBlock


def _cfg(n_layers=1, **kwargs):
    defaults = {
        "node_feature_len": 8,
        "edge_feature_len": 8,
        "n_layers": n_layers,
        "num_radial": 8,
        "num_angular": 4,
        "radial_cutoff": 3.0,
        "angle_cutoff": 3.0,
        "cutoff_coeff": 5,
    }
    defaults.update(kwargs)
    return GPTFFNetConfig(**defaults)


def _batch(a_cut=3.0):
    structure = Structure(
        Lattice.cubic(3.0),
        ["Na", "Cl"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    graph = CrystalGraphConverter(r_cut=3.0, a_cut=a_cut).convert(structure)
    return CrystalGraphBatch.from_graphs([graph]).with_geometry()


def test_non_transformer_model_uses_interaction_blocks():
    model = GPTFFNet(_cfg(n_layers=2))

    assert len(model.interactions) == 2
    assert isinstance(model.interactions[0], InteractionBlock)
    assert isinstance(model.interactions[0].triplet_edge_norm, nn.LayerNorm)
    assert isinstance(model.interactions[0].pair_atom_norm, nn.LayerNorm)
    assert isinstance(model.interactions[0].pair_edge_norm, nn.LayerNorm)
    assert isinstance(model.interactions[0].atom_norm, nn.LayerNorm)
    assert isinstance(model.interactions[0].atom_edge_norm, nn.LayerNorm)
    assert model.interactions[0].residual_scale == 1.0
    assert model.interactions[0].aggregation_norm == "sqrt"


def test_interaction_block_returns_finite_atom_and_edge_features():
    graph = _batch()
    model = GPTFFNet(_cfg())

    atom_fea = model.atom_embedding(graph.atom_types)
    edge_basis = model.edge_rbf(graph.edge_lengths)
    angle_edge_basis = model.angle_edge_rbf(graph.edge_lengths)
    edge_modulation = model.edge_modulation(edge_basis)
    triplet_modulation = model.triplet_modulation(angle_edge_basis)
    edge_ij = model.edge_embedding(edge_basis)

    atom_out, edge_out = model.interactions[0](
        atom_fea,
        edge_ij,
        graph,
        edge_modulation,
        triplet_modulation,
    )

    assert atom_out.shape == atom_fea.shape
    assert edge_out.shape == edge_ij.shape
    assert torch.isfinite(atom_out).all()
    assert torch.isfinite(edge_out).all()


def test_interaction_block_handles_no_edge_graph():
    structure = Structure(
        Lattice.cubic(10.0),
        ["Na", "Cl"],
        [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
        coords_are_cartesian=True,
    )
    graph = CrystalGraphConverter(r_cut=1.0, a_cut=1.0).convert(structure)
    batch = CrystalGraphBatch.from_graphs([graph]).with_geometry()
    model = GPTFFNet(_cfg(radial_cutoff=1.0, angle_cutoff=1.0))

    energy = model(batch)

    assert graph.num_edges == 0
    assert energy.shape == (1, 1)
    assert torch.isfinite(energy).all()


def test_normalize_aggregation_modes_handle_zero_counts():
    values = torch.ones((3, 2))
    counts = torch.tensor([0, 1, 4])

    assert torch.equal(normalize_aggregation(values, counts, "sum"), values)
    assert torch.allclose(
        normalize_aggregation(values, counts, "mean"),
        torch.tensor([[1.0, 1.0], [1.0, 1.0], [0.25, 0.25]]),
    )
    assert torch.allclose(
        normalize_aggregation(values, counts, "sqrt"),
        torch.tensor([[1.0, 1.0], [1.0, 1.0], [0.5, 0.5]]),
    )


def test_edge_counts_per_center_counts_outgoing_edges():
    edge_index = torch.tensor([[0, 0, 2], [1, 2, 0]])
    counts = edge_counts_per_center(
        edge_index,
        num_atoms=4,
        reference=torch.zeros((4, 2)),
    )

    assert torch.equal(counts, torch.tensor([2.0, 0.0, 1.0, 0.0]))


def test_invalid_aggregation_norm_is_rejected():
    with pytest.raises(ValueError, match="aggregation_norm"):
        _cfg(aggregation_norm="invalid")


def test_three_body_edge_delta_is_zero_without_angle_triplets():
    graph = _batch(a_cut=1.0)
    model = GPTFFNet(_cfg())

    assert graph.triplet_edge_index.numel() == 0

    atom_fea = model.atom_embedding(graph.atom_types)
    edge_basis = model.edge_rbf(graph.edge_lengths)
    angle_edge_basis = model.angle_edge_rbf(graph.edge_lengths)
    triplet_modulation = model.triplet_modulation(angle_edge_basis)
    edge_ij = model.edge_embedding(edge_basis)
    delta = model.interactions[0].three_body(
        edge_ij,
        graph,
        triplet_modulation,
    )

    assert delta.shape == edge_ij.shape
    assert torch.equal(delta, torch.zeros_like(edge_ij))


def test_interaction_block_residual_scale_zero_returns_identity():
    graph = _batch()
    model = GPTFFNet(_cfg(residual_scale=0.0))

    atom_fea = model.atom_embedding(graph.atom_types)
    edge_basis = model.edge_rbf(graph.edge_lengths)
    angle_edge_basis = model.angle_edge_rbf(graph.edge_lengths)
    edge_modulation = model.edge_modulation(edge_basis)
    triplet_modulation = model.triplet_modulation(angle_edge_basis)
    edge_ij = model.edge_embedding(edge_basis)

    atom_out, edge_out = model.interactions[0](
        atom_fea,
        edge_ij,
        graph,
        edge_modulation,
        triplet_modulation,
    )

    assert torch.allclose(atom_out, atom_fea)
    assert torch.allclose(edge_out, edge_ij)


def test_zero_geometry_modulation_zeroes_interaction_deltas():
    graph = _batch()
    model = GPTFFNet(_cfg())
    block = model.interactions[0]

    atom_fea = model.atom_embedding(graph.atom_types)
    edge_basis = model.edge_rbf(graph.edge_lengths)
    edge_ij = model.edge_embedding(edge_basis)
    zero_modulation = EdgeModulation(
        atom=torch.zeros_like(atom_fea[graph.edge_index[0]]),
        edge=torch.zeros_like(edge_ij),
    )
    zero_triplet_modulation = torch.zeros_like(edge_ij)

    triplet_delta = block.three_body(
        block.triplet_edge_norm(edge_ij),
        graph,
        zero_triplet_modulation,
    )
    pair_delta = block.edge_update(
        block.pair_atom_norm(atom_fea),
        block.pair_edge_norm(edge_ij),
        graph,
        zero_modulation.edge,
    )
    atom_delta = block.atom_update(
        block.atom_norm(atom_fea),
        block.atom_edge_norm(edge_ij),
        zero_modulation.atom,
        graph,
    )

    assert torch.equal(triplet_delta, torch.zeros_like(triplet_delta))
    assert torch.equal(pair_delta, torch.zeros_like(pair_delta))
    assert torch.equal(atom_delta, torch.zeros_like(atom_delta))


def test_interaction_block_updates_pair_then_triplet_then_atom():
    graph = _batch()
    model = GPTFFNet(_cfg())
    block = model.interactions[0]
    block.pair_atom_norm = nn.Identity()
    block.pair_edge_norm = nn.Identity()
    block.triplet_edge_norm = nn.Identity()
    block.atom_norm = nn.Identity()
    block.atom_edge_norm = nn.Identity()
    block.edge_update = _ConstantPairDelta(value=1.0)
    block.three_body = _ConstantTripletDelta(value=2.0)
    block.atom_update = _RecordingAtomDelta()

    atom_fea = torch.zeros((graph.atom_types.shape[0], model.atom_fea_len))
    edge_ij = torch.zeros((graph.edge_index.shape[1], model.nbr_fea_len))
    edge_modulation = EdgeModulation(
        atom=torch.zeros_like(atom_fea[graph.edge_index[0]]),
        edge=torch.zeros_like(edge_ij),
    )
    triplet_modulation = torch.zeros_like(edge_ij)

    atom_out, edge_out = block(
        atom_fea,
        edge_ij,
        graph,
        edge_modulation,
        triplet_modulation,
    )

    assert torch.equal(block.three_body.seen_edge, torch.ones_like(edge_ij))
    assert torch.equal(block.atom_update.seen_edge, torch.full_like(edge_ij, 3.0))
    assert torch.equal(edge_out, torch.full_like(edge_ij, 3.0))
    assert torch.equal(atom_out, atom_fea)


class _ConstantPairDelta(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = float(value)

    def forward(self, atom_fea, edge_ij, graph, edge_modulation):
        return torch.full_like(edge_ij, self.value)


class _ConstantTripletDelta(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = float(value)
        self.seen_edge = None

    def forward(self, edge_ij, graph, triplet_modulation):
        self.seen_edge = edge_ij.detach().clone()
        return torch.full_like(edge_ij, self.value)


class _RecordingAtomDelta(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen_edge = None

    def forward(self, atom_fea, edge_ij, edge_modulation, graph):
        self.seen_edge = edge_ij.detach().clone()
        return torch.zeros_like(atom_fea)
