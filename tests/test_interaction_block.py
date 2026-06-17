import torch
import torch.nn as nn
from pymatgen.core import Lattice, Structure

from gptff.graph import CrystalGraphBatch, CrystalGraphConverter
from gptff.model import GPTFFNet, GPTFFNetConfig
from gptff.model.interaction import InteractionBlock


def _cfg(n_layers=1, **kwargs):
    return GPTFFNetConfig(
        node_feature_len=8,
        edge_feature_len=8,
        n_layers=n_layers,
        num_radial=8,
        num_angular=4,
        radial_cutoff=3.0,
        angle_cutoff=3.0,
        cutoff_coeff=5,
        **kwargs,
    )


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
    assert isinstance(model.interactions[0].triplet_atom_norm, nn.LayerNorm)
    assert isinstance(model.interactions[0].triplet_edge_norm, nn.LayerNorm)
    assert isinstance(model.interactions[0].pair_atom_norm, nn.LayerNorm)
    assert isinstance(model.interactions[0].pair_edge_norm, nn.LayerNorm)
    assert isinstance(model.interactions[0].atom_norm, nn.LayerNorm)
    assert isinstance(model.interactions[0].atom_edge_norm, nn.LayerNorm)
    assert model.interactions[0].residual_scale == 1.0


def test_interaction_block_returns_finite_atom_and_edge_features():
    graph = _batch()
    model = GPTFFNet(_cfg())

    atom_fea = model.atom_embedding(graph.atom_types)
    edge_basis = model.edge_rbf(graph.edge_lengths)
    triplet_basis_ij = model.triplet_rbf(graph.triplet_lengths_ij)
    triplet_basis_ik = model.triplet_rbf(graph.triplet_lengths_ik)

    edge_ij = model.edge_embedding(atom_fea, graph.edge_index, edge_basis)

    atom_out, edge_out = model.interactions[0](
        atom_fea,
        edge_ij,
        graph,
        edge_basis,
        triplet_basis_ij,
        triplet_basis_ik,
    )

    assert atom_out.shape == atom_fea.shape
    assert edge_out.shape == edge_ij.shape
    assert torch.isfinite(atom_out).all()
    assert torch.isfinite(edge_out).all()


def test_three_body_edge_delta_is_zero_without_angle_triplets():
    graph = _batch(a_cut=1.0)
    model = GPTFFNet(_cfg())

    assert graph.triplet_edge_index.numel() == 0

    atom_fea = model.atom_embedding(graph.atom_types)
    edge_basis = model.edge_rbf(graph.edge_lengths)
    edge_ij = model.edge_embedding(atom_fea, graph.edge_index, edge_basis)

    triplet_basis_ij = model.triplet_rbf(graph.triplet_lengths_ij)
    triplet_basis_ik = model.triplet_rbf(graph.triplet_lengths_ik)
    delta = model.interactions[0].three_body(
        atom_fea,
        edge_ij,
        graph,
        triplet_basis_ij,
        triplet_basis_ik,
    )

    assert delta.shape == edge_ij.shape
    assert torch.equal(delta, torch.zeros_like(edge_ij))


def test_interaction_block_residual_scale_zero_returns_identity():
    graph = _batch()
    model = GPTFFNet(_cfg(residual_scale=0.0))

    atom_fea = model.atom_embedding(graph.atom_types)
    edge_basis = model.edge_rbf(graph.edge_lengths)
    triplet_basis_ij = model.triplet_rbf(graph.triplet_lengths_ij)
    triplet_basis_ik = model.triplet_rbf(graph.triplet_lengths_ik)
    edge_ij = model.edge_embedding(atom_fea, graph.edge_index, edge_basis)

    atom_out, edge_out = model.interactions[0](
        atom_fea,
        edge_ij,
        graph,
        edge_basis,
        triplet_basis_ij,
        triplet_basis_ik,
    )

    assert torch.allclose(atom_out, atom_fea)
    assert torch.allclose(edge_out, edge_ij)
