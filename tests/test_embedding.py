import pytest
import torch
import torch.nn as nn

from gptff.model import GPTFF, GPTFFConfig
from gptff.model.encoders import (
    AtomEmbedding,
    EdgeEmbedding,
    EdgeModulationProjection,
    GeometryEmbedding,
    TripletModulationProjection,
)


def test_atom_embedding_has_no_max_norm_constraint():
    embedding = AtomEmbedding(atom_feature_dim=8, max_atomic_number=94)

    assert embedding.embedding.max_norm is None


def test_atom_embedding_rejects_invalid_atomic_numbers():
    embedding = AtomEmbedding(atom_feature_dim=8, max_atomic_number=10)

    with pytest.raises(AssertionError, match="Atomic numbers must be in the range"):
        embedding(torch.tensor([1, 11], dtype=torch.long))


def test_edge_embedding_returns_edge_features():
    edge_embedding = EdgeEmbedding(edge_feature_dim=6, num_radial=4)
    edge_basis = torch.randn(3, 4)

    edge_features = edge_embedding(edge_basis)

    assert edge_features.shape == (3, 6)
    assert torch.isfinite(edge_features).all()


def test_edge_embedding_preserves_radial_basis_scale():
    edge_embedding = EdgeEmbedding(edge_feature_dim=6, num_radial=4)
    with torch.no_grad():
        for layer in edge_embedding.edge_embedding:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.fill_(1.0)

    edge_basis = torch.ones(3, 4)
    small_edge_basis = 0.01 * edge_basis

    edge_features = edge_embedding(edge_basis)
    small_edge_features = edge_embedding(small_edge_basis)

    assert small_edge_features.norm() < edge_features.norm()


def test_edge_embedding_preserves_zero_basis():
    edge_embedding = EdgeEmbedding(edge_feature_dim=6, num_radial=4)
    edge_basis = torch.zeros(3, 4)

    edge_features = edge_embedding(edge_basis)

    assert torch.equal(edge_features, torch.zeros_like(edge_features))


def test_edge_modulation_projection_preserves_zero_basis():
    projection = EdgeModulationProjection(atom_feature_dim=8, edge_feature_dim=6, num_radial=4)
    edge_basis = torch.zeros(3, 4)

    modulation = projection(edge_basis)

    assert torch.equal(modulation.atom_message, torch.zeros_like(modulation.atom_message))
    assert torch.equal(modulation.edge_message, torch.zeros_like(modulation.edge_message))


def test_triplet_modulation_projection_preserves_zero_basis():
    projection = TripletModulationProjection(edge_feature_dim=6, num_radial=4)
    angle_radial_basis = torch.zeros(3, 4)

    modulation = projection(angle_radial_basis)

    assert torch.equal(modulation, torch.zeros_like(modulation))


def test_triplet_modulation_uses_angle_cutoff_independently():
    cfg = GPTFFConfig(
        atom_feature_dim=8,
        edge_feature_dim=8,
        num_interaction_blocks=1,
        num_radial=8,
        num_angular=4,
        radial_cutoff=3.0,
        angle_cutoff=2.0,
        cutoff_coeff=5,
        max_atomic_number=94,
    )
    model = GPTFF(cfg)
    distances = torch.tensor([2.5])

    edge_basis = model.geometry_embedding.edge_rbf(distances)
    angle_radial_basis = model.geometry_embedding.angle_edge_rbf(distances)
    triplet_modulation = model.geometry_embedding.triplet_modulation(angle_radial_basis)

    assert edge_basis.abs().sum() > 0
    assert torch.equal(angle_radial_basis, torch.zeros_like(angle_radial_basis))
    assert torch.equal(triplet_modulation, torch.zeros_like(triplet_modulation))


def test_non_transformer_model_uses_embedding_modules():
    cfg = GPTFFConfig(
        atom_feature_dim=8,
        edge_feature_dim=8,
        num_interaction_blocks=1,
        num_radial=8,
        num_angular=4,
        radial_cutoff=3.0,
        angle_cutoff=3.0,
        cutoff_coeff=5,
        max_atomic_number=94,
    )

    model = GPTFF(cfg)

    assert isinstance(model.atom_embedding, AtomEmbedding)
    assert isinstance(model.geometry_embedding, GeometryEmbedding)
    assert isinstance(model.geometry_embedding.edge_embedding, EdgeEmbedding)
    assert isinstance(model.geometry_embedding.edge_modulation, EdgeModulationProjection)
    assert isinstance(model.geometry_embedding.triplet_modulation, TripletModulationProjection)
    assert isinstance(model.readout_atom_norm, nn.LayerNorm)
    assert not hasattr(model, "edge_embedding")
    assert not hasattr(model, "edge_modulation")
    assert not hasattr(model, "triplet_modulation")
    assert not hasattr(model, "edge_rbf")
    assert not hasattr(model, "angle_edge_rbf")
    assert not hasattr(model, "w_b")
    assert not hasattr(model, "w_eij")
    assert not hasattr(model, "w_r")


def test_non_transformer_model_can_disable_readout_atom_norm():
    cfg = GPTFFConfig(
        atom_feature_dim=8,
        edge_feature_dim=8,
        num_interaction_blocks=1,
        num_radial=8,
        num_angular=4,
        radial_cutoff=3.0,
        angle_cutoff=3.0,
        cutoff_coeff=5,
        max_atomic_number=94,
        readout_atom_norm=False,
    )

    model = GPTFF(cfg)

    assert isinstance(model.readout_atom_norm, nn.Identity)
