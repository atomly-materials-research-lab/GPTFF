import pytest
import torch

from gptff.model import GPTFFNet, GPTFFNetConfig
from gptff.model.embedding import (
    AtomEmbedding,
    EdgeEmbedding,
    EdgeModulationProjection,
    TripletModulationProjection,
)


def test_atom_embedding_has_no_max_norm_constraint():
    embedding = AtomEmbedding(atom_fea_len=8, max_atomic_number=94)

    assert embedding.embedding.max_norm is None


def test_atom_embedding_rejects_invalid_atomic_numbers():
    embedding = AtomEmbedding(atom_fea_len=8, max_atomic_number=10)

    with pytest.raises(AssertionError, match="Atomic numbers must be in the range"):
        embedding(torch.tensor([1, 11], dtype=torch.long))


def test_edge_embedding_returns_edge_features():
    edge_embedding = EdgeEmbedding(nbr_fea_len=6, num_radial=4)
    edge_basis = torch.randn(3, 4)

    edge_fea = edge_embedding(edge_basis)

    assert edge_fea.shape == (3, 6)
    assert torch.isfinite(edge_fea).all()


def test_edge_embedding_preserves_radial_basis_scale():
    edge_embedding = EdgeEmbedding(nbr_fea_len=6, num_radial=4)
    with torch.no_grad():
        for layer in edge_embedding.edge_embedding:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.fill_(1.0)

    edge_basis = torch.ones(3, 4)
    small_edge_basis = 0.01 * edge_basis

    edge_fea = edge_embedding(edge_basis)
    small_edge_fea = edge_embedding(small_edge_basis)

    assert small_edge_fea.norm() < edge_fea.norm()


def test_edge_embedding_preserves_zero_basis():
    edge_embedding = EdgeEmbedding(nbr_fea_len=6, num_radial=4)
    edge_basis = torch.zeros(3, 4)

    edge_fea = edge_embedding(edge_basis)

    assert torch.equal(edge_fea, torch.zeros_like(edge_fea))


def test_edge_modulation_projection_preserves_zero_basis():
    projection = EdgeModulationProjection(atom_fea_len=8, nbr_fea_len=6, num_radial=4)
    edge_basis = torch.zeros(3, 4)

    modulation = projection(edge_basis)

    assert torch.equal(modulation.atom, torch.zeros_like(modulation.atom))
    assert torch.equal(modulation.edge, torch.zeros_like(modulation.edge))


def test_triplet_modulation_projection_preserves_zero_basis():
    projection = TripletModulationProjection(nbr_fea_len=6, num_radial=4)
    angle_edge_basis = torch.zeros(3, 4)

    modulation = projection(angle_edge_basis)

    assert torch.equal(modulation, torch.zeros_like(modulation))


def test_triplet_modulation_uses_angle_cutoff_independently():
    cfg = GPTFFNetConfig(
        node_feature_len=8,
        edge_feature_len=8,
        n_layers=1,
        num_radial=8,
        num_angular=4,
        radial_cutoff=3.0,
        angle_cutoff=2.0,
        cutoff_coeff=5,
        max_atomic_number=94,
    )
    model = GPTFFNet(cfg)
    distances = torch.tensor([2.5])

    edge_basis = model.edge_rbf(distances)
    angle_edge_basis = model.angle_edge_rbf(distances)
    triplet_modulation = model.triplet_modulation(angle_edge_basis)

    assert edge_basis.abs().sum() > 0
    assert torch.equal(angle_edge_basis, torch.zeros_like(angle_edge_basis))
    assert torch.equal(triplet_modulation, torch.zeros_like(triplet_modulation))


def test_non_transformer_model_uses_embedding_modules():
    cfg = GPTFFNetConfig(
        node_feature_len=8,
        edge_feature_len=8,
        n_layers=1,
        num_radial=8,
        num_angular=4,
        radial_cutoff=3.0,
        angle_cutoff=3.0,
        cutoff_coeff=5,
        max_atomic_number=94,
    )

    model = GPTFFNet(cfg)

    assert isinstance(model.atom_embedding, AtomEmbedding)
    assert isinstance(model.edge_embedding, EdgeEmbedding)
    assert isinstance(model.edge_modulation, EdgeModulationProjection)
    assert isinstance(model.triplet_modulation, TripletModulationProjection)
    assert not hasattr(model, "w_b")
    assert not hasattr(model, "w_eij")
    assert not hasattr(model, "w_r")
