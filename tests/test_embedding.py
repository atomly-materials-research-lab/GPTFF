import pytest
import torch

from gptff.model import GPTFFNet, GPTFFNetConfig
from gptff.model.embedding import AtomEmbedding, EdgeEmbedding, EdgeModulationProjection


def test_atom_embedding_has_no_max_norm_constraint():
    embedding = AtomEmbedding(atom_fea_len=8, max_atomic_number=94)

    assert embedding.embedding.max_norm is None


def test_atom_embedding_rejects_invalid_atomic_numbers():
    embedding = AtomEmbedding(atom_fea_len=8, max_atomic_number=10)

    with pytest.raises(AssertionError, match="Atomic numbers must be in the range"):
        embedding(torch.tensor([1, 11], dtype=torch.long))


def test_edge_embedding_returns_normalized_edge_features():
    edge_embedding = EdgeEmbedding(nbr_fea_len=6, num_radial=4)
    edge_basis = torch.randn(3, 4)

    edge_fea = edge_embedding(edge_basis)

    assert edge_fea.shape == (3, 6)
    assert torch.isfinite(edge_fea).all()


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
    assert not hasattr(model, "w_b")
    assert not hasattr(model, "w_eij")
    assert not hasattr(model, "w_r")
