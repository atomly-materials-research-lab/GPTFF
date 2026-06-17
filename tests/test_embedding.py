from types import SimpleNamespace

import pytest
import torch

from gptff.model import GPTFFNet
from gptff.model.embedding import AtomEmbedding, EdgeEmbedding


def test_atom_embedding_has_no_max_norm_constraint():
    embedding = AtomEmbedding(atom_fea_len=8, max_atomic_number=94)

    assert embedding.embedding.max_norm is None


def test_atom_embedding_rejects_invalid_atomic_numbers():
    embedding = AtomEmbedding(atom_fea_len=8, max_atomic_number=10)

    with pytest.raises(AssertionError, match="Atomic numbers must be in the range"):
        embedding(torch.tensor([1, 11], dtype=torch.long))


def test_edge_embedding_returns_normalized_edge_features():
    edge_embedding = EdgeEmbedding(atom_fea_len=8, nbr_fea_len=6, num_radial=4)
    atom_fea = torch.randn(3, 8)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    edge_basis = torch.randn(3, 4)

    edge_fea = edge_embedding(atom_fea, edge_index, edge_basis)

    assert edge_fea.shape == (3, 6)
    assert torch.isfinite(edge_fea).all()


def test_non_transformer_model_uses_embedding_modules():
    cfg = SimpleNamespace(
        node_feature_len=8,
        edge_feature_len=8,
        n_layers=1,
        num_radial=8,
        num_angular=4,
        radial_cutoff=3.0,
        angle_cutoff=3.0,
        cutoff_coeff=5,
        max_atomic_number=94,
        device="cpu",
    )

    model = GPTFFNet(cfg)

    assert isinstance(model.atom_embedding, AtomEmbedding)
    assert isinstance(model.edge_embedding, EdgeEmbedding)
    assert not hasattr(model, "w_b")
    assert not hasattr(model, "w_eij")
    assert not hasattr(model, "w_r")
