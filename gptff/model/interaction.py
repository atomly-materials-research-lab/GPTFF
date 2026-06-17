import torch
import torch.nn as nn

from gptff.model.aggregation import (
    edge_counts_per_center,
    normalize_aggregation,
    validate_aggregation_norm,
)
from gptff.model.basis import FourierAngleBasis
from gptff.model.mlp import GatedMLP, MLP


class EdgeUpdate(nn.Module):
    def __init__(self, atom_fea_len, nbr_fea_len, num_radial, dropout=0.0):
        super().__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.message_gate = GatedMLP(
            2 * atom_fea_len + nbr_fea_len,
            nbr_fea_len,
            dropout=dropout,
        )
        self.message_projection = MLP(
            nbr_fea_len,
            nbr_fea_len,
            dropout=dropout,
            activate_output=True,
        )
        self.radial_gate = nn.Linear(num_radial, nbr_fea_len, bias=False)

    def forward(self, atom_fea, edge_ij, graph, edge_basis):
        atom_nbr_fea = torch.cat([
            atom_fea[graph.edge_index[0]],
            atom_fea[graph.edge_index[1]],
            edge_ij,
        ], dim=-1)

        edge_msg = self.message_projection(self.message_gate(atom_nbr_fea))
        return edge_msg * self.radial_gate(edge_basis)


class ThreeBodyEdgeDelta(nn.Module):
    def __init__(
        self,
        atom_fea_len,
        nbr_fea_len,
        num_radial,
        num_angular,
        *,
        dropout=0.0,
        aggregation_norm="sqrt",
    ):
        super().__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.aggregation_norm = validate_aggregation_norm(aggregation_norm)
        self.angle_basis = FourierAngleBasis(num_angular)
        self.angle_embedding = nn.Linear(self.angle_basis.out_dim, nbr_fea_len, bias=False)
        self.bond_embedding_k = nn.Linear(num_radial, nbr_fea_len, bias=False)
        self.bond_embedding_j = nn.Linear(num_radial, nbr_fea_len, bias=False)

        self.triplet_encoder = MLP(
            3 * atom_fea_len + 2 * nbr_fea_len,
            nbr_fea_len,
            dropout=dropout,
            activate_output=True,
        )
        self.triplet_gate = GatedMLP(nbr_fea_len, nbr_fea_len, dropout=dropout)

    def forward(self, atom_fea, edge_ij, graph, triplet_basis_ij, triplet_basis_ik):
        if graph.triplet_edge_index.numel() == 0:
            return edge_ij.new_zeros(edge_ij.shape)

        edge_ij_indices = graph.triplet_edge_index[0]
        edge_ik_indices = graph.triplet_edge_index[1]
        triple_i_indices = graph.edge_index[0][edge_ij_indices]
        triple_j_indices = graph.edge_index[1][edge_ij_indices]
        triple_k_indices = graph.edge_index[1][edge_ik_indices]
        atom_fea_ik = torch.cat([
            atom_fea[triple_i_indices],
            atom_fea[triple_j_indices],
            atom_fea[triple_k_indices],
            edge_ij[edge_ij_indices],
            edge_ij[edge_ik_indices],
        ], dim=-1)

        atom_fea_ik = self.triplet_encoder(atom_fea_ik)

        angles_mat = self.angle_embedding(self.angle_basis(graph.triplet_cosine))
        bonds_mat_k = self.bond_embedding_k(triplet_basis_ik)
        bonds_mat_j = self.bond_embedding_j(triplet_basis_ij)
        atom_fea_ik = (
            self.triplet_gate(atom_fea_ik)
            * bonds_mat_j
            * bonds_mat_k
            * angles_mat
        )

        edge_delta = edge_ij.new_zeros(edge_ij.shape)
        edge_delta = torch.index_add(edge_delta, 0, edge_ij_indices, atom_fea_ik)
        return normalize_aggregation(
            edge_delta,
            graph.triplets_per_edge,
            self.aggregation_norm,
        )


class AtomFeatureDelta(nn.Module):
    def __init__(
        self,
        atom_fea_len,
        nbr_fea_len,
        num_radial,
        *,
        dropout=0.0,
        aggregation_norm="sqrt",
    ):
        super().__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.aggregation_norm = validate_aggregation_norm(aggregation_norm)
        self.message_encoder = MLP(
            2 * atom_fea_len + nbr_fea_len,
            2 * atom_fea_len,
            dropout=dropout,
            activate_output=True,
        )
        self.message_gate = GatedMLP(2 * atom_fea_len, atom_fea_len, dropout=dropout)
        self.radial_gate = nn.Linear(num_radial, atom_fea_len, bias=False)

    def forward(self, atom_fea, edge_ij, edge_basis, graph):
        atom_nbr_fea = torch.cat([
            atom_fea[graph.edge_index[0]],
            atom_fea[graph.edge_index[1]],
            edge_ij,
        ], dim=-1)
        atom_msg = self.message_gate(self.message_encoder(atom_nbr_fea))
        atom_msg = atom_msg * self.radial_gate(edge_basis)

        atom_delta = atom_fea.new_zeros(atom_fea.shape)
        atom_delta = torch.index_add(atom_delta, 0, graph.edge_index[0], atom_msg.to(atom_delta.dtype))
        return normalize_aggregation(
            atom_delta,
            edge_counts_per_center(graph.edge_index, atom_delta.shape[0], atom_delta),
            self.aggregation_norm,
        )


class InteractionBlock(nn.Module):
    def __init__(
        self,
        atom_fea_len,
        nbr_fea_len,
        num_radial,
        num_angular,
        *,
        dropout=0.0,
        residual_scale=1.0,
        aggregation_norm="sqrt",
    ):
        super().__init__()
        if residual_scale < 0:
            raise ValueError("residual_scale must be non-negative.")

        self.residual_scale = float(residual_scale)
        self.aggregation_norm = validate_aggregation_norm(aggregation_norm)
        self.residual_dropout = nn.Dropout(dropout)
        self.three_body = ThreeBodyEdgeDelta(
            atom_fea_len=atom_fea_len,
            nbr_fea_len=nbr_fea_len,
            num_radial=num_radial,
            num_angular=num_angular,
            dropout=dropout,
            aggregation_norm=self.aggregation_norm,
        )
        self.edge_update = EdgeUpdate(atom_fea_len, nbr_fea_len, num_radial, dropout=dropout)
        self.atom_update = AtomFeatureDelta(
            atom_fea_len,
            nbr_fea_len,
            num_radial,
            dropout=dropout,
            aggregation_norm=self.aggregation_norm,
        )
        self.triplet_atom_norm = nn.LayerNorm(atom_fea_len)
        self.triplet_edge_norm = nn.LayerNorm(nbr_fea_len)
        self.pair_atom_norm = nn.LayerNorm(atom_fea_len)
        self.pair_edge_norm = nn.LayerNorm(nbr_fea_len)
        self.atom_norm = nn.LayerNorm(atom_fea_len)
        self.atom_edge_norm = nn.LayerNorm(nbr_fea_len)

    def forward(self, atom_fea, edge_ij, graph, edge_basis, triplet_basis_ij, triplet_basis_ik):
        triplet_delta = self.three_body(
            self.triplet_atom_norm(atom_fea),
            self.triplet_edge_norm(edge_ij),
            graph,
            triplet_basis_ij,
            triplet_basis_ik,
        )
        edge_ij = edge_ij + self.residual_scale * self.residual_dropout(triplet_delta)

        pair_delta = self.edge_update(
            self.pair_atom_norm(atom_fea),
            self.pair_edge_norm(edge_ij),
            graph,
            edge_basis,
        )
        edge_ij = edge_ij + self.residual_scale * self.residual_dropout(pair_delta)

        atom_delta = self.atom_update(
            self.atom_norm(atom_fea),
            self.atom_edge_norm(edge_ij),
            edge_basis,
            graph,
        )
        atom_fea = atom_fea + self.residual_scale * self.residual_dropout(atom_delta)
        return atom_fea, edge_ij
