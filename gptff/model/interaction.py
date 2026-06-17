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
    def __init__(self, atom_fea_len, nbr_fea_len, num_radial=None, *, dropout=0.0):
        super().__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.modulation_projection = (
            nn.Linear(num_radial, nbr_fea_len, bias=False)
            if num_radial is not None
            else None
        )
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

    def forward(self, atom_fea, edge_ij, graph, edge_modulation):
        atom_nbr_fea = torch.cat([
            atom_fea[graph.edge_index[0]],
            atom_fea[graph.edge_index[1]],
            edge_ij,
        ], dim=-1)

        edge_msg = self.message_projection(self.message_gate(atom_nbr_fea))
        if edge_modulation.shape[-1] != self.nbr_fea_len:
            if self.modulation_projection is None:
                raise ValueError(
                    "edge_modulation must have edge feature dimension "
                    f"{self.nbr_fea_len}."
                )
            edge_modulation = self.modulation_projection(edge_modulation)
        return edge_msg * edge_modulation


class ThreeBodyEdgeDelta(nn.Module):
    def __init__(
        self,
        nbr_fea_len,
        num_angular,
        *,
        dropout=0.0,
        aggregation_norm="sqrt",
    ):
        super().__init__()
        self.nbr_fea_len = nbr_fea_len
        self.aggregation_norm = validate_aggregation_norm(aggregation_norm)
        self.angle_basis = FourierAngleBasis(num_angular)

        self.triplet_encoder = MLP(
            2 * nbr_fea_len + self.angle_basis.out_dim,
            nbr_fea_len,
            dropout=dropout,
            activate_output=True,
        )
        self.triplet_gate = GatedMLP(nbr_fea_len, nbr_fea_len, dropout=dropout)

    def forward(self, edge_ij, graph, edge_modulation):
        if graph.triplet_edge_index.numel() == 0:
            return edge_ij.new_zeros(edge_ij.shape)

        edge_ij_indices = graph.triplet_edge_index[0]
        edge_ik_indices = graph.triplet_edge_index[1]
        triplet_fea = torch.cat([
            edge_ij[edge_ij_indices],
            edge_ij[edge_ik_indices],
            self.angle_basis(graph.triplet_cosine),
        ], dim=-1)

        triplet_msg = self.triplet_gate(self.triplet_encoder(triplet_fea))
        triplet_msg = (
            triplet_msg
            * edge_modulation[edge_ij_indices]
            * edge_modulation[edge_ik_indices]
        )

        edge_delta = edge_ij.new_zeros(edge_ij.shape)
        edge_delta = torch.index_add(edge_delta, 0, edge_ij_indices, triplet_msg)
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

    def forward(self, atom_fea, edge_ij, edge_modulation, graph):
        atom_nbr_fea = torch.cat([
            atom_fea[graph.edge_index[0]],
            atom_fea[graph.edge_index[1]],
            edge_ij,
        ], dim=-1)
        atom_msg = self.message_gate(self.message_encoder(atom_nbr_fea))
        atom_msg = atom_msg * edge_modulation

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
            nbr_fea_len=nbr_fea_len,
            num_angular=num_angular,
            dropout=dropout,
            aggregation_norm=self.aggregation_norm,
        )
        self.edge_update = EdgeUpdate(atom_fea_len, nbr_fea_len, dropout=dropout)
        self.atom_update = AtomFeatureDelta(
            atom_fea_len,
            nbr_fea_len,
            dropout=dropout,
            aggregation_norm=self.aggregation_norm,
        )
        self.triplet_edge_norm = nn.LayerNorm(nbr_fea_len)
        self.pair_atom_norm = nn.LayerNorm(atom_fea_len)
        self.pair_edge_norm = nn.LayerNorm(nbr_fea_len)
        self.atom_norm = nn.LayerNorm(atom_fea_len)
        self.atom_edge_norm = nn.LayerNorm(nbr_fea_len)

    def forward(self, atom_fea, edge_ij, graph, edge_modulation):
        triplet_delta = self.three_body(
            self.triplet_edge_norm(edge_ij),
            graph,
            edge_modulation.edge,
        )
        edge_ij = edge_ij + self.residual_scale * self.residual_dropout(triplet_delta)

        pair_delta = self.edge_update(
            self.pair_atom_norm(atom_fea),
            self.pair_edge_norm(edge_ij),
            graph,
            edge_modulation.edge,
        )
        edge_ij = edge_ij + self.residual_scale * self.residual_dropout(pair_delta)

        atom_delta = self.atom_update(
            self.atom_norm(atom_fea),
            self.atom_edge_norm(edge_ij),
            edge_modulation.atom,
            graph,
        )
        atom_fea = atom_fea + self.residual_scale * self.residual_dropout(atom_delta)
        return atom_fea, edge_ij
