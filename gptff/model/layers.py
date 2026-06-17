from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from gptff.model.basis import FourierAngleBasis


AGGREGATION_NORMS = {"sum", "mean", "sqrt"}


def validate_aggregation_norm(aggregation_norm: str) -> str:
    aggregation_norm = str(aggregation_norm).lower()
    if aggregation_norm not in AGGREGATION_NORMS:
        supported = ", ".join(sorted(AGGREGATION_NORMS))
        raise ValueError(f"aggregation_norm must be one of: {supported}.")
    return aggregation_norm


def normalize_aggregation(values: torch.Tensor, counts: torch.Tensor, mode: str) -> torch.Tensor:
    mode = validate_aggregation_norm(mode)
    if mode == "sum":
        return values

    counts = counts.to(device=values.device, dtype=values.dtype).clamp_min(1)
    if mode == "sqrt":
        counts = torch.sqrt(counts)

    view_shape = counts.shape + (1,) * (values.ndim - counts.ndim)
    return values / counts.reshape(view_shape)


def edge_counts_per_center(edge_index: torch.Tensor, num_atoms: int, reference: torch.Tensor) -> torch.Tensor:
    counts = reference.new_zeros((num_atoms,))
    if edge_index.shape[1] == 0:
        return counts

    ones = reference.new_ones((edge_index.shape[1],))
    return torch.index_add(counts, 0, edge_index[0], ones)


def _as_hidden_dims(hidden_dims: int | Sequence[int] | None) -> tuple[int, ...]:
    if hidden_dims is None:
        return ()
    if isinstance(hidden_dims, int):
        return (hidden_dims,)
    return tuple(int(dim) for dim in hidden_dims)


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: int | Sequence[int] | None = None,
        *,
        dropout: float = 0.0,
        activate_output: bool = False,
        zero_init_output: bool = False,
    ):
        super().__init__()
        if dropout < 0 or dropout >= 1:
            raise ValueError("dropout must be in the range [0, 1).")

        layers = []
        prev_dim = int(input_dim)
        for hidden_dim in _as_hidden_dims(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, output_dim)
        self.output_activation = nn.SiLU() if activate_output else nn.Identity()
        self.output_dropout = nn.Dropout(dropout)
        if zero_init_output:
            nn.init.zeros_(self.output_layer.weight)
            nn.init.zeros_(self.output_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        x = self.output_activation(x)
        return self.output_dropout(x)


class GatedMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        dropout: float = 0.0,
        zero_init_output: bool = False,
    ):
        super().__init__()
        if dropout < 0 or dropout >= 1:
            raise ValueError("dropout must be in the range [0, 1).")

        self.value = nn.Linear(input_dim, output_dim)
        self.gate = nn.Linear(input_dim, output_dim)
        self.activation = nn.SiLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        if zero_init_output:
            nn.init.zeros_(self.value.weight)
            nn.init.zeros_(self.value.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.activation(self.value(x)) * self.sigmoid(self.gate(x)))


class EdgeUpdate(nn.Module):
    def __init__(
        self,
        atom_fea_len,
        nbr_fea_len,
        num_radial=None,
        *,
        dropout=0.0,
        zero_init_output=True,
    ):
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
            zero_init_output=zero_init_output,
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
        zero_init_output=True,
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
        self.triplet_gate = GatedMLP(
            nbr_fea_len,
            nbr_fea_len,
            dropout=dropout,
            zero_init_output=zero_init_output,
        )

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
        zero_init_output=True,
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
        self.message_gate = GatedMLP(
            2 * atom_fea_len,
            atom_fea_len,
            dropout=dropout,
            zero_init_output=zero_init_output,
        )

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
        residual_zero_init=True,
    ):
        super().__init__()
        if residual_scale < 0:
            raise ValueError("residual_scale must be non-negative.")

        self.residual_scale = float(residual_scale)
        self.residual_zero_init = bool(residual_zero_init)
        self.aggregation_norm = validate_aggregation_norm(aggregation_norm)
        self.residual_dropout = nn.Dropout(dropout)
        self.three_body = ThreeBodyEdgeDelta(
            nbr_fea_len=nbr_fea_len,
            num_angular=num_angular,
            dropout=dropout,
            aggregation_norm=self.aggregation_norm,
            zero_init_output=self.residual_zero_init,
        )
        self.edge_update = EdgeUpdate(
            atom_fea_len,
            nbr_fea_len,
            dropout=dropout,
            zero_init_output=self.residual_zero_init,
        )
        self.atom_update = AtomFeatureDelta(
            atom_fea_len,
            nbr_fea_len,
            dropout=dropout,
            aggregation_norm=self.aggregation_norm,
            zero_init_output=self.residual_zero_init,
        )
        self.pair_atom_norm = nn.LayerNorm(atom_fea_len)
        self.atom_norm = nn.LayerNorm(atom_fea_len)

    def forward(self, atom_fea, edge_ij, graph, geometry_features):
        pair_delta = self.edge_update(
            self.pair_atom_norm(atom_fea),
            edge_ij,
            graph,
            geometry_features.edge_modulation.edge,
        )
        edge_ij = edge_ij + self.residual_scale * self.residual_dropout(pair_delta)

        triplet_delta = self.three_body(
            edge_ij,
            graph,
            geometry_features.triplet_modulation,
        )
        edge_ij = edge_ij + self.residual_scale * self.residual_dropout(triplet_delta)

        atom_delta = self.atom_update(
            self.atom_norm(atom_fea),
            edge_ij,
            geometry_features.edge_modulation.atom,
            graph,
        )
        atom_fea = atom_fea + self.residual_scale * self.residual_dropout(atom_delta)
        return atom_fea, edge_ij
