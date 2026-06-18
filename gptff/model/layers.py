from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from gptff.model.basis import FourierAngleBasis


def sum_aggregation(
    values: torch.Tensor,
    indices: torch.Tensor,
    dim_size: int,
    reference: torch.Tensor,
) -> torch.Tensor:
    output = reference.new_zeros((dim_size, *values.shape[1:]))
    return torch.index_add(output, 0, indices, values.to(output.dtype))


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
        bias: bool = True,
    ):
        super().__init__()
        if dropout < 0 or dropout >= 1:
            raise ValueError("dropout must be in the range [0, 1).")

        self.value = nn.Linear(input_dim, output_dim, bias=bias)
        self.gate = nn.Linear(input_dim, output_dim, bias=bias)
        self.activation = nn.SiLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.activation(self.value(x)) * self.sigmoid(self.gate(x)))


class EdgeUpdate(nn.Module):
    def __init__(
        self,
        atom_feature_dim,
        edge_feature_dim,
        num_radial=None,
        *,
        dropout=0.0,
    ):
        super().__init__()
        self.atom_feature_dim = atom_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.modulation_projection = (
            nn.Linear(num_radial, edge_feature_dim, bias=False)
            if num_radial is not None
            else None
        )
        self.message_gate = GatedMLP(
            2 * atom_feature_dim + edge_feature_dim,
            edge_feature_dim,
            dropout=dropout,
        )

    def forward(self, atom_features, edge_features, graph, edge_modulation):
        pair_features = torch.cat([
            atom_features[graph.edge_index[0]],
            atom_features[graph.edge_index[1]],
            edge_features,
        ], dim=-1)

        edge_message = self.message_gate(pair_features)
        if edge_modulation.shape[-1] != self.edge_feature_dim:
            if self.modulation_projection is None:
                raise ValueError(
                    "edge_modulation must have edge feature dimension "
                    f"{self.edge_feature_dim}."
                )
            edge_modulation = self.modulation_projection(edge_modulation)
        return edge_message * edge_modulation


class ThreeBodyEdgeDelta(nn.Module):
    def __init__(
        self,
        atom_feature_dim,
        edge_feature_dim,
        num_angular,
        *,
        dropout=0.0,
    ):
        super().__init__()
        self.atom_feature_dim = atom_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.angle_basis = FourierAngleBasis(num_angular)

        self.target_encoder = MLP(
            2 * atom_feature_dim + edge_feature_dim,
            edge_feature_dim,
            dropout=dropout,
            activate_output=True,
        )
        self.source_encoder = MLP(
            atom_feature_dim + edge_feature_dim + self.angle_basis.out_dim,
            edge_feature_dim,
            dropout=dropout,
            activate_output=True,
        )
        self.output_gate = GatedMLP(
            edge_feature_dim,
            edge_feature_dim,
            dropout=dropout,
            bias=False,
        )

    def forward(self, atom_features, edge_features, graph, triplet_modulation):
        if graph.triplet_edge_index.numel() == 0:
            return edge_features.new_zeros(edge_features.shape)

        target_edge_indices = graph.triplet_edge_index[0]
        source_edge_indices = graph.triplet_edge_index[1]

        target_features = torch.cat([
            atom_features[graph.edge_index[0]],
            atom_features[graph.edge_index[1]],
            edge_features,
        ], dim=-1)
        target_message = self.target_encoder(target_features) * triplet_modulation

        source_features = torch.cat([
            atom_features[graph.edge_index[1][source_edge_indices]],
            edge_features[source_edge_indices],
            self.angle_basis(graph.triplet_cosine),
        ], dim=-1)
        source_message = self.source_encoder(source_features) * triplet_modulation[source_edge_indices]

        aggregated_source_message = sum_aggregation(
            source_message,
            target_edge_indices,
            dim_size=edge_features.shape[0],
            reference=edge_features,
        )
        return self.output_gate(target_message * aggregated_source_message)


class AtomFeatureDelta(nn.Module):
    def __init__(
        self,
        atom_feature_dim,
        edge_feature_dim,
        *,
        dropout=0.0,
    ):
        super().__init__()
        self.atom_feature_dim = atom_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.message_encoder = MLP(
            2 * atom_feature_dim + edge_feature_dim,
            2 * atom_feature_dim,
            dropout=dropout,
            activate_output=True,
        )
        self.message_gate = GatedMLP(
            2 * atom_feature_dim,
            atom_feature_dim,
            dropout=dropout,
        )

    def forward(self, atom_features, edge_features, edge_modulation, graph):
        pair_features = torch.cat([
            atom_features[graph.edge_index[0]],
            atom_features[graph.edge_index[1]],
            edge_features,
        ], dim=-1)
        atom_message = self.message_gate(self.message_encoder(pair_features))
        atom_message = atom_message * edge_modulation

        return sum_aggregation(
            atom_message,
            graph.edge_index[0],
            dim_size=atom_features.shape[0],
            reference=atom_features,
        )


class InteractionBlock(nn.Module):
    def __init__(
        self,
        atom_feature_dim,
        edge_feature_dim,
        num_angular,
        *,
        dropout=0.0,
    ):
        super().__init__()

        self.residual_dropout = nn.Dropout(dropout)
        self.three_body = ThreeBodyEdgeDelta(
            atom_feature_dim=atom_feature_dim,
            edge_feature_dim=edge_feature_dim,
            num_angular=num_angular,
            dropout=dropout,
        )
        self.edge_update = EdgeUpdate(
            atom_feature_dim,
            edge_feature_dim,
            dropout=dropout,
        )
        self.atom_update = AtomFeatureDelta(
            atom_feature_dim,
            edge_feature_dim,
            dropout=dropout,
        )
        self.pair_atom_norm = nn.LayerNorm(atom_feature_dim)
        self.triplet_atom_norm = nn.LayerNorm(atom_feature_dim)
        self.atom_norm = nn.LayerNorm(atom_feature_dim)

    def forward(self, atom_features, edge_features, graph, geometry_features):
        triplet_delta = self.three_body(
            self.triplet_atom_norm(atom_features),
            edge_features,
            graph,
            geometry_features.triplet_modulation,
        )
        edge_features = edge_features + self.residual_dropout(triplet_delta)

        pair_delta = self.edge_update(
            self.pair_atom_norm(atom_features),
            edge_features,
            graph,
            geometry_features.edge_modulation.edge_message,
        )
        edge_features = edge_features + self.residual_dropout(pair_delta)

        atom_delta = self.atom_update(
            self.atom_norm(atom_features),
            edge_features,
            geometry_features.edge_modulation.atom_message,
            graph,
        )
        atom_features = atom_features + self.residual_dropout(atom_delta)
        return atom_features, edge_features
