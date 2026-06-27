from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from gptff.model.basis import LegendreAngleBasis


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
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                ]
            )
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
            nn.Linear(num_radial, edge_feature_dim, bias=False) if num_radial is not None else None
        )
        self.message_gate = GatedMLP(
            2 * atom_feature_dim + edge_feature_dim,
            edge_feature_dim,
            dropout=dropout,
        )

    def forward(self, atom_features, edge_features, graph, edge_modulation):
        pair_features = torch.cat(
            [
                atom_features[graph.edge_index[0]],
                atom_features[graph.edge_index[1]],
                edge_features,
            ],
            dim=-1,
        )

        edge_message = self.message_gate(pair_features)
        if edge_modulation.shape[-1] != self.edge_feature_dim:
            if self.modulation_projection is None:
                raise ValueError(
                    f"edge_modulation must have edge feature dimension {self.edge_feature_dim}."
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
        self.angle_basis = LegendreAngleBasis(num_angular)

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

        target_features = torch.cat(
            [
                atom_features[graph.edge_index[0]],
                atom_features[graph.edge_index[1]],
                edge_features,
            ],
            dim=-1,
        )
        target_message = self.target_encoder(target_features) * triplet_modulation

        source_features = torch.cat(
            [
                atom_features[graph.edge_index[1][source_edge_indices]],
                edge_features[source_edge_indices],
                self.angle_basis(graph.triplet_cosine),
            ],
            dim=-1,
        )
        source_message = (
            self.source_encoder(source_features) * triplet_modulation[source_edge_indices]
        )

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
        pair_features = torch.cat(
            [
                atom_features[graph.edge_index[0]],
                atom_features[graph.edge_index[1]],
                edge_features,
            ],
            dim=-1,
        )
        atom_message = self.message_gate(self.message_encoder(pair_features))
        atom_message = atom_message * edge_modulation

        return sum_aggregation(
            atom_message,
            graph.edge_index[0],
            dim_size=atom_features.shape[0],
            reference=atom_features,
        )


class RadialDensityFeatures(nn.Module):
    def __init__(
        self,
        atom_feature_dim,
        num_radial,
    ):
        super().__init__()
        self.atom_feature_dim = int(atom_feature_dim)
        self.num_radial = int(num_radial)
        self.radial_density = nn.Linear(num_radial, atom_feature_dim, bias=False)
        self.out_dim = atom_feature_dim + 1

    def forward(
        self,
        atom_features,
        graph,
        edge_basis,
        edge_cutoff,
    ):
        if graph.edge_index.numel() == 0:
            return atom_features.new_zeros((atom_features.shape[0], self.out_dim))

        center_indices = graph.edge_index[0]
        radial_density = sum_aggregation(
            self.radial_density(edge_basis),
            center_indices,
            dim_size=atom_features.shape[0],
            reference=atom_features,
        )
        continuous_degree = sum_aggregation(
            edge_cutoff.reshape(-1, 1),
            center_indices,
            dim_size=atom_features.shape[0],
            reference=atom_features[:, :1],
        )
        return torch.cat([continuous_degree, radial_density], dim=-1)


class DensityContext(nn.Module):
    def __init__(
        self,
        density_feature_dim,
        atom_feature_dim,
        *,
        scale_init=0.1,
        dropout=0.0,
    ):
        super().__init__()
        if scale_init < 0:
            raise ValueError("scale_init must be non-negative.")
        if dropout < 0 or dropout >= 1:
            raise ValueError("dropout must be in the range [0, 1).")

        self.density_scale = nn.Parameter(torch.tensor(float(scale_init)))
        self.hidden = nn.Linear(density_feature_dim, atom_feature_dim, bias=False)
        self.output = nn.Linear(atom_feature_dim, atom_feature_dim, bias=False)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, density_features):
        density_context = self.hidden(density_features)
        density_context = self.dropout(self.activation(density_context))
        return self.output(density_context)

    def compute_scale(self, density_context):
        return 1.0 + self.density_scale * torch.tanh(density_context)


def cutoff_weighted_softmax(
    logits: torch.Tensor,
    indices: torch.Tensor,
    dim_size: int,
    edge_cutoff: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    if logits.numel() == 0:
        return logits

    if logits.ndim != 2:
        raise ValueError("logits must have shape [num_edges, num_heads].")

    edge_cutoff = edge_cutoff.reshape(-1, 1).to(dtype=logits.dtype, device=logits.device)
    if edge_cutoff.shape[0] != logits.shape[0]:
        raise ValueError("edge_cutoff must have one value per edge.")

    expanded_indices = indices.reshape(-1, 1).expand(-1, logits.shape[1])
    positive_cutoff = edge_cutoff > 0
    weighted_logits = torch.where(
        positive_cutoff,
        logits + torch.log(edge_cutoff.clamp_min(eps)),
        torch.full_like(logits, -torch.inf),
    )
    with torch.no_grad():
        max_logits = logits.new_full((dim_size, logits.shape[1]), -torch.inf)
        max_logits.scatter_reduce_(
            0,
            expanded_indices,
            weighted_logits,
            reduce="amax",
            include_self=True,
    )

    shifted_logits = weighted_logits - max_logits[indices]
    shifted_logits = torch.where(
        torch.isfinite(shifted_logits),
        shifted_logits,
        torch.zeros_like(shifted_logits),
    )
    weights = torch.where(positive_cutoff, torch.exp(shifted_logits), torch.zeros_like(logits))
    normalizer = logits.new_zeros((dim_size, logits.shape[1]))
    normalizer.index_add_(0, indices, weights)
    return weights / normalizer[indices].clamp_min(eps)


class AttentionAtomUpdate(nn.Module):
    def __init__(
        self,
        atom_feature_dim,
        edge_feature_dim,
        num_radial,
        *,
        num_heads=4,
        dropout=0.0,
        density_scale_init=0.1,
    ):
        super().__init__()
        if atom_feature_dim % num_heads != 0:
            raise ValueError("atom_feature_dim must be divisible by atom attention num_heads.")

        self.atom_feature_dim = int(atom_feature_dim)
        self.edge_feature_dim = int(edge_feature_dim)
        self.num_radial = int(num_radial)
        self.num_heads = int(num_heads)
        self.head_dim = self.atom_feature_dim // self.num_heads
        input_dim = 2 * atom_feature_dim + edge_feature_dim + num_radial

        self.radial_density_features = RadialDensityFeatures(
            atom_feature_dim=atom_feature_dim,
            num_radial=num_radial,
        )
        self.density_context = DensityContext(
            density_feature_dim=self.radial_density_features.out_dim,
            atom_feature_dim=atom_feature_dim,
            scale_init=density_scale_init,
            dropout=dropout,
        )
        self.score = MLP(
            input_dim,
            self.num_heads,
            hidden_dims=atom_feature_dim,
            dropout=dropout,
        )
        self.value = MLP(
            input_dim,
            atom_feature_dim,
            hidden_dims=atom_feature_dim,
            dropout=dropout,
            activate_output=True,
        )
        self.output_gate = GatedMLP(
            3 * atom_feature_dim,
            atom_feature_dim,
            dropout=dropout,
            bias=False,
        )

    def forward(
        self,
        atom_features,
        edge_features,
        edge_modulation,
        graph,
        edge_basis,
        edge_cutoff,
    ):
        if graph.edge_index.numel() == 0:
            return torch.zeros_like(atom_features)

        center_indices = graph.edge_index[0]
        neighbor_indices = graph.edge_index[1]
        pair_features = torch.cat(
            [
                atom_features[center_indices],
                atom_features[neighbor_indices],
                edge_features,
                edge_basis,
            ],
            dim=-1,
        )

        logits = self.score(pair_features)
        attention = cutoff_weighted_softmax(
            logits,
            center_indices,
            dim_size=atom_features.shape[0],
            edge_cutoff=edge_cutoff,
        )
        values = self.value(pair_features) * edge_modulation
        values = values.reshape(-1, self.num_heads, self.head_dim)
        messages = values * attention.unsqueeze(-1)
        attention_aggregate = sum_aggregation(
            messages.reshape(messages.shape[0], self.atom_feature_dim),
            center_indices,
            dim_size=atom_features.shape[0],
            reference=atom_features,
        )
        density_features = self.radial_density_features(
            atom_features,
            graph,
            edge_basis,
            edge_cutoff,
        )
        center_context = atom_features * torch.tanh(density_features[:, :1])
        density_context = self.density_context(density_features)
        scaled_attention = attention_aggregate * self.density_context.compute_scale(
            density_context
        )
        return self.output_gate(
            torch.cat([center_context, scaled_attention, density_context], dim=-1)
        )


class AtomFeedForward(nn.Module):
    def __init__(
        self,
        atom_feature_dim,
        hidden_dim=None,
        *,
        dropout=0.0,
    ):
        super().__init__()
        hidden_dim = int(hidden_dim or 2 * atom_feature_dim)
        self.ffn = MLP(
            atom_feature_dim,
            atom_feature_dim,
            hidden_dims=hidden_dim,
            dropout=dropout,
        )

    def forward(self, atom_features):
        return self.ffn(atom_features)


def _attention_config_value(config, key, default):
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


class InteractionBlock(nn.Module):
    def __init__(
        self,
        atom_feature_dim,
        edge_feature_dim,
        num_angular,
        num_radial,
        *,
        dropout=0.0,
        atom_attention_config=None,
    ):
        super().__init__()

        self.residual_dropout = nn.Dropout(dropout)
        attention_enabled = bool(
            _attention_config_value(
                atom_attention_config,
                "enabled",
                True,
            )
        )
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
        self.attention_enabled = attention_enabled
        self.pair_atom_norm = nn.LayerNorm(atom_feature_dim)
        self.triplet_atom_norm = nn.LayerNorm(atom_feature_dim)
        self.atom_norm = nn.LayerNorm(atom_feature_dim)
        self.atom_ffn = None
        self.atom_ffn_norm = None
        self.atom_ffn_residual_scale = None
        if attention_enabled:
            attention_dropout = float(
                _attention_config_value(
                    atom_attention_config,
                    "dropout",
                    0.0,
                )
            )
            attention_num_heads = int(
                _attention_config_value(
                    atom_attention_config,
                    "num_heads",
                    4,
                )
            )
            use_ffn = bool(
                _attention_config_value(
                    atom_attention_config,
                    "use_ffn",
                    True,
                )
            )
            ffn_hidden_dim = _attention_config_value(
                atom_attention_config,
                "ffn_hidden_dim",
                None,
            )
            density_scale_init = float(
                _attention_config_value(
                    atom_attention_config,
                    "density_scale_init",
                    0.1,
                )
            )
            ffn_residual_scale_init = float(
                _attention_config_value(
                    atom_attention_config,
                    "ffn_residual_scale_init",
                    1e-2,
                )
            )
            self.atom_update = AttentionAtomUpdate(
                atom_feature_dim=atom_feature_dim,
                edge_feature_dim=edge_feature_dim,
                num_radial=num_radial,
                num_heads=attention_num_heads,
                dropout=attention_dropout,
                density_scale_init=density_scale_init,
            )
            if use_ffn:
                self.atom_ffn = AtomFeedForward(
                    atom_feature_dim,
                    hidden_dim=ffn_hidden_dim,
                    dropout=attention_dropout,
                )
                self.atom_ffn_norm = nn.LayerNorm(atom_feature_dim)
                self.atom_ffn_residual_scale = nn.Parameter(torch.tensor(ffn_residual_scale_init))
        else:
            self.atom_update = AtomFeatureDelta(
                atom_feature_dim,
                edge_feature_dim,
                dropout=dropout,
            )

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

        normalized_atom_features = self.atom_norm(atom_features)
        if self.attention_enabled:
            atom_delta = self.atom_update(
                normalized_atom_features,
                edge_features,
                geometry_features.edge_modulation.atom_message,
                graph,
                geometry_features.edge_basis,
                geometry_features.edge_cutoff,
            )
        else:
            atom_delta = self.atom_update(
                normalized_atom_features,
                edge_features,
                geometry_features.edge_modulation.atom_message,
                graph,
            )
        atom_features = atom_features + self.residual_dropout(atom_delta)

        if self.atom_ffn is not None:
            atom_features = (
                atom_features
                + self.atom_ffn_residual_scale * self.atom_ffn(self.atom_ffn_norm(atom_features))
            )
        return atom_features, edge_features
