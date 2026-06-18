from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from gptff.model.basis import RadialBesselBasis


class AtomEmbedding(nn.Module):
    def __init__(self, atom_feature_dim, max_atomic_number=94, normalize=True):
        super().__init__()
        if max_atomic_number < 1:
            raise ValueError("max_atomic_number must be positive.")

        self.max_atomic_number = int(max_atomic_number)
        self.embedding = nn.Embedding(self.max_atomic_number + 1, atom_feature_dim)
        self.norm = nn.LayerNorm(atom_feature_dim) if normalize else nn.Identity()

    def forward(self, atom_types):
        if atom_types.numel() > 0:
            torch._assert(
                torch.all((atom_types >= 1) & (atom_types <= self.max_atomic_number)),
                f"Atomic numbers must be in the range [1, {self.max_atomic_number}].",
            )
        return self.norm(self.embedding(atom_types))


@dataclass(frozen=True)
class EdgeModulation:
    atom_message: torch.Tensor
    edge_message: torch.Tensor


@dataclass(frozen=True)
class GeometryFeatures:
    edge_basis: torch.Tensor
    edge_cutoff: torch.Tensor
    angle_radial_basis: torch.Tensor
    edge_features: torch.Tensor
    edge_modulation: EdgeModulation
    triplet_modulation: torch.Tensor


class EdgeModulationProjection(nn.Module):
    def __init__(self, atom_feature_dim, edge_feature_dim, num_radial):
        super().__init__()
        self.atom_message_weight = nn.Linear(num_radial, atom_feature_dim, bias=False)
        self.edge_message_weight = nn.Linear(num_radial, edge_feature_dim, bias=False)

    def forward(self, edge_basis):
        return EdgeModulation(
            atom_message=self.atom_message_weight(edge_basis),
            edge_message=self.edge_message_weight(edge_basis),
        )


class TripletModulationProjection(nn.Module):
    def __init__(self, edge_feature_dim, num_radial):
        super().__init__()
        self.triplet_weight = nn.Linear(num_radial, edge_feature_dim, bias=False)

    def forward(self, angle_radial_basis):
        return self.triplet_weight(angle_radial_basis)


class EdgeEmbedding(nn.Module):
    def __init__(self, edge_feature_dim, num_radial):
        super().__init__()
        self.edge_embedding = nn.Sequential(
            nn.Linear(num_radial, edge_feature_dim, bias=False),
            nn.SiLU(),
            nn.Linear(edge_feature_dim, edge_feature_dim, bias=False),
        )

    def forward(self, edge_basis):
        return self.edge_embedding(edge_basis)


class GeometryEmbedding(nn.Module):
    def __init__(
        self,
        atom_feature_dim,
        edge_feature_dim,
        num_radial,
        radial_cutoff,
        angle_cutoff,
        cutoff_coeff,
    ):
        super().__init__()
        self.edge_rbf = RadialBesselBasis(num_radial, radial_cutoff, cutoff_coeff)
        self.angle_edge_rbf = RadialBesselBasis(num_radial, angle_cutoff, cutoff_coeff)
        self.edge_embedding = EdgeEmbedding(edge_feature_dim, num_radial)
        self.edge_modulation = EdgeModulationProjection(atom_feature_dim, edge_feature_dim, num_radial)
        self.triplet_modulation = TripletModulationProjection(edge_feature_dim, num_radial)

    def forward(self, graph):
        edge_basis = self.edge_rbf(graph.edge_lengths)
        angle_radial_basis = self.angle_edge_rbf(graph.edge_lengths)
        edge_cutoff = self.edge_rbf.cutoff_fn(graph.edge_lengths.reshape(-1, 1))
        return GeometryFeatures(
            edge_basis=edge_basis,
            edge_cutoff=edge_cutoff,
            angle_radial_basis=angle_radial_basis,
            edge_features=self.edge_embedding(edge_basis),
            edge_modulation=self.edge_modulation(edge_basis),
            triplet_modulation=self.triplet_modulation(angle_radial_basis),
        )
