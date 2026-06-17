from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


class AtomEmbedding(nn.Module):
    def __init__(self, atom_fea_len, max_atomic_number=94, normalize=True):
        super().__init__()
        if max_atomic_number < 1:
            raise ValueError("max_atomic_number must be positive.")

        self.max_atomic_number = int(max_atomic_number)
        self.embedding = nn.Embedding(self.max_atomic_number + 1, atom_fea_len)
        self.norm = nn.LayerNorm(atom_fea_len) if normalize else nn.Identity()

    def forward(self, atom_types):
        if atom_types.numel() > 0:
            torch._assert(
                torch.all((atom_types >= 1) & (atom_types <= self.max_atomic_number)),
                f"Atomic numbers must be in the range [1, {self.max_atomic_number}].",
            )
        return self.norm(self.embedding(atom_types))


@dataclass(frozen=True)
class EdgeModulation:
    atom: torch.Tensor
    edge: torch.Tensor


class EdgeModulationProjection(nn.Module):
    def __init__(self, atom_fea_len, nbr_fea_len, num_radial):
        super().__init__()
        self.atom_weight = nn.Linear(num_radial, atom_fea_len, bias=False)
        self.edge_weight = nn.Linear(num_radial, nbr_fea_len, bias=False)

    def forward(self, edge_basis):
        return EdgeModulation(
            atom=self.atom_weight(edge_basis),
            edge=self.edge_weight(edge_basis),
        )


class EdgeEmbedding(nn.Module):
    def __init__(self, nbr_fea_len, num_radial, normalize=True):
        super().__init__()
        self.edge_embedding = nn.Sequential(
            nn.Linear(num_radial, nbr_fea_len, bias=False),
            nn.SiLU(),
            nn.Linear(nbr_fea_len, nbr_fea_len, bias=False),
        )
        self.norm = (
            nn.LayerNorm(nbr_fea_len, elementwise_affine=False)
            if normalize
            else nn.Identity()
        )

    def forward(self, edge_basis):
        return self.norm(self.edge_embedding(edge_basis))
