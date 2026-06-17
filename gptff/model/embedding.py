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


class EdgeEmbedding(nn.Module):
    def __init__(self, atom_fea_len, nbr_fea_len, num_radial, normalize=True):
        super().__init__()
        self.bond_embedding = nn.Linear(num_radial, nbr_fea_len, bias=False)
        self.edge_embedding = nn.Linear(2 * atom_fea_len + nbr_fea_len, nbr_fea_len)
        self.radial_gate = nn.Linear(num_radial, nbr_fea_len, bias=False)
        self.norm = nn.LayerNorm(nbr_fea_len) if normalize else nn.Identity()

    def forward(self, atom_fea, edge_index, edge_basis):
        bond_fea = self.bond_embedding(edge_basis)
        edge_fea = torch.cat([
            atom_fea[edge_index[0]],
            atom_fea[edge_index[1]],
            bond_fea,
        ], dim=-1)
        edge_fea = self.edge_embedding(edge_fea) * self.radial_gate(edge_basis)
        return self.norm(edge_fea)
