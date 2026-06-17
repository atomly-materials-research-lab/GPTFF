import torch
import torch.nn as nn

from gptff.model.basis import FourierAngleBasis


class EdgeUpdate(nn.Module):
    def __init__(self, atom_fea_len, nbr_fea_len, num_radial):
        super().__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.swish = nn.SiLU()
        self.sig = nn.Sigmoid()
        self.W_1 = nn.Linear(2 * atom_fea_len + nbr_fea_len, nbr_fea_len)
        self.W_2 = nn.Linear(2 * atom_fea_len + nbr_fea_len, nbr_fea_len)
        self.W_r = nn.Linear(num_radial, nbr_fea_len, bias=False)
        self.W_3 = nn.Linear(nbr_fea_len, nbr_fea_len)

    def forward(self, atom_fea, edge_ij, graph, edge_basis):
        atom_nbr_fea = torch.cat([
            atom_fea[graph.edge_index[0]],
            atom_fea[graph.edge_index[1]],
            edge_ij,
        ], dim=-1)

        edge_ij = self.swish(self.W_1(atom_nbr_fea)) * self.sig(self.W_2(atom_nbr_fea))
        edge_ij = self.swish(self.W_3(edge_ij)) * self.W_r(edge_basis)
        return edge_ij


class ThreeBodyEdgeDelta(nn.Module):
    def __init__(self, atom_fea_len, nbr_fea_len, num_radial, num_angular):
        super().__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.angle_basis = FourierAngleBasis(num_angular)
        self.angle_embedding = nn.Linear(self.angle_basis.out_dim, nbr_fea_len, bias=False)
        self.bond_embedding_k = nn.Linear(num_radial, nbr_fea_len, bias=False)
        self.bond_embedding_j = nn.Linear(num_radial, nbr_fea_len, bias=False)

        self.sig = nn.Sigmoid()
        self.swish = nn.SiLU()

        self.W_fea = nn.Linear(3 * atom_fea_len + 2 * nbr_fea_len, nbr_fea_len)
        self.W_1 = nn.Linear(nbr_fea_len, nbr_fea_len)
        self.W_2 = nn.Linear(nbr_fea_len, nbr_fea_len)

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

        atom_fea_ik = self.swish(self.W_fea(atom_fea_ik))

        angles_mat = self.angle_embedding(self.angle_basis(graph.triplet_cosine))
        bonds_mat_k = self.bond_embedding_k(triplet_basis_ik)
        bonds_mat_j = self.bond_embedding_j(triplet_basis_ij)
        atom_fea_ik = (
            self.sig(self.W_1(atom_fea_ik))
            * self.swish(self.W_2(atom_fea_ik))
            * bonds_mat_j
            * bonds_mat_k
            * angles_mat
        )

        edge_delta = edge_ij.new_zeros(edge_ij.shape)
        return torch.index_add(edge_delta, 0, edge_ij_indices, atom_fea_ik)


class AtomFeatureDelta(nn.Module):
    def __init__(self, atom_fea_len, nbr_fea_len, num_radial):
        super().__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full = nn.Linear(2 * atom_fea_len + nbr_fea_len, 2 * atom_fea_len)
        self.sig = nn.Sigmoid()
        self.swish = nn.SiLU()

        self.W_r = nn.Linear(num_radial, atom_fea_len, bias=False)
        self.W_1 = nn.Linear(2 * atom_fea_len, atom_fea_len)
        self.W_2 = nn.Linear(2 * atom_fea_len, atom_fea_len)

    def forward(self, atom_fea, edge_ij, edge_basis, graph):
        atom_nbr_fea = torch.cat([
            atom_fea[graph.edge_index[0]],
            atom_fea[graph.edge_index[1]],
            edge_ij,
        ], dim=-1)
        atom_gated_fea = self.swish(self.fc_full(atom_nbr_fea))

        atom_msg = self.swish(self.W_1(atom_gated_fea)) * self.sig(self.W_2(atom_gated_fea))
        atom_msg = atom_msg * self.W_r(edge_basis)

        atom_delta = atom_fea.new_zeros(atom_fea.shape)
        return torch.index_add(atom_delta, 0, graph.edge_index[0], atom_msg.to(atom_delta.dtype))


class InteractionBlock(nn.Module):
    def __init__(self, atom_fea_len, nbr_fea_len, num_radial, num_angular):
        super().__init__()
        self.three_body = ThreeBodyEdgeDelta(
            atom_fea_len=atom_fea_len,
            nbr_fea_len=nbr_fea_len,
            num_radial=num_radial,
            num_angular=num_angular,
        )
        self.edge_update = EdgeUpdate(atom_fea_len, nbr_fea_len, num_radial)
        self.atom_update = AtomFeatureDelta(atom_fea_len, nbr_fea_len, num_radial)
        self.triplet_edge_norm = nn.LayerNorm(nbr_fea_len)
        self.pair_edge_norm = nn.LayerNorm(nbr_fea_len)
        self.atom_norm = nn.LayerNorm(atom_fea_len)

    def forward(self, atom_fea, edge_ij, graph, edge_basis, triplet_basis_ij, triplet_basis_ik):
        triplet_delta = self.three_body(
            atom_fea,
            edge_ij,
            graph,
            triplet_basis_ij,
            triplet_basis_ik,
        )
        edge_ij = self.triplet_edge_norm(edge_ij + triplet_delta)

        pair_delta = self.edge_update(atom_fea, edge_ij, graph, edge_basis)
        edge_ij = self.pair_edge_norm(edge_ij + pair_delta)

        atom_delta = self.atom_update(atom_fea, edge_ij, edge_basis, graph)
        atom_fea = self.atom_norm(atom_fea + atom_delta)
        return atom_fea, edge_ij
