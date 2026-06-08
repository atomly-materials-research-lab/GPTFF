from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import numpy as np


def _batch_index(n_atoms):
    return torch.repeat_interleave(torch.arange(n_atoms.shape[0], device=n_atoms.device), n_atoms)


def _padding_mask(n_atoms):
    max_len = int(n_atoms.max().item())
    positions = torch.arange(max_len, device=n_atoms.device)
    return positions.unsqueeze(0) >= n_atoms.unsqueeze(1)


def _cuda_math_sdp_context():
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        return sdpa_kernel([SDPBackend.MATH])
    except (ImportError, AttributeError):
        if torch.cuda.is_available():
            return torch.backends.cuda.sdp_kernel(
                enable_flash=False,
                enable_math=True,
                enable_mem_efficient=False,
            )
        return nullcontext()


def _three_body_indices(nbr_atoms, bond_pairs_indices, n_bond_pairs_bond):
    nbr_i = nbr_atoms[:, 0]
    nbr_j = nbr_atoms[:, 1]
    pair_j = bond_pairs_indices[:, 0]
    pair_k = bond_pairs_indices[:, 1]
    bond_pair_target = torch.repeat_interleave(
        torch.arange(n_bond_pairs_bond.shape[0], device=n_bond_pairs_bond.device),
        n_bond_pairs_bond,
    )
    return nbr_i, nbr_j, nbr_i.index_select(0, pair_k), nbr_j.index_select(0, pair_j), nbr_j.index_select(0, pair_k), bond_pair_target


def ebf(d_ij, rcut, device=None, filters=None):
    if filters is None:
        filter_device = d_ij.device if device is None else device
        filters = torch.arange(16, device=filter_device, dtype=d_ij.dtype)
    else:
        filters = filters.to(device=d_ij.device, dtype=d_ij.dtype)
    scale = (2.0 / float(rcut)) ** 0.5
    return scale * torch.sin(filters * (torch.pi / float(rcut)) * d_ij) / d_ij


class _RuntimeModelBase(nn.Module):
    def _init_runtime_buffers(self):
        self.register_buffer("ebf_filters", torch.arange(16, dtype=torch.float32), persistent=False)

    def _radial_basis(self, distances, rcut):
        return ebf(distances, rcut, filters=self.ebf_filters)


class ThreeBody(nn.Module):
    def __init__(self, atom_fea_len, nbr_fea_len, device):
        super(ThreeBody, self).__init__()

        self.device = device
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.angle_embedding = nn.Linear(1, nbr_fea_len)
        self.bond_embedding_k = nn.Linear(16, nbr_fea_len)
        self.bond_embedding_j = nn.Linear(16, nbr_fea_len)

        self.sig = nn.Sigmoid()
        self.swish = nn.SiLU()

        self.W_fea = nn.Linear(3 * atom_fea_len + 2 * nbr_fea_len, nbr_fea_len)
        self.W_1 = nn.Linear(nbr_fea_len, nbr_fea_len)
        self.W_2 = nn.Linear(nbr_fea_len, nbr_fea_len)

    def forward(self, atom_fea, edge_ij, triple_dist_ij, triple_dist_ik, triple_a_jik, nbr_atoms, bond_pairs_indices, n_bond_pairs_bond, three_body_indices=None):
        """
        atom_fea: [N, atom_fea_len]
        edge_ij: [M, nbr_fea_len]
        bonds_r: [M]
        triple_dist_ik: [L, 16]
        triple_a_jik: [L]
        nbr_atoms: [M, 2]
        n_bond_pairs_bond: [M]
        """

        if three_body_indices is None:
            _, _, triple_i_indices, triple_j_indices, triple_k_indices, bond_pair_target = _three_body_indices(
                nbr_atoms, bond_pairs_indices, n_bond_pairs_bond
            )
        else:
            _, _, triple_i_indices, triple_j_indices, triple_k_indices, bond_pair_target = three_body_indices
        atom_fea_ik = torch.cat([atom_fea[triple_i_indices],
                             atom_fea[triple_j_indices],
                             atom_fea[triple_k_indices],
                             edge_ij[bond_pairs_indices[:, 0]],
                             edge_ij[bond_pairs_indices[:, 1]]], dim=-1)
        
        atom_fea_ik = self.swish(self.W_fea(atom_fea_ik))

        angles_mat = self.angle_embedding(triple_a_jik.unsqueeze(-1)) # L, nbr_fea_len
        bonds_mat_k = self.bond_embedding_k(triple_dist_ik) # L, nbr_fea_len
        bonds_mat_j = self.bond_embedding_j(triple_dist_ij)
        atom_fea_ik = self.sig(self.W_1(atom_fea_ik)) * self.swish(self.W_2(atom_fea_ik)) * bonds_mat_j * bonds_mat_k * angles_mat
        # mat = angles_mat * bonds_mat * atom_fea_ik

        # atom_nbr_fea_ik = self.sig(mat)  # N, M, M-1, atom_fea
        edge_ij = torch.index_add(edge_ij, 0, bond_pair_target, atom_fea_ik)
        return edge_ij

class EdgeUpdate(nn.Module):
    def __init__(self, atom_fea_len, nbr_fea_len) :
        super(EdgeUpdate, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.swish = nn.SiLU()
        self.sig = nn.Sigmoid()
        self.W_1 = nn.Linear(2 * atom_fea_len + nbr_fea_len, nbr_fea_len)
        self.W_2 = nn.Linear(2 * atom_fea_len + nbr_fea_len, nbr_fea_len)
        self.W_r = nn.Linear(16, nbr_fea_len)
        self.W_3 = nn.Linear(nbr_fea_len, nbr_fea_len)

    def forward(self, atom_fea, edge_ij, nbr_atoms, bonds_r, nbr_i=None, nbr_j=None):
        if nbr_i is None:
            nbr_i = nbr_atoms[:, 0]
        if nbr_j is None:
            nbr_j = nbr_atoms[:, 1]
        atom_nbr_fea = torch.cat([atom_fea[nbr_i],
                                  atom_fea[nbr_j],
                                  edge_ij], dim=-1)
        
        edge_ij = self.swish(self.W_1(atom_nbr_fea)) * self.sig(self.W_2(atom_nbr_fea))
        
        edge_ij = self.swish(self.W_3(edge_ij)) * self.W_r(bonds_r)

        return edge_ij

class ConvLayer(nn.Module):
    def __init__(self, atom_fea_len, nbr_fea_len):
        super(ConvLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full = nn.Linear(2 * atom_fea_len + nbr_fea_len,
                                 2 * atom_fea_len)
        self.sig = nn.Sigmoid()
        self.swish = nn.SiLU()

        self.fc_core = nn.Linear(atom_fea_len, atom_fea_len)
        self.W_r = nn.Linear(16, atom_fea_len)
        self.W_1 = nn.Linear(2 * atom_fea_len, atom_fea_len)
        self.W_2 = nn.Linear(2 * atom_fea_len, atom_fea_len)

    def forward(self, atom_fea, edge_ij, bonds_r, nbr_atoms, nbr_i=None, nbr_j=None):
        if nbr_i is None:
            nbr_i = nbr_atoms[:, 0]
        if nbr_j is None:
            nbr_j = nbr_atoms[:, 1]
        atom_nbr_fea = torch.cat([atom_fea[nbr_i],
                                  atom_fea[nbr_j],
                                  edge_ij], dim=-1)
        atom_gated_fea = self.swish(self.fc_full(atom_nbr_fea))

        nbr_all = self.swish(self.W_1(atom_gated_fea)) * self.sig(self.W_2(atom_gated_fea))
        nbr_all = nbr_all * self.W_r(bonds_r)
        atom_fea = torch.index_add(atom_fea, 0, nbr_i, nbr_all.float())

        return atom_fea

class Attention(nn.Module):
    def __init__(self, d_model, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((heads, 1, 1))), requires_grad=True)

        self.softmax = nn.Softmax(dim=-1)

        self.to_qkv = nn.Linear(d_model, inner_dim * 3, bias=True)
        self.to_out = nn.Linear(inner_dim, d_model, bias=False)

    def forward(self, x):
        B_, N, C = x.shape

        qkv = self.to_qkv(x)
        qkv = qkv.reshape(B_, N, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # scaled cosine attention
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale,
                                  max=torch.log(torch.tensor(1. / 0.01, device=self.logit_scale.device))).exp()
        attn = attn * logit_scale
        attn = self.softmax(attn)

        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.to_out(out)
        
class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1) -> None:
        super().__init__()

        self.self_attn = Attention(d_model, nhead, 64)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model),
        )

    def forward(self, x):
        # x: (B, N, d_model)
        x = x + self.norm1(self.self_attn(x))
        x = x + self.norm2(self.ffn(x))
        return x
    
class TransformerBlock(nn.Module):
    def __init__(self, atom_fea_len):
        super(TransformerBlock, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=atom_fea_len, nhead=2, dim_feedforward=256, activation='gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, atom_fea, masks):
        if self.training and atom_fea.is_cuda:
            with _cuda_math_sdp_context():
                out = self.encoder(atom_fea, src_key_padding_mask=masks)
        else:
            out = self.encoder(atom_fea, src_key_padding_mask=masks)
        return out

class tModLodaer_t(_RuntimeModelBase):
    def __init__(self, CFG
                        ):
        
        super(tModLodaer_t, self).__init__()
        self._init_runtime_buffers()

        atom_fea_len = CFG.node_feature_len
        nbr_fea_len = CFG.edge_feature_len
        n_layers = CFG.n_layers

        self.device = CFG.device

        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.atom_embedding = nn.Embedding(95, atom_fea_len, max_norm=True)
        self.w_b = nn.Linear(16, nbr_fea_len)
        self.w_eij = nn.Linear(nbr_fea_len* 3, nbr_fea_len)
        self.w_r = nn.Linear(16, nbr_fea_len)

        self.convs = nn.ModuleList([ConvLayer(atom_fea_len=atom_fea_len,
                                    nbr_fea_len=nbr_fea_len)
                                    for _ in range(n_layers)])
        
        self.three = nn.ModuleList([ThreeBody(atom_fea_len=atom_fea_len,
                                    nbr_fea_len=nbr_fea_len, device=self.device) for _ in range(n_layers)])

        self.transformers = nn.ModuleList([TransformerBlock(atom_fea_len) for _ in range(n_layers)])  

        self.edge_updates = nn.ModuleList(EdgeUpdate(atom_fea_len, nbr_fea_len)
                                          for _ in range(n_layers))
        
        self.norms1 = nn.ModuleList([nn.LayerNorm(atom_fea_len) for _ in range(n_layers)])
        self.norms2 = nn.ModuleList([nn.LayerNorm(atom_fea_len) for _ in range(n_layers)])

        self.swish = nn.SiLU()
        self.fc1 = nn.Linear(atom_fea_len, atom_fea_len)
        self.fc2 = nn.Linear(atom_fea_len, atom_fea_len)
        
        self.fc_out = nn.Linear(atom_fea_len, 1)

    def forward(self, atom_fea, bonds_r, n_atoms, triple_dist_ij, triple_dist_ik, triple_a_jik, nbr_atoms, n_bond_pairs_bond, bond_pairs_indices):
        """
        atom_fea: [N,1]
        bonds_r: [M]
        triple_dist_ij: [L]
        triple_dist_ik: [L]
        triple_a_jik: [L]
        nbr_atoms: [M, 2]
        n_bond_pairs_bond: [M]
        """

        atom_fea = atom_fea.view(-1)
        atom_fea = self.atom_embedding(atom_fea) # N, atom_fea_len
        bonds_dist = self._radial_basis(bonds_r.unsqueeze(-1), 5.0) # M, 16
        
        triple_dist_ij = self._radial_basis(triple_dist_ij.unsqueeze(-1), 3.5)
        triple_dist_ik = self._radial_basis(triple_dist_ik.unsqueeze(-1), 3.5)

        three_body_indices = _three_body_indices(nbr_atoms, bond_pairs_indices, n_bond_pairs_bond)
        nbr_i, nbr_j = three_body_indices[:2]
        atom_batch_index = _batch_index(n_atoms)

        edge_ij = self.w_b(bonds_dist)

        edge_ij = torch.cat([atom_fea[nbr_i],
                          atom_fea[nbr_j],
                          edge_ij
                          ], dim=-1)

        edge_ij = self.w_eij(edge_ij) * self.w_r(bonds_dist)
        
        masks = _padding_mask(n_atoms)
        n_atoms_list = [int(n_atom) for n_atom in n_atoms.detach().cpu().tolist()]

        for edge_func, conv, three, transformer, norm1, norm2 in zip(self.edge_updates, self.convs, self.three, self.transformers, self.norms1, self.norms2):
            edge_ij = three(
                atom_fea,
                edge_ij,
                triple_dist_ij,
                triple_dist_ik,
                triple_a_jik,
                nbr_atoms,
                bond_pairs_indices,
                n_bond_pairs_bond,
                three_body_indices=three_body_indices,
            )
            edge_ij = edge_ij + edge_func(atom_fea, edge_ij, nbr_atoms, bonds_dist, nbr_i=nbr_i, nbr_j=nbr_j)
            atom_fea = norm1(conv(atom_fea, edge_ij, bonds_dist, nbr_atoms, nbr_i=nbr_i, nbr_j=nbr_j))

            atom_fea_list = pad_sequence(torch.split(atom_fea, n_atoms_list), batch_first=True) # bs, seq_len, fea_len

            # atom_fea = torch.cat([transformer(torch.index_select(atom_fea, 0, idx_map)) for idx_map in crystal_atom_idx], dim=0) + atom_fea
            atom_fea = norm2(transformer(atom_fea_list, masks))[~masks, :] + atom_fea # [masks == False, :]

        cry_fea = atom_fea.new_zeros((n_atoms.shape[0], self.atom_fea_len))
        cry_fea = torch.index_add(cry_fea, 0, atom_batch_index, atom_fea)
        
        cry_fea = self.swish(self.fc1(cry_fea))
        cry_fea = self.swish(self.fc2(cry_fea))
        
        out = self.fc_out(cry_fea)

        return out


class tModLodaer(_RuntimeModelBase):
    def __init__(self, CFG
                        ):
        
        super(tModLodaer, self).__init__()
        self._init_runtime_buffers()
        
        atom_fea_len = CFG.node_feature_len
        nbr_fea_len = CFG.edge_feature_len
        n_layers = CFG.n_layers

        self.device = CFG.device

        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.atom_embedding = nn.Embedding(95, atom_fea_len, max_norm=True)
        self.w_b = nn.Linear(16, nbr_fea_len)
        self.w_eij = nn.Linear(nbr_fea_len* 3, nbr_fea_len)
        self.w_r = nn.Linear(16, nbr_fea_len)

        self.convs = nn.ModuleList([ConvLayer(atom_fea_len=atom_fea_len,
                                    nbr_fea_len=nbr_fea_len)
                                    for _ in range(n_layers)])
        
        self.three = nn.ModuleList([ThreeBody(atom_fea_len=atom_fea_len,
                                    nbr_fea_len=nbr_fea_len, device=self.device) for _ in range(n_layers)])

        self.edge_updates = nn.ModuleList(EdgeUpdate(atom_fea_len, nbr_fea_len)
                                          for _ in range(n_layers))

        self.swish = nn.SiLU()
        self.fc1 = nn.Linear(atom_fea_len, atom_fea_len)
        self.fc2 = nn.Linear(atom_fea_len, atom_fea_len)
        
        self.fc_out = nn.Linear(atom_fea_len, 1)

    def forward(self, atom_fea, bonds_r, n_atoms, triple_dist_ij, triple_dist_ik, triple_a_jik, nbr_atoms, n_bond_pairs_bond, bond_pairs_indices):
        """
        atom_fea: [N,1]
        bonds_r: [M]
        triple_dist_ij: [L]
        triple_dist_ik: [L]
        triple_a_jik: [L]
        nbr_atoms: [M, 2]
        n_bond_pairs_bond: [M]
        """

        atom_fea = atom_fea.view(-1)
        atom_fea = self.atom_embedding(atom_fea) # N, atom_fea_len
        bonds_dist = self._radial_basis(bonds_r.unsqueeze(-1), 5.0) # M, 16
        
        triple_dist_ij = self._radial_basis(triple_dist_ij.unsqueeze(-1), 3.5)
        triple_dist_ik = self._radial_basis(triple_dist_ik.unsqueeze(-1), 3.5)

        three_body_indices = _three_body_indices(nbr_atoms, bond_pairs_indices, n_bond_pairs_bond)
        nbr_i, nbr_j = three_body_indices[:2]
        atom_batch_index = _batch_index(n_atoms)

        edge_ij = self.w_b(bonds_dist)

        edge_ij = torch.cat([atom_fea[nbr_i],
                          atom_fea[nbr_j],
                          edge_ij
                          ], dim=-1)

        edge_ij = self.w_eij(edge_ij) * self.w_r(bonds_dist)
        
        for edge_func, conv, three in zip(self.edge_updates, self.convs, self.three):
            edge_ij = three(
                atom_fea,
                edge_ij,
                triple_dist_ij,
                triple_dist_ik,
                triple_a_jik,
                nbr_atoms,
                bond_pairs_indices,
                n_bond_pairs_bond,
                three_body_indices=three_body_indices,
            )
            edge_ij = edge_ij + edge_func(atom_fea, edge_ij, nbr_atoms, bonds_dist, nbr_i=nbr_i, nbr_j=nbr_j)
            atom_fea = conv(atom_fea, edge_ij, bonds_dist, nbr_atoms, nbr_i=nbr_i, nbr_j=nbr_j)

        cry_fea = atom_fea.new_zeros((n_atoms.shape[0], self.atom_fea_len))
        cry_fea = torch.index_add(cry_fea, 0, atom_batch_index, atom_fea)
        
        cry_fea = self.swish(self.fc1(cry_fea))
        cry_fea = self.swish(self.fc2(cry_fea))
        
        out = self.fc_out(cry_fea)

        return out
