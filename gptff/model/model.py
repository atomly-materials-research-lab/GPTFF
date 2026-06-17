import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from gptff.model.basis import FourierAngleBasis, RadialBesselBasis


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


class ThreeBody(nn.Module):
    def __init__(self, atom_fea_len, nbr_fea_len, num_radial, num_angular, device):
        super(ThreeBody, self).__init__()

        self.device = device
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
        """
        atom_fea: [N, atom_fea_len]
        edge_ij: [M, nbr_fea_len]
        graph.triplet_edge_index: [2, L]
        """

        if graph.triplet_edge_index.numel() == 0:
            return edge_ij

        edge_ij_indices = graph.triplet_edge_index[0]
        edge_ik_indices = graph.triplet_edge_index[1]
        triple_i_indices = graph.edge_index[0][edge_ij_indices]
        triple_j_indices = graph.edge_index[1][edge_ij_indices]
        triple_k_indices = graph.edge_index[1][edge_ik_indices]
        atom_fea_ik = torch.cat([atom_fea[triple_i_indices],
                             atom_fea[triple_j_indices],
                             atom_fea[triple_k_indices],
                             edge_ij[edge_ij_indices],
                             edge_ij[edge_ik_indices]], dim=-1)
        
        atom_fea_ik = self.swish(self.W_fea(atom_fea_ik))

        angles_mat = self.angle_embedding(self.angle_basis(graph.triplet_cosine)) # L, nbr_fea_len
        bonds_mat_k = self.bond_embedding_k(triplet_basis_ik) # L, nbr_fea_len
        bonds_mat_j = self.bond_embedding_j(triplet_basis_ij)
        atom_fea_ik = self.sig(self.W_1(atom_fea_ik)) * self.swish(self.W_2(atom_fea_ik)) * bonds_mat_j * bonds_mat_k * angles_mat

        edge_ij = torch.index_add(edge_ij, 0, edge_ij_indices, atom_fea_ik)
        return edge_ij

class EdgeUpdate(nn.Module):
    def __init__(self, atom_fea_len, nbr_fea_len, num_radial) :
        super(EdgeUpdate, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.swish = nn.SiLU()
        self.sig = nn.Sigmoid()
        self.W_1 = nn.Linear(2 * atom_fea_len + nbr_fea_len, nbr_fea_len)
        self.W_2 = nn.Linear(2 * atom_fea_len + nbr_fea_len, nbr_fea_len)
        self.W_r = nn.Linear(num_radial, nbr_fea_len, bias=False)
        self.W_3 = nn.Linear(nbr_fea_len, nbr_fea_len)

    def forward(self, atom_fea, edge_ij, graph, edge_basis):
        atom_nbr_fea = torch.cat([atom_fea[graph.edge_index[0]],
                                  atom_fea[graph.edge_index[1]],
                                  edge_ij], dim=-1)
        
        edge_ij = self.swish(self.W_1(atom_nbr_fea)) * self.sig(self.W_2(atom_nbr_fea))
        
        edge_ij = self.swish(self.W_3(edge_ij)) * self.W_r(edge_basis)

        return edge_ij

class ConvLayer(nn.Module):
    def __init__(self, atom_fea_len, nbr_fea_len, num_radial):
        super(ConvLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full = nn.Linear(2 * atom_fea_len + nbr_fea_len,
                                 2 * atom_fea_len)
        self.sig = nn.Sigmoid()
        self.swish = nn.SiLU()

        self.fc_core = nn.Linear(atom_fea_len, atom_fea_len)
        self.W_r = nn.Linear(num_radial, atom_fea_len, bias=False)
        self.W_1 = nn.Linear(2 * atom_fea_len, atom_fea_len)
        self.W_2 = nn.Linear(2 * atom_fea_len, atom_fea_len)

    def forward(self, atom_fea, edge_ij, edge_basis, graph):
        atom_nbr_fea = torch.cat([atom_fea[graph.edge_index[0]],
                                  atom_fea[graph.edge_index[1]],
                                  edge_ij], dim=-1)
        atom_gated_fea = self.swish(self.fc_full(atom_nbr_fea))

        nbr_all = self.swish(self.W_1(atom_gated_fea)) * self.sig(self.W_2(atom_gated_fea))
        nbr_all = nbr_all * self.W_r(edge_basis)
        atom_fea = torch.index_add(atom_fea, 0, graph.edge_index[0], nbr_all.float())

        return atom_fea


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


class AtomWiseReadout(nn.Module):
    def __init__(self, atom_fea_len):
        super().__init__()
        self.swish = nn.SiLU()
        self.fc1 = nn.Linear(atom_fea_len, atom_fea_len)
        self.fc2 = nn.Linear(atom_fea_len, atom_fea_len)
        self.fc_out = nn.Linear(atom_fea_len, 1)

    def forward(self, atom_fea, atom_batch, num_graphs):
        site_energy = self.swish(self.fc1(atom_fea))
        site_energy = self.swish(self.fc2(site_energy))
        site_energy = self.fc_out(site_energy)
        energy = torch.zeros(
            (num_graphs, 1),
            dtype=site_energy.dtype,
            device=site_energy.device,
        )
        return torch.index_add(energy, 0, atom_batch, site_energy)


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
        out = self.encoder(atom_fea, src_key_padding_mask=masks)
        return out

class tModLodaer_t(nn.Module):
    def __init__(self, CFG
                        ):
        
        super(tModLodaer_t, self).__init__()

        atom_fea_len = CFG.node_feature_len
        nbr_fea_len = CFG.edge_feature_len
        n_layers = CFG.n_layers
        num_radial = getattr(CFG, "num_radial", 16)
        num_angular = getattr(CFG, "num_angular", 4)
        radial_cutoff = getattr(CFG, "radial_cutoff", 5.0)
        angle_cutoff = getattr(CFG, "angle_cutoff", 3.5)
        cutoff_coeff = getattr(CFG, "cutoff_coeff", 5)

        self.device = CFG.device

        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.num_radial = num_radial
        self.num_angular = num_angular
        self.radial_cutoff = radial_cutoff
        self.angle_cutoff = angle_cutoff
        self.atom_embedding = nn.Embedding(95, atom_fea_len, max_norm=True)
        self.edge_rbf = RadialBesselBasis(num_radial, radial_cutoff, cutoff_coeff)
        self.triplet_rbf = RadialBesselBasis(num_radial, angle_cutoff, cutoff_coeff)
        self.w_b = nn.Linear(num_radial, nbr_fea_len, bias=False)
        self.w_eij = nn.Linear(nbr_fea_len* 3, nbr_fea_len)
        self.w_r = nn.Linear(num_radial, nbr_fea_len, bias=False)

        self.convs = nn.ModuleList([ConvLayer(atom_fea_len=atom_fea_len,
                                    nbr_fea_len=nbr_fea_len,
                                    num_radial=num_radial)
                                    for _ in range(n_layers)])
        
        self.three = nn.ModuleList([ThreeBody(atom_fea_len=atom_fea_len,
                                    nbr_fea_len=nbr_fea_len,
                                    num_radial=num_radial,
                                    num_angular=num_angular,
                                    device=self.device) for _ in range(n_layers)])

        self.transformers = nn.ModuleList([TransformerBlock(atom_fea_len) for _ in range(n_layers)])  

        self.edge_updates = nn.ModuleList(EdgeUpdate(atom_fea_len, nbr_fea_len, num_radial)
                                          for _ in range(n_layers))
        
        self.norms1 = nn.ModuleList([nn.LayerNorm(atom_fea_len) for _ in range(n_layers)])
        self.norms2 = nn.ModuleList([nn.LayerNorm(atom_fea_len) for _ in range(n_layers)])

        self.readout = AtomWiseReadout(atom_fea_len)

    def forward(self, graph):
        """
        graph: DifferentiableGraphBatch
        """

        atom_fea = graph.atom_types
        atom_fea = self.atom_embedding(atom_fea) # N, atom_fea_len
        edge_basis = self.edge_rbf(graph.edge_lengths) # M, num_radial
        
        triplet_basis_ij = self.triplet_rbf(graph.triplet_lengths_ij)
        triplet_basis_ik = self.triplet_rbf(graph.triplet_lengths_ik)

        edge_ij = self.w_b(edge_basis)

        edge_ij = torch.cat([atom_fea[graph.edge_index[0]],
                          atom_fea[graph.edge_index[1]],
                          edge_ij
                          ], dim=-1)

        edge_ij = self.w_eij(edge_ij) * self.w_r(edge_basis)
        
        max_len = int(torch.max(graph.num_atoms).item())
        masks = torch.ones((graph.num_atoms.shape[0], max_len), device=atom_fea.device)

        for ii in range(len(masks)):
            masks[ii, :graph.num_atoms[ii]] = 0.0

        masks = masks.to(torch.bool) # bs, max_len

        for edge_func, conv, three, transformer, norm1, norm2 in zip(self.edge_updates, self.convs, self.three, self.transformers, self.norms1, self.norms2):
            edge_ij = three(atom_fea, edge_ij, graph, triplet_basis_ij, triplet_basis_ik)
            edge_ij = edge_ij + edge_func(atom_fea, edge_ij, graph, edge_basis)
            atom_fea = norm1(conv(atom_fea, edge_ij, edge_basis, graph))

            atom_fea_list = []

            c_atom = 0
            for n_atom in graph.num_atoms.tolist():
                atom_fea_list.append(atom_fea[c_atom:c_atom+n_atom])
                c_atom += n_atom

            atom_fea_list = pad_sequence(atom_fea_list, batch_first=True) # bs, seq_len, fea_len

            atom_fea = norm2(transformer(atom_fea_list, masks))[masks == False, :] + atom_fea # [masks == False, :]

        return self.readout(atom_fea, graph.atom_batch, graph.num_atoms.shape[0])


class tModLodaer(nn.Module):
    def __init__(self, CFG
                        ):
        
        super(tModLodaer, self).__init__()
        
        atom_fea_len = CFG.node_feature_len
        nbr_fea_len = CFG.edge_feature_len
        n_layers = CFG.n_layers
        num_radial = getattr(CFG, "num_radial", 16)
        num_angular = getattr(CFG, "num_angular", 4)
        radial_cutoff = getattr(CFG, "radial_cutoff", 5.0)
        angle_cutoff = getattr(CFG, "angle_cutoff", 3.5)
        cutoff_coeff = getattr(CFG, "cutoff_coeff", 5)
        max_atomic_number = getattr(CFG, "max_atomic_number", 94)

        self.device = CFG.device

        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.num_radial = num_radial
        self.num_angular = num_angular
        self.radial_cutoff = radial_cutoff
        self.angle_cutoff = angle_cutoff
        self.max_atomic_number = max_atomic_number
        self.atom_embedding = AtomEmbedding(atom_fea_len, max_atomic_number=max_atomic_number)
        self.edge_embedding = EdgeEmbedding(atom_fea_len, nbr_fea_len, num_radial)
        self.edge_rbf = RadialBesselBasis(num_radial, radial_cutoff, cutoff_coeff)
        self.triplet_rbf = RadialBesselBasis(num_radial, angle_cutoff, cutoff_coeff)

        self.interactions = nn.ModuleList([
            InteractionBlock(
                atom_fea_len=atom_fea_len,
                nbr_fea_len=nbr_fea_len,
                num_radial=num_radial,
                num_angular=num_angular,
            )
            for _ in range(n_layers)
        ])

        self.readout = AtomWiseReadout(atom_fea_len)

    def forward(self, graph):
        """
        graph: DifferentiableGraphBatch
        """

        atom_fea = self.atom_embedding(graph.atom_types)
        edge_basis = self.edge_rbf(graph.edge_lengths)
        
        triplet_basis_ij = self.triplet_rbf(graph.triplet_lengths_ij)
        triplet_basis_ik = self.triplet_rbf(graph.triplet_lengths_ik)
        edge_ij = self.edge_embedding(atom_fea, graph.edge_index, edge_basis)
        
        for interaction in self.interactions:
            atom_fea, edge_ij = interaction(
                atom_fea,
                edge_ij,
                graph,
                edge_basis,
                triplet_basis_ij,
                triplet_basis_ik,
            )

        return self.readout(atom_fea, graph.atom_batch, graph.num_atoms.shape[0])
