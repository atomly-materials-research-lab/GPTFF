"""TorchSim interface for GPTFF models.

This module is intentionally optional: importing GPTFF's ASE calculator does
not require TorchSim, while users who have torch-sim-atomistic installed can
use GPTFFTorchSimModel directly with TorchSim integrators.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import Tensor

from gptff.model import model
from gptff.model.mpredict import CFG, _load_checkpoint, atom_refs

try:
    import torch_sim as ts
    from torch_sim.models.interface import ModelInterface
    from torch_sim.neighbors import torchsim_nl
except ImportError as exc:  # pragma: no cover - exercised only without TorchSim
    raise ImportError(
        "GPTFFTorchSimModel requires torch-sim-atomistic. "
        "Install GPTFF with the torchsim extra or install torch-sim-atomistic separately."
    ) from exc


NeighborListFn = Callable[
    [Tensor, Tensor, Tensor, Tensor, Tensor, bool],
    tuple[Tensor, Tensor, Tensor],
]


def _sort_edges_by_center(
    nbr_atoms: Tensor,
    pair_vec: Tensor,
    pair_dist: Tensor,
    mapping_system: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Keep all edge-aligned tensors in center-atom order."""

    if nbr_atoms.shape[0] <= 1:
        return nbr_atoms, pair_vec, pair_dist, mapping_system

    centers = nbr_atoms[:, 0]
    try:
        order = torch.argsort(centers, stable=True)
    except TypeError:  # older PyTorch fallback
        order = torch.argsort(centers)
    return nbr_atoms[order], pair_vec[order], pair_dist[order], mapping_system[order]


def _angle_edge_pairs(nbr_atoms: Tensor, pair_dist: Tensor, angle_cutoff: float) -> tuple[Tensor, Tensor]:
    """Build GPTFF ordered three-body edge pairs from TorchSim edges.

    GPTFF represents an angle j-i-k by two directed bonds i->j and i->k.  For
    every angle-cutoff bond used as the target i->j, all other angle-cutoff
    bonds with the same center atom become i->k candidates.  The returned rows
    are ordered by target edge, matching GPTFF's n_bond_pairs_bond contract.
    """

    num_edges = int(nbr_atoms.shape[0])
    device = nbr_atoms.device
    n_bond_pairs_bond = torch.zeros(num_edges, dtype=torch.long, device=device)
    if num_edges == 0:
        return torch.empty((0, 2), dtype=torch.long, device=device), n_bond_pairs_bond

    eligible = torch.nonzero(pair_dist <= angle_cutoff, as_tuple=False).view(-1)
    if eligible.numel() == 0:
        return torch.empty((0, 2), dtype=torch.long, device=device), n_bond_pairs_bond

    centers = nbr_atoms[eligible, 0]
    _, counts = torch.unique_consecutive(centers, return_counts=True)
    if counts.numel() == 0 or int(counts.max().item()) < 2:
        return torch.empty((0, 2), dtype=torch.long, device=device), n_bond_pairs_bond

    counts_per_edge = torch.repeat_interleave(counts, counts)
    n_bond_pairs_bond[eligible] = counts_per_edge - 1

    group_count = int(counts.shape[0])
    max_count = int(counts.max().item())
    group_ids = torch.repeat_interleave(torch.arange(group_count, device=device), counts)
    group_starts = torch.repeat_interleave(torch.cumsum(counts, dim=0) - counts, counts)
    positions_in_group = torch.arange(eligible.numel(), device=device) - group_starts

    padded = eligible.new_full((group_count, max_count), -1)
    padded[group_ids, positions_in_group] = eligible

    target_edges = padded[:, :, None].expand(-1, -1, max_count)
    other_edges = padded[:, None, :].expand(-1, max_count, -1)
    keep = (target_edges >= 0) & (other_edges >= 0) & (target_edges != other_edges)
    bond_pairs_indices = torch.stack((target_edges[keep], other_edges[keep]), dim=1)
    return bond_pairs_indices.to(dtype=torch.long), n_bond_pairs_bond


def _triple_features(pair_vec: Tensor, pair_dist: Tensor, bond_pairs_indices: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    if bond_pairs_indices.numel() == 0:
        empty = pair_dist.new_empty((0,))
        return empty, empty, empty

    pair_j = bond_pairs_indices[:, 0]
    pair_k = bond_pairs_indices[:, 1]
    triple_vec_ij = pair_vec[pair_j]
    triple_vec_ik = pair_vec[pair_k]
    triple_dist_ij = pair_dist[pair_j]
    triple_dist_ik = pair_dist[pair_k]
    triple_a_jik = (triple_vec_ij * triple_vec_ik).sum(dim=1) / (triple_dist_ij * triple_dist_ik)
    triple_a_jik = torch.clamp(triple_a_jik, -1.0, 1.0) * (1 - 1e-6)
    return triple_dist_ij, triple_dist_ik, triple_a_jik


class GPTFFTorchSimModel(ModelInterface):
    """TorchSim-native GPTFF model wrapper.

    The wrapper reuses GPTFF's pretrained V1/V2 model classes and checkpoint
    format.  It replaces the ASE/Cython graph path with TorchSim's batched
    neighbor list, so a single SimState can contain one structure or a batch of
    structures/replicas.
    """

    def __init__(
        self,
        model_path: str | Path,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float32,
        compute_forces: bool = True,
        compute_stress: bool = False,
        neighbor_list_fn: NeighborListFn = torchsim_nl,
        r_cut: float = 5.0,
        a_cut: float = 3.5,
        reference_energies: np.ndarray = atom_refs,
    ) -> None:
        super().__init__()
        if compute_stress:
            raise NotImplementedError("GPTFFTorchSimModel currently supports energy and forces, but not stress.")

        self._device = torch.device(device)
        self._dtype = dtype
        self._compute_forces = bool(compute_forces)
        self._compute_stress = bool(compute_stress)
        self._memory_scales_with = "n_atoms_x_density"
        self.r_cut = float(r_cut)
        self.a_cut = float(a_cut)
        self.neighbor_list_fn = neighbor_list_fn

        state = _load_checkpoint(model_path, self._device)
        cfg = CFG(state["cfg"])
        cfg.device = self._device
        if state["cfg"]["transformer_activate"]:
            self.gptff_model = model.tModLodaer_t(cfg)
        else:
            self.gptff_model = model.tModLodaer(cfg)
        self.gptff_model.load_state_dict(state["state_dict"])
        self.gptff_model = self.gptff_model.to(device=self._device, dtype=self._dtype)
        for parameter in self.gptff_model.parameters():
            parameter.requires_grad_(False)
        self.gptff_model.eval()

        refs = torch.as_tensor(reference_energies, dtype=self._dtype, device=self._device)
        self.register_buffer("atom_refs", refs, persistent=False)

    def _build_gptff_inputs(self, state: ts.SimState, positions: Tensor) -> tuple[Tensor, ...]:
        system_idx = state.system_idx.to(device=self._device, dtype=torch.long)
        atomic_numbers = state.atomic_numbers.to(device=self._device, dtype=torch.long).view(-1)
        cell = state.row_vector_cell.to(device=self._device, dtype=self._dtype)
        pbc = state.pbc.to(device=self._device)
        cutoff = torch.tensor(self.r_cut, device=self._device, dtype=self._dtype)

        edge_index, mapping_system, unit_shifts = self.neighbor_list_fn(
            positions,
            cell,
            pbc,
            cutoff,
            system_idx,
            False,
        )
        nbr_atoms = edge_index.T.to(dtype=torch.long)
        mapping_system = mapping_system.to(device=self._device, dtype=torch.long)
        unit_shifts = unit_shifts.to(device=self._device, dtype=self._dtype)

        # Match GPTFF's Cython neighbor search: exclude only the true zero-shift
        # self edge, while retaining periodic images of the same atom.
        zero_shift = (unit_shifts == 0).all(dim=1)
        zero_self_edge = (nbr_atoms[:, 0] == nbr_atoms[:, 1]) & zero_shift
        keep_edges = ~zero_self_edge
        nbr_atoms = nbr_atoms[keep_edges]
        mapping_system = mapping_system[keep_edges]
        unit_shifts = unit_shifts[keep_edges]

        shifts = ts.transforms.compute_cell_shifts(cell, unit_shifts, mapping_system)

        pair_vec = positions[nbr_atoms[:, 1]] + shifts - positions[nbr_atoms[:, 0]]
        pair_dist = torch.linalg.norm(pair_vec, dim=1)
        nbr_atoms, pair_vec, pair_dist, mapping_system = _sort_edges_by_center(
            nbr_atoms, pair_vec, pair_dist, mapping_system
        )
        del mapping_system

        bond_pairs_indices, n_bond_pairs_bond = _angle_edge_pairs(nbr_atoms, pair_dist, self.a_cut)
        triple_dist_ij, triple_dist_ik, triple_a_jik = _triple_features(
            pair_vec,
            pair_dist,
            bond_pairs_indices,
        )

        n_systems = int(state.n_systems)
        n_atoms = torch.bincount(system_idx, minlength=n_systems).to(dtype=torch.long)
        ref_per_atom = self.atom_refs.index_select(0, atomic_numbers)
        ref_energy = ref_per_atom.new_zeros((n_systems,))
        ref_energy = torch.index_add(ref_energy, 0, system_idx, ref_per_atom)

        return (
            atomic_numbers.view(-1, 1),
            pair_dist,
            n_atoms,
            triple_dist_ij,
            triple_dist_ik,
            triple_a_jik,
            nbr_atoms,
            n_bond_pairs_bond,
            bond_pairs_indices,
            ref_energy,
        )

    def forward(self, state: ts.SimState, **kwargs) -> dict[str, Tensor]:
        del kwargs
        positions = state.positions.to(device=self._device, dtype=self._dtype).detach().clone()
        positions.requires_grad_(self._compute_forces)

        (
            atom_fea,
            pair_dist,
            n_atoms,
            triple_dist_ij,
            triple_dist_ik,
            triple_a_jik,
            nbr_atoms,
            n_bond_pairs_bond,
            bond_pairs_indices,
            ref_energy,
        ) = self._build_gptff_inputs(state, positions)

        energy = self.gptff_model(
            atom_fea,
            pair_dist,
            n_atoms,
            triple_dist_ij,
            triple_dist_ik,
            triple_a_jik,
            nbr_atoms,
            n_bond_pairs_bond,
            bond_pairs_indices,
        ).view(-1) + ref_energy

        results: dict[str, Tensor] = {"energy": energy.detach()}
        if self._compute_forces:
            forces = -torch.autograd.grad(
                energy,
                positions,
                grad_outputs=torch.ones_like(energy),
                retain_graph=False,
                create_graph=False,
            )[0]
            results["forces"] = forces.detach()
        return results


__all__ = ["GPTFFTorchSimModel"]
