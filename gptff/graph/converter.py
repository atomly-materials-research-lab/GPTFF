from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from gptff.graph.containers import CrystalGraph


@dataclass(frozen=True)
class CrystalGraphConverter:
    r_cut: float = 5.0
    a_cut: float = 3.5
    numerical_tol: float = 1e-8

    def convert(self, structure: Structure) -> CrystalGraph:
        center, neighbor, offsets, distances = structure.get_neighbor_list(
            r=self.r_cut,
            numerical_tol=self.numerical_tol,
            exclude_self=True,
        )
        center = np.asarray(center, dtype=np.int64)
        neighbor = np.asarray(neighbor, dtype=np.int64)
        offsets = np.asarray(offsets, dtype=np.int64)
        distances = np.asarray(distances, dtype=np.float32)

        if center.size:
            order = np.lexsort(
                (
                    offsets[:, 2],
                    offsets[:, 1],
                    offsets[:, 0],
                    neighbor,
                    center,
                )
            )
            center = center[order]
            neighbor = neighbor[order]
            offsets = offsets[order]
            distances = distances[order]
            edge_index = np.stack([center, neighbor], axis=0)
        else:
            edge_index = np.empty((2, 0), dtype=np.int64)
            offsets = np.empty((0, 3), dtype=np.int64)
            distances = np.empty((0,), dtype=np.float32)

        triplet_edge_index, triplets_per_atom, triplets_per_edge = enumerate_triplets(
            edge_index=edge_index,
            edge_distances=distances,
            angle_cutoff=self.a_cut,
            num_atoms=len(structure),
            numerical_tol=self.numerical_tol,
        )

        return CrystalGraph(
            atom_types=np.asarray([site.specie.number for site in structure], dtype=np.int64),
            positions=np.asarray(structure.cart_coords, dtype=np.float32),
            lattice=np.asarray(structure.lattice.matrix, dtype=np.float32),
            radial_cutoff=float(self.r_cut),
            angle_cutoff=float(self.a_cut),
            edge_index=edge_index,
            edge_offsets=offsets.astype(np.float32, copy=False),
            edge_distances=distances,
            triplet_edge_index=triplet_edge_index,
            triplets_per_atom=triplets_per_atom,
            triplets_per_edge=triplets_per_edge,
        )

    def convert_ase_atoms(self, atoms) -> CrystalGraph:
        structure = AseAtomsAdaptor().get_structure(atoms)
        return self.convert(structure)


def enumerate_triplets(
    edge_index: np.ndarray,
    edge_distances: np.ndarray,
    angle_cutoff: float,
    num_atoms: int,
    numerical_tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_edges = edge_index.shape[1]
    triplets_per_atom = np.zeros(num_atoms, dtype=np.int64)
    triplets_per_edge = np.zeros(num_edges, dtype=np.int64)

    if num_edges == 0:
        return np.empty((2, 0), dtype=np.int64), triplets_per_atom, triplets_per_edge

    angle_edge_ids = np.flatnonzero(edge_distances <= angle_cutoff + numerical_tol)
    edges_by_center = [[] for _ in range(num_atoms)]
    centers = edge_index[0]

    for edge_id in angle_edge_ids:
        edges_by_center[int(centers[edge_id])].append(int(edge_id))

    triplets: list[tuple[int, int]] = []
    for atom_id, edge_ids in enumerate(edges_by_center):
        num_angle_edges = len(edge_ids)
        triplets_per_atom[atom_id] = num_angle_edges * (num_angle_edges - 1)
        if num_angle_edges < 2:
            continue

        for edge_ij in edge_ids:
            triplets_per_edge[edge_ij] = num_angle_edges - 1
            for edge_ik in edge_ids:
                if edge_ij != edge_ik:
                    triplets.append((edge_ij, edge_ik))

    if not triplets:
        return np.empty((2, 0), dtype=np.int64), triplets_per_atom, triplets_per_edge

    return np.asarray(triplets, dtype=np.int64).T, triplets_per_atom, triplets_per_edge
