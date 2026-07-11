from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from gptff.graph.containers import CrystalGraph


@dataclass(frozen=True)
class CrystalGraphConverter:
    radial_cutoff: float = 5.0
    angle_cutoff: float = 3.5
    numerical_tol: float = 1e-8

    def convert(self, structure: Structure) -> CrystalGraph:
        center, neighbor, offsets, distances = structure.get_neighbor_list(
            r=self.radial_cutoff,
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

        triplet_edge_index = enumerate_triplets(
            edge_index=edge_index,
            edge_distances=distances,
            angle_cutoff=self.angle_cutoff,
            num_atoms=len(structure),
            numerical_tol=self.numerical_tol,
        )

        return CrystalGraph(
            atom_types=np.asarray([site.specie.number for site in structure], dtype=np.int64),
            positions=np.asarray(structure.cart_coords, dtype=np.float32),
            lattice=np.asarray(structure.lattice.matrix, dtype=np.float32),
            radial_cutoff=float(self.radial_cutoff),
            angle_cutoff=float(self.angle_cutoff),
            edge_index=edge_index,
            edge_offsets=offsets.astype(np.float32, copy=False),
            triplet_edge_index=triplet_edge_index,
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
) -> np.ndarray:
    num_edges = edge_index.shape[1]

    if num_edges == 0:
        return np.empty((2, 0), dtype=np.int64)

    angle_edge_ids = np.flatnonzero(edge_distances <= angle_cutoff + numerical_tol)
    edges_by_center = [[] for _ in range(num_atoms)]
    centers = edge_index[0]

    for edge_id in angle_edge_ids:
        edges_by_center[int(centers[edge_id])].append(int(edge_id))

    triplets: list[tuple[int, int]] = []
    for edge_ids in edges_by_center:
        num_angle_edges = len(edge_ids)
        if num_angle_edges < 2:
            continue

        for edge_ij in edge_ids:
            for edge_ik in edge_ids:
                if edge_ij != edge_ik:
                    triplets.append((edge_ij, edge_ik))

    if not triplets:
        return np.empty((2, 0), dtype=np.int64)

    return np.asarray(triplets, dtype=np.int64).T
