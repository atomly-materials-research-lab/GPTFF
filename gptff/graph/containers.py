from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class CrystalGraph:
    atom_types: np.ndarray
    positions: np.ndarray
    lattice: np.ndarray
    radial_cutoff: float
    angle_cutoff: float
    edge_index: np.ndarray
    edge_offsets: np.ndarray
    triplet_edge_index: np.ndarray

    @property
    def num_atoms(self) -> int:
        return int(self.atom_types.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    @property
    def num_triplets(self) -> int:
        return int(self.triplet_edge_index.shape[1])


@dataclass(frozen=True)
class GraphSample:
    graph: CrystalGraph
    energy: float | None = None
    forces: np.ndarray | None = None
    stress: np.ndarray | None = None


@dataclass(frozen=True)
class CrystalGraphBatch:
    atom_types: torch.Tensor
    positions: torch.Tensor
    lattice: torch.Tensor
    radial_cutoff: float
    angle_cutoff: float
    edge_index: torch.Tensor
    edge_offsets: torch.Tensor
    triplet_edge_index: torch.Tensor
    num_atoms: torch.Tensor
    num_edges: torch.Tensor
    atom_batch: torch.Tensor
    edge_batch: torch.Tensor
    energy: torch.Tensor | None = None
    forces: torch.Tensor | None = None
    stress: torch.Tensor | None = None

    @classmethod
    def from_graphs(
        cls,
        graphs: Sequence[CrystalGraph],
        energies: Sequence[float] | None = None,
        forces: Sequence[np.ndarray] | None = None,
        stresses: Sequence[np.ndarray] | None = None,
    ) -> CrystalGraphBatch:
        if len(graphs) == 0:
            raise ValueError("Cannot batch an empty graph list.")

        radial_cutoff = _shared_cutoff(graphs, "radial_cutoff")
        angle_cutoff = _shared_cutoff(graphs, "angle_cutoff")

        atom_counts = np.asarray([graph.num_atoms for graph in graphs], dtype=np.int64)
        edge_counts = np.asarray([graph.num_edges for graph in graphs], dtype=np.int64)
        atom_offsets = np.cumsum(np.concatenate([[0], atom_counts[:-1]])).astype(np.int64)
        edge_index_offsets = np.cumsum(np.concatenate([[0], edge_counts[:-1]])).astype(np.int64)

        atom_types = np.concatenate([graph.atom_types for graph in graphs])
        positions = np.concatenate([graph.positions for graph in graphs], axis=0)
        lattice = np.stack([graph.lattice for graph in graphs], axis=0)
        periodic_edge_offsets = np.concatenate([graph.edge_offsets for graph in graphs], axis=0)

        shifted_edges = []
        shifted_triplets = []
        for graph, atom_offset, edge_index_offset in zip(
            graphs,
            atom_offsets,
            edge_index_offsets,
        ):
            shifted_edges.append(graph.edge_index + atom_offset)
            shifted_triplets.append(graph.triplet_edge_index + edge_index_offset)

        edge_index = _concat_axis1(shifted_edges, rows=2, dtype=np.int64)
        triplet_edge_index = _concat_axis1(shifted_triplets, rows=2, dtype=np.int64)

        atom_batch = np.repeat(np.arange(len(graphs), dtype=np.int64), atom_counts)
        edge_batch = np.repeat(np.arange(len(graphs), dtype=np.int64), edge_counts)

        return cls(
            atom_types=torch.tensor(atom_types, dtype=torch.long),
            positions=torch.tensor(positions, dtype=torch.float32),
            lattice=torch.tensor(lattice, dtype=torch.float32),
            radial_cutoff=radial_cutoff,
            angle_cutoff=angle_cutoff,
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_offsets=torch.tensor(periodic_edge_offsets, dtype=torch.float32),
            triplet_edge_index=torch.tensor(triplet_edge_index, dtype=torch.long),
            num_atoms=torch.tensor(atom_counts, dtype=torch.long),
            num_edges=torch.tensor(edge_counts, dtype=torch.long),
            atom_batch=torch.tensor(atom_batch, dtype=torch.long),
            edge_batch=torch.tensor(edge_batch, dtype=torch.long),
            energy=_optional_float_tensor(energies),
            forces=_optional_concat_tensor(forces),
            stress=_optional_stack_tensor(stresses),
        )

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> CrystalGraphBatch:
        fields = {
            name: _tensor_to(value, device, non_blocking=non_blocking)
            for name, value in self.__dict__.items()
        }
        return type(self)(**fields)

    def pin_memory(self) -> CrystalGraphBatch:
        fields = {
            name: _pin_memory(value)
            for name, value in self.__dict__.items()
        }
        return type(self)(**fields)

    def with_geometry(
        self,
        *,
        positions_requires_grad: bool = True,
        strain_requires_grad: bool = True,
    ) -> DifferentiableGraphBatch:
        positions = self.positions.detach().clone().requires_grad_(positions_requires_grad)
        strain = torch.zeros_like(
            self.lattice,
            dtype=self.lattice.dtype,
        ).requires_grad_(strain_requires_grad)
        eye = torch.eye(3, dtype=self.lattice.dtype, device=self.lattice.device)

        strained_lattice = self.lattice @ (eye.unsqueeze(0) + strain)
        volumes = torch.abs(torch.linalg.det(strained_lattice))

        atom_strain = strain[self.atom_batch]
        strained_positions = torch.matmul(
            positions.unsqueeze(1),
            eye.unsqueeze(0) + atom_strain,
        ).squeeze(1)

        edge_lattice = strained_lattice[self.edge_batch]
        edge_translation = torch.matmul(self.edge_offsets.unsqueeze(1), edge_lattice).squeeze(1)
        edge_vectors = (
            strained_positions[self.edge_index[1]]
            + edge_translation
            - strained_positions[self.edge_index[0]]
        )
        edge_lengths = torch.linalg.norm(edge_vectors, dim=1)

        if self.triplet_edge_index.numel() == 0:
            triplet_lengths_ij = edge_lengths.new_empty((0,))
            triplet_lengths_ik = edge_lengths.new_empty((0,))
            triplet_cosine = edge_lengths.new_empty((0,))
        else:
            edge_ij = self.triplet_edge_index[0]
            edge_ik = self.triplet_edge_index[1]
            triplet_vec_ij = edge_vectors[edge_ij]
            triplet_vec_ik = edge_vectors[edge_ik]
            triplet_lengths_ij = edge_lengths[edge_ij]
            triplet_lengths_ik = edge_lengths[edge_ik]
            numerator = torch.sum(triplet_vec_ij * triplet_vec_ik, dim=1)
            denominator = triplet_lengths_ij * triplet_lengths_ik
            triplet_cosine = numerator / denominator.clamp_min(1e-12)
            triplet_cosine = torch.clamp(triplet_cosine, -1.0, 1.0)

        fields = dict(self.__dict__)
        fields["positions"] = strained_positions
        fields.update(
            strain=strain,
            volumes=volumes,
            strained_lattice=strained_lattice,
            edge_vectors=edge_vectors,
            edge_lengths=edge_lengths,
            triplet_lengths_ij=triplet_lengths_ij,
            triplet_lengths_ik=triplet_lengths_ik,
            triplet_cosine=triplet_cosine,
        )
        return DifferentiableGraphBatch(**fields)


@dataclass(frozen=True)
class DifferentiableGraphBatch(CrystalGraphBatch):
    strain: torch.Tensor = None
    volumes: torch.Tensor = None
    strained_lattice: torch.Tensor = None
    edge_vectors: torch.Tensor = None
    edge_lengths: torch.Tensor = None
    triplet_lengths_ij: torch.Tensor = None
    triplet_lengths_ik: torch.Tensor = None
    triplet_cosine: torch.Tensor = None

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> DifferentiableGraphBatch:
        fields = {
            name: _tensor_to(value, device, non_blocking=non_blocking)
            for name, value in self.__dict__.items()
        }
        return type(self)(**fields)


def batch_graphs(graphs: Sequence[CrystalGraph]) -> CrystalGraphBatch:
    return CrystalGraphBatch.from_graphs(graphs)


def batch_samples(samples: Sequence[GraphSample]) -> CrystalGraphBatch:
    return CrystalGraphBatch.from_graphs(
        [sample.graph for sample in samples],
        energies=_collect_optional_sample_field(samples, "energy"),
        forces=_collect_optional_sample_field(samples, "forces"),
        stresses=_collect_optional_sample_field(samples, "stress"),
    )


def _collect_optional_sample_field(samples: Sequence[GraphSample], field_name: str):
    values = [getattr(sample, field_name) for sample in samples]
    present = [value is not None for value in values]
    if all(present):
        return values
    if not any(present):
        return None
    raise ValueError(f"Cannot batch samples with partially missing {field_name} labels.")


def _concat_axis1(arrays: Sequence[np.ndarray], rows: int, dtype: np.dtype) -> np.ndarray:
    non_empty = [array for array in arrays if array.shape[1] > 0]
    if not non_empty:
        return np.empty((rows, 0), dtype=dtype)
    return np.concatenate(non_empty, axis=1).astype(dtype, copy=False)


def _shared_cutoff(graphs: Sequence[CrystalGraph], field_name: str) -> float:
    reference = float(getattr(graphs[0], field_name))
    for graph in graphs[1:]:
        value = float(getattr(graph, field_name))
        if not np.isclose(value, reference, rtol=0.0, atol=1e-8):
            raise ValueError(
                f"Cannot batch graphs with different {field_name} values: {reference} and {value}."
            )
    return reference


def _optional_float_tensor(values: Sequence[float] | None) -> torch.Tensor | None:
    if values is None:
        return None
    return torch.tensor(np.asarray(values, dtype=np.float32), dtype=torch.float32)


def _optional_concat_tensor(values: Sequence[np.ndarray] | None) -> torch.Tensor | None:
    if values is None:
        return None
    return torch.tensor(np.concatenate(values, axis=0), dtype=torch.float32)


def _optional_stack_tensor(values: Sequence[np.ndarray] | None) -> torch.Tensor | None:
    if values is None:
        return None
    return torch.tensor(np.stack(values, axis=0), dtype=torch.float32)


def _tensor_to(
    value,
    device: torch.device | str,
    *,
    non_blocking: bool,
):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=non_blocking)
    return value


def _pin_memory(value):
    if isinstance(value, torch.Tensor) and not value.is_cuda and torch.cuda.is_available():
        return value.pin_memory()
    return value
