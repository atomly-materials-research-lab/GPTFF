from __future__ import annotations

import json
import math
import shutil
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import h5py
import numpy as np
from torch.utils.data import Dataset

from gptff.graph import CrystalGraph, GraphSample
from gptff.utils.labels import convert_vasp_stress_to_gpa

PathLike = str | Path

METADATA_FILE = "metadata.json"
INDEX_FILE = "index.jsonl"
SHARDS_DIR = "shards"


@dataclass(frozen=True)
class ShardedGraphIndexRecord:
    sample_id: str
    material_id: str | None
    shard: str
    group: str
    energy: float
    has_stress: bool
    num_atoms: int
    composition: Mapping[str, int]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ShardedGraphIndexRecord:
        return cls(
            sample_id=str(data["sample_id"]),
            material_id=None if data.get("material_id") is None else str(data["material_id"]),
            shard=str(data["shard"]),
            group=str(data["group"]),
            energy=float(data["energy"]),
            has_stress=bool(data.get("has_stress", False)),
            num_atoms=int(data["num_atoms"]),
            composition={str(key): int(value) for key, value in data["composition"].items()},
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "material_id": self.material_id,
            "shard": self.shard,
            "group": self.group,
            "energy": self.energy,
            "has_stress": self.has_stress,
            "num_atoms": self.num_atoms,
            "composition": dict(self.composition),
        }


class ShardedGraphDataset(Dataset):
    """Lazy dataset backed by precomputed GPTFF graph shards."""

    def __init__(
        self,
        root: PathLike,
        *,
        records: Sequence[ShardedGraphIndexRecord] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.root = Path(root)
        self.metadata = dict(load_sharded_graph_metadata(self.root) if metadata is None else metadata)
        self.records = tuple(load_sharded_graph_index(self.root) if records is None else records)
        if not self.records:
            raise ValueError("ShardedGraphDataset must contain at least one sample.")
        self._shard_datasets, self._lookup = self._build_shard_datasets()

    def __getstate__(self):
        state = dict(self.__dict__)
        return state

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> GraphSample:
        shard_index, local_index = self._lookup[int(index)]
        try:
            return self._shard_datasets[shard_index][local_index]
        except Exception as exc:
            record = self.records[int(index)]
            raise ValueError(f"Failed to load graph sample '{record.sample_id}'.") from exc

    def _build_shard_datasets(
        self,
    ) -> tuple[tuple[HDF5GraphShardDataset, ...], tuple[tuple[int, int], ...]]:
        records_by_shard: dict[str, list[tuple[int, ShardedGraphIndexRecord]]] = {}
        for global_index, record in enumerate(self.records):
            records_by_shard.setdefault(record.shard, []).append((global_index, record))

        shard_datasets = []
        lookup: list[tuple[int, int] | None] = [None] * len(self.records)
        for shard_index, (shard, indexed_records) in enumerate(records_by_shard.items()):
            local_records = tuple(record for _, record in indexed_records)
            shard_datasets.append(
                HDF5GraphShardDataset(
                    root=self.root,
                    shard=shard,
                    records=local_records,
                )
            )
            for local_index, (global_index, _) in enumerate(indexed_records):
                lookup[global_index] = (shard_index, local_index)

        return tuple(shard_datasets), tuple(_require_lookup_entry(entry) for entry in lookup)

    def subset(
        self,
        indices: Sequence[int],
        *,
        name: str | None = None,
    ) -> ShardedGraphDataset:
        metadata = dict(self.metadata)
        if name is not None:
            metadata["name"] = name
        return type(self)(
            self.root,
            records=tuple(self.records[int(index)] for index in indices),
            metadata=metadata,
        )

    def sample_key(self, index: int) -> str:
        return self.records[int(index)].sample_id

    def material_id(self, index: int) -> str | None:
        return self.records[int(index)].material_id

    def has_stress(self, index: int) -> bool:
        return self.records[int(index)].has_stress

    def element_ref_records(self):
        return (
            SimpleNamespace(composition=record.composition, energy=record.energy)
            for record in self.records
        )

    def close(self) -> None:
        for shard_dataset in self._shard_datasets:
            shard_dataset.close()


class HDF5GraphShardDataset(Dataset):
    """Lazy view of one HDF5 graph shard."""

    def __init__(
        self,
        *,
        root: PathLike,
        shard: str,
        records: Sequence[ShardedGraphIndexRecord],
    ) -> None:
        self.root = Path(root)
        self.shard = str(shard)
        self.records = tuple(records)
        if not self.records:
            raise ValueError("HDF5GraphShardDataset must contain at least one sample.")
        self._file: h5py.File | None = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_file"] = None
        return state

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> GraphSample:
        record = self.records[int(index)]
        group = self.file[record.group]
        return read_graph_sample_group(group)

    @property
    def file(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.root / self.shard, "r")
        return self._file

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


class ShardedGraphDatasetWriter:
    """Streaming writer for precomputed GPTFF graph shards."""

    def __init__(
        self,
        root: PathLike,
        *,
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        samples_per_shard: int,
        overwrite: bool = False,
    ) -> None:
        self.root = Path(root)
        if self.root.exists():
            if not overwrite:
                raise FileExistsError(f"Output directory already exists: {self.root}")
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True)
        self.shards_dir = self.root / SHARDS_DIR
        self.shards_dir.mkdir()
        self.name = name
        self.metadata = dict(metadata or {})
        self.samples_per_shard = _positive_int(samples_per_shard, "samples_per_shard")
        self.sample_count = 0
        self.shard_count = 0
        self._current_file: h5py.File | None = None
        self._current_shard_name: str | None = None
        self._index_file = open(self.root / INDEX_FILE, "w", encoding="utf-8")

    def __enter__(self) -> ShardedGraphDatasetWriter:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()

    def add(
        self,
        *,
        graph: CrystalGraph,
        energy: float,
        forces: np.ndarray,
        stress: np.ndarray | None,
        sample_id: str,
        material_id: str | None,
    ) -> None:
        self._validate_or_set_cutoffs(graph)
        group_name = f"sample_{self.sample_count:08d}"
        shard_name = self._ensure_shard()
        group = self._current_file.create_group(group_name)
        write_graph_sample_group(
            group,
            graph=graph,
            energy=energy,
            forces=forces,
            stress=stress,
            sample_id=sample_id,
            material_id=material_id,
        )
        record = ShardedGraphIndexRecord(
            sample_id=sample_id,
            material_id=material_id,
            shard=f"{SHARDS_DIR}/{shard_name}",
            group=group_name,
            energy=float(energy),
            has_stress=stress is not None,
            num_atoms=graph.num_atoms,
            composition=composition_from_atom_types(graph.atom_types),
        )
        self._index_file.write(json.dumps(record.as_dict(), separators=(",", ":")) + "\n")
        self.sample_count += 1

    def close(self) -> None:
        if self._current_file is not None:
            self._current_file.close()
            self._current_file = None
        self._index_file.close()
        write_sharded_graph_metadata(
            self.root,
            {
                **self.metadata,
                "name": self.name,
                "format": "gptff_sharded_hdf5_graph",
                "num_samples": self.sample_count,
                "num_shards": self.shard_count,
                "samples_per_shard": self.samples_per_shard,
            },
        )

    def abort(self) -> None:
        if self._current_file is not None:
            self._current_file.close()
            self._current_file = None
        self._index_file.close()
        shutil.rmtree(self.root, ignore_errors=True)

    def _ensure_shard(self) -> str:
        if self.sample_count % self.samples_per_shard == 0:
            if self._current_file is not None:
                self._current_file.close()
            shard_name = f"shard_{self.shard_count:06d}.h5"
            self._current_file = h5py.File(self.shards_dir / shard_name, "w")
            self._current_shard_name = shard_name
            self.shard_count += 1
        return self._current_shard_name

    def _validate_or_set_cutoffs(self, graph: CrystalGraph) -> None:
        self._validate_or_set_cutoff("radial_cutoff", graph.radial_cutoff)
        self._validate_or_set_cutoff("angle_cutoff", graph.angle_cutoff)

    def _validate_or_set_cutoff(self, key: str, value: float) -> None:
        if key not in self.metadata:
            self.metadata[key] = float(value)
            return
        expected = float(self.metadata[key])
        actual = float(value)
        if math.isclose(expected, actual, rel_tol=0.0, abs_tol=1e-8):
            return
        raise ValueError(
            f"All graphs in a sharded dataset must use the same {key}; "
            f"metadata has {expected:g}, graph has {actual:g}."
        )


def _require_lookup_entry(entry: tuple[int, int] | None) -> tuple[int, int]:
    if entry is None:
        raise RuntimeError("Internal sharded dataset lookup construction failed.")
    return entry


def write_graph_sample_group(
    group,
    *,
    graph: CrystalGraph,
    energy: float,
    forces: np.ndarray,
    stress: np.ndarray | None,
    sample_id: str,
    material_id: str | None,
) -> None:
    group.attrs["sample_id"] = str(sample_id)
    group.attrs["material_id"] = "" if material_id is None else str(material_id)
    group.attrs["energy"] = float(energy)
    group.attrs["radial_cutoff"] = float(graph.radial_cutoff)
    group.attrs["angle_cutoff"] = float(graph.angle_cutoff)

    _dataset(group, "atom_types", graph.atom_types, dtype=np.int64)
    _dataset(group, "positions", graph.positions, dtype=np.float32)
    _dataset(group, "lattice", graph.lattice, dtype=np.float32)
    _dataset(group, "edge_index", graph.edge_index, dtype=np.int64)
    _dataset(group, "edge_offsets", graph.edge_offsets, dtype=np.float32)
    _dataset(group, "edge_distances", graph.edge_distances, dtype=np.float32)
    _dataset(group, "triplet_edge_index", graph.triplet_edge_index, dtype=np.int64)
    _dataset(group, "triplets_per_atom", graph.triplets_per_atom, dtype=np.int64)
    _dataset(group, "triplets_per_edge", graph.triplets_per_edge, dtype=np.int64)
    _dataset(group, "forces", np.asarray(forces, dtype=np.float32), dtype=np.float32)
    if stress is not None:
        _dataset(group, "stress", np.asarray(stress, dtype=np.float32), dtype=np.float32)


def read_graph_sample_group(group) -> GraphSample:
    graph = CrystalGraph(
        atom_types=group["atom_types"][()].astype(np.int64, copy=False),
        positions=group["positions"][()].astype(np.float32, copy=False),
        lattice=group["lattice"][()].astype(np.float32, copy=False),
        radial_cutoff=float(group.attrs["radial_cutoff"]),
        angle_cutoff=float(group.attrs["angle_cutoff"]),
        edge_index=group["edge_index"][()].astype(np.int64, copy=False),
        edge_offsets=group["edge_offsets"][()].astype(np.float32, copy=False),
        edge_distances=group["edge_distances"][()].astype(np.float32, copy=False),
        triplet_edge_index=group["triplet_edge_index"][()].astype(np.int64, copy=False),
        triplets_per_atom=group["triplets_per_atom"][()].astype(np.int64, copy=False),
        triplets_per_edge=group["triplets_per_edge"][()].astype(np.int64, copy=False),
    )
    raw_stress = group["stress"][()] if "stress" in group else None
    return GraphSample(
        graph=graph,
        energy=float(group.attrs["energy"]),
        forces=group["forces"][()].astype(np.float32, copy=False),
        stress=None if raw_stress is None else convert_vasp_stress_to_gpa(raw_stress),
    )


def load_sharded_graph_index(root: PathLike) -> tuple[ShardedGraphIndexRecord, ...]:
    index_path = Path(root) / INDEX_FILE
    with open(index_path, encoding="utf-8") as file:
        return tuple(
            ShardedGraphIndexRecord.from_dict(json.loads(line))
            for line in file
            if line.strip()
        )


def load_sharded_graph_metadata(root: PathLike) -> dict[str, Any]:
    with open(Path(root) / METADATA_FILE, encoding="utf-8") as file:
        return json.load(file)


def write_sharded_graph_metadata(root: PathLike, metadata: Mapping[str, Any]) -> None:
    with open(Path(root) / METADATA_FILE, "w", encoding="utf-8") as file:
        json.dump(dict(metadata), file, indent=2, sort_keys=True)


def composition_from_atom_types(atom_types: Iterable[int]) -> dict[str, int]:
    unique, counts = np.unique(np.asarray(list(atom_types), dtype=np.int64), return_counts=True)
    return {str(int(atomic_number)): int(count) for atomic_number, count in zip(unique, counts)}


def _dataset(group, name: str, value, *, dtype) -> None:
    group.create_dataset(name, data=np.asarray(value, dtype=dtype))


def _positive_int(value: int, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value
