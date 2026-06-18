from __future__ import annotations

from collections import Counter, OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from monty.json import MSONable
from monty.serialization import dumpfn, loadfn
from pymatgen.core import Structure
from torch.utils.data import Dataset

from gptff.graph import CrystalGraphConverter, GraphSample, batch_samples
from gptff.utils.labels import convert_vasp_stress_to_gpa

PathLike = str | Path


@dataclass(frozen=True)
class AtomicSample(MSONable):
    """One labeled periodic structure using raw VASP label conventions.

    Energy is the total structure energy in eV, forces are in eV/angstrom,
    and stress is the raw VASP stress in kBar.
    """

    structure: Structure
    energy: float
    forces: np.ndarray
    stress: np.ndarray | None = None
    sample_id: str | None = None
    material_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.structure, Structure):
            raise TypeError("structure must be a pymatgen Structure.")
        if self.structure.num_sites == 0:
            raise ValueError("structure must contain at least one atom.")

        energy = float(self.energy)
        if not np.isfinite(energy):
            raise ValueError("energy must be finite.")

        forces = np.asarray(self.forces, dtype=np.float32)
        expected_force_shape = (self.structure.num_sites, 3)
        if forces.shape != expected_force_shape:
            raise ValueError(f"forces must have shape {expected_force_shape}, got {forces.shape}.")
        if not np.all(np.isfinite(forces)):
            raise ValueError("forces must contain only finite values.")

        stress = None
        if self.stress is not None:
            stress = _stress_to_matrix(np.asarray(self.stress, dtype=np.float32))
            if not np.all(np.isfinite(stress)):
                raise ValueError("stress must contain only finite values.")

        sample_id = _optional_identifier(self.sample_id, "sample_id")
        material_id = _optional_identifier(self.material_id, "material_id")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping.")

        object.__setattr__(self, "energy", energy)
        object.__setattr__(self, "forces", forces.copy())
        object.__setattr__(self, "stress", None if stress is None else stress.copy())
        object.__setattr__(self, "sample_id", sample_id)
        object.__setattr__(self, "material_id", material_id)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "@module": type(self).__module__,
            "@class": type(self).__name__,
            "structure": self.structure.as_dict(),
            "energy": self.energy,
            "forces": self.forces.tolist(),
            "stress": None if self.stress is None else self.stress.tolist(),
            "sample_id": self.sample_id,
            "material_id": self.material_id,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AtomicSample:
        structure = data["structure"]
        if isinstance(structure, Mapping):
            structure = Structure.from_dict(structure)
        return cls(
            structure=structure,
            energy=data["energy"],
            forces=data["forces"],
            stress=data.get("stress"),
            sample_id=data.get("sample_id"),
            material_id=data.get("material_id"),
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True)
class AtomicDataset(MSONable, Sequence[AtomicSample]):
    """Complete labeled dataset before graph conversion."""

    samples: tuple[AtomicSample, ...]
    name: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        samples = tuple(self.samples)
        if not samples:
            raise ValueError("AtomicDataset must contain at least one sample.")
        if not all(isinstance(sample, AtomicSample) for sample in samples):
            raise TypeError("samples must contain only AtomicSample objects.")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping.")

        sample_ids = Counter(sample.sample_id for sample in samples if sample.sample_id is not None)
        duplicates = sorted(sample_id for sample_id, count in sample_ids.items() if count > 1)
        if duplicates:
            raise ValueError(f"sample_id values must be unique; duplicates: {duplicates}.")

        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "name", _optional_identifier(self.name, "name"))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return type(self)(
                samples=self.samples[index],
                name=self.name,
                metadata=self.metadata,
            )
        return self.samples[index]

    def subset(self, indices: Sequence[int], *, name: str | None = None) -> AtomicDataset:
        return type(self)(
            samples=tuple(self.samples[int(index)] for index in indices),
            name=name,
            metadata=self.metadata,
        )

    def sample_key(self, index: int) -> str:
        sample_id = self.samples[index].sample_id
        return sample_id if sample_id is not None else f"index:{index}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "@module": type(self).__module__,
            "@class": type(self).__name__,
            "samples": [sample.as_dict() for sample in self.samples],
            "name": self.name,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AtomicDataset:
        samples = tuple(
            sample if isinstance(sample, AtomicSample) else AtomicSample.from_dict(sample)
            for sample in data["samples"]
        )
        return cls(
            samples=samples,
            name=data.get("name"),
            metadata=data.get("metadata", {}),
        )

    def to_file(self, filename: PathLike) -> None:
        dumpfn(self, filename)

    @classmethod
    def from_file(cls, filename: PathLike) -> AtomicDataset:
        dataset = loadfn(filename)
        if isinstance(dataset, cls):
            return dataset
        if isinstance(dataset, Mapping):
            return cls.from_dict(dataset)
        raise TypeError(f"{filename} does not contain an AtomicDataset.")


class GraphDataset(Dataset):
    """Lazily convert an AtomicDataset into model-ready graph samples."""

    def __init__(
        self,
        atomic_dataset: AtomicDataset,
        *,
        radial_cutoff: float = 5.0,
        angle_cutoff: float = 3.5,
        numerical_tol: float = 1e-8,
        cache_graphs: bool = False,
        cache_size: int | None = None,
    ) -> None:
        if not isinstance(atomic_dataset, AtomicDataset):
            raise TypeError("atomic_dataset must be an AtomicDataset.")
        self.atomic_dataset = atomic_dataset
        self.cache_graphs = bool(cache_graphs)
        self.cache_size = _normalize_cache_size(cache_size)
        self._sample_cache: OrderedDict[int, GraphSample] = OrderedDict()
        self.converter = CrystalGraphConverter(
            radial_cutoff=radial_cutoff,
            angle_cutoff=angle_cutoff,
            numerical_tol=numerical_tol,
        )

    def __len__(self) -> int:
        return len(self.atomic_dataset)

    def __getitem__(self, index: int) -> GraphSample:
        index = int(index)
        if not self.cache_graphs:
            return self._load_sample(index)

        if index in self._sample_cache:
            sample = self._sample_cache.pop(index)
            self._sample_cache[index] = sample
            return sample

        sample = self._load_sample(index)
        if self.cache_size != 0:
            self._sample_cache[index] = sample
            if self.cache_size is not None:
                while len(self._sample_cache) > self.cache_size:
                    self._sample_cache.popitem(last=False)
        return sample

    def _load_sample(self, index: int) -> GraphSample:
        try:
            sample = self.atomic_dataset[index]
            return GraphSample(
                graph=self.converter.convert(sample.structure),
                energy=sample.energy,
                forces=sample.forces,
                stress=(
                    None if sample.stress is None else convert_vasp_stress_to_gpa(sample.stress)
                ),
            )
        except Exception as exc:
            sample_key = self.atomic_dataset.sample_key(index)
            raise ValueError(f"Failed to convert atomic sample '{sample_key}'.") from exc


def collate_graph_samples(samples: Sequence[GraphSample]):
    return batch_samples(samples)


def _stress_to_matrix(stress: np.ndarray) -> np.ndarray:
    if stress.shape == (3, 3):
        return stress
    if stress.shape == (6,):
        xx, yy, zz, yz, xz, xy = stress
        return np.asarray(
            [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]],
            dtype=stress.dtype,
        )
    raise ValueError(f"stress must have shape (3, 3) or (6,), got {stress.shape}.")


def _optional_identifier(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty.")
    return normalized


def _normalize_cache_size(cache_size: int | None) -> int | None:
    if cache_size is None:
        return None
    cache_size = int(cache_size)
    if cache_size < 0:
        raise ValueError("cache_size must be non-negative or None.")
    return cache_size
