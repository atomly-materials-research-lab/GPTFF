import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from gptff.data import (
    AtomicDataset,
    AtomicSample,
    GraphDataset,
    split_atomic_dataset,
)
from gptff.graph import CrystalGraphConverter


def test_atomic_sample_validates_and_normalizes_labels():
    sample = AtomicSample(
        structure=_structure("Na"),
        energy=-1,
        forces=[[0, 0, 0]],
        stress=[1, 2, 3, 4, 5, 6],
        sample_id=" frame-1 ",
    )

    assert sample.energy == -1.0
    assert sample.forces.dtype == np.float32
    assert sample.sample_id == "frame-1"
    assert np.allclose(
        sample.stress,
        [[1, 6, 5], [6, 2, 4], [5, 4, 3]],
    )


def test_atomic_sample_rejects_force_shape_mismatch():
    with pytest.raises(ValueError, match="forces must have shape"):
        AtomicSample(
            structure=_structure("Na"),
            energy=-1.0,
            forces=[[0.0, 0.0]],
        )


def test_atomic_sample_rejects_nonfinite_labels():
    with pytest.raises(ValueError, match="energy must be finite"):
        AtomicSample(
            structure=_structure("Na"),
            energy=float("nan"),
            forces=[[0.0, 0.0, 0.0]],
        )


def test_atomic_dataset_rejects_duplicate_sample_ids():
    with pytest.raises(ValueError, match="sample_id values must be unique"):
        AtomicDataset((_sample(0, sample_id="same"), _sample(1, sample_id="same")))


def test_atomic_sample_dict_roundtrip():
    original = _sample(0, material_id="mat-a")

    restored = AtomicSample.from_dict(original.as_dict())

    assert restored.structure == original.structure
    assert restored.energy == original.energy
    assert np.array_equal(restored.forces, original.forces)
    assert np.array_equal(restored.stress, original.stress)
    assert restored.material_id == "mat-a"


@pytest.mark.parametrize("suffix", [".json", ".json.gz"])
def test_atomic_dataset_monty_file_roundtrip(tmp_path, suffix):
    original = AtomicDataset(
        (_sample(0, material_id="mat-a"), _sample(1, material_id="mat-b")),
        name="training-data",
        metadata={"source": "vasp"},
    )
    path = tmp_path / f"dataset{suffix}"

    original.to_file(path)
    restored = AtomicDataset.from_file(path)

    assert restored.name == original.name
    assert restored.metadata == original.metadata
    assert restored.as_dict() == original.as_dict()
    assert len(restored) == 2


def test_graph_dataset_converts_raw_vasp_stress_to_gpa():
    dataset = GraphDataset(
        AtomicDataset((_sample(0, stress=np.eye(3)),)),
        radial_cutoff=2.0,
        angle_cutoff=2.0,
    )

    sample = dataset[0]

    assert sample.energy == 0.0
    assert sample.forces.shape == (1, 3)
    assert np.allclose(sample.stress, -0.1 * np.eye(3, dtype=np.float32))


def test_graph_dataset_allows_missing_stress():
    dataset = GraphDataset(
        AtomicDataset((_sample(0, stress=None),)),
        radial_cutoff=2.0,
        angle_cutoff=2.0,
    )

    assert dataset[0].stress is None


def test_graph_dataset_does_not_cache_by_default():
    dataset = GraphDataset(
        AtomicDataset((_sample(0),)),
        radial_cutoff=2.0,
        angle_cutoff=2.0,
    )
    converter = _CountingConverter(radial_cutoff=2.0, angle_cutoff=2.0)
    dataset.converter = converter

    dataset[0]
    dataset[0]

    assert converter.calls == 2


def test_graph_dataset_lru_cache_size_is_enforced():
    dataset = GraphDataset(
        AtomicDataset((_sample(0), _sample(1))),
        radial_cutoff=2.0,
        angle_cutoff=2.0,
        cache_graphs=True,
        cache_size=1,
    )
    converter = _CountingConverter(radial_cutoff=2.0, angle_cutoff=2.0)
    dataset.converter = converter

    dataset[0]
    dataset[1]
    dataset[0]

    assert converter.calls == 3


def test_graph_dataset_rejects_negative_cache_size():
    with pytest.raises(ValueError, match="cache_size"):
        GraphDataset(AtomicDataset((_sample(0),)), cache_graphs=True, cache_size=-1)


def test_random_split_is_deterministic_and_complete():
    dataset = _dataset(10)

    first = split_atomic_dataset(
        dataset,
        validation_fraction=0.2,
        test_fraction=0.1,
        seed=7,
    )
    second = split_atomic_dataset(
        dataset,
        validation_fraction=0.2,
        test_fraction=0.1,
        seed=7,
    )

    assert first == second
    assert len(first.train_indices) == 7
    assert len(first.validation_indices) == 2
    assert len(first.test_indices) == 1
    first.validate_for_dataset(dataset)


def test_grouped_split_keeps_material_frames_together():
    samples = tuple(_sample(index, material_id=f"material-{index // 3}") for index in range(12))
    dataset = AtomicDataset(samples)

    split = split_atomic_dataset(
        dataset,
        validation_fraction=0.25,
        test_fraction=0.25,
        seed=11,
        group_by_material=True,
    )

    partitions = {}
    for partition_name, indices in (
        ("train", split.train_indices),
        ("validation", split.validation_indices),
        ("test", split.test_indices),
    ):
        for index in indices:
            material_id = dataset[index].material_id
            assert material_id not in partitions or partitions[material_id] == partition_name
            partitions[material_id] = partition_name


def test_grouped_split_requires_material_ids():
    with pytest.raises(ValueError, match="requires material_id"):
        split_atomic_dataset(
            _dataset(4),
            validation_fraction=0.25,
            group_by_material=True,
        )


class _CountingConverter:
    def __init__(self, **kwargs):
        self.converter = CrystalGraphConverter(**kwargs)
        self.calls = 0

    def convert(self, structure):
        self.calls += 1
        return self.converter.convert(structure)


def _structure(species):
    return Structure(Lattice.cubic(3.0), [species], [[0.0, 0.0, 0.0]])


def _sample(index, *, sample_id=None, material_id=None, stress=np.zeros((3, 3))):
    return AtomicSample(
        structure=_structure("Na" if index % 2 == 0 else "Cl"),
        energy=-float(index),
        forces=np.zeros((1, 3)),
        stress=stress,
        sample_id=sample_id or f"frame-{index}",
        material_id=material_id,
    )


def _dataset(size):
    return AtomicDataset(tuple(_sample(index) for index in range(size)))
