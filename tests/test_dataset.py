from dataclasses import replace

import h5py
import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from gptff.data import (
    AtomicDataset,
    AtomicSample,
    DistributedSequentialSampler,
    GraphDataset,
    HDF5GraphShardDataset,
    ShardedGraphDataset,
    ShardedGraphDatasetWriter,
    build_graph_datasets,
    build_loaders,
    load_training_dataset,
    split_atomic_dataset,
    split_dataset_indices,
)
from gptff.graph import CrystalGraphConverter
from gptff.trainer.config import TrainingConfig
from gptff.trainer.distributed import DistributedContext
from gptff.utils.reproducibility import create_data_loader_generators


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


def test_sharded_graph_dataset_roundtrip_and_subset(tmp_path):
    atomic_dataset = AtomicDataset(
        (
            _sample(0, material_id="mat-a", stress=np.eye(3)),
            _sample(1, material_id="mat-b", stress=2 * np.eye(3)),
        )
    )
    converter = CrystalGraphConverter(radial_cutoff=2.0, angle_cutoff=2.0)
    output = tmp_path / "graphs.gptff"

    with ShardedGraphDatasetWriter(output, name="graphs", samples_per_shard=1) as writer:
        for sample in atomic_dataset:
            writer.add(
                graph=converter.convert(sample.structure),
                energy=sample.energy,
                forces=sample.forces,
                stress=sample.stress,
                sample_id=sample.sample_id,
                material_id=sample.material_id,
            )

    with h5py.File(output / "shards" / "shard_000000.h5", "r") as shard:
        stored_fields = set(shard["sample_00000000"])
    assert stored_fields.isdisjoint(
        {"edge_distances", "triplets_per_atom", "triplets_per_edge"}
    )

    dataset = ShardedGraphDataset(output)
    subset = dataset.subset([1], name="validation")
    sample = subset[0]
    _ = dataset[0]
    _ = dataset[1]

    assert len(dataset) == 2
    assert len(subset) == 1
    assert len(dataset._shard_datasets) == 2
    assert all(isinstance(shard, HDF5GraphShardDataset) for shard in dataset._shard_datasets)
    assert [shard._file is not None for shard in dataset._shard_datasets] == [True, True]
    dataset.close()
    assert all(shard._file is None for shard in dataset._shard_datasets)
    assert dataset.metadata["num_shards"] == 2
    assert dataset.metadata["samples_per_shard"] == 1
    assert dataset.metadata["num_samples"] == len(dataset) == 2
    assert dataset.metadata["radial_cutoff"] == pytest.approx(2.0)
    assert dataset.metadata["angle_cutoff"] == pytest.approx(2.0)
    assert subset.metadata["num_samples"] == len(subset) == 1
    assert subset.sample_key(0) == "frame-1"
    assert subset.material_id(0) == "mat-b"
    assert sample.energy == pytest.approx(-1.0)
    assert sample.forces.shape == (1, 3)
    assert np.allclose(sample.stress, -0.2 * np.eye(3, dtype=np.float32))


def test_sharded_graph_dataset_element_ref_records_do_not_load_graphs(tmp_path):
    atomic_dataset = AtomicDataset(
        (
            _sample(0, material_id="mat-a"),
            _sample(1, material_id="mat-b"),
        )
    )
    converter = CrystalGraphConverter(radial_cutoff=2.0, angle_cutoff=2.0)
    output = tmp_path / "graphs.gptff"

    with ShardedGraphDatasetWriter(output, samples_per_shard=2) as writer:
        for sample in atomic_dataset:
            writer.add(
                graph=converter.convert(sample.structure),
                energy=sample.energy,
                forces=sample.forces,
                stress=sample.stress,
                sample_id=sample.sample_id,
                material_id=sample.material_id,
            )

    records = list(ShardedGraphDataset(output).element_ref_records())

    assert [record.energy for record in records] == [0.0, -1.0]
    assert records[0].composition == {"11": 1}
    assert records[1].composition == {"17": 1}


def test_sharded_graph_dataset_writer_rejects_mixed_cutoffs(tmp_path):
    first_converter = CrystalGraphConverter(radial_cutoff=2.0, angle_cutoff=2.0)
    second_converter = CrystalGraphConverter(radial_cutoff=3.0, angle_cutoff=2.0)
    output = tmp_path / "graphs.gptff"
    sample = _sample(0)

    with ShardedGraphDatasetWriter(output, samples_per_shard=2) as writer:
        writer.add(
            graph=first_converter.convert(sample.structure),
            energy=sample.energy,
            forces=sample.forces,
            stress=sample.stress,
            sample_id="first",
            material_id=None,
        )
        with pytest.raises(ValueError, match="radial_cutoff"):
            writer.add(
                graph=second_converter.convert(sample.structure),
                energy=sample.energy,
                forces=sample.forces,
                stress=sample.stress,
                sample_id="second",
                material_id=None,
            )


def test_sharded_graph_dataset_training_loader_entrypoint(tmp_path):
    converter = CrystalGraphConverter(radial_cutoff=2.0, angle_cutoff=2.0)
    output = tmp_path / "graphs.gptff"

    with ShardedGraphDatasetWriter(
        output,
        name="graphs",
        metadata={"radial_cutoff": 2.0, "angle_cutoff": 2.0},
        samples_per_shard=2,
    ) as writer:
        for index in range(6):
            sample = _sample(index, material_id=f"material-{index // 2}")
            writer.add(
                graph=converter.convert(sample.structure),
                energy=sample.energy,
                forces=sample.forces,
                stress=sample.stress,
                sample_id=sample.sample_id,
                material_id=sample.material_id,
            )

    config = _sharded_training_config(
        output,
        validation_fraction=1.0 / 3.0,
        test_fraction=1.0 / 3.0,
        group_by_material=True,
    )

    dataset = load_training_dataset(config)
    splits = build_graph_datasets(dataset, config)
    loaders = build_loaders(
        config,
        splits,
        generators=create_data_loader_generators(config.training.seed),
    )
    batch = next(iter(loaders.train))

    assert isinstance(dataset, ShardedGraphDataset)
    assert isinstance(splits.train, ShardedGraphDataset)
    assert len(splits.train) == 2
    assert len(splits.validation) == 2
    assert splits.test is not None
    assert len(splits.test) == 2
    assert batch.energy is not None
    assert batch.forces is not None
    assert batch.stress is not None
    assert batch.num_atoms.tolist() == [1, 1]
    assert loaders.train.persistent_workers is False
    assert batch.radial_cutoff == pytest.approx(2.0)
    assert batch.angle_cutoff == pytest.approx(2.0)


def test_build_loaders_respects_persistent_workers_config(tmp_path):
    converter = CrystalGraphConverter(radial_cutoff=2.0, angle_cutoff=2.0)
    output = tmp_path / "graphs.gptff"

    with ShardedGraphDatasetWriter(
        output,
        metadata={"radial_cutoff": 2.0, "angle_cutoff": 2.0},
        samples_per_shard=2,
    ) as writer:
        for index in range(4):
            sample = _sample(index)
            writer.add(
                graph=converter.convert(sample.structure),
                energy=sample.energy,
                forces=sample.forces,
                stress=sample.stress,
                sample_id=sample.sample_id,
                material_id=sample.material_id,
            )

    config = _sharded_training_config(
        output,
        validation_fraction=0.25,
        persistent_workers=True,
        num_workers=1,
    )
    dataset = load_training_dataset(config)
    splits = build_graph_datasets(dataset, config)

    loaders = build_loaders(
        config,
        splits,
        generators=create_data_loader_generators(config.training.seed),
    )

    assert loaders.train.persistent_workers is True
    assert loaders.validation.persistent_workers is True


def test_graph_cache_requires_persistent_multiprocessing_workers():
    config = _sharded_training_config("unused", num_workers=1, persistent_workers=False)
    config.data = replace(config.data, cache_graphs=True)
    datasets = build_graph_datasets(_dataset(4), config)

    with pytest.raises(ValueError, match="cache_graphs.*persistent_workers"):
        build_loaders(
            config,
            datasets,
            generators=create_data_loader_generators(config.training.seed),
        )


def test_build_loaders_uses_distributed_samplers_without_eval_padding(tmp_path):
    converter = CrystalGraphConverter(radial_cutoff=2.0, angle_cutoff=2.0)
    output = tmp_path / "graphs.gptff"

    with ShardedGraphDatasetWriter(
        output,
        metadata={"radial_cutoff": 2.0, "angle_cutoff": 2.0},
        samples_per_shard=2,
    ) as writer:
        for index in range(6):
            sample = _sample(index, material_id=f"material-{index // 2}")
            writer.add(
                graph=converter.convert(sample.structure),
                energy=sample.energy,
                forces=sample.forces,
                stress=sample.stress,
                sample_id=sample.sample_id,
                material_id=sample.material_id,
            )

    config = _sharded_training_config(
        output,
        validation_fraction=1.0 / 3.0,
        test_fraction=1.0 / 3.0,
        group_by_material=True,
    )
    dataset = load_training_dataset(config)
    splits = build_graph_datasets(dataset, config)
    context = DistributedContext(enabled=True, rank=3, local_rank=3, world_size=4, device="cpu")

    loaders = build_loaders(
        config,
        splits,
        generators=create_data_loader_generators(config.training.seed),
        distributed=context,
    )

    assert loaders.train_sampler is not None
    assert isinstance(loaders.validation.sampler, DistributedSequentialSampler)
    assert len(loaders.validation.sampler) == 0
    assert list(loaders.validation) == []


def test_load_training_dataset_rejects_sharded_cutoff_mismatch(tmp_path):
    converter = CrystalGraphConverter(radial_cutoff=2.0, angle_cutoff=2.0)
    output = tmp_path / "graphs.gptff"
    sample = _sample(0)

    with ShardedGraphDatasetWriter(
        output,
        metadata={"radial_cutoff": 2.0, "angle_cutoff": 2.0},
        samples_per_shard=1,
    ) as writer:
        writer.add(
            graph=converter.convert(sample.structure),
            energy=sample.energy,
            forces=sample.forces,
            stress=sample.stress,
            sample_id=sample.sample_id,
            material_id=sample.material_id,
        )

    config = _sharded_training_config(output, radial_cutoff=3.0)

    with pytest.raises(ValueError, match="radial_cutoff=2"):
        load_training_dataset(config)


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


def test_split_dataset_indices_groups_material_ids():
    split = split_dataset_indices(
        4,
        validation_fraction=0.25,
        test_fraction=0.25,
        seed=3,
        material_ids=("a", "a", "b", "c"),
    )

    partitions = {}
    for partition_name, indices in (
        ("train", split.train_indices),
        ("validation", split.validation_indices),
        ("test", split.test_indices),
    ):
        for index in indices:
            material_id = ("a", "a", "b", "c")[index]
            assert material_id not in partitions or partitions[material_id] == partition_name
            partitions[material_id] = partition_name


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


def _sharded_training_config(
    dataset_path,
    *,
    radial_cutoff=2.0,
    angle_cutoff=2.0,
    validation_fraction=0.5,
    test_fraction=0.0,
    group_by_material=False,
    num_workers=0,
    persistent_workers=False,
):
    return TrainingConfig.from_dict(
        {
            "model": {
                "atom_feature_dim": 8,
                "edge_feature_dim": 8,
                "num_interaction_blocks": 1,
                "num_radial": 4,
                "num_angular": 3,
                "radial_cutoff": radial_cutoff,
                "angle_cutoff": angle_cutoff,
                "cutoff_coeff": 5,
                "max_atomic_number": 94,
                "num_readout_layers": 2,
                "readout_atom_norm": True,
                "interaction_dropout": 0.0,
                "atom_attention": {"enabled": False},
            },
            "optimizer": {"learning_rate": 1e-3},
            "training": {
                "epochs": 1,
                "batch_size": 2,
                "num_workers": num_workers,
                "persistent_workers": persistent_workers,
                "device": "cpu",
            },
            "loss": {
                "energy_loss_weight": 1.0,
                "force_loss_weight": 1.0,
                "stress_loss_weight": 0.1,
            },
            "data": {
                "dataset_path": str(dataset_path),
                "dataset_format": "sharded-hdf5-graph",
                "validation_fraction": validation_fraction,
                "test_fraction": test_fraction,
                "split_seed": 7,
                "group_by_material": group_by_material,
                "cache_graphs": False,
                "graph_cache_size": None,
            },
        }
    )
