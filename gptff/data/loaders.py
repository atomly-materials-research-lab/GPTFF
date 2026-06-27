from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler

from gptff.data.dataset import (
    AtomicDataset,
    GraphDataset,
    collate_graph_samples,
)
from gptff.data.sharded_graph import ShardedGraphDataset
from gptff.data.split import split_atomic_dataset, split_dataset_indices
from gptff.model.readout import fit_element_refs_from_samples
from gptff.utils.reproducibility import seed_data_loader_worker

if TYPE_CHECKING:
    from gptff.trainer.config import TrainingConfig


@dataclass(frozen=True)
class GraphDatasetSplits:
    train: GraphDataset | ShardedGraphDataset
    validation: GraphDataset | ShardedGraphDataset
    test: GraphDataset | ShardedGraphDataset | None


@dataclass(frozen=True)
class DataLoaders:
    train: DataLoader
    validation: DataLoader
    test: DataLoader | None
    train_sampler: DistributedSampler | None = None


def load_atomic_dataset(config: TrainingConfig) -> AtomicDataset:
    if config.dataset_path is None:
        raise ValueError(
            "data.dataset_path is required when Trainer.fit() is called without an AtomicDataset."
        )
    return AtomicDataset.from_file(config.dataset_path)


def load_training_dataset(config: TrainingConfig) -> AtomicDataset | ShardedGraphDataset:
    if config.dataset_path is None:
        raise ValueError(
            "data.dataset_path is required when Trainer.fit() is called without a dataset."
        )
    if config.dataset_format == "atomic_json":
        return AtomicDataset.from_file(config.dataset_path)
    if config.dataset_format == "sharded_hdf5_graph":
        dataset = ShardedGraphDataset(
            config.dataset_path,
            max_open_files=config.max_open_files,
        )
        _validate_sharded_dataset_cutoffs(dataset, config)
        return dataset
    raise ValueError(f"Unsupported data.dataset_format: {config.dataset_format!r}")


def build_graph_datasets(
    dataset: AtomicDataset | ShardedGraphDataset,
    config: TrainingConfig,
) -> GraphDatasetSplits:
    _validate_training_labels(dataset, require_stress=config.stress_loss_weight > 0.0)
    if isinstance(dataset, ShardedGraphDataset):
        return _build_sharded_graph_datasets(dataset, config)

    split = split_atomic_dataset(
        dataset,
        validation_fraction=config.validation_fraction,
        test_fraction=config.test_fraction,
        seed=config.split_seed,
        group_by_material=config.group_by_material,
    )

    train = _build_graph_dataset(
        dataset.subset(split.train_indices, name="train"),
        config,
    )
    validation = _build_graph_dataset(
        dataset.subset(split.validation_indices, name="validation"),
        config,
    )
    test = None
    if split.test_indices:
        test = _build_graph_dataset(
            dataset.subset(split.test_indices, name="test"),
            config,
        )
    return GraphDatasetSplits(
        train=train,
        validation=validation,
        test=test,
    )


def _build_sharded_graph_datasets(
    dataset: ShardedGraphDataset,
    config: TrainingConfig,
) -> GraphDatasetSplits:
    material_ids = None
    if config.group_by_material:
        material_ids = [record.material_id for record in dataset.records]
    split = split_dataset_indices(
        len(dataset),
        validation_fraction=config.validation_fraction,
        test_fraction=config.test_fraction,
        seed=config.split_seed,
        material_ids=material_ids,
    )
    return GraphDatasetSplits(
        train=dataset.subset(split.train_indices, name="train"),
        validation=dataset.subset(split.validation_indices, name="validation"),
        test=None if not split.test_indices else dataset.subset(split.test_indices, name="test"),
    )


def apply_fitted_element_refs(config: TrainingConfig, dataset, *, verbose: bool = True) -> None:
    if not config.element_references.fit_from_training_data:
        return
    if config.element_refs is not None:
        raise ValueError(
            "element_references.source='fit' cannot be combined with preloaded element refs."
        )
    if verbose:
        print("Fitting element_refs from the full dataset.")
    samples = dataset.element_ref_records() if hasattr(dataset, "element_ref_records") else dataset
    config.element_refs = fit_element_refs_from_samples(
        samples,
        max_atomic_number=config.max_atomic_number,
    )


def build_loaders(
    config: TrainingConfig,
    datasets: GraphDatasetSplits,
    *,
    generators: dict[str, torch.Generator],
    distributed=None,
) -> DataLoaders:
    distributed = distributed or _NoDistributedContext()
    pin_memory = torch.device(config.device).type == "cuda"
    common = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "collate_fn": collate_graph_samples,
        "pin_memory": pin_memory,
        "worker_init_fn": seed_data_loader_worker,
    }
    if config.num_workers > 0:
        common["persistent_workers"] = config.persistent_workers
        common["prefetch_factor"] = 2
    train_sampler = None
    if distributed.enabled:
        train_sampler = DistributedSampler(
            datasets.train,
            num_replicas=distributed.world_size,
            rank=distributed.rank,
            shuffle=True,
            seed=config.seed,
            drop_last=False,
        )
        train = DataLoader(
            datasets.train,
            shuffle=False,
            sampler=train_sampler,
            generator=generators["train"],
            **common,
        )
    else:
        train = DataLoader(
            datasets.train,
            shuffle=True,
            generator=generators["train"],
            **common,
        )
    validation_sampler = _eval_sampler(datasets.validation, distributed)
    validation = DataLoader(
        datasets.validation,
        shuffle=False,
        sampler=validation_sampler,
        generator=generators["validation"],
        **common,
    )
    test = None
    if datasets.test is not None:
        test_sampler = _eval_sampler(datasets.test, distributed)
        test = DataLoader(
            datasets.test,
            shuffle=False,
            sampler=test_sampler,
            generator=generators["test"],
            **common,
        )
    return DataLoaders(train=train, validation=validation, test=test, train_sampler=train_sampler)


class DistributedSequentialSampler(Sampler[int]):
    """No-padding sequential eval sampler for rank-local dataset indices."""

    def __init__(self, dataset, context) -> None:
        self.indices = tuple(range(context.rank, len(dataset), context.world_size))

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def _eval_sampler(dataset, distributed):
    if not distributed.enabled:
        return None
    return DistributedSequentialSampler(dataset, distributed)


@dataclass(frozen=True)
class _NoDistributedContext:
    enabled: bool = False
    rank: int = 0
    world_size: int = 1


def _build_graph_dataset(
    dataset: AtomicDataset,
    config: TrainingConfig,
) -> GraphDataset:
    return GraphDataset(
        dataset,
        radial_cutoff=config.radial_cutoff,
        angle_cutoff=config.angle_cutoff,
        cache_graphs=config.cache_graphs,
        cache_size=config.graph_cache_size,
    )


def _validate_sharded_dataset_cutoffs(
    dataset: ShardedGraphDataset,
    config: TrainingConfig,
) -> None:
    _validate_sharded_cutoff(
        dataset,
        metadata_key="radial_cutoff",
        expected=config.radial_cutoff,
        config_key="model.radial_cutoff",
    )
    _validate_sharded_cutoff(
        dataset,
        metadata_key="angle_cutoff",
        expected=config.angle_cutoff,
        config_key="model.angle_cutoff",
    )


def _validate_sharded_cutoff(
    dataset: ShardedGraphDataset,
    *,
    metadata_key: str,
    expected: float,
    config_key: str,
) -> None:
    if metadata_key not in dataset.metadata:
        return
    actual = float(dataset.metadata[metadata_key])
    if math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-8):
        return
    raise ValueError(
        f"Sharded graph dataset {metadata_key}={actual:g} does not match "
        f"{config_key}={expected:g}. Regenerate the sharded dataset with matching "
        "cutoffs or update the training config."
    )


def _validate_training_labels(
    dataset: AtomicDataset | ShardedGraphDataset,
    *,
    require_stress: bool,
) -> None:
    if not require_stress:
        return
    if isinstance(dataset, ShardedGraphDataset):
        missing = [
            dataset.sample_key(index) for index in range(len(dataset)) if not dataset.has_stress(index)
        ]
        _raise_missing_stress(missing)
        return

    missing = [
        dataset.sample_key(index) for index, sample in enumerate(dataset) if sample.stress is None
    ]
    if missing:
        _raise_missing_stress(missing)


def _raise_missing_stress(missing: list[str]) -> None:
    if not missing:
        return
    preview = ", ".join(missing[:5])
    suffix = "" if len(missing) <= 5 else f" and {len(missing) - 5} more"
    raise ValueError(
        "stress labels are required when stress_loss_weight is positive; "
        f"missing for {preview}{suffix}."
    )
