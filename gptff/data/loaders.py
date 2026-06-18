from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader

from gptff.data.dataset import (
    AtomicDataset,
    GraphDataset,
    collate_graph_samples,
)
from gptff.data.split import split_atomic_dataset
from gptff.model.readout import fit_element_refs_from_samples
from gptff.utils.reproducibility import seed_data_loader_worker

if TYPE_CHECKING:
    from gptff.trainer.config import TrainingConfig


@dataclass(frozen=True)
class GraphDatasetSplits:
    train: GraphDataset
    validation: GraphDataset
    test: GraphDataset | None


@dataclass(frozen=True)
class DataLoaders:
    train: DataLoader
    validation: DataLoader
    test: DataLoader | None


def load_atomic_dataset(config: TrainingConfig) -> AtomicDataset:
    if config.dataset_path is None:
        raise ValueError(
            "data.dataset_path is required when Trainer.fit() is called without an AtomicDataset."
        )
    return AtomicDataset.from_file(config.dataset_path)


def build_graph_datasets(
    dataset: AtomicDataset,
    config: TrainingConfig,
) -> GraphDatasetSplits:
    _validate_training_labels(dataset, require_stress=config.stress_loss_weight > 0.0)
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


def apply_fitted_element_refs(config: TrainingConfig, train_dataset) -> None:
    if not config.element_references.fit_from_training_data:
        return
    if config.element_refs is not None:
        raise ValueError(
            "element_references.source='fit' cannot be combined with preloaded element refs."
        )
    print("Fitting element_refs from the training dataset.")
    config.element_refs = fit_element_refs_from_samples(
        train_dataset,
        max_atomic_number=config.max_atomic_number,
        ridge=config.element_references.ridge,
    )


def build_loaders(
    config: TrainingConfig,
    datasets: GraphDatasetSplits,
    *,
    generators: dict[str, torch.Generator],
) -> DataLoaders:
    pin_memory = torch.device(config.device).type == "cuda"
    common = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "collate_fn": collate_graph_samples,
        "pin_memory": pin_memory,
        "worker_init_fn": seed_data_loader_worker,
    }
    train = DataLoader(
        datasets.train,
        shuffle=True,
        generator=generators["train"],
        **common,
    )
    validation = DataLoader(
        datasets.validation,
        shuffle=False,
        generator=generators["validation"],
        **common,
    )
    test = None
    if datasets.test is not None:
        test = DataLoader(
            datasets.test,
            shuffle=False,
            generator=generators["test"],
            **common,
        )
    return DataLoaders(train=train, validation=validation, test=test)


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


def _validate_training_labels(
    dataset: AtomicDataset,
    *,
    require_stress: bool,
) -> None:
    if not require_stress:
        return
    missing = [
        dataset.sample_key(index) for index, sample in enumerate(dataset) if sample.stress is None
    ]
    if missing:
        preview = ", ".join(missing[:5])
        suffix = "" if len(missing) <= 5 else f" and {len(missing) - 5} more"
        raise ValueError(
            "stress labels are required when stress_loss_weight is positive; "
            f"missing for {preview}{suffix}."
        )
