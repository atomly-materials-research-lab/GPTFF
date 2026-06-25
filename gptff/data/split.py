from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DatasetSplit:
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    test_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        partitions = (
            self.train_indices,
            self.validation_indices,
            self.test_indices,
        )
        flattened = [index for partition in partitions for index in partition]
        if len(flattened) != len(set(flattened)):
            raise ValueError("Dataset split indices must not overlap.")
        if any(index < 0 for index in flattened):
            raise ValueError("Dataset split indices must be non-negative.")
        if not self.train_indices:
            raise ValueError("Training split must not be empty.")
        if not self.validation_indices:
            raise ValueError("Validation split must not be empty.")

    def validate_for_dataset(self, dataset) -> None:
        self.validate_for_size(len(dataset))

    def validate_for_size(self, dataset_size: int) -> None:
        all_indices = sorted(self.train_indices + self.validation_indices + self.test_indices)
        if all_indices != list(range(int(dataset_size))):
            raise ValueError("Dataset split must contain every sample exactly once.")


def split_atomic_dataset(
    dataset,
    *,
    validation_fraction: float,
    test_fraction: float = 0.0,
    seed: int = 42,
    group_by_material: bool = False,
) -> DatasetSplit:
    material_ids = None
    if group_by_material:
        material_ids = []
        for sample in dataset:
            if sample.material_id is None:
                raise ValueError("group_by_material requires material_id for every AtomicSample.")
            material_ids.append(sample.material_id)
    return split_dataset_indices(
        len(dataset),
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        seed=seed,
        material_ids=material_ids,
    )


def split_dataset_indices(
    dataset_size: int,
    *,
    validation_fraction: float,
    test_fraction: float = 0.0,
    seed: int = 42,
    material_ids: Sequence[str | None] | None = None,
) -> DatasetSplit:
    fractions = _validate_split_fractions(validation_fraction, test_fraction)
    dataset_size = int(dataset_size)
    target_counts = _target_partition_counts(dataset_size, fractions)
    rng = np.random.default_rng(int(seed))

    if material_ids is not None:
        split = _grouped_split(material_ids, target_counts, rng)
    else:
        split = _random_split(dataset_size, target_counts, rng)
    split.validate_for_size(dataset_size)
    return split


def _random_split(
    dataset_size: int,
    target_counts: Sequence[int],
    rng: np.random.Generator,
) -> DatasetSplit:
    shuffled = rng.permutation(dataset_size).tolist()
    train_count, validation_count, _ = target_counts
    train_end = train_count
    validation_end = train_end + validation_count
    return DatasetSplit(
        train_indices=tuple(sorted(shuffled[:train_end])),
        validation_indices=tuple(sorted(shuffled[train_end:validation_end])),
        test_indices=tuple(sorted(shuffled[validation_end:])),
    )


def _grouped_split(
    material_ids: Sequence[str | None],
    target_counts: Sequence[int],
    rng: np.random.Generator,
) -> DatasetSplit:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, material_id in enumerate(material_ids):
        if material_id is None:
            raise ValueError("group_by_material requires material_id for every sample.")
        groups[str(material_id)].append(index)

    active_partitions = [index for index, target in enumerate(target_counts) if target > 0]
    if len(groups) < len(active_partitions):
        raise ValueError("Not enough distinct material_id groups to populate all requested splits.")

    shuffled_groups = list(groups.values())
    rng.shuffle(shuffled_groups)
    shuffled_groups.sort(key=len, reverse=True)

    partitions: list[list[int]] = [[], [], []]
    counts = np.zeros(3, dtype=np.int64)
    targets = np.asarray(target_counts, dtype=np.float64)

    for group_index, group in enumerate(shuffled_groups):
        remaining_groups = len(shuffled_groups) - group_index
        empty_active = [index for index in active_partitions if counts[index] == 0]
        candidates = empty_active if remaining_groups == len(empty_active) else active_partitions

        candidate_order = rng.permutation(candidates).tolist()
        best_partition = min(
            candidate_order,
            key=lambda partition: _assignment_score(
                counts,
                targets,
                partition,
                len(group),
            ),
        )
        partitions[best_partition].extend(group)
        counts[best_partition] += len(group)

    return DatasetSplit(
        train_indices=tuple(sorted(partitions[0])),
        validation_indices=tuple(sorted(partitions[1])),
        test_indices=tuple(sorted(partitions[2])),
    )


def _assignment_score(
    counts: np.ndarray,
    targets: np.ndarray,
    partition: int,
    group_size: int,
) -> float:
    proposed = counts.astype(np.float64, copy=True)
    proposed[partition] += group_size
    scale = np.maximum(targets, 1.0)
    return float(np.sum(((proposed - targets) / scale) ** 2))


def _validate_split_fractions(
    validation_fraction: float,
    test_fraction: float,
) -> tuple[float, float, float]:
    validation_fraction = float(validation_fraction)
    test_fraction = float(test_fraction)
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if not 0.0 <= test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1.")
    train_fraction = 1.0 - validation_fraction - test_fraction
    if train_fraction <= 0.0:
        raise ValueError("validation_fraction + test_fraction must be less than 1.")
    return train_fraction, validation_fraction, test_fraction


def _target_partition_counts(
    dataset_size: int,
    fractions: Sequence[float],
) -> tuple[int, int, int]:
    active_count = sum(fraction > 0.0 for fraction in fractions)
    if dataset_size < active_count:
        raise ValueError(
            f"Dataset has {dataset_size} samples but {active_count} non-empty splits "
            "were requested."
        )

    ideal = np.asarray(fractions, dtype=np.float64) * dataset_size
    counts = np.floor(ideal).astype(np.int64)
    for partition, fraction in enumerate(fractions):
        if fraction > 0.0 and counts[partition] == 0:
            counts[partition] = 1

    while counts.sum() < dataset_size:
        residual = ideal - counts
        counts[int(np.argmax(residual))] += 1
    while counts.sum() > dataset_size:
        removable = [
            index
            for index, fraction in enumerate(fractions)
            if counts[index] > (1 if fraction > 0.0 else 0)
        ]
        if not removable:
            raise ValueError("Dataset is too small for the requested split fractions.")
        partition = min(removable, key=lambda index: ideal[index] - counts[index])
        counts[partition] -= 1
    return tuple(int(value) for value in counts)
