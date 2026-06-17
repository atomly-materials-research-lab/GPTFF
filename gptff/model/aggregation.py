from __future__ import annotations

import torch


AGGREGATION_NORMS = {"sum", "mean", "sqrt"}


def validate_aggregation_norm(aggregation_norm: str) -> str:
    aggregation_norm = str(aggregation_norm).lower()
    if aggregation_norm not in AGGREGATION_NORMS:
        supported = ", ".join(sorted(AGGREGATION_NORMS))
        raise ValueError(f"aggregation_norm must be one of: {supported}.")
    return aggregation_norm


def normalize_aggregation(values: torch.Tensor, counts: torch.Tensor, mode: str) -> torch.Tensor:
    mode = validate_aggregation_norm(mode)
    if mode == "sum":
        return values

    counts = counts.to(device=values.device, dtype=values.dtype).clamp_min(1)
    if mode == "sqrt":
        counts = torch.sqrt(counts)

    view_shape = counts.shape + (1,) * (values.ndim - counts.ndim)
    return values / counts.reshape(view_shape)


def edge_counts_per_center(edge_index: torch.Tensor, num_atoms: int, reference: torch.Tensor) -> torch.Tensor:
    counts = reference.new_zeros((num_atoms,))
    if edge_index.shape[1] == 0:
        return counts

    ones = reference.new_ones((edge_index.shape[1],))
    return torch.index_add(counts, 0, edge_index[0], ones)
