from __future__ import annotations

from typing import Any, Mapping

import torch
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ExponentialLR,
    MultiStepLR,
    _LRScheduler,
)

Scheduler = _LRScheduler | CosineAnnealingWarmRestarts


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    scheduler: str,
    learning_rate: float,
    epochs: int,
    scheduler_params: Mapping[str, Any] | None = None,
) -> Scheduler | None:
    """Build the configured learning-rate scheduler."""
    name = scheduler.lower()
    params = dict(scheduler_params or {})

    if name in {"none", "off", "constant"}:
        return None

    if name in {"cosineannealinglr", "coslr", "cos", "cosine"}:
        decay_fraction = float(params.pop("decay_fraction", 1e-2))
        t_max = int(params.pop("T_max", params.pop("t_max", 10 * epochs)))
        return CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=decay_fraction * learning_rate,
            **params,
        )

    if name in {"cosrestartlr", "cosineannealingwarmrestarts", "cosrestart"}:
        decay_fraction = float(params.pop("decay_fraction", 1e-2))
        params.setdefault("T_0", 10)
        params.setdefault("T_mult", 2)
        return CosineAnnealingWarmRestarts(
            optimizer,
            eta_min=decay_fraction * learning_rate,
            **params,
        )

    if name in {"exponentiallr", "exp", "exponential"}:
        params.setdefault("gamma", 0.98)
        return ExponentialLR(optimizer, **params)

    if name in {"multisteplr", "multistep"}:
        params.setdefault("milestones", [4 * epochs, 6 * epochs, 8 * epochs, 9 * epochs])
        params.setdefault("gamma", 0.3)
        return MultiStepLR(optimizer, **params)

    raise ValueError(f"Unsupported scheduler: {scheduler}")
