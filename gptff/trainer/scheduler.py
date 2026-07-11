from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ExponentialLR,
    LambdaLR,
    MultiStepLR,
    _LRScheduler,
)

Scheduler = _LRScheduler | CosineAnnealingWarmRestarts
DEFAULT_SCHEDULER_STEPS_PER_EPOCH = 10
COSINE_SCHEDULERS = frozenset({"cosineannealinglr", "coslr", "cos", "cosine"})
COSINE_RESTART_SCHEDULERS = frozenset(
    {"cosrestartlr", "cosineannealingwarmrestarts", "cosrestart"}
)
EXPONENTIAL_SCHEDULERS = frozenset({"exponentiallr", "exp", "exponential"})
MULTISTEP_SCHEDULERS = frozenset({"multisteplr", "multistep"})


def scheduler_steps_per_epoch(
    scheduler_params: Mapping[str, Any] | None = None,
) -> int:
    """Return the configured number of scheduler steps per epoch."""
    params = dict(scheduler_params or {})
    steps_per_epoch = int(params.get("steps_per_epoch", DEFAULT_SCHEDULER_STEPS_PER_EPOCH))
    if steps_per_epoch <= 0:
        raise ValueError("steps_per_epoch must be positive.")
    return steps_per_epoch


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

    if name in EXPONENTIAL_SCHEDULERS | MULTISTEP_SCHEDULERS and "decay_fraction" in params:
        raise ValueError(
            "decay_fraction, min_learning_rate, and min_lr are only supported by "
            "cosine schedulers."
        )

    steps_per_epoch = scheduler_steps_per_epoch(params)
    params.pop("steps_per_epoch", None)

    if name in COSINE_SCHEDULERS:
        decay_fraction = float(params.pop("decay_fraction", 1e-2))
        t_max = int(params.pop("T_max", params.pop("t_max", steps_per_epoch * epochs)))
        warmup_steps = int(params.pop("warmup_steps", 0))
        warmup_epochs = float(params.pop("warmup_epochs", 0.0))
        if warmup_steps <= 0 and warmup_epochs > 0.0:
            warmup_steps = int(math.ceil(warmup_epochs * steps_per_epoch))
        warmup_start_factor = float(
            params.pop("warmup_start_factor", params.pop("warmup_factor", 0.1))
        )

        if warmup_steps <= 0:
            return CosineAnnealingLR(
                optimizer,
                T_max=t_max,
                eta_min=decay_fraction * learning_rate,
                **params,
            )

        if not 0.0 <= warmup_start_factor <= 1.0:
            raise ValueError("warmup_start_factor must be in the interval [0, 1].")
        if warmup_steps >= t_max:
            raise ValueError("warmup_steps must be smaller than T_max.")

        cosine_steps = max(1, t_max - warmup_steps)

        def lr_lambda(step: int) -> float:
            if step <= warmup_steps:
                alpha = step / float(warmup_steps)
                return warmup_start_factor * (1.0 - alpha) + alpha
            progress = min((step - warmup_steps) / float(cosine_steps), 1.0)
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            return decay_fraction + (1.0 - decay_fraction) * cosine_factor

        return LambdaLR(optimizer, lr_lambda=lr_lambda)

    if name in COSINE_RESTART_SCHEDULERS:
        decay_fraction = float(params.pop("decay_fraction", 1e-2))
        params.setdefault("T_0", 10)
        params.setdefault("T_mult", 2)
        return CosineAnnealingWarmRestarts(
            optimizer,
            eta_min=decay_fraction * learning_rate,
            **params,
        )

    if name in EXPONENTIAL_SCHEDULERS:
        params.setdefault("gamma", 0.98)
        return ExponentialLR(optimizer, **params)

    if name in MULTISTEP_SCHEDULERS:
        params.setdefault("milestones", [4 * epochs, 6 * epochs, 8 * epochs, 9 * epochs])
        params.setdefault("gamma", 0.3)
        return MultiStepLR(optimizer, **params)

    raise ValueError(f"Unsupported scheduler: {scheduler}")
