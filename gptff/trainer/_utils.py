from __future__ import annotations

import math


def normalize_distributed_mode(mode: str | bool) -> str:
    if isinstance(mode, bool):
        return "true" if mode else "false"
    normalized = str(mode).strip().lower()
    if normalized in {"auto", "true", "false"}:
        return normalized
    raise ValueError("training.distributed must be one of: auto, true, false.")


def format_value(value: float, precision: int) -> str:
    if math.isnan(value):
        return "n/a"
    return f"{value:.{precision}f}"
