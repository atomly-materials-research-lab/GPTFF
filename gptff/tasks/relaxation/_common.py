from __future__ import annotations

import math
from typing import Any

import numpy as np


def normalize_external_pressure_gpa(value: float) -> float:
    pressure = float(value)
    if not math.isfinite(pressure):
        raise ValueError("external_pressure_gpa must be finite.")
    return pressure


def max_row_norm(values: Any | None) -> float:
    if values is None:
        return float("nan")
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return 0.0
    return float(np.linalg.norm(np.atleast_2d(array), axis=1).max())


def relaxation_warnings(
    *,
    is_converged: bool,
    relaxation_requested: bool,
    max_steps: int,
    atomic_max_force: float,
    optimizer_max_force: float,
    optimizer_force_source: str,
    optimizer_force_is_atomic_only: bool,
    optimizer_units_label: str,
    fmax: float,
) -> tuple[str, ...]:
    if is_converged:
        return ()
    if relaxation_requested:
        if not optimizer_force_is_atomic_only:
            return (
                f"Relaxation did not converge within {max_steps} steps: "
                f"optimizer_max_force={optimizer_max_force:.6g} from {optimizer_force_source}, "
                f"atomic_max_force={atomic_max_force:.6g} eV/angstrom, "
                f"fmax={fmax:.6g} in {optimizer_units_label}.",
            )
        return (
            f"Relaxation did not converge within {max_steps} steps: "
            f"max_force={atomic_max_force:.6g} eV/angstrom, "
            f"fmax={fmax:.6g} eV/angstrom.",
        )
    return (
        "Static force check exceeded the requested threshold: "
        f"max_force={atomic_max_force:.6g} eV/angstrom, "
        f"fmax={fmax:.6g} eV/angstrom.",
    )
