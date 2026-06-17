from __future__ import annotations

from dataclasses import dataclass

import numpy as np


EV_PER_ANG3_TO_GPA = 160.21766208
_STRESS_UNIT_TO_GPA = {
    "gpa": 1.0,
    "kbar": 0.1,
    "ev_per_ang3": EV_PER_ANG3_TO_GPA,
}
_STRESS_UNIT_ALIASES = {
    "gpa": "gpa",
    "kbar": "kbar",
    "kilobar": "kbar",
    "ev/a^3": "ev_per_ang3",
    "ev/a3": "ev_per_ang3",
    "ev/ang^3": "ev_per_ang3",
    "ev/angstrom^3": "ev_per_ang3",
    "ev_per_a3": "ev_per_ang3",
    "ev_per_ang3": "ev_per_ang3",
    "ev_per_angstrom3": "ev_per_ang3",
}


@dataclass(frozen=True)
class LabelConfig:
    energy_unit: str = "ev"
    force_unit: str = "ev_per_ang"
    stress_unit: str = "kbar"
    stress_sign: float = -1.0

    def __post_init__(self) -> None:
        normalize_energy_unit(self.energy_unit)
        normalize_force_unit(self.force_unit)
        normalize_stress_unit(self.stress_unit)


def convert_stress_to_gpa(
    stress,
    *,
    unit: str,
    sign: float = 1.0,
    dtype=np.float32,
) -> np.ndarray:
    factor = stress_unit_factor_to_gpa(unit)
    return np.asarray(stress, dtype=dtype) * float(sign) * factor


def convert_energy_to_ev(energy, *, unit: str) -> float:
    key = normalize_energy_unit(unit)
    if key == "ev":
        return float(energy)
    raise AssertionError(f"Unhandled normalized energy unit: {key}.")


def convert_forces_to_ev_per_ang(forces, *, unit: str, dtype=np.float32) -> np.ndarray:
    key = normalize_force_unit(unit)
    if key == "ev_per_ang":
        return np.asarray(forces, dtype=dtype)
    raise AssertionError(f"Unhandled normalized force unit: {key}.")


def stress_unit_factor_to_gpa(unit: str) -> float:
    key = normalize_stress_unit(unit)
    return _STRESS_UNIT_TO_GPA[key]


def normalize_stress_unit(unit: str) -> str:
    key = str(unit).strip().lower()
    key = key.replace(" ", "").replace("-", "_")
    if key not in _STRESS_UNIT_ALIASES:
        allowed = ", ".join(sorted(_STRESS_UNIT_TO_GPA))
        raise ValueError(f"Unknown stress_unit '{unit}'. Supported units: {allowed}.")
    return _STRESS_UNIT_ALIASES[key]


def normalize_energy_unit(unit: str) -> str:
    key = str(unit).strip().lower()
    key = key.replace(" ", "").replace("-", "_")
    if key != "ev":
        raise ValueError("Only energy_unit='ev' is currently supported.")
    return key


def normalize_force_unit(unit: str) -> str:
    key = str(unit).strip().lower()
    key = key.replace(" ", "").replace("-", "_").replace("/", "_per_")
    aliases = {
        "ev_per_ang": "ev_per_ang",
        "ev_per_a": "ev_per_ang",
        "ev_per_angstrom": "ev_per_ang",
    }
    if key not in aliases:
        raise ValueError("Only force_unit='ev_per_ang' is currently supported.")
    return aliases[key]
