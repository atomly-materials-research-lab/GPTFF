from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RelaxationResult:
    """Result of a structure relaxation task."""

    final_structure: Any
    initial_structure: Any
    energy: float | None = None
    forces: Any | None = None
    stress: Any | None = None
    max_force: float | None = None
    is_converged: bool = False
    relaxation_requested: bool = False
    was_relaxed: bool = False
    n_steps: int = 0
    engine_name: str = ""
    model_name: str | None = None
    model_path: str | None = None
    units: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    warnings: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        units = {
            key: value
            for key, value in (
                ("energy", "eV"),
                ("forces", "eV/angstrom"),
                ("stress", "GPa"),
            )
            if getattr(self, key) is not None
        }
        if self.max_force is not None:
            units["max_force"] = "eV/angstrom"
        units.update(dict(self.units))
        self.units = units
        self.metadata = dict(self.metadata)
        self.warnings = tuple(self.warnings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_structure": self.final_structure,
            "initial_structure": self.initial_structure,
            "energy": self.energy,
            "forces": self.forces,
            "stress": self.stress,
            "max_force": self.max_force,
            "is_converged": self.is_converged,
            "relaxation_requested": self.relaxation_requested,
            "was_relaxed": self.was_relaxed,
            "n_steps": self.n_steps,
            "engine_name": self.engine_name,
            "model_name": self.model_name,
            "model_path": self.model_path,
            "units": dict(self.units),
            "metadata": dict(self.metadata),
            "warnings": tuple(self.warnings),
        }
