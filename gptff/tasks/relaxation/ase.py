from __future__ import annotations

from collections.abc import Mapping
from inspect import isclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import ase.optimize
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.constraints import FixSymmetry
from ase.filters import FrechetCellFilter
from ase.optimize.optimize import Optimizer
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from gptff.interfaces import ASECalculator
from gptff.runtime import GPTFFPotential
from gptff.tasks.relaxation._common import (
    max_row_norm,
    normalize_external_pressure_gpa,
    relaxation_warnings,
)
from gptff.tasks.relaxation.result import RelaxationResult
from gptff.utils.labels import EV_PER_ANG3_TO_GPA

if TYPE_CHECKING:
    from ase.filters import Filter

GPA_TO_EV_PER_ANG3 = 1.0 / EV_PER_ANG3_TO_GPA


def relax_with_ase(
    structure: Structure | Atoms,
    *,
    potential: GPTFFPotential | None = None,
    ase_calculator: Calculator | None = None,
    model_name: str | None = None,
    model_path: str | Path | None = None,
    device: str | None = None,
    optimizer: str | type[Optimizer] = "FIRE",
    fmax: float = 0.05,
    max_steps: int = 500,
    relax_atoms: bool = True,
    relax_cell: bool = True,
    fix_symmetry: bool = False,
    symprec: float = 1e-2,
    external_pressure_gpa: float = 0.0,
    cell_filter: type[Filter] = FrechetCellFilter,
    cell_filter_kwargs: Mapping[str, Any] | None = None,
    logfile: str | None = None,
    trajectory: str | None = None,
) -> RelaxationResult:
    """Relax a pymatgen Structure or ASE Atoms object with the ASE engine."""

    if not relax_atoms and relax_cell:
        raise ValueError("Cell-only ASE relaxation is not supported.")

    pressure = normalize_external_pressure_gpa(external_pressure_gpa)
    if pressure != 0.0 and not relax_cell:
        raise ValueError("external_pressure_gpa requires relax_cell=True.")

    kwargs = dict(cell_filter_kwargs or {})
    if "scalar_pressure" in kwargs:
        raise ValueError(
            "Use external_pressure_gpa instead of cell_filter_kwargs['scalar_pressure']."
        )
    if pressure != 0.0:
        kwargs["scalar_pressure"] = pressure * GPA_TO_EV_PER_ANG3

    calculator = _resolve_ase_calculator(
        potential=potential,
        ase_calculator=ase_calculator,
        model_name=model_name,
        model_path=model_path,
        device=device,
    )
    _validate_calculator_properties(calculator, require_stress=relax_cell)

    atoms = _to_ase_atoms(structure)
    initial_structure = _to_pmg_structure(atoms)
    if fix_symmetry:
        atoms.set_constraint(FixSymmetry(atoms, symprec=symprec))
    atoms.calc = calculator

    relaxation_requested = relax_atoms or relax_cell
    include_stress = "stress" in calculator.implemented_properties
    observer = _RelaxationObserver(atoms, include_stress=include_stress)
    n_steps = 0
    run_converged: bool | None = None
    optimizer_max_force: float | None = None
    optimizer_force_source = "atoms.get_forces"
    if relaxation_requested:
        opt_target = cell_filter(atoms, **kwargs) if relax_cell else atoms
        if relax_cell:
            optimizer_force_source = f"{cell_filter.__name__}.get_forces"
        optimizer_cls = _get_ase_optimizer(optimizer)
        ase_optimizer = optimizer_cls(
            opt_target,
            logfile=logfile,
            trajectory=trajectory,
        )
        ase_optimizer.attach(observer, interval=1)
        run_result = ase_optimizer.run(fmax=fmax, steps=max_steps)
        run_converged = None if run_result is None else bool(run_result)
        n_steps = int(getattr(ase_optimizer, "nsteps", 0))
        if hasattr(opt_target, "atoms"):
            atoms = opt_target.atoms
        observer.atoms = atoms
        if not observer.has_snapshot:
            observer()
        optimizer_max_force = max_row_norm(opt_target.get_forces())
    else:
        observer()

    max_force = max_row_norm(observer.forces)
    if optimizer_max_force is None:
        optimizer_max_force = max_force
    is_converged = run_converged if run_converged is not None else max_force <= fmax
    warnings = relaxation_warnings(
        is_converged=is_converged,
        relaxation_requested=relaxation_requested,
        max_steps=max_steps,
        atomic_max_force=max_force,
        optimizer_max_force=optimizer_max_force,
        optimizer_force_source=optimizer_force_source,
        optimizer_force_is_atomic_only=not relax_cell,
        optimizer_units_label="ASE optimizer units",
        fmax=fmax,
    )
    metadata = _relaxation_metadata(
        calculator,
        optimizer=optimizer,
        fmax=fmax,
        max_steps=max_steps,
        relax_atoms=relax_atoms,
        relax_cell=relax_cell,
        fix_symmetry=fix_symmetry,
        symprec=symprec,
        external_pressure_gpa=pressure,
        relaxation_requested=relaxation_requested,
        was_relaxed=n_steps > 0,
        cell_filter=cell_filter,
        include_stress=include_stress,
        optimizer_max_force=optimizer_max_force,
        optimizer_force_source=optimizer_force_source,
        optimizer_force_includes_cell=relax_cell,
    )

    return RelaxationResult(
        initial_structure=initial_structure,
        final_structure=_to_pmg_structure(_without_fix_symmetry_constraint(atoms)),
        energy=observer.energy,
        forces=observer.forces,
        stress=observer.stress,
        max_force=max_force,
        is_converged=is_converged,
        relaxation_requested=relaxation_requested,
        was_relaxed=n_steps > 0,
        n_steps=n_steps,
        engine_name="ase",
        model_name=metadata.get("model_name"),
        model_path=metadata.get("model_path"),
        metadata=metadata,
        warnings=warnings,
    )


def _resolve_ase_calculator(
    *,
    potential: GPTFFPotential | None,
    ase_calculator: Calculator | None,
    model_name: str | None,
    model_path: str | Path | None,
    device: str | None,
) -> Calculator:
    if ase_calculator is not None and (
        potential is not None
        or model_name is not None
        or model_path is not None
        or device is not None
    ):
        raise ValueError("Pass either ase_calculator or GPTFF model selection arguments, not both.")
    if ase_calculator is not None:
        return ase_calculator
    return ASECalculator(
        potential=potential,
        model_name=model_name,
        model_path=model_path,
        device=device,
    )


def _validate_calculator_properties(calculator: Calculator, *, require_stress: bool) -> None:
    implemented = set(calculator.implemented_properties)
    missing = {"energy", "forces"} - implemented
    if require_stress and "stress" not in implemented:
        missing.add("stress")
    if missing:
        raise ValueError(f"ASE relaxation calculator is missing properties: {sorted(missing)}.")


def _to_ase_atoms(structure: Structure | Atoms) -> Atoms:
    if isinstance(structure, Atoms):
        return structure.copy()
    if isinstance(structure, Structure):
        return AseAtomsAdaptor().get_atoms(structure)
    raise TypeError("structure must be a pymatgen Structure or ASE Atoms object.")


def _to_pmg_structure(atoms: Atoms) -> Structure:
    return AseAtomsAdaptor().get_structure(atoms)


def _get_ase_optimizer(optimizer: str | type[Optimizer]) -> type[Optimizer]:
    if isclass(optimizer) and issubclass(optimizer, Optimizer):
        return optimizer
    if isinstance(optimizer, str):
        optimizer_cls = getattr(ase.optimize, optimizer, None)
        if isclass(optimizer_cls) and issubclass(optimizer_cls, Optimizer):
            return optimizer_cls
    raise ValueError(f"Unknown ASE optimizer {optimizer!r}.")


def _optimizer_name(optimizer: str | type[Optimizer]) -> str:
    return optimizer if isinstance(optimizer, str) else optimizer.__name__


def _without_fix_symmetry_constraint(atoms: Atoms) -> Atoms:
    constraints = list(getattr(atoms, "constraints", ()))
    if not constraints:
        return atoms

    kept_constraints = [
        constraint for constraint in constraints if not isinstance(constraint, FixSymmetry)
    ]
    if len(kept_constraints) == len(constraints):
        return atoms

    unconstrained_atoms = atoms.copy()
    unconstrained_atoms.set_constraint(kept_constraints)
    return unconstrained_atoms


def _relaxation_metadata(
    calculator: Calculator,
    *,
    optimizer: str | type[Optimizer],
    fmax: float,
    max_steps: int,
    relax_atoms: bool,
    relax_cell: bool,
    fix_symmetry: bool,
    symprec: float,
    external_pressure_gpa: float,
    relaxation_requested: bool,
    was_relaxed: bool,
    cell_filter: type[Filter],
    include_stress: bool,
    optimizer_max_force: float,
    optimizer_force_source: str,
    optimizer_force_includes_cell: bool,
) -> dict[str, Any]:
    potential = getattr(calculator, "potential", None)
    metadata = {
        "task": "relaxation",
        "engine": "ase",
        "optimizer": _optimizer_name(optimizer),
        "fmax": float(fmax),
        "max_steps": int(max_steps),
        "relax_atoms": bool(relax_atoms),
        "relax_cell": bool(relax_cell),
        "fix_symmetry": bool(fix_symmetry),
        "symprec": float(symprec) if fix_symmetry else None,
        "external_pressure_gpa": float(external_pressure_gpa),
        "external_pressure_unit": "GPa",
        "relaxation_requested": bool(relaxation_requested),
        "was_relaxed": bool(was_relaxed),
        "cell_filter": cell_filter.__name__ if relax_cell else None,
        "ase_stress_input_unit": "eV/angstrom^3" if include_stress else None,
        "max_force_kind": "atomic",
        "max_force_source": "atoms.get_forces",
        "optimizer_max_force": float(optimizer_max_force),
        "optimizer_max_force_unit": (
            "ASE generalized force" if optimizer_force_includes_cell else "eV/angstrom"
        ),
        "optimizer_force_source": optimizer_force_source,
        "optimizer_force_includes_cell": bool(optimizer_force_includes_cell),
        "model_name": getattr(potential, "model_name", None),
        "model_path": (
            None if getattr(potential, "model_path", None) is None else str(potential.model_path)
        ),
    }
    return metadata


class _RelaxationObserver:
    def __init__(self, atoms: Atoms, *, include_stress: bool) -> None:
        self.atoms = atoms
        self.include_stress = include_stress
        self.energy: float | None = None
        self.forces: np.ndarray | None = None
        self.stress: np.ndarray | None = None

    @property
    def has_snapshot(self) -> bool:
        return self.forces is not None

    def __call__(self) -> None:
        self.energy = float(self.atoms.get_potential_energy())
        self.forces = np.asarray(self.atoms.get_forces(), dtype=float)
        if self.include_stress:
            stress = np.asarray(self.atoms.get_stress(voigt=False), dtype=float)
            self.stress = stress * EV_PER_ANG3_TO_GPA
