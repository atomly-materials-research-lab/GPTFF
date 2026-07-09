import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.filters import FrechetCellFilter
from pymatgen.core import Lattice, Structure

import gptff.tasks.relaxation.ase as ase_relaxation_module
from gptff.tasks import ASERelaxationRunner as top_level_ASERelaxationRunner
from gptff.tasks.relaxation import ASERelaxationRunner
from gptff.utils.labels import EV_PER_ANG3_TO_GPA


def test_ase_relaxation_runner_is_reexported_from_tasks():
    assert top_level_ASERelaxationRunner is ASERelaxationRunner


def test_ase_relaxation_runner_static_returns_metrics_and_pmg_structure():
    structure = Structure(Lattice.cubic(3.0), ["Na"], [[0.0, 0.0, 0.0]])
    stress_gpa = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    calculator = _ConstantCalculator(
        energy=1.25,
        forces=np.zeros((1, 3)),
        stress=stress_gpa / EV_PER_ANG3_TO_GPA,
    )

    runner = ASERelaxationRunner(
        ase_calculator=calculator,
        relax_atoms=False,
        relax_cell=False,
    )
    result = runner.run(structure)

    assert isinstance(result.final_structure, Structure)
    assert isinstance(result.initial_structure, Structure)
    assert result.energy == pytest.approx(1.25)
    assert result.max_force == pytest.approx(0.0)
    assert result.is_converged is True
    assert result.relaxation_requested is False
    assert result.was_relaxed is False
    assert result.n_steps == 0
    assert result.engine_name == "ase"
    assert result.units["stress"] == "GPa"
    assert np.allclose(
        result.stress,
        np.array(
            [
                [1.0, 6.0, 5.0],
                [6.0, 2.0, 4.0],
                [5.0, 4.0, 3.0],
            ]
        ),
    )


def test_ase_relaxation_runner_accepts_ase_atoms():
    atoms = Atoms(
        "Na",
        positions=[[0.0, 0.0, 0.0]],
        cell=[3.0, 3.0, 3.0],
        pbc=True,
    )

    runner = ASERelaxationRunner(
        ase_calculator=_ConstantCalculator(),
        relax_atoms=False,
        relax_cell=False,
    )
    result = runner.run(atoms)

    assert isinstance(result.final_structure, Structure)
    assert result.final_structure.formula == "Na1"


def test_ase_relaxation_runner_runs_atom_optimizer():
    atoms = Atoms(
        "Na",
        positions=[[0.2, 0.0, 0.0]],
        cell=[3.0, 3.0, 3.0],
        pbc=True,
    )

    runner = ASERelaxationRunner(
        ase_calculator=_HarmonicCalculator(),
        optimizer="FIRE",
        fmax=0.02,
        max_steps=200,
        relax_atoms=True,
        relax_cell=False,
    )
    result = runner.run(atoms)

    assert result.is_converged is True
    assert result.was_relaxed is True
    assert result.n_steps > 0
    assert result.max_force < 0.02
    assert result.metadata["max_force_kind"] == "atomic"
    assert result.metadata["optimizer_max_force"] == pytest.approx(result.max_force)
    assert result.metadata["optimizer_max_force_unit"] == "eV/angstrom"
    assert result.metadata["optimizer_force_includes_cell"] is False


def test_ase_relaxation_runner_cell_filter_tracks_generalized_force_and_pressure():
    _RecordingCellFilter.last_kwargs = None

    runner = ASERelaxationRunner(
        ase_calculator=_ConstantCalculator(
            forces=np.zeros((1, 3)),
            stress=np.zeros(6),
        ),
        max_steps=0,
        relax_atoms=True,
        relax_cell=True,
        external_pressure_gpa=2.0,
        cell_filter=_RecordingCellFilter,
    )
    result = runner.run(_structure())

    assert _RecordingCellFilter.last_kwargs is not None
    assert _RecordingCellFilter.last_kwargs["scalar_pressure"] == pytest.approx(
        2.0 / EV_PER_ANG3_TO_GPA
    )
    assert result.is_converged is False
    assert result.max_force == pytest.approx(0.0)
    assert result.metadata["optimizer_force_includes_cell"] is True
    assert result.metadata["optimizer_force_source"] == "_RecordingCellFilter.get_forces"
    assert result.metadata["optimizer_max_force_unit"] == "ASE generalized force"
    assert result.metadata["optimizer_max_force"] > result.max_force
    assert "optimizer_max_force" in result.warnings[0]
    assert "atomic_max_force" in result.warnings[0]


def test_ase_relaxation_runner_lazily_creates_and_reuses_default_calculator(monkeypatch):
    created = []

    class _FakeGPTFFASECalculator(_ConstantCalculator):
        def __init__(self, **kwargs):
            created.append(kwargs)
            super().__init__(forces=np.zeros((1, 3)), stress=np.zeros(6))
            self.potential = None

    monkeypatch.setattr(ase_relaxation_module, "ASECalculator", _FakeGPTFFASECalculator)

    runner = ASERelaxationRunner(
        relax_atoms=False,
        relax_cell=False,
        device="cpu",
    )

    assert created == []
    runner.run(_structure())
    runner.run(_structure())

    assert len(created) == 1
    assert created[0]["device"] == "cpu"


def test_ase_relaxation_runner_rejects_unknown_optimizer_before_default_calculator(
    monkeypatch,
):
    def fail_if_loaded(**kwargs):
        raise AssertionError("ASECalculator should not be created for an invalid optimizer.")

    monkeypatch.setattr(ase_relaxation_module, "ASECalculator", fail_if_loaded)

    with pytest.raises(ValueError, match="Unknown ASE optimizer"):
        ASERelaxationRunner(optimizer="NoSuchOptimizer")


def test_ase_relaxation_runner_rejects_cell_only_relaxation():
    with pytest.raises(ValueError, match="Cell-only ASE relaxation"):
        ASERelaxationRunner(
            ase_calculator=_ConstantCalculator(),
            relax_atoms=False,
            relax_cell=True,
        )


def test_ase_relaxation_runner_rejects_pressure_without_cell_relaxation():
    with pytest.raises(ValueError, match="external_pressure_gpa requires relax_cell"):
        ASERelaxationRunner(
            ase_calculator=_ConstantCalculator(),
            relax_atoms=True,
            relax_cell=False,
            external_pressure_gpa=1.0,
        )


def test_ase_relaxation_runner_requires_stress_for_cell_relaxation():
    runner = ASERelaxationRunner(
        ase_calculator=_NoStressCalculator(),
        relax_atoms=True,
        relax_cell=True,
    )

    with pytest.raises(ValueError, match="stress"):
        runner.run(_structure())


def test_ase_relaxation_runner_rejects_mixed_calculator_and_model_args():
    with pytest.raises(ValueError, match="ase_calculator or GPTFF model selection"):
        ASERelaxationRunner(
            ase_calculator=_ConstantCalculator(),
            model_path="custom.pt",
        )


def test_ase_relaxation_runner_rejects_mixed_potential_and_model_args():
    with pytest.raises(ValueError, match="potential or model selection"):
        ASERelaxationRunner(
            potential=object(),
            device="cpu",
        )


def test_ase_relaxation_runner_rejects_mixed_model_name_and_model_path():
    with pytest.raises(ValueError, match="model_name or model_path"):
        ASERelaxationRunner(
            model_name="default",
            model_path="custom.pt",
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"fmax": 0.0}, "fmax"),
        ({"max_steps": -1}, "max_steps"),
        ({"symprec": 0.0}, "symprec"),
    ],
)
def test_ase_relaxation_runner_validates_numeric_options(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ASERelaxationRunner(**kwargs)


def test_ase_relaxation_runner_rejects_scalar_pressure_cell_filter_kwarg():
    with pytest.raises(ValueError, match="external_pressure_gpa"):
        ASERelaxationRunner(
            cell_filter_kwargs={"scalar_pressure": 1.0},
        )


def _structure():
    return Structure(Lattice.cubic(3.0), ["Na"], [[0.0, 0.0, 0.0]])


class _ConstantCalculator(Calculator):
    implemented_properties = ["energy", "forces", "stress"]

    def __init__(
        self,
        *,
        energy=0.0,
        forces=None,
        stress=None,
    ):
        super().__init__()
        self.energy = float(energy)
        self.forces = None if forces is None else np.asarray(forces, dtype=float)
        self.stress = (
            np.zeros(6, dtype=float) if stress is None else np.asarray(stress, dtype=float)
        )

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms=atoms, properties=properties, system_changes=system_changes)
        forces = self.forces
        if forces is None:
            forces = np.zeros((len(atoms), 3), dtype=float)
        self.results = {
            "energy": self.energy,
            "forces": forces,
            "stress": self.stress,
        }


class _NoStressCalculator(_ConstantCalculator):
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms=atoms, properties=properties, system_changes=system_changes)
        self.results.pop("stress", None)


class _HarmonicCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, *, spring_constant=1.0):
        super().__init__()
        self.spring_constant = float(spring_constant)

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms=atoms, properties=properties, system_changes=system_changes)
        positions = atoms.get_positions()
        energy = 0.5 * self.spring_constant * float(np.sum(positions**2))
        forces = -self.spring_constant * positions
        self.results = {
            "energy": energy,
            "forces": forces,
        }


class _RecordingCellFilter(FrechetCellFilter):
    last_kwargs = None

    def __init__(self, atoms, **kwargs):
        type(self).last_kwargs = dict(kwargs)
        super().__init__(atoms, **kwargs)
