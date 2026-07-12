from pathlib import Path

import torch
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from gptff.runtime import GPTFFPotential
from gptff.utils.labels import stress_matrix_to_ase_voigt


class ASECalculator(Calculator):
    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(
        self,
        potential: GPTFFPotential | None = None,
        *,
        model_name: str | None = None,
        model_path: str | Path | None = None,
        device: str | torch.device | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if potential is not None and (
            model_name is not None or model_path is not None or device is not None
        ):
            raise ValueError("Pass either a potential or model selection arguments, not both.")
        self.potential = potential or GPTFFPotential.from_pretrained(
            model_name=model_name,
            model_path=model_path,
            device=device,
        )
        self.device = self.potential.device
        self.model_path = self.potential.model_path
        self.model_config = self.potential.model_config
        self.model = self.potential.model

    def predict_properties(self, batch, *, compute_stress: bool = True):
        return self.potential.predict_batch(
            batch,
            compute_stress=compute_stress,
        )

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list | None = None,
        system_changes: list | None = None,
    ):

        properties = properties or ["energy", "forces"]
        system_changes = system_changes or all_changes
        super().calculate(atoms=atoms, properties=properties, system_changes=system_changes)

        graph = self.potential.graph_converter.convert_ase_atoms(atoms)
        batch = self.potential.batch_graphs([graph])
        compute_stress = "stress" in properties
        energy, forces, stress = self.predict_properties(
            batch,
            compute_stress=compute_stress,
        )

        energy_value = float(energy.detach().cpu().item())
        results = dict(
            energy=energy_value,
            free_energy=energy_value,
            forces=forces.detach().cpu().numpy(),
        )
        if stress is not None:
            results["stress"] = stress_matrix_to_ase_voigt(stress[0].detach().cpu().numpy())
        self.results.update(results)
