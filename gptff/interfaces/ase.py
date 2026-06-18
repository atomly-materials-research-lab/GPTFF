import torch
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from gptff.graph import CrystalGraphBatch, CrystalGraphConverter
from gptff.inference import predict_energy_forces_stress
from gptff.model.config import GPTFFConfig
from gptff.model.model import GPTFF
from gptff.utils.labels import stress_gpa_to_ase_voigt


class ASECalculator(Calculator):
    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(self, model_path, device="cuda", **kwargs):
        super().__init__(**kwargs)

        self.state = torch.load(model_path, map_location=torch.device(device))

        self.model_config = GPTFFConfig.from_dict(self.state["model_config"])
        self.model = GPTFF(self.model_config)
        self.device = device
        self.model.load_state_dict(self.state["state_dict"])
        self.model = self.model.to(device)
        self.model.eval()
        self.graph_converter = CrystalGraphConverter(
            radial_cutoff=self.model_config.radial_cutoff,
            angle_cutoff=self.model_config.angle_cutoff,
        )

    def predict_properties(self, batch):
        return predict_energy_forces_stress(
            self.model,
            batch,
            create_graph=False,
            compute_stress=True,
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

        graph = self.graph_converter.convert_ase_atoms(atoms)
        batch = CrystalGraphBatch.from_graphs([graph]).to(self.device)
        energy, forces, stress = self.predict_properties(batch)

        self.results.update(
            energy=float(energy.detach().cpu().item()),
            free_energy=float(energy.detach().cpu().item()),
            forces=forces.detach().cpu().numpy(),
            stress=stress_gpa_to_ase_voigt(stress[0].detach().cpu().numpy()),
        )
