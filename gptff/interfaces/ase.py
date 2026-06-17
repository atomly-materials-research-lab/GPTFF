from types import SimpleNamespace
from typing import Optional

import torch
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from gptff.graph import CrystalGraphBatch, CrystalGraphConverter
from gptff.model.config import GPTFFNetConfig
from gptff.model.model import GPTFFNet, tModLodaer_t
from gptff.inference import predict_energy_forces_stress
from gptff.utils.labels import stress_gpa_to_ase_voigt


class ASECalculator(Calculator):

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(self, model_path, device="cuda", **kwargs):
        super().__init__(**kwargs)

        self.state = torch.load(model_path, map_location=torch.device(device))

        training_config = dict(self.state.get("training_config", self.state["cfg"]))
        self.unit_trans = float(training_config.get("unit_trans", 160.21766208))
        if training_config["transformer_activate"]:
            cfg = SimpleNamespace(**training_config)
            cfg.device = device
            self.model_config = GPTFFNetConfig.from_dict(training_config)
            self.model = tModLodaer_t(cfg)
        else:
            self.model_config = GPTFFNetConfig.from_dict(self.state["model_config"])
            self.model = GPTFFNet(self.model_config)
        self.device = device
        self.model.load_state_dict(self.state["state_dict"])
        self.model = self.model.to(device)
        self.graph_converter = CrystalGraphConverter(
            r_cut=self.model_config.radial_cutoff,
            a_cut=self.model_config.angle_cutoff,
        )

    def get_efs(self, batch):
        return predict_energy_forces_stress(
            self.model,
            batch,
            unit_trans=self.unit_trans,
            create_graph=False,
            compute_stress=True,
        )

    def calculate(
        self,
        atoms: Optional[Atoms] = None,
        properties: Optional[list] = None,
        system_changes: Optional[list] = None,
    ):

        properties = properties or ["energy", "forces"]
        system_changes = system_changes or all_changes
        super().calculate(atoms=atoms, properties=properties, system_changes=system_changes)

        graph = self.graph_converter.convert_ase_atoms(atoms)
        batch = CrystalGraphBatch.from_graphs([graph]).to(self.device)
        ener, force, stress = self.get_efs(batch)

        self.results.update(
            energy=float(ener.detach().cpu().item()),
            free_energy=float(ener.detach().cpu().item()),
            forces=force.detach().cpu().numpy(),
            stress=stress_gpa_to_ase_voigt(stress[0].detach().cpu().numpy()),
        )
