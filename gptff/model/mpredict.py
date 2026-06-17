from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
import torch
from typing import Optional

from gptff.graph import CrystalGraphBatch, CrystalGraphConverter
from gptff.model import GPTFFNet, tModLodaer_t
from gptff.model.prediction import predict_energy_forces_stress


class CFG:
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)

class ASECalculator(Calculator):

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(self, model_path, device='cuda', **kwargs):
        super().__init__(**kwargs)

        self.state = torch.load(model_path, map_location=torch.device(device))

        cfg = CFG(self.state['cfg'])
        cfg.device = device
        self.cfg = cfg
        if self.state['cfg']['transformer_activate']:
            self.model = tModLodaer_t(cfg)
        else:
            self.model = GPTFFNet(cfg)
        self.device = device
        self.model.load_state_dict(self.state['state_dict'])
        self.model = self.model.to(device)
        self.graph_converter = CrystalGraphConverter(
            r_cut=getattr(cfg, "radial_cutoff", 5.0),
            a_cut=getattr(cfg, "angle_cutoff", 3.5),
        )

    def get_efs(self, batch):
        return predict_energy_forces_stress(
            self.model,
            batch,
            unit_trans=160.21766208,
            create_graph=False,
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
            stress=stress[0].detach().cpu().numpy() 
        )
