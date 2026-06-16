import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from gptff.model import model 
import torch
from typing import Optional

from gptff.graph import CrystalGraphBatch, CrystalGraphConverter
from gptff.model.prediction import predict_energy_forces_stress


class CFG:
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)

atom_refs = np.array([ 0.00000000e+00, -3.46535853e+00, -7.56101906e-01, -3.46224791e+00,
       -4.77600176e+00, -8.03619240e+00, -8.40374071e+00, -7.76814618e+00,
       -7.38918302e+00, -4.94725878e+00, -2.92883670e-02, -2.47830716e+00,
       -2.02015956e+00, -5.15479820e+00, -7.91209653e+00, -6.91345095e+00,
       -4.62278149e+00, -3.01552069e+00, -6.27971322e-02, -2.31732442e+00,
       -4.75968073e+00, -8.17421803e+00, -1.14207788e+01, -8.92294483e+00,
       -8.48981509e+00, -8.16635547e+00, -6.58248850e+00, -5.26139665e+00,
       -4.48412068e+00, -3.27367370e+00, -1.34976438e+00, -3.62637456e+00,
       -4.67270042e+00, -4.13166577e+00, -3.67546394e+00, -2.80302539e+00,
        6.47272418e+00, -2.24681188e+00, -4.25110577e+00, -1.02452951e+01,
       -1.16658385e+01, -1.18015760e+01, -8.65537518e+00, -9.36409198e+00,
       -7.57165084e+00, -5.69907599e+00, -4.97159232e+00, -1.88700594e+00,
       -6.79483530e-01, -2.74880153e+00, -3.79441765e+00, -3.38825264e+00,
       -2.55867271e+00, -1.96213610e+00,  9.97909972e+00, -2.55677995e+00,
       -4.88030347e+00, -8.86033743e+00, -9.05368602e+00, -7.94309693e+00,
       -8.12585485e+00, -6.31826210e+00, -8.30242223e+00, -1.22893251e+01,
       -1.73097460e+01, -7.55105974e+00, -8.19580521e+00, -8.34926874e+00,
       -7.25911206e+00, -8.41697224e+00, -3.38725429e+00, -7.68222088e+00,
       -1.26297007e+01, -1.36257602e+01, -9.52985029e+00, -1.18396814e+01,
       -9.79914325e+00, -7.55608603e+00, -5.46902454e+00, -2.65092136e+00,
        4.17472161e-01, -2.32548971e+00, -3.48299933e+00, -3.18067109e+00,
        3.57605604e-15,  9.96350211e-16,  1.18278079e-15, -1.44201673e-15,
       -6.73760309e-18, -5.48347781e+00, -1.03346396e+01, -1.11296117e+01,
       -1.43116273e+01, -1.47003999e+01, -1.54726487e+01])

class ASECalculator(Calculator):

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(self, model_path, device='cuda', **kwargs):
        super().__init__(**kwargs)

        self.state = torch.load(model_path, map_location=torch.device(device))

        cfg = CFG(self.state['cfg'])
        cfg.device = device
        if self.state['cfg']['transformer_activate']:
            self.model = model.tModLodaer_t(cfg)
        else:
            self.model = model.tModLodaer(cfg)
        self.device = device
        self.model.load_state_dict(self.state['state_dict'])
        self.model = self.model.to(device)
        self.graph_converter = CrystalGraphConverter()

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
        ref_energy = float(np.sum(atom_refs[graph.atom_types]))
        batch = CrystalGraphBatch.from_graphs([graph], ref_energies=[ref_energy]).to(self.device)
        ener, force, stress = self.get_efs(batch)

        self.results.update(
            energy=float(ener.detach().cpu().item()),
            free_energy=float(ener.detach().cpu().item()),
            forces=force.detach().cpu().numpy(),
            stress=stress[0].detach().cpu().numpy() 
        )
