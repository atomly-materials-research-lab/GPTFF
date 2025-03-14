import numpy as np
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Structure
from gptff.model import model 
import ast
import pandas as pd
import json
import torch
from typing import Optional, Union
from ase.optimize.bfgs import BFGS
from ase.optimize.bfgslinesearch import BFGSLineSearch
from ase.optimize.fire import FIRE
from ase.optimize.lbfgs import LBFGS, LBFGSLineSearch
from ase.optimize.mdmin import MDMin
from ase.optimize.optimize import Optimizer
from ase.optimize.sciopt import SciPyFminBFGS, SciPyFminCG
from ase.constraints import ExpCellFilter, StrainFilter
from ase.phonons import Phonons
from tqdm import tqdm
# import matplotlib.pyplot as plt
from pymatgen.core.composition import Composition
from scipy import interpolate
from gptff.utils_.compute_tp import compute_tp_cc
from gptff.utils_.compute_nb import find_neighbors


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

class custom_graph(object):
    def __init__(self, pbc=[1, 1, 1], r_cut=5.0, a_cut=3.5, atom_refs=atom_refs):
        """
        r_cut: cutoff for bonds
        a_cut: cutoff for angles
        """

        self.r_cut = r_cut
        self.a_cut = a_cut
        self.pbc = pbc
        self.adp = AseAtomsAdaptor()
        self.atom_refs = atom_refs
        
    def transform(self, crystal):
        struc = self.adp.get_structure(crystal)
        i, j, offsets, d_ij = find_neighbors(np.array(struc.cart_coords), np.array(struc.lattice.matrix), self.r_cut, np.array(self.pbc, dtype=np.int32))
        nbr_atoms = np.array([i, j], dtype=np.int32).T
        
        n_bond_pairs_atom, n_bond_pairs_bond, bond_pairs_indices = compute_tp_cc(nbr_atoms, d_ij, self.a_cut, len(struc))
        
        n_bond_pairs_struc = np.array([np.sum(n_bond_pairs_atom)], dtype=np.int32)

        atom_fea = np.vstack([struc[i].specie.number
                              for i in range(len(struc))])
        
        ref_energy = np.sum(self.atom_refs[atom_fea])
        
        return atom_fea, np.array(struc.cart_coords), d_ij, offsets, np.array(struc.lattice.matrix), nbr_atoms, bond_pairs_indices, n_bond_pairs_atom, n_bond_pairs_bond, n_bond_pairs_struc, ref_energy

def collate_fn(data):
    """
    Collate for batch
    """
    batch_atom_fea, batch_coords, batch_d_ij, batch_offset, batch_lattices, batch_nbr_atoms, batch_bond_pairs_indices, batch_n_bond_pairs_atom, batch_n_bond_pairs_bond, batch_n_bond_pairs_struc = [], [], [], [], [], [], [], [], [], []

    n_atoms, pairs_count = [], []

    batch_ref_energy = []

    atom_fea, coords, d_ij, offset, lattices, nbr_atoms, bond_pairs_indices, n_bond_pairs_atom, n_bond_pairs_bond, n_bond_pairs_struc, ref_energy = data

    batch_atom_fea.append(atom_fea)
    batch_coords.append(coords)
    batch_d_ij.append(d_ij)
    batch_offset.append(offset)
    batch_lattices.append(lattices)
    batch_nbr_atoms.append(nbr_atoms)
    batch_bond_pairs_indices.append(bond_pairs_indices)

    batch_n_bond_pairs_atom.append(n_bond_pairs_atom)
    batch_n_bond_pairs_bond.append(n_bond_pairs_bond)
    batch_n_bond_pairs_struc.append(n_bond_pairs_struc)

    n_atoms.append(coords.shape[0])
    pairs_count.append(d_ij.shape[0])

    batch_ref_energy.append(ref_energy)

    batch_ref_energy = np.stack(batch_ref_energy)
    n_atoms = np.array(n_atoms)
    pairs_count = np.array(pairs_count)

    n_atom_cumsum = np.cumsum(np.concatenate([[0], n_atoms[:-1]]))
    nbr_atoms = np.concatenate(batch_nbr_atoms, axis=0)

    nbr_atoms += np.repeat(n_atom_cumsum, pairs_count)[:, None]

    bond_pairs_indices = np.concatenate(batch_bond_pairs_indices)
    pairs_cumsum = np.cumsum(np.concatenate([[0], pairs_count[:-1]]))
    n_bond_pairs_struc = np.concatenate(batch_n_bond_pairs_struc)
    n_bond_pairs_struc_temp = np.array([i for i in n_bond_pairs_struc])
    bond_pairs_indices += np.repeat(pairs_cumsum, n_bond_pairs_struc_temp)[:, None]
    
    n_bond_pairs_atom = np.concatenate(batch_n_bond_pairs_atom)
    n_bond_pairs_bond = np.concatenate(batch_n_bond_pairs_bond)

    atom_fea = np.concatenate(batch_atom_fea)
    coords = np.concatenate(batch_coords)
    d_ij = np.concatenate(batch_d_ij)
    offset = np.concatenate(batch_offset)
    lattices = np.stack(batch_lattices)


    return torch.tensor(atom_fea, dtype=torch.long), \
           torch.tensor(coords, dtype=torch.float32), \
           torch.tensor(d_ij, dtype=torch.float32), \
           torch.tensor(offset, dtype=torch.float32), \
           torch.tensor(lattices, dtype=torch.float32), \
           torch.tensor(n_atoms, dtype=torch.long), \
           torch.tensor(pairs_count, dtype=torch.long), \
           torch.tensor(nbr_atoms, dtype=torch.long), \
           torch.tensor(bond_pairs_indices, dtype=torch.long), \
           torch.tensor(n_bond_pairs_struc, dtype=torch.long), \
           torch.tensor(n_bond_pairs_atom, dtype=torch.long), \
           torch.tensor(n_bond_pairs_bond, dtype=torch.long), \
           torch.tensor(batch_ref_energy, dtype=torch.float32)


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
        self.graph = custom_graph()

    def get_efs(self, data):
        
        atom_fea, coords, d_ij, offsets, lattice, n_atoms, pairs_count, nbr_atoms, bond_pairs_indices, n_bond_pairs_struc, n_bond_pairs_atom, n_bond_pairs_bond, ref_energy = data
        
        coords = coords.requires_grad_(True)
        strain = torch.zeros_like(lattice, dtype=torch.float32).requires_grad_(True)
    
        lattices = torch.matmul(lattice, torch.eye(3, dtype=torch.float32)[None, :, :].to(self.device) + strain)

        volumes = torch.linalg.det(lattices)
        strains = torch.repeat_interleave(strain, n_atoms, dim=0)
        coords = torch.matmul(coords.unsqueeze(1), torch.eye(3, dtype=torch.float32)[None, :, :].to(self.device) + strains).squeeze()

        lattices = lattices[torch.repeat_interleave(torch.arange(pairs_count.shape[0]).to(self.device), pairs_count)]
        offset_dist = torch.matmul(offsets[:, None, :], lattices).squeeze()

        vec_diff_ij = (
                coords[nbr_atoms[:, 1], :]
                + offset_dist
                - coords[nbr_atoms[:, 0], :]
            )

        pair_vec_ij = vec_diff_ij
        pair_dist_ij = torch.sqrt(torch.matmul(pair_vec_ij[:, None, :], pair_vec_ij[:, :, None])).squeeze()
        triple_vec_ij = pair_vec_ij[bond_pairs_indices[:, 0]].squeeze()
        triple_vec_ik = pair_vec_ij[bond_pairs_indices[:, 1]].squeeze()
        triple_dist_ij = pair_dist_ij[bond_pairs_indices[:, 0]].squeeze()
        triple_dist_ik = pair_dist_ij[bond_pairs_indices[:, 1]].squeeze()
        triple_a_jik = torch.matmul(triple_vec_ij[:, None, :], triple_vec_ik[:, :, None]).squeeze(-1) / (triple_dist_ij[:, None] * triple_dist_ik[:, None])
        triple_a_jik = torch.clamp(triple_a_jik, -1.0, 1.0) * (1 - 1e-6)
        triple_a_jik = triple_a_jik.squeeze()
                
        ener_pred = self.model(atom_fea, pair_dist_ij, n_atoms, triple_dist_ij, triple_dist_ik, triple_a_jik, nbr_atoms, n_bond_pairs_bond, bond_pairs_indices)

        ener_pred = ener_pred.squeeze() + ref_energy
        force_pred, stress_pred = torch.autograd.grad(ener_pred, [coords, strain], torch.ones_like(ener_pred), retain_graph=True, create_graph=True)
        force_pred = -1.0 * force_pred
        stress_pred = 1. / volumes[:, None, None] * stress_pred * 160.21766208
        return ener_pred, force_pred, stress_pred

    def calculate(
        self,
        atoms: Optional[Atoms] = None,
        properties: Optional[list] = None,
        system_changes: Optional[list] = None,
    ):    

        properties = properties or ["energy", "forces"]
        system_changes = system_changes or all_changes
        super().calculate(atoms=atoms, properties=properties, system_changes=system_changes)

        data = self.graph.transform(atoms)
        data = collate_fn(data)
        data = [x.to(self.device) for x in data]
        ener, force, stress = self.get_efs(data)

        self.results.update(
            energy=ener.detach().cpu().numpy().ravel().item(),
            free_energy=ener.detach().cpu().numpy().ravel(),
            forces=force.detach().cpu().numpy(),
            stress=stress[0].detach().cpu().numpy() 
        )
