from __future__ import print_function, division

import functools
import json
import os
import random
import warnings
import ast

import numpy as np
import torch
from pymatgen.core.structure import Structure
from pymatgen.core.periodic_table import Element
from torch.utils.data import Dataset
import pandas as pd
from gptff.utils_.compute_tp import compute_tp_cc
from gptff.utils_.compute_nb import find_neighbors

from torch.optim.lr_scheduler import ReduceLROnPlateau,_LRScheduler
import math


@functools.lru_cache(maxsize=None)
def _atomic_number(symbol):
    return int(Element(symbol).Z)


def _literal(value):
    if isinstance(value, str):
        return ast.literal_eval(value)
    return value


def _array_literal(value, dtype=np.float32):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = ast.literal_eval(value)
    return np.asarray(value, dtype=dtype)


def _site_symbol(site):
    species = site.get("species")
    if isinstance(species, list) and len(species) == 1:
        specie = species[0]
        if isinstance(specie, dict):
            return specie.get("element") or specie.get("label")
    if isinstance(species, dict):
        return species.get("element") or species.get("label")
    return site.get("label")


def structure_arrays(structure_value):
    structure_dict = _literal(structure_value)
    try:
        sites = structure_dict["sites"]
        coords = np.asarray([site["xyz"] for site in sites], dtype=np.float64)
        lattice = np.asarray(structure_dict["lattice"]["matrix"], dtype=np.float64)
        symbols = [_site_symbol(site) for site in sites]
        if any(symbol is None for symbol in symbols):
            raise KeyError("missing site element")
        atom_fea = np.asarray([[_atomic_number(symbol)] for symbol in symbols], dtype=np.int64)
        return atom_fea, coords, lattice
    except Exception:
        structure = Structure.from_dict(structure_dict)
        atom_fea = np.asarray([[site.specie.number] for site in structure], dtype=np.int64)
        return atom_fea, np.asarray(structure.cart_coords, dtype=np.float64), np.asarray(structure.lattice.matrix, dtype=np.float64)


class Mydataset(Dataset):
    def __init__(self, df, pbc=[1, 1, 1], r_cut=5.0, a_cut=3.5):
        """
        r_cut: cutoff for bonds
        a_cut: cutoff for angles
        """

        self.df = df.reset_index(drop=True)
        self.r_cut = r_cut
        self.a_cut = a_cut
        self.pbc = np.asarray(pbc, dtype=np.int32)
        self.structures = self.df["structure"].tolist()
        self.energies = self.df["energy"].to_numpy(dtype=np.float32)
        self.forces = self.df["forces"].tolist()
        self.stresses = self.df["stress"].tolist()
        self.ref_energies = self.df["ref_energy"].to_numpy(dtype=np.float32)

    def __len__(self):
        return len(self.structures)

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        try:
            atom_fea, coords, lattice = structure_arrays(self.structures[idx])
            energy = self.energies[idx]
            forces = _array_literal(self.forces[idx], dtype=np.float32)
            stress = _array_literal(self.stresses[idx], dtype=np.float32) * -0.1
            ref_energy = self.ref_energies[idx]

        except Exception:
            idx = random.randint(0, len(self) - 1)
            return self.__getitem__(idx)
        
        i, j, offset, d_ij = find_neighbors(coords, lattice, self.r_cut, self.pbc)
        nbr_atoms = np.array([i, j], dtype=np.int32).T

        if len(nbr_atoms) == 0:
            # when there is no neighbor pair, keep consistent var names and shapes
            n_bond_pairs_atom = np.zeros(coords.shape[0], dtype=np.int32)
            n_bond_pairs_bond = np.array([], dtype=np.int32)
            bond_pairs_indices = np.array([], dtype=np.int32).reshape(-1, 2)
        else:
            n_bond_pairs_atom, n_bond_pairs_bond, bond_pairs_indices = compute_tp_cc(nbr_atoms, d_ij, self.a_cut, coords.shape[0])
        

        n_bond_pairs_struc = np.array([np.sum(n_bond_pairs_atom)], dtype=np.int32)
        
        return atom_fea, coords, d_ij, offset, lattice, nbr_atoms, bond_pairs_indices, n_bond_pairs_atom, n_bond_pairs_bond, n_bond_pairs_struc, energy, forces, stress, ref_energy
    

def collate_fn(data):
    """
    Collate for batch
    """
    batch_atom_fea, batch_coords, batch_d_ij, batch_offset, batch_lattices, batch_nbr_atoms, batch_bond_pairs_indices, batch_n_bond_pairs_atom, batch_n_bond_pairs_bond, batch_n_bond_pairs_struc = [], [], [], [], [], [], [], [], [], []

    n_atoms, pairs_count = [], []
    batch_energy, batch_forces, batch_stress = [], [], []
    batch_ref_energy = []

    for _, (atom_fea, coords, d_ij, offset, lattices, nbr_atoms, bond_pairs_indices, n_bond_pairs_atom, n_bond_pairs_bond, n_bond_pairs_struc, energy, forces, stress, ref_energy) in enumerate(data):
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

        batch_energy.append(energy)
        batch_forces.append(forces)
        batch_stress.append(stress)

        batch_ref_energy.append(ref_energy)
    
    batch_energy = np.stack(batch_energy)
    batch_forces = np.concatenate(batch_forces)
    batch_stress = np.stack(batch_stress)

    batch_ref_energy = np.stack(batch_ref_energy)
    # n_atoms = np.array([i.shape[0] for i in batch_coords])
    n_atoms = np.array(n_atoms)
    pairs_count = np.array(pairs_count)

    # n_atoms = n_atoms[:-1]
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
           torch.tensor(offset, dtype=torch.float32), \
           torch.tensor(lattices, dtype=torch.float32), \
           torch.tensor(n_atoms, dtype=torch.long), \
           torch.tensor(pairs_count, dtype=torch.long), \
           torch.tensor(nbr_atoms, dtype=torch.long), \
           torch.tensor(bond_pairs_indices, dtype=torch.long), \
           torch.tensor(n_bond_pairs_bond, dtype=torch.long), \
           torch.tensor(batch_energy, dtype=torch.float32), \
           torch.tensor(batch_forces, dtype=torch.float32), \
           torch.tensor(batch_stress, dtype=torch.float32), \
           torch.tensor(batch_ref_energy, dtype=torch.float32)


class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 first_cycle_steps: int,
                 cycle_mult: float = 1.,
                 max_lr: float = 0.1,
                 min_lr: float = 0.001,
                 warmup_steps: int = 0,
                 gamma: float = 1.,
                 last_epoch: int = -1
                 ):
        assert warmup_steps < first_cycle_steps

        self.first_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle_mult = cycle_mult  # cycle steps magnification
        self.base_max_lr = max_lr  # first max learning rate
        self.max_lr = max_lr  # max learning rate in the current cycle
        self.min_lr = min_lr  # min learning rate
        self.warmup_steps = warmup_steps  # warmup step size
        self.gamma = gamma  # decrease rate of max learning rate by cycle

        self.cur_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle = 0  # cycle count
        self.step_in_cycle = last_epoch  # step size of the current cycle

        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)

        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr)
                    * (1 + math.cos(math.pi * (self.step_in_cycle - self.warmup_steps)
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
