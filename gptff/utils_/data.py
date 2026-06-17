from __future__ import division, print_function

import ast
import functools
import math
from typing import Any

import numpy as np
import pandas as pd
import torch
from pymatgen.core.structure import Structure
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import Dataset

from gptff.graph import CrystalGraphConverter, GraphSample, batch_samples
from gptff.utils_.labels import (
    LabelConfig,
    convert_energy_to_ev,
    convert_forces_to_ev_per_ang,
    convert_stress_to_gpa,
)


class StructureDataset(Dataset):
    def __init__(
        self,
        df,
        r_cut=5.0,
        a_cut=3.5,
        numerical_tol=1e-8,
        label_config=None,
    ):
        self.df = df.reset_index(drop=True)
        self.label_config = label_config or LabelConfig()
        self.converter = CrystalGraphConverter(
            r_cut=r_cut,
            a_cut=a_cut,
            numerical_tol=numerical_tol,
        )

    def __len__(self):
        return len(self.df)

    @functools.lru_cache(maxsize=None)
    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]
            structure = Structure.from_dict(ast.literal_eval(row["structure"]))
            graph = self.converter.convert(structure)
            return GraphSample(
                graph=graph,
                energy=_load_energy(row, self.label_config),
                forces=_load_forces(row, self.label_config),
                stress=_load_stress(row, self.label_config),
            )
        except Exception as exc:
            raise ValueError(f"Failed to load structure row {idx}.") from exc


def collate_graph_samples(samples):
    return batch_samples(samples)


def _load_energy(row, label_config: LabelConfig):
    if not _has_label_value(row, "energy"):
        return None
    return convert_energy_to_ev(
        row["energy"],
        unit=label_config.energy_unit,
    )


def _load_forces(row, label_config: LabelConfig):
    if not _has_label_value(row, "forces"):
        return None
    return convert_forces_to_ev_per_ang(
        _literal_eval_if_needed(row["forces"]),
        unit=label_config.force_unit,
    )


def _load_stress(row, label_config: LabelConfig):
    if not _has_label_value(row, "stress"):
        return None
    return convert_stress_to_gpa(
        _literal_eval_if_needed(row["stress"]),
        unit=label_config.stress_unit,
        sign=label_config.stress_sign,
    )


def _has_label_value(row, column: str) -> bool:
    if column not in row.index:
        return False
    value = row[column]
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() not in {"", "none", "null", "nan"}
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return True
    if isinstance(missing, (bool, np.bool_)):
        return not bool(missing)
    return not bool(np.asarray(missing).any())


def _literal_eval_if_needed(value: Any) -> Any:
    if isinstance(value, str):
        return ast.literal_eval(value)
    return value


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
