from __future__ import annotations

import ast
import json
from collections import OrderedDict
from typing import Any, Optional

import numpy as np
import pandas as pd
from pymatgen.core.structure import Structure
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
        cache_graphs=False,
        cache_size: Optional[int] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.label_config = label_config or LabelConfig()
        self.cache_graphs = bool(cache_graphs)
        self.cache_size = _normalize_cache_size(cache_size)
        self._sample_cache = OrderedDict()
        self.converter = CrystalGraphConverter(
            r_cut=r_cut,
            a_cut=a_cut,
            numerical_tol=numerical_tol,
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        idx = int(idx)
        if not self.cache_graphs:
            return self._load_sample(idx)

        if idx in self._sample_cache:
            sample = self._sample_cache.pop(idx)
            self._sample_cache[idx] = sample
            return sample

        sample = self._load_sample(idx)
        if self.cache_size != 0:
            self._sample_cache[idx] = sample
            if self.cache_size is not None:
                while len(self._sample_cache) > self.cache_size:
                    self._sample_cache.popitem(last=False)
        return sample

    def _load_sample(self, idx):
        try:
            row = self.df.iloc[idx]
            structure = parse_structure_value(row["structure"])
            graph = self.converter.convert(structure)
            return GraphSample(
                graph=graph,
                energy=parse_energy_label(row, self.label_config),
                forces=parse_forces_label(
                    row,
                    self.label_config,
                    num_atoms=structure.num_sites,
                ),
                stress=parse_stress_label(row, self.label_config),
            )
        except Exception as exc:
            raise ValueError(f"Failed to load structure row {idx}.") from exc


def collate_graph_samples(samples):
    return batch_samples(samples)


def _normalize_cache_size(cache_size: Optional[int]) -> Optional[int]:
    if cache_size is None:
        return None
    cache_size = int(cache_size)
    if cache_size < 0:
        raise ValueError("cache_size must be non-negative or None.")
    return cache_size


def validate_dataframe_schema(
    df,
    label_config: LabelConfig,
    *,
    require_energy: bool = False,
    require_forces: bool = False,
    require_stress: bool = False,
    max_errors: int = 10,
) -> None:
    if "structure" not in df.columns:
        raise ValueError("Dataframe must contain a 'structure' column.")

    required_columns = []
    if require_energy:
        required_columns.append("energy")
    if require_forces:
        required_columns.append("forces")
    if require_stress:
        required_columns.append("stress")

    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        columns = ", ".join(missing_columns)
        raise ValueError(f"Dataframe is missing required label column(s): {columns}.")

    errors = []
    for row_idx, row in df.iterrows():
        try:
            structure = parse_structure_value(row["structure"])
        except Exception as exc:
            errors.append(f"row {row_idx} structure: {exc}")
            if len(errors) >= max_errors:
                break
            continue

        _validate_label(
            errors,
            row_idx,
            "energy",
            lambda: parse_energy_label(row, label_config),
            required=require_energy,
        )
        _validate_label(
            errors,
            row_idx,
            "forces",
            lambda: parse_forces_label(row, label_config, num_atoms=structure.num_sites),
            required=require_forces,
        )
        _validate_label(
            errors,
            row_idx,
            "stress",
            lambda: parse_stress_label(row, label_config),
            required=require_stress,
        )
        if len(errors) >= max_errors:
            break

    if errors:
        details = "\n".join(f"- {error}" for error in errors)
        raise ValueError(f"Dataframe schema validation failed:\n{details}")


def _validate_label(errors, row_idx, column, parser, *, required: bool) -> None:
    try:
        value = parser()
    except Exception as exc:
        errors.append(f"row {row_idx} {column}: {exc}")
        return
    if required and value is None:
        errors.append(f"row {row_idx} {column}: required label is missing.")


def parse_structure_value(value: Any) -> Structure:
    if isinstance(value, Structure):
        return value
    if isinstance(value, dict):
        return Structure.from_dict(value)
    if isinstance(value, str):
        raw = value.strip()
        if raw == "":
            raise ValueError("empty structure value.")
        for loader in (json.loads, ast.literal_eval):
            try:
                parsed = loader(raw)
                break
            except (json.JSONDecodeError, SyntaxError, ValueError):
                parsed = None
        if isinstance(parsed, dict):
            return Structure.from_dict(parsed)
    raise ValueError("structure must be a pymatgen Structure, dict, JSON string, or Python literal dict string.")


def parse_energy_label(row, label_config: LabelConfig):
    if not _has_label_value(row, "energy"):
        return None
    energy = _literal_eval_if_needed(row["energy"])
    if np.asarray(energy).ndim != 0:
        raise ValueError("energy must be a scalar.")
    return convert_energy_to_ev(energy, unit=label_config.energy_unit)


def parse_forces_label(row, label_config: LabelConfig, *, num_atoms: Optional[int] = None):
    if not _has_label_value(row, "forces"):
        return None
    forces = convert_forces_to_ev_per_ang(
        _literal_eval_if_needed(row["forces"]),
        unit=label_config.force_unit,
    )
    expected_shape = (num_atoms, 3) if num_atoms is not None else None
    if expected_shape is not None and forces.shape != expected_shape:
        raise ValueError(f"forces must have shape {expected_shape}, got {forces.shape}.")
    if forces.ndim != 2 or forces.shape[1] != 3:
        raise ValueError(f"forces must have shape (num_atoms, 3), got {forces.shape}.")
    return forces


def parse_stress_label(row, label_config: LabelConfig):
    if not _has_label_value(row, "stress"):
        return None
    stress = convert_stress_to_gpa(
        _literal_eval_if_needed(row["stress"]),
        unit=label_config.stress_unit,
        sign=label_config.stress_sign,
    )
    return _stress_to_matrix(stress)


def _stress_to_matrix(stress: np.ndarray) -> np.ndarray:
    if stress.shape == (3, 3):
        return stress
    if stress.shape == (6,):
        xx, yy, zz, yz, xz, xy = stress
        return np.asarray(
            [
                [xx, xy, xz],
                [xy, yy, yz],
                [xz, yz, zz],
            ],
            dtype=stress.dtype,
        )
    raise ValueError(f"stress must have shape (3, 3) or (6,), got {stress.shape}.")


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
