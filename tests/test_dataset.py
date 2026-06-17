import json

import numpy as np
import pandas as pd
import pytest
from pymatgen.core import Lattice, Structure

from gptff.data import StructureDataset, validate_dataframe_schema
from gptff.graph import CrystalGraphConverter
from gptff.utils.labels import LabelConfig


def test_structure_dataset_converts_stress_with_label_config():
    structure = Structure(Lattice.cubic(3.0), ["Na"], [[0.0, 0.0, 0.0]])
    df = pd.DataFrame([
        {
            "structure": repr(structure.as_dict()),
            "energy": -1.0,
            "forces": repr([[0.0, 0.0, 0.0]]),
            "stress": repr(np.eye(3).tolist()),
        }
    ])
    dataset = StructureDataset(
        df,
        r_cut=2.0,
        a_cut=2.0,
        label_config=LabelConfig(stress_unit="kbar", stress_sign=-1.0),
    )

    sample = dataset[0]

    assert sample.energy == -1.0
    assert sample.forces.shape == (1, 3)
    assert np.allclose(sample.stress, -0.1 * np.eye(3, dtype=np.float32))


def test_structure_dataset_allows_missing_force_and_stress_columns():
    structure = Structure(Lattice.cubic(3.0), ["Na"], [[0.0, 0.0, 0.0]])
    df = pd.DataFrame([
        {
            "structure": repr(structure.as_dict()),
            "energy": -1.0,
        }
    ])
    dataset = StructureDataset(df, r_cut=2.0, a_cut=2.0)

    sample = dataset[0]

    assert sample.energy == -1.0
    assert sample.forces is None
    assert sample.stress is None


def test_structure_dataset_accepts_json_structure_strings():
    structure = Structure(Lattice.cubic(3.0), ["Na"], [[0.0, 0.0, 0.0]])
    df = pd.DataFrame([
        {
            "structure": json.dumps(structure.as_dict()),
            "energy": -1.0,
        }
    ])
    dataset = StructureDataset(df, r_cut=2.0, a_cut=2.0)

    sample = dataset[0]

    assert sample.graph.num_atoms == 1
    assert sample.energy == -1.0


def test_structure_dataset_accepts_structure_dicts():
    structure = Structure(Lattice.cubic(3.0), ["Na"], [[0.0, 0.0, 0.0]])
    df = pd.DataFrame([
        {
            "structure": structure.as_dict(),
            "energy": -1.0,
        }
    ])
    dataset = StructureDataset(df, r_cut=2.0, a_cut=2.0)

    sample = dataset[0]

    assert sample.graph.num_atoms == 1
    assert sample.energy == -1.0


def test_structure_dataset_converts_voigt_stress_to_matrix():
    structure = Structure(Lattice.cubic(3.0), ["Na"], [[0.0, 0.0, 0.0]])
    df = pd.DataFrame([
        {
            "structure": repr(structure.as_dict()),
            "energy": -1.0,
            "stress": repr([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        }
    ])
    dataset = StructureDataset(
        df,
        r_cut=2.0,
        a_cut=2.0,
        label_config=LabelConfig(stress_unit="gpa", stress_sign=1.0),
    )

    sample = dataset[0]

    assert np.allclose(
        sample.stress,
        np.asarray(
            [
                [1.0, 6.0, 5.0],
                [6.0, 2.0, 4.0],
                [5.0, 4.0, 3.0],
            ],
            dtype=np.float32,
        ),
    )


def test_validate_dataframe_schema_requires_enabled_label_columns():
    df = _single_structure_df()

    with pytest.raises(ValueError, match="forces"):
        validate_dataframe_schema(
            df,
            LabelConfig(),
            require_energy=True,
            require_forces=True,
        )


def test_validate_dataframe_schema_rejects_force_shape_mismatch():
    df = _single_structure_df()
    df["forces"] = [repr([[0.0, 0.0]])]

    with pytest.raises(ValueError, match="forces"):
        validate_dataframe_schema(
            df,
            LabelConfig(),
            require_energy=True,
            require_forces=True,
        )


def test_validate_dataframe_schema_rejects_bad_stress_shape():
    df = _single_structure_df()
    df["stress"] = [repr([1.0, 2.0, 3.0])]

    with pytest.raises(ValueError, match="stress"):
        validate_dataframe_schema(
            df,
            LabelConfig(),
            require_energy=True,
            require_stress=True,
        )


def test_structure_dataset_does_not_cache_by_default():
    dataset = StructureDataset(_single_structure_df(), r_cut=2.0, a_cut=2.0)
    converter = _CountingConverter(r_cut=2.0, a_cut=2.0)
    dataset.converter = converter

    dataset[0]
    dataset[0]

    assert converter.calls == 2


def test_structure_dataset_caches_samples_when_enabled():
    dataset = StructureDataset(
        _single_structure_df(),
        r_cut=2.0,
        a_cut=2.0,
        cache_graphs=True,
    )
    converter = _CountingConverter(r_cut=2.0, a_cut=2.0)
    dataset.converter = converter

    first = dataset[0]
    second = dataset[0]

    assert converter.calls == 1
    assert first is second


def test_structure_dataset_lru_cache_size_is_enforced():
    dataset = StructureDataset(
        _two_structure_df(),
        r_cut=2.0,
        a_cut=2.0,
        cache_graphs=True,
        cache_size=1,
    )
    converter = _CountingConverter(r_cut=2.0, a_cut=2.0)
    dataset.converter = converter

    dataset[0]
    dataset[1]
    dataset[0]

    assert converter.calls == 3


def test_structure_dataset_rejects_negative_cache_size():
    try:
        StructureDataset(_single_structure_df(), cache_graphs=True, cache_size=-1)
    except ValueError as exc:
        assert "cache_size" in str(exc)
    else:
        raise AssertionError("Expected ValueError for negative cache_size.")


class _CountingConverter:
    def __init__(self, **kwargs):
        self.converter = CrystalGraphConverter(**kwargs)
        self.calls = 0

    def convert(self, structure):
        self.calls += 1
        return self.converter.convert(structure)


def _single_structure_df():
    structure = Structure(Lattice.cubic(3.0), ["Na"], [[0.0, 0.0, 0.0]])
    return pd.DataFrame([
        {
            "structure": repr(structure.as_dict()),
            "energy": -1.0,
        }
    ])


def _two_structure_df():
    structure_a = Structure(Lattice.cubic(3.0), ["Na"], [[0.0, 0.0, 0.0]])
    structure_b = Structure(Lattice.cubic(3.0), ["Cl"], [[0.0, 0.0, 0.0]])
    return pd.DataFrame([
        {
            "structure": repr(structure_a.as_dict()),
            "energy": -1.0,
        },
        {
            "structure": repr(structure_b.as_dict()),
            "energy": -2.0,
        },
    ])
