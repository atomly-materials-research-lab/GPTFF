import numpy as np
import pandas as pd
from pymatgen.core import Lattice, Structure

from gptff.utils_.data import StructureDataset
from gptff.utils_.labels import LabelConfig


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
