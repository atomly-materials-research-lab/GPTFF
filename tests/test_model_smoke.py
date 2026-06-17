from types import SimpleNamespace

import torch
from pymatgen.core import Lattice, Structure

from gptff.graph import CrystalGraphBatch, CrystalGraphConverter
from gptff.model.model import tModLodaer
from gptff.model.prediction import predict_energy_forces_stress


def test_model_forward_and_efs_with_smooth_radial_basis():
    cfg = SimpleNamespace(
        node_feature_len=8,
        edge_feature_len=8,
        n_layers=1,
        num_radial=8,
        radial_cutoff=3.0,
        angle_cutoff=3.0,
        cutoff_coeff=5,
        device="cpu",
    )
    structure = Structure(
        Lattice.cubic(3.0),
        ["Na", "Cl"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    graph = CrystalGraphConverter(r_cut=3.0, a_cut=3.0).convert(structure)
    batch = CrystalGraphBatch.from_graphs([graph], ref_energies=[0.0])

    energy, forces, stress = predict_energy_forces_stress(
        tModLodaer(cfg),
        batch,
        create_graph=True,
    )

    assert energy.shape == (1,)
    assert forces.shape == (2, 3)
    assert stress.shape == (1, 3, 3)
    assert torch.isfinite(energy).all()
    assert torch.isfinite(forces).all()
    assert torch.isfinite(stress).all()
