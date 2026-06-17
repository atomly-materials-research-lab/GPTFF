import numpy as np
import torch
from ase import Atoms

from gptff.model import GPTFFNet, GPTFFNetConfig
from gptff.model.mpredict import ASECalculator
from gptff.utils_.labels import EV_PER_ANG3_TO_GPA


def test_ase_calculator_returns_stress_in_ase_voigt_units(tmp_path):
    cfg = GPTFFNetConfig(
        node_feature_len=8,
        edge_feature_len=8,
        n_layers=1,
        num_radial=8,
        num_angular=4,
        radial_cutoff=3.0,
        angle_cutoff=3.0,
        cutoff_coeff=5,
    )
    model = GPTFFNet(cfg)
    checkpoint_path = tmp_path / "checkpoint.pth"
    torch.save(
        {
            "epoch": 1,
            "state_dict": model.state_dict(),
            "best_mae_error": 0.0,
            "optimizer": {},
            "training_config": {
                "transformer_activate": False,
                "unit_trans": EV_PER_ANG3_TO_GPA,
            },
            "cfg": {
                "transformer_activate": False,
                "unit_trans": EV_PER_ANG3_TO_GPA,
            },
            "model_name": "GPTFFNet",
            "model_config": cfg.to_dict(),
        },
        checkpoint_path,
    )
    calc = ASECalculator(checkpoint_path, device="cpu")

    def fake_get_efs(batch):
        return (
            torch.tensor([1.0]),
            torch.zeros((2, 3)),
            torch.tensor(
                [
                    [
                        [1.0, 6.0, 5.0],
                        [6.0, 2.0, 4.0],
                        [5.0, 4.0, 3.0],
                    ]
                ],
                dtype=torch.float32,
            ),
        )

    calc.get_efs = fake_get_efs
    atoms = Atoms(
        "NaCl",
        positions=[[0, 0, 0], [1.5, 1.5, 1.5]],
        cell=[3, 3, 3],
        pbc=True,
    )
    atoms.calc = calc

    stress = atoms.get_stress()

    assert stress.shape == (6,)
    assert np.allclose(
        stress,
        np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) / EV_PER_ANG3_TO_GPA,
    )
