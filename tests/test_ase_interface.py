import numpy as np
import pytest
import torch
from ase import Atoms

from gptff.interfaces import ASECalculator
from gptff.model import GPTFF, GPTFFConfig
from gptff.pretrained import (
    DEFAULT_MODEL_NAME,
    get_model_spec,
    model_checksum,
    path_checksum,
)
from gptff.runtime import GPTFFPotential


def test_ase_calculator_uses_packaged_default_model():
    calc = ASECalculator(device="cpu")

    assert calc.potential.model_name == DEFAULT_MODEL_NAME
    assert calc.model_path is None
    assert calc.model.training is False


def test_packaged_default_model_matches_pretrained_directory_copy():
    checksum = model_checksum()
    assert checksum == get_model_spec().sha256
    assert checksum == path_checksum(
        "pretrained/MatPES-PBE-2025.2/GPTFF-MatPES_PBE_2025.2.pt"
    )


def test_packaged_default_checkpoint_loads_on_cpu():
    potential = GPTFFPotential.from_pretrained(device="cpu")

    assert all(tensor.device.type == "cpu" for tensor in potential.model.state_dict().values())


def test_default_potential_keeps_checkpoint_metadata_light():
    calc = ASECalculator(device="cpu")

    assert "state_dict" not in calc.potential.checkpoint_metadata
    assert not any(
        isinstance(value, torch.Tensor)
        for value in calc.potential.checkpoint_metadata.values()
    )


def test_custom_model_path_is_not_labeled_as_default(tmp_path):
    checkpoint_path = _write_test_checkpoint(tmp_path)

    calc = ASECalculator(model_path=checkpoint_path, device="cpu")

    assert calc.potential.model_name is None
    assert calc.model_path == checkpoint_path


def test_ase_calculator_rejects_potential_with_device_override(tmp_path):
    checkpoint_path = _write_test_checkpoint(tmp_path)
    potential = GPTFFPotential.from_pretrained(model_path=checkpoint_path, device="cpu")

    with pytest.raises(ValueError, match="Pass either a potential"):
        ASECalculator(potential=potential, device="cpu")


def test_ase_calculator_returns_stress_in_ase_voigt_units(tmp_path):
    checkpoint_path = _write_test_checkpoint(tmp_path)
    calc = ASECalculator(model_path=checkpoint_path, device="cpu")

    def fake_predict_properties(batch):
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

    calc.predict_properties = fake_predict_properties
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
        np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    )


def _write_test_checkpoint(tmp_path):
    cfg = GPTFFConfig(
        atom_feature_dim=8,
        edge_feature_dim=8,
        num_interaction_blocks=1,
        num_radial=8,
        num_angular=4,
        radial_cutoff=3.0,
        angle_cutoff=3.0,
        cutoff_coeff=5,
    )
    model = GPTFF(cfg)
    checkpoint_path = tmp_path / "checkpoint.pth"
    torch.save(
        {
            "epoch": 1,
            "state_dict": model.state_dict(),
            "best_validation_metric": 0.0,
            "optimizer": {},
            "training_config": {},
            "model_name": "GPTFF",
            "model_config": cfg.to_dict(),
        },
        checkpoint_path,
    )
    return checkpoint_path
