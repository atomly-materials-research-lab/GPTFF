from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gptff.model import GPTFFNetConfig
from gptff.trainer.trainer import TrainingConfig, apply_fitted_element_refs, save_checkpoint


def test_trainer_config_parses_legacy_json_keys_without_side_effects():
    config = TrainingConfig.from_dict(_raw_config())

    assert config.lr == 1e-3
    assert config.num_workers == 0
    assert config.num_train_steps == config.epochs
    assert config.energy_unit == "ev"
    assert config.force_unit == "ev_per_ang"
    assert config.stress_unit == "kbar"
    assert config.stress_sign == pytest.approx(-1.0)
    assert config.element_refs == "atomly"
    assert config.n_readout_layers == 4
    assert config.readout_zero_init is True
    assert config.interaction_dropout == pytest.approx(0.1)
    assert config.residual_scale == pytest.approx(0.5)
    assert config.checkpoint_dict()["data_file"] == "data.csv"


def test_training_config_builds_label_config_from_data_fields():
    config = TrainingConfig.from_dict(_raw_config())

    label_config = config.to_label_config()

    assert label_config.energy_unit == "ev"
    assert label_config.force_unit == "ev_per_ang"
    assert label_config.stress_unit == "kbar"
    assert label_config.stress_sign == pytest.approx(-1.0)


def test_training_config_builds_model_config_only_from_model_fields():
    config = TrainingConfig.from_dict(_raw_config())

    model_config = config.to_model_config()

    assert isinstance(model_config, GPTFFNetConfig)
    assert model_config.n_readout_layers == 4
    assert model_config.readout_zero_init is True
    assert model_config.interaction_dropout == pytest.approx(0.1)
    assert model_config.residual_scale == pytest.approx(0.5)
    assert model_config.element_refs == "atomly"
    assert "batch_size" not in model_config.to_dict()
    assert "device" not in model_config.to_dict()


def test_apply_fitted_element_refs_updates_checkpoint_config():
    raw_config = _raw_config()
    raw_config["training"]["element_refs"] = None
    raw_config["training"]["fit_element_refs"] = True
    config = TrainingConfig.from_dict(raw_config)
    samples = [
        _sample([1], -1.0),
        _sample([3], 2.0),
        _sample([1, 1, 3], 0.0),
    ]

    apply_fitted_element_refs(config, samples)

    assert config.element_refs == pytest.approx({"1": -1.0, "3": 2.0})
    assert config.checkpoint_dict()["element_refs"] == pytest.approx({"1": -1.0, "3": 2.0})


def test_apply_fitted_element_refs_rejects_manual_refs_conflict():
    raw_config = _raw_config()
    raw_config["training"]["fit_element_refs"] = True
    config = TrainingConfig.from_dict(raw_config)

    with pytest.raises(ValueError, match="Set either element_refs or fit_element_refs"):
        apply_fitted_element_refs(config, [_sample([1], -1.0)])


def test_save_checkpoint_writes_separate_model_config(tmp_path):
    config = TrainingConfig.from_dict(_raw_config())
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    save_checkpoint(
        tmp_path,
        model,
        optimizer,
        config,
        epoch=1,
        best_mae_error=0.1,
        is_best=True,
    )

    state = torch.load(tmp_path / "curr_checkpoint.pth", map_location="cpu")

    assert state["model_name"] == "GPTFFNet"
    assert state["model_config"]["n_readout_layers"] == 4
    assert state["model_config"]["readout_zero_init"] is True
    assert state["model_config"]["interaction_dropout"] == pytest.approx(0.1)
    assert state["model_config"]["residual_scale"] == pytest.approx(0.5)
    assert "device" not in state["model_config"]
    assert state["cfg"]["batch_size"] == 4
    assert (tmp_path / "best_checkpoint.pth").exists()


def _sample(atom_types, energy):
    return SimpleNamespace(
        graph=SimpleNamespace(atom_types=np.asarray(atom_types, dtype=np.int64)),
        energy=energy,
    )


def _raw_config():
    return {
        "training": {
            "workers": 0,
            "epochs": 2,
            "batch_size": 4,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "node_feature_len": 8,
            "edge_feature_len": 8,
            "num_radial": 8,
            "num_angular": 4,
            "radial_cutoff": 3.0,
            "angle_cutoff": 3.0,
            "cutoff_coeff": 5,
            "element_refs": "atomly",
            "fit_element_refs": False,
            "element_ref_ridge": 0.0,
            "n_readout_layers": 4,
            "readout_zero_init": True,
            "interaction_dropout": 0.1,
            "residual_scale": 0.5,
            "n_layers": 1,
            "warmup_steps": 0,
            "device": "cpu",
            "val_fold": 0,
            "resume": False,
            "transformer_activate": False,
            "start_epoch": 0,
            "weight_energy": 1.0,
            "weight_force": 1.0,
            "weight_stress": 1.0,
        },
        "data": {
            "data_path": ".",
            "data_file": "data.csv",
            "energy_unit": "ev",
            "force_unit": "ev_per_ang",
            "stress_unit": "kbar",
            "stress_sign": -1.0,
        },
    }
