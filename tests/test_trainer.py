from types import SimpleNamespace

import numpy as np
import pytest

from gptff.trainer.trainer import TrainingConfig, apply_fitted_element_refs


def test_trainer_config_parses_legacy_json_keys_without_side_effects():
    config = TrainingConfig.from_dict(_raw_config())

    assert config.lr == 1e-3
    assert config.num_workers == 0
    assert config.num_train_steps == config.epochs
    assert config.element_refs == "atomly"
    assert config.checkpoint_dict()["data_file"] == "data.csv"


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
        },
    }
