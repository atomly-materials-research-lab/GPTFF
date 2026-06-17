from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from pymatgen.core import Lattice, Structure

from gptff.data import apply_fitted_element_refs, build_datasets
from gptff.graph import CrystalGraphBatch, CrystalGraphConverter
from gptff.model import GPTFFNetConfig
from gptff.trainer.trainer import (
    TrainingConfig,
    compute_batch_loss,
    load_training_checkpoint,
    resolve_checkpoint_path,
    save_checkpoint,
)


def test_trainer_config_parses_legacy_json_keys_without_side_effects():
    config = TrainingConfig.from_dict(_raw_config())

    assert config.lr == 1e-3
    assert config.num_workers == 0
    assert config.num_train_steps == config.epochs
    assert config.energy_unit == "ev"
    assert config.force_unit == "ev_per_ang"
    assert config.stress_unit == "kbar"
    assert config.stress_sign == pytest.approx(-1.0)
    assert config.cache_graphs is True
    assert config.graph_cache_size == 16
    assert config.element_refs == "atomly"
    assert config.n_readout_layers == 4
    assert config.readout_zero_init is True
    assert config.final_atom_norm is True
    assert config.interaction_dropout == pytest.approx(0.1)
    assert config.residual_scale == pytest.approx(0.5)
    assert config.residual_zero_init is True
    assert config.aggregation_norm == "sqrt"
    assert config.checkpoint_path is None
    assert config.checkpoint_dict()["data_file"] == "data.csv"


def test_training_config_builds_label_config_from_data_fields():
    config = TrainingConfig.from_dict(_raw_config())

    label_config = config.to_label_config()

    assert label_config.energy_unit == "ev"
    assert label_config.force_unit == "ev_per_ang"
    assert label_config.stress_unit == "kbar"
    assert label_config.stress_sign == pytest.approx(-1.0)


def test_training_config_disables_graph_cache_by_default():
    raw_config = _raw_config()
    raw_config["data"].pop("cache_graphs")
    raw_config["data"].pop("graph_cache_size")

    config = TrainingConfig.from_dict(raw_config)

    assert config.cache_graphs is False
    assert config.graph_cache_size is None


def test_build_datasets_passes_graph_cache_config(monkeypatch):
    config = TrainingConfig.from_dict(_raw_config())
    monkeypatch.setattr("gptff.data.loaders.read_data", lambda _: _training_df())

    train_dataset, val_dataset = build_datasets(config)

    assert train_dataset.cache_graphs is True
    assert train_dataset.cache_size == 16
    assert val_dataset.cache_graphs is True
    assert val_dataset.cache_size == 16


def test_build_datasets_validates_required_labels_before_training(monkeypatch):
    config = TrainingConfig.from_dict(_raw_config())
    df = _training_df().drop(columns=["forces"])
    monkeypatch.setattr("gptff.data.loaders.read_data", lambda _: df)

    with pytest.raises(ValueError, match="forces"):
        build_datasets(config)


def test_training_config_builds_model_config_only_from_model_fields():
    config = TrainingConfig.from_dict(_raw_config())

    model_config = config.to_model_config()

    assert isinstance(model_config, GPTFFNetConfig)
    assert model_config.n_readout_layers == 4
    assert model_config.readout_zero_init is True
    assert model_config.final_atom_norm is True
    assert model_config.interaction_dropout == pytest.approx(0.1)
    assert model_config.residual_scale == pytest.approx(0.5)
    assert model_config.residual_zero_init is True
    assert model_config.aggregation_norm == "sqrt"
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
    assert state["model_config"]["final_atom_norm"] is True
    assert state["model_config"]["interaction_dropout"] == pytest.approx(0.1)
    assert state["model_config"]["residual_scale"] == pytest.approx(0.5)
    assert state["model_config"]["residual_zero_init"] is True
    assert state["model_config"]["aggregation_norm"] == "sqrt"
    assert state["training_config"]["batch_size"] == 4
    assert state["training_config"]["cache_graphs"] is True
    assert state["training_config"]["graph_cache_size"] == 16
    assert state["label_config"]["stress_unit"] == "kbar"
    assert "device" not in state["model_config"]
    assert state["cfg"]["batch_size"] == 4
    assert (tmp_path / "best_checkpoint.pth").exists()


def test_resolve_checkpoint_path_uses_default_current_checkpoint(tmp_path):
    raw_config = _raw_config()
    raw_config["training"]["output_dir"] = str(tmp_path)
    config = TrainingConfig.from_dict(raw_config)

    assert resolve_checkpoint_path(config, tmp_path) == tmp_path / "curr_checkpoint.pth"


def test_resolve_checkpoint_path_respects_explicit_path(tmp_path):
    raw_config = _raw_config()
    checkpoint_path = tmp_path / "custom.pth"
    raw_config["training"]["checkpoint_path"] = str(checkpoint_path)
    config = TrainingConfig.from_dict(raw_config)

    assert resolve_checkpoint_path(config, tmp_path) == checkpoint_path


def test_load_training_checkpoint_restores_model_and_optimizer(tmp_path):
    config = TrainingConfig.from_dict(_raw_config())
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    loss = model(torch.ones(1, 1)).sum()
    loss.backward()
    optimizer.step()
    expected_weight = model.weight.detach().clone()

    save_checkpoint(
        tmp_path,
        model,
        optimizer,
        config,
        epoch=3,
        best_mae_error=0.2,
        is_best=True,
    )

    restored_model = torch.nn.Linear(1, 1)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=0.1)
    checkpoint = load_training_checkpoint(
        tmp_path / "curr_checkpoint.pth",
        restored_model,
        restored_optimizer,
        device="cpu",
    )

    assert checkpoint.epoch == 3
    assert checkpoint.best_mae_error == pytest.approx(0.2)
    assert checkpoint.training_config["batch_size"] == 4
    assert checkpoint.training_config["cache_graphs"] is True
    assert checkpoint.training_config["graph_cache_size"] == 16
    assert checkpoint.model_config["n_readout_layers"] == 4
    assert checkpoint.label_config["stress_unit"] == "kbar"
    assert checkpoint.model_config["aggregation_norm"] == "sqrt"
    assert torch.allclose(restored_model.weight, expected_weight)
    assert restored_optimizer.state_dict()["state"]


def test_load_training_checkpoint_rejects_missing_file(tmp_path):
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)

    with pytest.raises(FileNotFoundError, match="Checkpoint file not found"):
        load_training_checkpoint(
            tmp_path / "missing.pth",
            model,
            optimizer,
            device="cpu",
        )


def test_compute_batch_loss_requires_energy_and_force_batches():
    config = TrainingConfig.from_dict(_raw_config())
    config.w3 = 0.0

    batch_loss = compute_batch_loss(
        _ConstantEnergyModel(),
        _energy_force_batch(),
        torch.nn.HuberLoss(),
        config,
        create_graph=False,
    )

    assert torch.isfinite(batch_loss.loss)
    assert batch_loss.energy_mae is not None
    assert batch_loss.force_mae is not None
    assert batch_loss.stress_mae is None
    assert batch_loss.force_count == 3
    assert batch_loss.stress_count == 0


def test_compute_batch_loss_requires_force_labels():
    config = TrainingConfig.from_dict(_raw_config())
    config.w3 = 0.0

    with pytest.raises(ValueError, match="force labels are required"):
        compute_batch_loss(
            _ConstantEnergyModel(),
            _energy_only_batch(),
            torch.nn.HuberLoss(),
            config,
            create_graph=False,
        )


def test_compute_batch_loss_requires_positive_energy_and_force_weights():
    config = TrainingConfig.from_dict(_raw_config())
    config.w1 = 0.0
    config.w3 = 0.0

    with pytest.raises(ValueError, match="weight_energy must be positive"):
        compute_batch_loss(
            _ConstantEnergyModel(),
            _energy_force_batch(),
            torch.nn.HuberLoss(),
            config,
            create_graph=False,
        )

    config.w1 = 1.0
    config.w2 = 0.0
    with pytest.raises(ValueError, match="weight_force must be positive"):
        compute_batch_loss(
            _ConstantEnergyModel(),
            _energy_force_batch(),
            torch.nn.HuberLoss(),
            config,
            create_graph=False,
        )


def test_compute_batch_loss_requires_stress_labels_when_enabled():
    config = TrainingConfig.from_dict(_raw_config())
    config.w3 = 1.0

    with pytest.raises(ValueError, match="stress labels are required"):
        compute_batch_loss(
            _ConstantEnergyModel(),
            _energy_force_batch(),
            torch.nn.HuberLoss(),
            config,
            create_graph=False,
        )


class _ConstantEnergyModel(torch.nn.Module):
    def forward(self, graph):
        return torch.zeros(
            (graph.num_atoms.shape[0], 1),
            dtype=graph.positions.dtype,
            device=graph.positions.device,
        )


def _energy_only_batch():
    structure = Structure(Lattice.cubic(3.0), ["Na"], [[0.0, 0.0, 0.0]])
    graph = CrystalGraphConverter(r_cut=2.0, a_cut=2.0).convert(structure)
    return CrystalGraphBatch.from_graphs([graph], energies=[-1.0])


def _energy_force_batch():
    structure = Structure(Lattice.cubic(3.0), ["Na"], [[0.0, 0.0, 0.0]])
    graph = CrystalGraphConverter(r_cut=2.0, a_cut=2.0).convert(structure)
    return CrystalGraphBatch.from_graphs(
        [graph],
        energies=[-1.0],
        forces=[np.zeros((1, 3), dtype=np.float32)],
    )


def _training_df():
    structure = Structure(Lattice.cubic(3.0), ["Na"], [[0.0, 0.0, 0.0]])
    row = {
        "structure": repr(structure.as_dict()),
        "energy": -1.0,
        "forces": repr([[0.0, 0.0, 0.0]]),
        "stress": repr(np.eye(3).tolist()),
        "fold": 0,
    }
    train_row = dict(row)
    train_row["fold"] = 1
    return pd.DataFrame([row, train_row])


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
            "final_atom_norm": True,
            "interaction_dropout": 0.1,
            "residual_scale": 0.5,
            "residual_zero_init": True,
            "aggregation_norm": "sqrt",
            "n_layers": 1,
            "warmup_steps": 0,
            "device": "cpu",
            "val_fold": 0,
            "resume": False,
            "checkpoint_path": None,
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
            "cache_graphs": True,
            "graph_cache_size": 16,
        },
    }
