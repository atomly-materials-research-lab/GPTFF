from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from pymatgen.core import Lattice, Structure

from gptff.data import (
    AtomicDataset,
    AtomicSample,
    apply_fitted_element_refs,
    build_graph_datasets,
)
from gptff.graph import CrystalGraphBatch, CrystalGraphConverter
from gptff.model import GPTFFConfig
from gptff.trainer.logger import (
    CompositeLogger,
    ConsoleLogger,
    CSVLogger,
    EpochLogRecord,
)
from gptff.trainer.loss import BatchLoss
from gptff.trainer.trainer import (
    Trainer,
    TrainingConfig,
    compute_batch_loss,
    has_nonfinite_loss,
    load_config,
    save_checkpoint,
    use_cuda_amp,
)


def test_trainer_config_parses_sections_without_side_effects():
    config = TrainingConfig.from_dict(_raw_config())

    assert config.lr == 1e-3
    assert config.num_workers == 0
    assert config.optimizer_name == "AdamW"
    assert config.scheduler == "CosLR"
    assert config.scheduler_params["decay_fraction"] == pytest.approx(0.01)
    assert config.dataset_path == "dataset.json"
    assert config.validation_fraction == pytest.approx(0.5)
    assert config.test_fraction == pytest.approx(0.0)
    assert config.split_seed == 42
    assert config.group_by_material is False
    assert config.cache_graphs is True
    assert config.graph_cache_size == 16
    assert config.element_references.source == "atomly"
    assert config.element_references.ridge == pytest.approx(0.0)
    assert config.element_refs == "atomly"
    assert config.num_readout_layers == 4
    assert config.readout_atom_norm is True
    assert config.interaction_dropout == pytest.approx(0.1)
    assert config.model.atom_attention.enabled is False
    assert config.amp is False
    assert config.seed == 42
    assert config.deterministic is True
    checkpoint_config = config.checkpoint_dict()
    assert checkpoint_config["data"]["dataset_path"] == "dataset.json"
    assert checkpoint_config["model"]["element_refs"] == "atomly"
    assert checkpoint_config["element_references"]["source"] == "atomly"
    assert checkpoint_config["optimizer"]["learning_rate"] == pytest.approx(1e-3)
    assert checkpoint_config["training"]["batch_size"] == 4
    assert checkpoint_config["loss"]["force_loss_weight"] == pytest.approx(1.0)


def test_load_config_reads_yaml(tmp_path):
    refs_path = tmp_path / "refs.yaml"
    refs_path.write_text(
        """
1: -1.0
3: 2.0
""".strip(),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model:
  atom_feature_dim: 8
  edge_feature_dim: 8
  num_interaction_blocks: 1
  num_radial: 8
  num_angular: 4
  radial_cutoff: 3.0
  angle_cutoff: 3.0
optimizer:
  learning_rate: 0.001
training:
  epochs: 2
  batch_size: 4
  num_workers: 0
  device: cpu
loss:
  energy_loss_weight: 1.0
  force_loss_weight: 1.0
  stress_loss_weight: 0.0
element_references:
  source: refs.yaml
data:
  dataset_path: dataset.json
  validation_fraction: 0.2
  split_seed: 7
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.optimizer.name == "AdamW"
    assert config.optimizer.learning_rate == pytest.approx(1e-3)
    assert config.optimizer.weight_decay == pytest.approx(1e-2)
    assert config.element_refs == pytest.approx({1: -1.0, 3: 2.0})


def test_element_reference_file_requires_atomic_number_keys(tmp_path):
    refs_path = tmp_path / "refs.yaml"
    refs_path.write_text("H: -1.0\n", encoding="utf-8")
    raw_config = _canonical_config()
    raw_config["element_references"] = {"source": str(refs_path)}

    with pytest.raises(ValueError, match="atomic numbers"):
        TrainingConfig.from_dict(raw_config)


def test_training_config_builds_split_config_from_data_fields():
    config = TrainingConfig.from_dict(_raw_config())

    assert config.validation_fraction == pytest.approx(0.5)
    assert config.test_fraction == pytest.approx(0.0)
    assert config.split_seed == 42
    assert config.group_by_material is False


def test_training_config_parses_canonical_sections_without_legacy_keys():
    config = TrainingConfig.from_dict(_canonical_config())

    assert config.data.dataset_path == "dataset.json"
    assert config.model.atom_feature_dim == 8
    assert config.optimizer.name == "Adam"
    assert config.optimizer.learning_rate == pytest.approx(1e-3)
    assert config.optimizer.scheduler == "CosLR"
    assert config.optimizer.scheduler_params["decay_fraction"] == pytest.approx(0.01)
    assert config.training.num_workers == 0
    assert config.loss.stress_loss_weight == pytest.approx(1.0)
    assert "cfg" not in config.checkpoint_dict()
    assert set(config.checkpoint_dict()) == {
        "data",
        "model",
        "optimizer",
        "training",
        "loss",
        "element_references",
    }


def test_training_config_defaults_num_workers_to_four():
    raw_config = _canonical_config()
    raw_config["training"].pop("num_workers")

    config = TrainingConfig.from_dict(raw_config)

    assert config.training.num_workers == 4


def test_training_config_uses_optimizer_specific_weight_decay_defaults():
    raw_config = _canonical_config()
    raw_config["optimizer"].pop("name")
    raw_config["optimizer"].pop("weight_decay")

    config = TrainingConfig.from_dict(raw_config)

    assert config.optimizer.name == "AdamW"
    assert config.optimizer.weight_decay == pytest.approx(1e-2)

    raw_config["optimizer"]["name"] = "Adam"
    config = TrainingConfig.from_dict(raw_config)

    assert config.optimizer.weight_decay == pytest.approx(0.0)


def test_training_config_disables_graph_cache_by_default():
    raw_config = _raw_config()
    raw_config["data"].pop("cache_graphs")
    raw_config["data"].pop("graph_cache_size")

    config = TrainingConfig.from_dict(raw_config)

    assert config.cache_graphs is False
    assert config.graph_cache_size is None


def test_cuda_amp_requires_explicit_amp_flag():
    config = TrainingConfig.from_dict(_raw_config())
    config.device = "cuda"
    config.amp = False

    assert use_cuda_amp(config) is False

    config.amp = True

    assert use_cuda_amp(config) is True


def test_only_nonfinite_losses_are_skipped():
    assert has_nonfinite_loss(_batch_loss(float("nan"))) is True
    assert has_nonfinite_loss(_batch_loss(float("inf"))) is True
    assert has_nonfinite_loss(_batch_loss(1e6)) is False


def test_build_graph_datasets_passes_graph_cache_config():
    config = TrainingConfig.from_dict(_raw_config())

    datasets = build_graph_datasets(_atomic_dataset(), config)

    assert datasets.train.cache_graphs is True
    assert datasets.train.cache_size == 16
    assert datasets.validation.cache_graphs is True
    assert datasets.validation.cache_size == 16


def test_build_graph_datasets_validates_required_stress_before_training():
    config = TrainingConfig.from_dict(_raw_config())
    dataset = AtomicDataset(
        (
            _atomic_sample(0, stress=None),
            _atomic_sample(1, stress=None),
        )
    )

    with pytest.raises(ValueError, match="stress labels are required"):
        build_graph_datasets(dataset, config)


def test_training_config_builds_model_config_only_from_model_fields():
    config = TrainingConfig.from_dict(_raw_config())

    model_config = config.to_model_config()

    assert isinstance(model_config, GPTFFConfig)
    assert model_config.num_readout_layers == 4
    assert model_config.readout_atom_norm is True
    assert model_config.interaction_dropout == pytest.approx(0.1)
    assert model_config.atom_attention.enabled is False
    assert model_config.element_refs == "atomly"
    assert "batch_size" not in model_config.to_dict()
    assert "device" not in model_config.to_dict()
    assert "readout_zero_init" not in model_config.to_dict()
    assert "residual_zero_init" not in model_config.to_dict()


def test_apply_fitted_element_refs_updates_checkpoint_config():
    raw_config = _raw_config()
    raw_config["element_references"] = {"source": "fit", "ridge": 0.0}
    config = TrainingConfig.from_dict(raw_config)
    samples = [
        _sample([1], -1.0),
        _sample([3], 2.0),
        _sample([1, 1, 3], 0.0),
    ]

    apply_fitted_element_refs(config, samples)

    assert config.element_refs == pytest.approx({"1": -1.0, "3": 2.0})
    assert config.checkpoint_dict()["model"]["element_refs"] == pytest.approx({"1": -1.0, "3": 2.0})
    assert config.checkpoint_dict()["element_references"]["source"] == "fit"


def test_trainer_fit_writes_history_and_checkpoints(tmp_path):
    raw_config = _raw_config()
    raw_config["training"]["output_dir"] = str(tmp_path)
    raw_config["training"]["epochs"] = 1
    raw_config["training"]["stress_loss_weight"] = 0.0
    raw_config["data"]["cache_graphs"] = False
    raw_config["data"]["graph_cache_size"] = None
    config = TrainingConfig.from_dict(raw_config)

    best_metric = Trainer(config).fit(_atomic_dataset())

    history = pd.read_csv(tmp_path / "history.csv")
    assert np.isfinite(best_metric)
    assert history.shape[0] == 1
    assert set(
        [
            "epoch",
            "lr",
            "train_loss",
            "val_loss",
            "train_force_mae",
            "val_force_mae",
        ]
    ).issubset(history.columns)
    assert (tmp_path / "last.pt").exists()
    assert (tmp_path / "best.pt").exists()


def test_csv_logger_appends_epoch_records(tmp_path):
    logger = CSVLogger(tmp_path)

    logger.log_epoch(_epoch_record(epoch=1))
    logger.log_epoch(_epoch_record(epoch=2))

    history = pd.read_csv(tmp_path / "history.csv")
    assert history["epoch"].tolist() == [1, 2]
    assert history["val_force_mae"].tolist() == pytest.approx([0.4, 0.4])


def test_console_logger_prints_epoch_summary(capsys):
    logger = ConsoleLogger()

    logger.log_epoch(_epoch_record(epoch=3))

    output = capsys.readouterr().out
    assert "Epoch 3:" in output
    assert "val_MAE(f)=0.40000" in output


def test_composite_logger_dispatches_and_closes():
    first = _MemoryLogger()
    second = _MemoryLogger()
    logger = CompositeLogger([first, second])
    record = _epoch_record(epoch=1)

    logger.log_epoch(record)
    logger.close()

    assert first.records == [record]
    assert second.records == [record]
    assert first.closed is True
    assert second.closed is True


def test_apply_fitted_element_refs_rejects_preloaded_refs_conflict():
    raw_config = _raw_config()
    raw_config["element_references"] = {"source": "fit", "ridge": 0.0}
    config = TrainingConfig.from_dict(raw_config)
    config.element_refs = "atomly"

    with pytest.raises(ValueError, match="source='fit'"):
        apply_fitted_element_refs(config, [_sample([1], -1.0)])


def test_save_checkpoint_writes_separate_model_config(tmp_path):
    config = TrainingConfig.from_dict(_raw_config())
    model = torch.nn.Linear(1, 1)

    save_checkpoint(
        tmp_path,
        model,
        config,
        epoch=1,
        best_validation_metric=0.1,
        is_best=True,
    )

    state = torch.load(tmp_path / "last.pt", map_location="cpu")

    assert state["model_name"] == "GPTFF"
    assert state["model_config"]["num_readout_layers"] == 4
    assert state["model_config"]["readout_atom_norm"] is True
    assert "final_atom_norm" not in state["model_config"]
    assert state["model_config"]["interaction_dropout"] == pytest.approx(0.1)
    assert state["model_config"]["atom_attention"]["enabled"] is False
    assert "readout_zero_init" not in state["model_config"]
    assert "residual_zero_init" not in state["model_config"]
    assert "cfg" not in state
    assert state["training_config"]["training"]["batch_size"] == 4
    assert state["training_config"]["data"]["cache_graphs"] is True
    assert state["training_config"]["data"]["graph_cache_size"] == 16
    assert state["training_config"]["loss"]["stress_loss_weight"] == pytest.approx(1.0)
    assert "label_config" not in state
    assert "optimizer" not in state
    assert "scheduler" not in state
    assert "scaler" not in state
    assert "random_state" not in state
    assert "device" not in state["model_config"]
    assert (tmp_path / "best.pt").exists()


def test_compute_batch_loss_requires_energy_and_force_batches():
    config = TrainingConfig.from_dict(_raw_config())
    config.stress_loss_weight = 0.0

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
    config.stress_loss_weight = 0.0

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
    config.energy_loss_weight = 0.0
    config.stress_loss_weight = 0.0

    with pytest.raises(ValueError, match="energy_loss_weight must be positive"):
        compute_batch_loss(
            _ConstantEnergyModel(),
            _energy_force_batch(),
            torch.nn.HuberLoss(),
            config,
            create_graph=False,
        )

    config.energy_loss_weight = 1.0
    config.force_loss_weight = 0.0
    with pytest.raises(ValueError, match="force_loss_weight must be positive"):
        compute_batch_loss(
            _ConstantEnergyModel(),
            _energy_force_batch(),
            torch.nn.HuberLoss(),
            config,
            create_graph=False,
        )


def test_compute_batch_loss_requires_stress_labels_when_enabled():
    config = TrainingConfig.from_dict(_raw_config())
    config.stress_loss_weight = 1.0

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
    graph = CrystalGraphConverter(radial_cutoff=2.0, angle_cutoff=2.0).convert(structure)
    return CrystalGraphBatch.from_graphs([graph], energies=[-1.0])


def _energy_force_batch():
    structure = Structure(Lattice.cubic(3.0), ["Na"], [[0.0, 0.0, 0.0]])
    graph = CrystalGraphConverter(radial_cutoff=2.0, angle_cutoff=2.0).convert(structure)
    return CrystalGraphBatch.from_graphs(
        [graph],
        energies=[-1.0],
        forces=[np.zeros((1, 3), dtype=np.float32)],
    )


class _MemoryLogger:
    def __init__(self):
        self.records = []
        self.closed = False

    def log_epoch(self, record):
        self.records.append(record)

    def close(self):
        self.closed = True


def _epoch_record(epoch):
    return EpochLogRecord(
        epoch=epoch,
        lr=1e-3,
        train_loss=1.0,
        train_energy_mae=0.1,
        train_force_mae=0.2,
        train_stress_mae=0.3,
        train_skipped_batches=0,
        val_loss=0.9,
        val_energy_mae=0.2,
        val_force_mae=0.4,
        val_stress_mae=0.6,
        val_skipped_batches=0,
    )


def _atomic_dataset():
    return AtomicDataset((_atomic_sample(0), _atomic_sample(1)))


def _atomic_sample(index, *, stress=np.eye(3)):
    structure = Structure(Lattice.cubic(3.0), ["Na"], [[0.0, 0.0, 0.0]])
    return AtomicSample(
        structure=structure,
        energy=-1.0,
        forces=np.zeros((1, 3)),
        stress=stress,
        sample_id=f"sample-{index}",
    )


def _sample(atom_types, energy):
    return SimpleNamespace(
        graph=SimpleNamespace(atom_types=np.asarray(atom_types, dtype=np.int64)),
        energy=energy,
    )


def _batch_loss(value):
    return BatchLoss(
        loss=torch.tensor(value),
        energy_mae=None,
        force_mae=None,
        stress_mae=None,
        batch_size=1,
    )


def _raw_config():
    return {
        "training": {
            "workers": 0,
            "epochs": 2,
            "batch_size": 4,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "atom_feature_dim": 8,
            "edge_feature_dim": 8,
            "num_radial": 8,
            "num_angular": 4,
            "radial_cutoff": 3.0,
            "angle_cutoff": 3.0,
            "cutoff_coeff": 5,
            "num_readout_layers": 4,
            "readout_atom_norm": True,
            "interaction_dropout": 0.1,
            "amp": False,
            "num_interaction_blocks": 1,
            "device": "cpu",
            "energy_loss_weight": 1.0,
            "force_loss_weight": 1.0,
            "stress_loss_weight": 1.0,
        },
        "element_references": {
            "source": "atomly",
            "ridge": 0.0,
        },
        "data": {
            "dataset_path": "dataset.json",
            "validation_fraction": 0.5,
            "test_fraction": 0.0,
            "split_seed": 42,
            "group_by_material": False,
            "cache_graphs": True,
            "graph_cache_size": 16,
        },
    }


def _canonical_config():
    raw = _raw_config()
    training = raw["training"]
    data = dict(raw["data"])
    return {
        "model": {
            "atom_feature_dim": training["atom_feature_dim"],
            "edge_feature_dim": training["edge_feature_dim"],
            "num_interaction_blocks": training["num_interaction_blocks"],
            "num_radial": training["num_radial"],
            "num_angular": training["num_angular"],
            "radial_cutoff": training["radial_cutoff"],
            "angle_cutoff": training["angle_cutoff"],
            "cutoff_coeff": training["cutoff_coeff"],
            "max_atomic_number": 94,
            "num_readout_layers": training["num_readout_layers"],
            "readout_atom_norm": training["readout_atom_norm"],
            "interaction_dropout": training["interaction_dropout"],
            "atom_attention": {
                "enabled": False,
                "num_heads": 4,
                "dropout": 0.0,
                "use_ffn": True,
                "ffn_hidden_dim": None,
            },
        },
        "optimizer": {
            "name": "Adam",
            "learning_rate": training["learning_rate"],
            "weight_decay": 0.0,
            "scheduler": "CosLR",
            "scheduler_params": {
                "decay_fraction": 0.01,
            },
        },
        "training": {
            "epochs": training["epochs"],
            "batch_size": training["batch_size"],
            "num_workers": training["workers"],
            "device": training["device"],
            "amp": training["amp"],
            "output_dir": training.get("output_dir", "."),
        },
        "loss": {
            "energy_loss_weight": training["energy_loss_weight"],
            "force_loss_weight": training["force_loss_weight"],
            "stress_loss_weight": training["stress_loss_weight"],
        },
        "element_references": dict(raw["element_references"]),
        "data": data,
    }
