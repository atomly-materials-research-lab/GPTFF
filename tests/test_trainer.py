import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from pymatgen.core import Lattice, Structure

import gptff.trainer.loss as loss_module
import gptff.trainer.trainer as trainer_module
from gptff.data import (
    AtomicDataset,
    AtomicSample,
    apply_fitted_element_refs,
    build_graph_datasets,
)
from gptff.graph import CrystalGraphBatch, CrystalGraphConverter
from gptff.model import GPTFFConfig
from gptff.model.model import GPTFF
from gptff.trainer import distributed as distributed_module
from gptff.trainer.config import OptimizerConfig, load_config
from gptff.trainer.distributed import (
    DistributedContext,
    barrier,
    cleanup_distributed,
    unwrap_model,
)
from gptff.trainer.evaluation import EvaluationRecord
from gptff.trainer.logger import (
    CompositeLogger,
    ConsoleLogger,
    CSVLogger,
    EpochLogRecord,
    NullLogger,
    WandBLogger,
)
from gptff.trainer.loss import BatchLoss
from gptff.trainer.scheduler import build_lr_scheduler, scheduler_steps_per_epoch
from gptff.trainer.trainer import (
    AverageMeter,
    EpochMetrics,
    Trainer,
    TrainingConfig,
    build_optimizer,
    build_scheduler,
    clip_gradients,
    compute_batch_loss,
    effective_scheduler_steps_per_epoch,
    has_nonfinite_loss,
    save_checkpoint,
    scheduler_step_batches,
    should_skip_optimizer_step,
    sync_epoch_metrics,
    use_cuda_amp,
)
from gptff.utils.labels import EV_PER_ANG3_TO_GPA


def test_trainer_config_parses_sections_without_side_effects():
    config = TrainingConfig.from_dict(_raw_config())

    assert config.optimizer.learning_rate == 1e-3
    assert config.training.num_workers == 0
    assert config.optimizer.name == "AdamW"
    assert config.optimizer.scheduler == "CosLR"
    assert config.optimizer.scheduler_params == {}
    assert config.data.dataset_path == "dataset.json"
    assert config.data.dataset_format == "atomic_json"
    assert config.data.validation_fraction == pytest.approx(0.5)
    assert config.data.test_fraction == pytest.approx(0.0)
    assert config.data.split_seed == 42
    assert config.data.group_by_material is False
    assert config.data.cache_graphs is True
    assert config.data.graph_cache_size == 16
    assert config.element_references.source == {"1": -1.0, "3": 2.0}
    assert config.model.element_refs == {"1": -1.0, "3": 2.0}
    assert config.model.num_readout_layers == 4
    assert config.model.readout_atom_norm is True
    assert config.model.interaction_dropout == pytest.approx(0.1)
    assert config.model.atom_attention.enabled is True
    assert config.model.atom_attention.use_ffn is False
    assert config.model.atom_attention.density_scale_init == pytest.approx(0.1)
    assert config.model.atom_attention.ffn_residual_scale_init == pytest.approx(1e-2)
    assert config.training.amp is False
    assert config.training.seed == 42
    assert config.training.deterministic is True
    assert config.training.distributed == "auto"
    assert config.training.persistent_workers is False
    assert config.logging.wandb.enabled is True
    assert config.logging.wandb.project == "gptff"
    checkpoint_config = config.checkpoint_dict()
    assert checkpoint_config["data"]["dataset_path"] == "dataset.json"
    assert checkpoint_config["data"]["dataset_format"] == "atomic_json"
    assert checkpoint_config["model"]["element_refs"] == {"1": -1.0, "3": 2.0}
    assert checkpoint_config["element_references"]["source"] == {"1": -1.0, "3": 2.0}
    assert checkpoint_config["logging"]["wandb"]["enabled"] is True
    assert checkpoint_config["optimizer"]["learning_rate"] == pytest.approx(1e-3)
    assert checkpoint_config["training"]["batch_size"] == 4
    assert checkpoint_config["training"]["distributed"] == "auto"
    assert checkpoint_config["training"]["persistent_workers"] is False
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
    assert config.model.element_refs == pytest.approx({1: -1.0, 3: 2.0})


def test_element_reference_file_requires_atomic_number_keys(tmp_path):
    refs_path = tmp_path / "refs.yaml"
    refs_path.write_text("H: -1.0\n", encoding="utf-8")
    raw_config = _canonical_config()
    raw_config["element_references"] = {"source": str(refs_path)}

    with pytest.raises(ValueError, match="atomic numbers"):
        TrainingConfig.from_dict(raw_config)


def test_training_config_builds_split_config_from_data_fields():
    raw_config = _raw_config()
    config = TrainingConfig.from_dict(raw_config)

    assert config.data.validation_fraction == pytest.approx(0.5)
    assert config.data.test_fraction == pytest.approx(0.0)
    assert config.data.split_seed == 42
    assert config.data.group_by_material is False


def test_training_config_rejects_removed_max_open_files():
    raw_config = _raw_config()
    raw_config["data"]["max_open_files"] = 32

    with pytest.raises(ValueError, match="max_open_files has been removed"):
        TrainingConfig.from_dict(raw_config)


def test_training_config_parses_sharded_graph_dataset_format():
    raw_config = _raw_config()
    raw_config["data"]["dataset_path"] = "dataset.gptff"
    raw_config["data"]["dataset_format"] = "sharded-hdf5-graph"

    config = TrainingConfig.from_dict(raw_config)

    assert config.data.dataset_path == "dataset.gptff"
    assert config.data.dataset_format == "sharded_hdf5_graph"


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
        "logging",
    }


def test_legacy_training_config_rejects_unknown_keys():
    raw_config = _raw_config()
    raw_config["training"]["num_head"] = 8

    with pytest.raises(ValueError, match="Unknown model config key.*num_head"):
        TrainingConfig.from_dict(raw_config)


def test_training_config_preserves_scheduler_warmup_params():
    raw_config = _canonical_config()
    raw_config["optimizer"]["scheduler_params"] = {
        "decay_fraction": 0.01,
        "warmup_epochs": 3,
        "warmup_start_factor": 0.1,
    }

    config = TrainingConfig.from_dict(raw_config)

    assert config.optimizer.scheduler_params["warmup_epochs"] == 3
    assert config.optimizer.scheduler_params["warmup_start_factor"] == pytest.approx(0.1)
    assert config.checkpoint_dict()["optimizer"]["scheduler_params"][
        "warmup_epochs"
    ] == 3


def test_training_config_defaults_num_workers_to_four():
    raw_config = _canonical_config()
    raw_config["training"].pop("num_workers")

    config = TrainingConfig.from_dict(raw_config)

    assert config.training.num_workers == 4


def test_training_config_parses_persistent_workers():
    raw_config = _canonical_config()
    raw_config["training"]["persistent_workers"] = True

    config = TrainingConfig.from_dict(raw_config)

    assert config.training.persistent_workers is True


def test_coslr_warmup_linearly_reaches_base_lr_before_cosine_decay():
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    scheduler = build_lr_scheduler(
        optimizer,
        scheduler="CosLR",
        learning_rate=1e-3,
        epochs=10,
        scheduler_params={
            "decay_fraction": 0.01,
            "steps_per_epoch": 10,
            "warmup_epochs": 0.3,
            "warmup_start_factor": 0.1,
        },
    )

    assert scheduler is not None
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)

    def step_scheduler() -> None:
        optimizer.step()
        scheduler.step()

    step_scheduler()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(4e-4)

    step_scheduler()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(7e-4)

    step_scheduler()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)

    step_scheduler()
    assert optimizer.param_groups[0]["lr"] < 1e-3


def test_configured_scheduler_steps_per_epoch_controls_train_step_batches():
    raw_config = _canonical_config()
    raw_config["optimizer"]["scheduler_params"] = {
        "decay_fraction": 0.01,
        "steps_per_epoch": 4,
        "warmup_epochs": 1,
    }
    config = TrainingConfig.from_dict(raw_config)

    steps_per_epoch = scheduler_steps_per_epoch(config.optimizer.scheduler_params)

    assert steps_per_epoch == 4
    assert scheduler_step_batches(10, steps_per_epoch=steps_per_epoch) == {3, 5, 8, 10}


def test_small_training_loader_uses_achievable_cosine_schedule_length():
    raw_config = _canonical_config()
    raw_config["training"]["epochs"] = 4
    config = TrainingConfig.from_dict(raw_config)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.optimizer.learning_rate)

    scheduler = build_scheduler(optimizer, config, num_batches=2)
    assert scheduler is not None
    assert effective_scheduler_steps_per_epoch(2, config.optimizer.scheduler_params) == 2

    for _ in range(2 * config.training.epochs):
        optimizer.step()
        scheduler.step()

    assert optimizer.param_groups[0]["lr"] == pytest.approx(
        0.01 * config.optimizer.learning_rate
    )


def test_trainer_small_loader_reaches_minimum_lr_end_to_end(tmp_path, monkeypatch):
    raw_config = _raw_config()
    raw_config["training"]["epochs"] = 4
    raw_config["training"]["batch_size"] = 1
    raw_config["training"]["output_dir"] = str(tmp_path)
    raw_config["training"]["stress_loss_weight"] = 0.0
    raw_config["data"]["validation_fraction"] = 0.25
    raw_config["data"]["cache_graphs"] = False
    raw_config["data"]["graph_cache_size"] = None
    config = TrainingConfig.from_dict(raw_config)

    def fake_compute_batch_loss(model, batch, _criterion, _config, *, create_graph):
        del create_graph
        loss = next(model.parameters()).sum() * 0.0 + 1.0
        zero = loss.detach() * 0.0
        return BatchLoss(
            loss=loss,
            energy_mae=zero,
            force_mae=zero,
            stress_mae=None,
            batch_size=int(batch.num_atoms.shape[0]),
            force_count=int(batch.forces.numel()),
        )

    monkeypatch.setattr(trainer_module, "compute_batch_loss", fake_compute_batch_loss)
    trainer = Trainer(config, logger=NullLogger())
    trainer.setup(_atomic_dataset(size=4))

    assert len(trainer.train_loader) == 3
    for epoch in range(1, config.training.epochs + 1):
        metrics = trainer.train_epoch(epoch)
        assert metrics.skipped_batches == 0

    assert trainer.scheduler.last_epoch == 3 * config.training.epochs
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(
        0.01 * config.optimizer.learning_rate
    )


def test_lr_cycle_epochs_uses_epoch_units_and_stays_at_minimum_lr():
    raw_config = _canonical_config()
    raw_config["training"]["epochs"] = 4
    raw_config["optimizer"].pop("scheduler_params")
    raw_config["optimizer"]["lr_cycle_epochs"] = 1
    config = TrainingConfig.from_dict(raw_config)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.optimizer.learning_rate)

    scheduler = build_lr_scheduler(
        optimizer,
        scheduler=config.optimizer.scheduler,
        learning_rate=config.optimizer.learning_rate,
        epochs=config.training.epochs,
        scheduler_params={**config.optimizer.scheduler_params, "steps_per_epoch": 2},
    )

    assert config.optimizer.scheduler_params == {"lr_cycle_epochs": 1.0}
    for _ in range(2):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(
        0.01 * config.optimizer.learning_rate
    )

    for _ in range(6):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(
        0.01 * config.optimizer.learning_rate
    )


def test_multistep_default_milestones_scale_with_effective_steps_per_epoch():
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    scheduler = build_lr_scheduler(
        optimizer,
        scheduler="multistep",
        learning_rate=1e-3,
        epochs=10,
        scheduler_params={"steps_per_epoch": 2},
    )

    assert scheduler is not None
    assert list(scheduler.milestones) == [8, 12, 16, 18]


def test_coslr_rejects_warmup_that_covers_full_schedule():
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    with pytest.raises(ValueError, match="warmup_steps must be smaller than T_max"):
        build_lr_scheduler(
            optimizer,
            scheduler="CosLR",
            learning_rate=1e-3,
            epochs=1,
            scheduler_params={
                "steps_per_epoch": 10,
                "warmup_epochs": 1,
            },
        )


@pytest.mark.parametrize(
    ("scheduler_name", "scheduler_type"),
    [
        ("exp", torch.optim.lr_scheduler.ExponentialLR),
        ("multistep", torch.optim.lr_scheduler.MultiStepLR),
    ],
)
def test_non_cosine_schedulers_build_with_default_config(
    scheduler_name,
    scheduler_type,
):
    raw_config = _canonical_config()
    raw_config["optimizer"]["scheduler"] = scheduler_name
    raw_config["optimizer"].pop("scheduler_params")
    config = TrainingConfig.from_dict(raw_config)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.optimizer.learning_rate)

    scheduler = build_lr_scheduler(
        optimizer,
        scheduler=config.optimizer.scheduler,
        learning_rate=config.optimizer.learning_rate,
        epochs=config.training.epochs,
        scheduler_params=config.optimizer.scheduler_params,
    )

    assert config.optimizer.scheduler_params == {}
    assert isinstance(scheduler, scheduler_type)


@pytest.mark.parametrize("scheduler_name", ["exp", "multistep"])
def test_non_cosine_schedulers_reject_cosine_minimum_lr_config(scheduler_name):
    raw_config = _canonical_config()
    raw_config["optimizer"]["scheduler"] = scheduler_name
    raw_config["optimizer"].pop("scheduler_params")
    raw_config["optimizer"]["min_learning_rate"] = 1e-5
    config = TrainingConfig.from_dict(raw_config)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.optimizer.learning_rate)

    with pytest.raises(ValueError, match="min_learning_rate.*only supported by cosine"):
        build_lr_scheduler(
            optimizer,
            scheduler=config.optimizer.scheduler,
            learning_rate=config.optimizer.learning_rate,
            epochs=config.training.epochs,
            scheduler_params=config.optimizer.scheduler_params,
        )


@pytest.mark.parametrize("scheduler_name", ["cosrestart", "exp", "multistep"])
def test_scheduler_control_params_are_not_forwarded_to_torch(scheduler_name):
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    scheduler = build_lr_scheduler(
        optimizer,
        scheduler=scheduler_name,
        learning_rate=1e-3,
        epochs=10,
        scheduler_params={"steps_per_epoch": 4},
    )

    assert scheduler is not None


def test_training_config_parses_distributed_modes():
    raw_config = _canonical_config()
    raw_config["training"]["distributed"] = True

    config = TrainingConfig.from_dict(raw_config)

    assert config.training.distributed == "true"

    raw_config["training"]["distributed"] = "false"
    config = TrainingConfig.from_dict(raw_config)

    assert config.training.distributed == "false"

    raw_config["training"]["distributed"] = "bad"
    with pytest.raises(ValueError, match="distributed"):
        TrainingConfig.from_dict(raw_config)


def test_training_config_parses_wandb_logging_config():
    raw_config = _canonical_config()
    raw_config["logging"] = {
        "wandb": {
            "enabled": False,
            "project": "custom-project",
            "entity": "team",
            "name": "debug-run",
            "group": "smoke",
            "tags": ["unit", "test"],
            "notes": "local smoke test",
            "mode": "offline",
            "job_type": "train-test",
            "init_kwargs": {"reinit": True},
        }
    }

    config = TrainingConfig.from_dict(raw_config)

    assert config.logging.wandb.enabled is False
    assert config.logging.wandb.project == "custom-project"
    assert config.logging.wandb.entity == "team"
    assert config.logging.wandb.name == "debug-run"
    assert config.logging.wandb.group == "smoke"
    assert config.logging.wandb.tags == ("unit", "test")
    assert config.logging.wandb.notes == "local smoke test"
    assert config.logging.wandb.mode == "offline"
    assert config.logging.wandb.job_type == "train-test"
    assert config.logging.wandb.init_kwargs == {"reinit": True}


def test_training_config_uses_optimizer_specific_weight_decay_defaults():
    raw_config = _canonical_config()
    raw_config["optimizer"].pop("name")
    raw_config["optimizer"].pop("weight_decay")

    config = TrainingConfig.from_dict(raw_config)

    assert config.optimizer.name == "AdamW"
    assert config.optimizer.weight_decay == pytest.approx(1e-2)
    assert OptimizerConfig(learning_rate=1e-3).weight_decay == pytest.approx(1e-2)

    raw_config["optimizer"]["name"] = "Adam"
    config = TrainingConfig.from_dict(raw_config)

    assert config.optimizer.weight_decay == pytest.approx(0.0)
    assert OptimizerConfig(learning_rate=1e-3, name="Adam").weight_decay == pytest.approx(0.0)


def test_build_optimizer_excludes_scales_norms_and_biases_from_weight_decay():
    raw_config = _raw_config()
    raw_config["training"]["atom_attention"] = {"enabled": True, "use_ffn": True}
    raw_config["optimizer"] = {
        "name": "AdamW",
        "learning_rate": 1e-3,
        "weight_decay": 1e-2,
    }
    config = TrainingConfig.from_dict(raw_config)
    model = GPTFF(config.model)

    optimizer = build_optimizer(model, config)

    assert len(optimizer.param_groups) == 2
    decay_group = next(group for group in optimizer.param_groups if group["weight_decay"] == 1e-2)
    no_decay_group = next(group for group in optimizer.param_groups if group["weight_decay"] == 0.0)
    decay_param_ids = {id(param) for param in decay_group["params"]}
    no_decay_param_ids = {id(param) for param in no_decay_group["params"]}
    named_params = dict(model.named_parameters())
    trainable_param_ids = {id(param) for param in model.parameters() if param.requires_grad}

    assert (
        id(named_params["interactions.0.atom_update.density_context.density_scale"])
        in no_decay_param_ids
    )
    assert id(named_params["interactions.0.atom_ffn_residual_scale"]) in no_decay_param_ids
    assert id(named_params["atom_embedding.embedding.weight"]) in no_decay_param_ids
    assert (
        id(named_params["geometry_embedding.edge_embedding.edge_embedding.0.weight"])
        in no_decay_param_ids
    )
    assert "geometry_embedding.edge_modulation.atom_message_weight.weight" not in named_params
    assert (
        id(named_params["geometry_embedding.edge_modulation.edge_message_weight.weight"])
        in no_decay_param_ids
    )
    assert id(named_params["readout.output_layer.weight"]) in no_decay_param_ids
    assert id(named_params["interactions.0.atom_norm.weight"]) in no_decay_param_ids
    assert id(named_params["interactions.0.edge_update.message_gate.value.bias"]) in no_decay_param_ids
    assert id(named_params["interactions.0.edge_update.message_gate.value.weight"]) in decay_param_ids
    assert id(named_params["interactions.0.three_body.target_encoder.output_layer.weight"]) in (
        decay_param_ids
    )
    assert id(named_params["interactions.0.atom_update.score.output_layer.weight"]) in (
        decay_param_ids
    )
    assert decay_param_ids.isdisjoint(no_decay_param_ids)
    assert decay_param_ids | no_decay_param_ids == trainable_param_ids


def test_training_config_disables_graph_cache_by_default():
    raw_config = _raw_config()
    raw_config["data"].pop("cache_graphs")
    raw_config["data"].pop("graph_cache_size")

    config = TrainingConfig.from_dict(raw_config)

    assert config.data.cache_graphs is False
    assert config.data.graph_cache_size is None


def test_cuda_amp_requires_explicit_amp_flag():
    config = TrainingConfig.from_dict(_raw_config())
    config.training = replace(config.training, device="cuda", amp=False)

    assert use_cuda_amp(config) is False

    config.training = replace(config.training, amp=True)

    assert use_cuda_amp(config) is True


def test_zero_grad_clip_norm_disables_gradient_clipping():
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.grad = torch.full_like(model.weight, 3.0)

    result = clip_gradients(model, max_norm=0.0)

    assert result is None
    assert torch.equal(model.weight.grad, torch.full_like(model.weight, 3.0))


@pytest.mark.parametrize("grad_clip_norm", [-1.0, float("nan"), float("inf")])
def test_training_config_rejects_invalid_grad_clip_norm(grad_clip_norm):
    raw_config = _canonical_config()
    raw_config["training"]["grad_clip_norm"] = grad_clip_norm

    with pytest.raises(ValueError, match="grad_clip_norm must be finite and non-negative"):
        TrainingConfig.from_dict(raw_config)


def test_only_nonfinite_losses_are_skipped():
    assert has_nonfinite_loss(_batch_loss(float("nan"))) is True
    assert has_nonfinite_loss(_batch_loss(float("inf"))) is True
    assert has_nonfinite_loss(_batch_loss(1e6)) is False


def test_should_skip_optimizer_step_is_local_without_ddp():
    context = DistributedContext.disabled(device="cpu")

    assert should_skip_optimizer_step(_batch_loss(float("nan")), context, device="cpu") is True
    assert should_skip_optimizer_step(_batch_loss(1.0), context, device="cpu") is False


def test_should_skip_optimizer_step_propagates_remote_nonfinite_loss(monkeypatch):
    context = DistributedContext(enabled=True, world_size=2, device="cpu")

    def fake_all_reduce_max(tensor, _context):
        tensor.fill_(1)
        return tensor

    monkeypatch.setattr(trainer_module, "all_reduce_max", fake_all_reduce_max)

    assert should_skip_optimizer_step(_batch_loss(1.0), context, device="cpu") is True


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

    model_config = config.model

    assert isinstance(model_config, GPTFFConfig)
    assert model_config.num_readout_layers == 4
    assert model_config.readout_atom_norm is True
    assert model_config.interaction_dropout == pytest.approx(0.1)
    assert model_config.atom_attention.enabled is True
    assert model_config.atom_attention.use_ffn is False
    assert model_config.atom_attention.density_scale_init == pytest.approx(0.1)
    assert model_config.atom_attention.ffn_residual_scale_init == pytest.approx(1e-2)
    assert model_config.element_refs == {"1": -1.0, "3": 2.0}
    assert "batch_size" not in model_config.to_dict()
    assert "device" not in model_config.to_dict()
    assert "readout_zero_init" not in model_config.to_dict()
    assert "residual_zero_init" not in model_config.to_dict()


def test_apply_fitted_element_refs_updates_checkpoint_config():
    raw_config = _raw_config()
    raw_config["element_references"] = {"source": "fit"}
    config = TrainingConfig.from_dict(raw_config)
    samples = [
        _sample([1], -1.0),
        _sample([3], 2.0),
        _sample([1, 1, 3], 0.0),
    ]

    apply_fitted_element_refs(config, samples)

    assert config.model.element_refs == pytest.approx({"1": -1.0, "3": 2.0})
    assert config.checkpoint_dict()["model"]["element_refs"] == pytest.approx({"1": -1.0, "3": 2.0})
    assert config.checkpoint_dict()["element_references"]["source"] == "fit"


def test_trainer_fit_writes_history_checkpoints_and_progress(tmp_path, monkeypatch, capsys):
    raw_config = _raw_config()
    raw_config["training"]["output_dir"] = str(tmp_path)
    raw_config["training"]["epochs"] = 1
    raw_config["training"]["stress_loss_weight"] = 0.0
    raw_config["data"]["cache_graphs"] = False
    raw_config["data"]["graph_cache_size"] = None
    config = TrainingConfig.from_dict(raw_config)
    progress_descriptions = []

    class Progress:
        def __init__(self, iterable):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable)

        def set_postfix(self, **_kwargs):
            return None

    def fake_tqdm(iterable, *, desc, **_kwargs):
        progress_descriptions.append(desc)
        return Progress(iterable)

    monkeypatch.setattr(trainer_module, "tqdm", fake_tqdm)

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
    assert (tmp_path / "bestE.pt").exists()
    assert (tmp_path / "bestF.pt").exists()
    assert progress_descriptions == ["Train 1/1", "Validation 1/1"]
    assert "Dataset samples: total=2, train=1, validation=1, test=0" in capsys.readouterr().out


def test_trainer_fit_writes_final_test_metrics(tmp_path, monkeypatch, capsys):
    raw_config = _raw_config()
    raw_config["training"]["output_dir"] = str(tmp_path)
    raw_config["training"]["epochs"] = 1
    raw_config["training"]["stress_loss_weight"] = 0.0
    raw_config["data"]["validation_fraction"] = 1 / 3
    raw_config["data"]["test_fraction"] = 1 / 3
    raw_config["data"]["cache_graphs"] = False
    raw_config["data"]["graph_cache_size"] = None
    config = TrainingConfig.from_dict(raw_config)
    progress_descriptions = []

    class Progress:
        def __init__(self, iterable):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable)

        def set_postfix(self, **_kwargs):
            return None

    def fake_tqdm(iterable, *, desc, **_kwargs):
        progress_descriptions.append(desc)
        return Progress(iterable)

    monkeypatch.setattr(trainer_module, "tqdm", fake_tqdm)

    Trainer(config).fit(_atomic_dataset(size=3))

    metrics = json.loads((tmp_path / "test_metrics.json").read_text())
    assert metrics["split"] == "test"
    assert metrics["checkpoint"] == "bestF.pt"
    assert metrics["epoch"] == 1
    assert np.isfinite(metrics["loss"])
    assert np.isfinite(metrics["energy_mae"])
    assert np.isfinite(metrics["force_mae"])
    assert metrics["stress_mae"] is None
    assert metrics["skipped_batches"] == 0
    assert progress_descriptions == ["Train 1/1", "Validation 1/1", "Test bestF.pt"]
    output = capsys.readouterr().out
    assert "Dataset samples: total=3, train=1, validation=1, test=1" in output
    assert "test checkpoint=bestF.pt" in output


def test_evaluate_test_set_restores_in_memory_model_state(tmp_path):
    raw_config = _raw_config()
    raw_config["training"]["output_dir"] = str(tmp_path)
    config = TrainingConfig.from_dict(raw_config)
    trainer = Trainer(config)
    trainer.model = torch.nn.Linear(1, 1)
    trainer.test_loader = [object()]
    trainer.logger = CSVLogger(tmp_path)
    trainer.model.train()
    with torch.no_grad():
        trainer.model.weight.fill_(2.0)
        trainer.model.bias.fill_(3.0)
    original_state = {
        key: value.detach().clone() for key, value in trainer.model.state_dict().items()
    }

    checkpoint_model = torch.nn.Linear(1, 1)
    with torch.no_grad():
        checkpoint_model.weight.fill_(-5.0)
        checkpoint_model.bias.fill_(-7.0)
    torch.save(
        {"state_dict": checkpoint_model.state_dict(), "epoch": 4},
        tmp_path / "bestF.pt",
    )
    observed = []

    def fake_test(*, progress_description):
        observed.append(
            (
                progress_description,
                trainer.model.weight.detach().clone(),
                trainer.model.bias.detach().clone(),
            )
        )
        trainer.model.eval()
        return _metrics_record()

    trainer.test = fake_test

    record = trainer.evaluate_test_set()

    assert observed[0][0] == "Test bestF.pt"
    assert observed[0][1].item() == pytest.approx(-5.0)
    assert observed[0][2].item() == pytest.approx(-7.0)
    for key, value in trainer.model.state_dict().items():
        assert torch.equal(value, original_state[key])
    assert trainer.model.training is True

    metrics = json.loads((tmp_path / "test_metrics.json").read_text())
    assert metrics["checkpoint"] == "bestF.pt"
    assert metrics["epoch"] == 4
    assert record.checkpoint == "bestF.pt"
    assert record.epoch == 4


def test_csv_logger_appends_epoch_records(tmp_path):
    logger = CSVLogger(tmp_path)

    logger.log_epoch(_epoch_record(epoch=1))
    logger.log_epoch(_epoch_record(epoch=2))

    history = pd.read_csv(tmp_path / "history.csv")
    assert history["epoch"].tolist() == [1, 2]
    assert history["val_force_mae"].tolist() == pytest.approx([0.4, 0.4])


def test_csv_logger_writes_evaluation_records(tmp_path):
    logger = CSVLogger(tmp_path)

    logger.log_evaluation(_evaluation_record())

    metrics = json.loads((tmp_path / "test_metrics.json").read_text())
    assert metrics["split"] == "test"
    assert metrics["checkpoint"] == "bestF.pt"
    assert metrics["force_mae"] == pytest.approx(0.2)


def test_console_logger_prints_epoch_summary(capsys):
    logger = ConsoleLogger()

    logger.log_epoch(_epoch_record(epoch=3))

    output = capsys.readouterr().out
    assert "Epoch 3:" in output
    assert "train_MAE(e)=0.10000" in output
    assert "train_MAE(f)=0.20000" in output
    assert "train_MAE(s)=0.300" in output
    assert "val_MAE(e)=0.20000" in output
    assert "val_MAE(f)=0.40000" in output
    assert "val_MAE(s)=0.600" in output
    assert "train_loss" not in output
    assert "val_loss" not in output
    assert output.count("\n") == 1


def test_console_logger_prints_evaluation_summary(capsys):
    logger = ConsoleLogger()

    logger.log_evaluation(_evaluation_record())

    output = capsys.readouterr().out
    assert "test checkpoint=bestF.pt" in output
    assert "MAE(e)=0.10000" in output
    assert "MAE(f)=0.20000" in output
    assert "MAE(s)=0.300" in output
    assert output.count("\n") == 1


def test_composite_logger_dispatches_and_closes():
    first = _MemoryLogger()
    second = _MemoryLogger()
    logger = CompositeLogger([first, second])
    record = _epoch_record(epoch=1)
    evaluation_record = _evaluation_record()

    logger.log_epoch(record)
    logger.log_evaluation(evaluation_record)
    logger.close()

    assert first.records == [record]
    assert second.records == [record]
    assert first.evaluation_records == [evaluation_record]
    assert second.evaluation_records == [evaluation_record]
    assert first.closed is True
    assert second.closed is True


def test_wandb_logger_can_be_disabled():
    logger = WandBLogger(_wandb_config(enabled=False), {})

    logger.log_epoch(_epoch_record(epoch=1))
    logger.close()


def test_wandb_evaluation_uses_next_step_instead_of_checkpoint_epoch():
    calls = []
    logger = object.__new__(WandBLogger)
    logger._wandb = SimpleNamespace(
        log=lambda payload, **kwargs: calls.append((payload, kwargs)),
    )
    logger._run = None

    logger.log_epoch(_epoch_record(epoch=5))
    logger.log_evaluation(_evaluation_record())

    assert calls[0][1] == {"step": 5}
    assert calls[1][0]["test_epoch"] == 4
    assert calls[1][1] == {}


def test_null_logger_is_noop():
    logger = NullLogger()

    logger.log_epoch(_epoch_record(epoch=1))
    logger.log_evaluation(_evaluation_record())
    logger.close()


def test_unwrap_model_returns_raw_model_for_non_ddp():
    model = torch.nn.Linear(1, 1)

    assert unwrap_model(model) is model


def test_cleanup_distributed_keeps_externally_owned_process_group(monkeypatch):
    destroyed = []

    monkeypatch.setattr(distributed_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(distributed_module.dist, "destroy_process_group", lambda: destroyed.append(True))

    cleanup_distributed(
        DistributedContext(
            enabled=True,
            rank=0,
            local_rank=0,
            world_size=2,
            device="cpu",
            owns_process_group=False,
        )
    )

    assert destroyed == []


def test_barrier_passes_cuda_device_ids(monkeypatch):
    calls = []

    monkeypatch.setattr(
        distributed_module.dist,
        "barrier",
        lambda **kwargs: calls.append(kwargs),
    )

    barrier(
        DistributedContext(
            enabled=True,
            rank=0,
            local_rank=2,
            world_size=4,
            device="cuda:2",
            owns_process_group=True,
        )
    )

    assert calls == [{"device_ids": [2]}]


def test_cleanup_distributed_warns_when_destroy_fails(monkeypatch, capsys):
    monkeypatch.setattr(distributed_module.dist, "is_initialized", lambda: True)

    def fail_destroy():
        raise RuntimeError("destroy failed")

    monkeypatch.setattr(distributed_module.dist, "destroy_process_group", fail_destroy)

    cleanup_distributed(
        DistributedContext(
            enabled=True,
            rank=0,
            local_rank=0,
            world_size=2,
            device="cpu",
            owns_process_group=True,
        )
    )

    assert "failed to destroy distributed process group" in capsys.readouterr().err


def test_sync_epoch_metrics_is_noop_when_distributed_disabled():
    metrics = EpochMetrics(
        loss=AverageMeter(),
        energy_mae=AverageMeter(),
        force_mae=AverageMeter(),
        stress_mae=AverageMeter(),
        skipped_batches=2,
    )
    metrics.loss.update(1.5, n=2)

    synced = sync_epoch_metrics(
        metrics,
        DistributedContext.disabled(device="cpu"),
        device="cpu",
    )

    assert synced is metrics
    assert metrics.loss.avg == pytest.approx(1.5)
    assert metrics.skipped_batches == 2


def test_apply_fitted_element_refs_rejects_preloaded_refs_conflict():
    raw_config = _raw_config()
    raw_config["element_references"] = {"source": "fit"}
    config = TrainingConfig.from_dict(raw_config)
    config.model = replace(config.model, element_refs={"1": -1.0})

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
        best_energy_mae=0.1,
        best_force_mae=0.2,
        is_best_energy=True,
        is_best_force=True,
    )

    state = torch.load(tmp_path / "last.pt", map_location="cpu")

    assert state["model_name"] == "GPTFF"
    assert state["best_energy_mae"] == pytest.approx(0.1)
    assert state["best_force_mae"] == pytest.approx(0.2)
    assert state["best_validation_metric"] == pytest.approx(0.2)
    assert state["model_config"]["num_readout_layers"] == 4
    assert state["model_config"]["readout_atom_norm"] is True
    assert "final_atom_norm" not in state["model_config"]
    assert state["model_config"]["interaction_dropout"] == pytest.approx(0.1)
    assert state["model_config"]["atom_attention"]["enabled"] is True
    assert state["model_config"]["atom_attention"]["use_ffn"] is False
    assert state["model_config"]["atom_attention"]["density_scale_init"] == pytest.approx(0.1)
    assert state["model_config"]["atom_attention"]["ffn_residual_scale_init"] == pytest.approx(
        1e-2
    )
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
    assert (tmp_path / "bestE.pt").exists()
    assert (tmp_path / "bestF.pt").exists()


def test_save_checkpoint_updates_best_energy_and_force_independently(tmp_path):
    config = TrainingConfig.from_dict(_raw_config())
    model = torch.nn.Linear(1, 1)

    save_checkpoint(
        tmp_path,
        model,
        config,
        epoch=1,
        best_energy_mae=0.1,
        best_force_mae=0.2,
        is_best_energy=True,
        is_best_force=False,
    )
    assert (tmp_path / "bestE.pt").exists()
    assert not (tmp_path / "bestF.pt").exists()

    save_checkpoint(
        tmp_path,
        model,
        config,
        epoch=2,
        best_energy_mae=0.1,
        best_force_mae=0.15,
        is_best_energy=False,
        is_best_force=True,
    )
    assert (tmp_path / "bestE.pt").exists()
    assert (tmp_path / "bestF.pt").exists()

    best_force_state = torch.load(tmp_path / "bestF.pt", map_location="cpu")
    assert best_force_state["epoch"] == 2
    assert best_force_state["best_force_mae"] == pytest.approx(0.15)


def test_compute_batch_loss_requires_energy_and_force_batches():
    config = TrainingConfig.from_dict(_raw_config())
    config.loss = replace(config.loss, stress_loss_weight=0.0)

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
    config.loss = replace(config.loss, stress_loss_weight=0.0)

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
    config.loss = replace(
        config.loss,
        energy_loss_weight=0.0,
        stress_loss_weight=0.0,
    )

    with pytest.raises(ValueError, match="energy_loss_weight must be positive"):
        compute_batch_loss(
            _ConstantEnergyModel(),
            _energy_force_batch(),
            torch.nn.HuberLoss(),
            config,
            create_graph=False,
        )

    config.loss = replace(
        config.loss,
        energy_loss_weight=1.0,
        force_loss_weight=0.0,
    )
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
    config.loss = replace(config.loss, stress_loss_weight=1.0)

    with pytest.raises(ValueError, match="stress labels are required"):
        compute_batch_loss(
            _ConstantEnergyModel(),
            _energy_force_batch(),
            torch.nn.HuberLoss(),
            config,
            create_graph=False,
        )


def test_compute_batch_loss_converts_native_stress_to_training_gpa(monkeypatch):
    config = TrainingConfig.from_dict(_raw_config())
    config.loss = replace(config.loss, stress_loss_weight=1.0)
    batch = _energy_force_stress_batch()

    def fake_predict(*_args, **_kwargs):
        return (
            batch.energy.clone(),
            batch.forces.clone(),
            batch.stress.clone() / EV_PER_ANG3_TO_GPA,
        )

    monkeypatch.setattr(loss_module, "predict_energy_forces_stress", fake_predict)

    batch_loss = compute_batch_loss(
        _ConstantEnergyModel(),
        batch,
        torch.nn.HuberLoss(),
        config,
        create_graph=False,
    )

    assert batch_loss.loss.item() == pytest.approx(0.0)
    assert batch_loss.stress_mae.item() == pytest.approx(0.0, abs=1e-6)


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


def _energy_force_stress_batch():
    structure = Structure(Lattice.cubic(3.0), ["Na"], [[0.0, 0.0, 0.0]])
    graph = CrystalGraphConverter(radial_cutoff=2.0, angle_cutoff=2.0).convert(structure)
    return CrystalGraphBatch.from_graphs(
        [graph],
        energies=[-1.0],
        forces=[np.zeros((1, 3), dtype=np.float32)],
        stresses=[np.eye(3, dtype=np.float32) * 2.0],
    )


class _MemoryLogger:
    def __init__(self):
        self.records = []
        self.evaluation_records = []
        self.closed = False

    def log_epoch(self, record):
        self.records.append(record)

    def log_evaluation(self, record):
        self.evaluation_records.append(record)

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


def _atomic_dataset(size=2):
    return AtomicDataset(tuple(_atomic_sample(index) for index in range(size)))


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


def _metrics_record():
    return SimpleNamespace(
        loss=SimpleNamespace(avg=1.0, count=1),
        energy_mae=SimpleNamespace(avg=0.1, count=1),
        force_mae=SimpleNamespace(avg=0.2, count=1),
        stress_mae=SimpleNamespace(avg=0.3, count=1),
        skipped_batches=0,
    )


def _evaluation_record():
    return EvaluationRecord(
        split="test",
        checkpoint="bestF.pt",
        epoch=4,
        loss=1.0,
        energy_mae=0.1,
        force_mae=0.2,
        stress_mae=0.3,
        skipped_batches=0,
    )


def _wandb_config(**kwargs):
    return SimpleNamespace(
        enabled=kwargs.get("enabled", True),
        project=kwargs.get("project", "gptff"),
        entity=kwargs.get("entity"),
        name=kwargs.get("name"),
        group=kwargs.get("group"),
        tags=kwargs.get("tags", ()),
        notes=kwargs.get("notes"),
        mode=kwargs.get("mode", "online"),
        job_type=kwargs.get("job_type", "train"),
        init_kwargs=kwargs.get("init_kwargs", {}),
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
            "source": {"1": -1.0, "3": 2.0},
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
                "enabled": True,
                "num_heads": 4,
                "dropout": 0.0,
                "use_ffn": True,
                "ffn_hidden_dim": None,
                "density_scale_init": 0.1,
                "ffn_residual_scale_init": 1e-2,
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
