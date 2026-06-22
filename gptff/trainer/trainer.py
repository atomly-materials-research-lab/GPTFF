from __future__ import annotations

import gc
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from tqdm.auto import tqdm

from gptff.data import (
    AtomicDataset,
    apply_fitted_element_refs,
    build_graph_datasets,
    build_loaders,
    load_atomic_dataset,
)
from gptff.model import GPTFF
from gptff.trainer.checkpoint import save_checkpoint
from gptff.trainer.config import TrainingConfig
from gptff.trainer.evaluation import (
    EvaluationRecord,
    build_evaluation_record,
)
from gptff.trainer.logger import (
    CompositeLogger,
    ConsoleLogger,
    CSVLogger,
    EpochLogRecord,
    TrainingLogger,
    WandBLogger,
)
from gptff.trainer.loss import (
    BatchLoss,
    compute_batch_loss,
)
from gptff.trainer.scheduler import Scheduler, build_lr_scheduler
from gptff.utils.reproducibility import (
    configure_reproducibility,
    create_data_loader_generators,
)


@dataclass
class EpochMetrics:
    loss: AverageMeter
    energy_mae: AverageMeter
    force_mae: AverageMeter
    stress_mae: AverageMeter
    skipped_batches: int = 0

    @classmethod
    def create(cls) -> EpochMetrics:
        return cls(
            loss=AverageMeter(),
            energy_mae=AverageMeter(),
            force_mae=AverageMeter(),
            stress_mae=AverageMeter(),
        )


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0


def _meter_avg_or_nan(meter: AverageMeter) -> float:
    return meter.avg if meter.count > 0 else float("nan")


def _meter_avg_or_inf(meter: AverageMeter) -> float:
    return meter.avg if meter.count > 0 else float("inf")


def build_model(config: TrainingConfig) -> torch.nn.Module:
    model = GPTFF(config.to_model_config())
    return model.to(config.device)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_optimizer(model: torch.nn.Module, config: TrainingConfig) -> optim.Optimizer:
    optimizer_name = config.optimizer_name.lower()
    if optimizer_name == "adam":
        return optim.Adam(model.parameters(), config.lr, weight_decay=config.weight_decay)
    if optimizer_name == "adamw":
        return optim.AdamW(model.parameters(), config.lr, weight_decay=config.weight_decay)
    if optimizer_name == "radam":
        return optim.RAdam(model.parameters(), config.lr, weight_decay=config.weight_decay)
    if optimizer_name == "sgd":
        return optim.SGD(
            model.parameters(), config.lr, momentum=0.9, weight_decay=config.weight_decay
        )
    raise ValueError(f"Unsupported optimizer: {config.optimizer_name}")


def build_scheduler(
    optimizer: optim.Optimizer,
    config: TrainingConfig,
) -> Scheduler | None:
    return build_lr_scheduler(
        optimizer,
        scheduler=config.scheduler,
        learning_rate=config.lr,
        epochs=config.epochs,
        scheduler_params=config.scheduler_params,
    )


def scheduler_step_batches(num_batches: int, steps_per_epoch: int = 10) -> set[int]:
    if num_batches <= 0 or steps_per_epoch <= 0:
        return set()
    return {
        min(num_batches, max(1, math.ceil(num_batches * step / steps_per_epoch)))
        for step in range(1, steps_per_epoch + 1)
    }


def use_cuda_amp(config: TrainingConfig) -> bool:
    return bool(config.amp) and torch.device(config.device).type == "cuda"


def has_nonfinite_loss(batch_loss: BatchLoss) -> bool:
    return not bool(torch.isfinite(batch_loss.loss).item())


def update_metrics(metrics: EpochMetrics, batch_loss: BatchLoss) -> None:
    metrics.loss.update(batch_loss.loss.detach().cpu().item(), batch_loss.batch_size)
    if batch_loss.energy_mae is not None:
        metrics.energy_mae.update(batch_loss.energy_mae.cpu().item(), batch_loss.batch_size)
    if batch_loss.force_mae is not None:
        metrics.force_mae.update(batch_loss.force_mae.cpu().item(), batch_loss.force_count)
    if batch_loss.stress_mae is not None:
        metrics.stress_mae.update(batch_loss.stress_mae.cpu().item(), batch_loss.stress_count)


def train_one_epoch(
    train_loader,
    model: torch.nn.Module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Scheduler | None,
    scaler: GradScaler,
    config: TrainingConfig,
    *,
    progress_description: str = "Train",
) -> EpochMetrics:
    model.train()
    metrics = EpochMetrics.create()
    scheduler_batches = scheduler_step_batches(len(train_loader))

    progress = tqdm(
        train_loader,
        desc=progress_description,
        unit="batch",
        dynamic_ncols=True,
        leave=False,
    )
    for batch_idx, batch in enumerate(progress, start=1):
        batch = batch.to(config.device)

        with autocast("cuda", enabled=use_cuda_amp(config)):
            batch_loss = compute_batch_loss(
                model,
                batch,
                criterion,
                config,
                create_graph=True,
            )

        if has_nonfinite_loss(batch_loss):
            metrics.skipped_batches += 1
            progress.set_postfix(skipped=metrics.skipped_batches, refresh=False)
            continue

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(batch_loss.loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None and batch_idx in scheduler_batches:
            scheduler.step()

        update_metrics(metrics, batch_loss)
        progress.set_postfix(loss=f"{metrics.loss.avg:.5f}", refresh=False)

    torch.cuda.empty_cache()
    gc.collect()
    return metrics


def validate(
    val_loader,
    model: torch.nn.Module,
    criterion: nn.Module,
    config: TrainingConfig,
    *,
    progress_description: str = "Validation",
) -> EpochMetrics:
    model.eval()
    metrics = EpochMetrics.create()

    progress = tqdm(
        val_loader,
        desc=progress_description,
        unit="batch",
        dynamic_ncols=True,
        leave=False,
    )
    for batch in progress:
        batch = batch.to(config.device)

        with autocast("cuda", enabled=use_cuda_amp(config)):
            batch_loss = compute_batch_loss(
                model,
                batch,
                criterion,
                config,
                create_graph=False,
            )

        if has_nonfinite_loss(batch_loss):
            metrics.skipped_batches += 1
            progress.set_postfix(skipped=metrics.skipped_batches, refresh=False)
            continue

        update_metrics(metrics, batch_loss)
        progress.set_postfix(loss=f"{metrics.loss.avg:.5f}", refresh=False)

    torch.cuda.empty_cache()
    gc.collect()
    return metrics


def optimizer_lr(optimizer: optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def build_epoch_log_record(
    *,
    epoch: int,
    lr: float,
    train_metrics: EpochMetrics,
    val_metrics: EpochMetrics,
) -> EpochLogRecord:
    return EpochLogRecord(
        epoch=int(epoch),
        lr=float(lr),
        train_loss=_meter_avg_or_nan(train_metrics.loss),
        train_energy_mae=_meter_avg_or_nan(train_metrics.energy_mae),
        train_force_mae=_meter_avg_or_nan(train_metrics.force_mae),
        train_stress_mae=_meter_avg_or_nan(train_metrics.stress_mae),
        train_skipped_batches=int(train_metrics.skipped_batches),
        val_loss=_meter_avg_or_nan(val_metrics.loss),
        val_energy_mae=_meter_avg_or_nan(val_metrics.energy_mae),
        val_force_mae=_meter_avg_or_nan(val_metrics.force_mae),
        val_stress_mae=_meter_avg_or_nan(val_metrics.stress_mae),
        val_skipped_batches=int(val_metrics.skipped_batches),
    )


class Trainer:
    def __init__(self, config: TrainingConfig, logger: TrainingLogger | None = None):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.criterion: nn.Module = nn.HuberLoss()
        self.model: torch.nn.Module | None = None
        self.optimizer: optim.Optimizer | None = None
        self.scheduler: Scheduler | None = None
        self.scaler: GradScaler | None = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.data_loader_generators: dict[str, torch.Generator] = {}
        self.best_energy_mae = float("inf")
        self.best_force_mae = float("inf")
        self.logger = logger

    def setup(self, dataset: AtomicDataset | None = None) -> None:
        configure_reproducibility(self.config.seed, self.config.deterministic)
        self.data_loader_generators = create_data_loader_generators(self.config.seed)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.logger is None:
            self.logger = CompositeLogger(
                [
                    CSVLogger(self.output_dir),
                    ConsoleLogger(),
                    WandBLogger(self.config.logging.wandb, self.config.checkpoint_dict()),
                ]
            )
        if dataset is None:
            dataset = load_atomic_dataset(self.config)
        if not isinstance(dataset, AtomicDataset):
            raise TypeError("dataset must be an AtomicDataset.")
        datasets = build_graph_datasets(
            dataset,
            self.config,
        )
        test_size = len(datasets.test) if datasets.test is not None else 0
        print(
            "Dataset samples: "
            f"total={len(dataset)}, "
            f"train={len(datasets.train)}, "
            f"validation={len(datasets.validation)}, "
            f"test={test_size}",
            flush=True,
        )
        apply_fitted_element_refs(self.config, datasets.train)
        loaders = build_loaders(
            self.config,
            datasets,
            generators=self.data_loader_generators,
        )
        self.train_loader = loaders.train
        self.val_loader = loaders.validation
        self.test_loader = loaders.test

        self.model = build_model(self.config)
        print(f"Number of Model parameters: {count_parameters(self.model)}", flush=True)

        self.optimizer = build_optimizer(self.model, self.config)
        self.scheduler = build_scheduler(self.optimizer, self.config)
        self.scaler = GradScaler("cuda", enabled=use_cuda_amp(self.config))

    def train_epoch(self, epoch: int) -> EpochMetrics:
        return train_one_epoch(
            self.train_loader,
            self.model,
            self.criterion,
            self.optimizer,
            self.scheduler,
            self.scaler,
            self.config,
            progress_description=f"Train {epoch}/{self.config.epochs}",
        )

    def validate(self, epoch: int) -> EpochMetrics:
        return validate(
            self.val_loader,
            self.model,
            self.criterion,
            self.config,
            progress_description=f"Validation {epoch}/{self.config.epochs}",
        )

    def test(self, *, progress_description: str = "Test") -> EpochMetrics | None:
        if self.test_loader is None:
            return None
        return validate(
            self.test_loader,
            self.model,
            self.criterion,
            self.config,
            progress_description=progress_description,
        )

    def evaluate_test_set(
        self,
        *,
        checkpoint_filename: str = "bestF.pt",
    ) -> EvaluationRecord | None:
        if self.test_loader is None:
            return None

        checkpoint_path = self.output_dir / checkpoint_filename
        original_state = _clone_state_dict(self.model.state_dict())
        original_training = self.model.training
        checkpoint_name = None
        checkpoint_epoch = None

        try:
            if checkpoint_path.exists():
                state = torch.load(checkpoint_path, map_location=torch.device(self.config.device))
                self.model.load_state_dict(state["state_dict"])
                checkpoint_name = checkpoint_filename
                checkpoint_epoch = int(state["epoch"]) if "epoch" in state else None

            test_metrics = self.test(
                progress_description=f"Test {checkpoint_name or 'current'}",
            )
        finally:
            self.model.load_state_dict(original_state)
            self.model.train(original_training)

        record = build_evaluation_record(
            split="test",
            checkpoint=checkpoint_name,
            epoch=checkpoint_epoch,
            loss=_meter_avg_or_nan(test_metrics.loss),
            energy_mae=_meter_avg_or_nan(test_metrics.energy_mae),
            force_mae=_meter_avg_or_nan(test_metrics.force_mae),
            stress_mae=_meter_avg_or_nan(test_metrics.stress_mae),
            skipped_batches=test_metrics.skipped_batches,
        )
        if self.logger is not None:
            self.logger.log_evaluation(record)
        return record

    def fit(self, dataset: AtomicDataset | None = None) -> float:
        self.setup(dataset)

        try:
            for epoch in range(self.config.epochs):
                lr = optimizer_lr(self.optimizer)
                current_epoch = epoch + 1
                print(
                    f"Epoch: [{current_epoch}/{self.config.epochs}], lr: {lr:.4e}",
                    flush=True,
                )

                train_metrics = self.train_epoch(current_epoch)
                val_metrics = self.validate(current_epoch)

                self.logger.log_epoch(
                    build_epoch_log_record(
                        epoch=current_epoch,
                        lr=lr,
                        train_metrics=train_metrics,
                        val_metrics=val_metrics,
                    )
                )

                validation_energy_mae = _meter_avg_or_inf(val_metrics.energy_mae)
                validation_force_mae = _meter_avg_or_inf(val_metrics.force_mae)
                is_best_energy = validation_energy_mae < self.best_energy_mae
                is_best_force = validation_force_mae < self.best_force_mae
                self.best_energy_mae = min(validation_energy_mae, self.best_energy_mae)
                self.best_force_mae = min(validation_force_mae, self.best_force_mae)
                save_checkpoint(
                    self.output_dir,
                    self.model,
                    self.config,
                    epoch=current_epoch,
                    best_energy_mae=self.best_energy_mae,
                    best_force_mae=self.best_force_mae,
                    is_best_energy=is_best_energy,
                    is_best_force=is_best_force,
                )
            self.evaluate_test_set()
        finally:
            self.logger.close()

        return self.best_force_mae


def run_training(
    config: TrainingConfig,
    dataset: AtomicDataset | None = None,
) -> float:
    return Trainer(config).fit(dataset)


def _clone_state_dict(state_dict):
    return {key: value.detach().cpu().clone() for key, value in state_dict.items()}
