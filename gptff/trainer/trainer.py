from __future__ import annotations

import argparse
import gc
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from gptff.data import (
    AtomicDataset,
    apply_fitted_element_refs,
    build_graph_datasets,
    build_loaders,
    load_atomic_dataset,
)
from gptff.model import GPTFF
from gptff.trainer.checkpoint import save_checkpoint
from gptff.trainer.config import TrainingConfig, load_config
from gptff.trainer.loss import (
    BatchLoss,
    compute_batch_loss,
)
from gptff.trainer.logger import (
    CSVLogger,
    CompositeLogger,
    ConsoleLogger,
    EpochLogRecord,
    TrainingLogger,
)
from gptff.utils.reproducibility import (
    configure_reproducibility,
    create_data_loader_generators,
)
from gptff.trainer.scheduler import Scheduler, build_lr_scheduler


@dataclass
class EpochMetrics:
    loss: "AverageMeter"
    energy_mae: "AverageMeter"
    force_mae: "AverageMeter"
    stress_mae: "AverageMeter"
    skipped_batches: int = 0

    @classmethod
    def create(cls) -> "EpochMetrics":
        return cls(
            loss=AverageMeter(),
            energy_mae=AverageMeter(),
            force_mae=AverageMeter(),
            stress_mae=AverageMeter(),
        )

    def as_postfix(self) -> Dict[str, str]:
        return {
            "loss": _format_meter(self.loss, precision=5),
            "MAE(e)": _format_meter(self.energy_mae, precision=5),
            "MAE(f)": _format_meter(self.force_mae, precision=5),
            "MAE(s)": _format_meter(self.stress_mae, precision=3),
            "skip": str(self.skipped_batches),
        }


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


def _format_meter(meter: AverageMeter, precision: int) -> str:
    if meter.count == 0:
        return "n/a"
    return f"{meter.val:.{precision}f} ({meter.avg:.{precision}f})"


def _meter_avg_or_nan(meter: AverageMeter) -> float:
    return meter.avg if meter.count > 0 else float("nan")


def select_validation_metric(metrics: EpochMetrics) -> float:
    for meter in (
        metrics.energy_mae,
        metrics.force_mae,
        metrics.stress_mae,
        metrics.loss,
    ):
        if meter.count > 0:
            return meter.avg
    return float("inf")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a GPTFF model.")
    parser.add_argument("config", metavar="CONFIG", help="YAML training configuration")
    return parser.parse_args(argv)


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
        return optim.SGD(model.parameters(), config.lr, momentum=0.9, weight_decay=config.weight_decay)
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
    progress=None,
) -> EpochMetrics:
    model.train()
    metrics = EpochMetrics.create()
    scheduler_batches = scheduler_step_batches(len(train_loader))

    for batch_idx, batch in enumerate(train_loader, start=1):
        batch = batch.to(config.device)
        if progress is not None:
            progress.update(1)

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
        if progress is not None:
            progress.set_description(f"[{batch_idx}/{len(train_loader)}]")
            progress.set_postfix(metrics.as_postfix())

    torch.cuda.empty_cache()
    gc.collect()
    return metrics


def validate(
    val_loader,
    model: torch.nn.Module,
    criterion: nn.Module,
    config: TrainingConfig,
    *,
    progress=None,
) -> EpochMetrics:
    model.eval()
    metrics = EpochMetrics.create()

    for batch_idx, batch in enumerate(val_loader, start=1):
        batch = batch.to(config.device)
        if progress is not None:
            progress.update(1)

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
            continue

        update_metrics(metrics, batch_loss)
        if progress is not None:
            progress.set_description(f"[{batch_idx}/{len(val_loader)}]")
            progress.set_postfix(metrics.as_postfix())

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
        self.best_validation_metric = 1e12
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
        print(f"Number of Model parameters: {count_parameters(self.model)}")

        self.optimizer = build_optimizer(self.model, self.config)
        self.scheduler = build_scheduler(self.optimizer, self.config)
        self.scaler = GradScaler("cuda", enabled=use_cuda_amp(self.config))

    def train_epoch(self, *, progress=None) -> EpochMetrics:
        return train_one_epoch(
            self.train_loader,
            self.model,
            self.criterion,
            self.optimizer,
            self.scheduler,
            self.scaler,
            self.config,
            progress=progress,
        )

    def validate(self, *, progress=None) -> EpochMetrics:
        return validate(
            self.val_loader,
            self.model,
            self.criterion,
            self.config,
            progress=progress,
        )

    def fit(self, dataset: AtomicDataset | None = None) -> float:
        self.setup(dataset)
        bar_format = "{l_bar}{bar:40}| [{elapsed}<{remaining}{postfix}]"

        try:
            for epoch in range(self.config.epochs):
                lr = optimizer_lr(self.optimizer)
                print(f"Epoch: [{epoch + 1}/ {self.config.epochs}], lr: {lr:.4e}")
                sys.stdout.flush()

                pbar_train = tqdm(
                    self.train_loader,
                    total=len(self.train_loader),
                    mininterval=0.1,
                    ascii=True,
                    position=0,
                    unit="s",
                    bar_format=bar_format,
                )
                pbar_val = tqdm(
                    self.val_loader,
                    total=len(self.val_loader),
                    mininterval=0.1,
                    ascii=True,
                    position=0,
                    unit="s",
                    bar_format=bar_format,
                    leave=False,
                )

                train_metrics = self.train_epoch(progress=pbar_train)
                val_metrics = self.validate(progress=pbar_val)
                pbar_train.close()
                pbar_val.close()

                self.logger.log_epoch(
                    build_epoch_log_record(
                        epoch=epoch + 1,
                        lr=lr,
                        train_metrics=train_metrics,
                        val_metrics=val_metrics,
                    )
                )

                validation_metric = select_validation_metric(val_metrics)
                is_best = validation_metric < self.best_validation_metric
                self.best_validation_metric = min(validation_metric, self.best_validation_metric)
                save_checkpoint(
                    self.output_dir,
                    self.model,
                    self.config,
                    epoch=epoch + 1,
                    best_validation_metric=self.best_validation_metric,
                    is_best=is_best,
                )
        finally:
            self.logger.close()

        return self.best_validation_metric


def run_training(
    config: TrainingConfig,
    dataset: AtomicDataset | None = None,
) -> float:
    return Trainer(config).fit(dataset)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
    run_training(config)


if __name__ == "__main__":
    main()
