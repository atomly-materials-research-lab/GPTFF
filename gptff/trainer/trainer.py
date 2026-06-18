from __future__ import annotations

import argparse
import gc
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from gptff.data import (
    apply_fitted_element_refs,
    build_datasets,
    build_loaders,
)
from gptff.model import GPTFF, tModLodaer_t
from gptff.trainer.checkpoint import (
    LoadedCheckpoint,
    load_training_checkpoint,
    resolve_checkpoint_path,
    save_checkpoint,
)
from gptff.trainer.config import TrainingConfig, load_config
from gptff.trainer.loss import (
    BatchLoss,
    compute_batch_loss,
    loss_weight_active,
    mae,
    validate_required_labels,
)
from gptff.trainer.scheduler import CosineAnnealingWarmupRestarts


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


def _format_meter_avg(meter: AverageMeter, precision: int) -> str:
    if meter.count == 0:
        return "n/a"
    return f"{meter.avg:.{precision}f}"


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
    parser = argparse.ArgumentParser(description="Graph-based Pretrained Transformer Force Field.")
    parser.add_argument("config", metavar="OPTIONS", help="Configs for training")
    return parser.parse_args(argv)


def build_model(config: TrainingConfig) -> torch.nn.Module:
    model = tModLodaer_t(config) if config.transformer_activate else GPTFF(config.to_model_config())
    return model.to(config.device)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_optimizer(model: torch.nn.Module, config: TrainingConfig) -> optim.Optimizer:
    return optim.AdamW(model.parameters(), config.lr, weight_decay=config.weight_decay)


def build_scheduler(
    optimizer: optim.Optimizer,
    config: TrainingConfig,
    *,
    start_epoch: Optional[int] = None,
) -> CosineAnnealingWarmupRestarts:
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=config.num_train_steps,
        cycle_mult=1,
        max_lr=config.lr,
        min_lr=config.min_lr,
        warmup_steps=config.warmup_steps,
        gamma=1.0,
    )
    scheduler.step(config.start_epoch if start_epoch is None else start_epoch)
    return scheduler


def use_cuda_amp(config: TrainingConfig) -> bool:
    return torch.device(config.device).type == "cuda"


def should_skip_batch(batch_loss: BatchLoss, config: TrainingConfig) -> bool:
    if not torch.isfinite(batch_loss.loss):
        return True
    if config.max_loss_skip is not None and batch_loss.loss.detach().item() > config.max_loss_skip:
        return True
    return False


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
    scaler: GradScaler,
    config: TrainingConfig,
    *,
    progress=None,
) -> EpochMetrics:
    model.train()
    metrics = EpochMetrics.create()

    for batch_idx, batch in enumerate(train_loader, start=1):
        batch = batch.to(config.device)
        if progress is not None:
            progress.update(1)

        with autocast(enabled=use_cuda_amp(config)):
            batch_loss = compute_batch_loss(
                model,
                batch,
                criterion,
                config,
                create_graph=True,
            )

        if should_skip_batch(batch_loss, config):
            metrics.skipped_batches += 1
            continue

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(batch_loss.loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

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

        with autocast(enabled=use_cuda_amp(config)):
            batch_loss = compute_batch_loss(
                model,
                batch,
                criterion,
                config,
                create_graph=False,
            )

        if should_skip_batch(batch_loss, config):
            metrics.skipped_batches += 1
            continue

        update_metrics(metrics, batch_loss)
        if progress is not None:
            progress.set_description(f"[{batch_idx}/{len(val_loader)}]")
            progress.set_postfix(metrics.as_postfix())

    torch.cuda.empty_cache()
    gc.collect()
    return metrics


def append_validation_history(output_dir: Path, metrics: EpochMetrics) -> None:
    with open(output_dir / "val_history.txt", "a+") as fp:
        fp.write(
            f"{_meter_avg_or_nan(metrics.energy_mae):.4f} "
            f"{_meter_avg_or_nan(metrics.force_mae):.4f} "
            f"{_meter_avg_or_nan(metrics.stress_mae):.4f}\n"
        )


def run_training(config: TrainingConfig) -> float:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, val_dataset = build_datasets(config)
    apply_fitted_element_refs(config, train_dataset)
    train_loader, val_loader = build_loaders(config, train_dataset, val_dataset)

    model = build_model(config)
    print(f"Number of Model parameters: {count_parameters(model)}")

    criterion = nn.HuberLoss()
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(
        optimizer,
        config,
        start_epoch=0 if config.resume else config.start_epoch,
    )
    scaler = GradScaler(enabled=use_cuda_amp(config))

    best_mae_error = 1e12
    start_epoch = config.start_epoch
    if config.resume:
        checkpoint_path = resolve_checkpoint_path(config, output_dir)
        checkpoint = load_training_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            device=config.device,
            scheduler=scheduler,
            scaler=scaler,
        )
        start_epoch = checkpoint.epoch
        best_mae_error = checkpoint.best_mae_error
        print(f"Resumed training from {checkpoint_path} at epoch {start_epoch}.")

    bar_format = "{l_bar}{bar:40}| [{elapsed}<{remaining}{postfix}]"

    for epoch in range(start_epoch, config.epochs):
        print(f"Epoch: [{epoch + 1}/ {config.epochs}], lr: {scheduler.get_lr()[0]:.4e}")
        sys.stdout.flush()

        pbar_train = tqdm(
            train_loader,
            total=len(train_loader),
            mininterval=0.1,
            ascii=True,
            position=0,
            unit="s",
            bar_format=bar_format,
        )
        pbar_val = tqdm(
            val_loader,
            total=len(val_loader),
            mininterval=0.1,
            ascii=True,
            position=0,
            unit="s",
            bar_format=bar_format,
            leave=False,
        )

        train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scaler,
            config,
            progress=pbar_train,
        )
        val_metrics = validate(
            val_loader,
            model,
            criterion,
            config,
            progress=pbar_val,
        )
        pbar_train.set_postfix_str(
            f"val_loss: {_format_meter_avg(val_metrics.loss, 5)} "
            f"val_MAE(e): {_format_meter_avg(val_metrics.energy_mae, 5)}, "
            f"val_MAE(f): {_format_meter_avg(val_metrics.force_mae, 5)}, "
            f"val_MAE(s): {_format_meter_avg(val_metrics.stress_mae, 3)}"
        )
        pbar_train.close()
        pbar_val.close()

        append_validation_history(output_dir, val_metrics)
        scheduler.step()

        validation_metric = select_validation_metric(val_metrics)
        is_best = validation_metric < best_mae_error
        best_mae_error = min(validation_metric, best_mae_error)
        save_checkpoint(
            output_dir,
            model,
            optimizer,
            config,
            epoch=epoch + 1,
            best_mae_error=best_mae_error,
            is_best=is_best,
            scheduler=scheduler,
            scaler=scaler,
        )

    return best_mae_error


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
    run_training(config)


if __name__ == "__main__":
    main()
