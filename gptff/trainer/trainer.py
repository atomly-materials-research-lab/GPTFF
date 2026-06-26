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
    ShardedGraphDataset,
    apply_fitted_element_refs,
    build_graph_datasets,
    build_loaders,
    load_training_dataset,
)
from gptff.model import GPTFF
from gptff.trainer.checkpoint import save_checkpoint
from gptff.trainer.config import TrainingConfig
from gptff.trainer.distributed import (
    DistributedContext,
    all_reduce_max,
    all_reduce_sum,
    barrier,
    cleanup_distributed,
    initialize_distributed,
    unwrap_model,
    wrap_distributed_model,
)
from gptff.trainer.evaluation import (
    EvaluationRecord,
    build_evaluation_record,
)
from gptff.trainer.logger import (
    CompositeLogger,
    ConsoleLogger,
    CSVLogger,
    EpochLogRecord,
    NullLogger,
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

    def set_sum_count(self, total: float, count: int) -> None:
        self.sum = float(total)
        self.count = int(count)
        self.avg = self.sum / self.count if self.count > 0 else 0.0
        self.val = self.avg


def _meter_avg_or_nan(meter: AverageMeter) -> float:
    return meter.avg if meter.count > 0 else float("nan")


def _meter_avg_or_inf(meter: AverageMeter) -> float:
    return meter.avg if meter.count > 0 else float("inf")


def build_model(config: TrainingConfig) -> torch.nn.Module:
    model = GPTFF(config.to_model_config())
    return model.to(config.device)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_weight_decay_param_groups(
    model: torch.nn.Module,
    weight_decay: float,
) -> list[dict[str, object]]:
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if _exclude_from_weight_decay(name, param):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    if weight_decay == 0:
        return [{"params": decay_params + no_decay_params, "weight_decay": 0.0}]

    param_groups: list[dict[str, object]] = []
    if decay_params:
        param_groups.append({"params": decay_params, "weight_decay": weight_decay})
    if no_decay_params:
        param_groups.append({"params": no_decay_params, "weight_decay": 0.0})
    return param_groups


def _exclude_from_weight_decay(name: str, param: torch.nn.Parameter) -> bool:
    no_decay_prefixes = (
        "atom_embedding.",
        "geometry_embedding.",
        "readout.",
    )
    return (
        name.endswith(".bias")
        or param.ndim == 1
        or "residual_scale" in name
        or "element_ref" in name
        or name.startswith(no_decay_prefixes)
    )


def build_optimizer(model: torch.nn.Module, config: TrainingConfig) -> optim.Optimizer:
    optimizer_name = config.optimizer_name.lower()
    param_groups = build_weight_decay_param_groups(model, config.weight_decay)
    if optimizer_name == "adam":
        return optim.Adam(param_groups, config.lr)
    if optimizer_name == "adamw":
        return optim.AdamW(param_groups, config.lr)
    if optimizer_name == "radam":
        return optim.RAdam(param_groups, config.lr)
    if optimizer_name == "sgd":
        return optim.SGD(param_groups, config.lr, momentum=0.9)
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


def use_non_blocking_transfer(config: TrainingConfig) -> bool:
    return torch.device(config.device).type == "cuda"


def has_nonfinite_loss(batch_loss: BatchLoss) -> bool:
    return not bool(torch.isfinite(batch_loss.loss).item())


def should_skip_optimizer_step(
    batch_loss: BatchLoss,
    context: DistributedContext,
    *,
    device: str,
) -> bool:
    local_skip = int(has_nonfinite_loss(batch_loss))
    if not context.enabled:
        return bool(local_skip)
    skip = torch.tensor([local_skip], dtype=torch.int64, device=torch.device(device))
    all_reduce_sum(skip, context)
    return bool(skip.item() > 0)


def update_metrics(metrics: EpochMetrics, batch_loss: BatchLoss) -> None:
    metrics.loss.update(batch_loss.loss.detach().cpu().item(), batch_loss.batch_size)
    if batch_loss.energy_mae is not None:
        metrics.energy_mae.update(batch_loss.energy_mae.cpu().item(), batch_loss.batch_size)
    if batch_loss.force_mae is not None:
        metrics.force_mae.update(batch_loss.force_mae.cpu().item(), batch_loss.force_count)
    if batch_loss.stress_mae is not None:
        metrics.stress_mae.update(batch_loss.stress_mae.cpu().item(), batch_loss.stress_count)


def sync_epoch_metrics(
    metrics: EpochMetrics,
    context: DistributedContext,
    *,
    device: str,
    skipped_reduce: str = "sum",
) -> EpochMetrics:
    if not context.enabled:
        return metrics
    torch_device = torch.device(device)
    for meter in (
        metrics.loss,
        metrics.energy_mae,
        metrics.force_mae,
        metrics.stress_mae,
    ):
        values = torch.tensor(
            [meter.sum, float(meter.count)],
            dtype=torch.float64,
            device=torch_device,
        )
        all_reduce_sum(values, context)
        meter.set_sum_count(values[0].item(), int(values[1].item()))
    skipped = torch.tensor(
        [metrics.skipped_batches],
        dtype=torch.int64,
        device=torch_device,
    )
    if skipped_reduce == "sum":
        all_reduce_sum(skipped, context)
    elif skipped_reduce == "max":
        all_reduce_max(skipped, context)
    else:
        raise ValueError("skipped_reduce must be 'sum' or 'max'.")
    metrics.skipped_batches = int(skipped.item())
    return metrics


class _ProgressDisabled:
    def __init__(self, iterable) -> None:
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable)

    def set_postfix(self, **_kwargs) -> None:
        return None


def _progress_iterator(iterable, *, show: bool, desc: str):
    if not show:
        return _ProgressDisabled(iterable)
    return tqdm(
        iterable,
        desc=desc,
        unit="batch",
        dynamic_ncols=True,
        leave=False,
    )


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
    show_progress: bool = True,
    distributed: DistributedContext | None = None,
) -> EpochMetrics:
    distributed = distributed or DistributedContext.disabled(device=config.device)
    model.train()
    metrics = EpochMetrics.create()
    scheduler_batches = scheduler_step_batches(len(train_loader))

    progress = _progress_iterator(
        train_loader,
        show=show_progress,
        desc=progress_description,
    )
    for batch_idx, batch in enumerate(progress, start=1):
        batch = batch.to(
            config.device,
            non_blocking=use_non_blocking_transfer(config),
        )

        with autocast("cuda", enabled=use_cuda_amp(config)):
            batch_loss = compute_batch_loss(
                model,
                batch,
                criterion,
                config,
                create_graph=True,
            )

        if should_skip_optimizer_step(batch_loss, distributed, device=config.device):
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
    show_progress: bool = True,
) -> EpochMetrics:
    model.eval()
    metrics = EpochMetrics.create()

    progress = _progress_iterator(
        val_loader,
        show=show_progress,
        desc=progress_description,
    )
    for batch in progress:
        batch = batch.to(
            config.device,
            non_blocking=use_non_blocking_transfer(config),
        )

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
        self.distributed = DistributedContext.disabled(device=config.device)
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.train_sampler = None
        self.data_loader_generators: dict[str, torch.Generator] = {}
        self.best_energy_mae = float("inf")
        self.best_force_mae = float("inf")
        self.logger = logger

    def setup(self, dataset: AtomicDataset | None = None) -> None:
        self.distributed = initialize_distributed(
            self.config.distributed,
            requested_device=self.config.device,
        )
        if self.distributed.device is not None:
            self.config.device = self.distributed.device
        configure_reproducibility(self.config.seed, self.config.deterministic)
        self.data_loader_generators = create_data_loader_generators(self.config.seed)
        if self.distributed.is_main_process:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        barrier(self.distributed)
        if self.logger is not None and not self.distributed.is_main_process:
            self.logger = NullLogger()
        if self.logger is None:
            if self.distributed.is_main_process:
                self.logger = CompositeLogger(
                    [
                        CSVLogger(self.output_dir),
                        ConsoleLogger(),
                        WandBLogger(self.config.logging.wandb, self.config.checkpoint_dict()),
                    ]
                )
            else:
                self.logger = NullLogger()
        if dataset is None:
            dataset = load_training_dataset(self.config)
        if not isinstance(dataset, (AtomicDataset, ShardedGraphDataset)):
            raise TypeError("dataset must be an AtomicDataset or ShardedGraphDataset.")
        apply_fitted_element_refs(
            self.config,
            dataset,
            verbose=self.distributed.is_main_process,
        )
        datasets = build_graph_datasets(
            dataset,
            self.config,
        )
        test_size = len(datasets.test) if datasets.test is not None else 0
        self._print(
            "Dataset samples: "
            f"total={len(dataset)}, "
            f"train={len(datasets.train)}, "
            f"validation={len(datasets.validation)}, "
            f"test={test_size}"
        )
        loaders = build_loaders(
            self.config,
            datasets,
            generators=self.data_loader_generators,
            distributed=self.distributed,
        )
        self.train_loader = loaders.train
        self.val_loader = loaders.validation
        self.test_loader = loaders.test
        self.train_sampler = loaders.train_sampler

        raw_model = build_model(self.config)
        self._print(f"Number of Model parameters: {count_parameters(raw_model)}")

        self.optimizer = build_optimizer(raw_model, self.config)
        self.scheduler = build_scheduler(self.optimizer, self.config)
        self.model = wrap_distributed_model(raw_model, self.distributed)
        self.scaler = GradScaler("cuda", enabled=use_cuda_amp(self.config))

    def _print(self, message: str) -> None:
        if self.distributed.is_main_process:
            print(message, flush=True)

    def train_epoch(self, epoch: int) -> EpochMetrics:
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)
        metrics = train_one_epoch(
            self.train_loader,
            self.model,
            self.criterion,
            self.optimizer,
            self.scheduler,
            self.scaler,
            self.config,
            progress_description=f"Train {epoch}/{self.config.epochs}",
            show_progress=self.distributed.is_main_process,
            distributed=self.distributed,
        )
        return sync_epoch_metrics(
            metrics,
            self.distributed,
            device=self.config.device,
            skipped_reduce="max",
        )

    def validate(self, epoch: int) -> EpochMetrics:
        metrics = validate(
            self.val_loader,
            unwrap_model(self.model),
            self.criterion,
            self.config,
            progress_description=f"Validation {epoch}/{self.config.epochs}",
            show_progress=self.distributed.is_main_process,
        )
        return sync_epoch_metrics(metrics, self.distributed, device=self.config.device)

    def test(self, *, progress_description: str = "Test") -> EpochMetrics | None:
        if self.test_loader is None:
            return None
        metrics = validate(
            self.test_loader,
            unwrap_model(self.model),
            self.criterion,
            self.config,
            progress_description=progress_description,
            show_progress=self.distributed.is_main_process,
        )
        return sync_epoch_metrics(metrics, self.distributed, device=self.config.device)

    def evaluate_test_set(
        self,
        *,
        checkpoint_filename: str = "bestF.pt",
    ) -> EvaluationRecord | None:
        if self.test_loader is None:
            return None

        barrier(self.distributed)
        raw_model = unwrap_model(self.model)
        checkpoint_path = self.output_dir / checkpoint_filename
        original_state = _clone_state_dict(raw_model.state_dict())
        original_training = raw_model.training
        checkpoint_name = None
        checkpoint_epoch = None

        try:
            if checkpoint_path.exists():
                state = torch.load(checkpoint_path, map_location=torch.device(self.config.device))
                raw_model.load_state_dict(state["state_dict"])
                checkpoint_name = checkpoint_filename
                checkpoint_epoch = int(state["epoch"]) if "epoch" in state else None

            test_metrics = self.test(
                progress_description=f"Test {checkpoint_name or 'current'}",
            )
        finally:
            raw_model.load_state_dict(original_state)
            raw_model.train(original_training)

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
        barrier(self.distributed)
        return record

    def fit(self, dataset: AtomicDataset | None = None) -> float:
        try:
            self.setup(dataset)
            for epoch in range(self.config.epochs):
                lr = optimizer_lr(self.optimizer)
                current_epoch = epoch + 1
                self._print(f"Epoch: [{current_epoch}/{self.config.epochs}], lr: {lr:.4e}")

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
                if self.distributed.is_main_process:
                    save_checkpoint(
                        self.output_dir,
                        unwrap_model(self.model),
                        self.config,
                        epoch=current_epoch,
                        best_energy_mae=self.best_energy_mae,
                        best_force_mae=self.best_force_mae,
                        is_best_energy=is_best_energy,
                        is_best_force=is_best_force,
                    )
                barrier(self.distributed)
            self.evaluate_test_set()
        finally:
            if self.logger is not None:
                self.logger.close()
            cleanup_distributed(self.distributed)

        return self.best_force_mae


def run_training(
    config: TrainingConfig,
    dataset: AtomicDataset | None = None,
) -> float:
    return Trainer(config).fit(dataset)


def _clone_state_dict(state_dict):
    return {key: value.detach().cpu().clone() for key, value in state_dict.items()}
