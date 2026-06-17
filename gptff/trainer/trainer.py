from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from gptff.model import GPTFFNet, GPTFFNetConfig, tModLodaer_t
from gptff.model.element_refs import fit_element_refs_from_samples
from gptff.model.prediction import predict_energy_forces_stress
from gptff.utils_.data import (
    CosineAnnealingWarmupRestarts,
    StructureDataset,
    collate_graph_samples,
    validate_dataframe_schema,
)
from gptff.utils_.labels import LabelConfig


@dataclass
class TrainingConfig:
    val_fold: int
    num_train_steps: int
    warmup_steps: int
    batch_size: int
    device: str
    data_path: str
    data_file: str
    energy_unit: str
    force_unit: str
    stress_unit: str
    stress_sign: float
    cache_graphs: bool
    graph_cache_size: Optional[int]
    num_workers: int
    lr: float
    weight_decay: float
    epochs: int
    start_epoch: int
    w1: float
    w2: float
    w3: float
    transformer_activate: bool
    node_feature_len: int
    edge_feature_len: int
    n_layers: int
    num_radial: int = 16
    num_angular: int = 4
    radial_cutoff: float = 5.0
    angle_cutoff: float = 3.5
    cutoff_coeff: int = 5
    max_atomic_number: int = 94
    element_refs: Any = None
    fit_element_refs: bool = False
    element_ref_ridge: float = 0.0
    n_readout_layers: int = 3
    readout_zero_init: bool = True
    interaction_dropout: float = 0.0
    residual_scale: float = 1.0
    aggregation_norm: str = "sqrt"
    unit_trans: float = 160.21766208
    output_dir: str = "."
    min_lr: float = 5e-6
    grad_clip_norm: float = 10.0
    max_loss_skip: Optional[float] = 10.0
    resume: bool = False
    checkpoint_path: Optional[str] = None

    @classmethod
    def from_dict(cls, raw_config: Dict[str, Any]) -> "TrainingConfig":
        training = raw_config["training"]
        data = raw_config["data"]
        epochs = int(training["epochs"])
        return cls(
            val_fold=int(training["val_fold"]),
            num_train_steps=int(training.get("num_train_steps", epochs)),
            warmup_steps=int(training["warmup_steps"]),
            batch_size=int(training["batch_size"]),
            device=str(training["device"]),
            data_path=str(data["data_path"]),
            data_file=str(data["data_file"]),
            energy_unit=str(data.get("energy_unit", "ev")),
            force_unit=str(data.get("force_unit", "ev_per_ang")),
            stress_unit=str(data.get("stress_unit", "kbar")),
            stress_sign=float(data.get("stress_sign", -1.0)),
            cache_graphs=bool(data.get("cache_graphs", False)),
            graph_cache_size=_optional_int(data.get("graph_cache_size", None)),
            num_workers=int(training["workers"]),
            lr=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
            epochs=epochs,
            start_epoch=int(training["start_epoch"]),
            w1=float(training["weight_energy"]),
            w2=float(training["weight_force"]),
            w3=float(training["weight_stress"]),
            transformer_activate=bool(training["transformer_activate"]),
            node_feature_len=int(training["node_feature_len"]),
            edge_feature_len=int(training["edge_feature_len"]),
            n_layers=int(training["n_layers"]),
            num_radial=int(training.get("num_radial", 16)),
            num_angular=int(training.get("num_angular", 4)),
            radial_cutoff=float(training.get("radial_cutoff", 5.0)),
            angle_cutoff=float(training.get("angle_cutoff", 3.5)),
            cutoff_coeff=int(training.get("cutoff_coeff", 5)),
            max_atomic_number=int(training.get("max_atomic_number", 94)),
            element_refs=training.get("element_refs", None),
            fit_element_refs=bool(training.get("fit_element_refs", False)),
            element_ref_ridge=float(training.get("element_ref_ridge", 0.0)),
            n_readout_layers=int(training.get("n_readout_layers", 3)),
            readout_zero_init=bool(training.get("readout_zero_init", True)),
            interaction_dropout=float(training.get("interaction_dropout", 0.0)),
            residual_scale=float(training.get("residual_scale", 1.0)),
            aggregation_norm=str(training.get("aggregation_norm", "sqrt")),
            unit_trans=float(training.get("unit_trans", 160.21766208)),
            output_dir=str(training.get("output_dir", raw_config.get("output_dir", "."))),
            min_lr=float(training.get("min_lr", 5e-6)),
            grad_clip_norm=float(training.get("grad_clip_norm", 10.0)),
            max_loss_skip=training.get("max_loss_skip", 10.0),
            resume=bool(training.get("resume", False)),
            checkpoint_path=training.get("checkpoint_path", None),
        )

    def checkpoint_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_label_config(self) -> LabelConfig:
        return LabelConfig(
            energy_unit=self.energy_unit,
            force_unit=self.force_unit,
            stress_unit=self.stress_unit,
            stress_sign=self.stress_sign,
        )

    def to_model_config(self) -> GPTFFNetConfig:
        return GPTFFNetConfig(
            node_feature_len=self.node_feature_len,
            edge_feature_len=self.edge_feature_len,
            n_layers=self.n_layers,
            num_radial=self.num_radial,
            num_angular=self.num_angular,
            radial_cutoff=self.radial_cutoff,
            angle_cutoff=self.angle_cutoff,
            cutoff_coeff=self.cutoff_coeff,
            max_atomic_number=self.max_atomic_number,
            element_refs=self.element_refs,
            n_readout_layers=self.n_readout_layers,
            readout_zero_init=self.readout_zero_init,
            interaction_dropout=self.interaction_dropout,
            residual_scale=self.residual_scale,
            aggregation_norm=self.aggregation_norm,
        )


@dataclass
class BatchLoss:
    loss: torch.Tensor
    energy_mae: Optional[torch.Tensor]
    force_mae: Optional[torch.Tensor]
    stress_mae: Optional[torch.Tensor]
    batch_size: int
    force_count: int = 0
    stress_count: int = 0


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


@dataclass(frozen=True)
class LoadedCheckpoint:
    epoch: int
    best_mae_error: float
    training_config: Optional[Dict[str, Any]]
    model_config: Optional[Dict[str, Any]]
    label_config: Optional[Dict[str, Any]]


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


def _optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    return int(value)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Graph-based Pretrained Transformer Force Field.")
    parser.add_argument("config", metavar="OPTIONS", help="Configs for training")
    return parser.parse_args(argv)


def load_config(config_file: Union[str, Path]) -> TrainingConfig:
    with open(config_file, "r") as fp:
        raw_config = json.load(fp)
    return TrainingConfig.from_dict(raw_config)


def read_data(config: TrainingConfig) -> pd.DataFrame:
    return pd.read_csv(Path(config.data_path) / config.data_file)


def build_datasets(config: TrainingConfig) -> Tuple[StructureDataset, StructureDataset]:
    df = read_data(config)
    if "fold" not in df.columns:
        raise ValueError("Dataframe must contain a 'fold' column.")
    label_config = config.to_label_config()
    validate_dataframe_schema(
        df,
        label_config,
        require_energy=config.w1 > 0.0,
        require_forces=config.w2 > 0.0,
        require_stress=config.w3 > 0.0,
    )
    df_train = df.loc[df["fold"] != config.val_fold].reset_index(drop=True)
    df_val = df.loc[df["fold"] == config.val_fold].reset_index(drop=True)
    train_dataset = StructureDataset(
        df_train,
        r_cut=config.radial_cutoff,
        a_cut=config.angle_cutoff,
        label_config=label_config,
        cache_graphs=config.cache_graphs,
        cache_size=config.graph_cache_size,
    )
    val_dataset = StructureDataset(
        df_val,
        r_cut=config.radial_cutoff,
        a_cut=config.angle_cutoff,
        label_config=label_config,
        cache_graphs=config.cache_graphs,
        cache_size=config.graph_cache_size,
    )
    return train_dataset, val_dataset


def apply_fitted_element_refs(config: TrainingConfig, train_dataset) -> None:
    if not config.fit_element_refs:
        return
    if config.element_refs is not None:
        raise ValueError("Set either element_refs or fit_element_refs, not both.")
    print("Fitting element_refs from the training dataset.")
    config.element_refs = fit_element_refs_from_samples(
        train_dataset,
        max_atomic_number=config.max_atomic_number,
        ridge=config.element_ref_ridge,
    )


def build_loaders(
    config: TrainingConfig,
    train_dataset,
    val_dataset,
) -> Tuple[DataLoader, DataLoader]:
    pin_memory = torch.device(config.device).type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
        collate_fn=collate_graph_samples,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_graph_samples,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def build_model(config: TrainingConfig) -> torch.nn.Module:
    model = tModLodaer_t(config) if config.transformer_activate else GPTFFNet(config.to_model_config())
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


def resolve_checkpoint_path(config: TrainingConfig, output_dir: Path) -> Path:
    if config.checkpoint_path:
        return Path(config.checkpoint_path)
    return output_dir / "curr_checkpoint.pth"


def load_training_checkpoint(
    checkpoint_path: Union[str, Path],
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    *,
    device: torch.device | str,
    scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
    scaler: Optional[GradScaler] = None,
) -> LoadedCheckpoint:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location=torch.device(device))
    model.load_state_dict(state["state_dict"])
    optimizer.load_state_dict(state["optimizer"])

    epoch = int(state.get("epoch", 0))
    if scheduler is not None:
        if "scheduler" in state:
            scheduler.load_state_dict(state["scheduler"])
        else:
            scheduler.step(epoch)
    if scaler is not None and "scaler" in state:
        scaler.load_state_dict(state["scaler"])

    return LoadedCheckpoint(
        epoch=epoch,
        best_mae_error=float(state.get("best_mae_error", 1e12)),
        training_config=state.get("training_config", state.get("cfg")),
        model_config=state.get("model_config"),
        label_config=state.get("label_config"),
    )


def use_cuda_amp(config: TrainingConfig) -> bool:
    return torch.device(config.device).type == "cuda"


def mae(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(target - prediction))


def loss_weight_active(weight: float) -> bool:
    return float(weight) > 0.0


def validate_required_labels(batch, config: TrainingConfig) -> None:
    active_terms = [
        loss_weight_active(config.w1),
        loss_weight_active(config.w2),
        loss_weight_active(config.w3),
    ]
    if not any(active_terms):
        raise ValueError("At least one loss weight must be positive.")
    if active_terms[0] and batch.energy is None:
        raise ValueError("energy labels are required when weight_energy > 0.")
    if active_terms[1] and batch.forces is None:
        raise ValueError("force labels are required when weight_force > 0.")
    if active_terms[2] and batch.stress is None:
        raise ValueError("stress labels are required when weight_stress > 0.")


def compute_batch_loss(
    model: torch.nn.Module,
    batch,
    criterion: nn.Module,
    config: TrainingConfig,
    *,
    create_graph: bool,
) -> BatchLoss:
    validate_required_labels(batch, config)
    compute_forces = loss_weight_active(config.w2)
    compute_stress = loss_weight_active(config.w3)

    energy_pred, force_pred, stress_pred = predict_energy_forces_stress(
        model,
        batch,
        unit_trans=config.unit_trans,
        create_graph=create_graph,
        compute_forces=compute_forces,
        compute_stress=compute_stress,
    )
    loss = energy_pred.new_zeros(())
    energy_mae = None
    force_mae = None
    stress_mae = None
    force_count = 0
    stress_count = 0

    if loss_weight_active(config.w1):
        num_atoms = batch.num_atoms.to(dtype=energy_pred.dtype)
        energy_per_atom = energy_pred.view(-1) / num_atoms
        target_energy = batch.energy.view(-1) / num_atoms
        energy_loss = criterion(energy_per_atom, target_energy)
        loss = loss + config.w1 * energy_loss
        energy_mae = mae(energy_per_atom.detach(), target_energy.detach())

    if compute_forces:
        force_loss = criterion(force_pred.reshape(-1), batch.forces.reshape(-1))
        loss = loss + config.w2 * force_loss
        force_mae = mae(force_pred.detach().reshape(-1), batch.forces.detach().reshape(-1))
        force_count = int(batch.forces.numel())

    if compute_stress:
        stress_loss = criterion(stress_pred.reshape(-1), batch.stress.reshape(-1))
        loss = loss + config.w3 * stress_loss
        stress_mae = mae(stress_pred.detach().reshape(-1), batch.stress.detach().reshape(-1))
        stress_count = int(batch.stress.numel())

    return BatchLoss(
        loss=loss,
        energy_mae=energy_mae,
        force_mae=force_mae,
        stress_mae=stress_mae,
        batch_size=int(batch.num_atoms.shape[0]),
        force_count=force_count,
        stress_count=stress_count,
    )


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


def save_checkpoint(
    output_dir: Path,
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    config: TrainingConfig,
    *,
    epoch: int,
    best_mae_error: float,
    is_best: bool,
    scheduler: Optional[CosineAnnealingWarmupRestarts] = None,
    scaler: Optional[GradScaler] = None,
) -> None:
    model_state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "best_mae_error": best_mae_error,
        "optimizer": optimizer.state_dict(),
        "cfg": config.checkpoint_dict(),
        "training_config": config.checkpoint_dict(),
        "label_config": asdict(config.to_label_config()),
        "model_name": "tModLodaer_t" if config.transformer_activate else "GPTFFNet",
        "model_config": config.to_model_config().to_dict(),
    }
    if scheduler is not None:
        model_state["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        model_state["scaler"] = scaler.state_dict()
    current_path = output_dir / "curr_checkpoint.pth"
    torch.save(model_state, current_path)
    if is_best:
        shutil.copyfile(current_path, output_dir / "best_checkpoint.pth")


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
