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
)


@dataclass
class TrainingConfig:
    val_fold: int
    num_train_steps: int
    warmup_steps: int
    batch_size: int
    device: str
    data_path: str
    data_file: str
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
    unit_trans: float = 160.21766208
    output_dir: str = "."
    min_lr: float = 5e-6
    grad_clip_norm: float = 10.0
    max_loss_skip: Optional[float] = 10.0
    resume: bool = False

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
            unit_trans=float(training.get("unit_trans", 160.21766208)),
            output_dir=str(training.get("output_dir", raw_config.get("output_dir", "."))),
            min_lr=float(training.get("min_lr", 5e-6)),
            grad_clip_norm=float(training.get("grad_clip_norm", 10.0)),
            max_loss_skip=training.get("max_loss_skip", 10.0),
            resume=bool(training.get("resume", False)),
        )

    def checkpoint_dict(self) -> Dict[str, Any]:
        return asdict(self)

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
        )


@dataclass
class BatchLoss:
    loss: torch.Tensor
    energy_mae: torch.Tensor
    force_mae: torch.Tensor
    stress_mae: torch.Tensor
    batch_size: int


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
            "loss": f"{self.loss.val:.5f} ({self.loss.avg:.5f})",
            "MAE(e)": f"{self.energy_mae.val:.5f} ({self.energy_mae.avg:.5f})",
            "MAE(f)": f"{self.force_mae.val:.5f} ({self.force_mae.avg:.5f})",
            "MAE(s)": f"{self.stress_mae.val:.3f} ({self.stress_mae.avg:.3f})",
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
    df_train = df.loc[df["fold"] != config.val_fold].reset_index(drop=True)
    df_val = df.loc[df["fold"] == config.val_fold].reset_index(drop=True)
    train_dataset = StructureDataset(
        df_train,
        r_cut=config.radial_cutoff,
        a_cut=config.angle_cutoff,
    )
    val_dataset = StructureDataset(
        df_val,
        r_cut=config.radial_cutoff,
        a_cut=config.angle_cutoff,
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
    scheduler.step(config.start_epoch)
    return scheduler


def use_cuda_amp(config: TrainingConfig) -> bool:
    return torch.device(config.device).type == "cuda"


def mae(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(target - prediction))


def compute_batch_loss(
    model: torch.nn.Module,
    batch,
    criterion: nn.Module,
    config: TrainingConfig,
    *,
    create_graph: bool,
) -> BatchLoss:
    energy_pred, force_pred, stress_pred = predict_energy_forces_stress(
        model,
        batch,
        unit_trans=config.unit_trans,
        create_graph=create_graph,
    )
    num_atoms = batch.num_atoms.to(dtype=energy_pred.dtype)
    energy_pred = energy_pred.view(-1) / num_atoms
    target_energy = batch.energy.view(-1) / num_atoms

    energy_loss = criterion(energy_pred, target_energy)
    force_loss = criterion(force_pred.reshape(-1), batch.forces.reshape(-1))
    stress_loss = criterion(stress_pred.reshape(-1), batch.stress.reshape(-1))
    loss = config.w1 * energy_loss + config.w2 * force_loss + config.w3 * stress_loss

    return BatchLoss(
        loss=loss,
        energy_mae=mae(energy_pred.detach(), target_energy.detach()),
        force_mae=mae(force_pred.detach().reshape(-1), batch.forces.detach().reshape(-1)),
        stress_mae=mae(stress_pred.detach().reshape(-1), batch.stress.detach().reshape(-1)),
        batch_size=int(target_energy.shape[0]),
    )


def should_skip_batch(batch_loss: BatchLoss, config: TrainingConfig) -> bool:
    if not torch.isfinite(batch_loss.loss):
        return True
    if config.max_loss_skip is not None and batch_loss.loss.detach().item() > config.max_loss_skip:
        return True
    return False


def update_metrics(metrics: EpochMetrics, batch_loss: BatchLoss) -> None:
    metrics.loss.update(batch_loss.loss.detach().cpu().item(), batch_loss.batch_size)
    metrics.energy_mae.update(batch_loss.energy_mae.cpu().item(), batch_loss.batch_size)
    metrics.force_mae.update(batch_loss.force_mae.cpu().item(), batch_loss.batch_size)
    metrics.stress_mae.update(batch_loss.stress_mae.cpu().item(), batch_loss.batch_size * 9)


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
            f"{metrics.energy_mae.avg:.4f} "
            f"{metrics.force_mae.avg:.4f} "
            f"{metrics.stress_mae.avg:.4f}\n"
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
) -> None:
    model_state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "best_mae_error": best_mae_error,
        "optimizer": optimizer.state_dict(),
        "cfg": config.checkpoint_dict(),
        "model_name": "tModLodaer_t" if config.transformer_activate else "GPTFFNet",
        "model_config": config.to_model_config().to_dict(),
    }
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
    scheduler = build_scheduler(optimizer, config)
    scaler = GradScaler(enabled=use_cuda_amp(config))

    best_mae_error = 1e12
    bar_format = "{l_bar}{bar:40}| [{elapsed}<{remaining}{postfix}]"

    for epoch in range(config.start_epoch, config.epochs):
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
            f"val_loss: {val_metrics.loss.avg:.5f} "
            f"val_MAE(e): {val_metrics.energy_mae.avg:.5f}, "
            f"val_MAE(f): {val_metrics.force_mae.avg:.5f}, "
            f"val_MAE(s): {val_metrics.stress_mae.avg:.3f}"
        )
        pbar_train.close()
        pbar_val.close()

        append_validation_history(output_dir, val_metrics)
        scheduler.step()

        is_best = val_metrics.energy_mae.avg < best_mae_error
        best_mae_error = min(val_metrics.energy_mae.avg, best_mae_error)
        save_checkpoint(
            output_dir,
            model,
            optimizer,
            config,
            epoch=epoch + 1,
            best_mae_error=best_mae_error,
            is_best=is_best,
        )

    return best_mae_error


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
    run_training(config)


if __name__ == "__main__":
    main()
