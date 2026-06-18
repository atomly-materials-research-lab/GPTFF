from __future__ import annotations

import csv
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, TextIO

HISTORY_FIELDS = (
    "epoch",
    "lr",
    "train_loss",
    "train_energy_mae",
    "train_force_mae",
    "train_stress_mae",
    "train_skipped_batches",
    "val_loss",
    "val_energy_mae",
    "val_force_mae",
    "val_stress_mae",
    "val_skipped_batches",
)


@dataclass(frozen=True)
class EpochLogRecord:
    epoch: int
    lr: float
    train_loss: float
    train_energy_mae: float
    train_force_mae: float
    train_stress_mae: float
    train_skipped_batches: int
    val_loss: float
    val_energy_mae: float
    val_force_mae: float
    val_stress_mae: float
    val_skipped_batches: int

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


class TrainingLogger(Protocol):
    def log_epoch(self, record: EpochLogRecord) -> None: ...

    def close(self) -> None: ...


class CSVLogger:
    def __init__(self, output_dir: str | Path, filename: str = "history.csv") -> None:
        self.path = Path(output_dir) / filename

    def log_epoch(self, record: EpochLogRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.path.exists()
        with open(self.path, "a", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=HISTORY_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(record.as_dict())

    def close(self) -> None:
        return None


class ConsoleLogger:
    def __init__(self, stream: TextIO | None = None) -> None:
        self.stream = stream if stream is not None else sys.stdout

    def log_epoch(self, record: EpochLogRecord) -> None:
        print(
            "Epoch "
            f"{record.epoch}: "
            f"train_loss={_format_value(record.train_loss, 5)} "
            f"val_loss={_format_value(record.val_loss, 5)}\n"
            f"  train_MAE(e)={_format_value(record.train_energy_mae, 5)} "
            f"train_MAE(f)={_format_value(record.train_force_mae, 5)} "
            f"train_MAE(s)={_format_value(record.train_stress_mae, 3)}\n"
            f"  val_MAE(e)={_format_value(record.val_energy_mae, 5)} "
            f"val_MAE(f)={_format_value(record.val_force_mae, 5)} "
            f"val_MAE(s)={_format_value(record.val_stress_mae, 3)}",
            file=self.stream,
            flush=True,
        )

    def close(self) -> None:
        return None


class WandBLogger:
    def __init__(self, config, run_config: Mapping[str, Any]) -> None:
        self._wandb = None
        self._run = None
        if not config.enabled:
            return

        try:
            import wandb
        except ImportError:
            print(
                "WandB logging is enabled but the 'wandb' package is not installed; "
                "continuing without wandb.",
                file=sys.stderr,
                flush=True,
            )
            return

        init_kwargs = dict(config.init_kwargs)
        init_kwargs.setdefault("project", config.project)
        init_kwargs.setdefault("config", dict(run_config))
        for key in ("entity", "name", "group", "notes", "mode", "job_type"):
            value = getattr(config, key)
            if value is not None:
                init_kwargs.setdefault(key, value)
        if config.tags:
            init_kwargs.setdefault("tags", list(config.tags))

        try:
            self._run = wandb.init(**init_kwargs)
        except Exception as exc:  # pragma: no cover - depends on user wandb setup.
            print(
                f"WandB logging could not be initialized ({exc}); continuing without wandb.",
                file=sys.stderr,
                flush=True,
            )
            return
        self._wandb = wandb

    def log_epoch(self, record: EpochLogRecord) -> None:
        if self._wandb is None:
            return
        self._wandb.log(record.as_dict(), step=record.epoch)

    def close(self) -> None:
        if self._run is not None:
            self._run.finish()


class CompositeLogger:
    def __init__(self, loggers: Sequence[TrainingLogger]) -> None:
        self.loggers = tuple(loggers)

    def log_epoch(self, record: EpochLogRecord) -> None:
        for logger in self.loggers:
            logger.log_epoch(record)

    def close(self) -> None:
        for logger in self.loggers:
            logger.close()


def _format_value(value: float, precision: int) -> str:
    if math.isnan(value):
        return "n/a"
    return f"{value:.{precision}f}"
