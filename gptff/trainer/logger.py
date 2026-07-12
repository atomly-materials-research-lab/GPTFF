from __future__ import annotations

import csv
import os
import sys
import tempfile
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, TextIO

from gptff.trainer._utils import format_value
from gptff.trainer.evaluation import (
    EvaluationRecord,
    format_evaluation_record,
    write_evaluation_record,
)

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

    def log_evaluation(self, record: EvaluationRecord) -> None: ...

    def close(self) -> None: ...


class NullLogger:
    def log_epoch(self, record: EpochLogRecord) -> None:
        return None

    def log_evaluation(self, record: EvaluationRecord) -> None:
        return None

    def close(self) -> None:
        return None


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

    def log_evaluation(self, record: EvaluationRecord) -> None:
        write_evaluation_record(
            self.path.parent,
            record,
            filename=f"{record.split}_metrics.json",
        )

    def close(self) -> None:
        return None


def reconcile_history_for_resume(history_path: str | Path, saved_epoch: int) -> None:
    path = Path(history_path)
    if not path.exists() or path.stat().st_size == 0:
        warnings.warn(
            "Training history is missing; a new history.csv will be created "
            f"starting after checkpoint epoch {saved_epoch}.",
            RuntimeWarning,
            stacklevel=2,
        )
        _atomic_write_history(path, [])
        return

    with open(path, newline="") as file:
        reader = csv.DictReader(file)
        if tuple(reader.fieldnames or ()) != HISTORY_FIELDS:
            raise ValueError("history.csv is corrupted: header does not match expected fields.")
        rows = list(reader)

    epochs = []
    for row_number, row in enumerate(rows, start=2):
        raw_epoch = row.get("epoch", "")
        try:
            epoch = int(raw_epoch)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"history.csv is corrupted: epoch on row {row_number} is not an integer."
            ) from exc
        epochs.append(epoch)

    if any(current <= previous for previous, current in zip(epochs, epochs[1:])):
        raise ValueError("history.csv is corrupted: epochs must be strictly increasing.")

    kept_rows = [row for row, epoch in zip(rows, epochs) if epoch <= saved_epoch]
    if len(kept_rows) != len(rows):
        _atomic_write_history(path, kept_rows)

    last_epoch = int(kept_rows[-1]["epoch"]) if kept_rows else 0
    if last_epoch < saved_epoch:
        warnings.warn(
            f"history.csv ends at epoch {last_epoch}, but checkpoint is at epoch "
            f"{saved_epoch}; the missing history will not be backfilled.",
            RuntimeWarning,
            stacklevel=2,
        )


def _atomic_write_history(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp.{os.getpid()}.",
        dir=path.parent,
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=HISTORY_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    except Exception:
        try:
            os.close(file_descriptor)
        except OSError:
            pass
        temporary_path.unlink(missing_ok=True)
        raise


class ConsoleLogger:
    def __init__(self, stream: TextIO | None = None) -> None:
        self.stream = stream if stream is not None else sys.stdout

    def log_epoch(self, record: EpochLogRecord) -> None:
        print(
            "Epoch "
            f"{record.epoch}: "
            f"train_MAE(e)={format_value(record.train_energy_mae, 5)} "
            f"train_MAE(f)={format_value(record.train_force_mae, 5)} "
            f"train_MAE(s)={format_value(record.train_stress_mae, 3)} "
            f"val_MAE(e)={format_value(record.val_energy_mae, 5)} "
            f"val_MAE(f)={format_value(record.val_force_mae, 5)} "
            f"val_MAE(s)={format_value(record.val_stress_mae, 3)}",
            file=self.stream,
            flush=True,
        )

    def log_evaluation(self, record: EvaluationRecord) -> None:
        print(format_evaluation_record(record), file=self.stream, flush=True)

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

    def log_evaluation(self, record: EvaluationRecord) -> None:
        if self._wandb is None:
            return
        payload = {
            f"{record.split}_{key}": value
            for key, value in record.as_dict().items()
            if key != "split"
        }
        # Evaluation can describe an earlier checkpoint than the latest logged epoch.
        # Let W&B assign the next step so it does not discard an out-of-order record.
        self._wandb.log(payload)

    def close(self) -> None:
        if self._run is not None:
            self._run.finish()


class CompositeLogger:
    def __init__(self, loggers: Sequence[TrainingLogger]) -> None:
        self.loggers = tuple(loggers)

    def log_epoch(self, record: EpochLogRecord) -> None:
        for logger in self.loggers:
            logger.log_epoch(record)

    def log_evaluation(self, record: EvaluationRecord) -> None:
        for logger in self.loggers:
            logger.log_evaluation(record)

    def close(self) -> None:
        for logger in self.loggers:
            logger.close()
