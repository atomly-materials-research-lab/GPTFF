from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from gptff.trainer._utils import format_value


@dataclass(frozen=True)
class EvaluationRecord:
    split: str
    checkpoint: str | None
    epoch: int | None
    loss: float
    energy_mae: float
    force_mae: float
    stress_mae: float
    skipped_batches: int

    def as_dict(self) -> dict[str, str | float | int | None]:
        return asdict(self)


def build_evaluation_record(
    *,
    split: str,
    checkpoint: str | None,
    epoch: int | None,
    loss: float,
    energy_mae: float,
    force_mae: float,
    stress_mae: float,
    skipped_batches: int,
) -> EvaluationRecord:
    return EvaluationRecord(
        split=split,
        checkpoint=checkpoint,
        epoch=epoch,
        loss=float(loss),
        energy_mae=float(energy_mae),
        force_mae=float(force_mae),
        stress_mae=float(stress_mae),
        skipped_batches=int(skipped_batches),
    )


def write_evaluation_record(
    output_dir: str | Path,
    record: EvaluationRecord,
    *,
    filename: str,
) -> Path:
    path = Path(output_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(_json_safe(record.as_dict()), fp, indent=2, sort_keys=True)
        fp.write("\n")
    return path


def format_evaluation_record(record: EvaluationRecord) -> str:
    return (
        f"{record.split} "
        f"checkpoint={record.checkpoint or 'current'} "
        f"MAE(e)={format_value(record.energy_mae, 5)} "
        f"MAE(f)={format_value(record.force_mae, 5)} "
        f"MAE(s)={format_value(record.stress_mae, 3)}"
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    return value
