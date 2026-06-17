from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

from gptff.model import GPTFFNetConfig
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


def load_config(config_file: Union[str, Path]) -> TrainingConfig:
    with open(config_file, "r") as fp:
        raw_config = json.load(fp)
    return TrainingConfig.from_dict(raw_config)


def _optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    return int(value)
