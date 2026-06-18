from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import yaml

from gptff.model import GPTFFConfig


@dataclass(frozen=True)
class DataConfig:
    dataset_path: Optional[str] = None
    validation_fraction: float = 0.1
    test_fraction: float = 0.0
    split_seed: int = 42
    group_by_material: bool = False
    cache_graphs: bool = False
    graph_cache_size: Optional[int] = None

    def __post_init__(self) -> None:
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between 0 and 1.")
        if not 0.0 <= self.test_fraction < 1.0:
            raise ValueError("test_fraction must be between 0 and 1.")
        if self.validation_fraction + self.test_fraction >= 1.0:
            raise ValueError(
                "validation_fraction + test_fraction must be less than 1."
            )
        if self.graph_cache_size is not None and self.graph_cache_size < 0:
            raise ValueError("graph_cache_size must be non-negative or null.")


@dataclass(frozen=True)
class OptimizerConfig:
    learning_rate: float
    name: str = "AdamW"
    weight_decay: float = 1e-2
    scheduler: str = "CosLR"
    scheduler_params: dict[str, Any] = field(default_factory=lambda: {"decay_fraction": 1e-2})


@dataclass(frozen=True)
class TrainingLoopConfig:
    epochs: int
    batch_size: int
    num_workers: int
    device: str
    amp: bool = False
    output_dir: str = "."
    grad_clip_norm: float = 10.0
    seed: int = 42
    deterministic: bool = True


@dataclass(frozen=True)
class LossConfig:
    energy_loss_weight: float
    force_loss_weight: float
    stress_loss_weight: float


@dataclass(frozen=True)
class ElementReferenceConfig:
    source: Any = None
    ridge: float = 0.0

    def __post_init__(self) -> None:
        if self.ridge < 0:
            raise ValueError("element reference ridge must be non-negative.")

    @classmethod
    def from_dict(
        cls,
        raw_config: Any,
        *,
        base_dir: Path | None = None,
    ) -> "ElementReferenceConfig":
        if raw_config is None:
            return cls()
        if not isinstance(raw_config, Mapping):
            return cls(source=_resolve_element_reference_source(raw_config, base_dir))
        return cls(
            source=_resolve_element_reference_source(raw_config.get("source", None), base_dir),
            ridge=float(raw_config.get("ridge", 0.0)),
        )

    @property
    def fit_from_training_data(self) -> bool:
        return isinstance(self.source, str) and self.source.lower() == "fit"

    def model_element_refs(self) -> Any:
        if self.source is None:
            return None
        if isinstance(self.source, str) and self.source.lower() in {"none", "null"}:
            return None
        if self.fit_from_training_data:
            return None
        return self.source


@dataclass
class TrainingConfig:
    data: DataConfig
    model: GPTFFConfig
    optimizer: OptimizerConfig
    training: TrainingLoopConfig
    loss: LossConfig
    element_references: ElementReferenceConfig = field(default_factory=ElementReferenceConfig)

    @classmethod
    def from_dict(
        cls,
        raw_config: Mapping[str, Any],
        *,
        base_dir: Union[str, Path, None] = None,
    ) -> "TrainingConfig":
        training = raw_config.get("training", {})
        data = raw_config["data"]
        model = raw_config.get("model", training)
        optimizer = raw_config.get("optimizer", training)
        loss = raw_config.get("loss", training)
        epochs = int(_config_value(training, "epochs", "epochs"))
        optimizer_name = str(optimizer.get("name", optimizer.get("optimizer", "AdamW")))
        default_weight_decay = 1e-2 if optimizer_name.lower() == "adamw" else 0.0
        config_base_dir = None if base_dir is None else Path(base_dir)

        element_references = ElementReferenceConfig.from_dict(
            raw_config.get("element_references", None),
            base_dir=config_base_dir,
        )
        model_config = replace(
            GPTFFConfig.from_dict(model),
            element_refs=element_references.model_element_refs(),
        )

        return cls(
            data=DataConfig(
                dataset_path=_optional_str(data.get("dataset_path")),
                validation_fraction=float(data.get("validation_fraction", 0.1)),
                test_fraction=float(data.get("test_fraction", 0.0)),
                split_seed=int(data.get("split_seed", 42)),
                group_by_material=bool(data.get("group_by_material", False)),
                cache_graphs=bool(data.get("cache_graphs", False)),
                graph_cache_size=_optional_int(data.get("graph_cache_size", None)),
            ),
            model=model_config,
            optimizer=OptimizerConfig(
                name=optimizer_name,
                learning_rate=float(_config_value(optimizer, "learning_rate", "lr")),
                weight_decay=float(optimizer.get("weight_decay", default_weight_decay)),
                scheduler=str(optimizer.get("scheduler", "CosLR")),
                scheduler_params=_scheduler_params(optimizer, epochs),
            ),
            training=TrainingLoopConfig(
                epochs=epochs,
                batch_size=int(training["batch_size"]),
                num_workers=int(_config_value(training, "num_workers", "workers")),
                device=str(training["device"]),
                amp=bool(training.get("amp", False)),
                output_dir=str(training.get("output_dir", raw_config.get("output_dir", "."))),
                grad_clip_norm=float(training.get("grad_clip_norm", 10.0)),
                seed=int(training.get("seed", 42)),
                deterministic=bool(training.get("deterministic", True)),
            ),
            loss=LossConfig(
                energy_loss_weight=float(_config_value(loss, "energy_loss_weight", "weight_energy")),
                force_loss_weight=float(_config_value(loss, "force_loss_weight", "weight_force")),
                stress_loss_weight=float(_config_value(loss, "stress_loss_weight", "weight_stress")),
            ),
            element_references=element_references,
        )

    def checkpoint_dict(self) -> dict[str, Any]:
        return {
            "data": asdict(self.data),
            "model": self.model.to_dict(),
            "optimizer": asdict(self.optimizer),
            "training": asdict(self.training),
            "loss": asdict(self.loss),
            "element_references": asdict(self.element_references),
        }

    def to_model_config(self) -> GPTFFConfig:
        return self.model

    @property
    def dataset_path(self) -> Optional[str]:
        return self.data.dataset_path

    @property
    def validation_fraction(self) -> float:
        return self.data.validation_fraction

    @property
    def test_fraction(self) -> float:
        return self.data.test_fraction

    @property
    def split_seed(self) -> int:
        return self.data.split_seed

    @property
    def group_by_material(self) -> bool:
        return self.data.group_by_material

    @property
    def cache_graphs(self) -> bool:
        return self.data.cache_graphs

    @property
    def graph_cache_size(self) -> Optional[int]:
        return self.data.graph_cache_size

    @property
    def lr(self) -> float:
        return self.optimizer.learning_rate

    @property
    def weight_decay(self) -> float:
        return self.optimizer.weight_decay

    @property
    def optimizer_name(self) -> str:
        return self.optimizer.name

    @property
    def scheduler(self) -> str:
        return self.optimizer.scheduler

    @property
    def scheduler_params(self) -> dict[str, Any]:
        return dict(self.optimizer.scheduler_params)

    @property
    def epochs(self) -> int:
        return self.training.epochs

    @property
    def batch_size(self) -> int:
        return self.training.batch_size

    @property
    def num_workers(self) -> int:
        return self.training.num_workers

    @property
    def device(self) -> str:
        return self.training.device

    @device.setter
    def device(self, value: str) -> None:
        self.training = replace(self.training, device=value)

    @property
    def amp(self) -> bool:
        return self.training.amp

    @amp.setter
    def amp(self, value: bool) -> None:
        self.training = replace(self.training, amp=bool(value))

    @property
    def output_dir(self) -> str:
        return self.training.output_dir

    @property
    def grad_clip_norm(self) -> float:
        return self.training.grad_clip_norm

    @property
    def seed(self) -> int:
        return self.training.seed

    @property
    def deterministic(self) -> bool:
        return self.training.deterministic

    @property
    def energy_loss_weight(self) -> float:
        return self.loss.energy_loss_weight

    @energy_loss_weight.setter
    def energy_loss_weight(self, value: float) -> None:
        self.loss = replace(self.loss, energy_loss_weight=float(value))

    @property
    def force_loss_weight(self) -> float:
        return self.loss.force_loss_weight

    @force_loss_weight.setter
    def force_loss_weight(self, value: float) -> None:
        self.loss = replace(self.loss, force_loss_weight=float(value))

    @property
    def stress_loss_weight(self) -> float:
        return self.loss.stress_loss_weight

    @stress_loss_weight.setter
    def stress_loss_weight(self, value: float) -> None:
        self.loss = replace(self.loss, stress_loss_weight=float(value))

    @property
    def atom_feature_dim(self) -> int:
        return self.model.atom_feature_dim

    @property
    def edge_feature_dim(self) -> int:
        return self.model.edge_feature_dim

    @property
    def num_interaction_blocks(self) -> int:
        return self.model.num_interaction_blocks

    @property
    def num_radial(self) -> int:
        return self.model.num_radial

    @property
    def num_angular(self) -> int:
        return self.model.num_angular

    @property
    def radial_cutoff(self) -> float:
        return self.model.radial_cutoff

    @property
    def angle_cutoff(self) -> float:
        return self.model.angle_cutoff

    @property
    def cutoff_coeff(self) -> int:
        return self.model.cutoff_coeff

    @property
    def max_atomic_number(self) -> int:
        return self.model.max_atomic_number

    @property
    def element_refs(self) -> Any:
        return self.model.element_refs

    @element_refs.setter
    def element_refs(self, value: Any) -> None:
        self.model = replace(self.model, element_refs=value)

    @property
    def num_readout_layers(self) -> int:
        return self.model.num_readout_layers

    @property
    def readout_atom_norm(self) -> bool:
        return self.model.readout_atom_norm

    @property
    def interaction_dropout(self) -> float:
        return self.model.interaction_dropout

    @property
    def node_feature_len(self) -> int:
        return self.atom_feature_dim

    @property
    def edge_feature_len(self) -> int:
        return self.edge_feature_dim

    @property
    def n_layers(self) -> int:
        return self.num_interaction_blocks

    @property
    def n_readout_layers(self) -> int:
        return self.num_readout_layers


def load_config(config_file: Union[str, Path]) -> TrainingConfig:
    config_path = Path(config_file)
    with open(config_path, "r", encoding="utf-8") as fp:
        raw_config = yaml.safe_load(fp)
    if not isinstance(raw_config, Mapping):
        raise ValueError("Training config must contain a top-level mapping")
    return TrainingConfig.from_dict(raw_config, base_dir=config_path.parent)


def _resolve_element_reference_source(
    source: Any,
    base_dir: Path | None,
) -> Any:
    if source is None:
        return None
    if isinstance(source, Mapping):
        _validate_element_reference_mapping_keys(source)
        return source
    if not isinstance(source, str):
        return source

    normalized = source.strip()
    lower = normalized.lower()
    if lower in {"", "none", "null"}:
        return None
    if lower == "fit":
        return "fit"

    path = Path(normalized).expanduser()
    if base_dir is not None and not path.is_absolute():
        path = base_dir / path
    if path.exists() or path.suffix.lower() in {".yaml", ".yml", ".json"}:
        if not path.exists():
            raise FileNotFoundError(f"Element reference file not found: {path}")
        return _load_element_reference_file(path)

    return normalized


def _load_element_reference_file(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fp:
        refs = yaml.safe_load(fp)
    if refs is None:
        raise ValueError(f"Element reference file is empty: {path}")
    if isinstance(refs, Mapping):
        _validate_element_reference_mapping_keys(refs)
        return dict(refs)
    if isinstance(refs, list):
        return refs
    raise ValueError(
        "Element reference file must contain a mapping of atomic numbers to "
        "energies or a list of reference energies."
    )


def _validate_element_reference_mapping_keys(refs: Mapping[Any, Any]) -> None:
    for key in refs:
        if isinstance(key, bool):
            raise ValueError("Element reference mapping keys must be atomic numbers.")
        if isinstance(key, int):
            atomic_number = key
        elif isinstance(key, str) and key.strip().isdigit():
            atomic_number = int(key)
        else:
            raise ValueError(
                "Element reference mapping keys must be atomic numbers, "
                f"got {key!r}."
            )
        if atomic_number < 1:
            raise ValueError(
                "Element reference mapping keys must be positive atomic numbers, "
                f"got {key!r}."
            )


def _optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    return int(value)


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _scheduler_params(optimizer: Mapping[str, Any], epochs: int) -> dict[str, Any]:
    if "scheduler_params" in optimizer:
        return dict(optimizer["scheduler_params"])

    params: dict[str, Any] = {}
    learning_rate = float(_config_value(optimizer, "learning_rate", "lr"))

    if "min_learning_rate" in optimizer or "min_lr" in optimizer:
        min_learning_rate = float(_config_value(
            optimizer,
            "min_learning_rate",
            "min_lr",
        ))
        params["decay_fraction"] = min_learning_rate / learning_rate

    if "lr_cycle_epochs" in optimizer or "num_train_steps" in optimizer:
        params["T_max"] = int(_config_value(
            optimizer,
            "lr_cycle_epochs",
            "num_train_steps",
            default=10 * epochs,
        ))

    return params or {"decay_fraction": 1e-2}


def _config_value(
    config: Mapping[str, Any],
    key: str,
    legacy_key: str,
    *,
    fallback: Mapping[str, Any] | None = None,
    default: Any = None,
) -> Any:
    if key in config:
        return config[key]
    if legacy_key in config:
        return config[legacy_key]
    if fallback is not None:
        if key in fallback:
            return fallback[key]
        if legacy_key in fallback:
            return fallback[legacy_key]
    if default is not None:
        return default
    raise KeyError(key)
