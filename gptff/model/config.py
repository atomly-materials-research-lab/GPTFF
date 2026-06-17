from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from gptff.model.aggregation import validate_aggregation_norm


@dataclass(frozen=True)
class GPTFFNetConfig:
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
    n_readout_layers: int = 3
    readout_zero_init: bool = True
    interaction_dropout: float = 0.0
    residual_scale: float = 1.0
    residual_zero_init: bool = True
    aggregation_norm: str = "sqrt"

    def __post_init__(self) -> None:
        if self.node_feature_len <= 0:
            raise ValueError("node_feature_len must be positive.")
        if self.edge_feature_len <= 0:
            raise ValueError("edge_feature_len must be positive.")
        if self.n_layers <= 0:
            raise ValueError("n_layers must be positive.")
        if self.num_radial <= 0:
            raise ValueError("num_radial must be positive.")
        if self.num_angular <= 0:
            raise ValueError("num_angular must be positive.")
        if self.radial_cutoff <= 0:
            raise ValueError("radial_cutoff must be positive.")
        if self.angle_cutoff <= 0:
            raise ValueError("angle_cutoff must be positive.")
        if self.cutoff_coeff <= 0:
            raise ValueError("cutoff_coeff must be positive.")
        if self.max_atomic_number <= 0:
            raise ValueError("max_atomic_number must be positive.")
        if self.n_readout_layers <= 0:
            raise ValueError("n_readout_layers must be positive.")
        if self.interaction_dropout < 0 or self.interaction_dropout >= 1:
            raise ValueError("interaction_dropout must be in the range [0, 1).")
        if self.residual_scale < 0:
            raise ValueError("residual_scale must be non-negative.")
        object.__setattr__(
            self,
            "aggregation_norm",
            validate_aggregation_norm(self.aggregation_norm),
        )

    @classmethod
    def from_dict(cls, raw_config: Mapping[str, Any]) -> "GPTFFNetConfig":
        return cls(
            node_feature_len=int(raw_config["node_feature_len"]),
            edge_feature_len=int(raw_config["edge_feature_len"]),
            n_layers=int(raw_config["n_layers"]),
            num_radial=int(raw_config.get("num_radial", 16)),
            num_angular=int(raw_config.get("num_angular", 4)),
            radial_cutoff=float(raw_config.get("radial_cutoff", 5.0)),
            angle_cutoff=float(raw_config.get("angle_cutoff", 3.5)),
            cutoff_coeff=int(raw_config.get("cutoff_coeff", 5)),
            max_atomic_number=int(raw_config.get("max_atomic_number", 94)),
            element_refs=raw_config.get("element_refs", None),
            n_readout_layers=int(raw_config.get("n_readout_layers", 3)),
            readout_zero_init=bool(raw_config.get("readout_zero_init", True)),
            interaction_dropout=float(raw_config.get("interaction_dropout", 0.0)),
            residual_scale=float(raw_config.get("residual_scale", 1.0)),
            residual_zero_init=bool(raw_config.get("residual_zero_init", True)),
            aggregation_norm=str(raw_config.get("aggregation_norm", "sqrt")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
