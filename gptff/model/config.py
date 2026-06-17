from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


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
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
