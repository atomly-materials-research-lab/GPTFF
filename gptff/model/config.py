from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class AtomAttentionConfig:
    enabled: bool = True
    num_heads: int = 4
    dropout: float = 0.0
    use_ffn: bool = True
    ffn_hidden_dim: int | None = None

    def __post_init__(self) -> None:
        if self.num_heads <= 0:
            raise ValueError("atom_attention.num_heads must be positive.")
        if self.dropout < 0 or self.dropout >= 1:
            raise ValueError("atom_attention.dropout must be in the range [0, 1).")
        if self.ffn_hidden_dim is not None and self.ffn_hidden_dim <= 0:
            raise ValueError("atom_attention.ffn_hidden_dim must be positive or null.")

    @classmethod
    def from_dict(cls, raw_config: Any) -> AtomAttentionConfig:
        if isinstance(raw_config, cls):
            return raw_config
        if raw_config is None:
            return cls()
        if isinstance(raw_config, bool):
            return cls(enabled=raw_config)
        if not isinstance(raw_config, Mapping):
            raise TypeError("atom_attention must be a mapping, boolean, or null.")
        return cls(
            enabled=bool(raw_config.get("enabled", True)),
            num_heads=int(raw_config.get("num_heads", 4)),
            dropout=float(raw_config.get("dropout", 0.0)),
            use_ffn=bool(raw_config.get("use_ffn", True)),
            ffn_hidden_dim=_optional_int(raw_config.get("ffn_hidden_dim", None)),
        )


@dataclass(frozen=True)
class GPTFFConfig:
    atom_feature_dim: int
    edge_feature_dim: int
    num_interaction_blocks: int
    num_radial: int = 16
    num_angular: int = 6
    radial_cutoff: float = 5.0
    angle_cutoff: float = 3.5
    cutoff_coeff: int = 5
    max_atomic_number: int = 94
    element_refs: Any = None
    num_readout_layers: int = 3
    readout_atom_norm: bool = True
    interaction_dropout: float = 0.0
    atom_attention: AtomAttentionConfig = field(default_factory=AtomAttentionConfig)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "atom_attention",
            AtomAttentionConfig.from_dict(self.atom_attention),
        )
        if self.atom_feature_dim <= 0:
            raise ValueError("atom_feature_dim must be positive.")
        if self.edge_feature_dim <= 0:
            raise ValueError("edge_feature_dim must be positive.")
        if self.num_interaction_blocks <= 0:
            raise ValueError("num_interaction_blocks must be positive.")
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
        if self.num_readout_layers <= 0:
            raise ValueError("num_readout_layers must be positive.")
        if self.interaction_dropout < 0 or self.interaction_dropout >= 1:
            raise ValueError("interaction_dropout must be in the range [0, 1).")
        if (
            self.atom_attention.enabled
            and self.atom_feature_dim % self.atom_attention.num_heads != 0
        ):
            raise ValueError("atom_feature_dim must be divisible by atom_attention.num_heads.")

    @classmethod
    def from_dict(cls, raw_config: Mapping[str, Any]) -> GPTFFConfig:
        return cls(
            atom_feature_dim=int(_config_value(raw_config, "atom_feature_dim", "node_feature_len")),
            edge_feature_dim=int(_config_value(raw_config, "edge_feature_dim", "edge_feature_len")),
            num_interaction_blocks=int(
                _config_value(
                    raw_config,
                    "num_interaction_blocks",
                    "n_layers",
                )
            ),
            num_radial=int(raw_config.get("num_radial", 16)),
            num_angular=int(raw_config.get("num_angular", 6)),
            radial_cutoff=float(raw_config.get("radial_cutoff", 5.0)),
            angle_cutoff=float(raw_config.get("angle_cutoff", 3.5)),
            cutoff_coeff=int(raw_config.get("cutoff_coeff", 5)),
            max_atomic_number=int(raw_config.get("max_atomic_number", 94)),
            element_refs=raw_config.get("element_refs", None),
            num_readout_layers=int(
                _config_value(
                    raw_config,
                    "num_readout_layers",
                    "n_readout_layers",
                    default=3,
                )
            ),
            readout_atom_norm=bool(
                _config_value(
                    raw_config,
                    "readout_atom_norm",
                    "final_atom_norm",
                    default=True,
                )
            ),
            interaction_dropout=float(raw_config.get("interaction_dropout", 0.0)),
            atom_attention=AtomAttentionConfig.from_dict(raw_config.get("atom_attention", None)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _config_value(
    config: Mapping[str, Any],
    key: str,
    legacy_key: str,
    *,
    default: Any = None,
) -> Any:
    if key in config:
        return config[key]
    if legacy_key in config:
        return config[legacy_key]
    if default is not None:
        return default
    raise KeyError(key)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
