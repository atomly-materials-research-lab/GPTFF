from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


def _as_hidden_dims(hidden_dims: int | Sequence[int] | None) -> tuple[int, ...]:
    if hidden_dims is None:
        return ()
    if isinstance(hidden_dims, int):
        return (hidden_dims,)
    return tuple(int(dim) for dim in hidden_dims)


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: int | Sequence[int] | None = None,
        *,
        dropout: float = 0.0,
        activate_output: bool = False,
        zero_init_output: bool = False,
    ):
        super().__init__()
        if dropout < 0 or dropout >= 1:
            raise ValueError("dropout must be in the range [0, 1).")

        layers = []
        prev_dim = int(input_dim)
        for hidden_dim in _as_hidden_dims(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, output_dim)
        self.output_activation = nn.SiLU() if activate_output else nn.Identity()
        self.output_dropout = nn.Dropout(dropout)
        if zero_init_output:
            nn.init.zeros_(self.output_layer.weight)
            nn.init.zeros_(self.output_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        x = self.output_activation(x)
        return self.output_dropout(x)


class GatedMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        dropout: float = 0.0,
        zero_init_output: bool = False,
    ):
        super().__init__()
        if dropout < 0 or dropout >= 1:
            raise ValueError("dropout must be in the range [0, 1).")

        self.value = nn.Linear(input_dim, output_dim)
        self.gate = nn.Linear(input_dim, output_dim)
        self.activation = nn.SiLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        if zero_init_output:
            nn.init.zeros_(self.value.weight)
            nn.init.zeros_(self.value.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.activation(self.value(x)) * self.sigmoid(self.gate(x)))
