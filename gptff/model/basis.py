from __future__ import annotations

import math

import torch
from torch import nn


class PolynomialCutoff(nn.Module):
    def __init__(self, cutoff: float, cutoff_coeff: int = 5):
        super().__init__()
        if cutoff <= 0:
            raise ValueError("cutoff must be positive.")
        if cutoff_coeff < 0:
            raise ValueError("cutoff_coeff must be non-negative.")

        self.cutoff = float(cutoff)
        self.cutoff_coeff = int(cutoff_coeff)
        p = self.cutoff_coeff
        self.a = -(p + 1) * (p + 2) / 2
        self.b = p * (p + 2)
        self.c = -p * (p + 1) / 2

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        if self.cutoff_coeff == 0:
            return torch.ones_like(distances)

        scaled = distances / self.cutoff
        envelope = (
            1
            + self.a * scaled.pow(self.cutoff_coeff)
            + self.b * scaled.pow(self.cutoff_coeff + 1)
            + self.c * scaled.pow(self.cutoff_coeff + 2)
        )
        return torch.where(scaled < 1, envelope, torch.zeros_like(envelope))


class RadialBesselBasis(nn.Module):
    def __init__(
        self,
        num_radial: int = 16,
        cutoff: float = 5.0,
        cutoff_coeff: int = 5,
    ):
        super().__init__()
        if num_radial <= 0:
            raise ValueError("num_radial must be positive.")
        if cutoff <= 0:
            raise ValueError("cutoff must be positive.")

        self.num_radial = int(num_radial)
        self.cutoff = float(cutoff)
        self.norm_const = math.sqrt(2.0 / self.cutoff)
        self.cutoff_fn = PolynomialCutoff(cutoff=self.cutoff, cutoff_coeff=cutoff_coeff)
        self.register_buffer(
            "frequencies",
            math.pi * torch.arange(1, self.num_radial + 1, dtype=torch.float32),
        )

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        distances = distances.reshape(-1, 1)
        frequencies = self.frequencies.to(dtype=distances.dtype, device=distances.device)
        scaled = distances / self.cutoff
        numerator = torch.sin(frequencies * scaled)

        eps = torch.finfo(distances.dtype).eps
        radial = torch.where(
            distances.abs() > eps,
            numerator / distances.clamp_min(eps),
            frequencies / self.cutoff,
        )
        return self.cutoff_fn(distances) * self.norm_const * radial
