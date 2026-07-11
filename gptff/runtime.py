from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from gptff.graph import CrystalGraphBatch, CrystalGraphConverter
from gptff.inference import predict_energy_forces_stress
from gptff.model.config import GPTFFConfig
from gptff.model.model import GPTFF


class GPTFFPotential:
    """Engine-agnostic GPTFF inference wrapper."""

    def __init__(
        self,
        *,
        model: GPTFF,
        model_config: GPTFFConfig,
        device: str | torch.device,
        checkpoint_metadata: Mapping[str, Any],
        model_name: str | None = None,
        model_path: str | Path | None = None,
    ) -> None:
        self.model = model
        self.model_config = model_config
        self.device = torch.device(device)
        self.checkpoint_metadata = dict(checkpoint_metadata)
        self.model_name = model_name
        self.model_path = None if model_path is None else Path(model_path)
        self.graph_converter = CrystalGraphConverter(
            radial_cutoff=self.model_config.radial_cutoff,
            angle_cutoff=self.model_config.angle_cutoff,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Mapping[str, Any],
        *,
        device: str | torch.device | None = None,
        model_name: str | None = None,
        model_path: str | Path | None = None,
    ) -> GPTFFPotential:
        resolved_device = resolve_device(device)
        model_config = GPTFFConfig.from_dict(checkpoint["model_config"])
        model = GPTFF(model_config)
        model.load_state_dict(checkpoint["state_dict"])
        model = model.to(resolved_device)
        model.eval()
        checkpoint_metadata = {
            key: value
            for key, value in checkpoint.items()
            if key != "state_dict" and not _contains_tensor(value)
        }
        return cls(
            model=model,
            model_config=model_config,
            device=resolved_device,
            checkpoint_metadata=checkpoint_metadata,
            model_name=model_name,
            model_path=model_path,
        )

    @classmethod
    def from_pretrained(
        cls,
        *,
        model_name: str | None = None,
        model_path: str | Path | None = None,
        device: str | torch.device | None = None,
    ) -> GPTFFPotential:
        from gptff.pretrained import (
            DEFAULT_MODEL_NAME,
            _load_checkpoint_file,
            _load_packaged_checkpoint,
            resolve_model_path,
        )

        if model_name is not None and model_path is not None:
            raise ValueError("Pass either model_name or model_path, not both.")

        resolved_device = resolve_device(device)
        resolved_path = None if model_path is None else resolve_model_path(model_path)
        checkpoint = (
            _load_checkpoint_file(resolved_path)
            if resolved_path is not None
            else _load_packaged_checkpoint(model_name)
        )
        return cls.from_checkpoint(
            checkpoint,
            device=resolved_device,
            model_name=None if resolved_path is not None else model_name or DEFAULT_MODEL_NAME,
            model_path=resolved_path,
        )

    def batch_graphs(self, graphs: Sequence) -> CrystalGraphBatch:
        return CrystalGraphBatch.from_graphs(graphs).to(self.device)

    def predict_batch(
        self,
        batch,
        *,
        compute_stress: bool = True,
        create_graph: bool = False,
    ):
        """Predict energy, forces, and stress in eV, eV/Angstrom, and eV/Angstrom^3."""

        return predict_energy_forces_stress(
            self.model,
            batch,
            create_graph=create_graph,
            compute_stress=compute_stress,
        )


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _contains_tensor(value) -> bool:
    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, Mapping):
        return any(_contains_tensor(item) for item in value.values())
    if isinstance(value, list | tuple):
        return any(_contains_tensor(item) for item in value)
    return False
