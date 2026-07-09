from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any, Final

DEFAULT_MODEL_NAME: Final = "GPTFF-MatPES_PBE_2025.2"
DEFAULT_MODEL_FILENAME: Final = f"{DEFAULT_MODEL_NAME}.pt"
DEFAULT_MODEL_RESOURCE_PACKAGE: Final = "gptff.assets"
DEFAULT_MODEL_RESOURCE_PARTS: Final = ("MatPES-PBE-2025.2", DEFAULT_MODEL_FILENAME)
DEFAULT_MODEL_SHA256: Final = "e9740e9d1c84b5fca3d9c20a3412440dda54f8e2238ad0a080cec83222fed916"


@dataclass(frozen=True)
class PretrainedModelSpec:
    name: str
    filename: str
    resource_package: str
    resource_parts: tuple[str, ...]
    sha256: str

    @property
    def resource(self):
        return files(self.resource_package).joinpath(*self.resource_parts)


PRETRAINED_MODELS: Final[Mapping[str, PretrainedModelSpec]] = {
    DEFAULT_MODEL_NAME: PretrainedModelSpec(
        name=DEFAULT_MODEL_NAME,
        filename=DEFAULT_MODEL_FILENAME,
        resource_package=DEFAULT_MODEL_RESOURCE_PACKAGE,
        resource_parts=DEFAULT_MODEL_RESOURCE_PARTS,
        sha256=DEFAULT_MODEL_SHA256,
    )
}


def available_models() -> tuple[str, ...]:
    return tuple(PRETRAINED_MODELS)


def get_model_spec(model_name: str | None = None) -> PretrainedModelSpec:
    name = model_name or DEFAULT_MODEL_NAME
    try:
        return PRETRAINED_MODELS[name]
    except KeyError as exc:
        available = ", ".join(available_models())
        raise ValueError(f"Unknown GPTFF pretrained model {name!r}. Available models: {available}.") from exc


def resolve_model_path(model_path: str | Path) -> Path:
    """Resolve an explicit checkpoint path."""

    path = Path(model_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"GPTFF checkpoint not found: {path}")
    return path


def load_checkpoint(
    *,
    model_name: str | None = None,
    model_path: str | Path | None = None,
    verify: bool = True,
) -> dict[str, Any]:
    """Load a GPTFF checkpoint from an explicit path or the packaged registry."""

    import torch

    if model_name is not None and model_path is not None:
        raise ValueError("Pass either model_name or model_path, not both.")

    if model_path is not None:
        return torch.load(
            resolve_model_path(model_path),
            map_location=torch.device("cpu"),
            weights_only=True,
        )

    spec = get_model_spec(model_name)
    with as_file(spec.resource) as path:
        if not Path(path).is_file():
            raise FileNotFoundError(f"Packaged GPTFF checkpoint is missing: {path}")
        if verify and path_checksum(path) != spec.sha256:
            raise ValueError(
                f"Checksum mismatch for pretrained model {spec.name!r}. "
                f"Expected {spec.sha256}."
            )
        return torch.load(
            path,
            map_location=torch.device("cpu"),
            weights_only=True,
        )


def load_model(
    *,
    model_name: str | None = None,
    model_path: str | Path | None = None,
    device=None,
):
    """Load a pretrained GPTFF potential."""

    from gptff.runtime import GPTFFPotential

    return GPTFFPotential.from_pretrained(
        model_name=model_name,
        model_path=model_path,
        device=device,
    )


def model_checksum(model_name: str | None = None) -> str:
    spec = get_model_spec(model_name)
    with spec.resource.open("rb") as file:
        return _hash_stream(file)


def path_checksum(path: str | Path) -> str:
    with open(resolve_model_path(path), "rb") as file:
        return _hash_stream(file)


def _hash_stream(file) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: file.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()
