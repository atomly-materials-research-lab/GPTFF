from __future__ import annotations

import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gptff.pretrained import DEFAULT_MODEL_NAME, available_models

if TYPE_CHECKING:
    # GPTFFPotential is provided at runtime by __getattr__ and cached in globals().
    from gptff.runtime import GPTFFPotential

__all__ = [
    "DEFAULT_MODEL_NAME",
    "GPTFFPotential",
    "available_models",
]

def _resolve_version() -> str:
    try:
        return version("gptff")
    except PackageNotFoundError:
        pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
        try:
            with open(pyproject_path, "rb") as file:
                return str(tomllib.load(file)["project"]["version"])
        except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError):
            return "0+unknown"


__version__ = _resolve_version()


def __getattr__(name: str) -> Any:
    if name == "GPTFFPotential":
        from gptff.runtime import GPTFFPotential

        globals()[name] = GPTFFPotential
        return GPTFFPotential
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
