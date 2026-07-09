from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "build_parser",
    "main",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        main_module = import_module("gptff.cli.main")
        exports = {
            "build_parser": main_module.build_parser,
            "main": main_module.main,
        }
        globals().update(exports)
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
