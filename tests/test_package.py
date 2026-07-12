import subprocess
import sys
from importlib.metadata import PackageNotFoundError
from pathlib import Path

import gptff
from gptff.runtime import GPTFFPotential as RuntimeGPTFFPotential


def test_top_level_package_defines_version_and_public_api() -> None:
    assert gptff.__file__.endswith("gptff/__init__.py")
    assert isinstance(gptff.__version__, str)
    assert gptff.__version__
    assert set(gptff.__all__) == {"DEFAULT_MODEL_NAME", "GPTFFPotential", "available_models"}
    assert all(hasattr(gptff, name) for name in gptff.__all__)
    assert gptff.GPTFFPotential is RuntimeGPTFFPotential
    assert gptff.GPTFFPotential is gptff.GPTFFPotential
    assert gptff.DEFAULT_MODEL_NAME in gptff.available_models()


def test_top_level_potential_is_loaded_lazily_and_cached() -> None:
    code = """
import sys
import gptff

assert "gptff.runtime" not in sys.modules
first = gptff.GPTFFPotential
assert "gptff.runtime" in sys.modules
assert first is gptff.GPTFFPotential
"""

    subprocess.run([sys.executable, "-c", code], check=True)


def test_version_fallback_matches_package_layout(monkeypatch) -> None:
    def missing_distribution(_name: str) -> str:
        raise PackageNotFoundError

    monkeypatch.setattr(gptff, "version", missing_distribution)

    pyproject_path = Path(gptff.__file__).resolve().parent.parent / "pyproject.toml"
    expected = "0.1.0" if pyproject_path.is_file() else "0+unknown"
    assert gptff._resolve_version() == expected
