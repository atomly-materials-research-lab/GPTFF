from pathlib import Path

import pytest
from pymatgen.core import Lattice, Structure

import gptff.cli.main as cli_main
import gptff.tasks.relaxation as relaxation_task
import gptff.trainer.config as trainer_config
import gptff.trainer.trainer as trainer_module
from gptff.tasks.relaxation import RelaxationResult


def test_train_command_loads_config_and_starts_training(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    loaded_config = object()
    calls = []

    def fake_load_config(path: Path):
        calls.append(("load", path))
        return loaded_config

    def fake_run_training(config) -> None:
        calls.append(("train", config))

    monkeypatch.setattr(trainer_config, "load_config", fake_load_config)
    monkeypatch.setattr(trainer_module, "run_training", fake_run_training)

    cli_main.main(["train", str(config_path)])

    assert calls == [("load", config_path), ("train", loaded_config)]


def test_relaxation_command_calls_relaxation_and_writes_structure(
    monkeypatch,
    tmp_path,
) -> None:
    input_path = tmp_path / "input.cif"
    output_path = tmp_path / "relaxed.cif"
    structure = Structure(Lattice.cubic(3.0), ["Na"], [[0.0, 0.0, 0.0]])
    structure.to(filename=input_path)
    calls = []

    def fake_relax_with_ase(input_structure, **kwargs):
        calls.append((input_structure, kwargs))
        return RelaxationResult(
            initial_structure=input_structure,
            final_structure=input_structure,
            energy=-1.25,
            forces=[],
            max_force=0.0,
            is_converged=True,
            relaxation_requested=True,
            was_relaxed=True,
            n_steps=3,
            engine_name="ase",
        )

    monkeypatch.setattr(relaxation_task, "relax_with_ase", fake_relax_with_ase)

    cli_main.main(
        [
            "relaxation",
            str(input_path),
            "--output",
            str(output_path),
            "--fmax",
            "0.01",
            "--max-steps",
            "7",
            "--no-relax-cell",
            "--device",
            "cpu",
        ]
    )

    assert output_path.exists()
    assert Structure.from_file(output_path).formula == "Na1"
    assert len(calls) == 1
    _, kwargs = calls[0]
    assert kwargs["fmax"] == pytest.approx(0.01)
    assert kwargs["max_steps"] == 7
    assert kwargs["relax_cell"] is False
    assert kwargs["device"] == "cpu"


def test_main_module_entrypoint_help_smoke(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(["--help"])

    assert exc_info.value.code == 0
    assert "relaxation" in capsys.readouterr().out


def test_relaxation_parser_rejects_model_name_with_model_path() -> None:
    parser = cli_main.build_parser()

    with pytest.raises(SystemExit, match="2"):
        parser.parse_args(
            [
                "relaxation",
                "input.cif",
                "--output",
                "relaxed.cif",
                "--model-name",
                "default",
                "--model-path",
                "custom.pt",
            ]
        )


def test_relaxation_parser_rejects_unknown_optimizer() -> None:
    parser = cli_main.build_parser()

    with pytest.raises(SystemExit, match="2"):
        parser.parse_args(
            [
                "relaxation",
                "input.cif",
                "--output",
                "relaxed.cif",
                "--optimizer",
                "NoSuchOptimizer",
            ]
        )


def test_parser_requires_a_command() -> None:
    parser = cli_main.build_parser()

    with pytest.raises(SystemExit, match="2"):
        parser.parse_args([])
