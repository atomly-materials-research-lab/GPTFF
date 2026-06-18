from pathlib import Path

import pytest

import gptff.cli as cli


def test_train_command_loads_config_and_starts_training(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    loaded_config = object()
    calls = []

    def fake_load_config(path: Path):
        calls.append(("load", path))
        return loaded_config

    def fake_run_training(config) -> None:
        calls.append(("train", config))

    monkeypatch.setattr(cli, "load_config", fake_load_config)
    monkeypatch.setattr(cli, "run_training", fake_run_training)

    cli.main(["train", str(config_path)])

    assert calls == [("load", config_path), ("train", loaded_config)]


def test_parser_requires_a_command() -> None:
    parser = cli.build_parser()

    with pytest.raises(SystemExit, match="2"):
        parser.parse_args([])
