from __future__ import annotations

import argparse
from pathlib import Path


def register_train_command(commands: argparse._SubParsersAction) -> None:
    train_parser = commands.add_parser(
        "train",
        help="Train a model from a YAML configuration.",
    )
    train_parser.add_argument(
        "config",
        type=Path,
        metavar="CONFIG",
        help="path to the YAML training configuration",
    )
    train_parser.add_argument(
        "--resume",
        type=Path,
        metavar="CHECKPOINT",
        help="resume an interrupted run from a training checkpoint (typically last.pt)",
    )
    train_parser.set_defaults(handler=_train)


def _train(args: argparse.Namespace) -> None:
    from gptff.trainer.config import load_config
    from gptff.trainer.trainer import run_training

    config = load_config(args.config)
    run_training(config, resume_from=args.resume)
