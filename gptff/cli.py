from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path

from gptff.trainer.config import load_config
from gptff.trainer.trainer import run_training

CommandHandler = Callable[[argparse.Namespace], None]


def _train(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    run_training(config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gptff",
        description="GPTFF command-line interface.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

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
    train_parser.set_defaults(handler=_train)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    handler: CommandHandler = args.handler
    handler(args)


if __name__ == "__main__":
    main()
