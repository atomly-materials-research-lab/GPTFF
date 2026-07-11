from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence

from gptff import __version__

CommandHandler = Callable[[argparse.Namespace], None]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gptff",
        description="GPTFF command-line interface.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    from gptff.cli.relaxation import register_relaxation_command
    from gptff.cli.train import register_train_command

    register_train_command(commands)
    register_relaxation_command(commands)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    handler: CommandHandler = args.handler
    handler(args)
