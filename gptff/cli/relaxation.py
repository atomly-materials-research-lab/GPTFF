from __future__ import annotations

import argparse
from inspect import isclass
from pathlib import Path


def register_relaxation_command(commands: argparse._SubParsersAction) -> None:
    parser = commands.add_parser(
        "relaxation",
        help="Relax a structure with GPTFF.",
    )
    parser.add_argument(
        "input",
        type=Path,
        metavar="INPUT",
        help="input structure file readable by pymatgen",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="output structure file; format is inferred by pymatgen from the suffix",
    )
    model_selection = parser.add_mutually_exclusive_group()
    model_selection.add_argument(
        "--model-name",
        default=None,
        help="registered GPTFF model name; defaults to the packaged model",
    )
    model_selection.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="explicit checkpoint path; overrides the default packaged model",
    )
    parser.add_argument(
        "--device",
        default=None,
        help='torch device, for example "cpu" or "cuda"; defaults to auto selection',
    )
    parser.add_argument(
        "--optimizer",
        type=_ase_optimizer_name,
        default="FIRE",
        help="ASE optimizer class name",
    )
    parser.add_argument(
        "--fmax",
        type=float,
        default=0.05,
        help="force convergence threshold in eV/angstrom",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="maximum optimizer steps",
    )
    parser.add_argument(
        "--relax-atoms",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="relax atomic positions",
    )
    parser.add_argument(
        "--relax-cell",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="relax cell degrees of freedom",
    )
    parser.add_argument(
        "--fix-symmetry",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="apply ASE FixSymmetry during relaxation",
    )
    parser.add_argument(
        "--symprec",
        type=float,
        default=1e-2,
        help="symmetry tolerance used with --fix-symmetry",
    )
    parser.add_argument(
        "--external-pressure-gpa",
        type=float,
        default=0.0,
        help="external pressure in GPa for cell relaxation",
    )
    parser.add_argument(
        "--logfile",
        type=Path,
        default=None,
        help="ASE optimizer log file",
    )
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=None,
        help="ASE trajectory output file",
    )
    parser.set_defaults(handler=_relaxation)


def _relaxation(args: argparse.Namespace) -> None:
    from pymatgen.core import Structure

    from gptff.tasks.relaxation import relax_with_ase

    structure = Structure.from_file(args.input)
    result = relax_with_ase(
        structure,
        model_name=args.model_name,
        model_path=args.model_path,
        device=args.device,
        optimizer=args.optimizer,
        fmax=args.fmax,
        max_steps=args.max_steps,
        relax_atoms=args.relax_atoms,
        relax_cell=args.relax_cell,
        fix_symmetry=args.fix_symmetry,
        symprec=args.symprec,
        external_pressure_gpa=args.external_pressure_gpa,
        logfile=None if args.logfile is None else str(args.logfile),
        trajectory=None if args.trajectory is None else str(args.trajectory),
    )
    result.final_structure.to(filename=args.output)
    _print_relaxation_summary(result, output=args.output)


def _print_relaxation_summary(result, *, output: Path) -> None:
    energy = "nan" if result.energy is None else f"{result.energy:.10g}"
    max_force = "nan" if result.max_force is None else f"{result.max_force:.6g}"
    converged = "yes" if result.is_converged else "no"
    print(
        "Relaxation finished: "
        f"converged={converged} "
        f"steps={result.n_steps} "
        f"energy={energy} eV "
        f"max_force={max_force} eV/angstrom "
        f"output={output}"
    )
    for warning in result.warnings:
        print(f"Warning: {warning}")


def _ase_optimizer_name(name: str) -> str:
    import ase.optimize
    from ase.optimize.optimize import Optimizer

    optimizer_cls = getattr(ase.optimize, name, None)
    if isclass(optimizer_cls) and issubclass(optimizer_cls, Optimizer):
        return name
    raise argparse.ArgumentTypeError(f"Unknown ASE optimizer {name!r}.")
