#!/usr/bin/env python3
"""Convert MatPES JSON data into a GPTFF AtomicDataset JSON file.

MatPES entries are expected to contain pymatgen-serialized structures plus
total energy in eV, forces in eV/angstrom, and VASP-style stress in kbar Voigt
order [xx, yy, zz, yz, xz, xy]. GPTFF's AtomicSample stores raw VASP stress and
converts it to GPa when graph samples are built.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from pymatgen.core import Structure

from gptff.data import AtomicDataset, AtomicSample

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "MatPES-PBE-2025.2.json"
DEFAULT_OUTPUT = SCRIPT_DIR / "MatPES-PBE-2025.2.gptff.json"
LABEL_KEYS = {"structure", "energy", "forces", "stress"}
LARGE_OPTIONAL_KEYS = {
    "abs_forces",
    "bader_charges",
    "bader_magmoms",
    "cm5_partial_charges",
    "ddec6",
}
COMPACT_METADATA_KEYS = (
    "matpes_id",
    "functional",
    "formula_pretty",
    "formula_anonymous",
    "chemsys",
    "composition",
    "composition_reduced",
    "nsites",
    "elements",
    "nelements",
    "volume",
    "density",
    "density_atomic",
    "symmetry",
    "bandgap",
    "formation_energy_per_atom",
    "cohesive_energy_per_atom",
    "provenance",
    "builder_meta",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert MatPES JSON/JSON.GZ data into GPTFF AtomicDataset format."
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"MatPES JSON or JSON.GZ file. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output AtomicDataset JSON/JSON.GZ file. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="AtomicDataset name. Defaults to the input filename stem.",
    )
    parser.add_argument(
        "--functional",
        default=None,
        help="Optional functional filter, for example PBE or r2SCAN.",
    )
    parser.add_argument(
        "--metadata",
        choices=("compact", "all", "none"),
        default="compact",
        help="Metadata preservation mode. Default: compact.",
    )
    parser.add_argument(
        "--allow-missing-stress",
        action="store_true",
        help="Keep entries with missing stress as stress=None.",
    )
    parser.add_argument(
        "--skip-invalid",
        action="store_true",
        help="Skip invalid entries instead of failing at the first error.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    entries = load_matpes_entries(args.input)
    samples = []
    skipped = 0

    for index, entry in enumerate(entries):
        if args.functional and str(entry.get("functional", "")).lower() != args.functional.lower():
            continue

        try:
            samples.append(
                entry_to_atomic_sample(
                    entry,
                    index=index,
                    metadata_mode=args.metadata,
                    allow_missing_stress=args.allow_missing_stress,
                )
            )
        except Exception as exc:
            if not args.skip_invalid:
                sample_id = entry.get("matpes_id", f"entry:{index}")
                raise ValueError(f"Failed to convert MatPES entry {sample_id!r}.") from exc
            skipped += 1

    if not samples:
        raise ValueError("No MatPES entries were converted.")

    dataset = AtomicDataset(
        samples=tuple(samples),
        name=args.name or dataset_name_from_path(args.input),
        metadata={
            "source_format": "MatPES",
            "source_file": str(args.input),
            "num_converted": len(samples),
            "num_skipped": skipped,
            "stress_input_unit": "kbar",
            "stress_input_order": "[xx, yy, zz, yz, xz, xy]",
            "energy_unit": "eV",
            "force_unit": "eV/angstrom",
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_file(args.output)
    print(f"Converted {len(samples)} MatPES entries to GPTFF AtomicDataset: {args.output}")
    if skipped:
        print(f"Skipped {skipped} invalid entries.")


def load_matpes_entries(path: Path) -> Iterable[Mapping[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"MatPES input file not found: {path}")

    if path.suffix == ".gz":
        return load_matpes_entries_eager(path)
    if first_non_whitespace_char(path) == "[":
        return iter_json_array(path)
    return load_matpes_entries_eager(path)


def load_matpes_entries_eager(path: Path) -> Iterable[Mapping[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, list):
        return data
    if isinstance(data, Mapping):
        for key in ("data", "entries", "structures", "samples"):
            value = data.get(key)
            if isinstance(value, list):
                return value
    raise ValueError(
        "MatPES JSON must be a list of entries or a mapping with one of "
        "the keys: data, entries, structures, samples."
    )


def first_non_whitespace_char(path: Path) -> str:
    with open(path, encoding="utf-8") as file:
        while True:
            char = file.read(1)
            if char == "":
                raise ValueError(f"MatPES JSON file is empty: {path}")
            if not char.isspace():
                return char


def iter_json_array(path: Path, *, chunk_size: int = 1024 * 1024):
    decoder = json.JSONDecoder()
    buffer = ""
    array_started = False
    done = False

    with open(path, encoding="utf-8") as file:
        while not done:
            if not buffer:
                chunk = file.read(chunk_size)
                if chunk == "":
                    raise ValueError("Unexpected end of JSON while reading top-level array.")
                buffer += chunk

            buffer = buffer.lstrip()
            if not array_started:
                if not buffer:
                    continue
                if buffer[0] != "[":
                    raise ValueError("Expected a top-level JSON array.")
                buffer = buffer[1:]
                array_started = True
                continue

            buffer = buffer.lstrip()
            if not buffer:
                continue
            if buffer[0] == "]":
                done = True
                buffer = buffer[1:]
                continue
            if buffer[0] == ",":
                buffer = buffer[1:]
                continue

            while True:
                try:
                    entry, end = decoder.raw_decode(buffer)
                    break
                except json.JSONDecodeError:
                    chunk = file.read(chunk_size)
                    if chunk == "":
                        raise
                    buffer += chunk

            if not isinstance(entry, Mapping):
                raise ValueError("Each MatPES array item must be a JSON object.")
            yield entry
            buffer = buffer[end:]


def entry_to_atomic_sample(
    entry: Mapping[str, Any],
    *,
    index: int,
    metadata_mode: str,
    allow_missing_stress: bool,
) -> AtomicSample:
    missing = sorted(key for key in ("structure", "energy", "forces") if key not in entry)
    if missing:
        raise KeyError(f"Missing required MatPES keys: {missing}")

    stress = entry.get("stress")
    if stress is None and not allow_missing_stress:
        raise KeyError("Missing required MatPES key: stress")

    return AtomicSample(
        structure=Structure.from_dict(entry["structure"]),
        energy=entry["energy"],
        forces=entry["forces"],
        stress=stress,
        sample_id=sample_id(entry, index),
        material_id=material_id(entry),
        metadata=metadata(entry, metadata_mode),
    )


def sample_id(entry: Mapping[str, Any], index: int) -> str:
    value = entry.get("matpes_id") or entry.get("task_id") or entry.get("id")
    return str(value) if value is not None else f"matpes-entry-{index}"


def material_id(entry: Mapping[str, Any]) -> str | None:
    provenance = entry.get("provenance")
    if isinstance(provenance, Mapping):
        original_mp_id = provenance.get("original_mp_id")
        if original_mp_id is not None:
            return str(original_mp_id)
    return None


def metadata(entry: Mapping[str, Any], mode: str) -> dict[str, Any]:
    if mode == "none":
        return {}
    if mode == "all":
        return {key: value for key, value in entry.items() if key not in LABEL_KEYS}
    return {
        key: entry[key]
        for key in COMPACT_METADATA_KEYS
        if key in entry and key not in LARGE_OPTIONAL_KEYS
    }


def dataset_name_from_path(path: Path) -> str:
    if path.name.endswith(".json.gz"):
        return path.name.removesuffix(".json.gz")
    return path.stem


if __name__ == "__main__":
    main()
