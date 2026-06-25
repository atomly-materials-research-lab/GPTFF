#!/usr/bin/env python3
"""Convert MPtrj JSON data into a GPTFF AtomicDataset JSON file.

MPtrj is a nested JSON mapping:

    {
        "mp-id": {
            "frame-id": {
                "structure": pymatgen Structure dict,
                "corrected_total_energy": total energy in eV,
                "force": forces in eV/angstrom,
                "stress": raw VASP stress in kbar,
                ...
            },
            ...
        },
        ...
    }

GPTFF's AtomicSample stores total energy in eV, forces in eV/angstrom, and raw
VASP stress in kbar. The graph dataset handles VASP stress sign/unit conversion
when samples are loaded for training.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from pymatgen.core import Structure

from gptff.data import AtomicDataset, AtomicSample

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "MPtrj_2022.9_full.json"
DEFAULT_OUTPUT = SCRIPT_DIR / "MPtrj_2022.9.gptff.json"
LABEL_KEYS = {
    "structure",
    "corrected_total_energy",
    "uncorrected_total_energy",
    "force",
    "stress",
}
COMPACT_METADATA_KEYS = (
    "mp_id",
    "frame_id",
    "uncorrected_total_energy",
    "energy_per_atom",
    "ef_per_atom",
    "e_per_atom_relaxed",
    "ef_per_atom_relaxed",
    "bandgap",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert MPtrj JSON/JSON.GZ data into GPTFF AtomicDataset format."
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"MPtrj JSON or JSON.GZ file. Default: {DEFAULT_INPUT}",
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
        "--metadata",
        choices=("compact", "all", "none"),
        default="compact",
        help="Metadata preservation mode. Default: compact.",
    )
    parser.add_argument(
        "--energy-key",
        default="corrected_total_energy",
        choices=("corrected_total_energy", "uncorrected_total_energy"),
        help="Total-energy label to store in AtomicSample.energy. Default: corrected_total_energy.",
    )
    parser.add_argument(
        "--allow-missing-stress",
        action="store_true",
        help="Keep frames with missing stress as stress=None.",
    )
    parser.add_argument(
        "--skip-invalid",
        action="store_true",
        help="Skip invalid frames instead of failing at the first error.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Convert at most this many frames. Useful for quick checks.",
    )
    parser.add_argument(
        "--max-materials",
        type=int,
        default=None,
        help="Read at most this many outer mp-id groups.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = []
    skipped = 0
    materials_seen = 0

    for material_id, frames in iter_mptrj_materials(args.input):
        materials_seen += 1
        if args.max_materials is not None and materials_seen > args.max_materials:
            break
        if not isinstance(frames, Mapping):
            raise TypeError(f"MPtrj material {material_id!r} must contain a frame mapping.")

        for frame_id, frame in frames.items():
            if args.limit is not None and len(samples) >= args.limit:
                break
            if not isinstance(frame, Mapping):
                raise TypeError(f"MPtrj frame {frame_id!r} must be a mapping.")

            try:
                samples.append(
                    frame_to_atomic_sample(
                        frame,
                        material_id=str(material_id),
                        frame_id=str(frame_id),
                        metadata_mode=args.metadata,
                        energy_key=args.energy_key,
                        allow_missing_stress=args.allow_missing_stress,
                    )
                )
            except Exception as exc:
                if not args.skip_invalid:
                    raise ValueError(
                        f"Failed to convert MPtrj frame {material_id!r}/{frame_id!r}."
                    ) from exc
                skipped += 1

        if args.limit is not None and len(samples) >= args.limit:
            break

    if not samples:
        raise ValueError("No MPtrj frames were converted.")

    dataset = AtomicDataset(
        samples=tuple(samples),
        name=args.name or dataset_name_from_path(args.input),
        metadata={
            "source_format": "MPtrj",
            "source_file": str(args.input),
            "num_converted": len(samples),
            "num_skipped": skipped,
            "num_materials_read": materials_seen,
            "energy_key": args.energy_key,
            "energy_unit": "eV",
            "force_unit": "eV/angstrom",
            "stress_input_unit": "kbar",
            "stress_input_convention": "raw VASP stress",
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_file(args.output)
    print(f"Converted {len(samples)} MPtrj frames to GPTFF AtomicDataset: {args.output}")
    print(f"Read {materials_seen} material groups.")
    if skipped:
        print(f"Skipped {skipped} invalid frames.")


def frame_to_atomic_sample(
    frame: Mapping[str, Any],
    *,
    material_id: str,
    frame_id: str,
    metadata_mode: str,
    energy_key: str,
    allow_missing_stress: bool,
) -> AtomicSample:
    missing = sorted(key for key in ("structure", energy_key, "force") if key not in frame)
    if missing:
        raise KeyError(f"Missing required MPtrj keys: {missing}")

    stress = frame.get("stress")
    if stress is None and not allow_missing_stress:
        raise KeyError("Missing required MPtrj key: stress")

    frame_mp_id = frame.get("mp_id")
    if frame_mp_id is not None and str(frame_mp_id) != material_id:
        raise ValueError(
            f"Frame mp_id {frame_mp_id!r} does not match outer material_id {material_id!r}."
        )

    return AtomicSample(
        structure=Structure.from_dict(frame["structure"]),
        energy=frame[energy_key],
        forces=frame["force"],
        stress=stress,
        sample_id=frame_id,
        material_id=material_id,
        metadata=metadata(frame, metadata_mode, material_id=material_id, frame_id=frame_id),
    )


def metadata(
    frame: Mapping[str, Any],
    mode: str,
    *,
    material_id: str,
    frame_id: str,
) -> dict[str, Any]:
    if mode == "none":
        return {}
    if mode == "all":
        data = {key: value for key, value in frame.items() if key not in LABEL_KEYS}
    else:
        data = {key: frame[key] for key in COMPACT_METADATA_KEYS if key in frame}
    data.setdefault("mp_id", material_id)
    data.setdefault("frame_id", frame_id)
    return data


def iter_mptrj_materials(path: Path) -> Iterator[tuple[str, Mapping[str, Any]]]:
    if not path.exists():
        raise FileNotFoundError(f"MPtrj input file not found: {path}")
    if path.suffix == ".gz":
        yield from iter_mptrj_materials_eager(path)
        return
    yield from iter_json_object_items(path)


def iter_mptrj_materials_eager(path: Path) -> Iterator[tuple[str, Mapping[str, Any]]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, Mapping):
        raise ValueError("MPtrj JSON must be a top-level mapping from mp-id to frame mapping.")
    for material_id, frames in data.items():
        yield str(material_id), frames


def iter_json_object_items(path: Path, *, chunk_size: int = 1024 * 1024):
    """Stream top-level JSON object items without loading the full file."""

    decoder = json.JSONDecoder()
    buffer = ""
    object_started = False
    done = False

    with open(path, encoding="utf-8") as file:
        while not done:
            buffer = ensure_buffer(file, buffer, chunk_size)
            buffer = buffer.lstrip()

            if not object_started:
                if not buffer:
                    continue
                if buffer[0] != "{":
                    raise ValueError("Expected a top-level JSON object.")
                buffer = buffer[1:]
                object_started = True
                continue

            buffer = ensure_buffer(file, buffer, chunk_size).lstrip()
            if not buffer:
                continue
            if buffer[0] == "}":
                done = True
                buffer = buffer[1:]
                continue
            if buffer[0] == ",":
                buffer = buffer[1:]
                continue

            material_id, buffer = raw_decode_with_more(decoder, file, buffer, chunk_size)
            if not isinstance(material_id, str):
                raise ValueError("MPtrj top-level keys must be strings.")

            buffer = ensure_buffer(file, buffer, chunk_size).lstrip()
            if not buffer or buffer[0] != ":":
                raise ValueError(f"Expected ':' after material id {material_id!r}.")
            buffer = buffer[1:]

            frames, buffer = raw_decode_with_more(decoder, file, buffer, chunk_size)
            if not isinstance(frames, Mapping):
                raise ValueError(f"MPtrj material {material_id!r} must contain a mapping.")
            yield material_id, frames


def ensure_buffer(file, buffer: str, chunk_size: int) -> str:
    while not buffer:
        chunk = file.read(chunk_size)
        if chunk == "":
            raise ValueError("Unexpected end of JSON file.")
        buffer += chunk
    return buffer


def raw_decode_with_more(decoder: json.JSONDecoder, file, buffer: str, chunk_size: int):
    while True:
        stripped = buffer.lstrip()
        offset = len(buffer) - len(stripped)
        try:
            value, end = decoder.raw_decode(stripped)
            return value, buffer[offset + end :]
        except json.JSONDecodeError:
            chunk = file.read(chunk_size)
            if chunk == "":
                raise
            buffer += chunk


def dataset_name_from_path(path: Path) -> str:
    if path.name.endswith(".json.gz"):
        return path.name.removesuffix(".json.gz")
    return path.stem


if __name__ == "__main__":
    main()
