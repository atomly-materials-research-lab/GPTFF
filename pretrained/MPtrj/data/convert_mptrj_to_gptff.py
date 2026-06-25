#!/usr/bin/env python3
"""Convert MPtrj JSON data into GPTFF training data.

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

The default output is a sharded HDF5 graph dataset for large-scale training.
For small debugging jobs, ``--format atomic-json`` can still write a legacy
AtomicDataset JSON file.
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pymatgen.core import Structure
from tqdm.auto import tqdm

from gptff.data import AtomicSample, ShardedGraphDatasetWriter
from gptff.graph import CrystalGraphConverter

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "MPtrj_2022.9_full.json"
DEFAULT_OUTPUT = SCRIPT_DIR / "MPtrj_2022.9.gptff"
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


@dataclass(frozen=True)
class MPtrjFrame:
    material_id: str
    frame_id: str
    frame: Mapping[str, Any]
    material_index: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert MPtrj JSON/JSON.GZ data into GPTFF training data."
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
        help=f"Output dataset path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--format",
        choices=("sharded-hdf5-graph", "atomic-json"),
        default="sharded-hdf5-graph",
        help="Output format. Default: sharded-hdf5-graph.",
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
    parser.add_argument(
        "--radial-cutoff",
        type=float,
        default=5.0,
        help="Radial graph cutoff for sharded-hdf5-graph output. Default: 5.0.",
    )
    parser.add_argument(
        "--angle-cutoff",
        type=float,
        default=4.0,
        help="Angle graph cutoff for sharded-hdf5-graph output. Default: 4.0.",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=5000,
        help="Number of graph samples per HDF5 shard. Default: 5000.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_output_path(args)
    if args.format == "atomic-json":
        convert_to_atomic_json(args)
    else:
        convert_to_sharded_hdf5_graph(args)


def validate_output_path(args: argparse.Namespace) -> None:
    if args.format == "sharded-hdf5-graph" and (
        args.output.suffix == ".json" or args.output.name.endswith(".json.gz")
    ):
        raise ValueError(
            "sharded-hdf5-graph output must be a dataset directory path, for example "
            "'MPtrj_2022.9.gptff'. Use '--format atomic-json' for JSON output."
        )


def convert_to_atomic_json(args: argparse.Namespace) -> None:
    converted = 0
    skipped = 0
    materials_seen = 0
    output_tmp = temporary_output_path(args.output)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open_output(output_tmp, compressed=is_gzip_path(args.output)) as output_file:
            write_dataset_header(output_file)

            with tqdm(desc="Converting MPtrj", unit="frame", dynamic_ncols=True) as progress:
                for record in iter_mptrj_frames(args.input, max_materials=args.max_materials):
                    materials_seen = max(materials_seen, record.material_index)
                    if args.limit is not None and converted >= args.limit:
                        break
                    try:
                        sample = frame_to_atomic_sample(
                            record.frame,
                            material_id=record.material_id,
                            frame_id=record.frame_id,
                            metadata_mode=args.metadata,
                            energy_key=args.energy_key,
                            allow_missing_stress=args.allow_missing_stress,
                        )
                        write_dataset_sample(
                            output_file,
                            sample,
                            is_first_sample=converted == 0,
                        )
                        converted += 1
                    except Exception as exc:
                        if not args.skip_invalid:
                            raise ValueError(
                                f"Failed to convert MPtrj frame "
                                f"{record.material_id!r}/{record.frame_id!r}."
                            ) from exc
                        skipped += 1
                    finally:
                        progress.update()
                        progress.set_postfix(
                            converted=converted,
                            skipped=skipped,
                            materials=materials_seen,
                            refresh=False,
                        )

            if converted == 0:
                raise ValueError("No MPtrj frames were converted.")

            write_dataset_footer(
                output_file,
                name=args.name or dataset_name_from_path(args.input),
                metadata={
                    "source_format": "MPtrj",
                    "source_file": str(args.input),
                    "num_converted": converted,
                    "num_skipped": skipped,
                    "num_materials_read": materials_seen,
                    "energy_key": args.energy_key,
                    "energy_unit": "eV",
                    "force_unit": "eV/angstrom",
                    "stress_input_unit": "kbar",
                    "stress_input_convention": "raw VASP stress",
                },
            )
        output_tmp.replace(args.output)
    except Exception:
        output_tmp.unlink(missing_ok=True)
        raise

    print(f"Converted {converted} MPtrj frames to GPTFF AtomicDataset: {args.output}")
    print(f"Read {materials_seen} material groups.")
    if skipped:
        print(f"Skipped {skipped} invalid frames.")


def convert_to_sharded_hdf5_graph(args: argparse.Namespace) -> None:
    converted = 0
    skipped = 0
    materials_seen = 0
    output_tmp = temporary_output_path(args.output)
    converter = CrystalGraphConverter(
        radial_cutoff=args.radial_cutoff,
        angle_cutoff=args.angle_cutoff,
    )

    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output path already exists: {args.output}")
    if output_tmp.exists():
        if output_tmp.is_dir():
            shutil.rmtree(output_tmp)
        else:
            output_tmp.unlink()

    try:
        with ShardedGraphDatasetWriter(
            output_tmp,
            name=args.name or dataset_name_from_path(args.input),
            metadata={
                "source_format": "MPtrj",
                "source_file": str(args.input),
                "energy_key": args.energy_key,
                "energy_unit": "eV",
                "force_unit": "eV/angstrom",
                "stress_input_unit": "kbar",
                "stress_input_convention": "raw VASP stress",
                "radial_cutoff": args.radial_cutoff,
                "angle_cutoff": args.angle_cutoff,
            },
            shard_size=args.shard_size,
            overwrite=True,
        ) as writer:
            with tqdm(desc="Converting MPtrj", unit="frame", dynamic_ncols=True) as progress:
                for record in iter_mptrj_frames(args.input, max_materials=args.max_materials):
                    materials_seen = max(materials_seen, record.material_index)
                    if args.limit is not None and converted >= args.limit:
                        break
                    try:
                        sample = frame_to_atomic_sample(
                            record.frame,
                            material_id=record.material_id,
                            frame_id=record.frame_id,
                            metadata_mode=args.metadata,
                            energy_key=args.energy_key,
                            allow_missing_stress=args.allow_missing_stress,
                        )
                        writer.add(
                            graph=converter.convert(sample.structure),
                            energy=sample.energy,
                            forces=sample.forces,
                            stress=sample.stress,
                            sample_id=sample.sample_id,
                            material_id=sample.material_id,
                        )
                        converted += 1
                    except Exception as exc:
                        if not args.skip_invalid:
                            raise ValueError(
                                f"Failed to convert MPtrj frame "
                                f"{record.material_id!r}/{record.frame_id!r}."
                            ) from exc
                        skipped += 1
                    finally:
                        progress.update()
                        progress.set_postfix(
                            converted=converted,
                            skipped=skipped,
                            materials=materials_seen,
                            refresh=False,
                        )

            if converted == 0:
                raise ValueError("No MPtrj frames were converted.")
            writer.metadata.update(
                num_converted=converted,
                num_skipped=skipped,
                num_materials_read=materials_seen,
            )
        if args.output.exists():
            if args.output.is_dir():
                shutil.rmtree(args.output)
            else:
                args.output.unlink()
        output_tmp.replace(args.output)
    except Exception:
        shutil.rmtree(output_tmp, ignore_errors=True)
        raise

    print(f"Converted {converted} MPtrj frames to sharded GPTFF graph dataset: {args.output}")
    print(f"Read {materials_seen} material groups.")
    if skipped:
        print(f"Skipped {skipped} invalid frames.")


def write_dataset_header(output_file) -> None:
    output_file.write(
        '{"@module":"gptff.data.dataset",'
        '"@class":"AtomicDataset",'
        '"samples":['
    )


def write_dataset_sample(
    output_file,
    sample: AtomicSample,
    *,
    is_first_sample: bool,
) -> None:
    if not is_first_sample:
        output_file.write(",")
    json.dump(sample.as_dict(), output_file, separators=(",", ":"))


def write_dataset_footer(
    output_file,
    *,
    name: str | None,
    metadata: Mapping[str, Any],
) -> None:
    output_file.write('],"name":')
    json.dump(name, output_file, separators=(",", ":"))
    output_file.write(',"metadata":')
    json.dump(dict(metadata), output_file, separators=(",", ":"))
    output_file.write("}")


def open_output(path: Path, *, compressed: bool):
    if compressed:
        return gzip.open(path, "wt", encoding="utf-8")
    return open(path, "w", encoding="utf-8")


def temporary_output_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.tmp")


def is_gzip_path(path: Path) -> bool:
    return path.suffix == ".gz"


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


def iter_mptrj_frames(
    path: Path,
    *,
    max_materials: int | None = None,
) -> Iterator[MPtrjFrame]:
    if not path.exists():
        raise FileNotFoundError(f"MPtrj input file not found: {path}")
    yield from iter_json_frames(path, max_materials=max_materials)


def iter_json_frames(
    path: Path,
    *,
    max_materials: int | None = None,
    chunk_size: int = 1024 * 1024,
) -> Iterator[MPtrjFrame]:
    """Stream nested MPtrj material/frame items without loading material groups."""

    decoder = json.JSONDecoder()
    buffer = ""
    object_started = False
    material_index = 0
    done = False

    opener = gzip.open if is_gzip_path(path) else open
    with opener(path, "rt", encoding="utf-8") as file:
        while not done:
            if not object_started:
                char, buffer = pop_next_non_whitespace(file, buffer, chunk_size)
                if char != "{":
                    raise ValueError("Expected a top-level JSON object.")
                object_started = True
                continue

            char, buffer = pop_next_non_whitespace(file, buffer, chunk_size)
            if char == "}":
                done = True
                continue
            if char == ",":
                continue

            buffer = char + buffer
            material_id, buffer = raw_decode_with_more(decoder, file, buffer, chunk_size)
            if not isinstance(material_id, str):
                raise ValueError("MPtrj top-level keys must be strings.")
            material_index += 1
            if max_materials is not None and material_index > max_materials:
                break

            char, buffer = pop_next_non_whitespace(file, buffer, chunk_size)
            if char != ":":
                raise ValueError(f"Expected ':' after material id {material_id!r}.")

            buffer = yield from iter_material_frames(
                decoder,
                file,
                buffer,
                chunk_size,
                material_id=material_id,
                material_index=material_index,
            )


def ensure_buffer(file, buffer: str, chunk_size: int) -> str:
    while not buffer:
        chunk = file.read(chunk_size)
        if chunk == "":
            raise ValueError("Unexpected end of JSON file.")
        buffer += chunk
    return buffer


def pop_next_non_whitespace(file, buffer: str, chunk_size: int) -> tuple[str, str]:
    while True:
        buffer = ensure_buffer(file, buffer, chunk_size)
        stripped = buffer.lstrip()
        if stripped:
            return stripped[0], stripped[1:]
        buffer = ""


def iter_material_frames(
    decoder: json.JSONDecoder,
    file,
    buffer: str,
    chunk_size: int,
    *,
    material_id: str,
    material_index: int,
) -> Iterator[MPtrjFrame]:
    char, buffer = pop_next_non_whitespace(file, buffer, chunk_size)
    if char != "{":
        raise ValueError(f"MPtrj material {material_id!r} must contain a frame mapping.")

    while True:
        char, buffer = pop_next_non_whitespace(file, buffer, chunk_size)
        if char == "}":
            return buffer
        if char == ",":
            continue

        buffer = char + buffer
        frame_id, buffer = raw_decode_with_more(decoder, file, buffer, chunk_size)
        if not isinstance(frame_id, str):
            raise ValueError(f"MPtrj frame keys for material {material_id!r} must be strings.")

        char, buffer = pop_next_non_whitespace(file, buffer, chunk_size)
        if char != ":":
            raise ValueError(
                f"Expected ':' after frame id {material_id!r}/{frame_id!r}."
            )

        frame, buffer = raw_decode_with_more(decoder, file, buffer, chunk_size)
        if not isinstance(frame, Mapping):
            raise ValueError(f"MPtrj frame {material_id!r}/{frame_id!r} must be a mapping.")
        yield MPtrjFrame(
            material_id=material_id,
            frame_id=frame_id,
            frame=frame,
            material_index=material_index,
        )


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
