import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from pymatgen.core import Lattice, Structure

from gptff.data import ShardedGraphDataset


def test_iter_mptrj_frames_streams_nested_frames(tmp_path):
    converter = _load_mptrj_converter()
    path = tmp_path / "mptrj.json"
    path.write_text(
        json.dumps(
            {
                "mp-a": {
                    "frame-0": {"value": 0},
                    "frame-1": {"value": 1},
                },
                "mp-b": {
                    "frame-0": {"value": 2},
                },
            }
        ),
        encoding="utf-8",
    )

    frames = list(converter.iter_mptrj_frames(path))

    assert [
        (frame.material_id, frame.frame_id, frame.frame["value"], frame.material_index)
        for frame in frames
    ] == [
        ("mp-a", "frame-0", 0, 1),
        ("mp-a", "frame-1", 1, 1),
        ("mp-b", "frame-0", 2, 2),
    ]


def test_iter_mptrj_frames_respects_max_materials(tmp_path):
    converter = _load_mptrj_converter()
    path = tmp_path / "mptrj.json"
    path.write_text(
        json.dumps(
            {
                "mp-a": {"frame-0": {"value": 0}},
                "mp-b": {"frame-0": {"value": 1}},
            }
        ),
        encoding="utf-8",
    )

    frames = list(converter.iter_mptrj_frames(path, max_materials=1))

    assert [(frame.material_id, frame.frame_id) for frame in frames] == [
        ("mp-a", "frame-0")
    ]


def test_convert_to_sharded_hdf5_uses_num_process_for_shard_count(tmp_path):
    converter = _load_mptrj_converter()
    input_path = tmp_path / "mptrj.json"
    output_path = tmp_path / "mptrj.gptff"
    structure = Structure(Lattice.cubic(3.0), ["Na"], [[0.0, 0.0, 0.0]]).as_dict()
    input_path.write_text(
        json.dumps(
            {
                "mp-a": {
                    f"frame-{idx}": {
                        "structure": structure,
                        "corrected_total_energy": -float(idx),
                        "force": [[0.0, 0.0, 0.0]],
                        "stress": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    }
                    for idx in range(5)
                },
            }
        ),
        encoding="utf-8",
    )

    converter.convert_to_sharded_hdf5_graph(
        SimpleNamespace(
            input=input_path,
            output=output_path,
            name="small-mptrj",
            metadata="compact",
            energy_key="corrected_total_energy",
            allow_missing_stress=False,
            skip_invalid=False,
            limit=None,
            max_materials=None,
            radial_cutoff=2.0,
            angle_cutoff=2.0,
            num_process=3,
            num_samples=None,
            overwrite=False,
        )
    )

    dataset = ShardedGraphDataset(output_path)

    assert len(dataset) == 5
    assert dataset.metadata["num_process"] == 3
    assert dataset.metadata["samples_per_shard"] == 2
    assert dataset.metadata["num_shards"] == 3
    assert len(dataset._shard_datasets) == 3


def _load_mptrj_converter():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "pretrained"
        / "MPtrj"
        / "data"
        / "convert_mptrj_to_gptff.py"
    )
    spec = importlib.util.spec_from_file_location("convert_mptrj_to_gptff", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
