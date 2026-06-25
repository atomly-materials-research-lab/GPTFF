import importlib.util
import json
import sys
from pathlib import Path


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
