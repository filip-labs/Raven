import json
import struct
import sys
import zlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from raven.cli import main
from raven.perception.detection.predictions import save_detections
from raven.perception.visualization import comparison, scenes, viewer
from raven.perception.visualization.scenes import FrameSequence, SceneLoadError, export_screenshots


def png():
    def chunk(kind, data):
        return (
            struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"\0\xff\x00\x00"))
        + chunk(b"IEND", b"")
    )


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "training"
    for name in ["velodyne", "label_2", "calib", "predictions"]:
        (root / name).mkdir(parents=True)
    for index, label in [(0, "Car"), (1, "Cyclist")]:
        stem = f"{index:06d}"
        np.array([[index * 10, 0, 0, 0.2], [index * 10 + 1, 0, 0, 0.8]], dtype="<f4").tofile(
            root / "velodyne" / f"{stem}.bin"
        )
        (root / "label_2" / f"{stem}.txt").write_text(
            f"{label} 0 0 0 0 0 20 20 2 2 4 {index + 1} 2 10 0\n"
        )
        (root / "calib" / f"{stem}.txt").write_text(
            "R0_rect: 1 0 0 0 1 0 0 0 1\nTr_velo_to_cam: 1 0 0 0 0 1 0 0 0 0 1 0\n"
        )
        save_detections(root / "predictions" / f"{stem}.json", [], frame_id=stem)
    return root


def test_filename_sort_start_and_boundaries_load_only_active(dataset, monkeypatch, capsys):
    root = dataset / "velodyne"
    (root / "10.bin").write_bytes((root / "000000.bin").read_bytes())
    (root / "2.bin").write_bytes(b"corrupt")
    load = Mock(wraps=scenes.load_kitti_bin)
    monkeypatch.setattr(scenes, "load_kitti_bin", load)
    seq = FrameSequence(root, start_frame="000001")
    assert [p.name for p in seq.paths] == ["000000.bin", "000001.bin", "10.bin", "2.bin"]
    assert seq.status() == "000001 [2/4]" and load.call_count == 1
    seq.move(-1)
    assert seq.move(-1) is None and seq.status() == "000000 [1/4]"
    assert load.call_count == 2 and "no wrapping" in capsys.readouterr().out
    seq.move(1)
    seq.move(1)
    old = seq.active
    with pytest.raises(SceneLoadError, match="2.bin"):
        seq.move(1)
    assert seq.active is old


@pytest.mark.parametrize("kind", ["missing", "file", "empty", "start"])
def test_invalid_directory_and_start(dataset, tmp_path, kind):
    path = {
        "missing": tmp_path / "none",
        "file": dataset / "velodyne/000000.bin",
        "empty": tmp_path,
        "start": dataset / "velodyne",
    }[kind]
    with pytest.raises(ValueError):
        FrameSequence(path, start_frame="absent" if kind == "start" else None)


@pytest.mark.parametrize(
    "payload", [b"", b"bad", np.array([[np.nan, 0, 0, 1]], dtype="<f4").tobytes()]
)
def test_invalid_candidate_keeps_active(dataset, payload):
    seq = FrameSequence(dataset / "velodyne")
    original = seq.active
    (dataset / "velodyne/000001.bin").write_bytes(payload)
    with pytest.raises(SceneLoadError, match="000001.bin"):
        seq.move(1)
    assert seq.active is original


def test_disappeared_candidate_and_last_boundary(dataset):
    seq = FrameSequence(dataset / "velodyne", start_frame="000001")
    assert seq.move(1) is None
    (dataset / "velodyne/000000.bin").unlink()
    with pytest.raises(SceneLoadError, match="000000.bin"):
        seq.move(-1)
    assert seq.active.cloud.frame_id == "000001"


def test_ground_truth_prediction_files_are_reloaded_and_frame_matched(dataset):
    seq = FrameSequence(
        dataset / "velodyne", ground_truth=True, predictions_dir=dataset / "predictions"
    )
    assert seq.active.boxes[0].label == "Car"
    seq.move(1)
    assert seq.active.boxes[0].label == "Cyclist"
    (dataset / "predictions/000000.json").unlink()
    save_detections(dataset / "predictions/000000.json", [], frame_id="000099")
    old = seq.active
    with pytest.raises(SceneLoadError, match="frame"):
        seq.move(-1)
    assert seq.active is old


@pytest.mark.parametrize(
    "missing", ["label_2/000001.txt", "calib/000001.txt", "predictions/000001.json"]
)
def test_missing_overlay_never_reuses_previous_frame(dataset, missing):
    seq = FrameSequence(
        dataset / "velodyne", ground_truth=True, predictions_dir=dataset / "predictions"
    )
    (dataset / missing).unlink()
    old = seq.active
    with pytest.raises(SceneLoadError):
        seq.move(1)
    assert seq.active is old and old.boxes[0].label == "Car"


class Vis:
    def __init__(self):
        self.options = SimpleNamespace(point_size=2.0)
        self.state = {
            "trajectory": [
                {
                    "boundingbox_min": [-1, -1, -1],
                    "boundingbox_max": [1, 1, 1],
                    "lookat": [0, 0, 0],
                    "front": [0, 0, 1],
                    "up": [0, 1, 0],
                    "zoom": 0.7,
                    "field_of_view": 60,
                }
            ]
        }
        self.callbacks = {}
        self.live = {}
        self.create_window = Mock(return_value=True)
        self.add_geometry = Mock(side_effect=self.add)
        self.remove_geometry = Mock(side_effect=self.remove)
        self.update_geometry = Mock(return_value=True)
        self.update_renderer = Mock()
        self.destroy_window = Mock()
        self.run = Mock()
        self.poll_events = Mock(return_value=False)
        self.reset_view_point = Mock()
        self.capture_screen_image = Mock(
            side_effect=lambda path, do_render: Path(path).write_bytes(png())
        )

    def add(self, obj, **kw):
        self.live[id(obj)] = obj
        return True

    def remove(self, obj, **kw):
        self.live.pop(id(obj))
        return True

    def register_key_action_callback(self, key, fn):
        self.callbacks[key] = fn

    def get_render_option(self):
        return self.options

    def get_view_status(self):
        return json.dumps(self.state)

    def set_view_status(self, state):
        self.state = json.loads(state)


@pytest.fixture
def render(monkeypatch):
    objects = [Vis(), Vis()]
    factory = Mock(side_effect=objects)
    fake = SimpleNamespace(
        visualization=SimpleNamespace(Visualizer=factory, VisualizerWithKeyCallback=factory),
        geometry=SimpleNamespace(
            PointCloud=SimpleNamespace,
            LineSet=SimpleNamespace,
            TriangleMesh=SimpleNamespace(create_coordinate_frame=lambda **kw: object()),
        ),
        utility=SimpleNamespace(
            Vector3dVector=lambda v: np.array(v, copy=True),
            Vector2iVector=lambda v: np.array(v, copy=True),
        ),
    )
    monkeypatch.setitem(sys.modules, "open3d", fake)
    monkeypatch.setenv("DISPLAY", ":mock")
    return objects, factory


def test_viewer_navigation_preserves_camera_filters_and_replaces_geometry(
    dataset, render, tmp_path
):
    seq = FrameSequence(dataset / "velodyne", ground_truth=True)
    vis = render[0][0]
    original = (dataset / "velodyne/000000.bin").read_bytes()

    def actions():
        vis.state["trajectory"][0]["zoom"] = 0.3
        camera = vis.get_view_status()
        vis.callbacks[294](vis, 1, 0)  # Car only remains selected across the transition.
        vis.callbacks[300](vis, 1, 0)
        assert seq.status() == "000001 [2/2]" and vis.get_view_status() == camera
        assert len(vis.live) == 2  # cloud + axes; Cyclist is hidden by persistent Car filter.
        vis.callbacks[297](vis, 1, 0)
        assert len(vis.live) == 3
        vis.callbacks[299](vis, 1, 0)
        assert seq.active.boxes[0].label == "Car" and len(vis.live) == 3
        vis.callbacks[301](vis, 1, 0)
        vis.reset_view_point.assert_called_with(reset_bounding_box=True)
        vis.callbacks[301](vis, 1, 1)

    vis.run.side_effect = actions
    viewer.view_point_cloud(seq.active.cloud, sequence=seq, screenshot_dir=tmp_path / "shots")
    assert render[1].call_count == 1
    assert (dataset / "velodyne/000000.bin").read_bytes() == original
    assert (tmp_path / "shots/raven_000000_view_0001.png").is_file()
    vis.destroy_window.assert_called_once()


def test_comparison_navigation_runs_processing_once_per_selection_and_preserves_camera(
    dataset, render, monkeypatch, tmp_path
):
    run = Mock(wraps=scenes.run_preprocessing_pipeline)
    monkeypatch.setattr(scenes, "run_preprocessing_pipeline", run)
    seq = FrameSequence(
        dataset / "velodyne", stages=[{"name": "voxel_downsample", "voxel_size": 2}]
    )
    a, b = render[0]

    def events():
        a.state["trajectory"][0]["zoom"] = 0.4
        camera = a.get_view_status()
        a.callbacks[300](a, 1, 0)
        assert seq.active.cloud.frame_id == "000001" and run.call_count == 2
        assert a.get_view_status() == b.get_view_status() == camera
        assert len(a.live) == len(b.live) == 2
        a.callbacks[301](a, 1, 1)
        assert run.call_count == 2
        b.callbacks[299](b, 1, 0)
        assert seq.active.cloud.frame_id == "000000" and run.call_count == 3
        a.callbacks[301](a, 1, 0)
        assert a.state["trajectory"][0]["lookat"] == [0.5, 0, 0]
        return False

    a.poll_events.side_effect = events
    comparison.view_cloud_comparison(
        seq.active.cloud, seq.active.processed, sequence=seq, screenshot_dir=tmp_path / "shots"
    )
    assert render[1].call_count == 2
    assert sorted(p.name for p in (tmp_path / "shots").iterdir()) == [
        "raven_000001_processed_0001.png",
        "raven_000001_raw_0001.png",
    ]
    for vis in (a, b):
        vis.destroy_window.assert_called_once()


def test_candidate_error_in_callback_keeps_geometry_and_viewer_alive(dataset, render, capsys):
    seq = FrameSequence(dataset / "velodyne")
    (dataset / "velodyne/000001.bin").write_bytes(b"bad")
    vis = render[0][0]

    def actions():
        vis.callbacks[300](vis, 1, 0)
        assert seq.active.cloud.frame_id == "000000" and len(vis.live) == 2

    vis.run.side_effect = actions
    viewer.view_point_cloud(seq.active.cloud, sequence=seq)
    assert "keeping current frame" in capsys.readouterr().out
    vis.destroy_window.assert_called_once()


def test_export_numbering_pair_collision_and_no_overwrite(tmp_path):
    a, b = Vis(), Vis()
    existing = tmp_path / "raven_000000_processed_0001.png"
    existing.write_bytes(b"original")
    paths = export_screenshots([a, b], "000000", tmp_path, roles=("raw", "processed"))
    assert [p.name for p in paths] == [
        "raven_000000_raw_0002.png",
        "raven_000000_processed_0002.png",
    ]
    assert existing.read_bytes() == b"original"
    assert not (tmp_path / "raven_000000_raw_0001.png").exists()
    again = export_screenshots([a, b], "000000", tmp_path, roles=("raw", "processed"))
    assert all(p.name.endswith("0003.png") for p in again)
    a.capture_screen_image.assert_called_with(str(again[0]), do_render=True)


@pytest.mark.parametrize("failure", ["raise", "empty", "truncated", "crc"])
def test_export_failure_rolls_back_only_owned_pair_files(tmp_path, failure):
    a, b = Vis(), Vis()
    (tmp_path / "keep.png").write_bytes(b"keep")
    if failure == "raise":
        b.capture_screen_image.side_effect = RuntimeError("render failed")
    else:
        payload = (
            b""
            if failure == "empty"
            else png()[:-2]
            if failure == "truncated"
            else png()[:20] + b"bad" + png()[23:]
        )
        b.capture_screen_image.side_effect = lambda path, do_render: Path(path).write_bytes(payload)
    with pytest.raises(RuntimeError, match="Screenshot export failed"):
        export_screenshots([a, b], "000000", tmp_path, roles=("raw", "processed"))
    assert [p.name for p in tmp_path.iterdir()] == ["keep.png"]


def test_export_filesystem_and_filename_errors(tmp_path):
    file = tmp_path / "not-directory"
    file.write_text("keep")
    with pytest.raises(OSError):
        export_screenshots([Vis()], "000000", file)
    with pytest.raises(ValueError):
        export_screenshots([Vis()], "../frame", tmp_path)
    assert file.read_text() == "keep"


@pytest.mark.parametrize(
    "args",
    [
        ["--start-frame", "000000"],
        ["--predictions-dir", "preds"],
        ["--navigate", "--predictions", "one.json"],
        ["--navigate", "--ground-truth", "--calib", "one.txt"],
        ["--navigate", "--compare", "config.json", "--predictions-dir", "preds"],
    ],
)
def test_cli_rejects_unsafe_combinations(dataset, args):
    with pytest.raises(SystemExit) as error:
        main(["visualize", str(dataset / "velodyne"), *args])
    assert error.value.code == 2


@pytest.mark.parametrize("compare_mode", [False, True])
def test_cli_navigation_start_and_screenshots(dataset, monkeypatch, tmp_path, compare_mode):
    show = Mock()
    monkeypatch.setattr(
        comparison if compare_mode else viewer,
        "view_cloud_comparison" if compare_mode else "view_point_cloud",
        show,
    )
    args = [
        "visualize",
        str(dataset / "velodyne"),
        "--navigate",
        "--start-frame",
        "000001",
        "--screenshot-dir",
        str(tmp_path / "shots"),
    ]
    if compare_mode:
        config = tmp_path / "config.json"
        config.write_text("[]")
        args += ["--compare", str(config)]
    main(args)
    positional, options = show.call_args
    assert positional[0].frame_id == "000001"
    assert options["sequence"].status() == "000001 [2/2]"
    assert options["screenshot_dir"] == str(tmp_path / "shots")


def test_single_frame_screenshot_flag_keeps_normal_viewer_path(dataset, monkeypatch, tmp_path):
    show = Mock()
    monkeypatch.setattr(viewer, "view_point_cloud", show)
    main(
        [
            "visualize",
            str(dataset / "velodyne/000000.bin"),
            "--screenshot-dir",
            str(tmp_path / "shots"),
        ]
    )
    assert show.call_args.kwargs["screenshot_dir"] == str(tmp_path / "shots")


def test_screenshot_error_is_nonfatal_and_keys_ignore_releases(dataset, render, tmp_path, capsys):
    seq = FrameSequence(dataset / "velodyne")
    vis = render[0][0]
    vis.capture_screen_image.side_effect = RuntimeError("framebuffer unavailable")

    def actions():
        vis.callbacks[300](vis, 0, 0)
        vis.callbacks[300](vis, 2, 0)
        vis.callbacks[300](vis, 1, 2)
        assert seq.active.position == 0
        vis.callbacks[301](vis, 1, 1)
        assert seq.active.position == 0
        vis.callbacks[300](vis, 1, 0)
        assert seq.active.position == 1

    vis.run.side_effect = actions
    viewer.view_point_cloud(seq.active.cloud, sequence=seq, screenshot_dir=tmp_path / "shots")
    assert "Screenshot error:" in capsys.readouterr().out
    assert list((tmp_path / "shots").iterdir()) == []
    vis.destroy_window.assert_called_once()


@pytest.mark.parametrize("pair", [False, True])
def test_frame_geometry_failure_closes_all_windows(dataset, render, pair):
    seq = FrameSequence(dataset / "velodyne", stages=[] if pair else None)
    a, b = render[0]

    def actions():
        if pair:
            a.add_geometry.side_effect = None
            a.add_geometry.return_value = False
        else:
            a.update_geometry.return_value = False
        a.callbacks[300](a, 1, 0)
        return False

    if pair:
        a.poll_events.side_effect = actions
        with pytest.raises(RuntimeError, match="new frame geometry"):
            comparison.view_cloud_comparison(seq.active.cloud, seq.active.processed, sequence=seq)
        b.destroy_window.assert_called_once()
    else:
        a.run.side_effect = actions
        with pytest.raises(RuntimeError, match="update point colors"):
            viewer.view_point_cloud(seq.active.cloud, sequence=seq)
    a.destroy_window.assert_called_once()


def test_comparison_transition_to_empty_processed_and_back(dataset, render):
    stages = [{"name": "range_filter", "x_range": [-1, 2], "y_range": [-1, 1], "z_range": [-1, 1]}]
    seq = FrameSequence(dataset / "velodyne", stages=stages)
    a, b = render[0]

    def events():
        a.callbacks[300](a, 1, 0)
        assert seq.active.processed.count == 0 and len(b.live) == 1
        assert "EMPTY" in b.create_window.call_args.kwargs["window_name"]
        a.callbacks[299](a, 1, 0)
        assert seq.active.processed.count == 2 and len(b.live) == 2
        return False

    a.poll_events.side_effect = events
    comparison.view_cloud_comparison(seq.active.cloud, seq.active.processed, sequence=seq)
    for vis in (a, b):
        vis.destroy_window.assert_called_once()
