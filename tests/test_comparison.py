import copy
import json
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from raven.cli import main
from raven.core.point_cloud import PointCloud
from raven.perception.visualization.comparison import (
    _CameraLink,
    prepare_comparison,
    view_cloud_comparison,
)


def cloud(rows=None, frame="000000", coordinate_frame="kitti_velodyne"):
    rows = [[0, 0, 0, 0.2], [3, 4, 0, 0.5], [0, 0, 10, 0.9]] if rows is None else rows
    return PointCloud(np.array(rows).reshape(-1, 4), frame, coordinate_frame)


@pytest.mark.parametrize("rows,percent", [(None, 0), ([[3, 4, 0, 0.5]], 200 / 3), ([], 100)])
def test_counts_shared_color_scale_bounds_and_immutability(rows, percent):
    raw, processed = cloud(), cloud(rows)
    original = raw.points.tobytes(), processed.points.tobytes()
    data = prepare_comparison(raw, processed)
    assert data.raw_count == 3 and data.processed_count == processed.count
    assert data.reduction_percent == pytest.approx(percent)
    np.testing.assert_array_equal(data.bounds_min, [0, 0, 0])
    np.testing.assert_array_equal(data.bounds_max, [3, 4, 10])
    if processed.count == 1:
        np.testing.assert_allclose(data.processed_colors, [[0.5, 0.5, 0.2]])  # Same 10m max as raw.
    assert (raw.points.tobytes(), processed.points.tobytes()) == original
    assert not np.shares_memory(data.raw_colors, raw.points)
    assert not data.raw_colors.flags.writeable


@pytest.mark.parametrize("mode", ["intensity", "uniform"])
def test_coloring_semantics_match_existing_modes(mode):
    raw = cloud([[1, 0, 0, -2], [2, 0, 0, 0.5], [3, 0, 0, 2]])
    processed = cloud([[2, 0, 0, 0.5]])
    data = prepare_comparison(raw, processed, mode)
    expected = (
        [[0, 0, 0], [0.5, 0.5, 0.5], [1, 1, 1]] if mode == "intensity" else [[0.7, 0.7, 0.7]] * 3
    )
    np.testing.assert_allclose(data.raw_colors, expected)
    np.testing.assert_allclose(data.processed_colors, [expected[1]])


def test_union_camera_bounds_include_processed_noise_without_translating_points():
    raw, processed = cloud([[0, 0, 0, 1]]), cloud([[2, -3, 4, 1]])
    data = prepare_comparison(raw, processed)
    np.testing.assert_array_equal(data.bounds_min, [0, -3, 0])
    np.testing.assert_array_equal(data.bounds_max, [2, 0, 4])
    same = prepare_comparison(raw, raw)
    assert np.all(same.bounds_max > same.bounds_min)
    np.testing.assert_array_equal(raw.points, [[0, 0, 0, 1]])


@pytest.mark.parametrize(
    "processed,error",
    [(cloud(frame="other"), "frame IDs"), (cloud(coordinate_frame="camera"), "coordinate frames")],
)
def test_mismatched_frames_rejected_before_open3d(processed, error, monkeypatch):
    monkeypatch.setitem(sys.modules, "open3d", None)
    with pytest.raises(ValueError, match=error):
        view_cloud_comparison(cloud(), processed)


def test_empty_original_and_invalid_color_rejected():
    with pytest.raises(ValueError, match="nonempty original"):
        prepare_comparison(cloud([]), cloud([]))
    with pytest.raises(ValueError, match="Unknown color"):
        prepare_comparison(cloud(), cloud(), "bad")
    with pytest.raises(TypeError, match="PointCloud"):
        prepare_comparison(cloud(), None)


class Visualizer:
    def __init__(self):
        self.state = {
            "class_name": "ViewTrajectory",
            "trajectory": [
                {
                    "boundingbox_min": [-1, -1, -1],
                    "boundingbox_max": [1, 1, 1],
                    "front": [0, 0, 1],
                    "up": [0, 1, 0],
                    "lookat": [0, 0, 0],
                    "zoom": 0.7,
                    "field_of_view": 60,
                }
            ],
        }
        self.options = SimpleNamespace(point_size=2.0)
        self.create_window = Mock(return_value=True)
        self.add_geometry = Mock(return_value=True)
        self.destroy_window = Mock()
        self.update_renderer = Mock()
        self.poll_events = Mock(side_effect=[True, False])
        self.callbacks = {}

    def get_view_status(self):
        return json.dumps(self.state)

    def set_view_status(self, payload):
        self.state = json.loads(payload)

    def get_render_option(self):
        return self.options

    def register_key_action_callback(self, key, callback):
        self.callbacks[key] = callback


@pytest.fixture
def renderer(monkeypatch):
    visualizers = [Visualizer(), Visualizer()]
    geometries = []

    def point_cloud():
        obj = SimpleNamespace()
        geometries.append(obj)
        return obj

    def native_vector(values):
        assert values.dtype == np.float64 and values.flags.writeable
        return values.copy()

    fake = SimpleNamespace(
        visualization=SimpleNamespace(VisualizerWithKeyCallback=Mock(side_effect=visualizers)),
        geometry=SimpleNamespace(
            PointCloud=point_cloud,
            TriangleMesh=SimpleNamespace(create_coordinate_frame=Mock(return_value=object())),
        ),
        utility=SimpleNamespace(Vector3dVector=native_vector),
    )
    monkeypatch.setitem(sys.modules, "open3d", fake)
    monkeypatch.setenv("DISPLAY", ":mock")
    return visualizers, geometries


@pytest.mark.parametrize("empty", [False, True])
def test_render_boundary_no_mutation_shared_cameras_statistics_and_cleanup(renderer, empty, capsys):
    visualizers, geometries = renderer
    raw, processed = cloud(), cloud([] if empty else [[3, 4, 0, 0.5]])
    original = raw.points.tobytes(), processed.points.tobytes()
    view_cloud_comparison(raw, processed, point_size=3)
    assert len(geometries) == (1 if empty else 2)
    np.testing.assert_array_equal(geometries[0].points, raw.points[:, :3])
    if not empty:
        np.testing.assert_array_equal(geometries[1].points, processed.points[:, :3])
    assert (raw.points.tobytes(), processed.points.tobytes()) == original
    assert visualizers[0].state == visualizers[1].state
    frame = visualizers[0].state["trajectory"][0]
    assert frame["boundingbox_max"] == [3, 4, 10]
    assert frame["lookat"] == [1.5, 2, 5]
    assert "RAW 3" in capsys.readouterr().out
    for vis in visualizers:
        assert vis.options.point_size == 3
        assert vis.create_window.call_args.kwargs["width"] == 640
        vis.destroy_window.assert_called_once()
    title = visualizers[1].create_window.call_args.kwargs["window_name"]
    assert "PROCESSED" in title and ("EMPTY" in title if empty else "-66.67%" in title)


def test_camera_changes_in_either_window_copy_pose_zoom_fov_bounds_and_point_size():
    visualizers = [Visualizer(), Visualizer()]
    link = _CameraLink(visualizers, prepare_comparison(cloud(), cloud()))
    initial = copy.deepcopy(link.initial)
    for index in [0, 1]:
        frame = visualizers[index].state["trajectory"][0]
        frame.update(
            front=[1, 0, index],
            up=[0, 0, 1],
            lookat=[2, 3, index],
            zoom=0.3 + index / 10,
            field_of_view=35 + index,
        )
        visualizers[index].options.point_size = 4 + index
        link.tick()
        assert visualizers[0].state == visualizers[1].state
        assert visualizers[0].options.point_size == visualizers[1].options.point_size == 4 + index
        assert visualizers[index].state["trajectory"][0]["boundingbox_max"] == [3, 4, 10]
    assert link.initial == initial
    link.apply(link.initial, 2)
    assert visualizers[0].state == initial


def test_help_match_reset_callbacks_do_not_repeat_on_release_or_modifier(renderer):
    visualizers, _ = renderer
    view_cloud_comparison(cloud(), cloud())
    raw = visualizers[0]
    raw.state["trajectory"][0]["zoom"] = 0.2
    raw.callbacks[291](raw, 0, 0)  # release ignored
    assert visualizers[1].state["trajectory"][0]["zoom"] == 0.7
    raw.callbacks[291](raw, 1, 1)  # modified press ignored
    assert visualizers[1].state["trajectory"][0]["zoom"] == 0.7
    raw.callbacks[291](raw, 1, 0)
    assert visualizers[1].state["trajectory"][0]["zoom"] == 0.2
    raw.callbacks[292](raw, 1, 0)
    assert visualizers[0].state["trajectory"][0]["zoom"] == 0.7
    assert raw.create_window.call_count == 2


@pytest.mark.parametrize(
    "failure", ["first_window", "second_window", "geometry", "camera", "events", "cleanup"]
)
def test_gui_failures_cleanup_all_attempted_windows(renderer, failure):
    visualizers, _ = renderer
    if failure == "first_window":
        visualizers[0].create_window.return_value = False
    elif failure == "second_window":
        visualizers[1].create_window.return_value = False
    elif failure == "geometry":
        visualizers[1].add_geometry.return_value = False
    elif failure == "camera":
        visualizers[0].get_view_status = lambda: "{}"
    elif failure == "events":
        visualizers[0].poll_events.side_effect = RuntimeError("event failure")
    else:
        visualizers[0].destroy_window.side_effect = RuntimeError("cleanup failure")
    with pytest.raises(RuntimeError, match="Open3D"):
        view_cloud_comparison(cloud(), cloud())
    visualizers[0].destroy_window.assert_called_once()
    if failure != "first_window":
        visualizers[1].destroy_window.assert_called_once()


@pytest.mark.parametrize("size", [0, 26, True, np.nan, np.inf, [], "invalid"])
def test_invalid_point_size_before_dependency_import(size, monkeypatch):
    monkeypatch.setitem(sys.modules, "open3d", None)
    with pytest.raises(ValueError, match="point size"):
        view_cloud_comparison(cloud(), cloud(), point_size=size)


def test_missing_optional_open3d(monkeypatch):
    monkeypatch.setitem(sys.modules, "open3d", None)
    with pytest.raises(RuntimeError, match=r"pip install.*visualization"):
        view_cloud_comparison(cloud(), cloud())


@pytest.fixture
def frame(tmp_path):
    path = tmp_path / "000000.bin"
    cloud().points.astype("<f4").tofile(path)
    return path


@pytest.mark.parametrize(
    "stages,count",
    [
        ([], 3),
        ([{"name": "range_filter", "x_range": [0, 1], "y_range": [-1, 1], "z_range": [-1, 1]}], 1),
        ([{"name": "voxel_downsample", "voxel_size": 100}], 1),
        ([{"name": "noise", "position_std": 0.01, "seed": 42}], 3),
    ],
)
def test_cli_uses_existing_stage_list_and_runs_pipeline_once(
    frame, tmp_path, monkeypatch, stages, count
):
    from raven.perception.preprocessing import pipeline
    from raven.perception.visualization import comparison

    config = tmp_path / "stages.json"
    config.write_text(json.dumps(stages))
    run = Mock(wraps=pipeline.run_preprocessing_pipeline)
    show = Mock()
    monkeypatch.setattr(pipeline, "run_preprocessing_pipeline", run)
    monkeypatch.setattr(comparison, "view_cloud_comparison", show)
    original = frame.read_bytes()
    main(
        [
            "visualize",
            str(frame),
            "--compare",
            str(config),
            "--color",
            "intensity",
            "--point-size",
            "3",
        ]
    )
    run.assert_called_once()
    args, kwargs = show.call_args
    assert args[0].count == 3 and args[1].count == count
    assert kwargs == {"color": "intensity", "point_size": 3.0}
    assert frame.read_bytes() == original


@pytest.mark.parametrize(
    "payload",
    [
        "{",
        "{}",
        '[{"name":"noise","position_std":0.1}]',
        '[{"name":"unknown"}]',
        '[{"name":"noise","position_std":NaN,"seed":42}]',
        '[{"name":"noise","position_std":0,"seed":1,"seed":2}]',
    ],
)
def test_cli_rejects_malformed_config_before_viewer(frame, tmp_path, monkeypatch, capsys, payload):
    from raven.perception.visualization import comparison

    show = Mock()
    monkeypatch.setattr(comparison, "view_cloud_comparison", show)
    config = tmp_path / "stages.json"
    config.write_text(payload)
    with pytest.raises(SystemExit) as error:
        main(["visualize", str(frame), "--compare", str(config)])
    assert error.value.code == 2 and "error:" in capsys.readouterr().err
    show.assert_not_called()


def test_cli_missing_config_and_open3d_errors(frame, tmp_path, monkeypatch, capsys):
    path = tmp_path / "missing.json"
    with pytest.raises(SystemExit):
        main(["visualize", str(frame), "--compare", str(path)])
    assert "missing.json" in capsys.readouterr().err
    path.write_text("[]")
    monkeypatch.setitem(sys.modules, "open3d", None)
    with pytest.raises(SystemExit):
        main(["visualize", str(frame), "--compare", str(path)])
    assert "visualization" in capsys.readouterr().err


@pytest.mark.parametrize("flag", ["--ground-truth", "--predictions"])
def test_comparison_flags_cannot_silently_discard_overlays(frame, flag, capsys):
    extra = [flag] if flag == "--ground-truth" else [flag, "predictions.json"]
    with pytest.raises(SystemExit):
        main(["visualize", str(frame), "--compare", "stages.json", *extra])
    assert "cannot be combined" in capsys.readouterr().err


def test_headless_import_and_preparation_do_not_import_optional_frameworks():
    program = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'open3d','scipy','torch','pcdet','spconv','cumm'}:
            raise AssertionError('Unexpected optional dependency: '+fullname)
sys.meta_path.insert(0,Block())
import numpy as np
from raven.core.point_cloud import PointCloud
from raven.perception.visualization.comparison import prepare_comparison
raw=PointCloud(np.zeros((1,4)),'frame')
assert prepare_comparison(raw,raw).reduction_percent == 0
"""
    subprocess.run([sys.executable, "-c", program], check=True, capture_output=True, text=True)


def test_explicit_empty_config_path_does_not_fall_back_to_unprocessed_view(frame, monkeypatch):
    from raven.perception.visualization import viewer

    show = Mock()
    monkeypatch.setattr(viewer, "view_point_cloud", show)
    with pytest.raises(SystemExit):
        main(["visualize", str(frame), "--compare", ""])
    show.assert_not_called()


def test_worker_thread_rejected_before_open3d_import(monkeypatch):
    import threading

    monkeypatch.setitem(sys.modules, "open3d", None)
    errors = []

    def run():
        try:
            view_cloud_comparison(cloud(), cloud())
        except RuntimeError as error:
            errors.append(str(error))

    worker = threading.Thread(target=run)
    worker.start()
    worker.join()
    assert len(errors) == 1 and "main thread" in errors[0]


def test_linux_without_display_does_not_initialize_renderer(renderer, monkeypatch):
    from raven.perception.visualization import comparison

    monkeypatch.setattr(comparison.sys, "platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    with pytest.raises(RuntimeError, match="graphical desktop"):
        view_cloud_comparison(cloud(), cloud())
    for vis in renderer[0]:
        vis.create_window.assert_not_called()
