import json
import sys
from unittest.mock import Mock

import numpy as np
import pytest

from raven.cli import main
from raven.perception.visualization import viewer


@pytest.fixture
def frame(tmp_path):
    path = tmp_path / "000000.bin"
    np.array([[1, 2, 3, 0.5], [4, 5, 6, 0.7]], dtype="<f4").tofile(path)
    return path


@pytest.mark.parametrize(
    "args",
    [
        ["--help"],
        ["visualize", "--help"],
        ["inspect-labels", "--help"],
        ["inspect-calib", "--help"],
    ],
)
def test_help_without_open3d(args, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "open3d", None)
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 0
    expected = "visualize" if args == ["--help"] else args[0]
    assert expected in capsys.readouterr().out


def test_inspect_json_without_open3d(frame, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "open3d", None)
    main(["inspect", str(frame)])
    assert json.loads(capsys.readouterr().out) == {
        "frame_id": "000000",
        "coordinate_frame": "kitti_velodyne",
        "points": 2,
        "min_xyz": [1, 2, 3],
        "max_xyz": [4, 5, 6],
    }


def test_inspect_empty_json(frame, capsys):
    frame.write_bytes(b"")
    main(["inspect", str(frame)])
    summary = json.loads(capsys.readouterr().out)
    assert summary["points"] == 0
    assert summary["min_xyz"] is None and summary["max_xyz"] is None


@pytest.mark.parametrize("mode", ["distance", "intensity", "uniform"])
def test_visualize_passes_options(frame, monkeypatch, mode):
    render = Mock()
    monkeypatch.setattr(viewer, "view_point_cloud", render)
    main(["visualize", str(frame), "--color", mode, "--point-size", "3.5"])
    cloud = render.call_args.args[0]
    assert cloud.count == 2
    assert cloud.coordinate_frame == "kitti_velodyne"
    assert render.call_args.kwargs == {"color": mode, "point_size": 3.5}


def test_visualize_defaults(frame, monkeypatch):
    render = Mock()
    monkeypatch.setattr(viewer, "view_point_cloud", render)
    main(["visualize", str(frame)])
    assert render.call_args.kwargs == {"color": "distance", "point_size": 2.0}


@pytest.mark.parametrize(
    "options",
    [
        ["--color", "invalid"],
        ["--point-size", "0"],
        ["--point-size", "nan"],
        ["--point-size", "inf"],
        ["--point-size", "26"],
        ["--point-size", "abc"],
    ],
)
def test_invalid_options_do_not_render(frame, options, monkeypatch, capsys):
    render = Mock()
    monkeypatch.setattr(viewer, "view_point_cloud", render)
    with pytest.raises(SystemExit) as exc:
        main(["visualize", str(frame), *options])
    assert exc.value.code == 2
    assert "error:" in capsys.readouterr().err
    render.assert_not_called()


@pytest.mark.parametrize(
    ("command", "bad_file"),
    [
        (command, bad_file)
        for command in ["inspect", "visualize"]
        for bad_file in ["missing", "malformed", "nonfinite", "empty"]
        if (command, bad_file) != ("inspect", "empty")
    ],
)
def test_input_errors_are_actionable(command, bad_file, frame, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "open3d", None)
    if bad_file == "missing":
        path = frame.with_name("missing.bin")
        message = "not found"
    else:
        path = frame
        if bad_file == "malformed":
            path.write_bytes(b"123")
            message = "file size"
        elif bad_file == "nonfinite":
            np.array([[np.nan, 0, 0, 0]], dtype="<f4").tofile(path)
            message = "NaN or infinity"
        else:
            path.write_bytes(b"")
            message = "empty point cloud"
    with pytest.raises(SystemExit) as exc:
        main([command, str(path)])
    assert exc.value.code == 2
    stderr = capsys.readouterr().err
    assert message in stderr and "Traceback" not in stderr


def test_visualize_reports_optional_dependency(frame, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "open3d", None)
    with pytest.raises(SystemExit) as exc:
        main(["visualize", str(frame)])
    assert exc.value.code == 2
    assert "visualization" in capsys.readouterr().err


@pytest.mark.parametrize("option", ["--labels", "--calib"])
def test_explicit_ground_truth_paths_require_flag(frame, option, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["visualize", str(frame), option, "000000.txt"])
    assert exc.value.code == 2
    assert "require --ground-truth" in capsys.readouterr().err


def test_ground_truth_cli_explicit_paths_preserves_options(frame, tmp_path, monkeypatch):
    labels = tmp_path / "labels/000000.txt"
    calib = tmp_path / "calib/000000.txt"
    labels.parent.mkdir()
    calib.parent.mkdir()
    labels.write_text("Pedestrian 0 0 0 10 20 100 200 2 1 1 1 2 10 0")
    calib.write_text("R0_rect: 1 0 0 0 1 0 0 0 1\nTr_velo_to_cam: 0 -1 0 0 0 0 -1 0 1 0 0 0")
    render = Mock()
    monkeypatch.setattr(viewer, "view_point_cloud", render)
    main(
        [
            "visualize",
            str(frame),
            "--ground-truth",
            "--labels",
            str(labels),
            "--calib",
            str(calib),
            "--color",
            "intensity",
            "--point-size",
            "3.5",
        ]
    )
    assert render.call_args.kwargs["color"] == "intensity"
    assert render.call_args.kwargs["point_size"] == 3.5
    boxes = render.call_args.kwargs["boxes"]
    assert len(boxes) == 1 and boxes[0].label == "Pedestrian"
    assert boxes[0].coordinate_frame == render.call_args.args[0].coordinate_frame


def test_ground_truth_cli_error_does_not_render(frame, monkeypatch, capsys):
    render = Mock()
    monkeypatch.setattr(viewer, "view_point_cloud", render)
    with pytest.raises(SystemExit) as exc:
        main(["visualize", str(frame), "--ground-truth"])
    assert exc.value.code == 2
    assert "nonstandard layouts" in capsys.readouterr().err
    render.assert_not_called()
