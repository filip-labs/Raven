"""Synthetic CPU-only overlay geometry, state, CLI and mocked viewer tests."""

import json
import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from raven.cli import main
from raven.core.point_cloud import PointCloud
from raven.perception.data.kitti_annotations import parse_kitti_annotation
from raven.perception.data.kitti_calibration import (
    KittiCalibration,
    transform_annotation,
)
from raven.perception.detection.base import Detection3D
from raven.perception.detection.predictions import detections_to_document, save_detections
from raven.perception.visualization import viewer
from raven.perception.visualization.boxes import (
    BOX_EDGES,
    class_color,
    load_ground_truth_boxes,
    prediction_corners,
    prediction_wireframes,
)
from raven.perception.visualization.coloring import point_colors
from raven.perception.visualization.overlays import HELP, OverlayState


def prediction(label="Car", **kwargs):
    return Detection3D((10, -1, -2), (6, 4, 2), 0, label, 0.9, "kitti_velodyne", **kwargs)


@pytest.fixture
def ground_truth():
    axes = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=float)
    calib = KittiCalibration(
        {"R0_rect": np.eye(3), "Tr_velo_to_cam": np.column_stack((axes, [0, 0, 0]))}
    )
    annotation = parse_kitti_annotation("Car 0 0 0 10 20 100 200 2 4 6 1 3 10 0")
    return transform_annotation(annotation, calib)


@pytest.mark.parametrize("yaw", [0, np.pi / 2, -0.7, 2 * np.pi + 0.4])
def test_prediction_geometry_exact_pose_and_input_preservation(yaw):
    box = replace(prediction(), yaw_rad=yaw)
    points = prediction_corners(box, "kitti_velodyne")
    local = np.array(
        [
            [-3, -2, -1],
            [3, -2, -1],
            [3, 2, -1],
            [-3, 2, -1],
            [-3, -2, 1],
            [3, -2, 1],
            [3, 2, 1],
            [-3, 2, 1],
        ]
    )
    expected = np.empty_like(local, dtype=float)
    expected[:, 0] = local[:, 0] * np.cos(yaw) - local[:, 1] * np.sin(yaw) + 10
    expected[:, 1] = local[:, 0] * np.sin(yaw) + local[:, 1] * np.cos(yaw) - 1
    expected[:, 2] = local[:, 2] - 2
    np.testing.assert_allclose(points, expected, atol=1e-12)
    np.testing.assert_allclose(points.mean(axis=0), box.center_xyz)
    lengths = np.linalg.norm(
        points[np.array(BOX_EDGES)[:, 0]] - points[np.array(BOX_EDGES)[:, 1]], axis=1
    )
    np.testing.assert_allclose(sorted(lengths), [2] * 4 + [4] * 4 + [6] * 4)
    assert box.yaw_rad == yaw and box.score == 0.9


def test_ground_truth_camera_transform_and_prediction_alignment(ground_truth):
    # KITTI camera bottom (1,3,10), h=2 -> geometric center (1,2,10) -> (10,-1,-2).
    corners = prediction_corners(replace(prediction(), yaw_rad=-np.pi / 2), "kitti_velodyne")
    np.testing.assert_allclose(corners, ground_truth.corners, atol=1e-12)
    assert ground_truth.annotation.location_xyz == (1, 3, 10)
    assert ground_truth.source.position_reference == "bottom_center"


def test_dashed_style_does_not_offset_edges_and_glyph_stays_inside():
    box = prediction()
    corners = prediction_corners(box, "kitti_velodyne")
    points, edges, colors = prediction_wireframes([box], "kitti_velodyne")
    assert len(edges) == 52  # 12 edges * four dashes + four interior glyph segments.
    np.testing.assert_array_equal(colors, np.tile(class_color("Car"), (52, 1)))
    for index, (a, b) in enumerate(BOX_EDGES):
        for part in range(4):
            endpoints = points[edges[index * 4 + part]]
            np.testing.assert_allclose(
                endpoints,
                [
                    corners[a] + part / 4 * (corners[b] - corners[a]),
                    corners[a] + (part / 4 + 1 / 8) * (corners[b] - corners[a]),
                ],
            )
    glyph = points[edges[48:]].reshape(-1, 3)
    assert ((glyph >= corners.min(axis=0)) & (glyph <= corners.max(axis=0))).all()
    assert len(np.unique(glyph, axis=0)) > 2
    empty = prediction_wireframes([], "kitti_velodyne")
    assert [v.shape for v in empty] == [(0, 3), (0, 2), (0, 3)]


def test_prediction_wireframes_use_class_colors_without_changing_geometry():
    detections = [prediction("Car"), prediction("Pedestrian"), prediction("Cyclist"), prediction("Other")]
    points, edges, colors = prediction_wireframes(detections, "kitti_velodyne")
    assert points.shape == (4 * 52 * 2, 3)
    assert edges.shape == (4 * 52, 2)
    assert colors.shape == (4 * 52, 3)
    for index, detection in enumerate(detections):
        start, end = index * 52, (index + 1) * 52
        np.testing.assert_array_equal(
            colors[start:end], np.tile(class_color(detection.label), (52, 1))
        )
        expected_points, expected_edges, _ = prediction_wireframes(
            [detection], "kitti_velodyne"
        )
        np.testing.assert_allclose(points[start * 2 : end * 2], expected_points)
        np.testing.assert_array_equal(edges[start:end] - start * 2, expected_edges)


def test_overlay_help_explains_color_and_style_conventions():
    assert "Car green" in HELP
    assert "Pedestrian red" in HELP
    assert "Cyclist blue" in HELP
    assert "same class colors" in HELP
    assert "dashed" in HELP
    assert "F9" in HELP


@pytest.mark.parametrize("case", ["generic", "camera", "axis", "wrong_type"])
def test_invalid_prediction_geometry_before_open3d(case, monkeypatch):
    monkeypatch.setitem(sys.modules, "open3d", None)
    box = prediction()
    if case == "generic":
        box = replace(box, coordinate_frame="lidar")
    elif case == "camera":
        box = replace(box, coordinate_frame="camera")
    elif case == "axis":
        box = replace(box, rotation_axis="y")
    else:
        box = object()
    with pytest.raises((ValueError, TypeError)):
        viewer.view_point_cloud(
            PointCloud(np.ones((1, 4)), "000000", "kitti_velodyne"), predictions=[box]
        )


def test_independent_toggles_both_toggle_and_class_filters(ground_truth):
    boxes = [prediction("Car"), prediction("Pedestrian"), prediction("Cyclist")]
    state = OverlayState([ground_truth], boxes)
    state.apply_key(291)
    assert state.visible_ground_truth == () and len(state.visible_predictions) == 3
    state.apply_key(292)
    assert state.visible_predictions == ()
    state.apply_key(293)
    assert len(state.visible_ground_truth) == 1 and len(state.visible_predictions) == 3
    state.apply_key(293)
    assert state.visible_ground_truth == state.visible_predictions == ()
    state.apply_key(293)
    for key, label in [(294, "Car"), (295, "Pedestrian"), (296, "Cyclist")]:
        state.apply_key(key)
        assert [box.label for box in state.visible_predictions] == [label]
        assert len(state.visible_ground_truth) == (1 if label == "Car" else 0)
    state.apply_key(297)
    assert len(state.visible_predictions) == 3
    assert "score=0.9" in state.prediction_details()
    state.apply_key(292)
    assert "No visible predictions" in state.prediction_details()
    assert not state.apply_key(290) and not state.apply_key(298)
    assert "F1" in HELP and "F9" in HELP
    assert len(boxes) == 3 and state.ground_truth[0] is ground_truth


@pytest.fixture
def renderer(monkeypatch):
    point_geometry = SimpleNamespace()
    vis = Mock()
    for name in ("create_window", "add_geometry", "remove_geometry", "update_geometry"):
        getattr(vis, name).return_value = True
    vis.get_render_option.return_value = SimpleNamespace()
    callbacks = {}
    vis.register_key_action_callback.side_effect = lambda key, function: callbacks.setdefault(
        key, function
    )
    fake = SimpleNamespace(
        geometry=SimpleNamespace(
            PointCloud=lambda: point_geometry,
            LineSet=SimpleNamespace,
            TriangleMesh=SimpleNamespace(create_coordinate_frame=lambda **kw: object()),
        ),
        utility=SimpleNamespace(
            Vector3dVector=lambda a: np.array(a, copy=True),
            Vector2iVector=lambda a: np.array(a, copy=True),
        ),
        visualization=SimpleNamespace(
            Visualizer=lambda: vis, VisualizerWithKeyCallback=lambda: vis
        ),
    )
    monkeypatch.setitem(sys.modules, "open3d", fake)
    return vis, point_geometry, callbacks


def test_callbacks_preserve_camera_colors_point_size_and_confidence(renderer, ground_truth, capsys):
    vis, geometry, callbacks = renderer
    cloud = PointCloud(np.array([[0, 0, 0, 0.5], [1, 2, 3, 0.7]]), "000000", "kitti_velodyne")
    original = cloud.points.copy()

    def events():
        assert set(callbacks) == set(range(290, 299))
        calls = vis.remove_geometry.call_count
        callbacks[291](vis, 2, 0)  # Repeat ignored.
        callbacks[291](vis, 0, 0)  # Release ignored.
        callbacks[291](vis, 1, 2)  # Modified key ignored.
        assert vis.remove_geometry.call_count == calls
        callbacks[291](vis, 1, 0)  # GT off, predictions remain.
        np.testing.assert_allclose(geometry.colors, point_colors(cloud, "intensity") * 0.35)
        callbacks[292](vis, 1, 0)  # Both off restores colors.
        np.testing.assert_allclose(geometry.colors, point_colors(cloud, "intensity"))
        callbacks[293](vis, 1, 0)  # Both on.
        callbacks[295](vis, 1, 0)  # Pedestrian class absent -> no box geometry.
        np.testing.assert_allclose(geometry.colors, point_colors(cloud, "intensity"))
        callbacks[297](vis, 1, 0)  # Restore all.
        callbacks[298](vis, 1, 0)
        callbacks[290](vis, 1, 0)

    vis.run.side_effect = events
    viewer.view_point_cloud(
        cloud, color="intensity", point_size=3.5, boxes=[ground_truth], predictions=[prediction()]
    )
    assert vis.get_render_option.return_value.point_size == 3.5
    for call in vis.remove_geometry.call_args_list:
        assert call.kwargs == {"reset_bounding_box": False}
    for call in vis.add_geometry.call_args_list[2:]:
        assert call.kwargs == {"reset_bounding_box": False}
    vis.reset_view_point.assert_not_called()
    np.testing.assert_array_equal(cloud.points, original)
    vis.destroy_window.assert_called_once()
    text = capsys.readouterr().out
    assert "score=0.9" in text and "F9" in text and "Class: Pedestrian" in text


@pytest.fixture
def scene(tmp_path):
    base = tmp_path / "training"
    for kind in ("velodyne", "calib", "label_2"):
        (base / kind).mkdir(parents=True)
    scan = base / "velodyne/000000.bin"
    np.array([[10, -1, -2, 0.5]], dtype="<f4").tofile(scan)
    (base / "label_2/000000.txt").write_text("Car 0 0 0 10 20 100 200 2 4 6 1 3 10 0")
    (base / "calib/000000.txt").write_text(
        "R0_rect: 1 0 0 0 1 0 0 0 1\nTr_velo_to_cam: 0 -1 0 0 0 0 -1 0 1 0 0 0"
    )
    path = tmp_path / "synthetic_predictions.json"
    save_detections(path, [prediction()], frame_id="000000")
    return scan, path


@pytest.mark.parametrize("ground_truth", [False, True])
def test_cli_saved_predictions_and_both_overlays(scene, ground_truth, monkeypatch):
    scan, path = scene
    render = Mock()
    monkeypatch.setattr(viewer, "view_point_cloud", render)
    args = [
        "visualize",
        str(scan),
        "--predictions",
        str(path),
        "--color",
        "uniform",
        "--point-size",
        "4",
    ]
    if ground_truth:
        args.append("--ground-truth")
    main(args)
    assert render.call_args.kwargs["predictions"] == [prediction()]
    assert render.call_args.kwargs["point_size"] == 4
    if ground_truth:
        np.testing.assert_array_equal(
            render.call_args.kwargs["boxes"][0].corners, load_ground_truth_boxes(scan)[0].corners
        )


@pytest.mark.parametrize(
    "case", ["missing", "frame", "invalid_json", "bad_schema", "bad_conventions", "bad_type"]
)
def test_cli_rejects_invalid_capture_before_render(scene, case, monkeypatch, capsys):
    scan, path = scene
    render = Mock()
    monkeypatch.setattr(viewer, "view_point_cloud", render)
    if case == "missing":
        path.unlink()
    elif case == "invalid_json":
        path.write_text("{broken")
    else:
        doc = detections_to_document(
            [prediction()], frame_id="000001" if case == "frame" else "000000"
        )
        if case == "bad_schema":
            doc["schema_version"] = 2
        elif case == "bad_conventions":
            doc["rotation_axis"] = "y"
        elif case == "bad_type":
            doc["predictions"] = None
        path.write_text(json.dumps(doc))
    with pytest.raises(SystemExit) as exc:
        main(["visualize", str(scan), "--predictions", str(path)])
    assert exc.value.code == 2 and "error:" in capsys.readouterr().err
    render.assert_not_called()


def test_gpu_free_cli_saved_prediction_loading(scene):
    scan, path = scene
    program = """
import sys, importlib.abc
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "pcdet", "spconv", "cumm", "open3d"}:
            raise AssertionError("GPU/GUI import before render: " + fullname)
sys.meta_path.insert(0, Block())
from raven.perception.visualization import viewer
called = []
viewer.view_point_cloud = lambda *args, **kwargs: called.append(kwargs)
from raven.cli import main
main(["visualize", sys.argv[1], "--predictions", sys.argv[2]])
assert len(called[0]["predictions"]) == 1
"""
    subprocess.run(
        [sys.executable, "-c", program, str(scan), str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
