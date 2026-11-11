from dataclasses import replace

import numpy as np
import pytest

from raven.perception.data.kitti_annotations import parse_kitti_annotation
from raven.perception.data.kitti_calibration import (
    CAMERA_RECT_FRAME,
    VELODYNE_FRAME,
    KittiCalibration,
    transform_annotation,
)
from raven.perception.visualization.boxes import (
    box_wireframes,
    class_color,
    load_ground_truth_boxes,
    wireframe_from_corners,
)

LABEL = "Car 0 0 0 10 20 100 200 2 4 6 1 3 10 0"
IGNORE = "DontCare -1 -1 -10 10 20 100 200 -1 -1 -1 -1000 -1000 -1000 -10"
AXES = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=float)


@pytest.fixture
def calibration():
    return KittiCalibration(
        {"R0_rect": np.eye(3), "Tr_velo_to_cam": np.column_stack((AXES, [0, 0, 0]))}
    )


@pytest.fixture
def box(calibration):
    return transform_annotation(parse_kitti_annotation(LABEL), calibration)


@pytest.fixture
def scene(tmp_path, calibration):
    split = tmp_path / "training"
    for name in ("velodyne", "calib", "label_2"):
        (split / name).mkdir(parents=True)
    scan = split / "velodyne/000000.bin"
    np.array([[10, -1, -2, 0.5]], dtype="<f4").tofile(scan)
    labels = split / "label_2/000000.txt"
    labels.write_text(LABEL + "\n" + IGNORE + "\n")
    calib = split / "calib/000000.txt"
    calib.write_text(
        "\n".join(
            name + ": " + " ".join(str(v) for v in matrix.flat)
            for name, matrix in calibration.matrices.items()
        )
    )
    return scan, labels, calib


@pytest.mark.parametrize(
    "label,color",
    [
        ("Car", (0, 1, 0)),
        ("Pedestrian", (1, 0, 0)),
        ("Cyclist", (0, 0, 1)),
        ("Van", (1, 1, 0)),
        ("Truck", (1, 1, 0)),
        ("Unknown", (1, 1, 0)),
    ],
)
def test_class_colors(label, color):
    assert class_color(label) == color


def test_eight_corners_twelve_unique_edges_and_no_mutation(box):
    original = box.corners.copy()
    points, edges, colors = wireframe_from_corners(box.corners, box.label)
    assert points.shape == (8, 3) and edges.shape == (12, 2) and colors.shape == (12, 3)
    assert edges.min() == 0 and edges.max() == 7
    assert len({tuple(sorted(pair)) for pair in edges.tolist()}) == 12
    np.testing.assert_array_equal(np.bincount(edges.flatten(), minlength=8), [3] * 8)
    np.testing.assert_array_equal(colors, [[0, 1, 0]] * 12)
    lengths = np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1)
    np.testing.assert_allclose(sorted(lengths), [2] * 4 + [4] * 4 + [6] * 4)
    points[0] += 100
    np.testing.assert_array_equal(box.corners, original)


@pytest.mark.parametrize(
    "corners",
    [
        np.zeros((7, 3)),
        np.zeros((8, 4)),
        np.zeros((8, 3)),
        np.full((8, 3), np.nan),
        np.full((8, 3), np.inf),
        np.ones((8, 3), dtype=complex),
        ["invalid"],
    ],
)
def test_invalid_corners(corners):
    with pytest.raises(ValueError):
        wireframe_from_corners(corners, "Car")


def test_full_rotation_and_camera_to_velodyne_alignment(calibration):
    angle = 0.7
    annotation = parse_kitti_annotation(LABEL.rsplit(" ", 1)[0] + f" {angle}")
    c, s = np.cos(angle), np.sin(angle)
    camera_rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    # Independently construct camera-coordinate corners from local length/width/height.
    local = np.array(
        [
            [-3, 2, -2],
            [3, 2, -2],
            [3, 2, 2],
            [-3, 2, 2],
            [-3, 0, -2],
            [3, 0, -2],
            [3, 0, 2],
            [-3, 0, 2],
        ]
    )
    camera_corners = (local - np.array([0, 1, 0])) @ camera_rotation.T + [1, 2, 10]
    # Introduce roll via rectification to ensure the final box is not pure LiDAR yaw.
    tilt = 0.15
    rect = np.array([[1, 0, 0], [0, np.cos(tilt), -np.sin(tilt)], [0, np.sin(tilt), np.cos(tilt)]])
    tilted = KittiCalibration({**calibration.matrices, "R0_rect": rect})
    transformed = transform_annotation(annotation, tilted)
    points, _, _ = box_wireframes([transformed], VELODYNE_FRAME)
    expected = camera_corners @ np.linalg.inv(rect @ AXES).T
    np.testing.assert_allclose(points, expected, atol=1e-12)
    assert not np.allclose(transformed.rotation[:, 2], [0, 0, 1])
    assert transformed.annotation is annotation
    np.testing.assert_allclose(transformed.size_lwh, [6, 4, 2])


def test_batch_offsets_and_empty_boxes(box):
    second = replace(box, center_xyz=box.center_xyz + [1, 2, 3])
    points, edges, colors = box_wireframes([box, second], VELODYNE_FRAME)
    assert points.shape == (16, 3) and colors.shape == (24, 3)
    assert edges[:12].max() < 8 and edges[12:].min() >= 8 and edges[12:].max() == 15
    empty = box_wireframes([], VELODYNE_FRAME)
    assert [array.shape for array in empty] == [(0, 3), (0, 2), (0, 3)]


def test_misaligned_frames_rejected(calibration):
    camera_box = transform_annotation(
        parse_kitti_annotation(LABEL), calibration, target_frame=CAMERA_RECT_FRAME
    )
    with pytest.raises(ValueError, match="does not match cloud frame"):
        box_wireframes([camera_box], VELODYNE_FRAME)
    with pytest.raises(TypeError):
        box_wireframes([object()], VELODYNE_FRAME)


def test_standard_scene_resolves_siblings_and_skips_ignore(scene):
    scan, _labels, _calib = scene
    before = {path: path.read_bytes() for path in scene}
    boxes = load_ground_truth_boxes(scan)
    assert len(boxes) == 1 and boxes[0].label == "Car"
    np.testing.assert_array_equal(boxes[0].center_xyz, [10, -1, -2])
    assert boxes[0].coordinate_frame == VELODYNE_FRAME
    assert boxes[0].annotation.location_xyz == (1, 3, 10)
    assert all(path.read_bytes() == contents for path, contents in before.items())


@pytest.mark.parametrize("contents", ["", "\n  \n", IGNORE])
def test_empty_and_ignored_ground_truth(scene, contents):
    scan, labels, _ = scene
    labels.write_text(contents)
    assert load_ground_truth_boxes(scan) == ()


@pytest.mark.parametrize("role", ["labels", "calibration"])
def test_missing_matching_files(scene, role):
    scan, labels, calib = scene
    missing = (labels if role == "labels" else calib).with_name("000000_missing.txt")
    # Use a missing file with the matching stem in a different directory.
    missing = missing.parent / "absent/000000.txt"
    kwargs = {"labels_path" if role == "labels" else "calibration_path": missing}
    with pytest.raises(FileNotFoundError, match=f"Missing ground-truth {role}"):
        load_ground_truth_boxes(scan, **kwargs)


@pytest.mark.parametrize("role", ["labels_path", "calibration_path"])
def test_frame_id_mismatch(scene, role):
    scan, labels, _ = scene
    mismatch = labels.with_name("000001.txt")
    mismatch.write_text(LABEL)
    with pytest.raises(ValueError, match="Frame ID mismatch"):
        load_ground_truth_boxes(scan, **{role: mismatch})


def test_nonstandard_layout_requires_explicit_paths(scene, tmp_path):
    scan, labels, calib = scene
    other = tmp_path / "000000.bin"
    other.write_bytes(scan.read_bytes())
    with pytest.raises(ValueError, match="provide both --labels and --calib"):
        load_ground_truth_boxes(other)
    assert len(load_ground_truth_boxes(other, labels_path=labels, calibration_path=calib)) == 1


def test_cli_automatic_ground_truth_paths(scene, monkeypatch):
    from unittest.mock import Mock

    from raven.cli import main
    from raven.perception.visualization import viewer

    render = Mock()
    monkeypatch.setattr(viewer, "view_point_cloud", render)
    main(["visualize", str(scene[0]), "--ground-truth"])
    assert len(render.call_args.kwargs["boxes"]) == 1
    assert render.call_args.kwargs["boxes"][0].label == "Car"


def test_missing_calibration_cli_errors_before_render(scene, monkeypatch, capsys):
    from unittest.mock import Mock

    from raven.cli import main
    from raven.perception.visualization import viewer

    render = Mock()
    monkeypatch.setattr(viewer, "view_point_cloud", render)
    path = scene[2].parent / "absent/000000.txt"
    with pytest.raises(SystemExit) as exc:
        main(["visualize", str(scene[0]), "--ground-truth", "--calib", str(path)])
    assert exc.value.code == 2
    assert "Missing ground-truth calibration" in capsys.readouterr().err
    render.assert_not_called()
