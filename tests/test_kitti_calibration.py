from dataclasses import replace

import numpy as np
import pytest

from raven.cli import main
from raven.perception.data.kitti_annotations import parse_kitti_annotation
from raven.perception.data.kitti_calibration import (
    CAMERA_RECT_FRAME,
    VELODYNE_FRAME,
    KittiCalibration,
    load_kitti_calibration,
    transform_annotation,
    transform_box,
    transform_points,
)
from raven.perception.detection.base import GroundTruth3D

VALID_LABEL = "Car 0 0 0 10 20 100 200 2 4 6 1 3 10 0"
IGNORE_LABEL = "DontCare -1 -1 -10 10 20 100 200 -1 -1 -1 -1000 -1000 -1000 -10"
AXES = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=float)


def rotation(axis, angle):
    c, s = np.cos(angle), np.sin(angle)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


@pytest.fixture
def identity():
    return KittiCalibration({"R0_rect": np.eye(3), "Tr_velo_to_cam": np.eye(4)[:3]})


@pytest.fixture
def calibrated():
    extrinsic = np.column_stack((AXES, [1, 2, 3]))
    return KittiCalibration({"R0_rect": np.eye(3), "Tr_velo_to_cam": extrinsic})


def calibration_text(matrices):
    return "\n".join(
        f"{name}: " + " ".join(str(v) for v in values.flat) for name, values in matrices.items()
    )


@pytest.fixture
def calibration_file(tmp_path, calibrated):
    matrices = dict(calibrated.matrices)
    matrices.update({f"P{i}": np.eye(4)[:3] for i in range(4)})
    matrices["Tr_imu_to_velo"] = np.eye(4)[:3]
    matrices["extra_numeric"] = np.array([1.0, 2.0])
    path = tmp_path / "000000.txt"
    path.write_text(calibration_text(matrices) + "\n\n")
    return path


def test_parse_all_fields_and_preserve_extra(calibration_file):
    calibration = load_kitti_calibration(calibration_file)
    assert set(calibration.matrices) == {
        "P0",
        "P1",
        "P2",
        "P3",
        "R0_rect",
        "Tr_velo_to_cam",
        "Tr_imu_to_velo",
        "extra_numeric",
    }
    assert calibration.matrices["P2"].shape == (3, 4)
    assert calibration.matrices["R0_rect"].shape == (3, 3)
    np.testing.assert_array_equal(calibration.matrices["extra_numeric"], [1, 2])
    np.testing.assert_array_equal(calibration.velo_to_rect[3], [0, 0, 0, 1])


def test_calibration_is_defensively_immutable():
    values = np.eye(4)[:3].copy()
    calibration = KittiCalibration({"R0_rect": np.eye(3), "Tr_velo_to_cam": values})
    values[0, 3] = 100
    assert calibration.matrices["Tr_velo_to_cam"][0, 3] == 0
    with pytest.raises(TypeError):
        calibration.matrices["new"] = np.eye(3)
    with pytest.raises(ValueError):
        calibration.matrices["R0_rect"].setflags(write=True)


@pytest.mark.parametrize("missing", ["R0_rect", "Tr_velo_to_cam"])
def test_missing_required_fields(tmp_path, identity, missing):
    matrices = dict(identity.matrices)
    del matrices[missing]
    path = tmp_path / "missing.txt"
    path.write_text(calibration_text(matrices))
    with pytest.raises(ValueError, match="Missing required"):
        load_kitti_calibration(path)


@pytest.mark.parametrize(
    "entry",
    [
        "R0_rect: 1 0",
        "P0: 1 0",
        "Tr_velo_to_cam: " + "0 " * 16,
        "bad field: 1",
        "extra:",
        "extra: NaN",
        "extra: inf",
        "extra: nope",
        "extra 1 2",
        "extra: 1: 2",
    ],
)
def test_malformed_calibration_entries(tmp_path, identity, entry):
    path = tmp_path / "bad.txt"
    # The malformed field comes first, so duplicate checks cannot mask bad dimensions.
    path.write_text(entry + "\n" + calibration_text(identity.matrices))
    with pytest.raises(ValueError, match=r"bad\.txt:1:"):
        load_kitti_calibration(path)


def test_duplicate_field(calibration_file):
    calibration_file.write_text(calibration_file.read_text() + "P2: " + "0 " * 12)
    with pytest.raises(ValueError, match="Duplicate calibration field P2"):
        load_kitti_calibration(calibration_file)


@pytest.mark.parametrize("matrix", [np.zeros((3, 3)), np.diag([1, 1, -1]), np.eye(3) * 1.01])
@pytest.mark.parametrize("field", ["R0_rect", "Tr_velo_to_cam", "Tr_imu_to_velo"])
def test_invalid_rotations_rejected(identity, matrix, field):
    matrices = dict(identity.matrices)
    matrices[field] = matrix if field == "R0_rect" else np.column_stack((matrix, [0, 0, 0]))
    with pytest.raises(ValueError, match="proper orthonormal rotation"):
        KittiCalibration(matrices)


@pytest.mark.parametrize(
    "point", [np.array([1, 2, 3]), np.array([[1, 2, 3], [4, 5, 6]]), np.empty((0, 3))]
)
def test_identity_shape_and_immutability(identity, point):
    original = point.copy()
    result = transform_points(
        point, identity, source_frame=VELODYNE_FRAME, target_frame=CAMERA_RECT_FRAME
    )
    np.testing.assert_array_equal(result, original)
    np.testing.assert_array_equal(point, original)
    assert result.shape == point.shape and result.dtype == np.float64
    assert not np.shares_memory(result, point)


def test_known_forward_inverse_and_axis_mapping(calibrated):
    point = np.array([10, 2, 1], dtype=np.float32)
    camera = transform_points(
        point, calibrated, source_frame=VELODYNE_FRAME, target_frame=CAMERA_RECT_FRAME
    )
    np.testing.assert_allclose(camera, [-1, 1, 13])
    np.testing.assert_allclose(
        transform_points(
            camera, calibrated, source_frame=CAMERA_RECT_FRAME, target_frame=VELODYNE_FRAME
        ),
        point,
    )
    np.testing.assert_allclose(calibrated.rect_to_velo @ calibrated.velo_to_rect, np.eye(4))


def test_rectification_order_and_projection_independence(calibrated):
    matrices = dict(calibrated.matrices)
    rect = rotation("x", 0.1)
    matrices["R0_rect"] = rect
    matrices["P2"] = np.full((3, 4), 999.0)
    calibration = KittiCalibration(matrices)
    result = transform_points(
        np.array([10, 2, 1]),
        calibration,
        source_frame=VELODYNE_FRAME,
        target_frame=CAMERA_RECT_FRAME,
    )
    np.testing.assert_allclose(result, rect @ [-1, 1, 13])


def test_nontrivial_round_trip(calibrated):
    matrices = dict(calibrated.matrices)
    matrices["R0_rect"] = rotation("x", 0.03) @ rotation("z", -0.02)
    calibration = KittiCalibration(matrices)
    points = np.random.default_rng(42).normal(size=(50, 3))
    camera = transform_points(
        points, calibration, source_frame=VELODYNE_FRAME, target_frame=CAMERA_RECT_FRAME
    )
    recovered = transform_points(
        camera, calibration, source_frame=CAMERA_RECT_FRAME, target_frame=VELODYNE_FRAME
    )
    np.testing.assert_allclose(recovered, points, atol=1e-12)


@pytest.mark.parametrize(
    "points",
    [
        np.zeros(4),
        np.zeros((2, 4)),
        np.zeros((3, 1)),
        np.zeros((1, 2, 3)),
        np.array([np.nan, 1, 2]),
        np.array([np.inf, 1, 2]),
        np.array([]),
    ],
)
def test_invalid_point_arrays(identity, points):
    with pytest.raises(ValueError):
        transform_points(
            points, identity, source_frame=VELODYNE_FRAME, target_frame=CAMERA_RECT_FRAME
        )


@pytest.mark.parametrize(
    "source,target", [("wrong", CAMERA_RECT_FRAME), (VELODYNE_FRAME, "wrong"), ("wrong", "wrong")]
)
def test_unsupported_frames(identity, source, target):
    with pytest.raises(ValueError, match="Unsupported coordinate frame"):
        transform_points(np.empty((0, 3)), identity, source_frame=source, target_frame=target)


def test_legacy_camera_frame_alias(identity):
    point = np.array([1, 2, 3])
    result = transform_points(
        point, identity, source_frame="kitti_rectified_camera", target_frame=CAMERA_RECT_FRAME
    )
    np.testing.assert_array_equal(result, point)


def test_camera_box_center_corners_and_edge_lengths(identity):
    annotation = parse_kitti_annotation(VALID_LABEL)
    box = transform_annotation(annotation, identity, target_frame=CAMERA_RECT_FRAME)
    np.testing.assert_array_equal(box.center_xyz, [1, 2, 10])
    np.testing.assert_array_equal(box.size_lwh, [6, 4, 2])
    expected = np.array(
        [
            [-2, 3, 8],
            [4, 3, 8],
            [4, 3, 12],
            [-2, 3, 12],
            [-2, 1, 8],
            [4, 1, 8],
            [4, 1, 12],
            [-2, 1, 12],
        ]
    )
    np.testing.assert_array_equal(box.corners, expected)
    edges = box.corners[[1, 3, 4]] - box.corners[0]
    np.testing.assert_allclose(np.linalg.norm(edges, axis=1), [6, 4, 2])
    assert box.annotation is annotation and box.source is annotation.ground_truth
    assert annotation.location_xyz == (1, 3, 10)  # Original bottom center is intact.
    assert annotation.coordinate_frame == "kitti_rectified_camera"
    assert box.label == "Car"


def test_rotated_camera_box_and_full_calibration_pose(calibrated):
    annotation = parse_kitti_annotation(VALID_LABEL.rsplit(" ", 1)[0] + " 0.7")
    matrices = dict(calibrated.matrices)
    matrices["R0_rect"] = rotation("x", 0.2) @ rotation("z", 0.1)
    calibration = KittiCalibration(matrices)
    camera_box = transform_annotation(annotation, calibration, target_frame=CAMERA_RECT_FRAME)
    expected_length_axis = np.array([np.cos(0.7), 0, -np.sin(0.7)])
    np.testing.assert_allclose(camera_box.rotation[:, 0], expected_length_axis)
    lidar_box = transform_box(camera_box, calibration, target_frame=VELODYNE_FRAME)
    np.testing.assert_allclose(
        lidar_box.rotation, calibration.rect_to_velo[:3, :3] @ camera_box.rotation
    )
    expected_corners = transform_points(
        camera_box.corners, calibration, source_frame=CAMERA_RECT_FRAME, target_frame=VELODYNE_FRAME
    )
    np.testing.assert_allclose(lidar_box.corners, expected_corners, atol=1e-12)
    assert not np.allclose(lidar_box.rotation[:, 2], [0, 0, 1])  # Roll/pitch retained.
    recovered = transform_box(lidar_box, calibration, target_frame=CAMERA_RECT_FRAME)
    np.testing.assert_allclose(recovered.corners, camera_box.corners, atol=1e-12)
    assert recovered.annotation is annotation
    np.testing.assert_allclose(lidar_box.rotation.T @ lidar_box.rotation, np.eye(3), atol=1e-12)
    assert np.linalg.det(lidar_box.rotation) == pytest.approx(1)


def test_lidar_bottom_center_and_geometric_center(identity):
    source = GroundTruth3D(
        (1, 2, 3), (6, 4, 2), np.pi / 2, "Car", VELODYNE_FRAME, position_reference="bottom_center"
    )
    geometry = transform_box(source, identity, target_frame=VELODYNE_FRAME)
    np.testing.assert_allclose(geometry.center_xyz, [1, 2, 4])
    np.testing.assert_allclose(geometry.rotation[:, 0], [0, 1, 0], atol=1e-15)
    centered = replace(source, position_reference="geometric_center")
    np.testing.assert_array_equal(
        transform_box(centered, identity, target_frame=VELODYNE_FRAME).center_xyz, [1, 2, 3]
    )


def test_camera_geometric_center_not_shifted(identity):
    source = parse_kitti_annotation(VALID_LABEL).ground_truth
    source = replace(source, position_reference="geometric_center")
    box = transform_box(source, identity, target_frame=CAMERA_RECT_FRAME)
    np.testing.assert_array_equal(box.center_xyz, [1, 3, 10])


@pytest.mark.parametrize(
    "frame,axis", [("unsupported", "y"), (CAMERA_RECT_FRAME, "z"), (VELODYNE_FRAME, "y")]
)
def test_unsupported_box_conventions(identity, frame, axis):
    source = GroundTruth3D((0, 0, 0), (1, 1, 1), 0, "Car", frame, axis)
    with pytest.raises(ValueError):
        transform_box(source, identity, target_frame=VELODYNE_FRAME)


def test_ignore_regions_do_not_create_geometry(identity):
    annotation = parse_kitti_annotation(IGNORE_LABEL)
    assert transform_annotation(annotation, identity) is None
    with pytest.raises(ValueError):
        transform_annotation(annotation, identity, target_frame="wrong")


def test_box_geometry_immutable_and_validation(identity):
    box = transform_annotation(parse_kitti_annotation(VALID_LABEL), identity)
    with pytest.raises(ValueError):
        box.rotation.setflags(write=True)
    for change in (
        {"center_xyz": [1, 2]},
        {"size_lwh": [0, 1, 1]},
        {"rotation": np.diag([1, 1, -1])},
        {"coordinate_frame": "wrong"},
    ):
        with pytest.raises(ValueError):
            replace(box, **change)


def test_calibration_cli_without_open3d(calibration_file, monkeypatch, capsys):
    import json
    import sys

    monkeypatch.setitem(sys.modules, "open3d", None)
    main(["inspect-calib", str(calibration_file)])
    result = json.loads(capsys.readouterr().out)
    assert result["matrices"]["P2"] == [3, 4]
    assert result["source_frame"] == VELODYNE_FRAME
    assert result["target_frame"] == CAMERA_RECT_FRAME
    assert result["invertible"] is True
    assert np.asarray(result["velo_to_rect"]).shape == (4, 4)


@pytest.mark.parametrize("mode", ["missing", "malformed"])
def test_calibration_cli_errors(tmp_path, capsys, mode):
    path = tmp_path / "calib.txt"
    if mode == "malformed":
        path.write_text("bad field")
    with pytest.raises(SystemExit) as exc:
        main(["inspect-calib", str(path)])
    assert exc.value.code == 2
    stderr = capsys.readouterr().err
    assert "calib.txt" in stderr and "Traceback" not in stderr


def test_rounded_rotation_inverse_and_geometry():
    rotation_matrix = np.round(rotation("x", 0.12) @ rotation("y", -0.2), 7)
    calibration = KittiCalibration(
        {"R0_rect": rotation_matrix, "Tr_velo_to_cam": np.column_stack((AXES, [0.1, 0.2, 0.3]))}
    )
    points = np.array([[10.0, 2.0, 1.0]])
    rect = transform_points(
        points, calibration, source_frame=VELODYNE_FRAME, target_frame=CAMERA_RECT_FRAME
    )
    np.testing.assert_allclose(
        transform_points(
            rect, calibration, source_frame=CAMERA_RECT_FRAME, target_frame=VELODYNE_FRAME
        ),
        points,
        atol=1e-12,
    )
    box = transform_annotation(parse_kitti_annotation(VALID_LABEL), calibration)
    np.testing.assert_allclose(
        np.linalg.norm(box.corners[[1, 3, 4]] - box.corners[0], axis=1), [6, 4, 2], atol=1e-5
    )


def test_complex_points_and_calibrations_rejected(identity):
    with pytest.raises(ValueError, match="real"):
        transform_points(
            np.array([1 + 2j, 0, 0]),
            identity,
            source_frame=VELODYNE_FRAME,
            target_frame=CAMERA_RECT_FRAME,
        )
    with pytest.raises(ValueError):
        KittiCalibration({"R0_rect": np.eye(3, dtype=complex), "Tr_velo_to_cam": np.eye(4)[:3]})


def test_box_source_metadata_must_match(identity):
    box = transform_annotation(parse_kitti_annotation(VALID_LABEL), identity)
    with pytest.raises(TypeError):
        replace(box, source=None)
    with pytest.raises(ValueError, match="source ground-truth"):
        replace(box, annotation=parse_kitti_annotation(IGNORE_LABEL))
