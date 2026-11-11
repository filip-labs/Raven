from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from raven.perception.detection.base import Detection3D, GroundTruth3D


def test_detection_legacy_constructor_and_immutable_values():
    position = [1, 2, 3]
    box = Detection3D(position, [4, 2, 1], 0.5, "Car", 0.9)
    position[0] = 100
    assert box.center_xyz == (1.0, 2.0, 3.0)
    assert box.size_lwh == (4.0, 2.0, 1.0)
    assert box.coordinate_frame == "lidar" and box.rotation_axis == "z"
    with pytest.raises(FrozenInstanceError):
        box.label = "Truck"


def test_ground_truth_has_explicit_conventions_and_no_confidence():
    box = GroundTruth3D((1, 2, 3), (4, 2, 1), 0.5, "Car", "camera", "y", "bottom_center")
    assert box.position_xyz == (1, 2, 3)
    assert box.position_reference == "bottom_center"
    assert box.rotation_axis == "y"
    assert not hasattr(box, "score")


@pytest.mark.parametrize(
    "field,value",
    [
        ("center_xyz", (1, 2)),
        ("center_xyz", ((1, 2, 3),)),
        ("center_xyz", (float("nan"), 0, 0)),
        ("center_xyz", None),
        ("center_xyz", ("1", 2, 3)),
        ("center_xyz", (True, 2, 3)),
        ("size_lwh", (4, 0, 1)),
        ("size_lwh", (-4, 2, 1)),
        ("size_lwh", (4, 2, float("inf"))),
        ("size_lwh", (4, 2)),
        ("yaw_rad", float("inf")),
        ("yaw_rad", "0"),
        ("score", -0.1),
        ("score", 1.1),
        ("score", float("nan")),
        ("score", True),
        ("label", ""),
        ("label", "  "),
        ("coordinate_frame", ""),
        ("coordinate_frame", None),
        ("coordinate_frame", " lidar "),
        ("rotation_axis", "yaw"),
    ],
)
def test_detection_rejects_invalid_fields(field, value):
    args = {
        "center_xyz": (1, 2, 3),
        "size_lwh": (4, 2, 1),
        "yaw_rad": 0,
        "label": "Car",
        "score": 0.5,
    }
    args[field] = value
    with pytest.raises((ValueError, TypeError)):
        Detection3D(**args)


@pytest.mark.parametrize("score", [0, 1, np.float32(0.5)])
def test_confidence_boundaries(score):
    assert Detection3D((0, 0, 0), (1, 1, 1), 0, "Car", score).score == score


@pytest.mark.parametrize(
    "field,value",
    [
        ("position_xyz", (float("inf"), 2, 3)),
        ("position_xyz", (1, 2)),
        ("size_lwh", (1, -1, 2)),
        ("rotation_rad", float("nan")),
        ("coordinate_frame", ""),
        ("coordinate_frame", 1),
        ("rotation_axis", "invalid"),
        ("position_reference", "unknown"),
    ],
)
def test_ground_truth_validation(field, value):
    args = {
        "position_xyz": (1, 2, 3),
        "size_lwh": (4, 2, 1),
        "rotation_rad": 0,
        "label": "Car",
        "coordinate_frame": "camera",
    }
    args[field] = value
    with pytest.raises((ValueError, TypeError)):
        GroundTruth3D(**args)


def test_ground_truth_requires_coordinate_frame():
    with pytest.raises(TypeError):
        GroundTruth3D((1, 2, 3), (4, 2, 1), 0, "Car")
