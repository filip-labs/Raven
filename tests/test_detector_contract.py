import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from detector_doubles import FixedDetector

from raven.core.point_cloud import PointCloud
from raven.perception.detection.base import Detection3D, ObjectDetector


def detect_with_backend(detector: ObjectDetector, cloud: PointCloud) -> list[Detection3D]:
    """Exercise a structurally compatible backend through the shared API."""
    return detector.detect(cloud)


@pytest.fixture
def cloud():
    return PointCloud(np.array([[1, 2, 3, 0.5], [4, 5, 6, 0.9]]), "frame", "kitti_velodyne")


def test_structural_backend_multiple_boxes_metadata_and_input_preservation(cloud):
    boxes = [
        Detection3D((1, 2, 3), (4, 2, 1), 0.5, "Car", 0, "kitti_velodyne"),
        Detection3D((4, 5, 6), (0.5, 0.5, 1.8), -0.2, "Pedestrian", 1, "kitti_velodyne"),
    ]
    detector: ObjectDetector = FixedDetector(boxes)
    points = cloud.points.copy()
    original_array = cloud.points
    metadata = (cloud.frame_id, cloud.coordinate_frame, cloud.points.flags.writeable)
    result = detect_with_backend(detector, cloud)
    assert result == boxes
    assert [(box.label, box.coordinate_frame, box.rotation_axis) for box in result] == [
        ("Car", "kitti_velodyne", "z"),
        ("Pedestrian", "kitti_velodyne", "z"),
    ]
    assert result[0].center_xyz == (1, 2, 3)
    assert result[0].size_lwh == (4, 2, 1)
    assert result[0].yaw_rad == 0.5
    assert [box.score for box in result] == [0, 1]
    np.testing.assert_array_equal(cloud.points, points)
    assert cloud.points is original_array
    assert (cloud.frame_id, cloud.coordinate_frame, cloud.points.flags.writeable) == metadata
    assert not cloud.points.flags.writeable


@pytest.mark.parametrize("empty_cloud", [False, True])
def test_zero_detections(cloud, empty_cloud):
    if empty_cloud:
        cloud = PointCloud(np.empty((0, 4)), "empty", "kitti_velodyne")
    assert detect_with_backend(FixedDetector(), cloud) == []


def test_results_are_independent_of_caller_list_mutations(cloud):
    box = Detection3D((1, 2, 3), (4, 2, 1), 0, "Car", 0.5, "kitti_velodyne")
    configured = [box]
    detector = FixedDetector(configured)
    configured.clear()
    first = detector.detect(cloud)
    first.clear()
    assert detector.detect(cloud) == [box]


def test_incompatible_frames_fail_instead_of_relabeling(cloud):
    camera_box = Detection3D((1, 2, 3), (4, 2, 1), 0, "Car", 0.5, "camera", "y")
    detector = FixedDetector([camera_box])
    with pytest.raises(ValueError, match="coordinate frame"):
        detector.detect(cloud)
    assert camera_box.coordinate_frame == "camera"
    assert cloud.coordinate_frame == "kitti_velodyne"


def test_legacy_positional_detection_works_with_default_cloud_frame():
    box = Detection3D((0, 0, 0), (1, 2, 3), 0.25, "custom-class", 0.5)
    cloud = PointCloud(np.zeros((1, 4)), "legacy")
    assert detect_with_backend(FixedDetector([box]), cloud) == [box]
    assert box.coordinate_frame == "lidar"
    assert box.label == "custom-class"


def test_double_requires_validated_detections():
    with pytest.raises(TypeError, match="validated Detection3D"):
        FixedDetector([object()])


def test_cpu_import_and_inference_with_optional_dependencies_blocked():
    # A fresh interpreter avoids already-imported modules hiding eager imports.
    program = """
import importlib.abc
import sys

blocked = {"torch", "pcdet", "OpenPCDet", "spconv", "cumm", "open3d"}
class BlockOptionalImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in blocked:
            raise AssertionError("Unexpected optional import: " + fullname)
sys.meta_path.insert(0, BlockOptionalImports())
import numpy as np
import raven
from raven.core.point_cloud import PointCloud
from raven.perception.detection.base import Detection3D, ObjectDetector
from detector_doubles import FixedDetector
box = Detection3D((0, 0, 0), (1, 1, 1), 0, "Car", 1)
backend: ObjectDetector = FixedDetector([box])
assert backend.detect(PointCloud(np.zeros((1, 4)), "cpu")) == [box]
assert not any(name.split(".")[0] in blocked for name in sys.modules)
print("CPU-only detector import and inference passed")
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=Path(__file__).resolve().parent,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "CPU-only detector import and inference passed" in result.stdout
