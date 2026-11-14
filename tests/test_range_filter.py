import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from raven.core.point_cloud import PointCloud
from raven.perception.data.kitti_loader import load_kitti_bin
from raven.perception.preprocessing.range_filter import filter_range


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_each_axis_interior_boundaries_and_just_outside(axis):
    lower, upper = np.float32(-1), np.float32(1)
    values = [
        np.nextafter(lower, np.float32(-np.inf)),
        lower,
        0,
        upper,
        np.nextafter(upper, np.float32(np.inf)),
    ]
    points = np.zeros((5, 4), dtype=np.float32)
    points[:, axis] = values
    points[:, 3] = [0.11, 0.22, 0.33, 0.44, 0.55]
    cloud = PointCloud(points, "000123", "kitti_velodyne")
    result = filter_range(cloud, (-1, 1), (-1, 1), (-1, 1))
    np.testing.assert_array_equal(result.points, cloud.points[[1, 2, 3]])
    assert result.count == 3


def test_all_axes_apply_together_and_keep_unsorted_rows_and_intensity():
    points = np.array(
        [
            [0.5, 0.6, 0.7, -0.5],
            [2, 0, 0, 0.2],
            [-0.8, -0.9, -1, 3.5],
            [0, 2, 0, 0.4],
            [-1, 1, 1, 0.9],
            [0, 0, 2, 0.6],
            [0.5, 0.6, 0.7, 0.8],
        ]
    )
    cloud = PointCloud(points, "frame-identity", "caller-frame")
    original = cloud.points.copy()
    result = filter_range(cloud, (-1, 1), (-1, 1), (-1, 1))
    np.testing.assert_array_equal(result.points, cloud.points[[0, 2, 4, 6]])
    np.testing.assert_array_equal(cloud.points, original)
    assert result.points.dtype == np.float32 and not result.points.flags.writeable
    assert not cloud.points.flags.writeable
    assert result.frame_id == "frame-identity" and result.coordinate_frame == "caller-frame"
    assert result is not cloud and not np.shares_memory(result.points, cloud.points)
    repeated = filter_range(cloud, (-1, 1), (-1, 1), (-1, 1))
    np.testing.assert_array_equal(result.points, repeated.points)
    # A caller forcibly re-enabling output writes still cannot alter the input.
    result.points.setflags(write=True)
    result.points[0, 3] = 99
    np.testing.assert_array_equal(cloud.points, original)


@pytest.mark.parametrize("empty_input", [False, True])
def test_empty_result_preserves_shape_dtype_and_metadata(empty_input):
    cloud = PointCloud(
        np.empty((0, 4)) if empty_input else np.array([[10, 10, 10, 0.5]]),
        "empty",
        "original-frame",
    )
    result = filter_range(cloud, (-1, 1), (-1, 1), (-1, 1))
    assert result.count == 0 and result.points.shape == (0, 4)
    assert result.points.dtype == np.float32 and not result.points.flags.writeable
    assert result.frame_id == cloud.frame_id and result.coordinate_frame == cloud.coordinate_frame


def test_all_kept_still_independent_and_dtype_preserved():
    cloud = PointCloud(np.array([[1, 2, 3, 0.5], [-1, -2, -3, 0.1]]), "all")
    result = filter_range(cloud, (-10, 10), (-10, 10), (-10, 10))
    np.testing.assert_array_equal(result.points, cloud.points)
    assert not np.shares_memory(result.points, cloud.points)


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize(
    "invalid",
    [
        (1, 1),
        (2, 1),
        (np.nan, 1),
        (-1, np.nan),
        (np.inf, 1),
        (-1, np.inf),
        (-np.inf, 1),
        None,
        0,
        (),
        (0,),
        (0, 1, 2),
        "01",
        ("0", 1),
        (False, 1),
        (0, True),
        (np.bool_(False), 1),
        (0, 1j),
        ([0], 1),
        {0, 1},
        {0: "a", 1: "b"},
    ],
)
def test_invalid_bounds_are_rejected_on_every_axis_even_empty_input(axis, invalid):
    cloud = PointCloud(np.empty((0, 4)), "frame")
    bounds = [(-1, 1), (-1, 1), (-1, 1)]
    bounds[axis] = invalid
    with pytest.raises(
        (ValueError, TypeError), match=f"{'XYZ'[axis]} range.*finite real.*lower < upper"
    ):
        filter_range(cloud, *bounds)


def test_numpy_bounds_and_ordered_iterable_bounds():
    cloud = PointCloud(np.array([[0, 0, 0, 0.5]]), "a")
    result = filter_range(
        cloud, np.array([-1, 1], dtype=np.float32), [np.int64(-1), np.float64(1)], iter((-1, 1))
    )
    np.testing.assert_array_equal(result.points, cloud.points)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_float64_threshold_not_rounded_to_stored_float32(axis):
    points = np.zeros((1, 4), dtype=np.float32)
    points[0, axis] = np.float32(0.1)
    cloud = PointCloud(points, "precise")
    stored = float(cloud.points[0, axis])
    bounds = [(-1, 1), (-1, 1), (-1, 1)]
    bounds[axis] = (-1, np.nextafter(stored, -np.inf))
    assert filter_range(cloud, *bounds).count == 0
    bounds[axis] = (np.nextafter(stored, np.inf), 1)
    assert filter_range(cloud, *bounds).count == 0
    bounds[axis] = (stored, 1)
    assert filter_range(cloud, *bounds).count == 1


def test_kitti_binary_compatibility_with_synthetic_file(tmp_path):
    path = tmp_path / "000000.bin"
    source = np.array([[10, 2, -1, 0.1], [100, 0, 0, 0.8], [3, -2, -0.5, 0.9]], dtype="<f4")
    source.tofile(path)
    original = path.read_bytes()
    cloud = load_kitti_bin(path)
    result = filter_range(cloud, (0, 70), (-40, 40), (-3, 1))
    np.testing.assert_array_equal(result.points, source[[0, 2]])
    assert result.frame_id == "000000" and result.coordinate_frame == "kitti_velodyne"
    assert path.read_bytes() == original


def test_optional_real_local_kitti_scan():
    path = Path(__file__).resolve().parents[1] / "data/kitti/training/velodyne/000000.bin"
    if not path.is_file():
        pytest.skip("Local KITTI data is optional; no dataset download in CI")
    cloud = load_kitti_bin(path)
    result = filter_range(cloud, (0, 70), (-40, 40), (-3, 1))
    assert 0 < result.count <= cloud.count
    expected = cloud.points[
        np.array(
            [
                all(
                    lo <= float(value) <= hi
                    for value, (lo, hi) in zip(row[:3], ((0, 70), (-40, 40), (-3, 1)), strict=True)
                )
                for row in cloud.points
            ]
        )
    ]
    np.testing.assert_array_equal(result.points, expected)
    assert result.coordinate_frame == cloud.coordinate_frame and result.frame_id == cloud.frame_id


def test_filter_import_and_execution_without_optional_frameworks():
    program = """
import sys, importlib.abc
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "pcdet", "spconv", "cumm", "open3d"}:
            raise AssertionError("Optional dependency imported: " + fullname)
sys.meta_path.insert(0, Block())
import numpy as np
from raven.core.point_cloud import PointCloud
from raven.perception.preprocessing.range_filter import filter_range
cloud = PointCloud(np.array([[1,2,3,.5]]), "frame")
assert filter_range(cloud, (0,2), (0,3), (0,4)).count == 1
"""
    subprocess.run([sys.executable, "-c", program], check=True, capture_output=True, text=True)
