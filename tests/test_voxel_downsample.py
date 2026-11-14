import itertools
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from raven.core.point_cloud import PointCloud
from raven.perception.data.kitti_loader import load_kitti_bin
from raven.perception.preprocessing.range_filter import filter_range
from raven.perception.preprocessing.voxel_downsample import voxel_downsample


def cloud(points):
    return PointCloud(np.array(points, dtype=np.float32).reshape(-1, 4), "000000", "caller-frame")


def test_means_xyz_and_unclipped_intensity_with_lexicographic_voxel_order():
    source = cloud(
        [
            [1.1, 0.1, 0.1, 4],
            [0.1, 0.1, 0.1, -2],
            [-0.2, -0.2, -0.2, 10],
            [0.3, 0.5, 0.7, 6],
            [-0.8, -0.4, -0.6, 2],
            [1.7, 0.3, 0.5, 8],
        ]
    )
    result = voxel_downsample(source, 1)
    expected = np.array(
        [
            np.mean(source.points[[2, 4]].astype(np.float64), axis=0),
            np.mean(source.points[[1, 3]].astype(np.float64), axis=0),
            np.mean(source.points[[0, 5]].astype(np.float64), axis=0),
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(result.points, expected)
    assert result.count == 3 and result.points[0, 3] == 6
    assert result.points.dtype == np.float32 and not result.points.flags.writeable
    assert result.frame_id == source.frame_id and result.coordinate_frame == source.coordinate_frame


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_positive_negative_and_exact_half_open_boundaries(axis):
    values = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
    points = np.zeros((len(values), 4), dtype=np.float32)
    points[:, axis] = values
    points[:, 3] = np.arange(len(values))
    source = cloud(points)
    result = voxel_downsample(source, 0.5)
    groups = [[0, 1], [2, 3], [4, 5], [6, 7], [8]]
    expected = np.array(
        [source.points[group].astype(np.float64).mean(axis=0) for group in groups], dtype=np.float32
    )
    np.testing.assert_array_equal(result.points, expected)
    assert result.count == 5  # floor(-.25/.5)=-1, not truncation to zero.


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_adjacent_float32_values_around_boundary(axis):
    center = np.float32(0.5)
    points = np.zeros((3, 4), dtype=np.float32)
    points[:, axis] = [
        np.nextafter(center, np.float32(-np.inf)),
        center,
        np.nextafter(center, np.float32(np.inf)),
    ]
    points[:, 3] = [1, 3, 5]
    source = cloud(points)
    result = voxel_downsample(source, 0.5)
    assert result.count == 2
    np.testing.assert_array_equal(result.points[0], source.points[0])
    np.testing.assert_array_equal(
        result.points[1], source.points[1:].astype(np.float64).mean(axis=0).astype(np.float32)
    )


def test_anisotropic_sizes_and_full_xyz_cell_order():
    source = cloud(
        [
            [1.2, 2.2, 3.2, 1],
            [0.2, 2.2, 3.2, 2],
            [0.2, 0.2, 3.2, 3],
            [0.2, 0.2, 0.2, 4],
            [0.4, 0.4, 0.4, 6],
        ]
    )
    result = voxel_downsample(source, (1, 2, 3))
    expected = np.array(
        [
            source.points[[3, 4]].astype(float).mean(axis=0),
            source.points[2],
            source.points[1],
            source.points[0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(result.points, expected)


def test_voxel_origin_does_not_follow_cloud_minimum():
    source = cloud([[0.1, 0, 0, 1], [0.8, 0, 0, 3], [-100.1, 0, 0, 5]])
    result = voxel_downsample(source, 1)
    assert result.count == 2
    np.testing.assert_allclose(result.points[-1], [0.45, 0, 0, 2])


@pytest.mark.parametrize("points", [[], [[1, 2, 3, 0.5]], [[1, 2, 3, 0.5]] * 4])
def test_empty_single_and_duplicate_clouds(points):
    source = cloud(points)
    result = voxel_downsample(source, 0.2)
    assert result.count == (1 if points else 0)
    assert result.points.shape == (result.count, 4)
    assert result.points.dtype == np.float32 and not result.points.flags.writeable
    assert result.frame_id == source.frame_id and result.coordinate_frame == source.coordinate_frame
    assert not np.shares_memory(result.points, source.points)
    if points:
        np.testing.assert_array_equal(result.points[0], source.points[0])


def test_input_and_size_arrays_unchanged_and_output_has_independent_storage():
    source = cloud([[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5]])
    original = source.points.copy()
    sizes = np.array([1.0, 1.0, 1.0])
    before = sizes.copy()
    result = voxel_downsample(source, sizes)
    np.testing.assert_array_equal(source.points, original)
    np.testing.assert_array_equal(sizes, before)
    assert not source.points.flags.writeable
    result.points.setflags(write=True)
    result.points[0] = 99
    np.testing.assert_array_equal(source.points, original)


def test_repeat_and_all_permutations_produce_identical_values():
    # Cancellation-sensitive intensities require canonical within-cell ordering.
    source = cloud(
        [[0.1, 0.1, 0.1, 1e20], [0.2, 0.1, 0.1, -1e20], [0.3, 0.1, 0.1, 3], [2, 0, 0, 0.5]]
    )
    expected = voxel_downsample(source, 1).points.copy()
    for order in itertools.permutations(range(4)):
        result = voxel_downsample(cloud(source.points[list(order)]), 1)
        np.testing.assert_array_equal(result.points, expected)
    np.testing.assert_array_equal(voxel_downsample(source, 1).points, expected)


def test_float32_storage_after_float64_mean():
    source = cloud([[0.1, 0.2, 0.3, 0.1], [0.2, 0.4, 0.6, 0.2], [0.3, 0.6, 0.9, 0.3]])
    result = voxel_downsample(source, 1)
    expected64 = source.points.astype(np.float64).mean(axis=0)
    np.testing.assert_array_equal(result.points[0], expected64.astype(np.float32))
    assert float(result.points[0, 0]) != expected64[0]  # Final storage may round.


@pytest.mark.parametrize(
    "size",
    [
        0,
        -1,
        np.nan,
        np.inf,
        -np.inf,
        (1, 0, 1),
        (1, -1, 1),
        (1, np.nan, 1),
        (1, np.inf, 1),
        (),
        (1,),
        (1, 2),
        (1, 2, 3, 4),
        None,
        True,
        np.bool_(True),
        "0.2",
        ["1", 1, 1],
        [True, 1, 1],
        1j,
        [1j, 1, 1],
        [[1], [1], [1]],
        {1, 2, 3},
        {"x": 1, "y": 1, "z": 1},
        np.array(0.2),
        10**400,
    ],
)
@pytest.mark.parametrize("empty", [False, True])
def test_invalid_voxel_size_even_for_empty_cloud(size, empty):
    source = cloud([] if empty else [[1, 2, 3, 0.5]])
    with pytest.raises((ValueError, TypeError), match="voxel_size"):
        voxel_downsample(source, size)


@pytest.mark.parametrize(
    "size",
    [
        np.float32(0.5),
        np.float64(0.5),
        [0.5, 0.5, 0.5],
        np.array([0.5, 0.5, 0.5]),
        iter((0.5, 0.5, 0.5)),
    ],
)
def test_supported_scalar_and_ordered_per_axis_types(size):
    source = cloud([[0.1, 0.2, 0.3, 0.5], [0.2, 0.3, 0.4, 0.7]])
    np.testing.assert_array_equal(
        voxel_downsample(source, size).points, voxel_downsample(source, 0.5).points
    )


@pytest.mark.parametrize("size", [1e-300, 2**-53, 2**-63])
def test_overflow_or_unresolvable_voxel_indices_rejected(size):
    with pytest.raises(ValueError, match="voxel-index range"):
        voxel_downsample(cloud([[1, 0, 0, 0.5]]), size)


def test_very_large_finite_point_intensity_mean_stays_finite():
    limit = np.finfo(np.float32).max
    result = voxel_downsample(cloud([[0, 0, 0, limit], [0, 0, 0, limit]]), 1)
    assert np.isfinite(result.points).all() and result.points[0, 3] == limit


def test_range_then_voxel_composition_without_implicit_filtering():
    source = cloud([[0.1, 0, 0, 0.2], [0.3, 0, 0, 0.4], [100, 0, 0, 0.6]])
    assert voxel_downsample(source, 1).count == 2  # No hidden crop.
    cropped = filter_range(source, (-1, 1), (-1, 1), (-1, 1))
    result = voxel_downsample(cropped, 1)
    assert result.count == 1 and result.coordinate_frame == source.coordinate_frame
    np.testing.assert_allclose(result.points[0], [0.2, 0, 0, 0.3])


def test_synthetic_kitti_binary_compatibility(tmp_path):
    path = tmp_path / "000000.bin"
    np.array([[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.6]], dtype="<f4").tofile(path)
    original = path.read_bytes()
    source = load_kitti_bin(path)
    result = voxel_downsample(source, 1)
    assert result.count == 1 and result.coordinate_frame == "kitti_velodyne"
    assert result.frame_id == "000000" and path.read_bytes() == original


def test_optional_real_local_kitti_scan():
    path = Path(__file__).resolve().parents[1] / "data/kitti/training/velodyne/000000.bin"
    if not path.is_file():
        pytest.skip("Local KITTI data optional; no downloads in CI")
    source = load_kitti_bin(path)
    for size in (0.1, 0.2, 0.4):
        result = voxel_downsample(source, size)
        assert 0 < result.count <= source.count
        assert np.isfinite(result.points).all()
        assert (
            result.frame_id == source.frame_id
            and result.coordinate_frame == source.coordinate_frame
        )
        np.testing.assert_array_equal(result.points, voxel_downsample(source, size).points)


def test_gpu_free_import_and_execution():
    program = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "pcdet", "spconv", "cumm", "open3d"}:
            raise AssertionError("Optional framework import: " + fullname)
sys.meta_path.insert(0, Block())
import numpy as np
from raven.core.point_cloud import PointCloud
from raven.perception.preprocessing.voxel_downsample import voxel_downsample
cloud = PointCloud(np.array([[.1,.1,.1,.2],[.2,.2,.2,.4]]), "frame")
assert voxel_downsample(cloud, 1).count == 1
"""
    subprocess.run([sys.executable, "-c", program], check=True, capture_output=True, text=True)


def test_duplicate_returns_keep_their_weight_in_intensity_mean():
    source = cloud([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 6]])
    result = voxel_downsample(source, 1)
    np.testing.assert_array_equal(result.points, [[0, 0, 0, 2]])
