import importlib
import math
import subprocess
import sys

import numpy as np
import pytest

from raven.core.point_cloud import PointCloud
from raven.perception.preprocessing.statistical_outlier_removal import statistical_outlier_removal


def cloud(rows):
    return PointCloud(np.array(rows, dtype=np.float32).reshape(-1, 4), "000000", "caller-frame")


def oracle(source, k, ratio):
    """Independent exhaustive reference for tiny test fixtures only."""
    if source.count < 2:
        return source.points.copy()
    xyz = source.points[:, :3].astype(float).tolist()
    k = min(k, len(xyz) - 1)
    means = [
        math.fsum(sorted(math.dist(p, q) for j, q in enumerate(xyz) if i != j)[:k]) / k
        for i, p in enumerate(xyz)
    ]
    mean = math.fsum(means) / len(means)
    std = math.sqrt(math.fsum((value - mean) ** 2 for value in means) / len(means))
    return source.points[np.array(means) <= mean + ratio * std]


def test_isolated_outlier_and_intensity_order_metadata():
    source = cloud([[0, 0, 0, -2], [0.1, 0, 0, 10], [-0.1, 0, 0, 7], [100, 0, 0, 5]])
    result = statistical_outlier_removal(source, 2, 1)
    np.testing.assert_array_equal(result.points, source.points[:3])
    assert result.frame_id == source.frame_id and result.coordinate_frame == source.coordinate_frame
    assert result.points.dtype == np.float32 and not result.points.flags.writeable


@pytest.mark.parametrize("k", [1, 2, 6, 7, 100, 10**100])
@pytest.mark.parametrize("ratio", [0, 0.5, 1, 2])
def test_dense_cluster_equal_distances_and_duplicate_rows_against_oracle(k, ratio):
    source = cloud(
        [
            [0, 0, 0, 1],
            [0, 0, 0, 2],
            [1, 0, 0, 3],
            [-1, 0, 0, 4],
            [0, 1, 0, 5],
            [0, -1, 0, 6],
            [20, 20, 20, 7],
        ]
    )
    np.testing.assert_array_equal(
        statistical_outlier_removal(source, k, ratio).points, oracle(source, k, ratio)
    )


def test_threshold_equality_is_inclusive_and_std_is_population():
    # K=1 gives means [0,0,2,2], population mean=1 and std=1.
    source = cloud([[0, 0, 0, 1], [0, 0, 0, 2], [2, 0, 0, 3], [4, 0, 0, 4]])
    np.testing.assert_array_equal(statistical_outlier_removal(source, 1, 1).points, source.points)
    np.testing.assert_array_equal(
        statistical_outlier_removal(source, 1, 0).points, source.points[:2]
    )
    # Sample std would incorrectly keep the final two rows for ratio=.9.
    np.testing.assert_array_equal(
        statistical_outlier_removal(source, 1, 0.9).points, source.points[:2]
    )


@pytest.mark.parametrize("k", [1, 2, 15, 100])
def test_duplicate_ties_keep_other_zero_distance_rows(k):
    source = cloud([[1, 2, 3, i] for i in range(16)])
    np.testing.assert_array_equal(statistical_outlier_removal(source, k, 0).points, source.points)


@pytest.mark.parametrize("rows", [[], [[1, 2, 3, 4]], [[0, 0, 0, 1], [0.1, 0.2, 0.3, 2]]])
def test_empty_singleton_and_pair_are_independent_and_unchanged(rows):
    source = cloud(rows)
    result = statistical_outlier_removal(source, 100, 0)
    np.testing.assert_array_equal(result.points, source.points)
    assert result.points.shape == (source.count, 4)
    assert result.frame_id == source.frame_id and result.coordinate_frame == source.coordinate_frame
    assert not result.points.flags.writeable and not np.shares_memory(result.points, source.points)


def test_input_immutability_and_no_memory_alias():
    source = cloud([[0, 0, 0, 3], [1, 0, 0, -2], [100, 0, 0, 9]])
    original = source.points.copy()
    result = statistical_outlier_removal(source, 1, 1)
    assert not np.shares_memory(result.points, source.points)
    result.points.setflags(write=True)
    result.points[:] = 999
    np.testing.assert_array_equal(source.points, original)
    assert not source.points.flags.writeable


@pytest.mark.parametrize("seed", range(8))
def test_deterministic_random_small_cloud_against_independent_reference(seed):
    rng = np.random.default_rng(seed)
    source = cloud(np.column_stack([rng.normal(size=(17, 3)), np.arange(17)]))
    for k in (1, 4, 16, 20):
        for ratio in (0, 0.5, 2):
            result = statistical_outlier_removal(source, k, ratio)
            np.testing.assert_array_equal(result.points, oracle(source, k, ratio))
            np.testing.assert_array_equal(
                result.points, statistical_outlier_removal(source, k, ratio).points
            )
            assert result.count > 0  # Nonnegative ratio cannot remove every finite mean.
    order = rng.permutation(source.count)
    selected = statistical_outlier_removal(source, 4, 0.5).points[:, 3]
    permuted = cloud(source.points[order])
    expected = permuted.points[np.isin(permuted.points[:, 3], selected)]
    np.testing.assert_array_equal(statistical_outlier_removal(permuted, 4, 0.5).points, expected)


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize(
    "k", [0, -1, True, np.bool_(False), 1.5, np.nan, np.inf, "3", None, np.array(2)]
)
def test_invalid_neighbor_counts(k, empty):
    with pytest.raises((TypeError, ValueError), match="nb_neighbors"):
        statistical_outlier_removal(cloud([] if empty else [[0, 0, 0, 1]]), k, 1)


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize(
    "ratio", [-1, np.nan, np.inf, -np.inf, True, np.bool_(True), "1", 1j, None, 10**400]
)
def test_invalid_std_ratios(ratio, empty):
    with pytest.raises((TypeError, ValueError), match="std_ratio"):
        statistical_outlier_removal(cloud([] if empty else [[0, 0, 0, 1]]), 1, ratio)


def test_numpy_scalar_parameters_and_float32_rows_are_preserved_exactly():
    source = cloud([[0.1, 0.2, 0.3, 0.1], [0.2, 0.4, 0.6, 0.2], [50, 50, 50, 0.3]])
    result = statistical_outlier_removal(source, np.int64(1), np.float32(1))
    np.testing.assert_array_equal(result.points, source.points[:2])
    assert float(result.points[0, 0]) != 0.1  # Stored float32 is not reconstructed decimal data.


def test_bounded_queries_use_exact_single_worker_search(monkeypatch):
    from scipy import spatial

    module = importlib.import_module("raven.perception.preprocessing.statistical_outlier_removal")
    real_tree = spatial.KDTree
    calls = []

    class RecordingTree:
        def __init__(self, xyz, **kwargs):
            assert kwargs == {"copy_data": True, "balanced_tree": True, "compact_nodes": True}
            self.tree = real_tree(xyz, **kwargs)

        def query(self, xyz, **kwargs):
            assert kwargs == {"k": 5, "eps": 0, "p": 2, "workers": 1}
            calls.append(len(xyz))
            return self.tree.query(xyz, **kwargs)

    monkeypatch.setattr(spatial, "KDTree", RecordingTree)
    monkeypatch.setattr(module, "_QUERY_ENTRY_BUDGET", 12)
    source = cloud([[i, i % 3, 0, i] for i in range(13)])
    np.testing.assert_array_equal(
        statistical_outlier_removal(source, 4, 1).points, oracle(source, 4, 1)
    )
    assert len(calls) > 1 and max(calls) == 2


def test_large_finite_ratio_with_overflowing_threshold_rejected():
    source = cloud([[0, 0, 0, 1], [1, 0, 0, 2], [1e38, 0, 0, 3]])
    with pytest.raises(ValueError, match="nonfinite statistical threshold"):
        statistical_outlier_removal(source, 1, 1e308)


def test_missing_optional_backend_is_actionable_but_small_clouds_work(monkeypatch):
    monkeypatch.setitem(sys.modules, "scipy.spatial", None)
    assert statistical_outlier_removal(cloud([]), 1, 1).count == 0
    assert statistical_outlier_removal(cloud([[0, 0, 0, 1]]), 1, 1).count == 1
    with pytest.raises(RuntimeError, match=r"optional CPU SciPy.*\[preprocessing\]"):
        statistical_outlier_removal(cloud([[0, 0, 0, 1], [1, 0, 0, 2]]), 1, 1)


def test_core_import_without_scipy_and_execution_without_gpu_frameworks():
    program = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    block_scipy = True
    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".")[0]
        if root in {"torch", "pcdet", "spconv", "cumm", "open3d"} or (self.block_scipy and root == "scipy"):
            raise AssertionError("Unexpected dependency import: " + fullname)
block = Block()
sys.meta_path.insert(0, block)
import numpy as np
from raven.core.point_cloud import PointCloud
from raven.perception.preprocessing.statistical_outlier_removal import statistical_outlier_removal
from raven.perception.preprocessing.range_filter import filter_range
from raven.perception.preprocessing.voxel_downsample import voxel_downsample
empty = PointCloud(np.empty((0,4)), "frame")
assert statistical_outlier_removal(empty, 1, 0).count == 0
assert filter_range(empty, (-1,1), (-1,1), (-1,1)).count == 0
assert voxel_downsample(empty, 1).count == 0
block.block_scipy = False
cloud = PointCloud(np.array([[0,0,0,1],[1,0,0,2]]), "frame", "kitti_velodyne")
assert statistical_outlier_removal(cloud, 1, 0).count == 2
"""
    subprocess.run([sys.executable, "-c", program], check=True, capture_output=True, text=True)
