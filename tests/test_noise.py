import subprocess
import sys

import numpy as np
import pytest

from raven.core.point_cloud import PointCloud
from raven.perception.preprocessing.noise import inject_noise
from raven.perception.preprocessing.range_filter import filter_range
from raven.perception.preprocessing.statistical_outlier_removal import statistical_outlier_removal
from raven.perception.preprocessing.voxel_downsample import voxel_downsample


def cloud(rows):
    return PointCloud(np.array(rows, dtype=np.float32).reshape(-1, 4), "000000", "kitti_velodyne")


def test_same_seed_repeatability_and_different_seed_changes_xyz():
    source = cloud([[1, 2, 3, 0.4], [4, 5, 6, 0.7]])
    first = inject_noise(source, 0.05, 42)
    np.testing.assert_array_equal(first.points, inject_noise(source, 0.05, 42).points)
    assert not np.array_equal(first.points[:, :3], inject_noise(source, 0.05, 43).points[:, :3])
    assert not np.array_equal(first.points[:, :3], source.points[:, :3])
    np.testing.assert_array_equal(first.points[:, 3], source.points[:, 3])


def test_exact_local_pcg64_normal_parameterization_and_float64_addition():
    source = cloud([[0.1, 0.2, 0.3, -2], [1, 2, 3, 9], [-4, 5, -6, 0.4]])
    # Explicit reference checks scale=standard deviation (not variance), XYZ draw
    # shape/order and float64 addition before one final float32 rounding.
    noise = np.random.Generator(np.random.PCG64(17)).normal(0, 0.2, (3, 3))
    expected = source.points.astype(np.float64)
    expected[:, :3] += noise
    result = inject_noise(source, 0.2, 17)
    np.testing.assert_array_equal(result.points, expected.astype(np.float32))
    assert result.count == source.count
    assert result.frame_id == source.frame_id and result.coordinate_frame == source.coordinate_frame


def test_distribution_per_axis_spread_and_independence():
    # Broad deterministic bounds: test the model, without requiring sample moments
    # to equal theoretical values or imposing a brittle exact random fixture.
    source = cloud(np.zeros((30000, 4)))
    xyz = inject_noise(source, 0.2, 123).points[:, :3].astype(float)
    assert (np.abs(xyz.mean(axis=0)) < 0.004).all()
    np.testing.assert_allclose(xyz.std(axis=0), 0.2, rtol=0.025)
    corr = np.corrcoef(xyz.T)
    assert np.max(np.abs(corr - np.eye(3))) < 0.025


@pytest.mark.parametrize("std", [0, -0.0, np.float32(0)])
def test_zero_noise_bitwise_exact_independent_copy(std):
    source = cloud([[0.1, -0.0, -2, -0.0], [np.finfo(np.float32).max, 1, 0, 0.4]])
    result = inject_noise(source, std, 42)
    assert result.points.tobytes() == source.points.tobytes()
    assert not np.shares_memory(result.points, source.points)
    assert not result.points.flags.writeable


@pytest.mark.parametrize("rows", [[], [[1, 2, 3, 0.5]]])
@pytest.mark.parametrize("std", [0, 0.01])
def test_empty_and_singleton_preserve_invariants(rows, std):
    source = cloud(rows)
    result = inject_noise(source, std, 42)
    assert result.count == source.count and result.points.shape == (len(rows), 4)
    assert result.frame_id == source.frame_id and result.coordinate_frame == source.coordinate_frame
    assert result.points.dtype == np.float32 and not result.points.flags.writeable
    assert not np.shares_memory(result.points, source.points)
    np.testing.assert_array_equal(result.points[:, 3], source.points[:, 3])


def test_immutability_intensity_bits_and_original_row_correspondence():
    source = cloud([[i, i * 2, -i, i - 3] for i in range(8)])
    source.points.setflags(write=True)
    source.points[3, 3] = -0.0
    source.points.setflags(write=False)
    original = source.points.copy()
    result = inject_noise(source, 0.1, 3)
    assert result.points[:, 3].tobytes() == source.points[:, 3].tobytes()
    assert not np.shares_memory(result.points, source.points)
    np.testing.assert_array_equal(source.points, original)
    result.points.setflags(write=True)
    result.points[:] = 999
    np.testing.assert_array_equal(source.points, original)
    assert not source.points.flags.writeable


def test_global_rng_and_unrelated_generator_state_unchanged():
    before = np.random.get_state()
    generator = np.random.default_rng(99)
    expected = np.random.default_rng(99).random(4)
    source = cloud([[1, 2, 3, 4]])
    inject_noise(source, 0.1, 42)
    inject_noise(source, 0, 42)
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    np.testing.assert_array_equal(generator.random(4), expected)


@pytest.mark.parametrize(
    "std",
    [
        -1,
        np.nan,
        np.inf,
        -np.inf,
        True,
        np.bool_(True),
        "0.1",
        None,
        1j,
        [1],
        (1, 1, 1),
        np.array(0.1),
        10**400,
    ],
)
@pytest.mark.parametrize("empty", [False, True])
def test_invalid_standard_deviation(std, empty):
    with pytest.raises((TypeError, ValueError), match="position_std"):
        inject_noise(cloud([] if empty else [[0, 0, 0, 1]]), std, 42)


@pytest.mark.parametrize(
    "seed",
    [
        -1,
        -(10**100),
        None,
        True,
        np.bool_(False),
        1.0,
        0.5,
        np.nan,
        np.inf,
        "42",
        [42],
        np.array(42),
        1j,
        np.random.default_rng(42),
    ],
)
@pytest.mark.parametrize("std", [0, 0.1])
def test_invalid_seed_even_for_noop(seed, std):
    for source in (cloud([]), cloud([[0, 0, 0, 1]])):
        with pytest.raises((TypeError, ValueError), match="seed"):
            inject_noise(source, std, seed)


@pytest.mark.parametrize("seed", [0, np.int64(42), np.uint64(42), 10**100])
def test_explicit_nonnegative_integer_seeds(seed):
    source = cloud([[0, 0, 0, 1]])
    np.testing.assert_array_equal(
        inject_noise(source, 0.1, seed).points, inject_noise(source, 0.1, seed).points
    )


def test_missing_seed_is_rejected_by_signature():
    with pytest.raises(TypeError, match="seed"):
        inject_noise(cloud([]), 0.1)


@pytest.mark.parametrize("std", [np.float32(0.1), np.float64(0.1), 1])
def test_supported_numeric_std_types(std):
    source = cloud([[0, 0, 0, 1]])
    np.testing.assert_array_equal(
        inject_noise(source, std, 1).points, inject_noise(source, float(std), 1).points
    )


@pytest.mark.parametrize("std", [1e39, np.finfo(np.float64).max])
def test_extreme_noise_rejected_instead_of_clipped_or_resampled(std):
    source = cloud([[0, 0, 0, 1]] * 10)
    before = source.points.copy()
    with pytest.raises(ValueError, match="finite.*float32"):
        inject_noise(source, std, 42)
    np.testing.assert_array_equal(source.points, before)


def test_finite_noise_can_push_finite_input_out_of_float32_range():
    limit = np.finfo(np.float32).max
    source = cloud([[limit, 0, 0, 1]])
    # Seed 42 starts with positive X noise. Each operand is finite, but their
    # sum exceeds the maximum finite float32 value. Do not saturate silently.
    with pytest.raises(ValueError, match="float32 storage range"):
        inject_noise(source, float(limit) / 2, 42)


def test_small_noise_may_round_away_without_modifying_other_fields():
    source = cloud([[1e20, -1e20, 1e20, 7]])
    np.testing.assert_array_equal(inject_noise(source, 1e-100, 42).points, source.points)


def test_subnormal_std_is_valid_and_rounding_to_float32_zero_is_allowed():
    source = cloud([[0, 0, 0, 1]])
    result = inject_noise(source, np.nextafter(0.0, 1.0), 42)
    assert np.isfinite(result.points).all()
    np.testing.assert_array_equal(result.points, source.points)


def test_explicit_composition_with_existing_preprocessing_operations():
    source = cloud([[0.1, 0, 0, 1], [0.2, 0, 0, 3], [1.1, 0, 0, 5], [2.1, 0, 0, 7], [100, 0, 0, 9]])
    cropped = filter_range(source, (-1, 3), (-1, 1), (-1, 1))
    downsampled = voxel_downsample(cropped, 1)
    selected = statistical_outlier_removal(downsampled, 1, 1)
    result = inject_noise(selected, 0.01, 42)
    assert result.count == selected.count
    assert result.frame_id == source.frame_id and result.coordinate_frame == source.coordinate_frame
    np.testing.assert_array_equal(result.points[:, 3], selected.points[:, 3])
    # Noise alone performs none of the other stages automatically.
    assert inject_noise(source, 0.01, 42).count == source.count


def test_numpy_only_import_and_execution_without_optional_frameworks():
    program = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'scipy','torch','pcdet','spconv','cumm','open3d'}:
            raise AssertionError('Unexpected optional dependency: ' + fullname)
sys.meta_path.insert(0, Block())
import numpy as np
from raven.core.point_cloud import PointCloud
from raven.perception.preprocessing.noise import inject_noise
source = PointCloud(np.array([[1,2,3,.5]]), 'frame', 'caller-frame')
result = inject_noise(source, .01, 42)
assert result.count == 1 and result.coordinate_frame == 'caller-frame'
assert result.frame_id == 'frame' and result.points[0,3] == source.points[0,3]
"""
    subprocess.run([sys.executable, "-c", program], check=True, capture_output=True, text=True)


def test_nonzero_std_that_cannot_be_represented_in_float64_is_rejected():
    from fractions import Fraction

    for std in (Fraction(1, 10**400), Fraction(-1, 10**400)):
        with pytest.raises(ValueError, match="position_std"):
            inject_noise(cloud([]), std, 42)
