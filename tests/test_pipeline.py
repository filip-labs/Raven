import copy
import importlib
import json
import subprocess
import sys
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from raven.core.point_cloud import PointCloud
from raven.perception.preprocessing.noise import inject_noise
from raven.perception.preprocessing.pipeline import (
    PipelineConfigurationError,
    PipelineExecutionError,
    run_preprocessing_pipeline,
)
from raven.perception.preprocessing.range_filter import filter_range
from raven.perception.preprocessing.statistical_outlier_removal import statistical_outlier_removal
from raven.perception.preprocessing.voxel_downsample import voxel_downsample

MODULE = importlib.import_module("raven.perception.preprocessing.pipeline")
RANGE = {"name": "range_filter", "x_range": (-1, 3), "y_range": (-1, 1), "z_range": (-1, 1)}
VOXEL = {"name": "voxel_downsample", "voxel_size": 1}
STAT = {"name": "statistical_outlier_removal", "nb_neighbors": 1, "std_ratio": 1}
NOISE = {"name": "noise", "position_std": 0.01, "seed": 42}


def cloud(rows=None):
    if rows is None:
        rows = [[0.1, 0, 0, 1], [0.3, 0, 0, 3], [1.1, 0, 0, 5], [2.1, 0, 0, 7], [100, 0, 0, 9]]
    return PointCloud(np.array(rows, dtype=np.float32).reshape(-1, 4), "000000", "kitti_velodyne")


@pytest.mark.parametrize(
    "config,operation",
    [
        (RANGE, filter_range),
        (VOXEL, voxel_downsample),
        (STAT, statistical_outlier_removal),
        (NOISE, inject_noise),
    ],
)
def test_each_stage_matches_standalone_operation(config, operation):
    source = cloud()
    parameters = {k: v for k, v in config.items() if k != "name"}
    result = run_preprocessing_pipeline(source, [config])
    np.testing.assert_array_equal(result.cloud.points, operation(source, **parameters).points)
    (record,) = result.stages
    assert record.position == 1 and record.name == config["name"]
    assert record.input_count == source.count and record.output_count == result.cloud.count
    assert record.points_removed == (
        None if config["name"] == "noise" else source.count - result.cloud.count
    )
    assert record.frame_id == source.frame_id and record.coordinate_frame == source.coordinate_frame
    assert (
        result.cloud.frame_id == source.frame_id
        and result.cloud.coordinate_frame == source.coordinate_frame
    )


@pytest.mark.parametrize(
    "configs",
    [
        [RANGE, VOXEL, STAT, NOISE],
        [NOISE, STAT, VOXEL, RANGE],
        [VOXEL, RANGE],
        [RANGE, RANGE, NOISE, NOISE],
        [VOXEL, VOXEL],
    ],
)
def test_compositions_and_repetitions_run_in_exact_order(configs):
    source = cloud()
    operations = {
        "range_filter": filter_range,
        "voxel_downsample": voxel_downsample,
        "statistical_outlier_removal": statistical_outlier_removal,
        "noise": inject_noise,
    }
    expected = source
    counts = []
    for config in configs:
        before = expected.count
        expected = operations[config["name"]](
            expected, **{k: v for k, v in config.items() if k != "name"}
        )
        counts.append((before, expected.count))
    result = run_preprocessing_pipeline(source, configs)
    np.testing.assert_array_equal(result.cloud.points, expected.points)
    assert [(s.input_count, s.output_count) for s in result.stages] == counts
    assert [s.name for s in result.stages] == [c["name"] for c in configs]
    assert [s.position for s in result.stages] == list(range(1, len(configs) + 1))
    np.testing.assert_array_equal(
        result.cloud.points, run_preprocessing_pipeline(source, configs).cloud.points
    )


def test_independent_known_range_then_voxel_result_and_order_sensitivity():
    source = cloud([[0.1, 0, 0, 1], [0.3, 0, 0, 3], [1.1, 0, 0, 5]])
    crop = dict(RANGE, x_range=(0.2, 2))
    forward = run_preprocessing_pipeline(source, [crop, VOXEL])
    reverse = run_preprocessing_pipeline(source, [VOXEL, crop])
    np.testing.assert_array_equal(forward.cloud.points, source.points[1:])
    np.testing.assert_array_equal(
        reverse.cloud.points, np.array([[0.2, 0, 0, 2], [1.1, 0, 0, 5]], dtype=np.float32)
    )
    assert forward.stages[0].points_removed == 1


@pytest.mark.parametrize("rows", [[], [[1, 2, 3, 0.5]], None])
@pytest.mark.parametrize("configs", [[], [RANGE, VOXEL, STAT, NOISE]])
def test_empty_identity_empty_and_single_clouds(rows, configs):
    source = cloud(rows)
    result = run_preprocessing_pipeline(source, configs)
    assert not np.shares_memory(result.cloud.points, source.points)
    assert not result.cloud.points.flags.writeable
    assert (
        result.cloud.frame_id == source.frame_id
        and result.cloud.coordinate_frame == source.coordinate_frame
    )
    if not configs:
        np.testing.assert_array_equal(result.cloud.points, source.points)
        assert result.stages == ()
    if rows == []:
        assert result.cloud.points.shape == (0, 4)
        assert all(s.input_count == s.output_count == 0 for s in result.stages)


def test_noop_stages_preserve_single_point_and_intensity_bits():
    source = cloud([[0.25, 0, 0, -0.0]])
    result = run_preprocessing_pipeline(source, [RANGE, VOXEL, STAT, dict(NOISE, position_std=0)])
    assert result.cloud.points.tobytes() == source.points.tobytes()


def test_source_configuration_and_result_storage_are_independent():
    source = cloud()
    before = source.points.copy()
    configs = [
        dict(RANGE, x_range=[-1, 3]),
        dict(VOXEL, voxel_size=np.array([1.0, 1.0, 1.0])),
        NOISE.copy(),
    ]
    config_before = copy.deepcopy(configs)
    result = run_preprocessing_pipeline(source, configs)
    assert configs[0] == config_before[0] and configs[2] == config_before[2]
    np.testing.assert_array_equal(configs[1]["voxel_size"], config_before[1]["voxel_size"])
    configs[0]["x_range"][0] = -99
    configs[1]["voxel_size"][0] = 99
    assert result.stages[0].parameters["x_range"] == (-1.0, 3.0)
    assert result.stages[1].parameters["voxel_size"] == (1.0, 1.0, 1.0)
    with pytest.raises(TypeError):
        result.stages[0].parameters["x_range"] = (1, 2)
    with pytest.raises(TypeError):
        result.reproducibility["numpy"] = "different"
    with pytest.raises(FrozenInstanceError):
        result.stages[0].position = 9
    result.cloud.points.setflags(write=True)
    result.cloud.points[:] = 999
    np.testing.assert_array_equal(source.points, before)


@pytest.mark.parametrize(
    "bad",
    [
        {},
        {"name": "unknown"},
        {"name": ["noise"]},
        {"name": "Noise"},
        {"name": "filter_range"},
        {"name": "noise", "position_std": 0.1},
        dict(NOISE, seed=None),
        dict(NOISE, position_std=-1),
        dict(NOISE, extra=1),
        dict(NOISE, seed=True),
        dict(NOISE, position_std=np.nan),
        {"name": "voxel_downsample"},
        dict(VOXEL, voxel_size=0),
        dict(VOXEL, voxel_size=[1, 2]),
        dict(STAT, nb_neighbors=0),
        dict(STAT, std_ratio=np.inf),
        dict(STAT, nb_neighbors=True),
        dict(RANGE, x_range=(3, -1)),
        dict(RANGE, z_range=[0, np.inf]),
        dict(RANGE, y_range=[0]),
        dict(RANGE, x_range=True),
        dict(NOISE, params={"seed": 42}),
        None,
        "noise",
    ],
)
def test_invalid_later_configuration_prevents_all_processing(monkeypatch, bad):
    calls = []
    monkeypatch.setitem(MODULE._OPERATIONS, "range_filter", lambda *a, **kw: calls.append("ran"))
    source = cloud()
    before = source.points.copy()
    with pytest.raises(PipelineConfigurationError, match="Stage 2"):
        run_preprocessing_pipeline(source, [RANGE, bad])
    assert calls == []
    np.testing.assert_array_equal(source.points, before)


@pytest.mark.parametrize("configs", [None, "noise", {}, set(), iter([NOISE]), np.array([NOISE])])
def test_stage_container_must_be_ordered_reusable_sequence(configs):
    with pytest.raises(PipelineConfigurationError, match="ordered sequence"):
        run_preprocessing_pipeline(cloud(), configs)


def test_iterators_in_parameters_rejected_without_consuming_configuration():
    iterator = iter([-1, 3])
    with pytest.raises(PipelineConfigurationError, match="x_range"):
        run_preprocessing_pipeline(cloud(), [dict(RANGE, x_range=iterator)])
    assert list(iterator) == [-1, 3]


def test_tuple_sequence_numpy_parameters_and_explicit_seed_provenance():
    result = run_preprocessing_pipeline(
        cloud(), (dict(VOXEL, voxel_size=np.float32(0.5)), dict(NOISE, seed=np.int64(42)))
    )
    metadata = result.to_metadata_dict()
    assert metadata["stages"][0]["parameters"]["voxel_size"] == 0.5
    assert metadata["stages"][1]["parameters"]["seed"] == 42
    assert metadata["reproducibility"]["noise_rng"] == "numpy.random.Generator(PCG64)"
    json.dumps(metadata, allow_nan=False)


def test_metadata_export_is_detached_and_contains_no_point_data():
    result = run_preprocessing_pipeline(cloud(), [RANGE, STAT, NOISE])
    metadata = result.to_metadata_dict()
    assert metadata["input_count"] == 5 and metadata["output_count"] == result.cloud.count
    assert metadata["frame_id"] == "000000" and metadata["coordinate_frame"] == "kitti_velodyne"
    assert metadata["reproducibility"]["scipy"]
    metadata["stages"][0]["parameters"]["x_range"][0] = 999
    metadata["reproducibility"]["numpy"] = "modified"
    assert result.stages[0].parameters["x_range"] == (-1.0, 3.0)
    assert result.reproducibility["numpy"] == np.__version__
    assert "points" not in metadata
    assert all("points" not in stage for stage in metadata["stages"])
    assert run_preprocessing_pipeline(cloud(), []).to_metadata_dict()["stages"] == []


def test_monotonic_stage_and_separate_total_timings(monkeypatch):
    ticks = iter([10.0, 11.0, 13.0, 14.0, 17.0, 20.0])
    monkeypatch.setattr(MODULE, "perf_counter", lambda: next(ticks))
    result = run_preprocessing_pipeline(cloud(), [RANGE, NOISE])
    assert [s.duration_seconds for s in result.stages] == [2.0, 3.0]
    assert result.duration_seconds == 10.0
    assert result.duration_seconds >= sum(s.duration_seconds for s in result.stages)


def test_later_runtime_failure_identifies_position_preserves_source_and_cause():
    source = cloud()
    before = source.points.copy()
    with pytest.raises(PipelineExecutionError, match="Stage 2 .*noise.*float32") as error:
        run_preprocessing_pipeline(source, [RANGE, dict(NOISE, position_std=1e39)])
    assert isinstance(error.value.__cause__, ValueError)
    np.testing.assert_array_equal(source.points, before)


def test_optional_scipy_only_needed_by_nontrivial_statistical_stage(monkeypatch):
    monkeypatch.setitem(sys.modules, "scipy.spatial", None)
    assert run_preprocessing_pipeline(cloud(), [RANGE, VOXEL, NOISE]).cloud.count > 0
    assert run_preprocessing_pipeline(cloud([]), [STAT]).cloud.count == 0
    assert run_preprocessing_pipeline(cloud([[0, 0, 0, 1]]), [STAT]).cloud.count == 1
    with pytest.raises(
        PipelineExecutionError, match=r"Stage 2 .*statistical_outlier_removal.*\[preprocessing\]"
    ):
        run_preprocessing_pipeline(cloud(), [RANGE, STAT])


def test_pipeline_import_and_nonstatistical_execution_without_optional_frameworks():
    program = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'scipy','torch','pcdet','spconv','cumm','open3d'}:
            raise AssertionError('Unexpected dependency: ' + fullname)
sys.meta_path.insert(0, Block())
import numpy as np
from raven.core.point_cloud import PointCloud
from raven.perception.preprocessing.pipeline import run_preprocessing_pipeline
cloud = PointCloud(np.array([[0,0,0,1],[1,0,0,2]]), 'frame', 'caller-frame')
result = run_preprocessing_pipeline(cloud, [{'name':'noise','position_std':.01,'seed':42}])
assert result.cloud.count == 2 and result.cloud.coordinate_frame == 'caller-frame'
assert 'scipy' not in result.reproducibility
"""
    subprocess.run([sys.executable, "-c", program], check=True, capture_output=True, text=True)
