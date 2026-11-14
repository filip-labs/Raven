"""Independent, deliberately tiny references for end-to-end M2 regression checks."""

import copy
import importlib
import itertools
import json
import math
import sys

import numpy as np
import pytest

from raven.core.point_cloud import PointCloud
from raven.perception.preprocessing.pipeline import (
    PipelineConfigurationError,
    PipelineExecutionError,
    run_preprocessing_pipeline,
)

MODULE = importlib.import_module("raven.perception.preprocessing.pipeline")
RANGE = {"name": "range_filter", "x_range": [-1, 21], "y_range": [-1, 1], "z_range": [-1, 1]}
VOXEL = {"name": "voxel_downsample", "voxel_size": 1.0}
STAT = {"name": "statistical_outlier_removal", "nb_neighbors": 1, "std_ratio": 1.0}
NOISE = {"name": "noise", "position_std": 0.125, "seed": 42}
STAGES = [RANGE, VOXEL, STAT, NOISE]
ROWS = [
    [2.25, 0, 0, 80],
    [0.75, 0, 0, 6],
    [-0.75, 0, 0, 2],
    [0.25, 0, 0, 2],
    [-0.25, 0, 0, 6],
    [20.25, 0, 0, 100],
    [100.25, 0, 0, 1000],
]


def reference(points, stage):
    """No Raven operations/validators. Exhaustive distances only for <=10 rows.

    Python floor/dict cells, fsum centroids and math.dist/population statistics
    independently implement the mathematical contract. Noise uses PCG64 standard
    normals scaled explicitly, rather than Raven's Generator.normal call.
    Each stage ends with float32 storage, including intermediate pipeline stages.
    """
    assert len(points) <= 10, "This exhaustive oracle is forbidden for real scans"
    rows = points.tolist()
    name = stage["name"]
    if name == "range_filter":
        bounds = [stage[key] for key in ("x_range", "y_range", "z_range")]
        rows = [
            row
            for row in rows
            if all(lo <= v <= hi for v, (lo, hi) in zip(row[:3], bounds, strict=True))
        ]
    elif name == "voxel_downsample":
        size = stage["voxel_size"]
        sizes = [size] * 3 if np.isscalar(size) else size
        cells = {}
        for row in rows:
            cell = tuple(math.floor(v / s) for v, s in zip(row[:3], sizes, strict=True))
            cells.setdefault(cell, []).append(row)
        rows = [
            [math.fsum(row[col] for row in cells[cell]) / len(cells[cell]) for col in range(4)]
            for cell in sorted(cells)
        ]
    elif name == "statistical_outlier_removal" and len(rows) > 1:
        k = min(stage["nb_neighbors"], len(rows) - 1)
        means = [
            math.fsum(
                sorted(math.dist(row[:3], other[:3]) for j, other in enumerate(rows) if i != j)[:k]
            )
            / k
            for i, row in enumerate(rows)
        ]
        mean = math.fsum(means) / len(means)
        std = math.sqrt(math.fsum((value - mean) ** 2 for value in means) / len(means))
        rows = [
            row
            for row, distance in zip(rows, means, strict=True)
            if distance <= mean + stage["std_ratio"] * std
        ]
    elif name == "noise":
        samples = np.random.Generator(np.random.PCG64(stage["seed"])).standard_normal(
            (len(rows), 3)
        )
        rows = [
            [
                float(row[axis]) + float(samples[i, axis]) * stage["position_std"]
                for axis in range(3)
            ]
            + [row[3]]
            for i, row in enumerate(rows)
        ]
    return np.asarray(rows, dtype=np.float32).reshape(-1, 4)


ORDERS = list(itertools.permutations(range(4))) + [(0, 1, 2, 3, 3), (3, 0, 1, 2, 3)]


@pytest.mark.parametrize("order", ORDERS, ids=lambda order: "-".join(map(str, order)))
def test_full_pipeline_orders_and_repeats_against_independent_reference(order):
    configs = copy.deepcopy([STAGES[i] for i in order])
    if len(order) > 4:
        configs[-1]["seed"] = 7  # Each repeated noise occurrence is explicit.
    config_before = copy.deepcopy(configs)
    source = PointCloud(np.array(ROWS), "synthetic-frame", "caller-frame")
    before = source.points.tobytes()
    expected = source.points.copy()
    counts = []
    for stage in configs:
        input_count = len(expected)
        expected = reference(expected, stage)
        counts.append((input_count, len(expected)))
    result = run_preprocessing_pipeline(source, configs)
    assert result.cloud.points.tobytes() == expected.tobytes()
    assert (
        result.cloud.points.tobytes()
        == run_preprocessing_pipeline(source, configs).cloud.points.tobytes()
    )
    assert source.points.tobytes() == before and configs == config_before
    assert not np.shares_memory(source.points, result.cloud.points)
    assert not result.cloud.points.flags.writeable
    metadata = json.loads(json.dumps(result.to_metadata_dict(), allow_nan=False))
    # Exact allowlists protect provenance from accidentally serializing raw XYZI.
    assert set(metadata) == {
        "input_count",
        "output_count",
        "frame_id",
        "coordinate_frame",
        "duration_seconds",
        "reproducibility",
        "stages",
    }
    assert set(metadata["reproducibility"]) == {"python", "numpy", "scipy", "noise_rng"}
    assert all(isinstance(v, str) for v in metadata["reproducibility"].values())
    assert metadata["input_count"] == len(ROWS) and metadata["output_count"] == len(expected)
    assert result.cloud.frame_id == metadata["frame_id"] == source.frame_id
    assert result.cloud.coordinate_frame == metadata["coordinate_frame"] == source.coordinate_frame
    assert math.isfinite(metadata["duration_seconds"])
    assert metadata["duration_seconds"] >= sum(s["duration_seconds"] for s in metadata["stages"])
    for position, (record, config, (before_count, after_count)) in enumerate(
        zip(metadata["stages"], configs, counts, strict=True), 1
    ):
        assert set(record) == {
            "position",
            "name",
            "parameters",
            "input_count",
            "output_count",
            "points_removed",
            "duration_seconds",
            "frame_id",
            "coordinate_frame",
        }
        assert record["position"] == position and record["name"] == config["name"]
        assert record["parameters"] == {k: v for k, v in config.items() if k != "name"}
        assert (record["input_count"], record["output_count"]) == (before_count, after_count)
        assert record["points_removed"] == (
            None if config["name"] == "noise" else before_count - after_count
        )
        assert (
            record["frame_id"] == source.frame_id
            and record["coordinate_frame"] == source.coordinate_frame
        )
        assert math.isfinite(record["duration_seconds"]) and record["duration_seconds"] >= 0


def test_hand_calculated_four_stage_geometry_and_population_selection():
    source = PointCloud(np.array(ROWS), "000000", "kitti_velodyne")
    # Range removes only X=100.25. Voxel centers/intensities become:
    # (-.5,4), (.5,4), (2.25,80), (20.25,100). K=1 means are
    # [1,1,1.75,18]; mu=5.4375 and population variance=52.69921875.
    # At ratio=1, threshold is between 2 and 18: retain the first three.
    assert math.isclose(math.fsum((v - 5.4375) ** 2 for v in [1, 1, 1.75, 18]) / 4, 52.69921875)
    retained = np.array([[-0.5, 0, 0, 4], [0.5, 0, 0, 4], [2.25, 0, 0, 80]], dtype=np.float64)
    retained[:, :3] += np.random.Generator(np.random.PCG64(42)).standard_normal((3, 3)) / 8
    result = run_preprocessing_pipeline(source, STAGES)
    np.testing.assert_array_equal(result.cloud.points, retained.astype(np.float32))
    assert [(s.input_count, s.output_count) for s in result.stages] == [
        (7, 6),
        (6, 4),
        (4, 3),
        (3, 3),
    ]


def test_float32_rounding_between_stages_controls_following_inclusive_crop():
    # The exact mean of adjacent float32 coordinates rounds to the even lower
    # value. Keeping it in float64 across stage boundaries would fail this crop.
    lower = np.float32(0.5)
    upper = np.nextafter(lower, np.float32(np.inf))
    source = PointCloud(np.array([[lower, 0, 0, 2], [upper, 0, 0, 6]]), "rounding", "lidar")
    crop = dict(RANGE, x_range=[0, float(lower)])
    first = run_preprocessing_pipeline(source, [VOXEL, crop])
    second = run_preprocessing_pipeline(source, [crop, VOXEL])
    np.testing.assert_array_equal(first.cloud.points, [[0.5, 0, 0, 4]])
    np.testing.assert_array_equal(second.cloud.points, [[0.5, 0, 0, 2]])
    assert [s.output_count for s in first.stages] == [1, 1]
    assert [s.output_count for s in second.stages] == [1, 1]


@pytest.mark.parametrize(
    "rows", [ROWS, [[0, 0, 0, 0.5]], []], ids=["becomes-empty", "singleton", "already-empty"]
)
def test_empty_transition_and_singleton_flow_need_no_scipy(monkeypatch, rows):
    monkeypatch.setitem(sys.modules, "scipy.spatial", None)
    # This crop eliminates ROWS but keeps the singleton at the origin.
    crop = dict(RANGE, x_range=[-0.01, 0.01])
    configs = [crop, VOXEL, STAT, dict(NOISE, position_std=0)]
    source = PointCloud(np.array(rows).reshape(-1, 4), "empty-flow", "kitti_velodyne")
    result = run_preprocessing_pipeline(source, configs)
    count = 1 if len(rows) == 1 else 0
    assert result.cloud.count == count
    assert [s.output_count for s in result.stages] == [count] * 4
    assert [s.input_count for s in result.stages] == [len(rows), count, count, count]
    np.testing.assert_array_equal(
        result.cloud.points, np.array([[0, 0, 0, 0.5]] if count else []).reshape(-1, 4)
    )
    assert (
        result.cloud.coordinate_frame == "kitti_velodyne"
        and result.cloud.frame_id == source.frame_id
    )
    assert not np.shares_memory(source.points, result.cloud.points)


@pytest.mark.parametrize(
    "invalid",
    [
        dict(RANGE, x_range=[0, 0]),
        dict(VOXEL, voxel_size=-1),
        dict(STAT, nb_neighbors=0),
        {"name": "noise", "position_std": 0.1},
    ],
)
def test_invalid_tail_prevents_every_registered_operation(monkeypatch, invalid):
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("No operation may execute before complete validation")

    for name in MODULE._OPERATIONS:
        monkeypatch.setitem(MODULE._OPERATIONS, name, forbidden)
    source = PointCloud(np.array(ROWS), "prevalidation")
    with pytest.raises(PipelineConfigurationError, match="Stage 5"):
        run_preprocessing_pipeline(source, [*STAGES, invalid])
    assert calls == []


@pytest.mark.parametrize(
    "failing,rows,message",
    [
        (dict(VOXEL, voxel_size=1e-300), [[1, 0, 0, 0.5]], "voxel-index range"),
        (
            dict(STAT, std_ratio=1e308),
            [[0, 0, 0, 1], [1, 0, 0, 2], [1e38, 0, 0, 3]],
            "nonfinite statistical threshold",
        ),
    ],
)
def test_data_dependent_failures_after_valid_stage_preserve_source(failing, rows, message):
    source = PointCloud(np.array(rows), "runtime", "caller-frame")
    original = source.points.tobytes()
    with pytest.raises(
        PipelineExecutionError, match=f"Stage 2 .*{failing['name']}.*{message}"
    ) as error:
        run_preprocessing_pipeline(source, [dict(NOISE, position_std=0), failing])
    assert isinstance(error.value.__cause__, ValueError)
    assert source.points.tobytes() == original and not source.points.flags.writeable
