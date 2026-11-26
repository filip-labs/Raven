"""Synthetic correctness tests; fake clocks/GPU boundaries are never performance evidence."""

import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from raven.benchmark import experiment as benchmark
from raven.benchmark.experiment import StageObserver, benchmark_experiment
from raven.benchmark.gpu import LocalCudaBackend
from raven.benchmark.spec import BenchmarkSpec, load_benchmark_spec
from raven.benchmark.timing import MAX_DURATION_NS, Timer, summarize
from raven.cli import main
from raven.experiments.config import ExperimentConfig
from raven.perception.detection.base import Detection3D
from raven.perception.preprocessing import pipeline


class Clock:
    def __init__(self):
        self.value = 0

    def __call__(self):
        self.value += 100
        return self.value


def test_statistics_independent_linear_percentiles_and_population_std():
    # Ordered ms [1,2,3,4]: type-7 percentile index=(N-1)*p.
    result = summarize([4_000_000, 1_000_000, 3_000_000, 2_000_000])
    assert result.count == 4
    assert result.mean_ms == result.median_ms == result.p50_ms == 2.5
    assert result.minimum_ms == 1 and result.maximum_ms == 4
    assert result.std_ms == pytest.approx(math.sqrt(1.25))
    assert (result.p90_ms, result.p95_ms, result.p99_ms) == pytest.approx((3.7, 3.85, 3.97))
    assert result.percentile_method == "linear" and result.std_ddof == 0
    singleton = summarize([0])
    assert singleton.mean_ms == singleton.std_ms == singleton.p99_ms == 0


@pytest.mark.parametrize(
    "values",
    [
        [],
        [-1],
        [True],
        [1.0],
        [float("nan")],
        [float("inf")],
        ["100"],
        [None],
        [MAX_DURATION_NS + 1],
    ],
)
def test_invalid_or_empty_durations_never_produce_statistics(values):
    with pytest.raises((ValueError, TypeError)):
        summarize(values)


def test_monotonic_timer_exact_boundaries_with_synchronization():
    events = []
    readings = iter([100, 325])

    def clock():
        events.append("clock")
        return next(readings)

    def sync():
        events.append("sync")

    def operation():
        events.append("operation")
        return "value"

    assert Timer(clock).call(operation, sync) == ("value", 225)
    assert events == ["sync", "clock", "operation", "sync", "clock"]


@pytest.mark.parametrize("readings", [[100, 90], [0, MAX_DURATION_NS + 1], [True], [-1], [1.2]])
def test_bad_clock_samples_rejected(readings):
    timer = Timer(iter(readings).__next__)
    with pytest.raises(ValueError):
        timer.call(lambda: None)


@pytest.mark.parametrize(
    "field,value",
    [
        ("warmup", -1),
        ("warmup", 21),
        ("warmup", True),
        ("repetitions", 0),
        ("repetitions", 101),
        ("repetitions", 2.5),
        ("include_loading", 1),
        ("mode", "saved"),
        ("mode", "modal"),
        ("schema_version", 2),
        ("schema_version", True),
    ],
)
def test_spec_rejects_unsafe_or_unsupported_settings(field, value):
    with pytest.raises(ValueError):
        BenchmarkSpec(**{field: value})


@pytest.mark.parametrize(
    "raw",
    [
        "[]",
        "NaN",
        "{}",
        '{"warmup":1,"warmup":2}',
        '{"schema_version":1,"mode":"preprocessing","warmup":0,"repetitions":1,"include_loading":false,"extra":1}',
    ],
)
def test_spec_file_is_strict_json(tmp_path, raw):
    path = tmp_path / "spec.json"
    path.write_text(raw)
    with pytest.raises(ValueError):
        load_benchmark_spec(path)


def test_defaults_round_trip_and_work_caps(tmp_path):
    spec = BenchmarkSpec()
    assert spec.warmup == 5 and spec.repetitions == 30 and not spec.include_loading
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec.to_dict()))
    assert load_benchmark_spec(path) == spec
    with pytest.raises(ValueError):
        spec.validate_size([str(i) for i in range(33)])
    with pytest.raises(ValueError):
        BenchmarkSpec(warmup=20, repetitions=100).validate_size([str(i) for i in range(9)])


@pytest.fixture
def config(tmp_path):
    root = tmp_path / "dataset"
    scans = root / "training/velodyne"
    scans.mkdir(parents=True)
    for frame in ("000001", "000002"):
        np.array(
            [[0, 0, 0, 0.1], [1, 0, 0, 0.2], [3, 0, 0, 0.3], [100, 0, 0, 0.4]], dtype="<f4"
        ).tofile(scans / f"{frame}.bin")
    # Saved files/GT/calib intentionally absent: preprocessing needs only scans.
    return ExperimentConfig.from_dict(
        {
            "schema_version": 1,
            "name": "Synthetic benchmark",
            "experiment_id": "timing-test",
            "dataset": {"root": str(root), "split": "val"},
            "frames": {"ids": ["000001"]},
            "model": {
                "name": "pointpillars",
                "source": "saved",
                "predictions_dir": str(tmp_path / "absent-predictions"),
            },
            "preprocessing": [],
            "seed": 42,
            "output": {"directory": str(tmp_path / "runs"), "artifact_prefix": "experiment"},
        },
        base_dir=tmp_path,
    )


def with_stages(config, stages):
    doc = config.to_dict()
    # Existing v1 forbids saved predictions plus new processing. Local model paths are declarations only in CPU mode.
    doc["model"] = {
        "name": "pointpillars",
        "source": "inference",
        "config": "/unused/model.yaml",
        "checkpoint": "/unused/weights.pth",
        "checkpoint_sha256": "a" * 64,
        "openpcdet_revision": "233f849829b6ac19afb8af8837a0246890908755",
    }
    doc["preprocessing"] = stages
    return ExperimentConfig.from_dict(doc, base_dir=Path.cwd())


def result_document(result):
    return json.loads((Path(result.output_directory) / "benchmark.json").read_text())


def test_baseline_warmup_excluded_and_atomic_artifact_integrity(config):
    result = benchmark_experiment(
        config, specification=BenchmarkSpec(warmup=2, repetitions=3), _clock=Clock()
    )
    assert result.status == "success" and result.evidence_kind == "synthetic_test"
    frame = result.frames[0]
    assert [r["phase"] for r in frame.observations] == [
        "warmup",
        "warmup",
        "measured",
        "measured",
        "measured",
    ]
    assert len(frame.series) == 2  # No invented zero-operation series.
    for series in frame.series:
        assert series.summary["count"] == 3 and len(series.raw_ns) == 3
    assert all(v["input_count"] == v["output_count"] == 4 for v in frame.observations)
    assert len({v["processed_xyzi_sha256"] for v in frame.observations}) == 1
    assert result.initialization is None
    assert "inference_adapter" in result.unavailable
    assert not config.run_directory.exists()
    document = result_document(result)
    assert document == json.loads(json.dumps(result.to_dict()))
    assert document["benchmark_schema_version"] == 1
    manifest = json.loads((Path(result.output_directory) / "manifest.json").read_text())
    for identity in manifest["artifacts"].values():
        assert (
            hashlib.sha256(
                (Path(result.output_directory) / identity["file"]).read_bytes()
            ).hexdigest()
            == identity["sha256"]
        )
    detached = result.to_dict()
    detached["frames"][0]["source"]["sha256"] = "corrupted"
    assert result.frames[0].source["sha256"] != "corrupted"


def test_warmup_large_durations_do_not_affect_measured_summary(config):
    count = 0
    clock = Clock()

    def measured_pipeline(cloud, stages, **kwargs):
        nonlocal count
        count += 1
        if count <= 2:
            clock.value += 1_000_000_000
        return pipeline.run_preprocessing_pipeline(cloud, stages, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(benchmark, "run_preprocessing_pipeline", measured_pipeline)
        result = benchmark_experiment(
            config, specification=BenchmarkSpec(warmup=2, repetitions=2), _clock=clock
        )
    times = next(s for s in result.frames[0].series if s.stage == "preprocessing")
    assert times.raw_ns == (100, 100)
    assert result.frames[0].observations[0]["durations_ns"]["preprocessing"] > 1_000_000_000


@pytest.mark.parametrize(
    "stages,expected_count",
    [
        ([{"name": "range_filter", "x_range": [0, 3], "y_range": [-1, 1], "z_range": [-1, 1]}], 3),
        ([{"name": "voxel_downsample", "voxel_size": 2}], 3),
        ([{"name": "statistical_outlier_removal", "nb_neighbors": 1, "std_ratio": 1}], 3),
        ([{"name": "noise", "position_std": 0.01, "seed": 42}], 4),
        (
            [
                {"name": "range_filter", "x_range": [0, 3], "y_range": [-1, 1], "z_range": [-1, 1]},
                {"name": "voxel_downsample", "voxel_size": 2},
                {"name": "noise", "position_std": 0, "seed": 42},
            ],
            2,
        ),
    ],
)
def test_actual_operations_with_seeded_equivalent_inputs_and_stage_counts(
    config, stages, expected_count
):
    config = with_stages(config, stages)
    path = config.dataset.scan_directory / "000001.bin"
    before = path.read_bytes()
    result = benchmark_experiment(
        config, specification=BenchmarkSpec(warmup=1, repetitions=2), _clock=Clock()
    )
    assert result.status == "success", result.failures
    frame = result.frames[0]
    assert all(r["output_count"] == expected_count for r in frame.observations)
    assert len({r["processed_xyzi_sha256"] for r in frame.observations}) == 1
    operation_series = [s for s in frame.series if s.stage.startswith("operation/")]
    assert len(operation_series) == len(stages)
    assert all(s.summary["count"] == 2 for s in operation_series)
    for observation in frame.observations:
        records = observation["operations"]
        assert records[0]["input_count"] == 4 and records[-1]["output_count"] == expected_count
        assert observation["durations_ns"]["preprocessing"] >= sum(
            s["duration_ns"] for s in records
        )
    assert path.read_bytes() == before


def test_loading_boundary_excludes_preflight_and_hashing(config, monkeypatch):
    calls = []
    original = benchmark.load_kitti_bin

    def load(path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(benchmark, "load_kitti_bin", load)
    result = benchmark_experiment(
        config,
        specification=BenchmarkSpec(warmup=1, repetitions=2, include_loading=True),
        _clock=Clock(),
    )
    assert result.status == "success"
    assert len(calls) == 5  # preflight + source check + each of three timed loads.
    loading = next(s for s in result.frames[0].series if s.stage == "loading")
    assert loading.raw_ns == (100, 100)
    assert "loading" not in result.unavailable


def test_repeated_operation_names_have_distinct_position_series(config):
    config = with_stages(
        config,
        [
            {"name": "noise", "position_std": 0, "seed": 1},
            {"name": "noise", "position_std": 0, "seed": 2},
        ],
    )
    result = benchmark_experiment(
        config, specification=BenchmarkSpec(warmup=0, repetitions=1), _clock=Clock()
    )
    assert {s.stage for s in result.frames[0].series if s.stage.startswith("operation/")} == {
        "operation/1:noise",
        "operation/2:noise",
    }


def test_failed_operation_retains_completed_reps_without_partial_statistics(config, monkeypatch):
    original = benchmark.run_preprocessing_pipeline
    calls = 0

    def fail(cloud, stages, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("intentional stage failure")
        return original(cloud, stages, **kwargs)

    monkeypatch.setattr(benchmark, "run_preprocessing_pipeline", fail)
    result = benchmark_experiment(
        config, specification=BenchmarkSpec(warmup=0, repetitions=3), _clock=Clock()
    )
    assert result.status == "failed"
    assert len(result.frames[0].observations) == 2
    assert result.frames[0].observations[-1]["status"] == "failed"
    assert not result.frames[0].series
    assert "intentional stage failure" in result.failures[0]["message"]
    assert (
        json.loads((Path(result.output_directory) / "manifest.json").read_text())["status"]
        == "failed"
    )


def test_source_mutation_is_detected_and_failed(config, monkeypatch):
    original = benchmark.run_preprocessing_pipeline

    def bad(cloud, stages, **kwargs):
        cloud.points.setflags(write=True)
        cloud.points[0, 0] = 999
        return original(cloud, stages, **kwargs)

    monkeypatch.setattr(benchmark, "run_preprocessing_pipeline", bad)
    result = benchmark_experiment(
        config, specification=BenchmarkSpec(warmup=0, repetitions=1), _clock=Clock()
    )
    assert result.status == "failed" and "mutation" in result.failures[0]["message"]


class FakeGPU:
    def __init__(self, config):
        self.initializations = 0
        self.calls = 0
        self.syncs = 0
        self.closed = False

    def initialize(self):
        self.initializations += 1

    def synchronize(self):
        self.syncs += 1

    def detect(self, cloud):
        self.calls += 1
        assert self.initializations == 1
        return [Detection3D((1, 0, 0), (4, 2, 1), 0, "Car", 0.8, "kitti_velodyne")]

    @property
    def provenance(self):
        return {"synthetic_test": True, "gpu": "fake boundary, not real hardware"}

    def close(self):
        self.closed = True


def test_synthetic_gpu_initialization_warmup_repetitions_sync_and_outputs(config):
    backend = FakeGPU(config)
    result = benchmark_experiment(
        config,
        specification=BenchmarkSpec(mode="inference", warmup=2, repetitions=3),
        _clock=Clock(),
        _backend_factory=lambda config: backend,
    )
    assert result.status == "success" and result.evidence_kind == "synthetic_test"
    assert result.initialization.summary["count"] == 1
    assert backend.initializations == 1 and backend.calls == 5 and backend.closed
    assert (
        backend.syncs == 2 + 5 * 3
    )  # two initialization barriers; one E2E + two adapter barriers per iteration.
    series = next(s for s in result.frames[0].series if s.stage == "inference_adapter")
    assert series.raw_ns == (100, 100, 100)
    assert all(r["predictions"]["total_predictions"] == 1 for r in result.frames[0].observations)
    assert result.model_provenance["synthetic_test"]


@pytest.mark.parametrize("phase", ["initialize", "detect"])
def test_gpu_failures_never_become_success_and_cleanup_runs(config, phase):
    backend = FakeGPU(config)
    setattr(backend, phase, Mock(side_effect=RuntimeError("GPU failure")))
    result = benchmark_experiment(
        config,
        specification=BenchmarkSpec(mode="inference", warmup=0, repetitions=2),
        _clock=Clock(),
        _backend_factory=lambda config: backend,
    )
    assert result.status == "failed" and backend.closed
    if phase == "initialize":
        assert result.initialization is None and not result.frames
    else:
        assert not result.frames[0].series


def test_mutating_detector_is_rejected(config):
    backend = FakeGPU(config)

    def mutate(cloud):
        cloud.points.setflags(write=True)
        cloud.points[0, 0] += 1
        return []

    backend.detect = mutate
    result = benchmark_experiment(
        config,
        specification=BenchmarkSpec(mode="inference", warmup=0, repetitions=1),
        _clock=Clock(),
        _backend_factory=lambda config: backend,
    )
    assert result.status == "failed" and "detector mutated" in result.failures[0]["message"]


def test_saved_predictions_are_never_reported_as_inference(config):
    result = benchmark_experiment(
        config, specification=BenchmarkSpec(mode="inference", warmup=0, repetitions=1)
    )
    assert result.status == "blocked" and "saved predictions" in result.failures[0]["message"]
    assert (
        result.initialization is None and not result.frames and not config.output.directory.exists()
    )


def test_mac_gpu_mode_blocks_before_importing_torch(config, monkeypatch):
    monkeypatch.setattr("raven.benchmark.gpu.platform.system", lambda: "Darwin")
    config = with_stages(config, [])
    result = benchmark_experiment(
        config, specification=BenchmarkSpec(mode="inference", warmup=0, repetitions=1)
    )
    assert result.status == "blocked" and "macOS" in result.failures[0]["message"]
    assert not result.frames and result.initialization is None


def test_model_initialization_rechecks_config_checksum(config, monkeypatch):
    from raven.experiments import runner

    backend = LocalCudaBackend.__new__(LocalCudaBackend)
    backend.config = with_stages(config, [])
    backend.evidence = {"config_sha256": "a" * 64}
    monkeypatch.setattr("raven.experiments.artifacts.file_sha256", lambda path: "b" * 64)
    factory = Mock()
    monkeypatch.setattr(runner, "_build_detector", factory)
    with pytest.raises(ValueError, match="changed"):
        backend.initialize()
    factory.assert_not_called()


def test_collision_preserves_existing_benchmark_and_experiment_artifacts(config):
    config.run_directory.mkdir(parents=True)
    sentinel = config.run_directory / "keep.json"
    sentinel.write_text("untouched")
    first = benchmark_experiment(
        config, specification=BenchmarkSpec(warmup=0, repetitions=1), _clock=Clock()
    )
    before = (Path(first.output_directory) / "benchmark.json").read_bytes()
    second = benchmark_experiment(
        config, specification=BenchmarkSpec(warmup=0, repetitions=1), _clock=Clock()
    )
    assert first.status == "success" and second.status == "invalid"
    assert sentinel.read_text() == "untouched"
    assert (Path(first.output_directory) / "benchmark.json").read_bytes() == before
    assert (
        benchmark_experiment(
            config, specification=BenchmarkSpec(warmup=0, repetitions=1), benchmark_id="../bad"
        ).status
        == "invalid"
    )


def test_split_missing_scan_and_changed_source_preflight(config):
    doc = config.to_dict()
    doc["frames"]["ids"] = ["000000"]
    other = ExperimentConfig.from_dict(doc, base_dir=Path.cwd())
    assert benchmark_experiment(other).status == "invalid"  # train member declared val
    path = config.dataset.scan_directory / "000001.bin"
    path.unlink()
    assert benchmark_experiment(config).status == "invalid"
    assert not config.output.directory.exists()


def test_observer_rejects_incomplete_event_order():
    observer = StageObserver(Timer(Clock()))
    with pytest.raises(ValueError):
        observer("finish", 1, "noise", 3)
    observer("start", 1, "noise", 3)
    with pytest.raises(ValueError):
        observer("start", 2, "noise", 3)
    with pytest.raises(ValueError):
        observer("finish", 2, "noise", 3)
    observer("finish", 1, "noise", 3)
    assert observer.records[0]["duration_ns"] == 100


def test_cli_success_default_spec_override_and_blocked_exit(config, tmp_path, capsys):
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(config.to_dict()))
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(BenchmarkSpec(warmup=0, repetitions=2).to_dict()))
    main(["benchmark-experiment", str(path), "--spec", str(spec), "--benchmark-id", "cli-test"])
    value = json.loads(capsys.readouterr().out)
    assert value["status"] == "success" and value["evidence_kind"] == "local_cpu"
    spec.write_text(json.dumps(BenchmarkSpec(mode="inference", warmup=0, repetitions=1).to_dict()))
    with pytest.raises(SystemExit) as error:
        main(["benchmark-experiment", str(path), "--spec", str(spec), "--benchmark-id", "gpu-test"])
    assert error.value.code == 3
    assert json.loads(capsys.readouterr().out)["status"] == "blocked"
    with pytest.raises(SystemExit) as error:
        main(["benchmark-experiment", str(tmp_path / "missing.json")])
    assert error.value.code == 2


def test_cpu_execution_has_no_gpu_modal_gui_imports(config, tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config.to_dict()))
    script = """
import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self,name,*args):
        if name.split('.')[0] in {'torch','spconv','cumm','pcdet','modal','open3d'}:
            raise AssertionError('forbidden import '+name)
sys.meta_path.insert(0,Block())
from raven.benchmark.experiment import benchmark_experiment
from raven.benchmark.spec import BenchmarkSpec
r=benchmark_experiment(sys.argv[1],specification=BenchmarkSpec(warmup=0,repetitions=1))
assert r.status=='success',r.failures
"""
    env = dict(
        os.environ,
        PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"),
        PYTHONDONTWRITEBYTECODE="1",
    )
    subprocess.run(
        [sys.executable, "-c", script, str(path)], check=True, env=env, capture_output=True
    )


def test_invalid_detector_output_fails_instead_of_masquerading_as_no_inference(config):
    backend = FakeGPU(config)
    backend.detect = lambda cloud: None
    result = benchmark_experiment(
        config,
        specification=BenchmarkSpec(mode="inference", warmup=0, repetitions=1),
        _backend_factory=lambda config: backend,
    )
    assert result.status == "failed"
    assert not result.frames[0].series


def test_missing_cuda_on_non_mac_is_explicitly_blocked(config, monkeypatch):
    monkeypatch.setattr("raven.benchmark.gpu.platform.system", lambda: "Linux")
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    )
    result = benchmark_experiment(
        with_stages(config, []),
        specification=BenchmarkSpec(mode="inference", warmup=0, repetitions=1),
    )
    assert result.status == "blocked" and result.evidence_kind == "unavailable", result.failures
    assert "CUDA unavailable" in result.failures[0]["message"]
    assert result.environment["cuda_used"] is False


def test_missing_scipy_backend_and_empty_filtered_outputs(config, monkeypatch):
    sor = with_stages(
        config, [{"name": "statistical_outlier_removal", "nb_neighbors": 1, "std_ratio": 1}]
    )
    monkeypatch.setitem(sys.modules, "scipy.spatial", None)
    failed = benchmark_experiment(
        sor, specification=BenchmarkSpec(warmup=0, repetitions=1), _clock=Clock()
    )
    assert failed.status == "failed" and "SciPy" in failed.failures[0]["message"]
    assert not failed.frames[0].series
    empty = with_stages(
        config,
        [{"name": "range_filter", "x_range": [-20, -10], "y_range": [-1, 1], "z_range": [-1, 1]}],
    )
    result = benchmark_experiment(
        empty,
        specification=BenchmarkSpec(warmup=0, repetitions=1),
        benchmark_id="empty",
        _clock=Clock(),
    )
    assert result.status == "success" and result.frames[0].observations[0]["output_count"] == 0
    assert (
        result.frames[0].observations[0]["processed_xyzi_sha256"] == hashlib.sha256(b"").hexdigest()
    )


def test_output_symlink_rejected_without_touching_target(config, tmp_path):
    config.output.directory.mkdir()
    target = tmp_path / "target"
    target.mkdir()
    (config.output.directory / f"{config.experiment_id}.benchmark").symlink_to(
        target, target_is_directory=True
    )
    result = benchmark_experiment(config, specification=BenchmarkSpec(warmup=0, repetitions=1))
    assert result.status == "invalid" and not list(target.iterdir())


def test_persistence_failure_cannot_report_success(config, monkeypatch):
    original = benchmark.write_json_exclusive

    def write(path, doc):
        if path.name == "manifest.json":
            raise OSError("intentional publication failure")
        original(path, doc)

    monkeypatch.setattr(benchmark, "write_json_exclusive", write)
    result = benchmark_experiment(
        config, specification=BenchmarkSpec(warmup=0, repetitions=1), _clock=Clock()
    )
    assert result.status == "failed" and result.failures[-1]["phase"] == "persistence"
    assert not (Path(result.output_directory) / "manifest.json").exists()


def test_frame_order_and_non_equivalent_inputs_abort_before_later_frames(config, monkeypatch):
    cfg = replace(config, frames=replace(config.frames, ids=("000002", "000001")))
    good = benchmark_experiment(
        cfg,
        specification=BenchmarkSpec(warmup=0, repetitions=1),
        benchmark_id="order",
        _clock=Clock(),
    )
    assert [f.frame_id for f in good.frames] == ["000002", "000001"]
    original = benchmark.run_preprocessing_pipeline
    calls = 0

    def non_equivalent(cloud, stages, **kwargs):
        nonlocal calls
        calls += 1
        result = original(cloud, stages, **kwargs)
        if calls == 2:
            result.cloud.points.setflags(write=True)
            result.cloud.points[0, 0] = 99
        return result

    monkeypatch.setattr(benchmark, "run_preprocessing_pipeline", non_equivalent)
    failed = benchmark_experiment(
        cfg,
        specification=BenchmarkSpec(warmup=0, repetitions=3),
        benchmark_id="non-equivalent",
        _clock=Clock(),
    )
    assert failed.status == "failed" and len(failed.frames) == 1
    assert "non-equivalent" in failed.failures[0]["message"]
    assert len(failed.frames[0].observations) == 2 and not failed.frames[0].series


def test_real_cpu_environment_source_hashes_and_no_test_clock_claim(config):
    result = benchmark_experiment(config, specification=BenchmarkSpec(warmup=0, repetitions=1))
    assert result.status == "success" and result.evidence_kind == "local_cpu"
    env = result.environment
    assert env["logical_cpu_count"] >= 1 and env["machine"]
    assert env["effective_clock"] == "time.perf_counter_ns" and env["cuda_used"] is False
    assert env["benchmark_source_sha256"]
    source = Path(__file__).resolve().parents[1] / "src/raven"
    for name, sha in env["benchmark_source_sha256"].items():
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == sha


def test_optional_real_kitti_cpu_smoke(tmp_path):
    # Explicit opt-in: no dataset downloads or machine-specific standard CI assumptions.
    root = os.environ.get("RAVEN_KITTI_BENCHMARK_ROOT")
    if not root:
        pytest.skip("optional local KITTI benchmark root not configured")
    config = ExperimentConfig.from_dict(
        {
            "schema_version": 1,
            "name": "KITTI bounded CPU smoke",
            "experiment_id": "kitti-smoke",
            "dataset": {"root": root, "split": "train"},
            "frames": {"ids": ["000000"]},
            "model": {
                "name": "pointpillars",
                "source": "saved",
                "predictions_dir": str(tmp_path / "unused"),
            },
            "preprocessing": [],
            "seed": 42,
            "output": {"directory": str(tmp_path / "results"), "artifact_prefix": "cpu"},
        },
        base_dir=tmp_path,
    )
    result = benchmark_experiment(config, specification=BenchmarkSpec(warmup=1, repetitions=2))
    assert result.status == "success", result.failures
    frame = result.frames[0]
    assert frame.source["input_count"] == 115384
    assert (
        frame.source["sha256"] == "0e09c85e3f6078ecbdd1e706ee9624519f1bd29417437167a9ed7fbe6f54b4b1"
    )
    assert all(s.summary["count"] == 2 for s in frame.series)
