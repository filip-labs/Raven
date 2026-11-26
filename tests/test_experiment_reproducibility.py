"""Cross-stage M5 regressions. Synthetic detector/clock tests are never GPU evidence."""

import csv
import hashlib
import json
import os
import subprocess
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from test_benchmark import Clock
from test_kitti_evaluation import completed as completed  # noqa: PLC0414 -- Imported fixture.

from raven.benchmark.experiment import benchmark_experiment
from raven.benchmark.spec import BenchmarkSpec
from raven.cli import main
from raven.evaluation.evaluate import evaluate_experiment, save_evaluation_result
from raven.experiments import runner
from raven.experiments.config import ExperimentConfig, load_experiment_config
from raven.exporting.evidence import Snapshot, load_evidence
from raven.exporting.export import export_results
from raven.perception.detection.predictions import detections_to_document
from raven.reproducibility.policy import projection
from raven.reproducibility.verify import verify_results

STAGES = {
    "baseline": [],
    "range": [
        {"name": "range_filter", "x_range": [0, 70], "y_range": [-40, 40], "z_range": [-3, 1]}
    ],
    "voxel": [{"name": "voxel_downsample", "voxel_size": 0.2}],
    "statistical": [{"name": "statistical_outlier_removal", "nb_neighbors": 2, "std_ratio": 1}],
    "noise": [{"name": "noise", "position_std": 0.01, "seed": 42}],
}
STAGES["full"] = STAGES["range"] + STAGES["voxel"] + STAGES["statistical"] + STAGES["noise"]


@pytest.fixture
def experiment(tmp_path):
    # Every file is synthetic and exclusively created under pytest's own tmp_path.
    root = tmp_path / "synthetic-dataset"
    predictions = tmp_path / "synthetic-predictions"
    predictions.mkdir()
    points = np.array(
        [
            [20, 0, 0, 0.1],
            [20.01, 0, 0, 0.2],
            [-2, 3, 0, 0.3],
            [40, 5, 0, 0.4],
            [65, 2, 0, 0.5],
            [120, 0, 0, 0.6],
            [20, 0, 0, 0.7],
            [21, 0, 1, 0.8],
        ],
        dtype="<f4",
    )
    for folder in ("velodyne", "label_2", "calib"):
        (root / "training" / folder).mkdir(parents=True)
    for frame in ("000001", "000002"):
        scan = root / "training/velodyne" / f"{frame}.bin"
        with scan.open("xb") as stream:
            stream.write(points.tobytes())
        (root / "training/label_2" / f"{frame}.txt").write_text(
            "Car 0 0 0 0 0 100 100 2 2 4 0 1 20 0\n"
        )
        (root / "training/calib" / f"{frame}.txt").write_text(
            "R0_rect: 1 0 0 0 1 0 0 0 1\nTr_velo_to_cam: 0 -1 0 0 0 0 -1 0 1 0 0 0\n"
        )
        document = detections_to_document([], frame_id=frame)
        document["input"].update(sha256=hashlib.sha256(points.tobytes()).hexdigest(), points=8)
        (predictions / f"{frame}.json").write_text(json.dumps(document))
    model = tmp_path / "declared-only.yaml"
    model.write_text("synthetic model boundary, not executable weights")
    config = ExperimentConfig.from_dict(
        {
            "schema_version": 1,
            "name": "Controlled reproducibility fixture",
            "experiment_id": "controlled",
            "dataset": {"root": str(root), "split": "val"},
            "frames": {"ids": ["000002", "000001"]},
            "model": {
                "name": "pointpillars",
                "source": "saved",
                "predictions_dir": str(predictions),
            },
            "preprocessing": [],
            "seed": 42,
            "output": {"directory": str(tmp_path / "initial-output"), "artifact_prefix": "result"},
        },
        base_dir=tmp_path,
    )
    return config, model


def output(config, tmp_path, name):
    return replace(config, output=replace(config.output, directory=(tmp_path / name).resolve()))


def run(config):
    result = runner.run_experiment(config)
    assert result.status == "success", result.failures
    return config.run_directory / f"{config.output.artifact_prefix}.manifest.json"


def synthetic_live(config, model, stages, monkeypatch):
    document = config.to_dict()
    document["model"] = {
        "name": "pointpillars",
        "source": "inference",
        "config": str(model),
        "checkpoint": str(model.parent / "never-loaded.pth"),
        "checkpoint_sha256": "a" * 64,
        "openpcdet_revision": "233f849829b6ac19afb8af8837a0246890908755",
    }
    document["preprocessing"] = stages
    config = ExperimentConfig.from_dict(document, base_dir=model.parent)
    monkeypatch.setattr(
        runner,
        "_inference_preflight",
        lambda cfg: {
            "config_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
            "synthetic_test": True,
        },
    )

    class SyntheticDetector:
        def initialize(self):
            pass

        def detect(self, cloud):
            return []

        def close(self):
            pass

        @property
        def provenance(self):
            return {"synthetic_test": True, "not_real_gpu_inference": True}

    monkeypatch.setattr(runner, "_build_detector", lambda *args: SyntheticDetector())
    return config


@pytest.mark.parametrize("name", list(STAGES))
def test_independent_runner_replays_preserve_seeded_pipeline_and_input_chain(
    experiment, tmp_path, monkeypatch, name
):
    config, model = experiment
    config = synthetic_live(config, model, STAGES[name], monkeypatch)
    before = {p: p.read_bytes() for p in config.dataset.root.rglob("*") if p.is_file()}
    first = run(output(config, tmp_path, "first"))
    second = run(output(config, tmp_path, "second"))
    report = verify_results(first, compare=second, replay_preprocessing=True)
    assert report.status == "checks_passed", report.failures
    assert report.checks["deterministic_comparison"]["status"] == "matched"
    assert report.checks["preprocessing_replay"]["status"] == "verified"
    assert report.checks["actual_model_inference_reproduction"]["status"] == "not_performed"
    assert report.checks["cross_environment_reproducibility"]["status"] == "not_established"
    assert (
        report.environment_differences
    )  # Output paths/config hashes/time metadata remain visible.
    for path, data in before.items():
        assert path.read_bytes() == data
    original = load_evidence(first, Snapshot())
    other = load_evidence(second, Snapshot())
    assert (
        original.config_sha256 != other.config_sha256
    )  # Canonical config includes output directory.
    assert projection(original)[0] == projection(other)[0]
    assert [r["frame_id"] for r in original.payload["results"]["frames"]] == ["000002", "000001"]
    for row in original.payload["results"]["frames"]:
        assert (
            row["inputs"]["scan"]["sha256"]
            == hashlib.sha256(before[Path(row["inputs"]["scan"]["path"])]).hexdigest()
        )


def test_saved_runs_preserve_capture_identity_without_claiming_new_gpu(experiment, tmp_path):
    config, _ = experiment
    first = run(output(config, tmp_path, "a"))
    second = run(output(config, tmp_path, "b"))
    report = verify_results(first, compare=second, check_inputs=True)
    assert report.status == "checks_passed", report.failures
    assert report.sources[0]["evidence_kind"] == "saved_predictions"
    assert report.checks["preprocessing_replay"]["status"] == "not_requested"
    assert report.checks["actual_model_inference_reproduction"]["status"] == "not_performed"
    assert report.checks["deterministic_comparison"]["status"] == "matched"
    doc = load_evidence(first, Snapshot()).payload
    assert all(
        r["prediction_source"]["model_provenance"] == "declared_not_independently_verified"
        for r in doc["results"]["frames"]
    )


def test_seed_change_is_scientific_difference_even_when_metadata_is_similar(
    experiment, tmp_path, monkeypatch
):
    config, model = experiment
    first_config = synthetic_live(config, model, STAGES["noise"], monkeypatch)
    second_config = synthetic_live(
        config, model, [{"name": "noise", "position_std": 0.01, "seed": 43}], monkeypatch
    )
    first = run(output(first_config, tmp_path, "a"))
    second = run(output(second_config, tmp_path, "b"))
    result = verify_results(first, compare=second)
    assert (
        result.status == "failed"
        and result.checks["deterministic_comparison"]["status"] == "mismatch"
    )
    assert any(
        "processed_xyzi_sha256" in row["path"]
        for row in result.checks["deterministic_comparison"]["differences"]
    )


@pytest.mark.parametrize("seed", [None, True, -1, 1.5, "42"])
def test_invalid_noise_seed_cannot_start_a_controlled_run(experiment, monkeypatch, seed):
    config, model = experiment
    doc = config.to_dict()
    doc["model"] = {
        "name": "pointpillars",
        "source": "inference",
        "config": str(model),
        "checkpoint": str(model.parent / "absent.pth"),
        "checkpoint_sha256": "a" * 64,
        "openpcdet_revision": "b" * 40,
    }
    doc["preprocessing"] = [{"name": "noise", "position_std": 0.01, "seed": seed}]
    with pytest.raises((ValueError, TypeError)):
        ExperimentConfig.from_dict(doc, base_dir=model.parent)
    assert not config.output.directory.exists()


@pytest.mark.parametrize("name", list(STAGES))
def test_fresh_processes_match_processed_bytes_without_comparing_timers(experiment, tmp_path, name):
    config, model = experiment
    doc = config.to_dict()
    doc["model"] = {
        "name": "pointpillars",
        "source": "inference",
        "config": str(model),
        "checkpoint": str(model.parent / "absent.pth"),
        "checkpoint_sha256": "a" * 64,
        "openpcdet_revision": "b" * 40,
    }
    doc["preprocessing"] = STAGES[name]
    path = tmp_path / "config.json"
    path.write_text(json.dumps(doc))
    script = """
import json,sys,hashlib,numpy as np
from raven.experiments.config import load_experiment_config
from raven.perception.data.kitti_loader import load_kitti_bin
from raven.perception.preprocessing.pipeline import run_preprocessing_pipeline
from raven.experiments.splits import split_membership
c=load_experiment_config(sys.argv[1]);p=c.dataset.scan_directory/'000001.bin'
x=load_kitti_bin(p);before=x.points.tobytes();np.random.seed(777);state=np.random.get_state()
y=run_preprocessing_pipeline(x,c.preprocessing).cloud
assert x.points.tobytes()==before
assert np.array_equal(np.random.get_state()[1],state[1]) and np.random.get_state()[2:]==state[2:]
print(json.dumps({'configuration':c.to_dict(),'config_sha256':c.config_sha256,'frames':c.frame_ids,
    'selection':c.selection_provenance,'split':split_membership(c.dataset.split)[1],
    'input':hashlib.sha256(before).hexdigest(),'processed':hashlib.sha256(y.points.astype('<f4').tobytes()).hexdigest(),
    'frame_id':y.frame_id,'coordinate_frame':y.coordinate_frame,'count':y.count},sort_keys=True))
"""
    env = dict(
        os.environ,
        PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"),
        PYTHONDONTWRITEBYTECODE="1",
    )
    calls = [
        subprocess.run(
            [sys.executable, "-c", script, str(path)], env=env, check=True, capture_output=True
        ).stdout
        for _ in range(2)
    ]
    assert calls[0] == calls[1]
    result = json.loads(calls[0])
    assert result["frame_id"] == "000001" and result["coordinate_frame"] == "kitti_velodyne"


def test_repeated_config_loads_manifest_selection_and_hash_partition(experiment, tmp_path):
    config, _ = experiment
    manifest = tmp_path / "selection.txt"
    manifest.write_text("000002\n000001\n")
    doc = config.to_dict()
    doc["frames"] = {"manifest": str(manifest), "sample_size": 1}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(doc))
    left, right = load_experiment_config(path), load_experiment_config(path)
    assert left.to_dict() == right.to_dict() and left.config_sha256 == right.config_sha256
    assert (
        left.frame_ids == right.frame_ids
        and left.selection_provenance == right.selection_provenance
    )
    raw = manifest.read_bytes()
    manifest.write_bytes(b"000001\n000002\n")
    changed = load_experiment_config(path)
    assert changed.frames.sha256 != hashlib.sha256(raw).hexdigest()
    assert changed.config_sha256 != left.config_sha256


def test_repeated_evaluation_matches_identity_nulls_diagnostics_and_declared_tolerance(
    completed, tmp_path
):
    path, _ = completed
    outputs = []
    for name in ("a", "b"):
        value = evaluate_experiment(path)
        assert value.status == "success"
        target = tmp_path / f"{name}.json"
        save_evaluation_result(target, value)
        outputs.append(target)
    report = verify_results(outputs[0], compare=outputs[1])
    assert report.status == "checks_passed", report.failures
    assert report.checks["numerical_evaluation_comparison"]["status"] == "matched"
    assert report.checks["numerical_evaluation_comparison"]["matching_null_metrics"] > 0
    assert report.tolerance_policy["ap_r40_absolute_tolerance_percent"] == 1e-8
    first, second = [json.loads(p.read_text()) for p in outputs]
    assert first["metrics"] == second["metrics"] and first["diagnostics"] == second["diagnostics"]


@pytest.mark.parametrize(
    "delta,expected", [(0.0, "checks_passed"), (5e-9, "checks_passed"), (2e-8, "failed")]
)
def test_numerical_ap_tolerance_independent_changes(completed, tmp_path, delta, expected):
    run_path, _ = completed
    left = evaluate_experiment(run_path).to_dict()
    right = deepcopy(left)
    right["metrics"][0]["ap_r40_percent"] += delta
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(left))
    b.write_text(json.dumps(right))
    report = verify_results(a, compare=b)
    assert report.status == expected, report.failures
    assert report.checks["deterministic_comparison"]["status"] == "matched"


def test_metric_row_order_is_compared_by_identity_without_rewriting_source(completed, tmp_path):
    run_path, _ = completed
    value = evaluate_experiment(run_path).to_dict()
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(value))
    value["metrics"] = list(reversed(value["metrics"]))
    value["diagnostics"] = list(reversed(value["diagnostics"]))
    b.write_text(json.dumps(value))
    before = b.read_bytes()
    report = verify_results(a, compare=b)
    assert report.status == "checks_passed", report.failures
    assert b.read_bytes() == before


def test_blocked_evaluation_has_no_numerical_reproduction_claim(completed, tmp_path):
    path, config = completed
    image = config.dataset.root / "training/image_2/000001.png"
    assert tmp_path in image.parents
    image.rename(tmp_path / "synthetic-image-not-at-original-reference.png")
    result = evaluate_experiment(path)
    assert result.status == "blocked" and not result.metrics
    source = tmp_path / "blocked.json"
    save_evaluation_result(source, result)
    report = verify_results(source, compare=source)
    assert report.status == "unavailable"
    assert report.checks["numerical_evaluation_comparison"]["status"] == "not_requested"
    assert report.checks["actual_model_inference_reproduction"]["status"] == "not_performed"


def test_fake_clock_benchmark_math_repeated_sources_and_real_timing_is_not_bitwise_contract(
    experiment, tmp_path
):
    config, _ = experiment
    paths = []
    for name in ("a", "b"):
        cfg = output(config, tmp_path, name)
        value = benchmark_experiment(
            cfg, specification=BenchmarkSpec(warmup=2, repetitions=3), _clock=Clock()
        )
        assert value.status == "success"
        paths.append(Path(value.output_directory) / "manifest.json")
        for series in value.frames[0].series:
            assert series.summary["count"] == 3
            assert series.summary["mean_ms"] == sum(series.raw_ns) / 3 / 1_000_000
            assert (
                series.summary["std_ddof"] == 0 and series.summary["percentile_method"] == "linear"
            )
    report = verify_results(paths[0], compare=paths[1], replay_preprocessing=True)
    assert report.status == "checks_passed", report.failures
    for source in report.sources:
        assert source["evidence_kind"] == "synthetic_test"
    observed = []
    for name in ("real-a", "real-b"):
        value = benchmark_experiment(
            output(config, tmp_path, name), specification=BenchmarkSpec(warmup=0, repetitions=2)
        )
        observed.append(Path(value.output_directory) / "manifest.json")
    actual = verify_results(observed[0], compare=observed[1])
    assert actual.status == "checks_passed", actual.failures
    # Never require identical real latency, timestamps, source artifact bytes or summary values.
    assert actual.checks["cross_environment_reproducibility"]["status"] == "not_established"


def test_independent_exports_match_scientific_fields_but_not_raw_content_hash(experiment, tmp_path):
    config, _ = experiment
    manifests = [run(output(config, tmp_path, name)) for name in ("run-a", "run-b")]
    bundles = []
    for i, path in enumerate(manifests):
        dest = tmp_path / f"export-{i}"
        export_results([path], dest, format="all")
        bundles.append(dest / "manifest.json")
    report = verify_results(bundles[0], compare=bundles[1])
    assert report.status == "checks_passed", report.failures
    assert report.checks["deterministic_comparison"]["exact_export_content_sha256_equal"] is False
    assert report.environment_differences


def test_same_source_export_content_and_csv_bytes_are_stable_with_generation_separate(
    experiment, tmp_path
):
    config, _ = experiment
    path = run(output(config, tmp_path, "run"))
    destinations = [tmp_path / "a", tmp_path / "b"]
    for dest in destinations:
        export_results([path], dest, format="all")
    a, b = [json.loads((d / "results.json").read_text()) for d in destinations]
    assert a["content_sha256"] == b["content_sha256"] and a["content"] == b["content"]
    for name in ("experiment_summary.csv", "evaluation_metrics.csv", "benchmark_summaries.csv"):
        assert (destinations[0] / name).read_bytes() == (destinations[1] / name).read_bytes()
        with (destinations[0] / name).open(newline="") as stream:
            assert next(csv.reader(stream))[0] == "export_schema_version"
    report = verify_results(
        destinations[0] / "manifest.json", compare=destinations[1] / "manifest.json"
    )
    assert report.checks["deterministic_comparison"]["exact_export_content_sha256_equal"] is True


def test_noise_large_seed_is_supported_but_experiment_selection_seed_is_bounded(
    experiment, tmp_path
):
    from raven.perception.data.kitti_loader import load_kitti_bin
    from raven.perception.preprocessing.noise import inject_noise

    config, _ = experiment
    cloud = load_kitti_bin(config.dataset.scan_directory / "000001.bin")
    a, b = [inject_noise(cloud, 0.01, 2**80 + 42) for _ in range(2)]
    assert a.points.tobytes() == b.points.tobytes()
    doc = config.to_dict()
    doc["seed"] = 2**64
    with pytest.raises(ValueError):
        ExperimentConfig.from_dict(doc, base_dir=tmp_path)


def test_offline_bundle_inspection_survives_unreadable_originals_without_claiming_replay(
    experiment, tmp_path, monkeypatch
):
    config, _ = experiment
    source = run(output(config, tmp_path, "run"))
    destination = tmp_path / "archive"
    export_results([source], destination, format="all")
    original_paths = {Path(a["path"]) for a in load_evidence(source, Snapshot()).source_artifacts}
    read = Path.read_bytes

    def unavailable(path):
        if path in original_paths:
            raise FileNotFoundError("simulated archived original unavailable")
        return read(path)

    monkeypatch.setattr(Path, "read_bytes", unavailable)
    offline = verify_results(destination / "manifest.json")
    assert offline.status == "integrity_only", offline.failures
    assert offline.checks["artifact_integrity"]["status"] == "verified"
    assert offline.sources[0]["input_identities"]
    assert offline.checks["preprocessing_replay"]["status"] == "not_requested"
    linked = verify_results(destination / "manifest.json", check_sources=True)
    assert linked.status == "unavailable"
    assert linked.checks["original_source_references"]["status"] == "unavailable"
    assert linked.checks["actual_model_inference_reproduction"]["status"] == "not_performed"


def test_external_selection_manifest_retention_is_explicit(experiment, tmp_path, monkeypatch):
    config, _ = experiment
    selection = tmp_path / "selection.txt"
    selection.write_text("000002\n000001\n")
    cfg = replace(config, frames=type(config.frames)(manifest=selection.resolve()))
    source = run(output(cfg, tmp_path, "run"))
    destination = tmp_path / "archive"
    export_results([source], destination)
    original = Path.read_bytes

    def missing(path):
        if path == selection:
            raise FileNotFoundError("selection manifest unavailable")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", missing)
    assert verify_results(destination / "manifest.json").status == "integrity_only"
    result = verify_results(destination / "manifest.json", check_sources=True)
    assert result.status == "unavailable"
    assert "selection" in result.checks["original_source_references"]["reason"]


def test_environment_and_unknown_fields_are_not_silently_erased(experiment, tmp_path):
    from raven.exporting.evidence import Evidence

    config, _ = experiment
    source = run(output(config, tmp_path, "run"))
    original = load_evidence(source, Snapshot())
    changed = deepcopy(original.to_dict())
    changed["environment"]["machine"] = "different-architecture"
    changed["payload"]["results"]["frames"][0]["new_scientific_field"] = "new evidence"
    other = Evidence(**changed)
    science_a, env_a, _ = projection(original)
    science_b, env_b, _ = projection(other)
    assert env_a != env_b and science_a != science_b
    assert science_b["frames"][0]["new_scientific_field"] == "new evidence"


@pytest.fixture
def chain(completed, tmp_path):
    path, config = completed
    evaluation = tmp_path / "evaluation.json"
    save_evaluation_result(evaluation, evaluate_experiment(path))
    bench = benchmark_experiment(
        config, specification=BenchmarkSpec(warmup=1, repetitions=2), _clock=Clock()
    )
    benchmark = Path(bench.output_directory) / "manifest.json"
    archive = tmp_path / "archive"
    export_results([path, evaluation, benchmark], archive, format="all", raw_timings=True)
    return {
        "run": path,
        "evaluation": evaluation,
        "benchmark": benchmark,
        "archive": archive,
        "config": config,
    }


def corrupt_artifact(manifest_path, role, mutation):
    """Adversarial test only: a fresh checksum must not conceal semantic corruption."""
    manifest = json.loads(manifest_path.read_text())
    target = manifest_path.parent / manifest["artifacts"][role]["file"]
    value = json.loads(target.read_text())
    mutation(value)
    target.write_text(json.dumps(value))
    manifest["artifacts"][role]["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    if "bytes" in manifest["artifacts"][role]:
        manifest["artifacts"][role]["bytes"] = target.stat().st_size
    manifest_path.write_text(json.dumps(manifest))


@pytest.mark.parametrize(
    "problem",
    [
        "configuration",
        "raw_input",
        "selection",
        "missing",
        "truncated",
        "checksum",
        "schema",
        "provenance",
        "benchmark_samples",
        "statistics",
        "symlink",
        "export_content",
        "export_csv",
    ],
)
def test_complete_integrity_chain_detects_corruption_without_regenerating_scientific_evidence(
    chain, tmp_path, problem
):
    source = chain["run"]
    manifest = json.loads(source.read_text())
    if problem == "configuration":
        target = source.parent / manifest["artifacts"]["configuration"]["file"]
        value = json.loads(target.read_text())
        value["seed"] = 43
        target.write_text(json.dumps(value))
    elif problem == "raw_input":
        scan = chain["config"].dataset.scan_directory / "000001.bin"
        assert tmp_path in scan.parents  # Only the synthetic pytest dataset can be modified.
        raw = bytearray(scan.read_bytes())
        raw[0] ^= 1
        scan.write_bytes(raw)
    elif problem == "selection":
        target = source.parent / manifest["artifacts"]["frames"]["file"]
        target.write_text("000002\n")
    elif problem == "missing":
        target = source.parent / manifest["artifacts"]["results"]["file"]
        assert tmp_path in target.parents
        target.unlink()
    elif problem == "truncated":
        target = source.parent / manifest["artifacts"]["results"]["file"]
        target.write_bytes(target.read_bytes()[:20])
    elif problem == "checksum":
        manifest["artifacts"]["results"]["sha256"] = "0" * 64
        source.write_text(json.dumps(manifest))
    elif problem == "schema":
        manifest["run_schema_version"] = 99
        source.write_text(json.dumps(manifest))
    elif problem == "provenance":
        corrupt_artifact(source, "provenance", lambda d: d["configuration"].update(seed=43))
    elif problem in ("benchmark_samples", "statistics"):
        source = chain["benchmark"]

        def change(v):
            if problem == "benchmark_samples":
                v["frames"][0]["observations"][-1]["durations_ns"]["preprocessing"] += 100
            else:
                v["frames"][0]["series"][0]["summary"]["mean_ms"] += 1

        corrupt_artifact(source, "benchmark.json", change)
    elif problem == "symlink":
        target = source.parent / manifest["artifacts"]["results"]["file"]
        elsewhere = tmp_path / "copied-results.json"
        elsewhere.write_bytes(target.read_bytes())
        assert tmp_path in target.parents
        target.unlink()
        target.symlink_to(elsewhere)
    elif problem == "export_content":
        source = chain["archive"] / "manifest.json"
        corrupt_artifact(
            source,
            "results.json",
            lambda d: d["content"]["sources"][0].update(config_sha256="0" * 64),
        )
    else:
        source = chain["archive"] / "manifest.json"
        csv_path = chain["archive"] / "experiment_summary.csv"
        csv_path.write_text(csv_path.read_text().replace("saved_predictions", "fabricated_new_gpu"))
        content = json.loads(source.read_text())
        content["artifacts"][csv_path.name]["sha256"] = hashlib.sha256(
            csv_path.read_bytes()
        ).hexdigest()
        content["artifacts"][csv_path.name]["bytes"] = csv_path.stat().st_size
        source.write_text(json.dumps(content))
    result = verify_results(source, check_inputs=problem == "raw_input")
    assert result.status == "failed", result.to_dict()
    assert result.checks["actual_model_inference_reproduction"]["status"] == "not_performed"
    if problem == "raw_input":
        assert result.checks["original_input_integrity"]["status"] == "failed"


def test_evaluation_source_provenance_is_checked_without_recalculating_ap(chain):
    source = chain["evaluation"]
    value = json.loads(source.read_text())
    value["provenance"]["source_execution"]["config_sha256"] = "0" * 64
    source.write_text(json.dumps(value))
    result = verify_results(source)
    assert result.status == "failed" and "source_execution" in result.failures[0]


def test_cli_verification_report_is_exclusive_and_explicitly_scoped(experiment, tmp_path, capsys):
    config, _ = experiment
    source = run(output(config, tmp_path, "run"))
    target = tmp_path / "verification.json"
    main(["verify-results", str(source), "--check-inputs", "--output", str(target)])
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "checks_passed" and report["verification_schema_version"] == 1
    assert report["checks"]["actual_model_inference_reproduction"]["status"] == "not_performed"
    before = target.read_bytes()
    with pytest.raises(SystemExit) as error:
        main(["verify-results", str(source), "--output", str(target)])
    assert error.value.code == 2 and target.read_bytes() == before
    with pytest.raises(SystemExit) as error:
        main(["verify-results", str(tmp_path / "missing")])
    assert error.value.code == 1


def test_default_verification_imports_no_gpu_modal_gui_or_scipy(experiment, tmp_path):
    config, _ = experiment
    source = run(output(config, tmp_path, "run"))
    archive = tmp_path / "archive"
    export_results([source], archive, format="all")
    code = """
import sys,importlib.abc
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self,name,*args):
        if name.split('.')[0] in {'torch','pcdet','spconv','cumm','modal','open3d','scipy','shapely'}:
            raise AssertionError('forbidden import '+name)
sys.meta_path.insert(0,Block())
from raven.reproducibility.verify import verify_results
r=verify_results(sys.argv[1]);assert r.status=='integrity_only',r.failures
assert r.checks['actual_model_inference_reproduction']['status']=='not_performed'
"""
    env = dict(
        os.environ,
        PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"),
        PYTHONDONTWRITEBYTECODE="1",
    )
    subprocess.run(
        [sys.executable, "-c", code, str(archive / "manifest.json")],
        check=True,
        capture_output=True,
        env=env,
    )
