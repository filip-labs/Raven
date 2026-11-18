import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from raven.cli import main
from raven.experiments import artifacts, runner, splits
from raven.experiments.config import ExperimentConfig
from raven.perception.detection.base import Detection3D
from raven.perception.detection.predictions import detections_to_document


@pytest.fixture
def experiment(tmp_path):
    root = tmp_path / "dataset"
    for folder in ("velodyne", "label_2", "calib"):
        (root / "training" / folder).mkdir(parents=True)
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    for frame in ("000000", "000003"):  # Both in the actual pinned train manifest.
        scan = root / "training/velodyne" / f"{frame}.bin"
        np.array([[0, 0, 0, 0.2], [2, 1, 0, 0.8]], dtype="<f4").tofile(scan)
        (root / "training/label_2" / f"{frame}.txt").write_text(
            "Car 0 0 0 0 0 20 20 2 2 4 0 2 10 0\n"
        )
        (root / "training/calib" / f"{frame}.txt").write_text(
            "R0_rect: 1 0 0 0 1 0 0 0 1\nTr_velo_to_cam: 1 0 0 0 0 1 0 0 0 0 1 0\n"
        )
        box = Detection3D((2, 1, 0), (4, 2, 1), 0, "Car", 0.8, "kitti_velodyne")
        document = detections_to_document([box], frame_id=frame)
        document["input"].update(sha256=hashlib.sha256(scan.read_bytes()).hexdigest(), points=2)
        (predictions / f"{frame}.json").write_text(json.dumps(document))
    return ExperimentConfig.from_dict(
        {
            "schema_version": 1,
            "name": "Synthetic saved baseline",
            "experiment_id": "test-001",
            "dataset": {"root": str(root), "split": "train"},
            "frames": {"ids": ["000003", "000000"]},
            "model": {
                "name": "pointpillars",
                "source": "saved",
                "predictions_dir": str(predictions),
            },
            "preprocessing": [],
            "seed": 42,
            "output": {"directory": str(tmp_path / "runs"), "artifact_prefix": "baseline"},
        },
        base_dir=tmp_path,
    )


def path(config, frame="000000", kind="scan"):
    if kind == "predictions":
        return config.model.predictions_dir / f"{frame}.json"
    folder = {"scan": "velodyne", "labels": "label_2", "calibration": "calib"}[kind]
    return (
        config.dataset.root / "training" / folder / (frame + (".bin" if kind == "scan" else ".txt"))
    )


def inputs(config):
    return {p: p.read_bytes() for p in config.dataset.root.rglob("*") if p.is_file()} | {
        p: p.read_bytes() for p in config.model.predictions_dir.iterdir()
    }


def test_end_to_end_saved_baseline_truth_and_artifacts(experiment):
    before = inputs(experiment)
    preflight = runner.preflight_experiment(experiment)
    assert preflight.status == "ready" and len(preflight.frames) == 2
    assert not experiment.output.directory.exists()
    summary = runner.run_experiment(experiment)
    assert summary.status == "success" and summary.succeeded == 2 and summary.failed == 0
    results = json.loads(experiment.artifact_paths["results"].read_text())
    assert [f["frame_id"] for f in results["frames"]] == ["000003", "000000"]
    for frame in results["frames"]:
        assert frame["input_count"] == frame["processed_count"] == 2
        assert frame["prediction_count"] == 1 and frame["class_counts"] == {"Car": 1}
        assert frame["prediction_source"]["mode"] == "loaded_saved_predictions"
        assert frame["prediction_source"]["input_binding"] == "sha256_verified"
        assert frame["ground_truth"]["objects"][0]["position_reference"] == "bottom_center"
        assert "R0_rect" in frame["ground_truth"]["calibration"]
        assert frame["processed_xyzi_sha256"] == frame["inputs"]["scan"]["sha256"]
    manifest_path = experiment.run_directory / "baseline.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "success"
    for artifact in manifest["artifacts"].values():
        assert (
            artifacts.file_sha256(experiment.run_directory / artifact["file"]) == artifact["sha256"]
        )
    provenance = json.loads(experiment.artifact_paths["provenance"].read_text())
    assert provenance["split"]["status"] == "verified"
    assert provenance["environment"]["python"] and provenance["environment"]["raven_commit"]
    assert provenance["started_at_utc"] <= provenance["ended_at_utc"]
    assert provenance["config_sha256"] == experiment.config_sha256
    assert inputs(experiment) == before
    previous = {p: p.read_bytes() for p in experiment.run_directory.iterdir()}
    repeated = runner.run_experiment(experiment)
    assert repeated.status == "failed" and repeated.output_directory is None
    assert {p: p.read_bytes() for p in experiment.run_directory.iterdir()} == previous


def test_dry_run_and_deterministic_input_identities(experiment):
    a = runner.preflight_experiment(experiment)
    b = runner.preflight_experiment(experiment)
    assert a == b
    summary = runner.run_experiment(experiment, dry_run=True)
    assert summary.status == "ready" and summary.succeeded == 0
    assert summary.output_directory is None and not experiment.output.directory.exists()


@pytest.mark.parametrize("kind", ["scan", "labels", "calibration", "predictions"])
def test_missing_required_inputs_never_create_outputs(experiment, kind):
    path(experiment, kind=kind).unlink()
    summary = runner.run_experiment(experiment)
    assert summary.status == "failed" and summary.output_directory is None
    assert not experiment.output.directory.exists()
    assert summary.failures[0].frame_id == "000000"


@pytest.mark.parametrize("kind", ["scan", "labels", "calibration", "predictions"])
def test_corrupt_inputs_fail_preflight(experiment, kind):
    path(experiment, kind=kind).write_bytes(b"corrupt")
    summary = runner.run_experiment(experiment)
    assert summary.status == "failed" and not experiment.output.directory.exists()


@pytest.mark.parametrize(
    "change", ["frame", "coordinate", "hash", "points", "model", "metadata", "duplicates"]
)
def test_saved_prediction_schema_and_available_provenance(experiment, change):
    p = path(experiment, kind="predictions")
    d = json.loads(p.read_text())
    if change == "frame":
        d["input"]["frame_id"] = "000001"
    if change == "coordinate":
        d["coordinate_frame"] = "lidar"
    if change == "hash":
        d["input"]["sha256"] = "f" * 64
    if change == "points":
        d["input"]["points"] = 123
    if change == "model":
        d["effective_config"] = {"MODEL": {"NAME": "Other"}}
    if change == "metadata":
        d["effective_config"] = None
    p.write_text(json.dumps(d))
    if change == "duplicates":
        p.write_text(
            p.read_text().replace('"schema_version": 1', '"schema_version": 1, "schema_version": 1')
        )
    summary = runner.run_experiment(experiment)
    assert summary.status == "failed" and not experiment.output.directory.exists()


def test_missing_capture_hash_is_explicitly_unavailable(experiment):
    p = path(experiment, kind="predictions")
    d = json.loads(p.read_text())
    del d["input"]["sha256"]
    p.write_text(json.dumps(d))
    assert runner.run_experiment(experiment).status == "success"
    frames = json.loads(experiment.artifact_paths["results"].read_text())["frames"]
    assert frames[1]["prediction_source"]["input_binding"] == "unavailable"


def test_split_membership_is_not_training_directory_membership(experiment):
    c = replace(experiment, dataset=replace(experiment.dataset, split="val"))
    summary = runner.run_experiment(c)
    assert summary.status == "failed" and "not in declared val" in summary.failures[0].message
    assert not c.output.directory.exists()


def test_unavailable_test_split_is_blocked_not_labeled(experiment):
    c = replace(experiment, dataset=replace(experiment.dataset, split="test"))
    summary = runner.run_experiment(c)
    assert summary.status == "blocked" and "unlabeled" in summary.failures[0].message
    assert not c.output.directory.exists()


def test_missing_or_invalid_split_evidence(experiment, monkeypatch):
    for error, status in [
        (FileNotFoundError("missing evidence"), "blocked"),
        (ValueError("bad hash"), "failed"),
    ]:

        def fail(_, error=error):
            raise error

        monkeypatch.setattr(runner, "split_membership", fail)
        assert runner.run_experiment(experiment).status == status
        assert not experiment.output.directory.exists()


def live_config(config):
    d = config.to_dict()
    d["model"] = {
        "name": "pointpillars",
        "source": "inference",
        "config": "/nonexistent/pointpillar.yaml",
        "checkpoint": "/nonexistent/weights.pth",
        "checkpoint_sha256": "a" * 64,
        "openpcdet_revision": "233f849829b6ac19afb8af8837a0246890908755",
    }
    d["preprocessing"] = [
        {"name": "range_filter", "x_range": [0, 1], "y_range": [-1, 1], "z_range": [-1, 1]}
    ]
    return ExperimentConfig.from_dict(d, base_dir=Path.cwd())


def test_real_unsupported_inference_has_no_saved_fallback(experiment, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    summary = runner.run_experiment(live_config(experiment))
    assert summary.status == "blocked" and summary.succeeded == 0
    assert "unsupported inference environment" in summary.failures[0].message
    assert not experiment.output.directory.exists()


def mock_live(monkeypatch):
    monkeypatch.setattr(
        runner,
        "_inference_preflight",
        lambda c: {"synthetic_test": True, "config_sha256": "synthetic"},
    )
    model = Mock()
    model.provenance = {"synthetic_test": True, "not_real_gpu_inference": True}
    model.detect.return_value = []
    original_sha = runner.file_sha256
    monkeypatch.setattr(
        runner,
        "file_sha256",
        lambda p: "synthetic" if str(p).endswith("pointpillar.yaml") else original_sha(p),
    )
    monkeypatch.setattr(runner, "_build_detector", lambda c, report: model)
    return model


def test_mocked_live_uses_real_preprocessing_and_existing_detector_boundary(
    experiment, monkeypatch
):
    model = mock_live(monkeypatch)
    c = live_config(experiment)
    summary = runner.run_experiment(c)
    assert summary.status == "success"
    assert model.detect.call_count == 2
    for call in model.detect.call_args_list:
        np.testing.assert_array_equal(
            call.args[0].points, np.array([[0, 0, 0, 0.2]], dtype=np.float32)
        )
    frames = json.loads(c.artifact_paths["results"].read_text())["frames"]
    assert all(f["processed_count"] == 1 for f in frames)
    assert frames[0]["prediction_source"]["model_provenance"]["synthetic_test"] is True
    model.initialize.assert_called_once()
    model.close.assert_called_once()


def test_backend_initialization_and_cleanup_failure_is_not_success(experiment, monkeypatch):
    model = mock_live(monkeypatch)
    model.initialize.side_effect = RuntimeError("initialize failed")
    model.close.side_effect = RuntimeError("cleanup failed")
    summary = runner.run_experiment(live_config(experiment))
    assert summary.status == "blocked" and "cleanup failed" in summary.failures[0].message
    assert not experiment.output.directory.exists()


def test_partial_execution_failure_collects_without_fabrication(experiment, monkeypatch):
    original = runner._execute_frame

    def execute(config, frame, detector):
        if frame.frame_id == "000003":
            raise RuntimeError("controlled mandatory failure")
        return original(config, frame, detector)

    monkeypatch.setattr(runner, "_execute_frame", execute)
    summary = runner.run_experiment(experiment)
    assert summary.status == "failed" and (summary.succeeded, summary.failed) == (1, 1)
    frames = json.loads(experiment.artifact_paths["results"].read_text())["frames"]
    assert frames[0]["status"] == "failed" and "predictions" not in frames[0]
    assert frames[1]["status"] == "success"
    assert (
        json.loads((experiment.run_directory / "baseline.manifest.json").read_text())["status"]
        == "failed"
    )


def test_changed_input_after_preflight_is_a_frame_failure(experiment, monkeypatch):
    original = runner.preflight_experiment

    def preflight(config):
        report = original(config)
        p = path(config)
        values = np.fromfile(p, dtype="<f4")
        values[0] = 0.5
        values.tofile(p)
        return report

    monkeypatch.setattr(runner, "preflight_experiment", preflight)
    summary = runner.run_experiment(experiment)
    assert summary.status == "failed" and summary.failed == 1
    assert "changed since preflight" in summary.failures[0].message


def test_permission_failures_and_artifact_failure_never_commit_success(experiment, monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(runner.os, "access", lambda *a: False)
        assert runner.run_experiment(experiment).status == "failed"
        assert not experiment.output.directory.exists()

    original = artifacts.write_text_exclusive

    def fail(destination, text):
        if destination.name.endswith(".config.json"):
            raise PermissionError("controlled persistence denial")
        return original(destination, text)

    monkeypatch.setattr(artifacts, "write_text_exclusive", fail)
    summary = runner.run_experiment(experiment)
    assert summary.status == "failed" and summary.output_directory == experiment.run_directory
    assert not (experiment.run_directory / "baseline.manifest.json").exists()
    assert (experiment.run_directory / "baseline.started.json").exists()


def test_atomic_writer_refuses_existing_files_and_cleans_temps(tmp_path):
    destination = tmp_path / "existing.json"
    destination.write_text("protected")
    with pytest.raises(FileExistsError):
        artifacts.write_json_exclusive(destination, {"new": True})
    assert destination.read_text() == "protected"
    assert not list(tmp_path.glob(".raven-*"))


def test_cli_success_dry_run_errors_and_fresh_dependency_free_execution(
    experiment, tmp_path, capsys
):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(experiment.to_dict()))
    main(["run-experiment", str(config_path), "--dry-run"])
    assert json.loads(capsys.readouterr().out)["status"] == "ready"
    code = """import sys
class Block:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'torch','spconv','cumm','pcdet','open3d','scipy'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
from raven.cli import main
main(['run-experiment', sys.argv[1]])
"""
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    process = subprocess.run(
        [sys.executable, "-c", code, str(config_path)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout)["succeeded"] == 2
    with pytest.raises(SystemExit) as exc:
        main(["run-experiment", str(config_path)])
    assert exc.value.code == 1
    capsys.readouterr()
    with pytest.raises(SystemExit) as exc:
        main(["run-experiment", str(tmp_path / "missing.json")])
    assert exc.value.code == 2


def test_packaged_split_resource_path(monkeypatch, tmp_path):
    # Exercise the installed-wheel resource layout without trusting arbitrary split contents.
    import shutil

    folder = tmp_path / "kitti_splits"
    source = Path(__file__).resolve().parents[1] / "configs/kitti_splits"
    shutil.copytree(source, folder)
    monkeypatch.setattr(splits, "__file__", str(tmp_path / "splits.py"))
    ids, evidence = splits.split_membership("train")
    assert "000000" in ids and evidence["status"] == "verified"
    (folder / "train.txt").write_text("000000\n")
    with pytest.raises(ValueError, match="checksum/count"):
        splits.split_membership("train")


def test_existing_output_created_between_preflight_and_reservation_is_preserved(
    experiment, monkeypatch
):
    original = runner.preflight_experiment

    def preflight(config):
        report = original(config)
        config.run_directory.mkdir(parents=True)
        (config.run_directory / "protected.txt").write_text("existing results")
        return report

    monkeypatch.setattr(runner, "preflight_experiment", preflight)
    summary = runner.run_experiment(experiment)
    assert summary.status == "failed" and summary.output_directory is None
    assert (experiment.run_directory / "protected.txt").read_text() == "existing results"
    assert len(list(experiment.run_directory.iterdir())) == 1


def test_actual_scan_read_change_detected_before_output(experiment, monkeypatch):
    original = runner.load_kitti_bin

    def load(path):
        cloud = original(path)
        values = np.fromfile(path, dtype="<f4")
        values[0] = 0.25
        values.tofile(path)
        return cloud

    monkeypatch.setattr(runner, "load_kitti_bin", load)
    summary = runner.run_experiment(experiment)
    assert summary.status == "failed" and "changed during read" in summary.failures[0].message
    assert not experiment.output.directory.exists()


def test_backend_cleanup_failure_commits_failed_run(experiment, monkeypatch):
    model = mock_live(monkeypatch)
    model.close.side_effect = RuntimeError("controlled cleanup failure")
    c = live_config(experiment)
    summary = runner.run_experiment(c)
    assert summary.status == "failed" and summary.succeeded == 2
    assert summary.failures[0].phase == "backend_cleanup"
    assert (
        json.loads((c.run_directory / "baseline.manifest.json").read_text())["status"] == "failed"
    )


def test_manifest_publication_failure_is_uncommitted(experiment, monkeypatch):
    original = artifacts.write_json_exclusive

    def write(path, data):
        if path.name.endswith(".manifest.json"):
            raise PermissionError("controlled final commit failure")
        return original(path, data)

    monkeypatch.setattr(artifacts, "write_json_exclusive", write)
    summary = runner.run_experiment(experiment)
    assert summary.status == "failed"
    assert not (experiment.run_directory / "baseline.manifest.json").exists()
    assert (experiment.run_directory / "baseline.failure.json").exists()
    assert (experiment.run_directory / "baseline.started.json").exists()
    assert not list(experiment.run_directory.glob(".raven-*"))


def test_saved_preprocessing_incompatibility_cannot_bypass_runner(experiment):
    with pytest.raises(ValueError, match="saved predictions"):
        replace(experiment, preprocessing=({"name": "noise", "position_std": 0, "seed": 42},))
    assert not experiment.output.directory.exists()


def test_nonfinite_capture_metadata_fails_preflight(experiment):
    p = path(experiment, kind="predictions")
    p.write_text(p.read_text()[:-1] + ', "metadata": 1e400}')
    summary = runner.run_experiment(experiment)
    assert summary.status == "failed" and not experiment.output.directory.exists()


def test_live_mutated_input_is_rejected(experiment, monkeypatch):
    model = mock_live(monkeypatch)

    def detect(cloud):
        cloud.points.setflags(write=True)
        cloud.points[0, 0] = 5
        return []

    model.detect.side_effect = detect
    summary = runner.run_experiment(live_config(experiment))
    assert summary.status == "failed" and summary.failed == 2
    assert all("mutated" in f.message for f in summary.failures)


def test_missing_dataset_is_preflight_failure(experiment):
    c = replace(
        experiment, dataset=replace(experiment.dataset, root=experiment.dataset.root / "missing")
    )
    summary = runner.run_experiment(c)
    assert summary.status == "failed" and "dataset root unavailable" in summary.failures[0].message
    assert not c.output.directory.exists()


def test_seeded_selection_does_not_load_unselected_corrupt_scan(experiment):
    c = replace(experiment, frames=replace(experiment.frames, sample_size=1))
    assert c.frame_ids == ("000003",)
    path(experiment, "000000").write_bytes(b"unselected corrupt input")
    summary = runner.run_experiment(c)
    assert summary.status == "success" and summary.succeeded == 1
    assert summary.selected_frame_ids == ("000003",)


def test_valid_validation_member_executes_from_training_storage(experiment):
    import shutil

    for kind in ("scan", "labels", "calibration", "predictions"):
        shutil.copyfile(path(experiment, "000000", kind), path(experiment, "000001", kind))
    p = path(experiment, "000001", "predictions")
    d = json.loads(p.read_text())
    d["input"]["frame_id"] = "000001"
    p.write_text(json.dumps(d))
    c = replace(
        experiment,
        frames=replace(experiment.frames, ids=("000001",)),
        dataset=replace(experiment.dataset, split="val"),
    )
    assert runner.run_experiment(c).status == "success"


def test_nonregular_input_file_rejected_without_opening(tmp_path):
    fifo = tmp_path / "scan.bin"
    os.mkfifo(fifo)
    with pytest.raises(FileNotFoundError, match="regular"):
        artifacts.file_sha256(fifo)


@pytest.mark.parametrize("problem", [None, "cuda", "dependency", "revision", "checkpoint"])
def test_live_asset_preflight_with_synthetic_cpu_modules(
    experiment, tmp_path, monkeypatch, problem
):
    root = tmp_path / "OpenPCDet"
    (root / "pcdet").mkdir(parents=True)
    (root / "pcdet/__init__.py").write_text("# synthetic CPU preflight, not a GPU backend\n")
    (root / "tools").mkdir()
    config_path = root / "tools/pointpillar.yaml"
    config_path.write_text("# synthetic declared configuration\n")
    checkpoint = tmp_path / "synthetic.pth"
    checkpoint.write_bytes(b"synthetic, never deserialized")
    d = live_config(experiment).to_dict()
    d["model"].update(
        config=str(config_path),
        checkpoint=str(checkpoint),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    )
    if problem == "checkpoint":
        d["model"]["checkpoint_sha256"] = "b" * 64
    c = ExperimentConfig.from_dict(d, base_dir=tmp_path)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: problem != "cuda")),
    )
    original_spec = runner.importlib.util.find_spec
    monkeypatch.setattr(
        runner.importlib.util,
        "find_spec",
        lambda name: None if problem == "dependency" else object(),
    )
    monkeypatch.setattr(
        runner.subprocess,
        "check_output",
        lambda *a, **kw: "0" * 40 if problem == "revision" else c.model.openpcdet_revision,
    )
    if problem in ("cuda", "dependency"):
        with pytest.raises(RuntimeError):
            runner._inference_preflight(c)
    elif problem:
        with pytest.raises(ValueError):
            runner._inference_preflight(c)
    else:
        result = runner._inference_preflight(c)
        assert result["checkpoint_sha256"] == c.model.checkpoint_sha256
        assert result["openpcdet_root"] == str(root)
        assert result["pickle_loading_allowed"] is False
    monkeypatch.setattr(runner.importlib.util, "find_spec", original_spec)


def test_installed_module_does_not_inherit_unrelated_parent_git_identity(tmp_path, monkeypatch):
    module = tmp_path / "unpacked/raven/experiments/runner.py"
    module.parent.mkdir(parents=True)
    module.write_text("# synthetic installed module identity\n")
    monkeypatch.setattr(runner, "__file__", str(module))
    environment = runner._environment()
    assert environment["raven_commit"] is None
    assert environment["raven_worktree_status"] is None
    assert environment["source_path"] is None
    assert environment["runner_source_sha256"] == hashlib.sha256(module.read_bytes()).hexdigest()
