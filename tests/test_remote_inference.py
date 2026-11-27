"""Synthetic CPU contract/runner tests. No fixture represents real Modal/GPU evidence."""

import hashlib
import os
import struct
import subprocess
import sys
import zlib
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from raven.cli import main
from raven.core.point_cloud import PointCloud
from raven.evaluation.evaluate import evaluate_experiment
from raven.evaluation.inputs import load_completed_run, prepare_sources
from raven.experiments import runner
from raven.experiments.config import ExperimentConfig, ModalConfig
from raven.inference import contracts as c
from raven.inference.detector import RemoteDetector
from raven.inference.evidence import validate_remote_evidence
from raven.inference.json_native import normalize_json_containers
from raven.inference.modal_image import image_spec, recipe_sha256
from raven.inference.modal_transport import (
    ModalTransport,
    RemoteExecutionError,
    modal_preflight,
)
from raven.inference.worker import WorkerSession, runtime_source_sha256
from raven.perception.detection.base import Detection3D
from raven.perception.detection.predictions import detections_to_document, load_detections


@pytest.fixture
def cloud():
    return PointCloud(
        np.array([[20, 0, 0, 0.2], [30, 1, 0, 0.8]], dtype=np.float32), "000001", "kitti_velodyne"
    )


def make_request(cloud):
    return c.create_request(
        cloud,
        request_id="test-request",
        experiment_id="synthetic",
        config_sha256="a" * 64,
        preprocessing_sha256=c.digest([]),
        submitted_at_utc="2026-11-26T00:00:00+00:00",
    )


def make_response(request):
    box = Detection3D((20, 0, 0), (4, 2, 1.5), 0.1, "Car", 0.9, "kitti_velodyne")
    response = {
        "schema_version": 1,
        "status": "success",
        "request_id": request["request_id"],
        "frame_id": request["frame_id"],
        "request_sha256": request["request_sha256"],
        "input_sha256": request["input"]["sha256"],
        "model": c.model_identity(),
        "predictions": detections_to_document([box], frame_id=request["frame_id"]),
        "worker": {
            "provider": "modal",
            "worker_id": "synthetic-worker",
            "image_recipe_sha256": recipe_sha256(),
            "gpu": "Tesla T4",
            "compute_capability": [7, 5],
            "python": "3.10.16",
            "packages": dict(c.PACKAGES),
            "cuda_runtime": "11.8",
            "raven_source_sha256": runtime_source_sha256(),
            "model_provenance": {
                "openpcdet_commit": c.REVISION,
                "checkpoint_sha256": c.CHECKPOINT,
                "config": {"sha256": c.MODEL_CONFIG},
            },
        },
        "timings": {
            "worker_started_at_utc": "2026-11-26T00:00:01+00:00",
            "worker_finished_at_utc": "2026-11-26T00:00:02+00:00",
            "model_load_seconds": 2.0,
            "detect_seconds": 0.1,
            "worker_total_seconds": 0.2,
            "clock": "time.perf_counter",
            "gpu_synchronized": True,
        },
    }
    response = normalize_json_containers(response)
    response["response_sha256"] = c.response_digest(response)
    return response


class SyntheticTransport:
    def __init__(self):
        self.requests = []

    def submit(self, request):
        self.requests.append(request)
        return make_response(request), {
            "provider": "modal",
            "synthetic_test": True,
            "round_trip_seconds": 0.01,
            "function_call_id": "fc-synthetic",
            "call_id": "fc-synthetic",
            "submission_attempts": 1,
        }


def test_xyzi_binary_round_trip_has_independent_little_endian_bytes_and_no_mutation(cloud):
    before = cloud.points.copy()
    request = make_request(cloud)
    assert request["points"] == struct.pack("<8f", 20, 0, 0, 0.2, 30, 1, 0, 0.8)
    loaded = c.validate_request(request)
    np.testing.assert_array_equal(loaded.points, before)
    assert not np.shares_memory(loaded.points, cloud.points)
    assert not loaded.points.flags.writeable
    assert request["input"]["sha256"] == hashlib.sha256(request["points"]).hexdigest()
    np.testing.assert_array_equal(cloud.points, before)
    c.validate_request_metadata(c.request_metadata(request))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r.update(schema_version=True),
        lambda r: r.update(schema_version=2),
        lambda r: r.update(frame_id="1"),
        lambda r: r.update(coordinate_frame="camera"),
        lambda r: r.update(request_id="../bad"),
        lambda r: r.update(points=b"bad"),
        lambda r: r.update(points=bytearray(r["points"])),
        lambda r: r["input"].update(dtype="float64"),
        lambda r: r["input"].update(shape=[1, 8]),
        lambda r: r["input"].update(points=True),
        lambda r: r["input"].update(points=c.MAX_POINTS + 1),
        lambda r: r["input"].update(sha256="0" * 64),
        lambda r: r["model"].update(checkpoint_sha256="0" * 64),
        lambda r: r.update(preprocessing_sha256="bad"),
        lambda r: r.update(submitted_at_utc="2026-11-26"),
        lambda r: r.update(context={"unsupported": 1}),
        lambda r: r.update(extra="unsupported"),
    ],
)
def test_request_rejects_malformed_metadata_or_payload(cloud, mutate):
    request = make_request(cloud)
    mutate(request)
    with pytest.raises((ValueError, TypeError)):
        c.validate_request(request)


def test_nonfinite_payload_rejected_even_when_hashes_are_consistent(cloud):
    request = make_request(cloud)
    request["points"] = struct.pack("<8f", float("nan"), 0, 0, 1, 1, 1, 1, 1)
    request["input"]["sha256"] = hashlib.sha256(request["points"]).hexdigest()
    request["request_sha256"] = c.digest(
        {k: v for k, v in c.request_metadata(request).items() if k != "request_sha256"}
    )
    with pytest.raises(ValueError, match="finite"):
        c.validate_request(request)


def test_empty_input_rejected_before_transport():
    transport = SyntheticTransport()
    detector = RemoteDetector(
        transport,
        experiment_id="synthetic",
        config_sha256="a" * 64,
        preprocessing_sha256=c.digest([]),
    )
    with pytest.raises(ValueError, match="points"):
        detector.detect(PointCloud(np.empty((0, 4)), "000001", "kitti_velodyne"))
    assert not transport.requests


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r.update(status="failed"),
        lambda r: r.update(schema_version=True),
        lambda r: r.update(request_id="different"),
        lambda r: r.update(frame_id="000002"),
        lambda r: r.update(input_sha256="0" * 64),
        lambda r: r.update(request_sha256="0" * 64),
        lambda r: r["worker"].update(gpu="NVIDIA L4"),
        lambda r: r["worker"].update(packages={}),
        lambda r: r["worker"].update(image_recipe_sha256="0" * 64),
        lambda r: r["worker"].update(raven_source_sha256="0" * 64),
        lambda r: r["worker"]["model_provenance"].update(checkpoint_sha256="0" * 64),
        lambda r: r["worker"]["model_provenance"].update(synthetic_test=True),
        lambda r: r["timings"].update(detect_seconds=-1),
        lambda r: r["timings"].update(gpu_synchronized=False),
        lambda r: r["timings"].update(worker_finished_at_utc="2026-11-25T00:00:00+00:00"),
        lambda r: r["predictions"].update(rotation_axis="y"),
        lambda r: r["predictions"]["detections"][0].update(score=1.1),
    ],
)
def test_response_rejects_bad_geometry_identity_provenance_and_timing(cloud, mutate):
    request = make_request(cloud)
    response = make_response(request)
    mutate(response)
    response["response_sha256"] = c.response_digest(response)
    with pytest.raises((ValueError, TypeError)):
        c.validate_response(response, request)


def test_response_checksum_is_checked_and_prediction_reader_is_reused(cloud, tmp_path):
    request = make_request(cloud)
    response = make_response(request)
    detections = c.validate_response(response, request)
    path = tmp_path / "predictions.json"
    path.write_bytes(c.canonical(response["predictions"]))
    assert load_detections(path, expected_frame_id=cloud.frame_id) == detections
    response["response_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checksum"):
        c.validate_response(response, request)


def test_optional_vehicle_metadata_is_preserved_without_transforming_points(cloud):
    request = make_request(cloud)
    context = {
        "vehicle_id": "vehicle-A",
        "sensor_id": "lidar-0",
        "session_id": "session-1",
        "capture_timestamp_utc": "2026-11-26T00:00:00+00:00",
        "pose": {"coordinate_reference": "world", "matrix": np.eye(4).tolist()},
    }
    request = c.create_request(
        cloud,
        request_id="vehicle-request",
        experiment_id="synthetic",
        config_sha256="a" * 64,
        preprocessing_sha256=c.digest([]),
        submitted_at_utc=request["submitted_at_utc"],
        context=context,
    )
    assert request["context"] == context
    assert request["points"] == make_request(cloud)["points"]


def test_detector_rejects_stale_duplicate_and_detaches_provenance(cloud):
    transport = SyntheticTransport()
    detector = RemoteDetector(
        transport,
        experiment_id="synthetic",
        config_sha256="a" * 64,
        preprocessing_sha256=c.digest([]),
    )
    result = detector.detect(cloud)
    assert result[0].label == "Car"
    original = detector.provenance
    original["response"]["status"] = "corrupted"
    assert detector.provenance["response"]["status"] == "success"
    stale = detector.provenance["response"]
    transport.submit = lambda request: (stale, {})
    with pytest.raises(ValueError, match="duplicate"):
        detector.detect(cloud)
    with pytest.raises(RuntimeError, match="validated"):
        _ = detector.provenance
    detector.close()
    with pytest.raises(RuntimeError, match="closed"):
        detector.detect(cloud)


@pytest.fixture
def config(tmp_path):
    root = tmp_path / "dataset"
    for folder in ("velodyne", "label_2", "calib", "image_2"):
        (root / "training" / folder).mkdir(parents=True)
    for frame in ("000001", "000002"):
        np.array([[20, 0, 0, 0.2], [30, 1, 0, 0.8]], dtype="<f4").tofile(
            root / "training/velodyne" / f"{frame}.bin"
        )
        (root / "training/label_2" / f"{frame}.txt").write_text(
            "Car 0 0 0 0 0 100 100 1.5 2 4 0 1 20 0\n"
        )
        (root / "training/calib" / f"{frame}.txt").write_text(
            "R0_rect: 1 0 0 0 1 0 0 0 1\nTr_velo_to_cam: 0 -1 0 0 0 0 -1 0 1 0 0 0\n"
            "P2: 700 0 600 0 0 700 180 0 0 0 1 0\n"
        )

        def chunk(kind, data):
            return (
                struct.pack(">I", len(data))
                + kind
                + data
                + struct.pack(">I", zlib.crc32(kind + data))
            )

        png = b"\x89PNG\r\n\x1a\n" + chunk(
            b"IHDR", struct.pack(">IIBBBBB", 1242, 375, 8, 0, 0, 0, 0)
        )
        png += chunk(b"IDAT", zlib.compress(b"\0" * (1243 * 375))) + chunk(b"IEND", b"")
        (root / "training/image_2" / f"{frame}.png").write_bytes(png)
    return ExperimentConfig.from_dict(
        {
            "schema_version": 2,
            "name": "Synthetic remote contract",
            "experiment_id": "remote-test",
            "dataset": {"root": str(root), "split": "val"},
            "frames": {"ids": ["000001"]},
            "model": {
                "name": "pointpillars",
                "source": "modal",
                "remote": {"app": "raven-pointpillars", "environment": "main"},
            },
            "preprocessing": [
                {
                    "name": "range_filter",
                    "x_range": [0, 25],
                    "y_range": [-40, 40],
                    "z_range": [-3, 1],
                }
            ],
            "seed": 42,
            "output": {"directory": str(tmp_path / "runs"), "artifact_prefix": "modal"},
        },
        base_dir=tmp_path,
    )


def test_schema_v2_round_trip_and_frame_spending_guard(config):
    assert config.schema_version == 2
    assert ExperimentConfig.from_dict(config.to_dict(), base_dir=Path.cwd()) == config
    document = config.to_dict()
    document["frames"]["ids"].append("000002")
    with pytest.raises(ValueError, match="max_frames"):
        ExperimentConfig.from_dict(document, base_dir=Path.cwd())


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d.update(schema_version=1),
        lambda d: d["model"].update(checkpoint="weights.pth"),
        lambda d: d["model"].update(predictions_dir="saved"),
        lambda d: d["model"].update(remote=None),
        lambda d: d["model"]["remote"].update(timeout_seconds=0),
        lambda d: d["model"]["remote"].update(timeout_seconds=True),
        lambda d: d["model"]["remote"].update(timeout_seconds=601),
        lambda d: d["model"]["remote"].update(max_frames=0),
        lambda d: d["model"]["remote"].update(max_frames=9),
        lambda d: d["model"]["remote"].update(worker_class="Other"),
        lambda d: d["model"]["remote"].update(app="../bad"),
        lambda d: d["model"]["remote"].update(token="never-store-tokens"),
    ],
)
def test_incompatible_remote_configuration_rejected(config, mutation):
    document = config.to_dict()
    mutation(document)
    with pytest.raises((ValueError, TypeError)):
        ExperimentConfig.from_dict(document, base_dir=Path.cwd())


def test_unapproved_real_runner_is_blocked_without_rpc_or_outputs(
    config, monkeypatch, capsys, tmp_path
):
    monkeypatch.delenv("RAVEN_ALLOW_MODAL_GPU", raising=False)
    monkeypatch.delenv("RAVEN_MODAL_ASSET_RIGHTS_CONFIRMED", raising=False)
    result = runner.run_experiment(config)
    assert result.status == "blocked"
    assert "RAVEN_ALLOW_MODAL_GPU" in result.failures[0].message
    assert not config.output.directory.exists()
    path = tmp_path / "config.json"
    path.write_bytes(c.canonical(config.to_dict()))
    with pytest.raises(SystemExit) as error:
        main(["run-experiment", str(path)])
    assert error.value.code == 1
    assert '"status": "blocked"' in capsys.readouterr().out


def test_sdk_missing_wrong_version_and_rights_gate_are_nonbillable(monkeypatch):
    monkeypatch.setenv("RAVEN_ALLOW_MODAL_GPU", "1")
    monkeypatch.delenv("RAVEN_MODAL_ASSET_RIGHTS_CONFIRMED", raising=False)
    with pytest.raises(RuntimeError, match="RIGHTS"):
        modal_preflight()
    monkeypatch.setenv("RAVEN_MODAL_ASSET_RIGHTS_CONFIRMED", "1")

    def unavailable(name):
        import importlib.metadata

        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr("importlib.metadata.version", unavailable)
    with pytest.raises(RuntimeError, match="unavailable"):
        modal_preflight()
    monkeypatch.setattr("importlib.metadata.version", lambda name: "other")
    with pytest.raises(RuntimeError, match="SDK must"):
        modal_preflight()


@pytest.mark.parametrize("error", [TimeoutError, ConnectionError, PermissionError])
def test_modal_transport_never_resubmits_and_sanitizes_failures(monkeypatch, cloud, error):
    monkeypatch.setenv("RAVEN_ALLOW_MODAL_GPU", "1")
    monkeypatch.setenv("RAVEN_MODAL_ASSET_RIGHTS_CONFIRMED", "1")
    call = Mock(object_id="fc-synthetic")
    call.get.side_effect = error("secret-token-must-not-appear")
    spawn = Mock(return_value=call)
    lookup = Mock(return_value=lambda: SimpleNamespace(infer=SimpleNamespace(spawn=spawn)))
    monkeypatch.setitem(
        sys.modules, "modal", SimpleNamespace(Cls=SimpleNamespace(from_name=lookup))
    )
    with pytest.raises(RemoteExecutionError) as failure:
        ModalTransport(ModalConfig("raven-pointpillars", "main", timeout_seconds=7)).submit(
            make_request(cloud)
        )
    assert "secret-token" not in str(failure.value)
    assert "fc-synthetic" in str(failure.value)
    spawn.assert_called_once()
    call.get.assert_called_once_with(timeout=7)
    call.cancel.assert_called_once_with(terminate_containers=True)
    lookup.assert_called_once_with(
        "raven-pointpillars", "PointPillarsWorker", environment_name="main"
    )


def test_runner_sends_exact_processed_points_and_evaluator_accepts_contract(config, monkeypatch):
    transport = SyntheticTransport()
    monkeypatch.setattr(
        "raven.inference.modal_transport.modal_preflight", lambda: {"source": "modal"}
    )
    monkeypatch.setattr(
        "raven.inference.modal_transport.ModalTransport", lambda settings: transport
    )
    before = {p: p.read_bytes() for p in config.dataset.root.rglob("*") if p.is_file()}
    result = runner.run_experiment(config)
    assert result.status == "success"
    assert len(transport.requests) == 1
    request = transport.requests[0]
    assert request["input"]["points"] == 1
    assert request["points"] == struct.pack("<4f", 20, 0, 0, 0.2)
    assert request["preprocessing_sha256"] == c.digest(config.to_dict()["preprocessing"])
    assert request["experiment_config_sha256"] == config.config_sha256
    manifest = config.run_directory / "modal.manifest.json"
    run = load_completed_run(manifest)
    frame = run.results["frames"][0]
    assert frame["processed_xyzi_sha256"] == request["input"]["sha256"]
    assert frame["input_count"] == 2 and frame["processed_count"] == 1
    assert frame["client_preprocessing_timing"]["seconds"] >= 0
    assert "points" not in frame["prediction_source"]["model_provenance"]["request"]
    evaluated = evaluate_experiment(manifest)
    assert evaluated.status == "blocked", evaluated.failures
    assert evaluated.evaluated_frame_count == 0
    assert not evaluated.metrics
    assert any("mocked inference" in failure for failure in evaluated.failures)
    for path, raw in before.items():
        assert path.read_bytes() == raw
    # Evidence is synthetic test data, never a claimed real GPU run.
    model = frame["prediction_source"]["model_provenance"]
    with pytest.raises(ValueError, match="bound"):
        validate_remote_evidence(model, {**frame, "processed_xyzi_sha256": "0" * 64}, config)
    with pytest.raises(ValueError, match="bound"):
        validate_remote_evidence(model, frame, replace(config, seed=43))
    frame["prediction_source"]["model_provenance"]["response"]["worker"]["model_provenance"][
        "not_real_gpu_inference"
    ] = True
    response = frame["prediction_source"]["model_provenance"]["response"]
    response["response_sha256"] = c.response_digest(response)
    with pytest.raises(ValueError, match="provenance"):
        prepare_sources(run, allow_training_smoke=False)


def test_remote_failure_aborts_remaining_frames_without_fallback(config, monkeypatch):
    config = replace(
        config,
        frames=replace(config.frames, ids=("000001", "000002")),
        model=replace(config.model, remote=replace(config.model.remote, max_frames=2)),
    )
    monkeypatch.setattr(
        "raven.inference.modal_transport.modal_preflight", lambda: {"source": "modal"}
    )
    transport = Mock()
    transport.submit.side_effect = RemoteExecutionError("uncertain remote outcome")
    monkeypatch.setattr(
        "raven.inference.modal_transport.ModalTransport", lambda settings: transport
    )
    result = runner.run_experiment(config)
    assert result.status == "failed" and result.failed == 2
    transport.submit.assert_called_once()
    assert any(f.phase == "remote_aborted" for f in result.failures)
    assert evaluate_experiment(config.run_directory / "modal.manifest.json").status == "invalid"


def test_worker_replay_cache_and_input_conflict_without_model_reinitialization(cloud, monkeypatch):
    synchronize = Mock()
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(synchronize=synchronize))
    )
    worker = WorkerSession.__new__(WorkerSession)
    request = make_request(cloud)
    worker.metadata = make_response(request)["worker"]
    worker.load_seconds = 0.2
    worker.detector = Mock()
    worker.detector.detect.return_value = c.validate_response(make_response(request), request)
    worker.cache = {}
    first = worker.infer(request)
    assert worker.infer(request) == first
    worker.detector.detect.assert_called_once()
    changed = deepcopy(request)
    changed["experiment_id"] = "other-experiment"
    changed["request_sha256"] = c.digest(
        {k: v for k, v in c.request_metadata(changed).items() if k != "request_sha256"}
    )
    with pytest.raises(ValueError, match="reused"):
        worker.infer(changed)


def test_core_remote_imports_do_not_require_sdk_gpu_or_gui():
    code = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, *args):
        if name.split('.')[0] in {'modal','torch','spconv','cumm','pcdet','open3d'}:
            raise AssertionError('forbidden import '+name)
sys.meta_path.insert(0, Block())
from raven.inference.detector import RemoteDetector
from raven.inference.json_native import normalize_json_containers
from raven.inference.worker import runtime_source_sha256
from raven.experiments.runner import run_experiment
from raven.evaluation.evaluate import evaluate_experiment
assert len(runtime_source_sha256()) == 64
"""
    env = dict(
        os.environ,
        PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"),
        PYTHONDONTWRITEBYTECODE="1",
    )
    subprocess.run([sys.executable, "-c", code], env=env, check=True, capture_output=True)


def test_image_spec_keeps_cuda_stack_consistent_and_has_no_assets():
    spec = image_spec()
    assert spec["base"].startswith("nvidia/cuda@sha256:")
    assert spec["python"] == "3.10" and spec["torch"] == "2.2.2+cu118"
    assert spec["spconv-cu118"] == "2.3.6" and spec["cumm-cu118"] == "0.4.11"
    assert "not yet verified" in spec["status"]


def test_falsey_malformed_context_is_not_silently_replaced(cloud):
    with pytest.raises(ValueError, match="context"):
        c.create_request(
            cloud,
            request_id="request",
            experiment_id="test",
            config_sha256="a" * 64,
            preprocessing_sha256=c.digest([]),
            submitted_at_utc="2026-11-26T00:00:00+00:00",
            context=[],
        )


def test_uncertain_submission_and_failed_cancellation_never_start_second_call(monkeypatch, cloud):
    monkeypatch.setenv("RAVEN_ALLOW_MODAL_GPU", "1")
    monkeypatch.setenv("RAVEN_MODAL_ASSET_RIGHTS_CONFIRMED", "1")
    spawn = Mock(side_effect=ConnectionError("private token"))
    lookup = Mock(return_value=lambda: SimpleNamespace(infer=SimpleNamespace(spawn=spawn)))
    monkeypatch.setitem(
        sys.modules, "modal", SimpleNamespace(Cls=SimpleNamespace(from_name=lookup))
    )
    with pytest.raises(RemoteExecutionError, match="call_id=unavailable"):
        ModalTransport(ModalConfig("raven-pointpillars", "main")).submit(make_request(cloud))
    spawn.assert_called_once()
    call = Mock(object_id="fc-test")
    call.get.side_effect = TimeoutError()
    call.cancel.side_effect = ConnectionError("private token")
    spawn.side_effect = None
    spawn.return_value = call
    spawn.reset_mock()
    with pytest.raises(RemoteExecutionError, match="cancellation_requested=False") as failure:
        ModalTransport(ModalConfig("raven-pointpillars", "main")).submit(make_request(cloud))
    assert "private token" not in str(failure.value)
    spawn.assert_called_once()


def test_worker_startup_rejects_wrong_gpu_before_loading_checkpoint(monkeypatch):
    import raven.inference.worker as worker_module
    from raven.perception.detection.pointpillars import PointPillarsDetector

    monkeypatch.setattr(worker_module.platform, "python_version", lambda: "3.10.16")
    monkeypatch.setattr(worker_module.importlib.metadata, "version", lambda name: c.PACKAGES[name])
    torch = SimpleNamespace(
        version=SimpleNamespace(cuda="11.8"),
        cuda=SimpleNamespace(is_available=lambda: True, get_device_capability=lambda i: (8, 9)),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    initialize = Mock()
    monkeypatch.setattr(PointPillarsDetector, "initialize", initialize)
    with pytest.raises(worker_module.WorkerInitializationError, match="runtime_profile"):
        WorkerSession()
    initialize.assert_not_called()


def test_worker_reuses_initialized_model_and_validates_both_config_files(monkeypatch):
    import raven.inference.worker as worker_module
    import raven.perception.detection.pointpillars as adapter

    monkeypatch.setattr(worker_module.platform, "python_version", lambda: "3.10.16")
    monkeypatch.setattr(worker_module.importlib.metadata, "version", lambda name: c.PACKAGES[name])
    torch = SimpleNamespace(
        version=SimpleNamespace(cuda="11.8"),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            get_device_capability=lambda i: (7, 5),
            get_device_name=lambda i: "Tesla T4",
            synchronize=Mock(),
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    checked_source = {
        "openpcdet_commit": c.REVISION,
        "openpcdet_modified_files": ["pcdet/datasets/__init__.py"],
        "openpcdet_diff_sha256": "a" * 64,
        "argo_workaround": {"backup_sha256": "b" * 64, "patched_sha256": "c" * 64},
    }
    validation = Mock(return_value={"source": checked_source, "gpu_execution": "not_performed"})
    monkeypatch.setattr(worker_module, "validate_cpu_environment", validation)
    detector = Mock()
    detector.provenance = {
        "openpcdet_commit": c.REVISION,
        "checkpoint_sha256": c.CHECKPOINT,
        "config": {"sha256": c.MODEL_CONFIG},
        "openpcdet_modified_files": ["pcdet/datasets/__init__.py"],
    }
    factory = Mock(return_value=detector)
    monkeypatch.setattr(adapter, "PointPillarsDetector", factory)
    worker = WorkerSession()
    validation.assert_called_once_with(Path("/opt/OpenPCDet"))
    assert (
        worker.metadata["model_provenance"]["argo_workaround"] == checked_source["argo_workaround"]
    )
    detector.initialize.assert_called_once()
    assert worker.metadata["packages"] == c.PACKAGES
    assert factory.call_args.kwargs["checkpoint_sha256"] == c.CHECKPOINT
    worker.close()
    detector.close.assert_called_once()


def test_transport_cannot_rewrite_request_binding_before_returning(cloud):
    def submit(request):
        request["experiment_id"] = "rewritten-experiment"
        request["request_sha256"] = c.digest(
            {k: v for k, v in c.request_metadata(request).items() if k != "request_sha256"}
        )
        return make_response(request), {
            "provider": "modal",
            "function_call_id": "fc-test",
            "call_id": "fc-test",
            "submission_attempts": 1,
        }

    detector = RemoteDetector(
        SimpleNamespace(submit=submit),
        experiment_id="synthetic",
        config_sha256="a" * 64,
        preprocessing_sha256=c.digest([]),
    )
    with pytest.raises(ValueError, match="request_sha256 mismatch"):
        detector.detect(cloud)


def test_missing_invocation_identity_cannot_be_persisted_as_success(cloud):
    detector = RemoteDetector(
        SimpleNamespace(submit=lambda r: (make_response(r), {})),
        experiment_id="synthetic",
        config_sha256="a" * 64,
        preprocessing_sha256=c.digest([]),
    )
    with pytest.raises(ValueError, match="invocation identity"):
        detector.detect(cloud)


def test_explicit_argo_patch_is_guarded_preserves_backup_and_changes_only_registration(
    tmp_path, monkeypatch
):
    from raven.inference import patch_argo

    path = tmp_path / "pcdet/datasets/__init__.py"
    path.parent.mkdir(parents=True)
    raw = b"from .argo2.argo2_dataset import Argo2Dataset\nKEEP = True\n__all__ = {\n    'Argo2Dataset': Argo2Dataset\n}\n"
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="preimage"):
        patch_argo.patch(tmp_path)
    assert path.read_bytes() == raw
    monkeypatch.setattr(patch_argo, "PREIMAGE", hashlib.sha256(raw).hexdigest())
    patch_argo.patch(tmp_path)
    assert path.read_bytes() == b"KEEP = True\n__all__ = {\n}\n"
    assert path.with_suffix(".py.bak").read_bytes() == raw


def test_deployment_policy_is_bounded_and_local_authorization_does_not_break_container_import(
    monkeypatch,
):
    import runpy

    from raven.inference import modal_image

    captured = {}

    class App:
        def __init__(self, name):
            self.name = name

        def cls(self, **kwargs):
            captured.update(kwargs)
            return lambda cls: cls

    local = True
    fake = SimpleNamespace(
        App=App,
        is_local=lambda: local,
        Volume=SimpleNamespace(from_name=lambda name, **kwargs: (name, kwargs)),
        enter=lambda: lambda f: f,
        method=lambda: lambda f: f,
        exit=lambda: lambda f: f,
    )
    monkeypatch.setitem(sys.modules, "modal", fake)
    monkeypatch.setattr(modal_image, "build_image", lambda: "synthetic-image-graph")
    monkeypatch.delenv("RAVEN_ALLOW_MODAL_GPU", raising=False)
    monkeypatch.delenv("RAVEN_MODAL_ASSET_RIGHTS_CONFIRMED", raising=False)
    monkeypatch.delenv("RAVEN_MODAL_TRUSTED_LOADING_SECRET", raising=False)
    path = Path(__file__).resolve().parents[1] / "providers/modal/app.py"
    with pytest.raises(RuntimeError, match="RAVEN_ALLOW_MODAL_GPU"):
        runpy.run_path(str(path))
    assert not captured
    # Container runtime imports cannot depend on local-only approval environment variables.
    local = False
    runpy.run_path(str(path))
    assert captured["gpu"] == "T4"
    assert captured["max_containers"] == 1 and captured["min_containers"] == 0
    assert captured["buffer_containers"] == 0 and captured["scaledown_window"] == 2
    assert captured["timeout"] == 180 and captured["startup_timeout"] == 300
    assert captured["retries"] == 0 and captured["secrets"] == []
    assert captured["volumes"]["/weights"] == (
        "raven-pointpillars-weights",
        {"create_if_missing": False},
    )


def test_neutral_detector_accepts_another_explicit_provider_profile_without_modal_api(cloud):
    box = Detection3D((20, 0, 0), (4, 2, 1.5), 0, "Car", 0.9, "kitti_velodyne")

    def submit(request):
        return {"request_id": request["request_id"], "worker": {"provider": "another-provider"}}, {
            "provider": "another-provider",
            "call_id": "opaque-job-1",
            "submission_attempts": 1,
        }

    validator = Mock(return_value=[box])
    detector = RemoteDetector(
        SimpleNamespace(submit=submit),
        experiment_id="synthetic",
        config_sha256="a" * 64,
        preprocessing_sha256=c.digest([]),
        response_validator=validator,
    )
    assert detector.detect(cloud) == [box]
    validator.assert_called_once()
    assert detector.provenance["transport"]["provider"] == "another-provider"
