"""CPU-only M5 integration. All remote execution in this file is synthetic."""

import csv
import json
import subprocess
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
from test_remote_inference import SyntheticTransport
from test_remote_inference import config as config  # noqa: PLC0414 -- Imported fixture.

from raven.benchmark.experiment import benchmark_experiment
from raven.benchmark.gpu import LocalCudaBackend
from raven.benchmark.spec import BenchmarkSpec
from raven.cli import main
from raven.evaluation.evaluate import evaluate_experiment, save_evaluation_result
from raven.evaluation.inputs import load_completed_run
from raven.experiments import runner
from raven.exporting.evidence import Snapshot, load_evidence
from raven.exporting.export import export_results
from raven.inference import contracts
from raven.inference.evidence import validate_archived_remote_run, validate_remote_evidence
from raven.reproducibility.verify import verify_results


@pytest.fixture
def remote(config, monkeypatch):
    transport = SyntheticTransport()
    monkeypatch.setattr(
        "raven.inference.modal_transport.modal_preflight",
        lambda: {"source": "modal", "synthetic_test": True},
    )
    monkeypatch.setattr(
        "raven.inference.modal_transport.ModalTransport", lambda settings: transport
    )
    return config, transport


def execute(config):
    result = runner.run_experiment(config)
    assert result.status == "success", result.failures
    return config.run_directory / "modal.manifest.json"


def test_synthetic_remote_runner_export_evaluation_and_cpu_replay(remote, tmp_path, monkeypatch):
    config, transport = remote
    manifest = execute(config)
    run = load_completed_run(manifest)
    frame = run.results["frames"][0]
    metadata = frame["prediction_source"]["model_provenance"]
    assert metadata["synthetic_test"] is True
    assert metadata["response"]["model"]["checkpoint_sha256"] == contracts.CHECKPOINT
    assert frame["processed_xyzi_sha256"] == metadata["request"]["input"]["sha256"]
    assert (
        frame["client_end_to_end_timing"]["seconds"]
        >= frame["client_preprocessing_timing"]["seconds"]
    )
    assert frame["client_end_to_end_timing"]["clock_domain"] == "client_process"
    assert metadata["timing_domains"]["response.timings"] == "worker_process"
    assert validate_remote_evidence(metadata, frame, config)
    evaluation = evaluate_experiment(manifest)
    assert evaluation.status == "blocked" and not evaluation.metrics
    assert any("mocked inference" in reason for reason in evaluation.failures)
    evaluation_path = tmp_path / "evaluation.json"
    save_evaluation_result(evaluation_path, evaluation)
    evidence = load_evidence(manifest, Snapshot())
    assert evidence.evidence_kind == "synthetic_test"
    assert evidence.model["declared"]["source"] == "modal"
    assert "checkpoint" not in evidence.model["declared"]
    assert evidence.model["observed"][0]["response"] == metadata["response"]
    destination = tmp_path / "export"
    export_results([manifest, evaluation_path], destination, format="all")
    with (destination / "experiment_summary.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert all(row["evidence_kind"] == "synthetic_test" for row in rows)
    assert all(json.loads(row["model_json"])["declared"]["source"] == "modal" for row in rows)
    with (destination / "evaluation_metrics.csv").open(newline="") as stream:
        metrics = list(csv.DictReader(stream))
    assert metrics == []  # Existing exporter emits no invented metric rows for blocked evaluation.
    blocked_row = next(row for row in rows if row["source_type"] == "evaluation")
    assert blocked_row["source_status"] == "blocked"
    assert "mocked inference" in json.loads(blocked_row["unavailable_json"])["evaluation"]
    calls = len(transport.requests)
    # Replay and verification cannot invoke the transport, even with cloud flags absent.
    monkeypatch.setattr(transport, "submit", lambda _: pytest.fail("unexpected cloud submission"))
    monkeypatch.delenv("RAVEN_ALLOW_MODAL_GPU", raising=False)
    monkeypatch.delenv("RAVEN_MODAL_ASSET_RIGHTS_CONFIRMED", raising=False)
    replay = verify_results(manifest, replay_preprocessing=True)
    assert replay.status == "checks_passed", replay.failures
    assert replay.checks["actual_model_inference_reproduction"]["status"] == "not_performed"
    assert replay.checks["cross_environment_reproducibility"]["status"] == "not_established"
    assert len(transport.requests) == calls
    original_read = Path.read_bytes

    def offline_read(path):
        if config.dataset.root in path.parents or config.run_directory in path.parents:
            pytest.fail(f"offline verifier accessed original input/source: {path}")
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", offline_read)
    offline = verify_results(destination / "manifest.json")
    assert offline.status == "integrity_only", offline.failures
    assert offline.sources[0]["evidence_kind"] == "synthetic_test"


def test_independent_remote_invocations_compare_science_without_erasing_request_binding(
    remote, tmp_path
):
    config, transport = remote
    left = execute(config)
    right_config = replace(
        config, output=replace(config.output, directory=tmp_path / "second-runs")
    )
    right = execute(right_config)
    assert transport.requests[0]["request_id"] != transport.requests[1]["request_id"]
    assert transport.requests[0]["request_sha256"] != transport.requests[1]["request_sha256"]
    assert config.config_sha256 != right_config.config_sha256
    report = verify_results(left, compare=right, replay_preprocessing=True)
    assert report.status == "checks_passed", report.to_dict()
    assert report.checks["deterministic_comparison"]["status"] == "matched"
    assert any("request_id" in item["path"] for item in report.environment_differences)
    for path, name in [(left, "a"), (right, "b")]:
        export_results([path], tmp_path / name, format="all")
    assert (
        verify_results(tmp_path / "a/manifest.json", compare=tmp_path / "b/manifest.json").status
        == "checks_passed"
    )


@pytest.mark.parametrize(
    "mutation", ["input", "config", "model", "transport", "client_time", "synthetic"]
)
def test_export_and_verification_reject_rechecksummed_invalid_remote_provenance(remote, mutation):
    config, _ = remote
    manifest = execute(config)
    directory = manifest.parent
    assert config.dataset.root.parent in directory.parents
    document = json.loads(manifest.read_text())
    result_path = directory / document["artifacts"]["results"]["file"]
    results = json.loads(result_path.read_text())
    frame = results["frames"][0]
    metadata = frame["prediction_source"]["model_provenance"]
    if mutation == "input":
        frame["processed_xyzi_sha256"] = "a" * 64
    elif mutation == "config":
        request = metadata["request"]
        request["experiment_config_sha256"] = "a" * 64
        request["request_sha256"] = contracts.digest(
            {k: v for k, v in request.items() if k != "request_sha256"}
        )
        metadata["response"]["request_sha256"] = request["request_sha256"]
        metadata["response"]["response_sha256"] = contracts.response_digest(metadata["response"])
    elif mutation == "model":
        metadata["response"]["model"]["checkpoint_sha256"] = "a" * 64
        metadata["response"]["response_sha256"] = contracts.response_digest(metadata["response"])
    elif mutation == "transport":
        metadata["transport"]["call_id"] = "fc-other"
    elif mutation == "client_time":
        metadata["client_timings"]["serialization_seconds"] = -1
    else:
        metadata["synthetic_test"] = False
    result_path.write_text(json.dumps(results))
    import hashlib

    document["artifacts"]["results"]["sha256"] = hashlib.sha256(
        result_path.read_bytes()
    ).hexdigest()
    manifest.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        export_results([manifest], directory / "invalid-export")
    report = verify_results(manifest)
    assert report.status == "failed"


@pytest.mark.parametrize("mutation", ["kind", "response", "processed", "configuration"])
def test_embedded_remote_archive_revalidates_contract_without_original_paths(remote, mutation):
    config, _ = remote
    record = deepcopy(load_evidence(execute(config), Snapshot()))
    if mutation == "kind":
        record = replace(record, evidence_kind="declared_remote_modal_inference")
    elif mutation == "configuration":
        record.configuration["seed"] = 43
    else:
        frame = record.payload["results"]["frames"][0]
        if mutation == "response":
            response = frame["prediction_source"]["model_provenance"]["response"]
            response["input_sha256"] = "a" * 64
            response["response_sha256"] = contracts.response_digest(response)
        else:
            frame["processed_xyzi_sha256"] = "a" * 64
    with pytest.raises(ValueError):
        validate_archived_remote_run(record)


def test_modal_is_never_a_local_cuda_benchmark_but_cpu_preprocessing_remains_available(
    remote, tmp_path, monkeypatch
):
    config, transport = remote

    def no_backend(*args, **kwargs):
        pytest.fail("CPU preprocessing initialized remote/CUDA backend")

    monkeypatch.setattr("raven.benchmark.experiment.LocalCudaBackend", no_backend)
    cpu = benchmark_experiment(config, specification=BenchmarkSpec(warmup=0, repetitions=2))
    assert cpu.status == "success" and cpu.evidence_kind == "local_cpu"
    assert not transport.requests
    assert (
        verify_results(
            Path(cpu.output_directory) / "manifest.json", replay_preprocessing=True
        ).status
        == "checks_passed"
    )
    with pytest.raises(RuntimeError, match="local CUDA"):
        LocalCudaBackend(config)
    monkeypatch.undo()
    blocked_config = replace(
        config, output=replace(config.output, directory=tmp_path / "blocked-cuda")
    )
    blocked = benchmark_experiment(
        blocked_config, specification=BenchmarkSpec(mode="inference", warmup=0, repetitions=1)
    )
    assert blocked.status == "blocked" and blocked.evidence_kind == "unavailable"
    assert "Modal" in blocked.failures[0]["message"]
    assert not transport.requests


@pytest.mark.parametrize("failure", [TimeoutError("timeout"), RuntimeError("transport failed")])
def test_m5_remote_failures_do_not_publish_success_or_fall_back(remote, monkeypatch, failure):
    config, transport = remote

    def fail(request):
        transport.requests.append(request)
        raise failure

    monkeypatch.setattr(transport, "submit", fail)
    result = runner.run_experiment(config)
    assert result.status == "failed" and len(transport.requests) == 1
    manifest = config.run_directory / "modal.manifest.json"
    assert verify_results(manifest).status == "failed"
    assert evaluate_experiment(manifest).status == "invalid"


def test_m5_cpu_commands_import_without_modal_gpu_or_gui(tmp_path):
    script = """
import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in {'modal','torch','pcdet','spconv','cumm','open3d'}:
   raise AssertionError('forbidden import: '+fullname)
sys.meta_path.insert(0,Block())
from raven.cli import main
for command in ('inspect-experiment','run-experiment','evaluate-experiment','benchmark-experiment','export-results','verify-results'):
 try: main([command,'--help'])
 except SystemExit as e: assert e.code==0
from raven.experiments.config import load_experiment_config
cfg=load_experiment_config('configs/experiments/modal-pointpillars.json')
assert cfg.schema_version==2
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)


def test_verify_cli_accepts_synthetic_remote_source_without_enabling_cloud(remote, capsys):
    config, _ = remote
    manifest = execute(config)
    main(["verify-results", str(manifest), "--replay-preprocessing"])
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "checks_passed"
    assert report["sources"][0]["evidence_kind"] == "synthetic_test"


@pytest.mark.parametrize("mutation", ["input", "worker", "coordinate"])
def test_invalid_remote_reply_never_publishes_a_successful_run(remote, monkeypatch, mutation):
    config, transport = remote
    original_submit = transport.submit

    def invalid(request):
        response, metadata = original_submit(request)
        if mutation == "input":
            response["input_sha256"] = "a" * 64
        elif mutation == "worker":
            response["worker"]["model_provenance"]["checkpoint_sha256"] = "a" * 64
        else:
            response["predictions"]["coordinate_frame"] = "kitti_camera"
        response["response_sha256"] = contracts.response_digest(response)
        return response, metadata

    monkeypatch.setattr(transport, "submit", invalid)
    result = runner.run_experiment(config)
    assert result.status == "failed" and len(transport.requests) == 1
    manifest = config.run_directory / "modal.manifest.json"
    assert json.loads(manifest.read_text())["status"] == "failed"
    assert verify_results(manifest).status == "failed"
    with pytest.raises(ValueError):
        export_results([manifest], config.output.directory / "invalid-response-export")
