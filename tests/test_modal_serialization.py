"""CPU-only source/transport regressions. All inference fixtures are synthetic."""

import json
import pickle
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from test_remote_inference import config as config  # noqa: PLC0414 -- Imported fixture.
from test_remote_inference import make_request, make_response

from raven.evaluation.evaluate import evaluate_experiment
from raven.evaluation.inputs import load_completed_run
from raven.experiments import runner
from raven.exporting.export import export_results
from raven.inference import contracts as c
from raven.inference.json_native import normalize_json_containers, validate_json_native
from raven.inference.worker import WorkerSession
from raven.perception.detection import _openpcdet as backend
from raven.perception.detection.pointpillars import PointPillarsDetector
from raven.reproducibility.verify import verify_results

# gpu_backend mocks subprocess.run for upstream Git. Keep a genuine subprocess runner.
REAL_RUN = subprocess.run


class TorchVersionLike(str):
    """Mimics torch.torch_version.TorchVersion's cross-process pickle dependency."""


class FloatSubtype(float):
    pass


class IntSubtype(int):
    pass


class ListSubtype(list):
    pass


class DictSubtype(dict):
    pass


@pytest.mark.parametrize(
    "bad",
    [
        TorchVersionLike("2.2.2+cu118"),
        FloatSubtype(1),
        IntSubtype(1),
        ListSubtype([1]),
        DictSubtype(x=1),
        np.float64(1),
        np.float32(1),
        np.int64(1),
        np.bool_(True),
        np.array([1]),
        Path("version"),
        b"bytes",
        {1, 2},
        object(),
        (1, 2),
    ],
)
def test_strict_json_rejects_every_non_native_nested_value_with_path(bad):
    with pytest.raises(TypeError, match=r'\$\["worker"\]\["versions"\]\[0\]'):
        validate_json_native({"worker": {"versions": [bad]}})


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_strict_json_rejects_nonfinite_values_anywhere(bad):
    with pytest.raises(ValueError, match="finite"):
        validate_json_native({"worker": {"extra": [bad]}})


def test_strict_json_native_accepts_exact_types_and_shared_but_acyclic_containers():
    row = [None, True, False, 1, 0, -4, 1.25, "Ž"]
    value = {"left": row, "right": row}
    validate_json_native(value)
    with pytest.raises(TypeError, match="keys"):
        validate_json_native({TorchVersionLike("version"): "value"})
    cyclic = []
    cyclic.append(cyclic)
    with pytest.raises(ValueError, match="cyclic"):
        validate_json_native(cyclic)
    deep = 0
    for _ in range(66):
        deep = [deep]
    with pytest.raises(ValueError, match="nesting"):
        validate_json_native(deep)


def test_normalizer_only_converts_structural_tuples_and_preserves_json_hashes():
    original = {"geometry": [(1.25, 2, 3)], "values": (None, True, "version")}
    result = normalize_json_containers(original)
    assert result == {"geometry": [[1.25, 2, 3]], "values": [None, True, "version"]}
    validate_json_native(result)
    assert c.canonical(original) == c.canonical(result) and c.digest(original) == c.digest(result)
    result["geometry"][0][0] = 99
    assert original["geometry"][0][0] == 1.25
    with pytest.raises(TypeError, match="TorchVersionLike"):
        normalize_json_containers({"version": TorchVersionLike("x")})


def test_validator_does_not_call_subclass_pickle_or_stringification_hooks():
    class BadVersion(str):
        def __reduce__(self):
            raise AssertionError("pickle hook must not execute")

        def __str__(self):
            raise AssertionError("implicit stringification must not execute")

    with pytest.raises(TypeError, match="BadVersion"):
        normalize_json_containers({"bad": BadVersion("x")})


def test_plain_config_normalizes_known_scalar_subtypes_without_materializing_tensors():
    source = {
        "versions": [TorchVersionLike("v")],
        "floats": (np.float64(0.2), np.float32(0.3), FloatSubtype(0.4)),
        "int": np.int64(7),
        "bool": np.bool_(True),
        "path": Path("/synthetic"),
    }
    result = backend.plain_config(source)
    validate_json_native(result)
    assert result["floats"] == [0.2, float(np.float32(0.3)), 0.4]
    assert result["versions"] == ["v"] and result["int"] == 7 and result["bool"] is True
    for bad in [np.array([1]), object(), np.float64("nan")]:
        with pytest.raises(ValueError):
            backend.plain_config({"bad": bad})


def test_package_version_normalizes_distribution_version_subtypes(monkeypatch):
    monkeypatch.setattr(backend.importlib.metadata, "version", lambda name: TorchVersionLike("1.2"))
    assert type(backend.package_version("synthetic-package")) is str
    assert backend.package_version("synthetic-package") == "1.2"


@pytest.fixture
def source_provenance(gpu_backend, monkeypatch):
    gpu_backend.torch.__version__ = TorchVersionLike("2.2.2+cu118")
    gpu_backend.torch.version.cuda = TorchVersionLike("11.8")
    sys.modules["pcdet"].__version__ = TorchVersionLike("0.6.0+233f849")
    gpu_backend.cfg["extra_provenance_fixture"] = {
        "version": TorchVersionLike("nested"),
        "number": np.float64(0.25),
    }
    detector = PointPillarsDetector(**gpu_backend.options)
    detector.initialize()
    provenance = detector.provenance
    for key in ["pytorch", "torch_cuda_runtime", "openpcdet"]:
        assert type(provenance["environment"][key]) is str
    validate_json_native(provenance)
    # Synthetic profile declaration; not these fixture bytes' genuine checkpoint/config identities.
    provenance["checkpoint_sha256"] = c.CHECKPOINT
    provenance["config"]["sha256"] = c.MODEL_CONFIG
    return provenance


def worker_fixture(request, metadata, monkeypatch):
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(synchronize=Mock()))
    )
    worker = WorkerSession.__new__(WorkerSession)
    worker.metadata = metadata
    worker.load_seconds = 0.2
    worker.cache = {}
    worker.detector = Mock()
    worker.detector.detect.return_value = c.validate_response(make_response(request), request)
    return worker


def test_real_source_version_fix_and_full_worker_pickle_round_trip_in_torchless_process(
    source_provenance, gpu_backend, monkeypatch, tmp_path
):
    request = make_request(gpu_backend.cloud)
    metadata = make_response(request)["worker"]
    metadata["model_provenance"] = source_provenance
    worker = worker_fixture(request, metadata, monkeypatch)
    response = worker.infer(request)
    validate_json_native(response)
    assert type(response["predictions"]["detections"][0]["center_xyz"]) is list
    assert response["response_sha256"] == c.response_digest(response)
    good = tmp_path / "native.pkl"
    good.write_bytes(pickle.dumps((request, response)))
    bad = tmp_path / "old-subtype.pkl"
    bad.write_bytes(pickle.dumps({"pytorch": TorchVersionLike("2.2.2+cu118")}))
    script = """
import importlib.abc,json,pickle,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in {'torch','pcdet','spconv','cumm','open3d','modal','test_modal_serialization'}:
   raise AssertionError('forbidden import: '+fullname)
sys.meta_path.insert(0,Block())
with open(sys.argv[1],'rb') as stream: value=pickle.load(stream)
from raven.inference.contracts import validate_response
from raven.inference.json_native import validate_json_native
request,response=value
validate_json_native(response)
boxes=validate_response(response,request)
assert boxes and 'torch' not in sys.modules
print(json.dumps({'boxes':len(boxes),'response_sha256':response['response_sha256']}))
"""
    result = REAL_RUN(
        [sys.executable, "-c", script, str(good)], check=True, capture_output=True, text=True
    )
    assert json.loads(result.stdout)["response_sha256"] == response["response_sha256"]
    failure = REAL_RUN([sys.executable, "-c", script, str(bad)], capture_output=True, text=True)
    assert (
        failure.returncode != 0 and "forbidden import: test_modal_serialization" in failure.stderr
    )


@pytest.mark.parametrize("bad", [TorchVersionLike("v"), np.float64(0.5), float("nan"), object()])
def test_worker_rejects_nested_leaks_before_cache_or_return(monkeypatch, bad):
    from raven.core.point_cloud import PointCloud

    request = make_request(
        PointCloud(np.array([[20, 0, 0, 0.5]], dtype=np.float32), "000001", "kitti_velodyne")
    )
    metadata = make_response(request)["worker"]
    metadata["model_provenance"]["nested"] = {"bad": bad}
    worker = worker_fixture(request, metadata, monkeypatch)
    with pytest.raises((TypeError, ValueError), match="nested"):
        worker.infer(request)
    assert worker.cache == {}


def test_incoming_and_cached_response_validation_remains_strict(monkeypatch):
    from raven.core.point_cloud import PointCloud

    request = make_request(
        PointCloud(np.array([[20, 0, 0, 0.5]], dtype=np.float32), "000001", "kitti_velodyne")
    )
    response = make_response(request)
    corrupted = deepcopy(response)
    corrupted["worker"]["model_provenance"]["version"] = TorchVersionLike("v")
    corrupted["response_sha256"] = c.response_digest(corrupted)
    with pytest.raises(TypeError, match="TorchVersionLike"):
        c.validate_response(corrupted, request)
    historical = deepcopy(response)
    historical["worker"]["raven_source_sha256"] = "a" * 64
    historical["response_sha256"] = c.response_digest(historical)
    with pytest.raises(ValueError, match="source mismatch"):
        c.validate_response(historical, request)
    worker = worker_fixture(request, response["worker"], monkeypatch)
    first = worker.infer(request)
    worker.cache[request["request_id"]][1]["worker"]["extra"] = IntSubtype(3)
    with pytest.raises(TypeError, match="IntSubtype"):
        worker.infer(request)
    assert first["response_sha256"] == c.response_digest(first)
    worker.detector.detect.assert_called_once()


def test_complete_synthetic_response_through_real_client_validation_persistence_and_export(
    config, source_provenance, monkeypatch, tmp_path
):
    calls = []

    class Transport:
        def submit(self, request):
            calls.append(request)
            metadata = make_response(request)["worker"]
            metadata["model_provenance"] = source_provenance
            worker = worker_fixture(request, metadata, monkeypatch)
            # Exercise the SDK's pickle boundary, not just a direct dict return.
            response = pickle.loads(pickle.dumps(worker.infer(request)))
            return response, {
                "provider": "modal",
                "synthetic_test": True,
                "function_call_id": "fc-json-fixture",
                "call_id": "fc-json-fixture",
                "submission_attempts": 1,
                "round_trip_seconds": 0.01,
            }

    monkeypatch.setattr(
        "raven.inference.modal_transport.modal_preflight",
        lambda: {"source": "modal", "synthetic_test": True},
    )
    monkeypatch.setattr(
        "raven.inference.modal_transport.ModalTransport", lambda settings: Transport()
    )
    result = runner.run_experiment(config)
    assert result.status == "success", result.failures
    assert len(calls) == 1
    manifest = config.run_directory / "modal.manifest.json"
    loaded = load_completed_run(manifest)
    frame = loaded.results["frames"][0]
    metadata = frame["prediction_source"]["model_provenance"]
    assert metadata["synthetic_test"] is True
    assert type(metadata["response"]["worker"]["model_provenance"]["environment"]["pytorch"]) is str
    assert metadata["request"]["input"]["sha256"] == frame["processed_xyzi_sha256"]
    c.validate_response_metadata(metadata["response"], metadata["request"])
    validate_json_native(metadata)
    archive = tmp_path / "export"
    export_results([manifest], archive, format="all")
    assert verify_results(archive / "manifest.json").status == "integrity_only"
    assert verify_results(manifest, replay_preprocessing=True).status == "checks_passed"
    evaluation = evaluate_experiment(manifest)
    assert evaluation.status == "blocked" and not evaluation.metrics
    assert any("mocked inference" in reason for reason in evaluation.failures)
