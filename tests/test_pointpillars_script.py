import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest


@pytest.fixture
def runner():
    path = Path(__file__).resolve().parents[1] / "scripts/kaggle_pointpillars.py"
    spec = importlib.util.spec_from_file_location("kaggle_pointpillars", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def scan(tmp_path):
    path = tmp_path / "000000.bin"
    np.array([[1, 2, 3, 0.5], [4, 5, 6, 0.7]], dtype="<f4").tofile(path)
    return path


def predictions():
    return np.array([[1.0, 2, 3, 4, 2, 1, 0.5]]), np.array([0.9]), np.array([2])


def test_input_validation_and_checksum(runner, scan):
    original = scan.read_bytes()
    digest = runner.sha256_file(scan)
    cloud, manifest = runner.validate_input(scan, digest)
    assert cloud.count == 2 and manifest["sha256"] == digest
    assert manifest["coordinate_frame"] == "kitti_velodyne"
    assert scan.read_bytes() == original
    with pytest.raises(ValueError, match="Input SHA256 mismatch"):
        runner.validate_input(scan, "0" * 64)


@pytest.mark.parametrize(
    "contents", [b"", b"123", np.array([[np.nan, 0, 0, 0]], dtype="<f4").tobytes()]
)
def test_bad_inputs_rejected(runner, scan, contents):
    scan.write_bytes(contents)
    with pytest.raises(ValueError):
        runner.validate_input(scan)


def test_missing_input(runner, tmp_path):
    with pytest.raises(FileNotFoundError):
        runner.validate_input(tmp_path / "missing.bin")


def test_serialization_class_mapping_and_no_accuracy_claim(runner):
    result = runner.serialize_predictions(*predictions(), ["Car", "Pedestrian", "Cyclist"])
    assert result["total_predictions"] == 1 and result["class_counts"] == {"Pedestrian": 1}
    assert result["predictions"][0]["class_name"] == "Pedestrian"
    assert result["predictions"][0]["label_id"] == 2
    assert result["box_format"] == ["x", "y", "z", "dx", "dy", "dz", "heading"]
    assert result["position_reference"] == "geometric_center"
    assert result["coordinate_frame"] == "kitti_velodyne"
    assert "accuracy" not in result and "ground_truth" not in result
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_empty_predictions(runner):
    result = runner.serialize_predictions(np.empty((0, 7)), np.empty(0), np.empty(0), ["Car"])
    assert result["total_predictions"] == 0
    assert result["predictions"] == [] and result["class_counts"] == {}


@pytest.mark.parametrize(
    "field,value",
    [
        (0, np.zeros((1, 6))),
        (0, np.array([[1, 2, 3, 0, 2, 1, 0]])),
        (0, np.array([[np.nan, 2, 3, 4, 2, 1, 0]])),
        (1, np.array([1.2])),
        (1, np.array([np.inf])),
        (1, np.array([0.5, 0.4])),
        (2, np.array([0])),
        (2, np.array([4])),
        (2, np.array([1.5])),
        (2, np.array(["Car"])),
        (2, np.array([True])),
    ],
)
def test_bad_prediction_arrays(runner, field, value):
    arrays = list(predictions())
    arrays[field] = value
    with pytest.raises(ValueError):
        runner.serialize_predictions(*arrays, ["Car", "Pedestrian", "Cyclist"])


@pytest.mark.parametrize("names", [[], ["Car", "Car"], [""], [1]])
def test_invalid_class_names(runner, names):
    with pytest.raises(ValueError):
        runner.serialize_predictions(*predictions(), names)


def test_cli_help_has_no_gpu_import(runner, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "pcdet", None)
    with pytest.raises(SystemExit) as exc:
        runner.main(["--help"])
    assert exc.value.code == 0
    assert "--validate-input-only" in capsys.readouterr().out


def test_cli_validate_only_without_gpu(runner, scan, monkeypatch, capsys, tmp_path):
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "pcdet", None)
    backend = Mock()
    monkeypatch.setattr(runner, "run_inference", backend)
    assert runner.main(["--input", str(scan), "--validate-input-only"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["gpu_inference_executed"] is False
    assert result["input"]["points"] == 2
    backend.assert_not_called()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["000000.bin"]


@pytest.mark.parametrize(
    "flags,code,message",
    [
        ([], 1, "Inference requires"),
        (["--input-sha256", "0" * 64, "--validate-input-only"], 1, "Input SHA256 mismatch"),
        (["--input-sha256", "bad"], 2, "64 hexadecimal"),
    ],
)
def test_cli_invalid_arguments(runner, scan, flags, code, message, capsys):
    with pytest.raises(SystemExit) as exc:
        runner.main(["--input", str(scan), *flags])
    assert exc.value.code == code
    assert message in capsys.readouterr().err


@pytest.fixture
def inference_args(runner, scan, tmp_path):
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"synthetic checkpoint bytes for CPU-only validation")
    config = tmp_path / "pointpillar.yaml"
    config.write_text("synthetic config for mocked backend")
    checkout = tmp_path / "OpenPCDet"
    (checkout / "tools").mkdir(parents=True)
    return [
        "--input",
        str(scan),
        "--checkpoint",
        str(checkpoint),
        "--checkpoint-sha256",
        runner.sha256_file(checkpoint),
        "--checkpoint-source",
        "synthetic-test-fixture",
        "--config",
        str(config),
        "--openpcdet-root",
        str(checkout),
        "--output",
        str(tmp_path / "predictions.json"),
    ]


def test_cli_serializes_mocked_execution_without_overwriting(
    runner, inference_args, monkeypatch, capsys
):
    backend = Mock(return_value=(predictions(), ["Car", "Pedestrian", "Cyclist"], {"mocked": True}))
    monkeypatch.setattr(runner, "run_inference", backend)
    assert runner.main(inference_args) == 0
    output = Path(inference_args[-1])
    document = json.loads(output.read_text())
    assert document["mocked"] is True
    assert document["class_counts"] == {"Pedestrian": 1}
    assert document["checkpoint"]["source_status"] == "user_supplied"
    assert document["config"]["sha256"]
    original = output.read_bytes()
    with pytest.raises(SystemExit) as exc:
        runner.main(inference_args)
    assert exc.value.code == 1
    assert "Refusing to overwrite" in capsys.readouterr().err
    assert output.read_bytes() == original and backend.call_count == 1


def test_checkpoint_hash_checked_before_backend(runner, inference_args, monkeypatch, capsys):
    backend = Mock()
    monkeypatch.setattr(runner, "run_inference", backend)
    inference_args[inference_args.index("--checkpoint-sha256") + 1] = "0" * 64
    with pytest.raises(SystemExit) as exc:
        runner.main(inference_args)
    assert exc.value.code == 1
    assert "Checkpoint SHA256 mismatch" in capsys.readouterr().err
    backend.assert_not_called()


def test_backend_failure_does_not_write_predictions(runner, inference_args, monkeypatch):
    monkeypatch.setattr(runner, "run_inference", Mock(side_effect=RuntimeError("no CUDA")))
    with pytest.raises(SystemExit):
        runner.main(inference_args)
    assert not Path(inference_args[-1]).exists()


def test_flags_before_import_and_missing_dependency_error(runner, gpu_backend, monkeypatch):
    import os

    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(RuntimeError, match="GPU dependencies are unavailable"):
        runner.run_inference(
            SimpleNamespace(**gpu_backend.options, trust_checkpoint=False), gpu_backend.cloud
        )
    assert os.environ["CUMM_DISABLE_JIT"] == "1"
    assert os.environ["SPCONV_DISABLE_JIT"] == "1"
    assert os.environ["TORCH_CUDA_ARCH_LIST"] == "7.5"


@pytest.mark.parametrize("trust,load_error", [(False, False), (True, False), (False, True)])
def test_headless_backend_contract_with_fake_gpu_modules(runner, gpu_backend, trust, load_error):
    import pickle

    from raven.perception.detection._openpcdet import PINNED_REVISION

    fake = gpu_backend
    args = SimpleNamespace(**fake.options, trust_checkpoint=trust)
    previous = Path.cwd()
    if load_error:
        fake.loader.side_effect = pickle.UnpicklingError("synthetic unsupported metadata")
        with pytest.raises(RuntimeError, match="--trust-checkpoint"):
            runner.run_inference(args, fake.cloud)
        fake.loader.assert_called_once()
        fake.model.forward.assert_not_called()
    else:
        arrays, names, manifest = runner.run_inference(args, fake.cloud)
        assert names == ["Car", "Pedestrian", "Cyclist"]
        np.testing.assert_array_equal(arrays[0], fake.raw[0])
        fake.model.load_state_dict.assert_called_once_with({"synthetic_weight": 1}, strict=True)
        assert not np.shares_memory(
            fake.dataset.prepare_data.call_args.args[0]["points"], fake.cloud.points
        )
        assert manifest["openpcdet_commit"] == PINNED_REVISION
        assert manifest["environment"]["device_index"] == 0
        assert manifest["environment"]["environment_flags"]["SPCONV_DISABLE_JIT"] == "1"
        assert manifest["effective_config"]["ROOT_DIR"] == str(fake.root)
    assert fake.loader.call_args.kwargs == {"map_location": "cpu", "weights_only": not trust}
    assert Path.cwd() == previous


def test_adapter_cli_invokes_detect_and_serializes_validated_outputs(
    runner, gpu_backend, tmp_path, monkeypatch, capsys
):
    from raven.perception.detection.pointpillars import PointPillarsDetector

    fake = gpu_backend
    scan = tmp_path / "000000.bin"
    fake.cloud.points.tofile(scan)
    original = PointPillarsDetector.detect
    calls = []

    def detect(self, cloud):
        boxes = original(self, cloud)
        calls.append(boxes)
        return boxes

    monkeypatch.setattr(PointPillarsDetector, "detect", detect)
    output = tmp_path / "adapter_predictions.json"
    assert (
        runner.main(
            [
                "--adapter",
                "--input",
                str(scan),
                "--checkpoint",
                str(fake.options["checkpoint"]),
                "--checkpoint-sha256",
                fake.options["checkpoint_sha256"],
                "--checkpoint-source",
                "synthetic-test-fixture",
                "--config",
                str(fake.options["config"]),
                "--openpcdet-root",
                str(fake.root),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    document = json.loads(output.read_text())
    assert len(calls) == 1 and len(calls[0]) == 2
    assert document["inference_entrypoint"] == "PointPillarsDetector.detect"
    assert document["coordinate_frame"] == "kitti_velodyne"
    assert document["detections"][0]["coordinate_frame"] == "kitti_velodyne"
    assert document["detections"][0]["rotation_axis"] == "z"
    assert document["detections"][0]["center_xyz"] == [1, 2, 3]
    assert document["class_counts"] == {"Car": 1, "Pedestrian": 1}
    assert document["predictions"][0]["box"] == list(fake.raw[0][0])
    assert document["checkpoint"]["sha256"] == fake.options["checkpoint_sha256"]
    assert document["openpcdet_commit"] and document["effective_config"]
    assert "output" in capsys.readouterr().out
