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


def test_flags_before_import_and_missing_dependency_error(runner, scan, monkeypatch):
    for key in ("CUMM_DISABLE_JIT", "SPCONV_DISABLE_JIT", "TORCH_CUDA_ARCH_LIST"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setitem(sys.modules, "torch", None)
    cloud, _ = runner.validate_input(scan)
    with pytest.raises(RuntimeError, match="GPU dependencies are unavailable"):
        runner.run_inference(SimpleNamespace(openpcdet_root=scan.parent), cloud)
    assert runner.os.environ["CUMM_DISABLE_JIT"] == "1"
    assert runner.os.environ["SPCONV_DISABLE_JIT"] == "1"
    assert runner.os.environ["TORCH_CUDA_ARCH_LIST"] == "7.5"


@pytest.mark.parametrize("trust,load_error", [(False, False), (True, False), (False, True)])
def test_headless_backend_contract_with_fake_gpu_modules(
    runner, scan, tmp_path, monkeypatch, trust, load_error
):
    from contextlib import nullcontext

    class Config(dict):
        __getattr__ = dict.__getitem__

    root = tmp_path / "checkout"
    (root / "tools").mkdir(parents=True)
    config = Config(
        MODEL=Config(NAME="PointPillar", POST_PROCESSING=Config(OUTPUT_RAW_SCORE=False)),
        DATA_CONFIG=Config(),
        CLASS_NAMES=["Car", "Pedestrian", "Cyclist"],
        ROOT_DIR=root,
    )
    model = Mock()
    model.cuda.return_value = model
    model.eval.return_value = model
    tensors = [
        SimpleNamespace(
            detach=lambda value=value: SimpleNamespace(
                cpu=lambda: SimpleNamespace(numpy=lambda: value)
            )
        )
        for value in predictions()
    ]
    model.forward.return_value = (
        [dict(zip(("pred_boxes", "pred_scores", "pred_labels"), tensors, strict=True))],
        {},
    )
    dataset = Mock()
    dataset.prepare_data.side_effect = lambda data: data
    dataset.collate_batch.side_effect = lambda samples: {"sample": samples[0]}
    build = Mock(return_value=model)
    loader = Mock(return_value={"model_state": {"synthetic_weight": 1}})
    cuda = SimpleNamespace(
        is_available=lambda: True,
        get_device_name=lambda index: "Fake T4",
        get_device_capability=lambda index: (7, 5),
        device_count=lambda: 2,
    )
    fake_torch = SimpleNamespace(
        cuda=cuda,
        load=loader,
        no_grad=nullcontext,
        __version__="synthetic-test",
        version=SimpleNamespace(cuda="test"),
    )
    configure = Mock(side_effect=lambda *_: None)
    modules = {
        "torch": fake_torch,
        "pcdet": SimpleNamespace(__file__=str(root / "pcdet/__init__.py"), __version__="test"),
        "pcdet.config": SimpleNamespace(cfg=config, cfg_from_yaml_file=configure),
        "pcdet.datasets": SimpleNamespace(DatasetTemplate=Mock(return_value=dataset)),
        "pcdet.models": SimpleNamespace(build_network=build, load_data_to_gpu=Mock()),
        "pcdet.utils": SimpleNamespace(common_utils=SimpleNamespace(create_logger=Mock())),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    for name in ("CUMM_DISABLE_JIT", "SPCONV_DISABLE_JIT", "TORCH_CUDA_ARCH_LIST"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        Mock(
            side_effect=[
                SimpleNamespace(stdout="synthetic-commit\n"),
                SimpleNamespace(stdout="pcdet/datasets/__init__.py\n"),
            ]
        ),
    )
    monkeypatch.setattr(runner, "command_output", lambda _: None)
    monkeypatch.setattr(runner, "package_version", lambda _: None)
    arguments = SimpleNamespace(
        input=scan,
        openpcdet_root=root,
        config=root / "tools/model.yaml",
        checkpoint=tmp_path / "trusted.pth",
        trust_checkpoint=trust,
    )
    cloud, _ = runner.validate_input(scan)
    previous = Path.cwd()
    if load_error:
        import pickle

        loader.side_effect = pickle.UnpicklingError("synthetic unsupported checkpoint metadata")
        with pytest.raises(RuntimeError, match="--trust-checkpoint"):
            runner.run_inference(arguments, cloud)
        loader.assert_called_once()
        model.forward.assert_not_called()
        assert Path.cwd() == previous
        return
    arrays, names, manifest = runner.run_inference(arguments, cloud)
    assert Path.cwd() == previous
    assert names == ["Car", "Pedestrian", "Cyclist"]
    np.testing.assert_array_equal(arrays[0], predictions()[0])
    assert loader.call_args.kwargs == {"map_location": "cpu", "weights_only": not trust}
    model.load_state_dict.assert_called_once_with({"synthetic_weight": 1}, strict=True)
    assert not np.shares_memory(dataset.prepare_data.call_args.args[0]["points"], cloud.points)
    assert manifest["openpcdet_commit"] == "synthetic-commit"
    assert manifest["environment"]["device_index"] == 0
    assert manifest["environment"]["environment_flags"]["SPCONV_DISABLE_JIT"] == "1"
    assert manifest["effective_config"]["ROOT_DIR"] == str(root)
