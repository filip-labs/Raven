import json
import pickle
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from raven.core.point_cloud import PointCloud
from raven.perception.detection import _openpcdet as backend
from raven.perception.detection.base import Detection3D, ObjectDetector
from raven.perception.detection.pointpillars import PointPillarsDetector, convert_predictions


def test_construction_is_lazy_and_multiple_calls_reuse_model(gpu_backend):
    fake = gpu_backend
    detector: ObjectDetector = PointPillarsDetector(**fake.options)
    fake.loader.assert_not_called()
    fake.build.assert_not_called()
    with pytest.raises(RuntimeError, match="Initialize or detect"):
        _ = detector.provenance
    original = fake.cloud.points.copy()

    # Simulate an upstream preprocessor mutating its own point array.
    def prepare(data):
        data["points"][:] = 0
        return data

    fake.dataset.prepare_data.side_effect = prepare
    first = detector.detect(fake.cloud)
    second = detector.detect(fake.cloud)
    assert first == second
    assert all(isinstance(box, Detection3D) for box in first)
    assert [(box.label, box.score) for box in first] == [("Car", 0), ("Pedestrian", 1)]
    assert first[0].center_xyz == (1, 2, 3)
    assert first[0].size_lwh == (4, 2, 1)
    assert first[0].yaw_rad == 0.5
    assert first[0].coordinate_frame == "kitti_velodyne" and first[0].rotation_axis == "z"
    np.testing.assert_array_equal(fake.cloud.points, original)
    assert not fake.cloud.points.flags.writeable
    assert fake.cloud.frame_id == "000000" and fake.cloud.coordinate_frame == "kitti_velodyne"
    fake.loader.assert_called_once()
    fake.build.assert_called_once()
    fake.model.eval.assert_called_once()
    assert fake.model.forward.call_count == 2
    assert first is not second
    assert json.loads(json.dumps([asdict(box) for box in first], allow_nan=False))[0]["score"] == 0


def test_explicit_initialization_device_and_close(gpu_backend):
    fake = gpu_backend
    detector = PointPillarsDetector(**fake.options, device_index=1)
    detector.initialize()
    detector.initialize()
    fake.loader.assert_called_once()
    detector.detect(fake.cloud)
    assert all(call.args == (1,) for call in fake.cuda.device.call_args_list)
    assert fake.selected == []
    assert detector.provenance["environment"]["device_index"] == 1
    metadata = detector.provenance
    metadata["effective_config"]["CLASS_NAMES"].clear()
    assert len(detector.provenance["effective_config"]["CLASS_NAMES"]) == 3
    detector.close()
    detector.close()
    with pytest.raises(RuntimeError, match="closed"):
        detector.detect(fake.cloud)


def test_zero_predictions(gpu_backend):
    gpu_backend.set_predictions([np.empty((0, 7)), np.empty(0), np.empty(0)])
    assert PointPillarsDetector(**gpu_backend.options).detect(gpu_backend.cloud) == []


@pytest.mark.parametrize("frame", ["lidar", "camera", "kitti_camera_rect"])
def test_wrong_frame_fails_before_gpu_loading(gpu_backend, frame):
    detector = PointPillarsDetector(**gpu_backend.options)
    cloud = PointCloud(gpu_backend.cloud.points, "frame", frame)
    with pytest.raises(ValueError, match="kitti_velodyne"):
        detector.detect(cloud)
    gpu_backend.loader.assert_not_called()


def test_empty_cloud_fails_before_gpu_loading(gpu_backend):
    detector = PointPillarsDetector(**gpu_backend.options)
    with pytest.raises(ValueError, match="nonempty"):
        detector.detect(PointCloud(np.empty((0, 4)), "empty", "kitti_velodyne"))
    gpu_backend.loader.assert_not_called()


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("checkpoint", "absent.pth", "existing nonempty"),
        ("config", "absent.yaml", "configuration not found"),
        ("openpcdet_root", "absent-root", "checkout"),
        ("checkpoint_sha256", "invalid", "64 hexadecimal"),
        ("checkpoint_sha256", "0" * 64, "SHA256 mismatch"),
        ("device_index", -1, "nonnegative integer"),
        ("device_index", True, "nonnegative integer"),
        ("trust_checkpoint", "yes", "boolean"),
    ],
)
def test_invalid_options(gpu_backend, field, value, message):
    options = {**gpu_backend.options, field: value}
    with pytest.raises((ValueError, FileNotFoundError, TypeError), match=message):
        PointPillarsDetector(**options)
    gpu_backend.loader.assert_not_called()


def test_checkpoint_changed_after_construction_is_rejected(gpu_backend):
    detector = PointPillarsDetector(**gpu_backend.options)
    gpu_backend.options["checkpoint"].write_bytes(b"changed after validation")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        detector.detect(gpu_backend.cloud)
    gpu_backend.loader.assert_not_called()


@pytest.mark.parametrize("trust", [False, True])
def test_checkpoint_trust_is_explicit_and_never_retried(gpu_backend, trust):
    detector = PointPillarsDetector(**gpu_backend.options, trust_checkpoint=trust)
    gpu_backend.loader.side_effect = pickle.UnpicklingError("synthetic pickle metadata")
    with pytest.raises(RuntimeError, match="No unsafe fallback"):
        detector.detect(gpu_backend.cloud)
    gpu_backend.loader.assert_called_once()
    assert gpu_backend.loader.call_args.kwargs["weights_only"] is not trust
    gpu_backend.model.forward.assert_not_called()
    assert gpu_backend.selected == []


def test_initialization_failure_does_not_cache_partial_state(gpu_backend):
    detector = PointPillarsDetector(**gpu_backend.options)
    gpu_backend.model.load_state_dict.side_effect = RuntimeError("incompatible weights")
    with pytest.raises(RuntimeError, match="incompatible weights"):
        detector.detect(gpu_backend.cloud)
    gpu_backend.model.load_state_dict.side_effect = None
    detector.detect(gpu_backend.cloud)
    assert gpu_backend.loader.call_count == 2
    gpu_backend.model.load_state_dict.assert_called_with({"synthetic_weight": 1}, strict=True)


def test_forward_failure_propagates_and_restores_context(gpu_backend):
    detector = PointPillarsDetector(**gpu_backend.options)
    gpu_backend.model.forward.side_effect = RuntimeError("synthetic CUDA failure")
    with pytest.raises(RuntimeError, match="synthetic CUDA failure"):
        detector.detect(gpu_backend.cloud)
    assert gpu_backend.selected == []


@pytest.mark.parametrize("change", ["model", "dataset", "features", "raw_scores", "classes"])
def test_unsupported_config(gpu_backend, change):
    cfg = gpu_backend.cfg
    if change == "model":
        cfg.MODEL.NAME = "OtherModel"
    elif change == "dataset":
        cfg.DATA_CONFIG.DATASET = "OtherDataset"
    elif change == "features":
        cfg.DATA_CONFIG.POINT_FEATURE_ENCODING.used_feature_list = ["x", "y", "z"]
    elif change == "raw_scores":
        cfg.MODEL.POST_PROCESSING.OUTPUT_RAW_SCORE = True
    else:
        cfg.CLASS_NAMES.reverse()
    previous = Path.cwd()
    with pytest.raises(ValueError, match="Unsupported configuration"):
        PointPillarsDetector(**gpu_backend.options).detect(gpu_backend.cloud)
    assert Path.cwd() == previous
    gpu_backend.loader.assert_not_called()


def test_missing_dependencies(gpu_backend, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    original_path = list(sys.path)
    with pytest.raises(RuntimeError, match="External GPU dependencies are unavailable"):
        PointPillarsDetector(**gpu_backend.options).detect(gpu_backend.cloud)
    assert sys.path == original_path


@pytest.mark.parametrize("available,index", [(False, 0), (True, 2)])
def test_unavailable_device(gpu_backend, available, index):
    gpu_backend.cuda.is_available.return_value = available
    with pytest.raises((RuntimeError, ValueError), match="CUDA"):
        PointPillarsDetector(**gpu_backend.options, device_index=index).detect(gpu_backend.cloud)
    gpu_backend.loader.assert_not_called()


def test_unsupported_revision(gpu_backend, monkeypatch):
    monkeypatch.setattr(
        backend.subprocess,
        "run",
        lambda *a, **k: type("Result", (), {"stdout": "different-commit"})(),
    )
    with pytest.raises(ValueError, match="revision"):
        PointPillarsDetector(**gpu_backend.options).detect(gpu_backend.cloud)


def test_config_objects_are_not_shared_between_sessions(gpu_backend):
    first = PointPillarsDetector(**gpu_backend.options)
    second = PointPillarsDetector(**gpu_backend.options)
    first.initialize()
    second.initialize()
    configs = [call.args[1] for call in gpu_backend.configure.call_args_list]
    assert configs[0] is not configs[1] and configs[0] is not gpu_backend.cfg


@pytest.mark.parametrize(
    "field,value",
    [
        (0, np.zeros((1, 8))),
        (0, np.array([[1, 2, 3, -1, 2, 1, 0]])),
        (1, np.array([1.1])),
        (1, np.array([np.nan])),
        (2, np.array([0])),
        (2, np.array([4])),
        (2, np.array([1.5])),
    ],
)
def test_bad_predictions_rejected(field, value):
    arrays = [np.array([[1.0, 2, 3, 4, 2, 1, 0.5]]), np.array([0.9]), np.array([1])]
    arrays[field] = value
    with pytest.raises(ValueError):
        convert_predictions(*arrays, ["Car", "Pedestrian", "Cyclist"])


def test_lazy_imports_in_fresh_process(tmp_path):
    program = """
import importlib.abc
import sys
blocked = {"torch", "pcdet", "spconv", "cumm", "open3d"}
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in blocked:
            raise AssertionError("Unexpected GPU/GUI import: " + fullname)
sys.meta_path.insert(0, Block())
import json
import raven
from raven.perception.detection.pointpillars import PointPillarsDetector
detector = PointPillarsDetector(**json.loads(sys.argv[1]))
assert not any(name.split(".")[0] in blocked for name in sys.modules)
detector.close()
"""
    (tmp_path / "tools").mkdir()
    (tmp_path / "pcdet").mkdir()
    (tmp_path / "pcdet/__init__.py").write_text("# CPU-only construction fixture")
    config, checkpoint = tmp_path / "model.yaml", tmp_path / "model.pth"
    config.write_text("# not parsed during construction")
    checkpoint.write_bytes(b"not loaded during construction")
    options = {
        "openpcdet_root": str(tmp_path),
        "config": str(config),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": backend.sha256_file(checkpoint),
    }
    subprocess.run(
        [sys.executable, "-c", program, json.dumps(options)],
        check=True,
        capture_output=True,
        text=True,
    )


def test_conflicting_jit_flags_rejected_before_loading(gpu_backend, monkeypatch):
    monkeypatch.setenv("SPCONV_DISABLE_JIT", "0")
    with pytest.raises(ValueError, match="SPCONV_DISABLE_JIT=1"):
        PointPillarsDetector(**gpu_backend.options).initialize()
    gpu_backend.loader.assert_not_called()


def test_flags_cannot_be_set_after_spconv_import(gpu_backend, monkeypatch):
    monkeypatch.setitem(sys.modules, "spconv", object())
    with pytest.raises(RuntimeError, match="fresh process"):
        PointPillarsDetector(**gpu_backend.options).initialize()


def test_incorrect_checkout_import_is_rejected(gpu_backend, monkeypatch, tmp_path):
    from types import SimpleNamespace

    monkeypatch.setitem(sys.modules, "pcdet", SimpleNamespace(__file__=str(tmp_path / "other.py")))
    with pytest.raises(ValueError, match="does not match"):
        PointPillarsDetector(**gpu_backend.options).initialize()


def test_missing_config_fields_are_reported(gpu_backend):
    del gpu_backend.cfg.DATA_CONFIG["POINT_FEATURE_ENCODING"]
    with pytest.raises(ValueError, match="Incomplete KITTI"):
        PointPillarsDetector(**gpu_backend.options).initialize()


def test_unsupported_processor_is_rejected(gpu_backend):
    gpu_backend.cfg.DATA_CONFIG.DATA_PROCESSOR[0].NAME = "unknown_coordinate_transform"
    with pytest.raises(ValueError, match="Unsupported configuration"):
        PointPillarsDetector(**gpu_backend.options).initialize()


def test_wrong_batch_result_count(gpu_backend):
    gpu_backend.model.forward.return_value = ([], {})
    with pytest.raises(ValueError, match="one prediction result"):
        PointPillarsDetector(**gpu_backend.options).detect(gpu_backend.cloud)


def test_wrong_checkpoint_structure(gpu_backend):
    gpu_backend.loader.return_value = {"unrelated": {}}
    with pytest.raises(TypeError, match="model_state"):
        PointPillarsDetector(**gpu_backend.options).initialize()
