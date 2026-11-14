import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from raven.perception.detection import _openpcdet as backend
from raven.perception.detection.execution_provenance import execution_provenance
from raven.perception.detection.pointpillars import PointPillarsDetector
from raven.perception.detection.reference_validation import REFERENCE_ENTRYPOINT


@pytest.fixture
def reference_runner(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/openpcdet_reference.py"
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location("openpcdet_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_independent_upstream_dataset_and_forward_without_raven_inference(
    reference_runner, gpu_backend, monkeypatch, tmp_path
):
    fake = gpu_backend
    # A synthetic upstream class fixture; tests do not claim execution of a real GPU/model.
    source = """
raise AssertionError("Top-level demo/GUI code must not execute")
class DemoDataset(DatasetTemplate):
    def __init__(self, **kwargs):
        self.path = kwargs["root_path"]
    def __getitem__(self, index):
        return self.prepare_data({"points": np.fromfile(self.path, dtype=np.float32).reshape(-1,4),
                                  "frame_id": index})
"""
    (fake.root / "tools/demo.py").write_text(source)
    monkeypatch.setattr(
        reference_runner, "DEMO_SOURCE_SHA256", backend.sha256_file(fake.root / "tools/demo.py")
    )

    class Base:
        prepare_data = staticmethod(fake.dataset.prepare_data)
        collate_batch = staticmethod(fake.dataset.collate_batch)

    import sys

    monkeypatch.setattr(sys.modules["pcdet.datasets"], "DatasetTemplate", Base)
    forbidden = Mock(side_effect=AssertionError("Raven inference must not be called"))
    monkeypatch.setattr(backend.OpenPCDetSession, "predict", forbidden)
    monkeypatch.setattr(PointPillarsDetector, "detect", forbidden)
    monkeypatch.setattr(
        reference_runner,
        "command_output",
        lambda args: backend.PINNED_REVISION if "rev-parse" in args else "",
    )
    scan = tmp_path / "000000.bin"
    fake.cloud.points.tofile(scan)
    args = SimpleNamespace(
        **fake.options, adapter=False, trust_checkpoint=False, device_index=0, input=scan
    )
    previous = Path.cwd()
    arrays, names, metadata = reference_runner.run_reference(args, None)
    assert Path.cwd() == previous
    forbidden.assert_not_called()
    fake.model.forward.assert_called_once()
    assert fake.dataset.prepare_data.call_args.args[0]["frame_id"] == 0
    np.testing.assert_array_equal(
        fake.dataset.prepare_data.call_args.args[0]["points"], fake.cloud.points
    )
    np.testing.assert_array_equal(arrays[0], fake.raw[0])
    assert names == ["Car", "Pedestrian", "Cyclist"]
    assert metadata["inference_entrypoint"] == REFERENCE_ENTRYPOINT
    assert metadata["reference_dataset_sha256"]
    assert fake.loader.call_args.kwargs["weights_only"] is True


def test_reference_rejects_adapter_flag(reference_runner):
    with pytest.raises(ValueError, match="cannot use --adapter"):
        reference_runner.run_reference(SimpleNamespace(adapter=True), None)


def test_missing_upstream_demo_class_is_explicit(reference_runner, tmp_path):
    path = tmp_path / "demo.py"
    path.write_text("# missing class")
    with pytest.raises(ValueError, match="exactly one"):
        reference_runner.upstream_demo_dataset(path, object)


def test_provenance_missing_git_does_not_invent_identity(monkeypatch, tmp_path):
    import raven.perception.detection.execution_provenance as provenance

    monkeypatch.setattr(provenance, "_git", Mock(side_effect=OSError("git unavailable")))
    first, second = execution_provenance(tmp_path), execution_provenance(tmp_path)
    assert first["raven_git_commit"] is None and first["openpcdet_diff_sha256"] is None
    assert first["raven_modified_files"] is None
    assert first["inference_run_id"] != second["inference_run_id"]


def test_changed_upstream_demo_source_rejected(
    reference_runner, gpu_backend, monkeypatch, tmp_path
):
    (gpu_backend.root / "tools/demo.py").write_text("# altered upstream source")
    monkeypatch.setattr(
        reference_runner,
        "command_output",
        lambda args: backend.PINNED_REVISION if "rev-parse" in args else "",
    )
    args = SimpleNamespace(
        **gpu_backend.options,
        adapter=False,
        trust_checkpoint=False,
        device_index=0,
        input=tmp_path / "000000.bin",
    )
    with pytest.raises(ValueError, match="differs from the pinned"):
        reference_runner.run_reference(args, None)
    gpu_backend.loader.assert_not_called()
