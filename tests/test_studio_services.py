"""Studio's CPU adapters use existing backend contracts; all fixtures are synthetic."""

import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from test_modal_m5_integration import execute
from test_modal_m5_integration import remote as remote  # noqa: PLC0414
from test_remote_inference import config as config  # noqa: PLC0414 -- Imported fixture.

from raven.studio.services import (
    STAGES,
    ViewOptions,
    discover_dataset,
    export_prediction_copy,
    inspect_frame,
    load_pipeline,
    load_saved_run,
    pipeline_document,
    prediction_input,
    summarize_result,
    viewer_arguments,
)
from raven.studio.state import PipelineEditor


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "KITTI with spaces"
    for parent in ["training", "testing"]:
        scan = root / parent / "velodyne"
        scan.mkdir(parents=True)
        for frame in ["000000", "000001"]:
            np.array([[1, 2, 3, 0.5], [1.01, 2, 3, 0.7], [-1, 0, 0, 0.2]], dtype="<f4").tofile(
                scan / f"{frame}.bin"
            )
    return root


def snapshot(root):
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }


def test_dataset_discovery_is_lazy_and_uses_canonical_membership(dataset, monkeypatch):
    monkeypatch.setattr(
        "raven.studio.services.load_kitti_bin", lambda _: pytest.fail("discovery loaded a scan")
    )
    monkeypatch.setattr(
        "raven.studio.services.split_membership",
        lambda split: (frozenset({"000000"} if split == "train" else {"000001"}), {}),
    )
    train = discover_dataset(dataset, "train")
    val = discover_dataset(dataset, "val")
    assert [p.stem for p in train.paths] == ["000000"]
    assert [p.stem for p in val.paths] == ["000001"]
    assert train.config.scan_directory == val.config.scan_directory
    test = discover_dataset(dataset, "test")
    assert len(test.paths) == 2 and "unavailable" in test.membership


def test_real_canonical_split_and_invalid_dataset_paths(dataset, tmp_path):
    assert "000000" in [p.stem for p in discover_dataset(dataset, "train").paths]
    with pytest.raises(ValueError, match="split"):
        discover_dataset(dataset, "training")
    with pytest.raises(ValueError, match="Missing KITTI"):
        discover_dataset(tmp_path / "absent", "train")
    bad = tmp_path / "file"
    bad.write_text("no directory")
    with pytest.raises(ValueError, match="directory"):
        discover_dataset(bad, "train")


def test_frame_inspection_reports_real_bounds_intensity_and_optional_files(dataset):
    before = snapshot(dataset)
    scan = dataset / "training/velodyne/000000.bin"
    info = inspect_frame(scan)
    assert info.cloud.count == 3 and info.cloud.coordinate_frame == "kitti_velodyne"
    assert info.bounds == ((-1.0, 0.0, 0.0), (float(np.float32(1.01)), 2.0, 3.0))
    assert info.ground_truth is None and not info.calibration_available
    assert not info.cloud.points.flags.writeable
    assert snapshot(dataset) == before


def test_corrupt_scan_and_labels_report_errors(dataset):
    scan = dataset / "training/velodyne/000000.bin"
    scan.write_bytes(b"bad")
    with pytest.raises(ValueError, match="size"):
        inspect_frame(scan)
    np.array([[1, 2, 3, 0.5]], dtype="<f4").tofile(scan)
    labels = dataset / "training/label_2"
    labels.mkdir()
    (labels / "000000.txt").write_text("invalid annotation")
    with pytest.raises(ValueError):
        inspect_frame(scan)


def test_pipeline_editor_order_enable_repeat_and_backend_validation(tmp_path):
    editor = PipelineEditor()
    for stage in STAGES.values():
        editor.add(stage)
    editor.add(STAGES["Voxel downsampling"])
    assert editor.move(4, -1) == 3
    editor.rows[0]["enabled"] = False
    assert [s["name"] for s in editor.active()] == [
        "voxel_downsample",
        "statistical_outlier_removal",
        "voxel_downsample",
        "noise",
    ]
    detached = editor.active()
    detached[0]["voxel_size"] = 42
    assert editor.active()[0]["voxel_size"] == 0.2
    with pytest.raises(ValueError, match="Stage"):
        editor.replace(1, {"name": "voxel_downsample", "voxel_size": -1})
    config = tmp_path / "stages.json"
    config.write_text(json.dumps(editor.active()))
    assert load_pipeline(config) == editor.active()
    config.write_text('[{"name":"noise","position_std":NaN,"seed":42}]')
    with pytest.raises(ValueError):
        load_pipeline(config)
    with pytest.raises(ValueError):
        pipeline_document({"stages": []})


def test_saved_run_and_prediction_copy_are_validated_read_only_exclusive(remote, tmp_path):
    config, _ = remote
    config = replace(config, preprocessing=())
    manifest = execute(config)
    before = snapshot(config.run_directory)
    saved = load_saved_run(config.run_directory)
    summary = summarize_result(saved, "000001")
    assert summary["evidence"] == "synthetic_test" and summary["detections"] == 1
    assert summary["gpu"] == "Tesla T4" and summary["initialization"] == 2.0
    output = tmp_path / "new-predictions.json"
    scan, path = export_prediction_copy(saved, "000001", output)
    assert path == output and scan.stem == "000001"
    assert json.loads(output.read_text()) == saved.frame("000001")["predictions"]
    with pytest.raises(FileExistsError):
        export_prediction_copy(saved, "000001", output)
    with pytest.raises(ValueError, match="outside"):
        export_prediction_copy(saved, "000001", config.run_directory / "new.json")
    assert snapshot(config.run_directory) == before
    scan.write_bytes(b"changed input")
    with pytest.raises(ValueError, match="no longer matches"):
        prediction_input(saved, "000001")
    assert manifest.is_file()


def test_processed_prediction_overlay_is_not_silently_shown_on_raw_cloud(remote):
    config, _ = remote
    saved = load_saved_run(execute(config))
    with pytest.raises(ValueError, match="raw-baseline"):
        prediction_input(saved, "000001")


def test_saved_result_tampering_missing_and_failed_runs_rejected(remote, tmp_path):
    config, _ = remote
    manifest = execute(config)
    results = config.run_directory / "modal.results.json"
    results.write_text(
        results.read_text().replace('"prediction_count": 1', '"prediction_count": 99')
    )
    with pytest.raises(ValueError):
        load_saved_run(manifest)
    with pytest.raises(ValueError):
        load_saved_run(tmp_path)
    bad = tmp_path / "failed.manifest.json"
    bad.write_text(json.dumps({"run_schema_version": 1, "status": "failed"}))
    with pytest.raises(ValueError, match="successful"):
        load_saved_run(bad)


@pytest.mark.parametrize(
    "options",
    [
        ViewOptions(color="wrong"),
        ViewOptions(point_size=True),
        ViewOptions(point_size=float("nan")),
        ViewOptions(point_size=0),
    ],
)
def test_viewer_options_invalid(options):
    with pytest.raises(ValueError):
        options.arguments()


def test_viewer_command_uses_literal_argument_arrays_and_existing_cli(dataset):
    path = dataset / "training/velodyne/000000.bin"
    args = viewer_arguments(
        path, ViewOptions("intensity", 3, Path("/tmp/screenshots with spaces")), ground_truth=True
    )
    assert args[:4] == ["-m", "raven.cli", "visualize", str(path)]
    assert args[-1] == "--ground-truth" and "/tmp/screenshots with spaces" in args
    with pytest.raises(FileNotFoundError):
        viewer_arguments(path.with_name("missing.bin"), ViewOptions())


def test_cli_and_services_do_not_import_qt_gpu_or_cloud_and_missing_studio_is_actionable():
    script = """
import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, *args):
  if fullname.split('.')[0] in {'PySide6','torch','modal','pcdet','spconv','cumm','open3d'}:
   raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0,Block())
from raven.studio.services import discover_dataset
from raven.cli import main
try: main(['studio','--help'])
except SystemExit as e: assert e.code==0
try: main(['studio'])
except SystemExit as e: assert e.code==2
else: raise AssertionError('missing Qt was not reported')
assert 'PySide6' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    assert ".[studio,visualization,preprocessing]" in result.stderr


def test_empty_cloud_and_exact_per_axis_config_round_trip(dataset, tmp_path):
    scan = dataset / "training/velodyne/000000.bin"
    scan.write_bytes(b"")
    info = inspect_frame(scan)
    assert info.cloud.count == 0 and info.bounds is None
    config = [{"name": "voxel_downsample", "voxel_size": [0.123456789012345, 0.2, 0.4]}]
    path = tmp_path / "pipeline.json"
    path.write_text(json.dumps(config))
    assert load_pipeline(path) == config
