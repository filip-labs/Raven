"""Synthetic offline planning evidence only; no licensed data or remote requests."""

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from test_kitti_evaluation import png

from raven.cli import main
from raven.experiments.config import FrameSelection, load_experiment_config
from raven.experiments.splits import split_membership
from raven.inference.contracts import digest, model_identity
from raven.research.baseline import (
    SUBSET_SHA256,
    baseline_configuration,
    baseline_preflight,
    frozen_selection,
    inspect_baseline_inputs,
    prepare_baseline,
    validate_baseline_config,
)

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "configs/experiments/modal-pointpillars.json"


@pytest.fixture
def baseline(tmp_path, monkeypatch):
    monkeypatch.setenv("RAVEN_ALLOW_MODAL_GPU", "0")
    monkeypatch.setenv("RAVEN_MODAL_ASSET_RIGHTS_CONFIRMED", "0")
    root = tmp_path / "dataset"
    for folder in ("velodyne", "label_2", "calib", "image_2"):
        (root / "training" / folder).mkdir(parents=True)
    for frame in frozen_selection()[0][:8]:
        np.array([[1, 2, 3, 0.25], [4, 5, 6, 0.75]], dtype="<f4").tofile(
            root / "training/velodyne" / f"{frame}.bin"
        )
        (root / "training/label_2" / f"{frame}.txt").write_text(
            "Car 0.15 0 0 0 0 50 50 1.5 1.6 4 0 1 10 0\n"
            "Pedestrian 0 0 0 0 0 20 40 1.8 0.6 0.6 2 1 10 0\n"
            "Cyclist 0 2 0 0 0 20 30 1.8 0.6 1.7 3 1 10 0\n"
        )
        (root / "training/calib" / f"{frame}.txt").write_text(
            "R0_rect: 1 0 0 0 1 0 0 0 1\n"
            "Tr_velo_to_cam: 0 -1 0 0 0 0 -1 0 1 0 0 0\n"
            "P2: 700 0 640 0 0 700 360 0 0 0 1 0\n"
        )
        (root / "training/image_2" / f"{frame}.png").write_bytes(png())
    return baseline_configuration(
        TEMPLATE,
        dataset_root=root,
        output_directory=tmp_path / "runs",
        experiment_id="baseline-test-001",
    )


def test_frozen_selection_exact_bytes_membership_and_existing_seeded_algorithm():
    ids, record = frozen_selection()
    members, split = split_membership("val")
    assert len(ids) == len(set(ids)) == 256
    assert len(members) == 3769 and set(ids) <= members
    assert "000000" not in ids and "000000" not in members
    assert ids[:8] == (
        "000053",
        "000173",
        "000191",
        "000247",
        "000272",
        "000291",
        "000332",
        "000420",
    )
    expected = FrameSelection(ids=tuple(sorted(members)), sample_size=256).select(42)
    assert ids == expected == frozen_selection()[0]
    assert hashlib.sha256("".join(f"{i}\n" for i in ids).encode()).hexdigest() == SUBSET_SHA256
    assert record["split"] == split
    assert record["model"] == model_identity()
    assert record["preprocessing_sha256"] == digest([])


@pytest.mark.parametrize("count", [0, -1, 9, True, 1.5, "1"])
def test_config_generation_preserves_eight_frame_cap(baseline, tmp_path, count):
    with pytest.raises(ValueError, match="frame_count"):
        baseline_configuration(
            TEMPLATE,
            dataset_root=baseline.dataset.root,
            output_directory=tmp_path / "out",
            experiment_id="new",
            frame_count=count,
        )


def test_explicit_eight_frame_pilot_uses_frozen_order_and_raw_model(baseline, tmp_path):
    config = baseline_configuration(
        TEMPLATE,
        dataset_root=baseline.dataset.root,
        output_directory=tmp_path / "out",
        experiment_id="eight",
        frame_count=8,
    )
    assert config.frame_ids == frozen_selection()[0][:8]
    assert config.dataset.split == "val" and config.preprocessing == ()
    assert config.schema_version == 2 and config.model.source == "modal"
    assert config.model.remote.max_frames == 8
    assert json.loads(TEMPLATE.read_text())["model"]["remote"]["max_frames"] == 1
    assert not config.run_directory.exists()


@pytest.mark.parametrize(
    "changes",
    [
        {
            "dataset": {"root": "/tmp/kitti-baseline-synthetic", "split": "train"},
            "frames": {"ids": ["000000"]},
        },
        {"frames": {"ids": ["000000"]}},
        {"seed": 43},
        {"preprocessing": [{"name": "voxel_downsample", "voxel_size": 0.2}]},
    ],
)
def test_incompatible_baselines_rejected_before_reading(baseline, changes):
    document = baseline.to_dict() | changes
    config = type(baseline).from_dict(document, base_dir=ROOT)
    with pytest.raises(ValueError, match="baseline"):
        validate_baseline_config(config)


def test_different_deployment_rejected(baseline):
    remote = replace(baseline.model.remote, app="other-worker")
    config = replace(baseline, model=replace(baseline.model, remote=remote))
    with pytest.raises(ValueError, match="deployment"):
        validate_baseline_config(config)


def test_offline_eligibility_uses_exact_difficulty_boundaries_and_preserves_sources(baseline):
    before = {p: p.read_bytes() for p in baseline.dataset.root.rglob("*") if p.is_file()}
    result = baseline_preflight(baseline)
    assert result["inputs"]["status"] == "ready"
    # Easy excludes height == 40; moderate excludes occlusion == 2; hard accepts it.
    assert result["inputs"]["eligible_ground_truth"] == {
        "Car": {"easy": 1, "moderate": 1, "hard": 1},
        "Pedestrian": {"easy": 0, "moderate": 1, "hard": 1},
        "Cyclist": {"easy": 0, "moderate": 0, "hard": 1},
    }
    assert result["inputs"]["coverage_complete"]
    assert any("AP unsupported" in w for w in result["inputs"]["warnings"])
    assert result["evaluation_readiness"] == "inputs_ready_predictions_pending"
    assert result["runner_preflight"]["status"] == "blocked"
    assert result["cloud"]["authorization"] == "not_authorized_for_this_plan"
    assert result["metrics"] is None and result["cloud"]["operations_performed"] == 0
    assert {p: p.read_bytes() for p in before} == before
    assert not baseline.run_directory.exists()
    row = result["inputs"]["frames"][0]
    assert row["coordinate_frame"] == "kitti_velodyne" and row["points"] == 2
    assert set(row["files"]) == {"scan", "labels", "calibration", "image"}
    assert "points" not in json.dumps(row["files"])


@pytest.mark.parametrize(
    "folder,suffix", [("velodyne", "bin"), ("label_2", "txt"), ("calib", "txt"), ("image_2", "png")]
)
def test_missing_inputs_never_replaced(baseline, folder, suffix):
    frame = baseline.frame_ids[0]
    (baseline.dataset.root / "training" / folder / f"{frame}.{suffix}").unlink()
    result = inspect_baseline_inputs(baseline)
    assert result["status"] == "blocked" and result["frame_ids"] == [frame]
    assert result["frames"][0]["failures"]
    assert result["coverage_complete"] == (folder == "image_2")


@pytest.mark.parametrize(
    "folder,suffix,raw",
    [
        ("velodyne", "bin", b"bad"),
        ("label_2", "txt", b"bad\n"),
        ("calib", "txt", b"bad\n"),
        ("image_2", "png", b"fake dimensions"),
    ],
)
def test_malformed_inputs_report_blocked(baseline, folder, suffix, raw):
    (baseline.dataset.root / "training" / folder / f"{baseline.frame_ids[0]}.{suffix}").write_bytes(
        raw
    )
    assert baseline_preflight(baseline)["evaluation_readiness"] == "blocked_inputs"


def test_missing_dataset_no_substitution_or_fake_coverage(baseline, tmp_path):
    config = replace(baseline, dataset=replace(baseline.dataset, root=tmp_path / "absent"))
    result = baseline_preflight(config)
    assert result["runner_preflight"]["status"] == "failed"
    assert not result["inputs"]["coverage_complete"]
    assert result["inputs"]["verified_label_frames"] == 0
    assert result["metrics"] is None


def test_plan_roundtrip_exclusive_and_output_collision(baseline, tmp_path):
    args = {
        "dataset_root": baseline.dataset.root,
        "output_directory": baseline.output.directory,
        "experiment_id": baseline.experiment_id,
        "plan_directory": tmp_path / "plan",
    }
    result = prepare_baseline(TEMPLATE, **args)
    assert load_experiment_config(result["configuration"]) == baseline
    paths = sorted((tmp_path / "plan").iterdir())
    before = {p: p.read_bytes() for p in paths}
    with pytest.raises(FileExistsError):
        prepare_baseline(TEMPLATE, **args)
    assert {p: p.read_bytes() for p in paths} == before
    baseline.run_directory.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="run directory"):
        prepare_baseline(TEMPLATE, **(args | {"plan_directory": tmp_path / "new-plan"}))
    assert not (tmp_path / "new-plan").exists()


def test_plan_must_not_pollute_dataset_or_future_output(baseline):
    for protected in (baseline.dataset.root, baseline.output.directory):
        with pytest.raises(ValueError, match="disjoint"):
            prepare_baseline(
                TEMPLATE,
                dataset_root=baseline.dataset.root,
                output_directory=baseline.output.directory,
                experiment_id="safe",
                plan_directory=protected / "plan",
            )


def test_cli_persists_honest_blocked_plan_when_images_absent(baseline, tmp_path, capsys):
    (baseline.dataset.root / "training/image_2" / f"{baseline.frame_ids[0]}.png").unlink()
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "prepare-baseline",
                str(TEMPLATE),
                "--dataset-root",
                str(baseline.dataset.root),
                "--output-directory",
                str(baseline.output.directory),
                "--experiment-id",
                "cli-test",
                "--plan-directory",
                str(tmp_path / "plan"),
            ]
        )
    assert exc.value.code == 3
    assert (
        json.loads(capsys.readouterr().out)["cloud_authorization"] == "not_authorized_for_this_plan"
    )
    report = json.loads((tmp_path / "plan/preflight.json").read_text())
    assert report["evaluation_readiness"] == "blocked_inputs" and report["metrics"] is None


def test_planning_in_separate_process_never_imports_remote_or_gpu(baseline, tmp_path):
    script = """
import sys
class NoRemote:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'modal', 'torch', 'pcdet', 'spconv', 'open3d'}:
            raise AssertionError('forbidden planning import: ' + fullname)
sys.meta_path.insert(0, NoRemote())
from raven.research.baseline import prepare_baseline
result = prepare_baseline(sys.argv[1], dataset_root=sys.argv[2], output_directory=sys.argv[3],
    experiment_id='isolated', plan_directory=sys.argv[4])
assert result['cloud_authorization'] == 'not_authorized_for_this_plan'
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(TEMPLATE),
            str(baseline.dataset.root),
            str(baseline.output.directory),
            str(tmp_path / "isolated-plan"),
        ],
        env={
            **os.environ,
            "PYTHONPATH": str(ROOT / "src"),
            "RAVEN_ALLOW_MODAL_GPU": "0",
            "RAVEN_MODAL_ASSET_RIGHTS_CONFIRMED": "0",
        },
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr


def test_modified_selection_identity_is_rejected(monkeypatch):
    import raven.research.baseline as module

    monkeypatch.setattr(module, "SUBSET_SHA256", "f" * 64)
    with pytest.raises(ValueError, match="identity mismatch"):
        frozen_selection()


def test_invalid_baseline_fails_before_any_input_operation(baseline, monkeypatch):
    from unittest.mock import Mock

    import raven.research.baseline as module

    loader = Mock(side_effect=AssertionError("must not read inputs"))
    monkeypatch.setattr(module, "load_experiment_frame", loader)
    with pytest.raises(ValueError, match="baseline"):
        baseline_preflight(replace(baseline, seed=1))
    loader.assert_not_called()


def test_subset_audit_reports_all_missing_frames_without_replacement(baseline):
    report = baseline_preflight(baseline, inspect_subset=True)
    assert report["inputs"]["frame_ids"] == list(frozen_selection()[0])
    assert report["inputs"]["frame_count"] == 256
    assert report["inputs"]["verified_label_frames"] == 8
    assert not report["inputs"]["coverage_complete"]
    assert report["inputs"]["eligible_ground_truth"]["Car"] == dict.fromkeys(
        ["easy", "moderate", "hard"], 8
    )
    assert len([r for r in report["inputs"]["frames"] if r["failures"]]) == 248
    assert report["metrics"] is None


def test_portable_example_is_valid_schema_v2_and_never_runs():
    config = load_experiment_config(ROOT / "configs/experiments/m6-baseline-pilot.json")
    validate_baseline_config(config)
    assert config.frame_ids == ("000053",)
    assert config.model.remote.max_frames == 1
