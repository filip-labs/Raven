"""Synthetic numerical/provenance tests; none are GPU verification evidence."""

import itertools
import json
import math
import subprocess
import sys
from copy import deepcopy

import numpy as np
import pytest

from raven.perception.detection.base import Detection3D
from raven.perception.detection.predictions import detections_to_document
from raven.perception.detection.reference_validation import (
    CANDIDATE_ENTRYPOINT,
    REFERENCE_ENTRYPOINT,
    Tolerances,
    _assignment,
    compare_documents,
    compare_files,
    main,
)


def box(x=0, label="Car", score=0.9, yaw=0, dimensions=(4, 2, 1)):
    return Detection3D((x, 0, 0), dimensions, yaw, label, score, "kitti_velodyne")


def capture(boxes, reference=False):
    doc = detections_to_document(boxes, frame_id="000000")
    doc.update(
        {
            "input": {**doc["input"], "sha256": "a" * 64, "points": 115384},
            "checkpoint": {"sha256": "b" * 64},
            "config": {"path": "/synthetic/pointpillar.yaml", "sha256": "c" * 64},
            "effective_config": {
                "MODEL": {"NAME": "PointPillar", "threshold": 0.1},
                "DATA_CONFIG": {"range": [0, 70]},
                "CLASS_NAMES": ["Car", "Pedestrian", "Cyclist"],
            },
            "openpcdet_commit": "d" * 40,
            "raven_git_commit": "e" * 40,
            "openpcdet_diff_sha256": "f" * 64,
            "openpcdet_modified_files": [],
            "raven_modified_files": [],
            "runner_sha256": ("1" if reference else "2") * 64,
            "inference_run_id": ("3" if reference else "4") * 32,
            "inference_entrypoint": REFERENCE_ENTRYPOINT if reference else CANDIDATE_ENTRYPOINT,
            "environment": {
                "gpu": "synthetic T4",
                "python": "test",
                "pytorch": "test",
                "torch_cuda_runtime": "test",
                "openpcdet": "test",
                "numpy": "test",
                "spconv_cu126": "test",
                "environment_flags": {
                    "CUMM_DISABLE_JIT": "1",
                    "SPCONV_DISABLE_JIT": "1",
                    "TORCH_CUDA_ARCH_LIST": "7.5",
                },
            },
        }
    )
    if reference:
        doc["reference_dataset_sha256"] = "5" * 64
    return doc


def compare(left, right, settings=None):
    return compare_documents(capture(left, True), capture(right), settings)


def test_exact_matches_reordered_and_no_input_mutation():
    boxes = [box(1), box(4, "Pedestrian"), box(8, "Cyclist"), box(10)]
    reference, candidate = capture(boxes, True), capture(list(reversed(boxes)))
    original = deepcopy((reference, candidate))
    report = compare_documents(reference, candidate)
    assert report["pass"] and report["provenance_pass"]
    assert len(report["matched_detections"]) == 4
    assert report["metrics"]["max_position_error_m"] == 0
    assert report["metrics"]["max_yaw_error_rad"] == 0
    assert (reference, candidate) == original
    assert len({m["candidate_index"] for m in report["matched_detections"]}) == 4
    assert report == compare_documents(reference, candidate)


def test_float32_rounding_and_wrapped_yaw():
    report = compare(
        [box(0.1, yaw=math.pi - 1e-6)], [box(float(np.float32(0.1)), yaw=-math.pi + 1e-6)]
    )
    assert report["pass"]
    assert report["metrics"]["max_yaw_error_rad"] == pytest.approx(2e-6)
    assert compare([box(yaw=0)], [box(yaw=8 * math.pi)])["pass"]


@pytest.mark.parametrize(
    "candidate", [[box(0.01)], [box(score=0.8)], [box(dimensions=(4.1, 2, 1))], [box(yaw=0.1)]]
)
def test_same_count_different_numbers_fail(candidate):
    report = compare([box()], candidate)
    assert not report["pass"] and len(report["matched_detections"]) == 1
    assert "Numerical tolerance exceeded" in report["failures"][-1]


@pytest.mark.parametrize("candidate", [[], [box(), box()], [box(label="Cyclist")], [box(10)]])
def test_missing_duplicated_class_and_far_detections(candidate):
    report = compare([box()], candidate)
    assert not report["pass"]
    assert report["unmatched_reference_indices"] or report["unmatched_candidate_indices"]


def test_empty_sets_require_valid_provenance():
    report = compare([], [])
    assert report["pass"] and report["matched_detections"] == []
    assert all(value is None for value in report["metrics"].values())
    candidate = capture([])
    del candidate["checkpoint"]
    assert not compare_documents(capture([], True), candidate)["pass"]


def test_boundary_absolute_and_relative_tolerances():
    settings = Tolerances(position_atol=0.001, rtol=0)
    assert compare([box()], [box(0.001)], settings)["pass"]
    assert not compare([box()], [box(np.nextafter(0.001, np.inf))], settings)["pass"]
    settings = Tolerances(position_atol=0, rtol=0.01)
    assert compare([box(10)], [box(10.05)], settings)["pass"]
    assert not compare([box(10)], [box(10.2)], settings)["pass"]


def test_global_assignment_avoids_greedy_reuse_and_maximizes_cardinality():
    # Greedy 0 -> .1 would strand .2. Global matching instead uses 0 -> -.3.
    report = compare(
        [box(0), box(0.2)],
        [box(0.1), box(-0.3)],
        Tolerances(position_atol=0.4, rtol=0, max_match_distance=0.35),
    )
    assert report["pass"]
    assert [(r["reference_index"], r["candidate_index"]) for r in report["matched_detections"]] == [
        (0, 1),
        (1, 0),
    ]


def test_hungarian_cost_matches_exhaustive_small_assignment():
    cost = np.array([[3.0, 1, 9, 7], [1, 2, 3, 5], [2, 8, 4, 6]])
    result = _assignment(cost)
    actual = sum(cost[i, j] for i, j in result)
    expected = min(
        sum(cost[i, j] for i, j in enumerate(columns))
        for columns in itertools.permutations(range(4), 3)
    )
    assert actual == expected and len({j for _, j in result}) == 3


def test_symmetric_ties_deterministic():
    report = compare([box(), box()], [box(), box()])
    assert report["pass"]
    assert [(r["reference_index"], r["candidate_index"]) for r in report["matched_detections"]] == [
        (0, 0),
        (1, 1),
    ]


@pytest.mark.parametrize(
    "field",
    [
        "input",
        "checkpoint",
        "config",
        "effective_config",
        "raven_git_commit",
        "inference_run_id",
        "environment",
        "runner_sha256",
        "openpcdet_diff_sha256",
        "raven_modified_files",
    ],
)
def test_missing_provenance_cannot_pass(field):
    candidate = capture([box()])
    del candidate[field]
    if field == "input":
        with pytest.raises(TypeError, match="requires input"):
            compare_documents(capture([box()], True), candidate)
    else:
        report = compare_documents(capture([box()], True), candidate)
        assert not report["pass"] and not report["provenance_pass"]


@pytest.mark.parametrize(
    "path,value",
    [
        (("input", "sha256"), "0" * 64),
        (("input", "points"), 10),
        (("checkpoint", "sha256"), "0" * 64),
        (("config", "sha256"), "0" * 64),
        (("openpcdet_commit",), "0" * 40),
        (("raven_git_commit",), "0" * 40),
        (("openpcdet_diff_sha256",), "0" * 64),
        (("raven_modified_files",), ["modified.py"]),
        (("environment", "gpu"), "different GPU"),
        (("environment", "environment_flags", "TORCH_CUDA_ARCH_LIST"), "8.0"),
        (("effective_config", "MODEL", "threshold"), 0.2),
        (("effective_config", "DATA_CONFIG", "range"), [0, 50]),
        (("inference_entrypoint",), "OpenPCDetSession.predict"),
    ],
)
def test_mismatched_provenance_and_settings_fail(path, value):
    candidate = capture([box()])
    parent = candidate
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value
    report = compare_documents(capture([box()], True), candidate)
    assert not report["pass"] and not report["provenance_pass"]


def test_reexported_same_run_and_wrong_reference_role_fail():
    reference, candidate = capture([box()], True), capture([box()])
    reference["inference_run_id"] = candidate["inference_run_id"]
    reference["runner_sha256"] = candidate["runner_sha256"]
    reference["inference_entrypoint"] = CANDIDATE_ENTRYPOINT
    report = compare_documents(reference, candidate)
    assert not report["pass"]
    assert any("two exports" in error for error in report["failures"])
    assert any("independent" in error for error in report["failures"])


def test_wrong_frame_or_conventions_rejected():
    reference, candidate = capture([box()], True), capture([box()])
    candidate["input"]["frame_id"] = "000001"
    with pytest.raises(ValueError, match="frame_id must match"):
        compare_documents(reference, candidate)
    candidate["input"]["frame_id"] = "000000"
    candidate["rotation_axis"] = "y"
    with pytest.raises(ValueError, match="coordinate conventions"):
        compare_documents(reference, candidate)


@pytest.mark.parametrize(
    "values",
    [
        {"rtol": float("nan")},
        {"score_atol": -1},
        {"position_atol": True},
        {"max_match_distance": 0},
    ],
)
def test_invalid_tolerances(values):
    with pytest.raises((ValueError, TypeError)):
        Tolerances(**values)


def test_readonly_cli_success_failure_corruption_and_no_overwrite(tmp_path, capsys):
    reference, candidate, output = (
        tmp_path / name for name in ("ref.json", "candidate.json", "report.json")
    )
    reference.write_text(json.dumps(capture([box()], True)))
    candidate.write_text(json.dumps(capture([box()])))
    before = (reference.read_bytes(), candidate.read_bytes())
    assert main([str(reference), str(candidate), "--output", str(output)]) == 0
    assert json.loads(output.read_text())["status"] == "PASS"
    assert (reference.read_bytes(), candidate.read_bytes()) == before
    with pytest.raises(SystemExit) as error:
        main([str(reference), str(candidate), "--output", str(output)])
    assert error.value.code == 2
    candidate.write_text('{"broken":')
    failed_output = tmp_path / "failed.json"
    assert main([str(reference), str(candidate), "--output", str(failed_output)]) == 1
    assert not json.loads(failed_output.read_text())["pass"]
    assert "Input verification failed" in json.loads(failed_output.read_text())["failures"][0]
    assert "status" in capsys.readouterr().out


def test_file_report_records_exact_capture_hashes(tmp_path):
    from hashlib import sha256

    files = [tmp_path / "ref.json", tmp_path / "candidate.json"]
    for path, reference in zip(files, (True, False), strict=True):
        path.write_text(json.dumps(capture([], reference)))
    report = compare_files(*files)
    assert report["captures"]["reference"]["sha256"] == sha256(files[0].read_bytes()).hexdigest()


def test_cpu_only_comparison_in_fresh_interpreter(tmp_path):
    reference, candidate = tmp_path / "reference.json", tmp_path / "candidate.json"
    reference.write_text(json.dumps(capture([box()], True)))
    candidate.write_text(json.dumps(capture([box()])))
    program = """
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "pcdet", "spconv", "cumm", "open3d"}:
            raise AssertionError("GPU/GUI import attempted: " + fullname)
sys.meta_path.insert(0, Block())
from raven.perception.detection.reference_validation import compare_files
assert compare_files(sys.argv[1], sys.argv[2])["pass"]
assert "raven.perception.detection._openpcdet" not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", program, str(reference), str(candidate)],
        check=True,
        capture_output=True,
        text=True,
    )


def test_type_sensitive_resolved_settings_comparison():
    reference, candidate = capture([box()], True), capture([box()])
    reference["effective_config"]["MODEL"]["typed_setting"] = True
    candidate["effective_config"]["MODEL"]["typed_setting"] = 1
    assert not compare_documents(reference, candidate)["provenance_pass"]


def test_legacy_geometry_agreement_is_not_verified_success():
    reference, candidate = capture([box()], True), capture([box()])
    del reference["reference_dataset_sha256"]
    del candidate["inference_run_id"]
    report = compare_documents(reference, candidate)
    assert report["numerical_pass"] and not report["provenance_pass"] and not report["pass"]


def test_nonstandard_json_and_duplicate_keys_fail(tmp_path):
    reference, candidate, output = (
        tmp_path / name for name in ("ref.json", "candidate.json", "report.json")
    )
    reference.write_text(json.dumps(capture([box()], True)))
    candidate.write_text('{"schema_version":1,"schema_version":1}')
    assert main([str(reference), str(candidate), "--output", str(output)]) == 1
    assert "Duplicate JSON" in json.loads(output.read_text())["failures"][0]
    candidate.write_text('{"value":NaN}')
    with pytest.raises(ValueError, match="Nonfinite JSON"):
        compare_files(reference, candidate)
