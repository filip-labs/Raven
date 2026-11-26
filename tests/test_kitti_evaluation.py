import hashlib
import json
import math
import os
import struct
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

from raven.cli import main
from raven.evaluation.backend import (
    SOURCE_HASHES,
    evaluate_annotations,
    metric_reference,
    rotate_iou_cpu,
)
from raven.evaluation.evaluate import evaluate_experiment, save_evaluation_result
from raven.evaluation.geometry import CalibrationAdapter, image_metadata, prediction_annotation
from raven.evaluation.inputs import load_completed_run
from raven.experiments.config import ExperimentConfig
from raven.experiments.runner import run_experiment
from raven.perception.data.kitti_calibration import KittiCalibration
from raven.perception.detection.base import Detection3D
from raven.perception.detection.predictions import detections_to_document

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures/kitti_evaluation_reference.json").read_text()
)


def annotation(cls="Car", *, score=None, x=0.0, height=100.0):
    result = {
        "name": np.array([cls]),
        "bbox": np.array([[0.0, 0.0, 100.0, height]]),
        "dimensions": np.array([[4.0, 2.0, 2.0]]),
        "location": np.array([[x, 1.0, 20.0]]),
        "rotation_y": np.array([0.0]),
        "alpha": np.array([0.0]),
    }
    if score is None:
        result.update(occluded=np.array([0]), truncated=np.array([0.0]))
    else:
        result["score"] = np.array([score])
    return result


def empty_prediction():
    return {k: v[:0].copy() for k, v in annotation(score=0.9).items()}


@pytest.mark.parametrize(
    "case", ["perfect", "high_score_false_positive_each_frame", "wrong_class", "empty_predictions"]
)
def test_reference_ap_matches_independent_precision_recall_fixture(case):
    gt = [annotation() for _ in range(41)]
    dt = [annotation(score=0.9) for _ in gt]
    if case == "high_score_false_positive_each_frame":
        dt = [
            {k: np.concatenate([v, annotation(score=0.95, x=100.0)[k]]) for k, v in row.items()}
            for row in dt
        ]
    elif case == "wrong_class":
        dt = [annotation("Cyclist", score=0.9) for _ in gt]
    elif case == "empty_predictions":
        dt = [empty_prediction() for _ in gt]
    metrics, diagnostics = evaluate_annotations(gt, dt)
    for m in metrics:
        if m["class_name"] == "Car":
            assert m["ap_r40_percent"] == pytest.approx(FIXTURE["ap_r40"][case], abs=1e-8)
        else:
            assert m["ap_r40_percent"] is None and m["availability"] == "no_eligible_ground_truth"
    assert all(d["tp"] + d["fn"] == 1 for d in diagnostics if d["class_name"] == "Car")


def test_duplicate_detection_count_is_separate_from_ranked_ap():
    gt = [annotation() for _ in range(41)]
    dt = []
    for _ in gt:
        a, b = annotation(score=0.9), annotation(score=0.1)
        dt.append({k: np.concatenate([a[k], b[k]]) for k in a})
    metrics, diagnostics = evaluate_annotations(gt, dt, include_bev=False)
    assert next(m for m in metrics if m["class_name"] == "Car")["ap_r40_percent"] == 100
    assert all(
        (d["tp"], d["fp"], d["fn"]) == (1, 1, 0) for d in diagnostics if d["class_name"] == "Car"
    )


@pytest.mark.parametrize("overlap,matched", [(0.7, False), (0.7 - 1e-8, False), (0.7 + 1e-8, True)])
def test_iou_strict_boundary_with_independent_overlap_values(overlap, matched):
    ref = metric_reference()
    gt = np.array([[0, 0, 100, 100, 0.0]], dtype=float)
    dt = np.array([[0, 0, 100, 100, 0, 0.9]], dtype=float)
    tp, fp, fn, *_ = ref.compute_statistics_jit(
        np.array([[overlap]]),
        gt,
        dt,
        np.array([0]),
        np.array([0]),
        np.empty((0, 4)),
        2,
        0.7,
        compute_fp=True,
    )
    assert (tp, fp, fn) == ((1, 0, 0) if matched else (0, 1, 1))


@pytest.mark.parametrize(
    "difficulty,cutoff,max_occlusion,max_truncation",
    [(0, 40, 0, 0.15), (1, 25, 1, 0.3), (2, 25, 2, 0.5)],
)
def test_difficulty_boundaries_and_adjacent_ignored_classes(
    difficulty, cutoff, max_occlusion, max_truncation
):
    ref = metric_reference()
    gt = annotation(height=cutoff)
    dt = annotation(score=0.9, height=cutoff)
    assert ref.clean_data(gt, dt, 0, difficulty)[:3] == (0, [1], [0])
    gt["bbox"][0, 3] += 1
    gt["occluded"][0] = max_occlusion
    gt["truncated"][0] = max_truncation
    assert ref.clean_data(gt, dt, 0, difficulty)[0] == 1
    gt["occluded"][0] += 1
    assert ref.clean_data(gt, dt, 0, difficulty)[0] == 0
    assert ref.clean_data(annotation("Van"), dt, 0, difficulty)[1] == [1]
    assert ref.clean_data(annotation("Person_sitting"), dt, 1, difficulty)[1] == [1]


def test_dontcare_rule_is_not_silently_applied_to_3d_or_bev():
    ref = metric_reference()
    gt = np.array([[0, 0, 100, 100, 0.0]], float)
    dt = np.array([[0, 0, 100, 100, 0, 0.9]], float)
    dc = np.array([[0, 0, 100, 100]], float)
    for metric, expected_fp in [(0, 0), (1, 1), (2, 1)]:
        tp, fp, fn, *_ = ref.compute_statistics_jit(
            np.zeros((1, 1)),
            gt,
            dt,
            np.array([-1]),
            np.array([0]),
            dc,
            metric,
            0.7,
            compute_fp=True,
        )
        assert (tp, fp, fn) == (0, expected_fp, 0)


def polygon(box):
    x, y, l, w, angle = box
    points = np.array([[-l / 2, -w / 2], [l / 2, -w / 2], [l / 2, w / 2], [-l / 2, w / 2]])
    return points @ np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    ) + [x, y]


def oracle_iou(a, b):
    # Independent float64 Sutherland-Hodgman clipping + shoelace, not reference intersection/sort.
    points = list(polygon(a))
    clip = polygon(b)

    def cross(v, w):
        return v[0] * w[1] - v[1] * w[0]

    for i in range(4):
        if not points:
            return 0.0
        start, end = clip[i], clip[(i + 1) % 4]
        edge = end - start
        result = []
        prev = points[-1]
        for cur in points:
            dp, dc = cross(edge, prev - start), cross(edge, cur - start)
            if (dp >= 0) != (dc >= 0):
                result.append(prev + (cur - prev) * dp / (dp - dc))
            if dc >= 0:
                result.append(cur)
            prev = cur
        points = result
    if not points:
        return 0.0
    p = np.array(points)
    area = abs(np.dot(p[:, 0], np.roll(p[:, 1], 1)) - np.dot(p[:, 1], np.roll(p[:, 0], 1))) / 2
    return area / (a[2] * a[3] + b[2] * b[3] - area)


def test_rotated_geometry_against_closed_forms_and_independent_polygon_oracle():
    boxes = np.array([[0, 0, 2, 2, 0.0], [0, 0, 2, 2, np.pi / 4], [20, 0, 2, 2, 0.0]])
    overlaps = rotate_iou_cpu(boxes, boxes)
    np.testing.assert_allclose(np.diag(overlaps), 1, atol=1e-10)
    assert overlaps[0, 1] == pytest.approx(FIXTURE["geometry"]["square_45_iou"], abs=1e-10)
    assert overlaps[0, 2] == 0
    rng = np.random.Generator(np.random.PCG64(42))
    for _ in range(100):
        a = np.r_[rng.uniform(-3, 3, 2), rng.uniform(0.5, 6, 2), rng.uniform(-np.pi, np.pi)]
        b = np.r_[rng.uniform(-3, 3, 2), rng.uniform(0.5, 6, 2), rng.uniform(-np.pi, np.pi)]
        assert rotate_iou_cpu(a[None], b[None])[0, 0] == pytest.approx(oracle_iou(a, b), abs=1e-10)
    ref = metric_reference()
    a = np.array([[0.0, 1.0, 20.0, 4.0, 2.0, 2.0, 0.0]])
    b = a.copy()
    b[0, 1] += 1
    assert ref.d3_box_overlap(a, b)[0, 0] == pytest.approx(1 / 3, abs=1e-10)


def calibration(tilt=0.0):
    r0 = np.array([[1, 0, 0], [0, np.cos(tilt), -np.sin(tilt)], [0, np.sin(tilt), np.cos(tilt)]])
    return KittiCalibration(
        {
            "R0_rect": r0,
            "Tr_velo_to_cam": np.array([[0, -1, 0, 2], [0, 0, -1, 3], [1, 0, 0, 4]], float),
            "P2": np.array([[700, 0, 640, 0], [0, 700, 360, 0], [0, 0, 1, 0]], float),
        }
    )


def test_conversion_matches_independent_full_matrix_and_benchmark_axes_without_mutation():
    box = Detection3D((20.0, 1.0, 2.0), (4.0, 2.0, 2.0), 0.4, "Car", 0.9, "kitti_velodyne")
    calib = calibration(0.01)
    result, meta = prediction_annotation([box], calib, (720, 1280))
    original = np.array(box.center_xyz)
    bottom = original - [0, 0, 1]
    expected = calib.matrices["R0_rect"] @ (
        calib.matrices["Tr_velo_to_cam"][:, :3] @ bottom + calib.matrices["Tr_velo_to_cam"][:, 3]
    )
    np.testing.assert_allclose(result["location"][0], expected, atol=1e-12)
    np.testing.assert_array_equal(result["dimensions"][0], [4, 2, 2])
    assert result["rotation_y"][0] == -0.4 - np.pi / 2
    assert (
        meta["axis_deviation_rad"] == pytest.approx(0.01, abs=1e-12)
        and meta["full_so3_boxes"] is False
    )
    assert box.center_xyz == (20.0, 1.0, 2.0)
    # Independent analytical eight-corner projection of the benchmark's upright camera box.
    yaw = result["rotation_y"][0]
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    corners = (
        np.array([[x, y, z] for x in (-2, 2) for y in (0, -2) for z in (-1, 1)]) @ rotation.T
        + expected
    )
    projected = np.c_[corners, np.ones(8)] @ calib.matrices["P2"].T
    uv = projected[:, :2] / corners[:, 2, None]
    expected_bbox = np.r_[uv.min(0), uv.max(0)]
    np.testing.assert_allclose(result["bbox"][0], expected_bbox, atol=1e-4)
    with pytest.raises(ValueError, match="Velodyne"):
        prediction_annotation(
            [Detection3D(box.center_xyz, box.size_lwh, 0, "Car", 0.9, "lidar")], calib, (720, 1280)
        )


def png(width=1280, height=720):
    def chunk(kind, data):
        return (
            struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress((b"\0" + b"\0" * (width * 3)) * height))
        + chunk(b"IEND", b"")
    )


@pytest.fixture
def completed(tmp_path):
    root = tmp_path / "dataset"
    for folder in ("velodyne", "label_2", "calib", "image_2"):
        (root / "training" / folder).mkdir(parents=True)
    preds = tmp_path / "preds"
    preds.mkdir()
    frame = "000001"  # Canonical val member; fixture bytes are deliberately synthetic.
    scan = root / "training/velodyne" / f"{frame}.bin"
    np.array([[20, 0, 0, 0.4]], dtype="<f4").tofile(scan)
    (root / "training/label_2" / f"{frame}.txt").write_text(
        "Car 0 0 0 600 300 680 400 2 2 4 0 1 20 -1.5707963267948966\n"
    )
    (root / "training/calib" / f"{frame}.txt").write_text(
        "R0_rect: 1 0 0 0 1 0 0 0 1\nTr_velo_to_cam: 0 -1 0 0 0 0 -1 0 1 0 0 0\nP2: 700 0 640 0 0 700 360 0 0 0 1 0\n"
    )
    (root / "training/image_2" / f"{frame}.png").write_bytes(png())
    pred = detections_to_document(
        [Detection3D((20, 0, 0), (4, 2, 2), 0, "Car", 0.9, "kitti_velodyne")], frame_id=frame
    )
    pred["input"].update(sha256=hashlib.sha256(scan.read_bytes()).hexdigest(), points=1)
    (preds / f"{frame}.json").write_text(json.dumps(pred))
    config = ExperimentConfig.from_dict(
        {
            "schema_version": 1,
            "name": "Synthetic evaluation fixture",
            "experiment_id": "synthetic-eval",
            "dataset": {"root": str(root), "split": "val"},
            "frames": {"ids": [frame]},
            "model": {"name": "pointpillars", "source": "saved", "predictions_dir": str(preds)},
            "preprocessing": [],
            "seed": 42,
            "output": {"directory": str(tmp_path / "runs"), "artifact_prefix": "result"},
        },
        base_dir=tmp_path,
    )
    assert run_experiment(config).status == "success"
    return config.run_directory / "result.manifest.json", config


def change_artifact(manifest_path, key, change):
    manifest = json.loads(manifest_path.read_text())
    p = manifest_path.parent / manifest["artifacts"][key]["file"]
    d = json.loads(p.read_text())
    change(d)
    p.write_text(json.dumps(d))
    manifest["artifacts"][key]["sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))


def test_completed_run_integrity_and_evaluation_persistence(completed, tmp_path):
    path, config = completed
    before = {p: p.read_bytes() for p in path.parent.iterdir()}
    result = evaluate_experiment(path)
    assert result.status == "success" and result.purpose == "validation_subset"
    assert result.frame_ids == ("000001",) and len(result.metrics) == 18
    assert result.evaluated_frame_count == 1 and result.to_dict()["evaluated_frame_count"] == 1
    assert next(m for m in result.metrics if m.class_name == "Car").eligible_ground_truth == 1
    assert any("fewer than 41" in w for w in result.warnings)
    assert result.provenance["evaluation_inputs"][0]["image"]["source"] == "PNG-IHDR"
    output = tmp_path / "evidence.json"
    save_evaluation_result(output, result)
    assert json.loads(output.read_text())["evaluation_schema_version"] == 1
    with pytest.raises(FileExistsError):
        save_evaluation_result(output, result)
    assert {p: p.read_bytes() for p in path.parent.iterdir()} == before
    assert load_completed_run(path).config.config_sha256 == config.config_sha256


@pytest.mark.parametrize(
    "problem", ["checksum", "failed", "schema", "missing", "unsafe", "frame", "coordinate", "split"]
)
def test_corrupted_or_incompatible_runs_have_no_metrics(completed, problem):
    path, _ = completed
    m = json.loads(path.read_text())
    if problem == "checksum":
        (path.parent / m["artifacts"]["results"]["file"]).write_text("{}")
    if problem == "failed":
        m["status"] = "failed"
        path.write_text(json.dumps(m))
    if problem == "schema":
        m["run_schema_version"] = 2
        path.write_text(json.dumps(m))
    if problem == "missing":
        del m["artifacts"]["provenance"]
        path.write_text(json.dumps(m))
    if problem == "unsafe":
        m["artifacts"]["results"]["file"] = "../secret.json"
        path.write_text(json.dumps(m))
    if problem == "frame":
        change_artifact(path, "results", lambda d: d["frames"][0].update(frame_id="000002"))
    if problem == "coordinate":
        change_artifact(
            path,
            "results",
            lambda d: d["frames"][0]["predictions"].update(coordinate_frame="lidar"),
        )
    if problem == "split":
        change_artifact(path, "provenance", lambda d: d["split"].update(manifest_sha256="0" * 64))
    result = evaluate_experiment(path)
    assert result.status == "invalid" and result.metrics == () and result.failures


@pytest.mark.parametrize("missing", ["image", "P2", "source"])
def test_missing_genuine_metadata_blocks_without_invented_metrics(completed, missing):
    path, config = completed
    if missing == "image":
        (config.dataset.root / "training/image_2/000001.png").unlink()
    if missing == "source":
        (config.dataset.root / "training/velodyne/000001.bin").unlink()
    if missing == "P2":
        # Rewrite both source and persisted calibration identity to model a valid runner
        # result produced without optional P2, rather than a checksum-corrupt source.
        p = config.dataset.root / "training/calib/000001.txt"
        p.write_text(p.read_text().split("P2:")[0])
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        size = p.stat().st_size

        def results(d):
            d["frames"][0]["inputs"]["calibration"].update(sha256=h, bytes=size)
            del d["frames"][0]["ground_truth"]["calibration"]["P2"]

        def provenance(d):
            d["inputs"][0]["files"]["calibration"].update(sha256=h, bytes=size)

        change_artifact(path, "results", results)
        change_artifact(path, "provenance", provenance)
    result = evaluate_experiment(path)
    assert result.status == "blocked" and result.metrics == () and result.failures


def test_cli_exit_codes_and_fresh_process_without_gpu_or_jit(completed, tmp_path, capsys):
    path, _ = completed
    output = tmp_path / "eval.json"
    main(["evaluate-experiment", str(path), "--no-bev", "--output", str(output)])
    assert json.loads(capsys.readouterr().out)["status"] == "success"
    with pytest.raises(SystemExit) as e:
        main(["evaluate-experiment", str(path), "--output", str(output)])
    assert e.value.code == 2
    capsys.readouterr()
    with pytest.raises(SystemExit) as e:
        main(["evaluate-experiment", str(tmp_path / "missing.json")])
    assert e.value.code == 2
    capsys.readouterr()
    code = """import sys
class Block:
 def find_spec(self,fullname,*args):
  if fullname.split('.')[0] in {'numba','torch','spconv','pcdet','open3d','scipy'}: raise AssertionError(fullname)
sys.meta_path.insert(0,Block())
from raven.evaluation.evaluate import evaluate_experiment
result=evaluate_experiment(sys.argv[1]);assert result.status=='success', result.failures
print(result.status)
"""
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    p = subprocess.run(
        [sys.executable, "-c", code, str(path)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert p.returncode == 0, p.stderr


def test_bundled_reference_sources_are_unmodified_and_pinned():
    root = Path(__file__).resolve().parents[1] / "src/raven/evaluation/_vendor"
    assert all(
        hashlib.sha256((root / name).read_bytes()).hexdigest() == sha
        for name, sha in SOURCE_HASHES.items()
    )
    assert (root / "LICENSE.evaluator").read_text().startswith("MIT License")


def test_actual_image_dimensions_and_crc(tmp_path):
    p = tmp_path / "image.png"
    p.write_bytes(png(83, 47))
    assert image_metadata(p)[0] == (47, 83)
    raw = bytearray(p.read_bytes())
    raw[20] ^= 1
    p.write_bytes(raw)
    with pytest.raises(ValueError, match="checksum"):
        image_metadata(p)


@pytest.mark.parametrize("cls,expected", [("Car", 0), ("Pedestrian", 100), ("Cyclist", 100)])
def test_class_specific_strict_thresholds_at_independent_iou_point_six(cls, expected):
    gt = [annotation(cls) for _ in range(41)]
    dt = [annotation(cls, score=0.9, x=1.0) for _ in gt]  # IoU=(4-1)/(4+1)=0.6.
    metrics, _ = evaluate_annotations(gt, dt)
    assert all(m["ap_r40_percent"] == expected for m in metrics if m["class_name"] == cls)


def test_ignored_van_detection_consumed_without_false_positive():
    gt = []
    dt = []
    for _ in range(41):
        car, van = annotation(), annotation("Van", x=10)
        gt.append({k: np.concatenate([car[k], van[k]]) for k in car})
        a, b = annotation(score=0.9), annotation(score=0.9, x=10)
        dt.append({k: np.concatenate([a[k], b[k]]) for k in a})
    metrics, diagnostics = evaluate_annotations(gt, dt, include_bev=False)
    assert all(m["ap_r40_percent"] == 100 for m in metrics if m["class_name"] == "Car")
    assert all(
        (d["tp"], d["fp"], d["fn"]) == (1, 0, 0) for d in diagnostics if d["class_name"] == "Car"
    )


def test_no_eligible_gt_is_explicitly_unavailable():
    base = annotation()
    gt = [{k: v[:0].copy() for k, v in base.items()}]
    metrics, diagnostics = evaluate_annotations(gt, [annotation(score=0.9)])
    assert all(m["ap_r40_percent"] is None for m in metrics)
    assert all(m["availability"] == "no_eligible_ground_truth" for m in metrics)
    assert next(d for d in diagnostics if d["class_name"] == "Car")["fp"] == 1


def test_empty_predictions_can_evaluate_without_unneeded_image_dimensions(completed):
    from dataclasses import replace

    _, config = completed
    source = config.model.predictions_dir / "000001.json"
    d = detections_to_document([], frame_id="000001")
    d["input"].update(
        sha256=hashlib.sha256(
            (config.dataset.root / "training/velodyne/000001.bin").read_bytes()
        ).hexdigest(),
        points=1,
    )
    source.write_text(json.dumps(d))
    (config.dataset.root / "training/image_2/000001.png").unlink()
    c = replace(config, experiment_id="empty-predictions")
    assert run_experiment(c).status == "success"
    result = evaluate_experiment(c.run_directory / "result.manifest.json")
    assert result.status == "success"
    assert all(m.ap_r40_percent == 0 for m in result.metrics if m.class_name == "Car")
    assert (
        result.provenance["evaluation_inputs"][0]["image"]["availability"]
        == "not_required_for_empty_predictions"
    )


def test_training_smoke_explicit_and_cli_blocked_exit(completed, capsys):
    import shutil
    from dataclasses import replace

    _, config = completed
    for directory, extension in [
        ("velodyne", ".bin"),
        ("label_2", ".txt"),
        ("calib", ".txt"),
        ("image_2", ".png"),
    ]:
        root = config.dataset.root / "training" / directory
        shutil.copyfile(root / ("000001" + extension), root / ("000000" + extension))
    source = json.loads((config.model.predictions_dir / "000001.json").read_text())
    source["input"]["frame_id"] = "000000"
    (config.model.predictions_dir / "000000.json").write_text(json.dumps(source))
    c = replace(
        config,
        experiment_id="training-smoke",
        dataset=replace(config.dataset, split="train"),
        frames=replace(config.frames, ids=("000000",)),
    )
    assert run_experiment(c).status == "success"
    manifest = c.run_directory / "result.manifest.json"
    blocked = evaluate_experiment(manifest)
    assert blocked.status == "blocked" and not blocked.metrics
    smoke = evaluate_experiment(manifest, allow_training_smoke=True)
    assert smoke.status == "success" and smoke.purpose == "training_smoke"
    assert any("not representative" in w for w in smoke.warnings)
    with pytest.raises(SystemExit) as exc:
        main(["evaluate-experiment", str(manifest)])
    assert exc.value.code == 3
    assert json.loads(capsys.readouterr().out)["status"] == "blocked"


def test_test_split_is_never_measured_ap(completed):
    from dataclasses import replace

    from raven.evaluation.inputs import EvaluationBlocked, prepare_sources

    path, config = completed
    run = load_completed_run(path)
    run = replace(run, config=replace(config, dataset=replace(config.dataset, split="test")))
    with pytest.raises(EvaluationBlocked, match="unlabeled"):
        prepare_sources(run, allow_training_smoke=True)


def test_run_ground_truth_cannot_be_substituted_after_checksums_recomputed(completed):
    path, _ = completed
    change_artifact(
        path,
        "results",
        lambda d: d["frames"][0]["ground_truth"]["objects"][0].update(label="Cyclist"),
    )
    result = evaluate_experiment(path)
    assert result.status == "invalid" and not result.metrics
    assert "ground truth" in result.failures[0]


def test_detections_never_match_ground_truth_from_another_frame():
    g = annotation()
    empty_gt = {k: v[:0].copy() for k, v in g.items()}
    metrics, diagnostics = evaluate_annotations(
        [g, empty_gt], [empty_prediction(), annotation(score=0.9)], include_bev=False
    )
    assert all(m["ap_r40_percent"] == 0 for m in metrics if m["class_name"] == "Car")
    car = [d for d in diagnostics if d["class_name"] == "Car" and d["difficulty"] == "easy"]
    assert [(d["tp"], d["fp"], d["fn"]) for d in car] == [(0, 0, 1), (0, 1, 0)]


def test_visibility_sentinels_are_not_fabricated_as_easy_ground_truth(completed):
    from dataclasses import replace

    _, config = completed
    p = config.dataset.root / "training/label_2/000001.txt"
    p.write_text(p.read_text().replace("Car 0 0 0", "Car -1 -1 0"))
    c = replace(config, experiment_id="unavailable-visibility")
    assert run_experiment(c).status == "success"
    result = evaluate_experiment(c.run_directory / "result.manifest.json")
    assert result.status == "blocked" and not result.metrics
    assert "visibility" in result.failures[0]


def test_reordered_results_join_by_exact_frame_ids(completed):
    import shutil
    from dataclasses import replace

    _, config = completed
    for directory, extension in [
        ("velodyne", ".bin"),
        ("label_2", ".txt"),
        ("calib", ".txt"),
        ("image_2", ".png"),
    ]:
        root = config.dataset.root / "training" / directory
        shutil.copyfile(root / ("000001" + extension), root / ("000002" + extension))
    labels = config.dataset.root / "training/label_2/000002.txt"
    labels.write_text(labels.read_text().replace("Car", "Pedestrian"))
    doc = detections_to_document(
        [Detection3D((20, 0, 0), (4, 2, 2), 0, "Pedestrian", 0.9, "kitti_velodyne")],
        frame_id="000002",
    )
    doc["input"].update(
        sha256=hashlib.sha256(
            (config.dataset.root / "training/velodyne/000002.bin").read_bytes()
        ).hexdigest(),
        points=1,
    )
    (config.model.predictions_dir / "000002.json").write_text(json.dumps(doc))
    c = replace(
        config,
        experiment_id="two-frame-join",
        frames=replace(config.frames, ids=("000002", "000001")),
    )
    assert run_experiment(c).status == "success"
    manifest = c.run_directory / "result.manifest.json"
    change_artifact(manifest, "results", lambda d: d["frames"].reverse())
    result = evaluate_experiment(manifest)
    assert result.status == "success" and result.frame_ids == ("000002", "000001")
    diagnostics = [
        d for d in result.diagnostics if d["difficulty"] == "easy" and d["metric"] == "3d"
    ]
    assert (
        next(
            d for d in diagnostics if d["frame_id"] == "000002" and d["class_name"] == "Pedestrian"
        )["tp"]
        == 1
    )
    assert (
        next(d for d in diagnostics if d["frame_id"] == "000001" and d["class_name"] == "Car")["tp"]
        == 1
    )


def test_processed_saved_prediction_claims_cannot_bypass_config_rule(completed):
    path, _ = completed
    change_artifact(
        path, "results", lambda d: d["frames"][0].update(processed_xyzi_sha256="0" * 64)
    )
    result = evaluate_experiment(path)
    assert result.status == "invalid" and not result.metrics


def test_evaluation_options_do_not_silently_enable_training_smoke(completed):
    path, _ = completed
    result = evaluate_experiment(path, allow_training_smoke="false")
    assert result.status == "invalid" and not result.metrics


def test_rotated_self_iou_is_translation_and_rotation_invariant():
    for angle in [0.0, 0.3, np.pi / 2, -np.pi / 2, np.pi, -2.1]:
        box = np.array([[0.0, 20.0, 4.0, 2.0, angle]])
        assert rotate_iou_cpu(box, box)[0, 0] == pytest.approx(1, abs=1e-12)
        assert rotate_iou_cpu(box, box, 2)[0, 0] == pytest.approx(8, abs=1e-12)


def test_missing_pinned_cpu_geometry_is_blocked(completed, monkeypatch):
    from raven.evaluation.backend import geometry_engine

    geometry_engine.cache_clear()
    monkeypatch.setitem(sys.modules, "shapely", None)
    path, _ = completed
    result = evaluate_experiment(path)
    assert result.status == "blocked" and not result.metrics
    geometry_engine.cache_clear()


def test_nonzero_kitti_projection_depth_offset_is_retained():
    source = calibration()
    matrices = dict(source.matrices)
    projection = matrices["P2"].copy()
    projection[2, 3] = 0.004981016
    matrices["P2"] = projection
    adapter = CalibrationAdapter(KittiCalibration(matrices))
    xyz = np.array([[1.0, 2.0, 20.0]])
    uv, depth = adapter.rect_to_img(xyz)
    np.testing.assert_allclose(uv, [[675.0, 430.0]], atol=1e-12)
    np.testing.assert_allclose(depth, [20.0], atol=1e-12)


def test_frozen_reference_execution_is_reproducible_without_raven_backend(tmp_path):
    generator = Path(__file__).parent / "reference/generate_kitti_reference.py"
    output = tmp_path / "reference.json"
    subprocess.run([sys.executable, str(generator), str(output)], check=True)
    actual = json.loads(output.read_text())
    assert actual == FIXTURE["independent_reference_execution"]
    for case, values in actual["car_ap_r40_percent"].items():
        assert values == [FIXTURE["ap_r40"][case]] * 3
