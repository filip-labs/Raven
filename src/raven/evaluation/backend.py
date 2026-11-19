"""Pinned reference function bodies with CPU-only dependency/launch adapters."""

import ast
import copy
import hashlib
import io
import json
import math
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np

CLASSES = ("Car", "Pedestrian", "Cyclist")
DIFFICULTIES = ("easy", "moderate", "hard")
THRESHOLDS = (0.7, 0.5, 0.5)
REVISION = "233f849829b6ac19afb8af8837a0246890908755"
SOURCE_HASHES = {
    "eval.py": "d9260a41f68083d685fc43fecac1c417b14a1bc53df4915a46b2924f3a6a3c64",
    "rotate_iou.py": "d9b2820b5be5d15ee7233147643134c5d73eff129c773aef4b59ce73a8ecd7f4",
    "box_utils.py": "9d4d769dbcc63619d1715a6f3ceb952ae4e4790b938b770a0ec29b33683fb73d",
}
IDENTITY = {
    "name": "OpenPCDet KITTI evaluator / Raven CPU adapter",
    "version": "1",
    "reference_revision": REVISION,
    "source_sha256": SOURCE_HASHES,
    "protocol": "AP_R40",
    "official_server_equivalence": "not_certified",
    "overlap_precision": "float64 GEOS oriented polygon intersection + reference vertical overlap",
    "geometry_dependency": "shapely==2.1.2",
    "geometry_fixture_atol": 1e-10,
    "camera_center_fixture_atol_m": 1e-12,
    "projection_fixture_atol_px": 1e-4,
    "ap_fixture_atol_percent": 1e-8,
    "parameters": {
        "iou_thresholds": dict(zip(CLASSES, THRESHOLDS, strict=True)),
        "iou_match_comparison": ">",
        "recall_bins_averaged": list(range(1, 41)),
        "precision_interpolation": "maximum of remaining recall bins",
        "difficulty": {
            "easy": {
                "gt_height_exclusive_min": 40,
                "dt_height_inclusive_min": 40,
                "max_occlusion": 0,
                "max_truncation": 0.15,
            },
            "moderate": {
                "gt_height_exclusive_min": 25,
                "dt_height_inclusive_min": 25,
                "max_occlusion": 1,
                "max_truncation": 0.3,
            },
            "hard": {
                "gt_height_exclusive_min": 25,
                "dt_height_inclusive_min": 25,
                "max_occlusion": 2,
                "max_truncation": 0.5,
            },
        },
        "ignored_neighbors": {"Car": ["Van"], "Pedestrian": ["Person_sitting"]},
        "dontcare": "2D suppression only in this reference; no 3D/BEV suppression",
        "camera_conversion": "OpenPCDet-KITTI-upright-v1",
        "camera_axis_deviation_limit_rad": 0.05,
    },
}


def _functions(filename, namespace, allowed=None):
    path = Path(__file__).with_name("_vendor") / filename
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_HASHES[filename]:
        raise ValueError(f"pinned evaluator source checksum mismatch: {filename}")
    tree = ast.parse(raw, filename=str(path))
    nodes = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and (allowed is None or n.name in allowed)
    ]
    if allowed is not None and {n.name for n in nodes} != set(allowed):
        raise ValueError("pinned evaluator function inventory mismatch")
    for node in nodes:
        node.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    # Only immutable, checksum-pinned bundled source; never a caller-supplied path.
    exec(compile(module, str(path), "exec"), namespace)  # noqa: S102
    return SimpleNamespace(**{n.name: namespace[n.name] for n in nodes})


class EvaluationBackendUnavailable(RuntimeError):
    """The pinned optional CPU geometry dependency is unavailable."""


@lru_cache(maxsize=1)
def geometry_engine():
    try:
        import shapely
    except (ImportError, OSError) as exc:
        raise EvaluationBackendUnavailable(
            "CPU evaluation requires shapely==2.1.2; install .[evaluation]"
        ) from exc
    if shapely.__version__ != "2.1.2":
        raise EvaluationBackendUnavailable(f"expected shapely==2.1.2, got {shapely.__version__}")
    return shapely


def geometry_identity():
    engine = geometry_engine()
    return {
        "name": "GEOS oriented polygon intersection",
        "shapely": engine.__version__,
        "geos": engine.geos_version_string,
        "arithmetic": "float64",
    }


def rotate_iou_cpu(boxes, queries, criterion=-1):
    """Exact oriented-rectangle polygons; no AABB/voxel/raster overlap approximation.

    Criteria match the reference interface: -1 IoU, 0/1 coverage, else intersection
    area. Nonpositive DontCare sentinel dimensions yield unused zero cells.
    """
    boxes, queries = np.asarray(boxes, dtype=np.float64), np.asarray(queries, dtype=np.float64)
    if boxes.ndim != 2 or queries.ndim != 2 or boxes.shape[1] != 5 or queries.shape[1] != 5:
        raise ValueError("rotated overlap requires (N,5) camera x/z/length/width/yaw")
    if not np.isfinite(boxes).all() or not np.isfinite(queries).all():
        raise ValueError("overlap geometry must be finite")
    engine = geometry_engine()

    def polygon(box):
        x, z, length, width, angle = box
        if length <= 0 or width <= 0:
            return None  # Only irrelevant DontCare sentinel cells reach this boundary.
        local = np.array(
            [
                [-length / 2, -width / 2],
                [length / 2, -width / 2],
                [length / 2, width / 2],
                [-length / 2, width / 2],
            ]
        )
        c, s = math.cos(angle), math.sin(angle)
        vertices = local @ np.array([[c, -s], [s, c]]) + np.array([x, z])
        result = engine.Polygon(vertices)
        if not result.is_valid or result.area <= 0:
            raise ValueError("oriented geometry is unresolved in float64 precision")
        return result

    result = np.zeros((len(boxes), len(queries)), dtype=np.float64)
    a, b = [polygon(v) for v in boxes], [polygon(v) for v in queries]
    try:
        for i, p in enumerate(a):
            for j, q in enumerate(b):
                if p is None or q is None:
                    continue
                intersection = p.intersection(q).area
                denominator = (
                    p.area + q.area - intersection
                    if criterion == -1
                    else p.area
                    if criterion == 0
                    else q.area
                    if criterion == 1
                    else 1.0
                )
                result[i, j] = intersection / denominator
    except engine.errors.GEOSException as exc:
        raise ValueError(f"GEOS intersection failed: {exc}") from exc
    if not np.isfinite(result).all():
        raise ValueError("undefined oriented overlap")
    return result


@lru_cache(maxsize=1)
def metric_reference():
    return _functions("eval.py", {"np": np, "sysio": io, "rotate_iou_gpu_eval": rotate_iou_cpu})


@lru_cache(maxsize=1)
def conversion_reference():
    return _functions(
        "box_utils.py",
        {"np": np, "copy": copy},
        (
            "boxes3d_lidar_to_kitti_camera",
            "boxes3d_to_corners3d_kitti_camera",
            "boxes3d_kitti_camera_to_imageboxes",
        ),
    )


def evaluate_annotations(ground_truth, detections, *, include_bev=True):
    """Reference AP_R40 and independent-at-score-zero count diagnostics; no AOS/AP11."""
    if not ground_truth or len(ground_truth) != len(detections):
        raise ValueError("evaluation requires aligned nonempty frame sequences")
    reference = metric_reference()
    overlap_thresholds = np.tile(np.array(THRESHOLDS), (1, 3, 1))
    metrics, diagnostics = [], []
    for metric, name in ((2, "3d"), (1, "bev")):
        if metric == 1 and not include_bev:
            continue
        # One part per frame avoids unused cross-frame pairs without changing matching/AP.
        evaluated = reference.eval_class(
            ground_truth,
            detections,
            [0, 1, 2],
            [0, 1, 2],
            metric,
            overlap_thresholds,
            num_parts=len(ground_truth),
        )
        values = reference.get_mAP_R40(evaluated["precision"])
        overlaps = reference.calculate_iou_partly(
            detections, ground_truth, metric, num_parts=len(ground_truth)
        )[0]
        for c, cls in enumerate(CLASSES):
            for d, difficulty in enumerate(DIFFICULTIES):
                prepared = reference._prepare_data(ground_truth, detections, c, d)
                eligible = int(prepared[-1])
                metrics.append(
                    {
                        "metric": name,
                        "class_name": cls,
                        "difficulty": difficulty,
                        "iou_threshold": THRESHOLDS[c],
                        "eligible_ground_truth": eligible,
                        "availability": "available" if eligible else "no_eligible_ground_truth",
                        "ap_r40_percent": float(values[c, d, 0]) if eligible else None,
                    }
                )
                for i in range(len(ground_truth)):
                    gt_data, dt_data, ignored_gt, ignored_dt, dc = (
                        prepared[k][i] for k in range(5)
                    )
                    tp, fp, fn, _, _ = reference.compute_statistics_jit(
                        overlaps[i],
                        gt_data,
                        dt_data,
                        ignored_gt,
                        ignored_dt,
                        dc,
                        metric,
                        THRESHOLDS[c],
                        thresh=0.0,
                        compute_fp=True,
                    )
                    diagnostics.append(
                        {
                            "frame_index": i,
                            "metric": name,
                            "class_name": cls,
                            "difficulty": difficulty,
                            "score_threshold": 0.0,
                            "tp": int(tp),
                            "fp": int(fp),
                            "fn": int(fn),
                            "matching_details": "not_exposed_by_reference",
                        }
                    )
    # Force finite JSON values; undefined metrics are deliberately null.
    json.dumps(metrics, allow_nan=False)
    return metrics, diagnostics
