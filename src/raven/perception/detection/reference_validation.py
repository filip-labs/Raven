"""CPU-only, provenance-gated numerical comparison of schema-v1 prediction captures."""

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from raven.perception.detection.predictions import (
    _json_object,
    _reject_constant,
    detections_from_document,
)

REFERENCE_ENTRYPOINT = "OpenPCDet.DemoDataset+model.forward"
CANDIDATE_ENTRYPOINT = "PointPillarsDetector.detect"


@dataclass(frozen=True)
class Tolerances:
    position_atol: float = 1e-4
    dimension_atol: float = 1e-4
    yaw_atol: float = 1e-4
    score_atol: float = 1e-5
    rtol: float = 1e-5
    max_match_distance: float = 1.0

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a finite number")
            if (
                not math.isfinite(value)
                or value < 0
                or (name == "max_match_distance" and value == 0)
            ):
                raise ValueError(f"{name} must be finite and nonnegative (matching distance > 0)")


def _wrap(angle):
    return math.remainder(angle, 2 * math.pi)


def _assignment(cost):
    """Rectangular Hungarian minimum-cost assignment, rows <= columns.

    Strict comparisons and ascending traversal resolve equal-cost ties repeatably.
    """
    n, m = cost.shape
    u, v = np.zeros(n + 1), np.zeros(m + 1)
    p, way = np.zeros(m + 1, dtype=int), np.zeros(m + 1, dtype=int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minimum = np.full(m + 1, np.inf)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], np.inf, 0
            for j in range(1, m + 1):
                if not used[j]:
                    current = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if current < minimum[j]:
                        minimum[j], way[j] = current, j0
                    if minimum[j] < delta:
                        delta, j1 = minimum[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minimum[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    return [(p[j] - 1, j - 1) for j in range(1, m + 1) if p[j]]


def _match(reference, candidate, gate):
    pairs = []

    def key(box):
        return (*box.center_xyz, *box.size_lwh, _wrap(box.yaw_rad), box.score)

    for label in sorted({box.label for box in reference} | {box.label for box in candidate}):
        left = sorted(
            (i for i, box in enumerate(reference) if box.label == label),
            key=lambda i: (key(reference[i]), i),
        )
        right = sorted(
            (i for i, box in enumerate(candidate) if box.label == label),
            key=lambda i: (key(candidate[i]), i),
        )
        if not left or not right:
            continue
        distances = np.array(
            [
                [math.dist(reference[i].center_xyz, candidate[j].center_xyz) for j in right]
                for i in left
            ]
        )
        # A dummy costs more than every possible real-distance improvement combined.
        penalty = (len(left) + len(right) + 1) * (gate + 1)
        if not math.isfinite(2 * penalty):
            raise ValueError("Matching distance/count exceeds finite assignment range")
        cost = np.full((len(left), len(right) + len(left)), penalty)
        cost[:, : len(right)] = np.where(distances <= gate, distances, 2 * penalty)
        for i, j in _assignment(cost):
            if j < len(right) and distances[i, j] <= gate:
                pairs.append((left[i], right[j]))
    return sorted(pairs)


def _field(document, path):
    value = document
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _provenance(reference, candidate):
    errors = []
    digests = {
        "input.sha256": 64,
        "checkpoint.sha256": 64,
        "config.sha256": 64,
        "openpcdet_commit": 40,
        "raven_git_commit": 40,
        "openpcdet_diff_sha256": 64,
        "runner_sha256": 64,
    }
    required_text = [
        "config.path",
        "environment.gpu",
        "environment.python",
        "environment.pytorch",
        "environment.torch_cuda_runtime",
        "environment.openpcdet",
        "environment.numpy",
        "environment.spconv_cu126",
    ]
    for role, doc in (("reference", reference), ("candidate", candidate)):
        for path, width in digests.items():
            value = _field(doc, path)
            if not isinstance(value, str) or not re.fullmatch(rf"[0-9a-f]{{{width}}}", value):
                errors.append(f"{role}: missing/invalid {path}")
        for path in required_text:
            value = _field(doc, path)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{role}: missing/invalid {path}")
        if type(_field(doc, "input.points")) is not int or _field(doc, "input.points") <= 0:
            errors.append(f"{role}: missing/invalid input.points")
        flags = _field(doc, "environment.environment_flags")
        if (
            not isinstance(flags, dict)
            or flags.get("CUMM_DISABLE_JIT") != "1"
            or flags.get("SPCONV_DISABLE_JIT") != "1"
            or not isinstance(flags.get("TORCH_CUDA_ARCH_LIST"), str)
            or not flags["TORCH_CUDA_ARCH_LIST"]
        ):
            errors.append(f"{role}: missing/invalid environment.environment_flags")
        run_id = doc.get("inference_run_id")
        if not isinstance(run_id, str) or not re.fullmatch(r"[0-9a-f]{32}", run_id):
            errors.append(f"{role}: missing/invalid inference_run_id")
        if doc.get("raven_modified_files") != []:
            errors.append(f"{role}: missing or dirty raven_modified_files")
        if not isinstance(doc.get("openpcdet_modified_files"), list):
            errors.append(f"{role}: missing openpcdet_modified_files")
        config = doc.get("effective_config")
        for section in ("MODEL", "DATA_CONFIG", "CLASS_NAMES"):
            if (
                not isinstance(config, dict)
                or not config.get(section)
                or (section != "CLASS_NAMES" and not isinstance(config[section], dict))
            ):
                errors.append(f"{role}: missing effective_config.{section}")
    dataset_digest = reference.get("reference_dataset_sha256")
    if not isinstance(dataset_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", dataset_digest):
        errors.append("reference: missing/invalid reference_dataset_sha256")
    equal_paths = [
        "input.frame_id",
        "input.points",
        "environment.environment_flags",
        "input.sha256",
        "checkpoint.sha256",
        "config.sha256",
        "openpcdet_commit",
        "raven_git_commit",
        "openpcdet_diff_sha256",
        "openpcdet_modified_files",
        *required_text[1:],
    ]
    for path in equal_paths:
        if _field(reference, path) != _field(candidate, path):
            errors.append(f"provenance mismatch: {path}")
    for section in ("MODEL", "DATA_CONFIG", "CLASS_NAMES"):
        path = f"effective_config.{section}"
        if json.dumps(_field(reference, path), sort_keys=True, allow_nan=False) != json.dumps(
            _field(candidate, path), sort_keys=True, allow_nan=False
        ):
            errors.append(f"inference settings mismatch: {path}")
    if reference.get("inference_entrypoint") != REFERENCE_ENTRYPOINT:
        errors.append("reference must use the independent OpenPCDet reference entrypoint")
    if candidate.get("inference_entrypoint") != CANDIDATE_ENTRYPOINT:
        errors.append("candidate must invoke PointPillarsDetector.detect")
    if reference.get("inference_run_id") == candidate.get("inference_run_id"):
        errors.append("inference_run_id must differ: two exports of one run are not a reference")
    if reference.get("runner_sha256") == candidate.get("runner_sha256"):
        errors.append("reference and candidate must identify different runner source files")
    return errors


def compare_documents(reference, candidate, tolerances=None):
    tolerances = tolerances or Tolerances()
    left, right = detections_from_document(reference), detections_from_document(candidate)
    if reference["input"]["frame_id"] != candidate["input"]["frame_id"]:
        raise ValueError("Reference and candidate frame_id must match")
    failures = _provenance(reference, candidate)
    provenance_pass = not failures
    pairs = _match(left, right, tolerances.max_match_distance)
    unmatched_left = sorted(set(range(len(left))) - {i for i, _ in pairs})
    unmatched_right = sorted(set(range(len(right))) - {j for _, j in pairs})
    left_counts, right_counts = (
        dict(Counter(b.label for b in left)),
        dict(Counter(b.label for b in right)),
    )
    if len(left) != len(right):
        failures.append("Prediction counts differ")
    if left_counts != right_counts:
        failures.append("Class distributions differ")
    if unmatched_left or unmatched_right:
        failures.append("Unmatched detections")
    records = []
    for i, j in pairs:
        a, b = left[i], right[j]
        position = [abs(x - y) for x, y in zip(a.center_xyz, b.center_xyz, strict=True)]
        dimensions = [abs(x - y) for x, y in zip(a.size_lwh, b.size_lwh, strict=True)]
        # Wrap operands first to avoid overflow and make 2pi-equivalent representations equal.
        yaw = abs(_wrap(_wrap(a.yaw_rad) - _wrap(b.yaw_rad)))
        score = abs(a.score - b.score)
        valid = (
            all(
                d <= tolerances.position_atol + tolerances.rtol * abs(x)
                for d, x in zip(position, a.center_xyz, strict=True)
            )
            and all(
                d <= tolerances.dimension_atol + tolerances.rtol * abs(x)
                for d, x in zip(dimensions, a.size_lwh, strict=True)
            )
            and yaw <= tolerances.yaw_atol + tolerances.rtol * abs(_wrap(a.yaw_rad))
            and score <= tolerances.score_atol + tolerances.rtol * abs(a.score)
        )
        record = {
            "reference_index": i,
            "candidate_index": j,
            "label": a.label,
            "position_error_m": math.dist(a.center_xyz, b.center_xyz),
            "position_component_errors_m": position,
            "max_dimension_error_m": max(dimensions),
            "yaw_error_rad": yaw,
            "confidence_difference": score,
            "pass": valid,
        }
        records.append(record)
        if not valid:
            failures.append(f"Numerical tolerance exceeded: reference[{i}] candidate[{j}]")

    def maximum(name):
        return max((row[name] for row in records), default=None)

    return {
        "report_schema_version": 1,
        "status": "FAIL" if failures else "PASS",
        "pass": not failures,
        "provenance_pass": provenance_pass,
        "numerical_pass": (
            len(left) == len(right)
            and left_counts == right_counts
            and not unmatched_left
            and not unmatched_right
            and all(row["pass"] for row in records)
        ),
        "failures": failures,
        "frame_id": reference["input"]["frame_id"],
        "tolerances": asdict(tolerances),
        "counts": {
            "reference": len(left),
            "candidate": len(right),
            "reference_classes": left_counts,
            "candidate_classes": right_counts,
        },
        "matched_detections": records,
        "unmatched_reference_indices": unmatched_left,
        "unmatched_candidate_indices": unmatched_right,
        "metrics": {
            "max_position_error_m": maximum("position_error_m"),
            "mean_position_error_m": (
                sum(r["position_error_m"] for r in records) / len(records) if records else None
            ),
            "max_dimension_error_m": maximum("max_dimension_error_m"),
            "max_yaw_error_rad": maximum("yaw_error_rad"),
            "max_confidence_difference": maximum("confidence_difference"),
        },
        "provenance": {
            "reference": _provenance_record(reference),
            "candidate": _provenance_record(candidate),
        },
    }


def _provenance_record(document):
    return {
        key: document.get(key)
        for key in (
            "input",
            "checkpoint",
            "config",
            "effective_config",
            "reference_dataset_sha256",
            "openpcdet_commit",
            "openpcdet_diff_sha256",
            "openpcdet_modified_files",
            "raven_git_commit",
            "raven_modified_files",
            "runner_sha256",
            "environment",
            "inference_entrypoint",
            "inference_run_id",
        )
    }


def compare_files(reference_path, candidate_path, tolerances=None):
    documents, identities = [], {}
    for role, path in (("reference", reference_path), ("candidate", candidate_path)):
        content = Path(path).read_bytes()
        documents.append(
            json.loads(content, object_pairs_hook=_json_object, parse_constant=_reject_constant)
        )
        identities[role] = {"path": str(path), "sha256": hashlib.sha256(content).hexdigest()}
    report = compare_documents(*documents, tolerances=tolerances)
    report["captures"] = identities
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    for name, default in asdict(Tolerances()).items():
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=default)
    args = parser.parse_args(argv)
    try:
        tolerances = Tolerances(**{key: getattr(args, key) for key in asdict(Tolerances())})
        if args.output.exists():
            raise FileExistsError(f"Refusing to overwrite {args.output}")
        try:
            report = compare_files(args.reference, args.candidate, tolerances)
        except (OSError, ValueError, TypeError, OverflowError) as exc:
            report = {
                "report_schema_version": 1,
                "status": "FAIL",
                "pass": False,
                "failures": [f"Input verification failed: {exc}"],
                "tolerances": asdict(tolerances),
            }
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps({"status": report["status"], "output": str(args.output)}))
        return 0 if report["pass"] else 1
    except (OSError, ValueError, TypeError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
