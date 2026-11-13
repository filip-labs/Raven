"""CPU-only KITTI prediction conversion and RV-022 schema-v1 JSON interoperability."""

import json
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

import numpy as np

from raven.perception.detection.base import Detection3D

CLASS_NAMES = ("Car", "Pedestrian", "Cyclist")
BOX_FORMAT = ["x", "y", "z", "dx", "dy", "dz", "heading"]
FRAME = "kitti_velodyne"
_DETECTION_FIELDS = {
    "center_xyz",
    "size_lwh",
    "yaw_rad",
    "label",
    "score",
    "coordinate_frame",
    "rotation_axis",
}


def _class_names(value):
    if isinstance(value, (str, bytes)):
        raise TypeError("Class names must be a sequence of unique nonempty strings")
    try:
        names = list(value)
    except TypeError as exc:
        raise ValueError("Class names must be a sequence of unique nonempty strings") from exc
    if (
        not names
        or any(
            not isinstance(name, str) or not name.strip() or name != name.strip() for name in names
        )
        or len(set(names)) != len(names)
    ):
        raise ValueError(
            "Class names must be unique nonempty strings without surrounding whitespace"
        )
    return names


def _numeric_array(value, name):
    try:
        array = np.asarray(value)
        # Detect booleans before mixed Python sequences coerce them to integers/floats.
        mixed_bool = not isinstance(value, np.ndarray) and any(
            isinstance(item, (bool, np.bool_)) for item in np.asarray(value, dtype=object).flat
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a rectangular real numeric array") from exc
    if mixed_bool or array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must contain real numeric values, not booleans/strings/objects")
    with np.errstate(over="ignore", invalid="ignore"):
        finite = np.isfinite(array).all() and np.isfinite(array.astype(np.float64)).all()
    if not finite:
        raise ValueError(f"{name} must contain finite numbers representable as Python floats")
    return array


def _validate_arrays(boxes, scores, labels, class_names):
    names = _class_names(class_names)
    boxes = _numeric_array(boxes, "Prediction boxes")
    scores = _numeric_array(scores, "Scores")
    labels = _numeric_array(labels, "Labels")
    if boxes.ndim != 2 or boxes.shape[1] != 7:
        raise ValueError("Prediction boxes must have shape (N, 7)")
    count = len(boxes)
    if scores.shape != (count,) or labels.shape != (count,):
        raise ValueError("Scores and labels must have shape (N,) matching boxes")
    if (boxes[:, 3:6] <= 0).any():
        raise ValueError("Predicted box dimensions must be positive")
    if ((scores < 0) | (scores > 1)).any():
        raise ValueError("Expected probability scores in [0, 1]; raw logits are unsupported")
    if (labels != np.floor(labels)).any() or ((labels < 1) | (labels > len(names))).any():
        raise ValueError("Labels must be one-based class indices from the model configuration")
    return boxes, scores, labels, names


def convert_predictions(boxes, scores, labels, class_names) -> list[Detection3D]:
    """Convert seven-column geometric-center KITTI Velodyne boxes without transforms."""
    boxes, scores, labels, names = _validate_arrays(boxes, scores, labels, class_names)
    return [
        Detection3D(
            tuple(box[:3]),
            tuple(box[3:6]),
            box[6],
            names[int(label) - 1],
            score,
            coordinate_frame=FRAME,
            rotation_axis="z",
        )
        for box, score, label in zip(boxes, scores, labels, strict=True)
    ]


def serialize_predictions(boxes, scores, labels, class_names) -> dict:
    """Retain the existing raw prediction fields; explicit class_names is additive."""
    boxes, scores, labels, names = _validate_arrays(boxes, scores, labels, class_names)
    rows = [
        {
            "box": [float(value) for value in box],
            "score": float(score),
            "label_id": int(label),
            "class_name": names[int(label) - 1],
        }
        for box, score, label in zip(boxes, scores, labels, strict=True)
    ]
    return {
        "coordinate_frame": FRAME,
        "box_format": list(BOX_FORMAT),
        "position_reference": "geometric_center",
        "rotation_axis": "z",
        "units": {"position": "m", "heading": "rad"},
        "class_names": names,
        "total_predictions": len(rows),
        "class_counts": dict(sorted(Counter(row["class_name"] for row in rows).items())),
        "predictions": rows,
    }


def _frame_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]{6}", value):
        raise ValueError("frame_id must be a six-digit KITTI identifier")
    return value


def detections_to_document(detections, *, frame_id: str, class_names=CLASS_NAMES) -> dict:
    """Export the existing schema-v1 envelope, without inventing runtime provenance."""
    _frame_id(frame_id)
    names = _class_names(class_names)
    detections = list(detections)
    for index, box in enumerate(detections):
        if not isinstance(box, Detection3D):
            raise TypeError(f"detections[{index}] must be a validated Detection3D")
        if box.coordinate_frame != FRAME or box.rotation_axis != "z":
            raise ValueError(f"detections[{index}] must use {FRAME} and rotation_axis=z")
        if box.label not in names:
            raise ValueError(f"detections[{index}] label is missing from class_names")
    boxes = np.array(
        [(*box.center_xyz, *box.size_lwh, box.yaw_rad) for box in detections], dtype=np.float64
    ).reshape(-1, 7)
    raw = serialize_predictions(
        boxes,
        [box.score for box in detections],
        [names.index(box.label) + 1 for box in detections],
        names,
    )
    return {
        "schema_version": 1,
        "input": {"frame_id": frame_id, "coordinate_frame": FRAME},
        **raw,
        "detections": [asdict(box) for box in detections],
    }


def detections_from_document(
    document, *, expected_frame_id: str | None = None
) -> list[Detection3D]:
    """Validate a runner/export envelope, including consistency of duplicate representations.

    Older RV-022 documents obtain class names from effective_config.CLASS_NAMES.
    Metadata outside the prediction schema (e.g. runtime provenance) is retained
    by the caller's document, but is not authenticated by this reader.
    """
    if not isinstance(document, Mapping):
        raise TypeError("Prediction document must be an object")
    if type(document.get("schema_version")) is not int or document["schema_version"] != 1:
        raise ValueError("Unsupported prediction schema_version; expected integer 1")
    source = document.get("input")
    if not isinstance(source, Mapping):
        raise TypeError("Prediction document requires input.frame_id and input.coordinate_frame")
    frame_id = _frame_id(source.get("frame_id"))
    if expected_frame_id is not None and frame_id != _frame_id(expected_frame_id):
        raise ValueError("Prediction frame_id does not match expected_frame_id")
    if (
        source.get("coordinate_frame") != FRAME
        or document.get("coordinate_frame") != FRAME
        or document.get("rotation_axis") != "z"
        or document.get("position_reference") != "geometric_center"
        or document.get("box_format") != BOX_FORMAT
        or document.get("units") != {"position": "m", "heading": "rad"}
    ):
        raise ValueError(
            "Prediction coordinate conventions must be KITTI Velodyne, geometric "
            "center, LWH meters and rotation about +Z in radians"
        )
    config = document.get("effective_config", {})
    legacy_names = config.get("CLASS_NAMES") if isinstance(config, Mapping) else None
    names = _class_names(document.get("class_names", legacy_names))
    if legacy_names is not None and names != _class_names(legacy_names):
        raise ValueError("class_names disagree with effective_config.CLASS_NAMES")
    rows = document.get("predictions")
    if not isinstance(rows, list):
        raise TypeError("predictions must be a list")
    for index, row in enumerate(rows):
        if (
            not isinstance(row, Mapping)
            or set(row) != {"box", "score", "label_id", "class_name"}
            or type(row["label_id"]) is not int
        ):
            raise ValueError(
                f"predictions[{index}] requires box, score, integer label_id, class_name"
            )
    raw_boxes = [row["box"] for row in rows] if rows else np.empty((0, 7))
    converted = convert_predictions(
        raw_boxes, [row["score"] for row in rows], [row["label_id"] for row in rows], names
    )
    for index, (row, box) in enumerate(zip(rows, converted, strict=True)):
        if row["class_name"] != box.label:
            raise ValueError(f"predictions[{index}] class_name disagrees with label_id")
    total, counts = document.get("total_predictions"), document.get("class_counts")
    expected_counts = dict(Counter(box.label for box in converted))
    if (
        type(total) is not int
        or total != len(converted)
        or not isinstance(counts, dict)
        or any(type(value) is not int for value in counts.values())
        or counts != expected_counts
    ):
        raise ValueError("total_predictions/class_counts disagree with prediction records")
    if "detections" not in document:
        return converted
    records = document["detections"]
    if not isinstance(records, list):
        raise TypeError("detections must be a list")
    detections = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping) or set(record) != _DETECTION_FIELDS:
            raise ValueError(f"detections[{index}] requires all seven Detection3D fields")
        try:
            detections.append(Detection3D(**record))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"detections[{index}]: {exc}") from exc
    if detections != converted:
        raise ValueError(
            "detections disagree with raw predictions (geometry, class, score or frame)"
        )
    return detections


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"Nonfinite JSON constant is unsupported: {value}")


def load_detections(path: str | Path, *, expected_frame_id: str | None = None) -> list[Detection3D]:
    """Read a JSON capture without GPU/GUI imports or source-file modifications."""
    with Path(path).open(encoding="utf-8") as stream:
        document = json.load(
            stream, object_pairs_hook=_json_object, parse_constant=_reject_constant
        )
    return detections_from_document(document, expected_frame_id=expected_frame_id)


def save_detections(
    path: str | Path, detections, *, frame_id: str, class_names=CLASS_NAMES
) -> None:
    """Write a new schema-v1 file; never overwrite an existing capture."""
    document = detections_to_document(detections, frame_id=frame_id, class_names=class_names)
    text = json.dumps(document, indent=2, allow_nan=False) + "\n"
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(text)
