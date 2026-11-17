"""Headless KITTI ground-truth loading and wireframe construction."""

from collections.abc import Iterable
from pathlib import Path

import numpy as np

from raven.perception.data.kitti_annotations import load_kitti_labels
from raven.perception.data.kitti_calibration import (
    VELODYNE_FRAME,
    OrientedBox3D,
    load_kitti_calibration,
    transform_annotation,
)

# OrientedBox3D lower face 0..3, upper face 4..7, then corresponding vertical edges.
BOX_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)


def class_color(label: str) -> tuple[float, float, float]:
    """Car green, Pedestrian red, Cyclist blue; all other names yellow."""
    return {"Car": (0.0, 1.0, 0.0), "Pedestrian": (1.0, 0.0, 0.0), "Cyclist": (0.0, 0.0, 1.0)}.get(
        label, (1.0, 1.0, 0.0)
    )


def wireframe_from_corners(corners: np.ndarray, label: str):
    """Copy eight ordered finite distinct corners into a twelve-edge wireframe."""
    if np.iscomplexobj(corners):
        raise ValueError("Box corners must be real-valued")
    try:
        points = np.array(corners, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Box corners must contain real finite numbers") from exc
    if points.shape != (8, 3) or not np.isfinite(points).all():
        raise ValueError("Box corners must have shape (8, 3) and finite coordinates")
    if len(np.unique(points, axis=0)) != 8:
        raise ValueError("Box corners must contain eight distinct vertices")
    edges = np.array(BOX_EDGES, dtype=np.int32)
    colors = np.tile(class_color(label), (12, 1))
    return points, edges, colors


def box_wireframes(boxes: Iterable[OrientedBox3D], coordinate_frame: str):
    """Batch box geometry into one line set, checking alignment before rendering."""
    vertices, lines, colors = [], [], []
    for index, box in enumerate(boxes):
        if not isinstance(box, OrientedBox3D):
            raise TypeError("Boxes must use OrientedBox3D geometry")
        if box.coordinate_frame != coordinate_frame:
            raise ValueError(
                f"Box frame {box.coordinate_frame!r} does not match cloud frame "
                f"{coordinate_frame!r}; transform boxes before rendering"
            )
        points, edges, rgb = wireframe_from_corners(box.corners, box.label)
        vertices.append(points)
        lines.append(edges + index * 8)
        colors.append(rgb)
    if not vertices:
        return np.empty((0, 3)), np.empty((0, 2), dtype=np.int32), np.empty((0, 3))
    return np.concatenate(vertices), np.concatenate(lines), np.concatenate(colors)


def load_ground_truth_boxes(
    scan_path: str | Path,
    *,
    labels_path: str | Path | None = None,
    calibration_path: str | Path | None = None,
) -> tuple[OrientedBox3D, ...]:
    """Resolve matching KITTI files and transform valid annotations into Velodyne.

    Automatic paths require split/velodyne/FRAME.bin. Explicit paths may override
    either sibling file; all three filename stems must match exactly.
    """
    scan_path = Path(scan_path)
    if not scan_path.is_file():
        raise FileNotFoundError(f"LiDAR scan not found: {scan_path}")
    if (labels_path is None or calibration_path is None) and scan_path.parent.name != "velodyne":
        raise ValueError(
            "Automatic ground-truth paths require a velodyne/ directory; "
            "provide both --labels and --calib for nonstandard layouts"
        )
    split = scan_path.parent.parent
    labels_path = (
        Path(labels_path)
        if labels_path is not None
        else split / "label_2" / f"{scan_path.stem}.txt"
    )
    calibration_path = (
        Path(calibration_path)
        if calibration_path is not None
        else split / "calib" / f"{scan_path.stem}.txt"
    )
    for role, path in (("labels", labels_path), ("calibration", calibration_path)):
        if path.stem != scan_path.stem:
            raise ValueError(f"Frame ID mismatch: scan {scan_path.stem!r}, {role} {path.stem!r}")
        if not path.is_file():
            raise FileNotFoundError(f"Missing ground-truth {role} file: {path}")
    labels = load_kitti_labels(labels_path)
    calibration = load_kitti_calibration(calibration_path)
    boxes = []
    for record in labels.records:
        geometry = transform_annotation(record, calibration, target_frame=VELODYNE_FRAME)
        if geometry is not None:
            boxes.append(geometry)
    return tuple(boxes)



def prediction_corners(detection, coordinate_frame: str):
    """Use the existing corner geometry with an exact +Z rotation; no frame transform."""
    from raven.perception.data.kitti_calibration import box_corners
    from raven.perception.detection.base import Detection3D

    if not isinstance(detection, Detection3D):
        raise TypeError("Predictions must use Detection3D")
    if detection.coordinate_frame != coordinate_frame or coordinate_frame != VELODYNE_FRAME:
        raise ValueError("Prediction and cloud frames must both be kitti_velodyne")
    if detection.rotation_axis != "z":
        raise ValueError("KITTI Velodyne predictions must rotate about Z")
    c, s = np.cos(detection.yaw_rad), np.sin(detection.yaw_rad)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return box_corners(detection.center_xyz, detection.size_lwh, rotation)


def prediction_wireframes(predictions, coordinate_frame: str):
    """Class-colored dashed box edges plus an interior heading arrow and center crossbar.

    Four exact collinear dash segments per edge. The glyph stays inside the box,
    helping distinguish coincident layers without moving or inflating the box.
    """
    segments = []
    segment_colors = []
    for detection in predictions:
        corners = prediction_corners(detection, coordinate_frame)
        points, edges, _ = wireframe_from_corners(corners, detection.label)
        before = len(segments)
        for start, end in edges:
            vector = points[end] - points[start]
            for part in range(0, 8, 2):
                segments.append(
                    [points[start] + (part / 8) * vector, points[start] + ((part + 1) / 8) * vector]
                )
        center = np.asarray(detection.center_xyz)
        # Derive glyph axes from the same ordered corners, retaining exact pose.
        length_axis = (corners[1] - corners[0]) / 2
        width_axis = (corners[3] - corners[0]) / 2
        tip = center + length_axis
        segments.extend(
            [
                [center, tip],
                [tip, tip - 0.3 * length_axis + 0.3 * width_axis],
                [tip, tip - 0.3 * length_axis - 0.3 * width_axis],
                [center - 0.2 * width_axis, center + 0.2 * width_axis],
            ]
        )
        segment_colors.extend([class_color(detection.label)] * (len(segments) - before))
    if not segments:
        return np.empty((0, 3)), np.empty((0, 2), dtype=np.int32), np.empty((0, 3))
    vertices = np.asarray(segments).reshape(-1, 3)
    lines = np.arange(len(vertices), dtype=np.int32).reshape(-1, 2)
    return vertices, lines, np.asarray(segment_colors, dtype=np.float64)
