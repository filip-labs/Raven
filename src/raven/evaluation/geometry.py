"""Explicit pinned KITTI upright conversion, full calibration centers and genuine image metadata."""

import hashlib
import struct
import zlib
from pathlib import Path

import numpy as np

from raven.evaluation.backend import conversion_reference
from raven.perception.data.kitti_calibration import (
    CAMERA_RECT_FRAME,
    VELODYNE_FRAME,
    KittiCalibration,
    transform_points,
)


class CalibrationAdapter:
    def __init__(self, calibration: KittiCalibration):
        self.calibration = calibration
        if "P2" not in calibration.matrices:
            raise ValueError("evaluation requires genuine P2 projection calibration")
        self.P2 = calibration.matrices["P2"]
        if not np.allclose(self.P2[2, :3], [0, 0, 1], atol=1e-9, rtol=0):
            raise ValueError("unsupported nonstandard KITTI P2 depth convention")
        if self.P2[0, 0] <= 0 or self.P2[1, 1] <= 0:
            raise ValueError("KITTI camera focal lengths must be positive")
        canonical = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=float)
        deviation = np.arccos(
            np.clip((np.trace(calibration.velo_to_rect[:3, :3] @ canonical.T) - 1) / 2, -1, 1)
        )
        if deviation > 0.05:
            raise ValueError("calibration is outside the supported KITTI upright-axis convention")
        self.axis_deviation_rad = float(deviation)

    def lidar_to_rect(self, points):
        return transform_points(
            points, self.calibration, source_frame=VELODYNE_FRAME, target_frame=CAMERA_RECT_FRAME
        )

    def rect_to_img(self, points):
        homogeneous = np.column_stack((points, np.ones(len(points))))
        projected = homogeneous @ self.P2.T
        with np.errstate(divide="ignore", invalid="ignore"):
            image = projected[:, :2] / points[:, 2, None]
        return image, projected[:, 2] - self.P2[2, 3]


def prediction_annotation(detections, calibration, image_shape):
    adapter = CalibrationAdapter(calibration)
    boxes = np.array(
        [(*b.center_xyz, *b.size_lwh, b.yaw_rad) for b in detections], dtype=float
    ).reshape(-1, 7)
    if any(b.coordinate_frame != VELODYNE_FRAME or b.rotation_axis != "z" for b in detections):
        raise ValueError("evaluation predictions require KITTI Velodyne/+Z conventions")
    reference = conversion_reference()
    # Benchmark contract: subtract half-height in LiDAR, transform bottom center
    # with full R0/Tr, then r_y=-heading-pi/2; NOT arbitrary SO(3) box conversion.
    camera = reference.boxes3d_lidar_to_kitti_camera(boxes, adapter)
    if len(detections) and image_shape is None:
        raise ValueError("nonempty predictions require genuine image dimensions")
    if len(detections):
        bbox = reference.boxes3d_kitti_camera_to_imageboxes(camera, adapter, image_shape)
    else:
        bbox = np.empty((0, 4))
    if not np.isfinite(camera).all() or not np.isfinite(bbox).all():
        raise ValueError(
            "undefined projection/camera geometry; no near-plane approximation is substituted"
        )
    return {
        "name": np.array([b.label for b in detections]),
        "bbox": bbox,
        "dimensions": camera[:, 3:6],
        "location": camera[:, :3],
        "rotation_y": camera[:, 6],
        "alpha": np.full(len(detections), -10.0),
        "score": np.array([b.score for b in detections]),
    }, {
        "convention": "OpenPCDet-KITTI-upright-v1",
        "axis_deviation_rad": adapter.axis_deviation_rad,
        "full_so3_boxes": False,
    }


def ground_truth_annotation(labels):
    records = labels.records
    return {
        "name": np.array([r.object_type for r in records]),
        "bbox": np.array([r.bbox_xyxy for r in records]).reshape(-1, 4),
        "dimensions": np.array(
            [[r.dimensions_hwl[2], r.dimensions_hwl[0], r.dimensions_hwl[1]] for r in records]
        ).reshape(-1, 3),
        "location": np.array([r.location_xyz for r in records]).reshape(-1, 3),
        "rotation_y": np.array([r.rotation_y for r in records]),
        "alpha": np.array([r.alpha for r in records]),
        "occluded": np.array([r.occlusion for r in records]),
        "truncated": np.array([r.truncation for r in records]),
    }


def image_metadata(path: str | Path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"genuine image file unavailable: {path}")
    raw = path.read_bytes()
    if len(raw) < 33 or raw[:8] != b"\x89PNG\r\n\x1a\n" or raw[8:16] != b"\x00\x00\x00\x0dIHDR":
        raise ValueError(f"invalid PNG image header: {path}")
    if zlib.crc32(raw[12:29]) != struct.unpack(">I", raw[29:33])[0]:
        raise ValueError(f"PNG IHDR checksum mismatch: {path}")
    width, height = struct.unpack(">II", raw[16:24])
    if not 0 < width < 2**31 or not 0 < height < 2**31:
        raise ValueError("image dimensions must be positive PNG dimensions")
    offset, seen_data, seen_end = 8, False, False
    while offset < len(raw):
        if offset + 12 > len(raw):
            raise ValueError("truncated PNG chunk")
        length = struct.unpack(">I", raw[offset : offset + 4])[0]
        end = offset + 12 + length
        if end > len(raw):
            raise ValueError("truncated PNG payload")
        kind = raw[offset + 4 : offset + 8]
        if zlib.crc32(raw[offset + 4 : end - 4]) != struct.unpack(">I", raw[end - 4 : end])[0]:
            raise ValueError("PNG chunk checksum mismatch")
        seen_data |= kind == b"IDAT"
        if kind == b"IEND":
            seen_end = length == 0 and end == len(raw)
            break
        offset = end
    if not seen_data or not seen_end:
        raise ValueError("PNG requires complete IDAT/IEND evidence")
    return (height, width), {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "height": height,
        "width": width,
        "source": "PNG-IHDR",
    }
