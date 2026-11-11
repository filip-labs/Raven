"""KITTI object calibration, rigid point transforms, and full-pose box geometry."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType

import numpy as np

from raven.perception.data.kitti_annotations import KITTI_CAMERA_FRAME, KittiAnnotation
from raven.perception.detection.base import GroundTruth3D

VELODYNE_FRAME = "kitti_velodyne"
CAMERA_RECT_FRAME = "kitti_camera_rect"
# KITTI text calibration contains rounded rotations; do not demand machine precision.
ROTATION_ATOL = 1e-5
_MATRIX_SHAPES = {
    **{f"P{i}": (3, 4) for i in range(4)},
    "R0_rect": (3, 3),
    "Tr_velo_to_cam": (3, 4),
    "Tr_imu_to_velo": (3, 4),
}


def _array(value, shape: tuple[int, ...], name: str) -> np.ndarray:
    try:
        if np.iscomplexobj(value):
            raise ValueError(f"{name} must be real-valued")
        array = np.array(value, dtype=np.float64, copy=True)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{name} must contain finite numeric values") from exc
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must have shape {shape} and finite numeric values")
    # A bytes-backed view cannot have its writable flag re-enabled by a caller.
    return np.frombuffer(array.tobytes(), dtype=np.float64).reshape(shape)


def _rotation(value, name: str) -> np.ndarray:
    rotation = _array(value, (3, 3), name)
    if not np.allclose(
        rotation.T @ rotation, np.eye(3), atol=ROTATION_ATOL, rtol=0
    ) or not np.isclose(np.linalg.det(rotation), 1.0, atol=ROTATION_ATOL, rtol=0):
        raise ValueError(f"{name} must be a proper orthonormal rotation (determinant +1)")
    return rotation


def _frame(name: str) -> str:
    if name == KITTI_CAMERA_FRAME:
        return CAMERA_RECT_FRAME
    if name not in (VELODYNE_FRAME, CAMERA_RECT_FRAME):
        raise ValueError(f"Unsupported coordinate frame {name!r}")
    return name


@dataclass(frozen=True, eq=False)
class KittiCalibration:
    """Immutable object-detection calibration matrices in their original shapes.

    R0_rect and Tr_velo_to_cam are required. Projection and IMU matrices are
    optional but validated when supplied. Additional numeric fields are retained
    as finite one-dimensional arrays. Projection matrices never define a rigid
    transform. All computations use float64.
    """

    matrices: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        missing = {"R0_rect", "Tr_velo_to_cam"} - self.matrices.keys()
        if missing:
            raise ValueError(f"Missing required KITTI matrices: {', '.join(sorted(missing))}")
        matrices = {}
        for name, values in self.matrices.items():
            shape = _MATRIX_SHAPES.get(name, (np.asarray(values).size,))
            matrix = _array(values, shape, name)
            if name == "R0_rect":
                _rotation(matrix, name)
            elif name in ("Tr_velo_to_cam", "Tr_imu_to_velo"):
                _rotation(matrix[:, :3], name)
            matrices[name] = matrix
        object.__setattr__(self, "matrices", MappingProxyType(matrices))

    @property
    def velo_to_rect(self) -> np.ndarray:
        """Homogeneous transform: R0_rect_h @ Tr_velo_to_cam_h."""
        rectification = np.eye(4)
        rectification[:3, :3] = self.matrices["R0_rect"]
        extrinsic = np.eye(4)
        extrinsic[:3, :] = self.matrices["Tr_velo_to_cam"]
        return _array(rectification @ extrinsic, (4, 4), "velo_to_rect")

    @property
    def rect_to_velo(self) -> np.ndarray:
        """Numerical inverse, retaining the precision of rounded input matrices."""
        return _array(np.linalg.inv(self.velo_to_rect), (4, 4), "rect_to_velo")

    def transform(self, source_frame: str, target_frame: str) -> np.ndarray:
        source, target = _frame(source_frame), _frame(target_frame)
        if source == target:
            return _array(np.eye(4), (4, 4), "identity")
        return self.velo_to_rect if source == VELODYNE_FRAME else self.rect_to_velo


def load_kitti_calibration(path: str | Path) -> KittiCalibration:
    """Read colon-separated KITTI object-detection fields, rejecting duplicates."""
    path = Path(path)
    matrices = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                if line.count(":") != 1:
                    raise ValueError("Expected one calibration field separator ':'")
                name, text = line.split(":")
                name = name.strip()
                if not name or not name.replace("_", "").isalnum():
                    raise ValueError("Malformed calibration field name")
                if name in matrices:
                    raise ValueError(f"Duplicate calibration field {name}")
                tokens = text.split()
                if not tokens:
                    raise ValueError(f"Empty calibration field {name}")
                values = np.array([float(token) for token in tokens], dtype=np.float64)
                shape = _MATRIX_SHAPES.get(name, (len(values),))
                if values.size != np.prod(shape):
                    raise ValueError(f"{name} requires shape {shape}, got {values.size} values")
                matrix = _array(values.reshape(shape), shape, name)
                if name == "R0_rect":
                    _rotation(matrix, name)
                elif name in ("Tr_velo_to_cam", "Tr_imu_to_velo"):
                    _rotation(matrix[:, :3], name)
                matrices[name] = matrix
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    try:
        return KittiCalibration(matrices)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc


def transform_points(
    points: np.ndarray,
    calibration: KittiCalibration,
    *,
    source_frame: str,
    target_frame: str,
) -> np.ndarray:
    """Transform (3,) or (N, 3) XYZ arrays, returning float64 without mutation."""
    try:
        if np.iscomplexobj(points):
            raise ValueError("points must be real-valued")
        points = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("points must contain finite real numbers") from exc
    if points.shape != (3,) and (points.ndim != 2 or points.shape[1] != 3):
        raise ValueError("points must have shape (3,) or (N, 3)")
    if not np.isfinite(points).all():
        raise ValueError("points must be finite")
    matrix = calibration.transform(source_frame, target_frame)
    # Equivalent to homogeneous [x, y, z, 1] multiplication without an N×4 buffer.
    with np.errstate(over="ignore", invalid="ignore"):
        result = points @ matrix[:3, :3].T + matrix[:3, 3]
    if not np.isfinite(result).all():
        raise ValueError("transformed points exceed finite float64 range")
    return result


@dataclass(frozen=True, eq=False)
class OrientedBox3D:
    """Geometric-center box with full 3D orientation, retaining source metadata.

    rotation columns are length, width, height unit directions in coordinate_frame.
    source remains the original GroundTruth3D; annotation, when present, retains
    all original KITTI metadata. Neither source object is modified.
    """

    center_xyz: np.ndarray
    size_lwh: np.ndarray
    rotation: np.ndarray
    coordinate_frame: str
    source: GroundTruth3D
    annotation: KittiAnnotation | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source, GroundTruth3D):
            raise TypeError("source must be the original GroundTruth3D")
        if self.annotation is not None and self.annotation.ground_truth != self.source:
            raise ValueError("annotation must describe the source ground-truth box")
        object.__setattr__(self, "center_xyz", _array(self.center_xyz, (3,), "center_xyz"))
        size = _array(self.size_lwh, (3,), "size_lwh")
        if (size <= 0).any():
            raise ValueError("size_lwh dimensions must be positive")
        object.__setattr__(self, "size_lwh", size)
        object.__setattr__(self, "rotation", _rotation(self.rotation, "box rotation"))
        object.__setattr__(self, "coordinate_frame", _frame(self.coordinate_frame))

    @property
    def label(self) -> str:
        return self.source.label

    @property
    def corners(self) -> np.ndarray:
        """Eight corners: lower-height face 0..3, matching upper face 4..7."""
        signs = np.array(
            [
                [-1, -1, -1],
                [1, -1, -1],
                [1, 1, -1],
                [-1, 1, -1],
                [-1, -1, 1],
                [1, -1, 1],
                [1, 1, 1],
                [-1, 1, 1],
            ]
        )
        return (signs * (self.size_lwh / 2)) @ self.rotation.T + self.center_xyz


def _ground_truth_geometry(box: GroundTruth3D) -> OrientedBox3D:
    frame = _frame(box.coordinate_frame)
    angle = box.rotation_rad
    c, s = np.cos(angle), np.sin(angle)
    center = np.array(box.position_xyz, dtype=np.float64)
    if frame == CAMERA_RECT_FRAME:
        if box.rotation_axis != "y":
            raise ValueError("KITTI camera ground truth must rotate about Y")
        rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        # Local L/W/H axes are camera +X/+Z/-Y before rotation; det(basis)=+1.
        rotation = rotation @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        if box.position_reference == "bottom_center":
            center[1] -= box.size_lwh[2] / 2
    else:
        if box.rotation_axis != "z":
            raise ValueError("KITTI Velodyne ground truth must rotate about Z")
        rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        if box.position_reference == "bottom_center":
            center[2] += box.size_lwh[2] / 2
    return OrientedBox3D(center, box.size_lwh, rotation, frame, box)


def transform_box(
    box: GroundTruth3D | OrientedBox3D,
    calibration: KittiCalibration,
    *,
    target_frame: str,
) -> OrientedBox3D:
    """Transform the center AND full orientation, never compressing to yaw."""
    geometry = _ground_truth_geometry(box) if isinstance(box, GroundTruth3D) else box
    matrix = calibration.transform(geometry.coordinate_frame, target_frame)
    return replace(
        geometry,
        center_xyz=transform_points(
            geometry.center_xyz,
            calibration,
            source_frame=geometry.coordinate_frame,
            target_frame=target_frame,
        ),
        rotation=matrix[:3, :3] @ geometry.rotation,
        coordinate_frame=target_frame,
    )


def transform_annotation(
    annotation: KittiAnnotation,
    calibration: KittiCalibration,
    *,
    target_frame: str = VELODYNE_FRAME,
) -> OrientedBox3D | None:
    """Convert an annotation, preserving its metadata; DontCare returns None."""
    _frame(target_frame)
    if annotation.is_ignored:
        return None
    geometry = transform_box(annotation.ground_truth, calibration, target_frame=target_frame)
    return replace(geometry, annotation=annotation)
