"""Validated boxes and the detection backend contract.

Distances are meters and angles are radians. No coordinate transforms are
performed here. Optional inference backends are implemented in separate modules.
"""

import math
from dataclasses import dataclass
from numbers import Real
from typing import Protocol

from raven.core.point_cloud import PointCloud


def _number(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite real number")
    return number


def _vector(value: tuple[float, ...], name: str) -> tuple[float, float, float]:
    try:
        items = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{name} must contain three finite numbers") from exc
    if len(items) != 3:
        raise ValueError(f"{name} must contain three finite numbers")
    return tuple(_number(item, name) for item in items)


def _text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a nonempty string without surrounding whitespace")


def _validate_box(box, position_name: str, angle_name: str) -> None:
    object.__setattr__(box, position_name, _vector(getattr(box, position_name), position_name))
    size = _vector(box.size_lwh, "size_lwh")
    if any(dimension <= 0 for dimension in size):
        raise ValueError("size_lwh dimensions must be positive meters")
    object.__setattr__(box, "size_lwh", size)
    object.__setattr__(box, angle_name, _number(getattr(box, angle_name), angle_name))
    _text(box.label, "label")
    _text(box.coordinate_frame, "coordinate_frame")
    if box.rotation_axis not in ("x", "y", "z"):
        raise ValueError("rotation_axis must be x, y, or z")


@dataclass(frozen=True)
class Detection3D:
    """Predicted geometric-center box with confidence in [0, 1].

    The original five positional arguments remain valid. The legacy default is
    a LiDAR frame with yaw around +Z; other frames/axes must be supplied by the
    backend. Positive rotation follows the right-hand rule about rotation_axis.
    size_lwh always means length, width, height in the object's local axes.
    """

    center_xyz: tuple[float, float, float]
    size_lwh: tuple[float, float, float]
    yaw_rad: float
    label: str
    score: float
    coordinate_frame: str = "lidar"
    rotation_axis: str = "z"

    def __post_init__(self) -> None:
        _validate_box(self, "center_xyz", "yaw_rad")
        score = _number(self.score, "score")
        if not 0 <= score <= 1:
            raise ValueError("score must be between 0 and 1")
        object.__setattr__(self, "score", score)


@dataclass(frozen=True)
class GroundTruth3D:
    """Dataset ground truth with explicit position and orientation conventions.

    position_xyz refers to position_reference, not necessarily a geometric
    center. size_lwh is length, width, height in meters. rotation_rad follows
    the source frame's right-hand rule about rotation_axis. There is no
    prediction confidence. Dataset-specific metadata belongs to the loader's
    annotation record, which retains this box alongside its original fields.
    """

    position_xyz: tuple[float, float, float]
    size_lwh: tuple[float, float, float]
    rotation_rad: float
    label: str
    coordinate_frame: str
    rotation_axis: str = "z"
    position_reference: str = "geometric_center"

    def __post_init__(self) -> None:
        _validate_box(self, "position_xyz", "rotation_rad")
        if self.position_reference not in ("geometric_center", "bottom_center"):
            raise ValueError("position_reference must be geometric_center or bottom_center")


class ObjectDetector(Protocol):
    """Model-independent, synchronous inference on a validated point cloud.

    Implementations preserve cloud points (including intensity), frame_id and
    coordinate_frame. Results are a list of validated Detection3D values; an
    empty list means no detections. Every box must use cloud.coordinate_frame.
    Backends own any required coordinate/unit/box-convention conversion and
    must report failure rather than silently return an incompatible frame.
    This protocol documents obligations; it performs no runtime validation.
    """

    def detect(self, cloud: PointCloud) -> list[Detection3D]:
        """Return geometric-center predictions without modifying the input."""
        ...
