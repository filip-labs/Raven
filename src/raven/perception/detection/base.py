"""Future contract for KITTI and OPV2V detection backends.

PointPillars integration is intentionally NOT included in the initial milestone.
"""
from dataclasses import dataclass
from typing import Protocol

from raven.core.point_cloud import PointCloud


@dataclass(frozen=True)
class Detection3D:
    center_xyz: tuple[float, float, float]
    size_lwh: tuple[float, float, float]
    yaw_rad: float
    label: str
    score: float


class ObjectDetector(Protocol):
    def detect(self, cloud: PointCloud) -> list[Detection3D]: ...
