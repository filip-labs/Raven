"""Dataset-neutral LiDAR point cloud representation.

Canonical frame: right-handed sensor frame supplied by the source dataset.
The loader records the frame name; it does not silently change coordinate axes.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PointCloud:
    points: np.ndarray
    frame_id: str
    coordinate_frame: str = "lidar"

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 4:
            raise ValueError("points must have shape (N, 4): x, y, z, intensity")
        if not np.isfinite(points).all():
            raise ValueError("point cloud contains NaN or infinity")
        if not self.frame_id or not self.coordinate_frame:
            raise ValueError("frame_id and coordinate_frame must be nonempty")
        points = points.copy()
        points.setflags(write=False)
        object.__setattr__(self, "points", points)

    @property
    def count(self) -> int:
        return int(self.points.shape[0])
