"""Read KITTI Velodyne .bin frames (float32 x, y, z, reflectance)."""
from pathlib import Path

import numpy as np

from raven.core.point_cloud import PointCloud


def load_kitti_bin(path: str | Path) -> PointCloud:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"KITTI frame not found: {path}")
    if path.suffix.lower() != ".bin":
        raise ValueError("Expected a KITTI .bin file")
    if path.stat().st_size % 16 != 0:
        raise ValueError("Invalid KITTI file size: expected 4 float32 values per point")
    points = np.fromfile(path, dtype="<f4").reshape(-1, 4)
    return PointCloud(points=points, frame_id=path.stem, coordinate_frame="kitti_velodyne")
