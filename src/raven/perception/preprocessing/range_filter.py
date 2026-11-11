"""Spatial region-of-interest filter; coordinates are in meters."""
import numpy as np

from raven.core.point_cloud import PointCloud


def filter_range(
    cloud: PointCloud,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float],
) -> PointCloud:
    bounds = (x_range, y_range, z_range)
    if any(lo >= hi or not np.isfinite([lo, hi]).all() for lo, hi in bounds):
        raise ValueError("Each range must have finite lower < upper bounds")
    xyz = cloud.points[:, :3]
    keep = np.ones(cloud.count, dtype=bool)
    for axis, (lo, hi) in enumerate(bounds):
        keep &= (xyz[:, axis] >= lo) & (xyz[:, axis] <= hi)
    return PointCloud(cloud.points[keep], cloud.frame_id, cloud.coordinate_frame)
