"""Deterministic RGB coloring without a rendering dependency."""

import numpy as np

from raven.core.point_cloud import PointCloud

COLOR_MODES = ("distance", "intensity", "uniform")


def point_colors(cloud: PointCloud, mode: str = "distance") -> np.ndarray:
    """Return (N, 3) RGB values in [0, 1] without modifying the cloud.

    Distance maps zero to green and the frame's maximum Euclidean distance to
    red, with blue fixed at 0.2. An all-origin cloud is green. Intensity uses
    grayscale reflectance clipped to [0, 1], including constant intensities.
    Uniform uses neutral gray (0.7, 0.7, 0.7).
    """
    if mode not in COLOR_MODES:
        raise ValueError(f"Unknown color mode {mode!r}; choose from {', '.join(COLOR_MODES)}")
    if not isinstance(cloud, PointCloud):
        raise TypeError("cloud must be a Raven PointCloud")
    points = cloud.points
    if points.ndim != 2 or points.shape[1] != 4:
        raise ValueError("points must have shape (N, 4): x, y, z, intensity")
    if not np.isfinite(points).all():
        raise ValueError("point cloud contains NaN or infinity")
    colors = np.empty((cloud.count, 3), dtype=np.float64)
    if not cloud.count:
        return colors
    if mode == "uniform":
        colors[:] = 0.7
    elif mode == "intensity":
        colors[:] = np.clip(points[:, 3], 0.0, 1.0)[:, None]
    else:
        # Float64 avoids overflow when squaring large, finite float32 coordinates.
        distances = np.linalg.norm(points[:, :3].astype(np.float64), axis=1)
        maximum = distances.max()
        fraction = distances / maximum if maximum else np.zeros_like(distances)
        colors[:, 0] = fraction
        colors[:, 1] = 1.0 - fraction
        colors[:, 2] = 0.2
    return colors
