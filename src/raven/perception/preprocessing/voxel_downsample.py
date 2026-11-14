"""Deterministic origin-anchored voxel means in the caller's coordinate frame."""

import math
from collections.abc import Mapping
from collections.abc import Set as AbstractSet
from numbers import Real

import numpy as np

from raven.core.point_cloud import PointCloud


def _voxel_sizes(value) -> np.ndarray:
    message = "voxel_size must be a positive finite scalar or three positive finite real sizes"
    if isinstance(value, (bool, np.bool_, str, bytes, Mapping, AbstractSet)):
        raise TypeError(message)
    if isinstance(value, Real):
        values = (value,) * 3
    else:
        try:
            values = tuple(value)
        except TypeError as exc:
            raise TypeError(message) from exc
        if len(values) != 3:
            raise ValueError(message)
    sizes = []
    for item in values:
        if isinstance(item, (bool, np.bool_)) or not isinstance(item, Real):
            raise TypeError(message)
        try:
            size = float(item)
        except (OverflowError, ValueError) as exc:
            raise ValueError(message) from exc
        if not math.isfinite(size) or size <= 0:
            raise ValueError(message)
        sizes.append(size)
    return np.array(sizes, dtype=np.float64)


def voxel_downsample(
    cloud: PointCloud, voxel_size: float | tuple[float, float, float]
) -> PointCloud:
    """Return one mean XYZI row per occupied, zero-origin, half-open voxel.

    Index each stored coordinate with floor(float64(XYZ) / size), without snapping
    or coordinate transforms. Cell k spans [k*size, (k+1)*size) on each axis.
    Output cells sort lexicographically by (ix, iy, iz); within each cell sort
    XYZI before float64 summation so input permutations do not change aggregation
    order. Cast means through PointCloud's float32 storage, preserving metadata.
    Validate sizes even for empty input; return independent read-only storage.

    Reject scaled-coordinate underflow/nonfinite values or |XYZ/size| >= 2**53: float64 cannot
    resolve consecutive integer voxel indices beyond this conservative range.
    No OpenPCDet voxelization, intensity normalization or hash iteration is used.
    """
    sizes = _voxel_sizes(voxel_size)
    if not cloud.count:
        return PointCloud(
            np.empty((0, 4), dtype=np.float32), cloud.frame_id, cloud.coordinate_frame
        )
    points = cloud.points.astype(np.float64)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        scaled = points[:, :3] / sizes
    underflow = (points[:, :3] != 0) & (scaled == 0)
    if underflow.any() or not np.isfinite(scaled).all() or (np.abs(scaled) >= 2**53).any():
        raise ValueError("voxel_size/coordinates exceed the supported float64 voxel-index range")
    indices = np.floor(scaled).astype(np.int64)
    # np.lexsort's last key is primary: cells XYZ, then individual XYZI values.
    order = np.lexsort(
        (
            points[:, 3],
            points[:, 2],
            points[:, 1],
            points[:, 0],
            indices[:, 2],
            indices[:, 1],
            indices[:, 0],
        )
    )
    ordered_indices = indices[order]
    starts = np.r_[
        0, np.flatnonzero(np.any(ordered_indices[1:] != ordered_indices[:-1], axis=1)) + 1
    ]
    counts = np.diff(np.r_[starts, cloud.count])
    means = np.add.reduceat(points[order], starts, axis=0) / counts[:, None]
    return PointCloud(means, cloud.frame_id, cloud.coordinate_frame)
