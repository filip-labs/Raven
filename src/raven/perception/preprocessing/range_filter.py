"""Deterministic inclusive XYZ filtering in the caller's existing coordinate frame."""

import math
from collections.abc import Mapping
from collections.abc import Set as AbstractSet
from numbers import Real

import numpy as np

from raven.core.point_cloud import PointCloud


def _validated_range(bounds, axis: str) -> tuple[float, float]:
    message = f"{axis} range must contain two finite real bounds with lower < upper"
    if isinstance(bounds, (str, bytes, Mapping, AbstractSet)):
        raise TypeError(message)
    try:
        lower, upper = bounds
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    normalized = []
    for value in (lower, upper):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise TypeError(message)
        try:
            number = float(value)
        except (OverflowError, ValueError) as exc:
            raise ValueError(message) from exc
        if not math.isfinite(number):
            raise ValueError(message)
        normalized.append(number)
    lower, upper = normalized
    if lower >= upper:
        raise ValueError(message)
    return lower, upper


def filter_range(
    cloud: PointCloud,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float],
) -> PointCloud:
    """Keep points inside all three inclusive [lower, upper] intervals.

    Bounds must be ordered pairs of finite real numbers with lower < upper;
    booleans, strings and unordered collections are invalid. Validate even for
    empty input. Comparisons promote stored float32 XYZ to float64 so a precise
    bound is not silently rounded to float32. Retained XYZI values, row order and
    frame metadata are unchanged. Return an independent PointCloud, including a
    valid shape-(0, 4) result when no points survive. No coordinate transforms or
    OpenPCDet preprocessing are performed. Physical units follow the input
    (meters for Raven's KITTI data).
    """
    bounds = tuple(
        _validated_range(value, axis)
        for value, axis in ((x_range, "X"), (y_range, "Y"), (z_range, "Z"))
    )
    xyz = cloud.points[:, :3].astype(np.float64)
    keep = np.ones(cloud.count, dtype=bool)
    for axis, (lower, upper) in enumerate(bounds):
        keep &= (xyz[:, axis] >= lower) & (xyz[:, axis] <= upper)
    return PointCloud(cloud.points[keep], cloud.frame_id, cloud.coordinate_frame)
