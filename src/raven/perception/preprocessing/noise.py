"""Seeded independent Gaussian XYZ perturbations in the caller's spatial units."""

import math
from numbers import Integral, Real

import numpy as np

from raven.core.point_cloud import PointCloud


def _parameters(position_std, seed):
    message = "position_std must be a finite nonnegative real scalar representable in float64"
    if isinstance(position_std, (bool, np.bool_)) or not isinstance(position_std, Real):
        raise TypeError(message)
    if position_std < 0:
        raise ValueError(message)
    try:
        std = float(position_std)
    except (OverflowError, ValueError) as exc:
        raise ValueError(message) from exc
    if not math.isfinite(std) or (std == 0 and position_std != 0):
        raise ValueError(message)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral):
        raise TypeError("seed must be an explicit nonnegative integer")
    if seed < 0:
        raise ValueError("seed must be an explicit nonnegative integer")
    return std, int(seed)


def inject_noise(cloud: PointCloud, position_std: float, seed: int) -> PointCloud:
    """Return independent XYZI rows with Normal(0, position_std**2) XYZ noise.

    Use a fresh local Generator(PCG64(seed)), drawing (N,3) float64 samples in
    row/XYZ order. Intensity, row count/order and frame metadata are unchanged.
    Empty and zero-noise inputs return independent copies after validation.
    Reject nonfinite intermediates and XYZ outside the finite float32 range;
    ordinary float32 rounding/underflow is allowed. Never clip or resample.
    Reproducibility requires the same input, parameters and software environment.
    """
    std, seed = _parameters(position_std, seed)
    if std == 0 or not cloud.count:
        return PointCloud(cloud.points, cloud.frame_id, cloud.coordinate_frame)
    rng = np.random.Generator(np.random.PCG64(seed))
    points = cloud.points.astype(np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        points[:, :3] += rng.normal(loc=0.0, scale=std, size=(cloud.count, 3))
    xyz = points[:, :3]
    if not np.isfinite(xyz).all() or (np.abs(xyz) > np.finfo(np.float32).max).any():
        raise ValueError("Noisy XYZ must be finite and within the float32 storage range")
    return PointCloud(points, cloud.frame_id, cloud.coordinate_frame)
