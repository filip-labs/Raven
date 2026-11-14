"""Exact CPU KD-tree statistical filtering, with bounded neighbor-query buffers."""

import math
from numbers import Integral, Real

import numpy as np

from raven.core.point_cloud import PointCloud

_QUERY_ENTRY_BUDGET = 262144
_MAX_QUERY_ROWS = 4096


def _parameters(nb_neighbors, std_ratio):
    if isinstance(nb_neighbors, (bool, np.bool_)) or not isinstance(nb_neighbors, Integral):
        raise TypeError("nb_neighbors must be a positive integer")
    if nb_neighbors < 1:
        raise ValueError("nb_neighbors must be a positive integer")
    if isinstance(std_ratio, (bool, np.bool_)) or not isinstance(std_ratio, Real):
        raise TypeError("std_ratio must be a finite nonnegative real number")
    try:
        ratio = float(std_ratio)
    except (OverflowError, ValueError) as exc:
        raise ValueError("std_ratio must be a finite nonnegative real number") from exc
    if not math.isfinite(ratio) or ratio < 0:
        raise ValueError("std_ratio must be a finite nonnegative real number")
    return int(nb_neighbors), ratio


def statistical_outlier_removal(
    cloud: PointCloud, nb_neighbors: int, std_ratio: float
) -> PointCloud:
    """Keep rows whose mean K-neighbor distance is <= mean + ratio * population std.

    K excludes the query row by index, retaining distinct duplicate-coordinate
    rows as valid zero-distance neighbors. K is capped at N-1. Empty/singleton
    inputs return independent copies after parameter validation, without SciPy.
    For larger inputs, exact Euclidean KDTree queries run single-threaded in
    bounded chunks. Sorted distances and sorted global means fix reduction order.
    Preserve XYZI row order and frame metadata without transforms/resampling.
    """
    requested, ratio = _parameters(nb_neighbors, std_ratio)
    if cloud.count < 2:
        return PointCloud(cloud.points, cloud.frame_id, cloud.coordinate_frame)
    try:
        from scipy.spatial import KDTree
    except ImportError as exc:
        raise RuntimeError(
            "Statistical outlier removal requires the optional CPU SciPy backend; "
            'install with python -m pip install -e ".[preprocessing]". '
            f"Original error: {exc}"
        ) from exc
    k = min(requested, cloud.count - 1)
    xyz = cloud.points[:, :3].astype(np.float64)
    tree = KDTree(xyz, copy_data=True, balanced_tree=True, compact_nodes=True)
    means = np.empty(cloud.count, dtype=np.float64)
    block_rows = max(1, min(_MAX_QUERY_ROWS, _QUERY_ENTRY_BUDGET // (k + 1)))
    for start in range(0, cloud.count, block_rows):
        stop = min(start + block_rows, cloud.count)
        distances, indices = tree.query(xyz[start:stop], k=k + 1, eps=0, p=2, workers=1)
        distances[indices == np.arange(start, stop)[:, None]] = np.inf
        # Self can be absent among K+1 zero-distance ties. Taking K still gives
        # K distinct other rows in that case; never discard all zero distances.
        distances.sort(axis=1)
        means[start:stop] = distances[:, :k].mean(axis=1, dtype=np.float64)
    if not np.isfinite(means).all():
        raise ValueError("Neighbor search returned nonfinite mean distances")
    ordered = np.sort(means)
    # Shift by the minimum: equivalent population statistics, with an exact
    # constant-distance case and a mean that cannot round below the minimum.
    offsets = ordered - ordered[0]
    mean_offset = offsets.mean(dtype=np.float64)
    global_mean = ordered[0] + mean_offset
    global_std = np.sqrt(np.mean((offsets - mean_offset) ** 2, dtype=np.float64))
    with np.errstate(over="ignore", invalid="ignore"):
        threshold = global_mean + ratio * global_std
    if not np.isfinite(threshold):
        raise ValueError("std_ratio produces a nonfinite statistical threshold")
    return PointCloud(cloud.points[means <= threshold], cloud.frame_id, cloud.coordinate_frame)
