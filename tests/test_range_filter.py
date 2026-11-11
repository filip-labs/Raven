import numpy as np
import pytest

from raven.core.point_cloud import PointCloud
from raven.perception.preprocessing.range_filter import filter_range


def test_filter_range_keeps_only_region():
    cloud = PointCloud(np.array([[1, 2, 0, .5], [11, 2, 0, .7], [0, -9, 0, .1]]), "a")
    result = filter_range(cloud, (0, 10), (-5, 5), (-1, 1))
    assert result.count == 1
    assert result.points[0, 3] == pytest.approx(.5)


def test_invalid_range():
    cloud = PointCloud(np.empty((0, 4)), "a")
    with pytest.raises(ValueError):
        filter_range(cloud, (1, 1), (-5, 5), (-1, 1))
