import numpy as np
import pytest

from raven.core.point_cloud import PointCloud


def test_valid_point_cloud_is_immutable():
    original = np.array([[1, 2, 3, 0.4]], dtype=np.float32)
    cloud = PointCloud(original, frame_id="000001")
    original[0, 0] = 9
    assert cloud.count == 1
    assert cloud.points[0, 0] == 1
    assert not cloud.points.flags.writeable


@pytest.mark.parametrize(
    "points",
    [np.array([1, 2, 3, 4]), np.zeros((3, 3)), np.array([[np.nan, 0, 0, 0]])],
)
def test_invalid_cloud_rejected(points):
    with pytest.raises(ValueError):
        PointCloud(points, "invalid")
