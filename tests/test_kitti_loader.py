import numpy as np
import pytest

from raven.perception.data.kitti_loader import load_kitti_bin


def test_load_kitti_frame(tmp_path):
    data = np.array([[1, 2, 3, 0.5], [4, 5, 6, 0.7]], dtype="<f4")
    path = tmp_path / "000128.bin"
    data.tofile(path)
    cloud = load_kitti_bin(path)
    assert cloud.frame_id == "000128"
    assert cloud.coordinate_frame == "kitti_velodyne"
    np.testing.assert_allclose(cloud.points, data)


def test_malformed_file(tmp_path):
    path = tmp_path / "bad.bin"
    path.write_bytes(b"123")
    with pytest.raises(ValueError, match="file size"):
        load_kitti_bin(path)
