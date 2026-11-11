import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from raven.core.point_cloud import PointCloud
from raven.perception.data.kitti_loader import load_kitti_bin
from raven.perception.visualization.coloring import COLOR_MODES, point_colors
from raven.perception.visualization.viewer import view_point_cloud


def test_distance_colors_use_euclidean_distance():
    cloud = PointCloud(np.array([[0, 0, 0, 0.1], [3, 4, 0, 0.2], [0, 0, 10, 0.3]]), "a")
    np.testing.assert_allclose(point_colors(cloud), [[0, 1, 0.2], [0.5, 0.5, 0.2], [1, 0, 0.2]])


@pytest.mark.parametrize("mode", COLOR_MODES)
def test_colors_are_bounded_and_preserve_cloud(mode):
    cloud = PointCloud(np.array([[1e38, 1e38, 1e38, -1], [0, 0, 0, 2]]), "a")
    original = cloud.points.copy()
    colors = point_colors(cloud, mode)
    assert colors.shape == (2, 3)
    assert np.isfinite(colors).all()
    assert ((colors >= 0) & (colors <= 1)).all()
    np.testing.assert_array_equal(cloud.points, original)
    assert not cloud.points.flags.writeable


def test_all_origin_distance_colors():
    cloud = PointCloud(np.zeros((2, 4)), "a")
    np.testing.assert_allclose(point_colors(cloud), [[0, 1, 0.2]] * 2)


def test_intensity_is_clipped_grayscale():
    cloud = PointCloud(np.array([[0, 0, 0, n] for n in [-1, 0, 0.5, 1, 2]]), "a")
    np.testing.assert_allclose(
        point_colors(cloud, "intensity"), np.repeat([[0], [0], [0.5], [1], [1]], 3, axis=1)
    )


@pytest.mark.parametrize("value", [0, 0.5, 1])
def test_constant_intensity(value):
    cloud = PointCloud(np.array([[0, 0, 0, value]] * 2), "a")
    np.testing.assert_allclose(point_colors(cloud, "intensity"), np.full((2, 3), value))


def test_uniform_is_neutral_gray():
    cloud = PointCloud(np.array([[1, 2, 3, 0.9]]), "a")
    np.testing.assert_allclose(point_colors(cloud, "uniform"), [[0.7, 0.7, 0.7]])


@pytest.mark.parametrize("mode", COLOR_MODES)
def test_empty_colors_and_viewer_rejection_without_open3d(mode, monkeypatch):
    monkeypatch.setitem(sys.modules, "open3d", None)
    cloud = PointCloud(np.empty((0, 4)), "empty")
    assert point_colors(cloud, mode).shape == (0, 3)
    with pytest.raises(ValueError, match="empty point cloud"):
        view_point_cloud(cloud, color=mode)


def test_invalid_mode():
    with pytest.raises(ValueError, match="Unknown color mode"):
        point_colors(PointCloud(np.empty((0, 4)), "a"), "rainbow")


def test_missing_open3d_is_actionable(monkeypatch):
    monkeypatch.setitem(sys.modules, "open3d", None)
    with pytest.raises(RuntimeError, match=r"pip install.*visualization"):
        view_point_cloud(PointCloud(np.zeros((1, 4)), "a"))


@pytest.mark.parametrize("size", [0, -1, 26, float("nan"), float("inf")])
def test_invalid_point_size_before_open3d(size, monkeypatch):
    monkeypatch.setitem(sys.modules, "open3d", None)
    with pytest.raises(ValueError, match="point size"):
        view_point_cloud(PointCloud(np.zeros((1, 4)), "a"), point_size=size)


@pytest.fixture
def renderer(monkeypatch):
    geometry = SimpleNamespace()
    options = SimpleNamespace()
    visualizer = Mock()
    visualizer.create_window.return_value = True
    visualizer.add_geometry.return_value = True
    visualizer.get_render_option.return_value = options
    axes = object()
    fake = SimpleNamespace(
        geometry=SimpleNamespace(
            PointCloud=lambda: geometry,
            TriangleMesh=SimpleNamespace(create_coordinate_frame=Mock(return_value=axes)),
        ),
        utility=SimpleNamespace(Vector3dVector=lambda values: np.array(values, copy=True)),
        visualization=SimpleNamespace(Visualizer=lambda: visualizer),
    )
    monkeypatch.setitem(sys.modules, "open3d", fake)
    return fake, geometry, axes, options, visualizer


def test_render_boundary_preserves_xyz_intensity_and_metadata(renderer):
    fake, geometry, axes, options, visualizer = renderer
    cloud = PointCloud(np.array([[1, 2, 3, 0.5]]), "000000", "kitti_velodyne")
    view_point_cloud(cloud, color="uniform", point_size=3)
    np.testing.assert_array_equal(geometry.points, cloud.points[:, :3])
    np.testing.assert_allclose(geometry.colors, [[0.7, 0.7, 0.7]])
    assert cloud.points[0, 3] == 0.5
    assert options.point_size == 3
    assert "000000 (kitti_velodyne)" in visualizer.create_window.call_args.kwargs["window_name"]
    fake.geometry.TriangleMesh.create_coordinate_frame.assert_called_once_with(
        size=2.0,
        origin=[0, 0, 0],
    )
    assert visualizer.add_geometry.call_args_list[1].args == (axes,)
    visualizer.run.assert_called_once()
    visualizer.destroy_window.assert_called_once()


@pytest.mark.parametrize("failure", ["window", "geometry", "run"])
def test_renderer_failure_cleans_up(renderer, failure):
    _, _, _, _, visualizer = renderer
    if failure == "window":
        visualizer.create_window.return_value = False
    elif failure == "geometry":
        visualizer.add_geometry.return_value = False
    else:
        visualizer.run.side_effect = RuntimeError("render failed")
    with pytest.raises(RuntimeError):
        view_point_cloud(PointCloud(np.zeros((1, 4)), "a"))
    visualizer.destroy_window.assert_called_once()


def test_real_local_kitti_coloring_smoke():
    path = Path(__file__).resolve().parents[1] / "data/kitti/training/velodyne/000000.bin"
    if not path.is_file():
        pytest.skip("Local KITTI frame is optional; no dataset required in CI")
    cloud = load_kitti_bin(path)
    assert cloud.count > 0
    assert cloud.frame_id == "000000"
    assert cloud.coordinate_frame == "kitti_velodyne"
    for mode in COLOR_MODES:
        colors = point_colors(cloud, mode)
        assert colors.shape == (cloud.count, 3)
        assert np.isfinite(colors).all()
        assert ((colors >= 0) & (colors <= 1)).all()
