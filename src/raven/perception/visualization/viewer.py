"""Interactive Open3D boundary; importing this module does not import Open3D."""

import numpy as np

from raven.core.point_cloud import PointCloud
from raven.perception.visualization.coloring import point_colors


def view_point_cloud(
    cloud: PointCloud, *, color: str = "distance", point_size: float = 2.0
) -> None:
    """Display a Raven cloud and sensor-origin axes until the window is closed.

    Must run on the main thread in a graphical desktop session. Empty clouds
    are rejected before importing Open3D or creating a window. Intensity and
    coordinate metadata remain in the caller's original PointCloud.
    """
    if not np.isfinite(point_size) or not 1.0 <= point_size <= 25.0:
        raise ValueError("point size must be finite and between 1 and 25 pixels")
    colors = point_colors(cloud, color)
    if not cloud.count:
        raise ValueError("Cannot visualize an empty point cloud")
    try:
        import open3d as o3d
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "Open3D is unavailable; install visualization dependencies with "
            'python -m pip install -e ".[visualization]". '
            f"Original error: {exc}"
        ) from exc

    geometry = o3d.geometry.PointCloud()
    # Convert only XYZ at the Open3D boundary; do not copy the entire Raven cloud.
    geometry.points = o3d.utility.Vector3dVector(cloud.points[:, :3])
    geometry.colors = o3d.utility.Vector3dVector(colors)
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])
    visualizer = o3d.visualization.Visualizer()
    try:
        if not visualizer.create_window(
            window_name=f"Raven — {cloud.frame_id} ({cloud.coordinate_frame})",
            width=1280,
            height=800,
        ):
            raise RuntimeError(
                "Open3D could not create a window. Run from a graphical desktop session "
                "with OpenGL support; headless rendering is not supported by this command."
            )
        if not visualizer.add_geometry(geometry) or not visualizer.add_geometry(axes):
            raise RuntimeError("Open3D could not add the point cloud and coordinate frame")
        options = visualizer.get_render_option()
        if options is None:
            raise RuntimeError("Open3D render options are unavailable")
        options.point_size = point_size
        options.background_color = np.array([0.05, 0.05, 0.05])
        visualizer.run()
    finally:
        visualizer.destroy_window()
