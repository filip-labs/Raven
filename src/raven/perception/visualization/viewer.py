"""Interactive Open3D boundary; importing this module does not import Open3D."""

import threading
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from raven.core.point_cloud import PointCloud
from raven.perception.data.kitti_calibration import OrientedBox3D
from raven.perception.detection.base import Detection3D
from raven.perception.visualization.boxes import box_wireframes, prediction_wireframes
from raven.perception.visualization.coloring import point_colors
from raven.perception.visualization.overlays import HELP, OverlayState
from raven.perception.visualization.scenes import SCENE_HELP, bind_scene_controls


def view_point_cloud(
    cloud: PointCloud,
    *,
    color: str = "distance",
    point_size: float = 2.0,
    boxes: Iterable[OrientedBox3D] = (),
    predictions: Iterable[Detection3D] = (),
    sequence=None,
    screenshot_dir=None,
) -> None:
    """Display a cloud, axes, and independently controlled GT/prediction overlays.

    Must run on the main thread in a graphical desktop session. Empty clouds
    are rejected before importing Open3D or creating a window. Intensity and
    coordinate metadata remain in the caller's original PointCloud. Box corners
    must already be in the cloud frame. When boxes are present, point RGB values
    are dimmed to 35% so overlay wires stand out; hiding all boxes restores colors.
    """
    if sequence is not None:
        if cloud is not sequence.active.cloud:
            raise ValueError("Viewer cloud must be the active sequence frame")
        if tuple(boxes) or tuple(predictions):
            raise ValueError("Configure sequence overlays in FrameSequence")
        boxes, predictions = sequence.active.boxes, sequence.active.predictions
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("Open3D viewer must run on the main thread")
    output_dir = Path(screenshot_dir or "outputs/screenshots").expanduser().resolve()
    if not np.isfinite(point_size) or not 1.0 <= point_size <= 25.0:
        raise ValueError("point size must be finite and between 1 and 25 pixels")
    colors = point_colors(cloud, color)
    if not cloud.count:
        raise ValueError("Cannot visualize an empty point cloud")
    state = OverlayState(tuple(boxes), tuple(predictions))
    wire_points, wire_edges, wire_colors = box_wireframes(
        state.ground_truth, cloud.coordinate_frame
    )
    pred_points, pred_edges, pred_colors = prediction_wireframes(
        state.predictions, cloud.coordinate_frame
    )
    if len(wire_points) or len(pred_points):
        colors *= 0.35
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
    wireframe = None
    if len(wire_points):
        wireframe = o3d.geometry.LineSet()
        wireframe.points = o3d.utility.Vector3dVector(wire_points)
        wireframe.lines = o3d.utility.Vector2iVector(wire_edges)
        wireframe.colors = o3d.utility.Vector3dVector(wire_colors)
    prediction_geometry = None
    if len(pred_points):
        prediction_geometry = o3d.geometry.LineSet()
        prediction_geometry.points = o3d.utility.Vector3dVector(pred_points)
        prediction_geometry.lines = o3d.utility.Vector2iVector(pred_edges)
        prediction_geometry.colors = o3d.utility.Vector3dVector(pred_colors)
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])
    interactive = bool(
        state.ground_truth
        or state.predictions
        or sequence is not None
        or screenshot_dir is not None
    )
    visualizer = (
        o3d.visualization.VisualizerWithKeyCallback()
        if interactive
        else o3d.visualization.Visualizer()
    )
    try:

        def title():
            return f"Raven — {sequence.status() if sequence is not None else cloud.frame_id} ({cloud.coordinate_frame})"

        if not visualizer.create_window(
            window_name=title(),
            width=1280,
            height=800,
        ):
            raise RuntimeError(
                "Open3D could not create a window. Run from a graphical desktop session "
                "with OpenGL support; headless rendering is not supported by this command."
            )
        if not visualizer.add_geometry(geometry) or not visualizer.add_geometry(axes):
            raise RuntimeError("Open3D could not add the point cloud and coordinate frame")
        if wireframe is not None and not visualizer.add_geometry(
            wireframe, reset_bounding_box=False
        ):
            raise RuntimeError("Open3D could not add ground-truth box wireframes")
        if prediction_geometry is not None and not visualizer.add_geometry(
            prediction_geometry, reset_bounding_box=False
        ):
            raise RuntimeError("Open3D could not add prediction wireframes")
        if interactive:
            geometries = {"gt": wireframe, "pred": prediction_geometry}

            def refresh(vis):
                for name, arrays in (
                    ("gt", box_wireframes(state.visible_ground_truth, cloud.coordinate_frame)),
                    (
                        "pred",
                        prediction_wireframes(state.visible_predictions, cloud.coordinate_frame),
                    ),
                ):
                    existing = geometries[name]
                    if existing is not None:
                        if not vis.remove_geometry(existing, reset_bounding_box=False):
                            raise RuntimeError("Open3D could not remove overlay geometry")
                        geometries[name] = None
                    points, lines, rgb = arrays
                    if len(points):
                        updated = o3d.geometry.LineSet()
                        updated.points = o3d.utility.Vector3dVector(points)
                        updated.lines = o3d.utility.Vector2iVector(lines)
                        updated.colors = o3d.utility.Vector3dVector(rgb)
                        if not vis.add_geometry(updated, reset_bounding_box=False):
                            raise RuntimeError("Open3D could not refresh overlay geometry")
                        geometries[name] = updated
                visible = bool(state.visible_ground_truth or state.visible_predictions)
                geometry.colors = o3d.utility.Vector3dVector(
                    point_colors(cloud, color) * (0.35 if visible else 1)
                )
                if not vis.update_geometry(geometry):
                    raise RuntimeError("Open3D could not update point colors")
                print(state.status())

            def callback(key):
                def on_key(vis, action, mods):
                    if action != 1 or mods != 0:  # Only a plain press, never repeats/releases.
                        return False
                    if key == 290:
                        print(HELP)
                        if sequence is not None or screenshot_dir is not None:
                            print(SCENE_HELP)
                    elif key == 298:
                        print(state.prediction_details())
                    elif state.apply_key(key):
                        refresh(vis)
                    return False  # Geometry already updated; avoid a global UpdateGeometry call.

                return on_key

            for key in range(290, 299):
                visualizer.register_key_action_callback(key, callback(key))
            print(HELP)
            print(state.status())
        if sequence is not None or screenshot_dir is not None:

            def move(delta):
                nonlocal cloud
                frame = sequence.move(delta)
                if frame is None:
                    return
                camera = visualizer.get_view_status()
                cloud = frame.cloud
                state.ground_truth, state.predictions = frame.boxes, frame.predictions
                geometry.points = o3d.utility.Vector3dVector(cloud.points[:, :3].astype(np.float64))
                refresh(visualizer)
                if not visualizer.create_window(window_name=title(), width=1280, height=800):
                    raise RuntimeError("Open3D could not update the active frame window")
                visualizer.set_view_status(camera)
                print(f"Active frame: {sequence.status()} | points {cloud.count}")

            bind_scene_controls(
                visualizer,
                get_frame_id=lambda: cloud.frame_id,
                output_dir=output_dir,
                move=move if sequence is not None else None,
                reset=lambda: visualizer.reset_view_point(reset_bounding_box=True),
            )
            if sequence is not None:
                print(f"Active frame: {sequence.status()} | points {cloud.count}")
        options = visualizer.get_render_option()
        if options is None:
            raise RuntimeError("Open3D render options are unavailable")
        options.point_size = point_size
        options.background_color = np.array([0.05, 0.05, 0.05])
        visualizer.run()
    finally:
        visualizer.destroy_window()
