"""Matched two-window comparison; no Open3D import until opening the viewer."""

import json
import os
import sys
import threading
from dataclasses import dataclass
from time import sleep

import numpy as np

from raven.core.point_cloud import PointCloud
from raven.perception.visualization.coloring import point_colors


@dataclass(frozen=True)
class ComparisonData:
    raw_count: int
    processed_count: int
    reduction_percent: float
    raw_colors: np.ndarray
    processed_colors: np.ndarray
    bounds_min: np.ndarray
    bounds_max: np.ndarray


def prepare_comparison(raw: PointCloud, processed: PointCloud, color="distance") -> ComparisonData:
    """Validate frames and compute shared colors/bounds without moving either cloud."""
    if not isinstance(raw, PointCloud) or not isinstance(processed, PointCloud):
        raise TypeError("Comparison inputs must be Raven PointCloud objects")
    if raw.frame_id != processed.frame_id:
        raise ValueError("Comparison frame IDs must match")
    if raw.coordinate_frame != processed.coordinate_frame:
        raise ValueError("Comparison coordinate frames must match")
    if not raw.count:
        raise ValueError("Comparison requires a nonempty original point cloud")
    combined = PointCloud(
        np.concatenate((raw.points, processed.points)), raw.frame_id, raw.coordinate_frame
    )
    colors = point_colors(combined, color)  # One distance maximum for both panels.
    bounds_min = combined.points[:, :3].astype(np.float64).min(axis=0)
    bounds_max = combined.points[:, :3].astype(np.float64).max(axis=0)
    if np.array_equal(bounds_min, bounds_max):
        padding = max(0.5, float(np.abs(bounds_min).max()) * 1e-6)
        bounds_min -= padding  # Camera-only padding; no source XYZ change.
        bounds_max += padding
    arrays = [colors[: raw.count].copy(), colors[raw.count :].copy(), bounds_min, bounds_max]
    for array in arrays:
        array.setflags(write=False)
    return ComparisonData(
        raw.count, processed.count, 100 * (1 - processed.count / raw.count), *arrays
    )


HELP = (
    "Comparison: drag either window to orbit; Ctrl-drag to pan; wheel to zoom. "
    "F1: help | F2: match from focused window | F3: restore equal window sizes/view. "
    "Close either window (or Q/Esc) to finish both. Keep equal window sizes for matched projection."
)


class _CameraLink:
    """Synchronize complete legacy ViewTrajectory settings, including common bounds."""

    def __init__(self, visualizers, data):
        self.visualizers = visualizers
        self.bounds = data.bounds_min.tolist(), data.bounds_max.tolist()
        self.previous = None
        self.initial = self._read(visualizers[0])
        frame = self.initial["trajectory"][0]
        frame["boundingbox_min"], frame["boundingbox_max"] = self.bounds
        frame["lookat"] = ((data.bounds_min + data.bounds_max) / 2).tolist()
        self.apply(self.initial, visualizers[0].get_render_option().point_size)

    @staticmethod
    def _read(vis):
        state = json.loads(vis.get_view_status())
        if not isinstance(state, dict) or len(state.get("trajectory", [])) != 1:
            raise RuntimeError("Open3D did not provide a single comparison camera view")
        return state

    def capture(self):
        return [(self._read(vis), vis.get_render_option().point_size) for vis in self.visualizers]

    def apply(self, state, point_size):
        # Detach the payload before adding bounds; never modify the reset snapshot.
        state = json.loads(json.dumps(state))
        frame = state["trajectory"][0]
        frame["boundingbox_min"], frame["boundingbox_max"] = self.bounds
        for vis in self.visualizers:
            vis.set_view_status(json.dumps(state))
            vis.get_render_option().point_size = point_size
            vis.update_renderer()
        self.previous = self.capture()

    def synchronize(self, source):
        self.apply(
            self._read(self.visualizers[source]),
            self.visualizers[source].get_render_option().point_size,
        )

    def tick(self):
        current = self.capture()
        changed = [a != b for a, b in zip(current, self.previous, strict=True)]
        if any(changed):
            self.synchronize(0 if changed[0] else 1)  # Raw wins simultaneous updates.


def view_cloud_comparison(
    raw: PointCloud, processed: PointCloud, *, color="distance", point_size=2.0
) -> None:
    """Show equal-size side-by-side GLFW windows with linked legacy cameras.

    Raw is left; processed is right, including a blank processed scene. Reuse
    the existing Open3D legacy renderer, with one color scale and union bounds.
    Inputs are already processed; callbacks never invoke preprocessing.
    """
    if isinstance(point_size, (bool, np.bool_)) or not np.isscalar(point_size):
        raise ValueError("point size must be finite and between 1 and 25 pixels")
    try:
        size = float(point_size)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("point size must be finite and between 1 and 25 pixels") from exc
    if not np.isfinite(size) or not 1 <= size <= 25:
        raise ValueError("point size must be finite and between 1 and 25 pixels")
    data = prepare_comparison(raw, processed, color)
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("Open3D comparison must run on the main thread in a graphical desktop")
    try:
        import open3d as o3d
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            'Open3D is unavailable; install with python -m pip install -e ".[visualization]". '
            f"Original error: {exc}"
        ) from exc
    if sys.platform.startswith("linux") and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        raise RuntimeError(
            "Open3D comparison requires a graphical desktop; headless rendering is not supported"
        )
    titles = [
        f"RAW {data.raw_count} | {raw.frame_id}",
        f"PROCESSED {data.processed_count} | {-data.reduction_percent:+.2f}%"
        + (" | EMPTY" if not processed.count else ""),
    ]
    visualizers = []
    failure = None
    cleanup_errors = []
    try:
        for index, (cloud, colors) in enumerate(
            zip((raw, processed), (data.raw_colors, data.processed_colors), strict=True)
        ):
            vis = o3d.visualization.VisualizerWithKeyCallback()
            visualizers.append(vis)
            if not vis.create_window(
                window_name=titles[index], width=640, height=720, left=30 + index * 660, top=60
            ):
                raise RuntimeError(
                    "Open3D could not create a comparison window; a graphical desktop is required"
                )
            if cloud.count:
                geometry = o3d.geometry.PointCloud()
                # Native bindings can require writable float64 buffers. Own both.
                geometry.points = o3d.utility.Vector3dVector(cloud.points[:, :3].astype(np.float64))
                geometry.colors = o3d.utility.Vector3dVector(colors.copy())
                if not vis.add_geometry(geometry):
                    raise RuntimeError("Open3D could not add comparison point geometry")
            axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])
            if not vis.add_geometry(axes):
                raise RuntimeError("Open3D could not add comparison axes")
            options = vis.get_render_option()
            if options is None:
                raise RuntimeError("Open3D comparison render options are unavailable")
            options.point_size = size
            options.background_color = np.array([0.05, 0.05, 0.05])
        link = _CameraLink(visualizers, data)

        def callback(source, key):
            def on_key(vis, action, mods):
                if action != 1 or mods != 0:
                    return False
                if key == 290:
                    print(HELP)
                elif key == 291:
                    link.synchronize(source)
                else:
                    for index, window in enumerate(visualizers):
                        if not window.create_window(
                            window_name=titles[index],
                            width=640,
                            height=720,
                            left=30 + index * 660,
                            top=60,
                        ):
                            raise RuntimeError("Open3D could not restore matched window sizes")
                    link.apply(link.initial, size)
                return False

            return on_key

        for index, vis in enumerate(visualizers):
            for key in range(290, 293):
                vis.register_key_action_callback(key, callback(index, key))
        print(
            f"RAW {data.raw_count} | PROCESSED {data.processed_count} | reduction {data.reduction_percent:.2f}%"
        )
        print(HELP)
        while all(vis.poll_events() for vis in visualizers):
            link.tick()
            for vis in visualizers:
                vis.update_renderer()
            sleep(0.01)
    except (RuntimeError, OSError, ValueError, TypeError, AttributeError) as exc:
        failure = exc
    finally:
        for vis in visualizers:
            try:
                vis.destroy_window()
            except (RuntimeError, OSError) as exc:
                cleanup_errors.append(exc)
    if failure is not None:
        detail = f" Cleanup also failed: {cleanup_errors}" if cleanup_errors else ""
        raise RuntimeError(f"Open3D comparison failed: {failure}.{detail}") from failure
    if cleanup_errors:
        raise RuntimeError(
            f"Open3D comparison window cleanup failed: {cleanup_errors}"
        ) from cleanup_errors[0]
