"""Lazy frame selection and framebuffer export, independent of Open3D imports."""

import re
import struct
import zlib
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

from raven.core.point_cloud import PointCloud
from raven.perception.data.kitti_loader import load_kitti_bin
from raven.perception.detection.predictions import load_detections
from raven.perception.preprocessing.pipeline import _validate, run_preprocessing_pipeline
from raven.perception.visualization.boxes import load_ground_truth_boxes

SCENE_HELP = (
    "F10 previous | F11 next (no wrap) | F12 reset current view | "
    "Shift+F12 export rendered PNG(s). Focus window; macOS may require Fn."
)


class SceneLoadError(ValueError):
    """A candidate failed validation; the active scene has not changed."""


@dataclass(frozen=True)
class SceneFrame:
    cloud: PointCloud
    position: int
    boxes: tuple = ()
    predictions: tuple = ()
    processed: PointCloud | None = None


class FrameSequence:
    """Snapshot ordered filenames; retain only the active scan and optional processed scan."""

    def __init__(
        self, directory, *, start_frame=None, ground_truth=False, predictions_dir=None, stages=None
    ):
        if stages is not None and (ground_truth or predictions_dir is not None):
            raise ValueError("Comparison navigation cannot be combined with overlays")
        root = Path(directory).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"Navigation requires a Velodyne directory: {root}")
        self.paths = tuple(sorted(root.glob("*.bin"), key=lambda p: p.name))
        if not self.paths:
            raise ValueError(f"No .bin frames in {root}")
        self.ground_truth = ground_truth
        self.predictions_dir = (
            Path(predictions_dir).expanduser().resolve() if predictions_dir is not None else None
        )
        if self.predictions_dir is not None and not self.predictions_dir.is_dir():
            raise ValueError(f"Prediction directory does not exist: {self.predictions_dir}")
        self.stages = (
            None
            if stages is None
            else tuple({"name": name, **params} for name, params in _validate(stages))
        )
        index = 0
        if start_frame is not None:
            matches = [i for i, path in enumerate(self.paths) if path.stem == start_frame]
            if not matches:
                raise ValueError(f"Start frame {start_frame!r} is not in {root}")
            index = matches[0]
        self.active = self._load(index)

    def _load(self, index):
        path = self.paths[index]
        try:
            cloud = load_kitti_bin(path)
            if not cloud.count:
                raise ValueError("empty point cloud")
            boxes = tuple(load_ground_truth_boxes(path)) if self.ground_truth else ()
            predictions = (
                tuple(
                    load_detections(
                        self.predictions_dir / f"{cloud.frame_id}.json",
                        expected_frame_id=cloud.frame_id,
                    )
                )
                if self.predictions_dir is not None
                else ()
            )
            processed = (
                run_preprocessing_pipeline(cloud, self.stages).cloud
                if self.stages is not None
                else None
            )
        except (OSError, ValueError, TypeError, RuntimeError) as exc:
            raise SceneLoadError(f"Cannot load frame {path.name}: {exc}") from exc
        return SceneFrame(cloud, index, boxes, predictions, processed)

    def move(self, delta):
        if isinstance(delta, bool) or not isinstance(delta, Integral) or delta not in (-1, 1):
            raise ValueError("Navigation step must be -1 or +1")
        index = self.active.position + int(delta)
        if not 0 <= index < len(self.paths):
            print(f"Frame boundary: {self.status()} (no wrapping)")
            return None
        candidate = self._load(index)
        self.active = candidate
        return candidate

    def status(self):
        return f"{self.active.cloud.frame_id} [{self.active.position + 1}/{len(self.paths)}]"


def _png_valid(path):
    content = path.read_bytes()
    if content[:8] != b"\x89PNG\r\n\x1a\n":
        raise RuntimeError(f"Renderer did not produce a PNG: {path}")
    offset, kinds = 8, []
    while offset + 12 <= len(content):
        size = struct.unpack(">I", content[offset : offset + 4])[0]
        kind = content[offset + 4 : offset + 8]
        end = offset + 8 + size
        if (
            end + 4 > len(content)
            or zlib.crc32(content[offset + 4 : end])
            != struct.unpack(">I", content[end : end + 4])[0]
        ):
            raise RuntimeError(f"Incomplete/corrupt screenshot PNG: {path}")
        if not kinds and (
            kind != b"IHDR"
            or size != 13
            or not all(struct.unpack(">II", content[offset + 8 : offset + 16]))
        ):
            raise RuntimeError(f"Invalid screenshot PNG dimensions: {path}")
        kinds.append(kind)
        offset = end + 4
        if kind == b"IEND":
            if size != 0 or offset != len(content) or b"IDAT" not in kinds:
                break
            return
    raise RuntimeError(f"Incomplete screenshot PNG: {path}")


def export_screenshots(visualizers, frame_id, output_dir, *, roles=("view",)):
    """Reserve a matching numbered filename set exclusively, then capture real framebuffers.

    Never overwrite preexisting files. Remove this attempt's files if any capture
    fails. Requires a working graphical renderer; PNG shape/signature are checked.
    """
    if not isinstance(frame_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", frame_id):
        raise ValueError(
            "Screenshot frame ID must contain only letters, digits, dot, dash or underscore"
        )
    visualizers, roles = tuple(visualizers), tuple(roles)
    if (
        len(visualizers) != len(roles)
        or not roles
        or len(set(roles)) != len(roles)
        or any(r not in ("view", "raw", "processed") for r in roles)
    ):
        raise ValueError("Screenshot roles must match unique view/raw/processed renderers")
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    number = 1
    while True:
        paths = tuple(directory / f"raven_{frame_id}_{role}_{number:04d}.png" for role in roles)
        owned = []
        try:
            for path in paths:
                with path.open("xb"):
                    owned.append(path)
        except FileExistsError:
            for path in owned:
                path.unlink()
            number += 1
            continue
        except OSError:
            for path in owned:
                path.unlink()
            raise
        break
    try:
        for vis, path in zip(visualizers, paths, strict=True):
            vis.capture_screen_image(str(path), do_render=True)
            _png_valid(path)
    except (OSError, RuntimeError, ValueError) as exc:
        for path in owned:
            path.unlink(missing_ok=True)
        raise RuntimeError(f"Screenshot export failed: {exc}") from exc
    for path in paths:
        print(f"Screenshot saved: {path}")
    return paths


def bind_scene_controls(
    vis,
    *,
    get_frame_id,
    output_dir,
    move=None,
    reset=None,
    renderers=None,
    roles=("view",),
    before_export=None,
):
    """Register nonconflicting keys; load/export errors keep the current scene visible."""

    def callback(window, action, mods, key):
        if action != 1:
            return False
        if key == 301 and mods == 1:  # GLFW Shift+F12
            try:
                if before_export is not None:
                    before_export()
                export_screenshots(renderers or (window,), get_frame_id(), output_dir, roles=roles)
            except (OSError, RuntimeError, ValueError) as exc:
                print(f"Screenshot error: {exc}")
        elif mods == 0:
            if key in (299, 300):
                if move is None:
                    print("Navigation disabled; open a directory with --navigate")
                else:
                    try:
                        move(-1 if key == 299 else 1)
                    except SceneLoadError as exc:
                        print(f"{exc}; keeping current frame")
            elif key == 301 and reset is not None:
                reset()
        return False

    for key in (299, 300, 301):
        vis.register_key_action_callback(
            key, lambda window, action, mods, key=key: callback(window, action, mods, key)
        )
    print(SCENE_HELP)
