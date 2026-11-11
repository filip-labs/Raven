"""Dependency-light CLI for inspecting and optionally visualizing KITTI frames."""

import argparse
import json
import math
from collections import Counter
from collections.abc import Sequence

from raven.perception.data.kitti_annotations import load_kitti_labels
from raven.perception.data.kitti_calibration import (
    CAMERA_RECT_FRAME,
    VELODYNE_FRAME,
    load_kitti_calibration,
)
from raven.perception.data.kitti_loader import load_kitti_bin


def _point_size(value: str) -> float:
    try:
        size = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("point size must be a number") from exc
    if not math.isfinite(size) or not 1.0 <= size <= 25.0:
        raise argparse.ArgumentTypeError("point size must be finite and between 1 and 25 pixels")
    return size


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="raven", description="Raven Vision CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect", help="Read and summarize a KITTI LiDAR .bin frame")
    inspect.add_argument("path", help="Path to KITTI velodyne/000000.bin")
    labels = sub.add_parser("inspect-labels", help="Summarize KITTI label_2 annotations as JSON")
    labels.add_argument("path", help="Path to a KITTI label_2/000000.txt file")
    calib = sub.add_parser(
        "inspect-calib", help="Inspect KITTI calibration and its rigid transform"
    )
    calib.add_argument("path", help="Path to a KITTI calib/000000.txt file")
    visualize = sub.add_parser("visualize", help="Explore a KITTI LiDAR frame in an Open3D window")
    visualize.add_argument("path", help="Path to a KITTI Velodyne .bin frame")
    visualize.add_argument(
        "--color",
        choices=("distance", "intensity", "uniform"),
        default="distance",
        help="Point coloring (default: distance)",
    )
    visualize.add_argument(
        "--point-size",
        type=_point_size,
        default=2.0,
        help="Point size in pixels, from 1 to 25 (default: 2)",
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect-calib":
            import numpy as np

            calibration = load_kitti_calibration(args.path)
            matrix = calibration.velo_to_rect
            print(
                json.dumps(
                    {
                        "matrices": {
                            name: list(value.shape) for name, value in calibration.matrices.items()
                        },
                        "source_frame": VELODYNE_FRAME,
                        "target_frame": CAMERA_RECT_FRAME,
                        "invertible": bool(
                            np.allclose(calibration.rect_to_velo @ matrix, np.eye(4))
                        ),
                        "velo_to_rect": matrix.tolist(),
                    },
                    indent=2,
                )
            )
            return
        if args.command == "inspect-labels":
            annotations = load_kitti_labels(args.path)
            print(
                json.dumps(
                    {
                        "frame_id": annotations.frame_id,
                        "coordinate_frame": annotations.coordinate_frame,
                        "objects": len(annotations.objects),
                        "class_counts": dict(
                            sorted(Counter(obj.label for obj in annotations.objects).items())
                        ),
                        "ignored_regions": len(annotations.ignored_regions),
                        "unknown_classes": sorted(
                            {
                                record.object_type
                                for record in annotations.records
                                if not record.is_known_class
                            }
                        ),
                    },
                    indent=2,
                )
            )
            return
        cloud = load_kitti_bin(args.path)
        if args.command == "visualize":
            from raven.perception.visualization.viewer import view_point_cloud

            view_point_cloud(cloud, color=args.color, point_size=args.point_size)
            return
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    if args.command == "inspect":
        print(
            json.dumps(
                {
                    "frame_id": cloud.frame_id,
                    "coordinate_frame": cloud.coordinate_frame,
                    "points": cloud.count,
                    "min_xyz": cloud.points[:, :3].min(axis=0).tolist() if cloud.count else None,
                    "max_xyz": cloud.points[:, :3].max(axis=0).tolist() if cloud.count else None,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
