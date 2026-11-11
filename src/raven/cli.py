"""Dependency-light CLI for inspecting and optionally visualizing KITTI frames."""

import argparse
import json
import math
from collections.abc import Sequence

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
