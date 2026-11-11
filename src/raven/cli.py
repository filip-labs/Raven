"""Minimal, dependency-light CLI for inspecting a KITTI frame."""
import argparse
import json

from raven.perception.data.kitti_loader import load_kitti_bin


def main() -> None:
    parser = argparse.ArgumentParser(prog="raven", description="Raven Vision CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect", help="Read and summarize a KITTI LiDAR .bin frame")
    inspect.add_argument("path", help="Path to KITTI velodyne/000000.bin")
    args = parser.parse_args()
    if args.command == "inspect":
        cloud = load_kitti_bin(args.path)
        print(json.dumps({
            "frame_id": cloud.frame_id,
            "coordinate_frame": cloud.coordinate_frame,
            "points": cloud.count,
            "min_xyz": cloud.points[:, :3].min(axis=0).tolist() if cloud.count else None,
            "max_xyz": cloud.points[:, :3].max(axis=0).tolist() if cloud.count else None,
        }, indent=2))


if __name__ == "__main__":
    main()
