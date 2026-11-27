"""Main-thread Open3D comparison of an already computed cloud; no Qt or inference."""

import argparse
import hashlib
from pathlib import Path

from raven.perception.data.kitti_loader import load_kitti_bin
from raven.perception.visualization.comparison import view_cloud_comparison


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--processed", required=True)
    parser.add_argument("--raw-sha256", required=True)
    parser.add_argument("--color", default="distance")
    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument("--screenshot-dir")
    args = parser.parse_args()
    try:
        if hashlib.sha256(Path(args.raw).read_bytes()).hexdigest() != args.raw_sha256:
            raise ValueError("Source scan changed after processing; apply the pipeline again.")
        raw, processed = load_kitti_bin(args.raw), load_kitti_bin(args.processed)
        if raw.frame_id != processed.frame_id or raw.coordinate_frame != processed.coordinate_frame:
            raise ValueError("Comparison frame mismatch.")
        view_cloud_comparison(
            raw,
            processed,
            color=args.color,
            point_size=args.point_size,
            screenshot_dir=args.screenshot_dir,
        )
    except (OSError, ValueError, RuntimeError, TypeError) as exc:
        parser.exit(1, f"{type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    main()
