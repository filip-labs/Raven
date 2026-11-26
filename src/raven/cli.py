"""Dependency-light CLI for inspecting and optionally visualizing KITTI frames."""

import argparse
import json
import math
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

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


def _config_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate preprocessing configuration key: {key}")
        result[key] = value
    return result


def _config_constant(value):
    raise ValueError(f"Nonfinite JSON constant in preprocessing configuration: {value}")


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
    experiment = sub.add_parser(
        "inspect-experiment", help="Validate experiment JSON and inspect identity/frame selection"
    )
    experiment.add_argument("path", help="Path to a schema-v1 experiment JSON configuration")
    run = sub.add_parser("run-experiment", help="Preflight and execute a schema-v1 experiment")
    run.add_argument("path", help="Path to experiment JSON")
    run.add_argument("--dry-run", action="store_true", help="Preflight only; create no artifacts")
    evaluation = sub.add_parser(
        "evaluate-experiment", help="Verify and evaluate a completed KITTI run"
    )
    evaluation.add_argument("path", help="Path to completed run manifest")
    evaluation.add_argument(
        "--allow-training-smoke",
        action="store_true",
        help="Allow explicitly labeled training-only development metrics",
    )
    evaluation.add_argument("--no-bev", action="store_true", help="Compute 3D AP_R40 only")
    evaluation.add_argument("--output", help="New JSON evidence path; never overwrite")
    benchmark = sub.add_parser(
        "benchmark-experiment", help="Measure bounded local preprocessing or CUDA adapter latency"
    )
    benchmark.add_argument("path", help="Validated schema-v1 experiment JSON")
    benchmark.add_argument(
        "--spec", help="Benchmark specification JSON; defaults to CPU, warmup 5, repetitions 30"
    )
    benchmark.add_argument(
        "--benchmark-id",
        help="New filename-safe output identity; never overwrite an existing benchmark",
    )
    visualize = sub.add_parser("visualize", help="Explore a KITTI LiDAR frame in an Open3D window")
    visualize.add_argument("path", help="Path to a KITTI Velodyne .bin frame")
    visualize.add_argument(
        "--ground-truth", action="store_true", help="Display matching KITTI ground-truth boxes"
    )
    visualize.add_argument(
        "--compare",
        metavar="CONFIG.json",
        help="Compare raw/processed using an ordered pipeline stage JSON array",
    )
    visualize.add_argument(
        "--navigate", action="store_true", help="Navigate a Velodyne directory with F10/F11"
    )
    visualize.add_argument("--start-frame", help="Starting frame ID (requires --navigate)")
    visualize.add_argument(
        "--predictions-dir", help="Frame-matched FRAME.json directory (requires --navigate)"
    )
    visualize.add_argument(
        "--screenshot-dir",
        help="Enable Shift+F12 PNG export to this directory (navigation default: outputs/screenshots)",
    )
    visualize.add_argument("--predictions", help="Saved schema-v1 KITTI prediction JSON")
    visualize.add_argument("--labels", help="Explicit label_2 file (requires --ground-truth)")
    visualize.add_argument("--calib", help="Explicit calibration file (requires --ground-truth)")
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
    if args.command == "visualize" and (args.labels or args.calib) and not args.ground_truth:
        parser.error("--labels and --calib require --ground-truth")
    if (
        args.command == "visualize"
        and args.compare is not None
        and (args.ground_truth or args.predictions is not None or args.predictions_dir is not None)
    ):
        parser.error(
            "--compare cannot be combined with --ground-truth or --predictions; use the overlay viewer separately"
        )
    if args.command == "visualize":
        if not args.navigate and (args.start_frame is not None or args.predictions_dir is not None):
            parser.error("--start-frame and --predictions-dir require --navigate")
        if args.navigate and (
            args.predictions is not None or args.labels is not None or args.calib is not None
        ):
            parser.error(
                "Navigation uses frame-matched sibling labels/calib and --predictions-dir; explicit single-frame files are unsupported"
            )
    try:
        if args.command == "benchmark-experiment":
            from raven.benchmark.experiment import benchmark_experiment
            from raven.benchmark.spec import BenchmarkSpec, load_benchmark_spec

            spec = load_benchmark_spec(args.spec) if args.spec else BenchmarkSpec()
            result = benchmark_experiment(
                args.path, specification=spec, benchmark_id=args.benchmark_id
            )
            print(json.dumps(result.to_dict(), indent=2, allow_nan=False))
            if result.status != "success":
                parser.exit(
                    {"blocked": 3, "invalid": 2, "failed": 1}[result.status],
                    "; ".join(v["message"] for v in result.failures) + "\n",
                )
            return
        if args.command == "evaluate-experiment":
            from raven.evaluation.evaluate import evaluate_experiment, save_evaluation_result

            result = evaluate_experiment(
                args.path,
                allow_training_smoke=args.allow_training_smoke,
                include_bev=not args.no_bev,
            )
            output = None
            if result.status == "success" or args.output is not None:
                output = (
                    Path(args.output)
                    if args.output
                    else Path(args.path)
                    .resolve()
                    .with_name(result.experiment_identity["artifact_prefix"] + ".evaluation.json")
                )
                save_evaluation_result(output, result)
            document = result.to_dict()
            print(
                json.dumps(
                    {
                        k: document[k]
                        for k in (
                            "status",
                            "purpose",
                            "evaluated_frame_count",
                            "frame_ids",
                            "metrics",
                            "warnings",
                            "failures",
                        )
                    }
                    | {"output": str(output) if output else None},
                    indent=2,
                )
            )
            if result.status != "success":
                parser.exit(
                    3 if result.status == "blocked" else 2, "; ".join(result.failures) + "\n"
                )
            return
        if args.command == "run-experiment":
            from raven.experiments.runner import run_experiment

            summary = run_experiment(args.path, dry_run=args.dry_run)
            print(json.dumps(summary.to_dict(), indent=2))
            if summary.status not in ("ready", "success"):
                parser.exit(
                    1,
                    "Experiment "
                    + summary.status
                    + ": "
                    + "; ".join(failure.message for failure in summary.failures)
                    + "\n",
                )
            return
        if args.command == "inspect-experiment":
            from raven.experiments.config import load_experiment_config

            config = load_experiment_config(args.path)
            print(
                json.dumps(
                    {
                        "configuration": config.to_dict(),
                        "config_sha256": config.config_sha256,
                        "selection_provenance": config.selection_provenance,
                        "frame_ids": list(config.frame_ids),
                        "scan_directory": str(config.dataset.scan_directory),
                        "artifacts": {k: str(v) for k, v in config.artifact_paths.items()},
                    },
                    indent=2,
                )
            )
            return
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
        if args.command == "visualize" and args.navigate:
            from raven.perception.visualization.scenes import FrameSequence

            stages = None
            if args.compare is not None:
                with Path(args.compare).open(encoding="utf-8") as stream:
                    stages = json.load(
                        stream, object_pairs_hook=_config_object, parse_constant=_config_constant
                    )
            sequence = FrameSequence(
                args.path,
                start_frame=args.start_frame,
                ground_truth=args.ground_truth,
                predictions_dir=args.predictions_dir,
                stages=stages,
            )
            active = sequence.active
            options = {
                "color": args.color,
                "point_size": args.point_size,
                "sequence": sequence,
                "screenshot_dir": args.screenshot_dir or "outputs/screenshots",
            }
            if args.compare is not None:
                from raven.perception.visualization.comparison import view_cloud_comparison

                view_cloud_comparison(active.cloud, active.processed, **options)
            else:
                from raven.perception.visualization.viewer import view_point_cloud

                view_point_cloud(active.cloud, **options)
            return
        cloud = load_kitti_bin(args.path)
        if args.command == "visualize":
            if args.compare is not None:
                from raven.perception.preprocessing.pipeline import run_preprocessing_pipeline
                from raven.perception.visualization.comparison import view_cloud_comparison

                with Path(args.compare).open(encoding="utf-8") as stream:
                    stages = json.load(
                        stream, object_pairs_hook=_config_object, parse_constant=_config_constant
                    )
                result = run_preprocessing_pipeline(cloud, stages)
                view_cloud_comparison(
                    cloud,
                    result.cloud,
                    color=args.color,
                    point_size=args.point_size,
                    **(
                        {"screenshot_dir": args.screenshot_dir}
                        if args.screenshot_dir is not None
                        else {}
                    ),
                )
                return
            from raven.perception.visualization.viewer import view_point_cloud

            overlays = {}
            if args.ground_truth:
                from raven.perception.visualization.boxes import load_ground_truth_boxes

                overlays["boxes"] = load_ground_truth_boxes(
                    args.path, labels_path=args.labels, calibration_path=args.calib
                )
            if args.predictions:
                from raven.perception.detection.predictions import load_detections

                overlays["predictions"] = load_detections(
                    args.predictions, expected_frame_id=cloud.frame_id
                )
            if args.screenshot_dir is not None:
                overlays["screenshot_dir"] = args.screenshot_dir
            view_point_cloud(cloud, color=args.color, point_size=args.point_size, **overlays)
            return
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
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
