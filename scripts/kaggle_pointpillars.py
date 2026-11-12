"""Headless OpenPCDet runner with optional ObjectDetector adapter validation.

Install Raven's NumPy-only core first. GPU packages belong only in the external
Kaggle environment. --adapter invokes the reusable PointPillarsDetector API.
The original runner was T4-validated; the shared backend/adapter needs a new GPU run.
"""

import argparse
import json
import re
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from raven.perception.data.kitti_loader import load_kitti_bin
from raven.perception.detection._openpcdet import (
    OpenPCDetSession,
    serialize_predictions,
    sha256_file,
)
from raven.perception.detection.pointpillars import PointPillarsDetector


def sha256_argument(value: str) -> str:
    if not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise argparse.ArgumentTypeError("SHA256 must contain exactly 64 hexadecimal characters")
    return value.lower()


def validate_input(path: Path, expected_sha256: str | None = None):
    cloud = load_kitti_bin(path)
    if not cloud.count:
        raise ValueError("Inference requires a nonempty KITTI frame")
    checksum = sha256_file(path)
    if expected_sha256 is not None and checksum != expected_sha256:
        raise ValueError("Input SHA256 mismatch")
    return cloud, {
        "frame_id": cloud.frame_id,
        "points": cloud.count,
        "coordinate_frame": cloud.coordinate_frame,
        "sha256": checksum,
    }


def run_inference(args, cloud):
    """Shared headless backend, optionally exercising the production detector API."""
    options = {
        "openpcdet_root": args.openpcdet_root,
        "config": args.config,
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": args.checkpoint_sha256,
        "device_index": getattr(args, "device_index", 0),
        "trust_checkpoint": args.trust_checkpoint,
    }
    if getattr(args, "adapter", False):
        detector = PointPillarsDetector(**options)
        try:
            predictions = detector.detect(cloud)
            names = detector.provenance["effective_config"]["CLASS_NAMES"]
            arrays = (
                np.array(
                    [(*box.center_xyz, *box.size_lwh, box.yaw_rad) for box in predictions],
                    dtype=float,
                ).reshape(-1, 7),
                np.array([box.score for box in predictions], dtype=float),
                np.array([names.index(box.label) + 1 for box in predictions], dtype=int),
            )
            return (
                arrays,
                names,
                {
                    **detector.provenance,
                    "inference_entrypoint": "PointPillarsDetector.detect",
                    "detections": [asdict(box) for box in predictions],
                },
            )
        finally:
            detector.close()
    session = OpenPCDetSession(**options)
    try:
        arrays, names = session.predict(cloud)
        return (
            arrays,
            names,
            {**session.provenance, "inference_entrypoint": "OpenPCDetSession.predict"},
        )
    finally:
        session.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="KITTI float32 XYZI .bin frame")
    parser.add_argument("--input-sha256", type=sha256_argument)
    parser.add_argument(
        "--validate-input-only",
        action="store_true",
        help="Validate/hash input without GPU imports or writing predictions",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-sha256", type=sha256_argument)
    parser.add_argument(
        "--checkpoint-source", help="Source URL/reference to record; not verified by the script"
    )
    parser.add_argument(
        "--config", type=Path, help="Matching PointPillar YAML, including its base files"
    )
    parser.add_argument("--openpcdet-root", type=Path, help="External OpenPCDet Git checkout")
    parser.add_argument(
        "--output", type=Path, help="New JSON path; default outputs/raven_predictions_FRAME.json"
    )
    parser.add_argument(
        "--trust-checkpoint",
        action="store_true",
        help="Allow pickle loading only for a trusted, checksum-verified checkpoint",
    )
    parser.add_argument(
        "--adapter",
        action="store_true",
        help="Invoke PointPillarsDetector.detect and serialize its validated outputs",
    )
    parser.add_argument("--device-index", type=int, default=0, help="Visible CUDA device index")
    args = parser.parse_args(argv)
    try:
        args.input = args.input.resolve()
        cloud, input_manifest = validate_input(args.input, args.input_sha256)
        if args.validate_input_only:
            print(json.dumps({"input": input_manifest, "gpu_inference_executed": False}, indent=2))
            return 0
        required = (
            "checkpoint",
            "checkpoint_sha256",
            "checkpoint_source",
            "config",
            "openpcdet_root",
        )
        missing = [name.replace("_", "-") for name in required if not getattr(args, name)]
        if missing:
            raise ValueError("Inference requires: " + ", ".join("--" + name for name in missing))
        args.checkpoint, args.config = args.checkpoint.resolve(), args.config.resolve()
        args.openpcdet_root = args.openpcdet_root.resolve()
        if not args.checkpoint.is_file() or not args.checkpoint.stat().st_size:
            raise ValueError("Checkpoint must be an existing nonempty file")
        checksum = sha256_file(args.checkpoint)
        if checksum != args.checkpoint_sha256:
            raise ValueError("Checkpoint SHA256 mismatch; inference was not started")
        if not args.config.is_file() or not (args.openpcdet_root / "tools").is_dir():
            raise ValueError("Provide an existing configuration and OpenPCDet tools/ directory")
        output = (
            args.output or Path("outputs") / f"raven_predictions_{cloud.frame_id}.json"
        ).resolve()
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite {output}")
        arrays, names, runtime = run_inference(args, cloud)
        document = {
            "schema_version": 1,
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "input": input_manifest,
            "checkpoint": {
                "sha256": checksum,
                "source": args.checkpoint_source,
                "source_status": "user_supplied",
                "pickle_loading_allowed": args.trust_checkpoint,
            },
            "config": {"path": str(args.config), "sha256": sha256_file(args.config)},
            **runtime,
            **serialize_predictions(*arrays, names),
        }
        text = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            stream.write(text)
        print(
            json.dumps(
                {
                    "output": str(output),
                    "total_predictions": document["total_predictions"],
                    "class_counts": document["class_counts"],
                }
            )
        )
        return 0
    except (
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        subprocess.CalledProcessError,
    ) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
