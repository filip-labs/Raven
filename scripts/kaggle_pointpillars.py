"""Headless, single-frame OpenPCDet experiment runner; GPU execution is unverified.

Install Raven's NumPy-only core first. GPU packages belong only in the external
Kaggle environment. This is an experiment script, not a Raven detector adapter.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import pickle
import platform
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from raven.perception.data.kitti_loader import load_kitti_bin


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def serialize_predictions(boxes, scores, labels, class_names) -> dict:
    """Serialize raw OpenPCDet KITTI boxes, without evaluation or axis conversion."""
    names = list(class_names)
    if (
        not names
        or any(not isinstance(name, str) or not name.strip() for name in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("Class names must be unique nonempty strings")
    boxes, scores, labels = np.asarray(boxes), np.asarray(scores), np.asarray(labels)
    if boxes.ndim != 2 or boxes.shape[1] != 7:
        raise ValueError("Prediction boxes must have shape (N, 7)")
    count = len(boxes)
    if scores.shape != (count,) or labels.shape != (count,):
        raise ValueError("Scores and labels must have shape (N,) matching boxes")
    for value in (boxes, scores, labels):
        if value.dtype.kind not in "fiu" or not np.isfinite(value).all():
            raise ValueError("Predictions must contain finite real numeric values")
    if (boxes[:, 3:6] <= 0).any():
        raise ValueError("Predicted box dimensions must be positive")
    if ((scores < 0) | (scores > 1)).any():
        raise ValueError("Expected probability scores in [0, 1]; raw logits are unsupported")
    if (labels != np.floor(labels)).any() or ((labels < 1) | (labels > len(names))).any():
        raise ValueError("Labels must be one-based class indices from the model configuration")
    predictions = [
        {
            "box": box.tolist(),
            "score": float(score),
            "label_id": int(label),
            "class_name": names[int(label) - 1],
        }
        for box, score, label in zip(boxes, scores, labels, strict=True)
    ]
    return {
        "coordinate_frame": "kitti_velodyne",
        "box_format": ["x", "y", "z", "dx", "dy", "dz", "heading"],
        "position_reference": "geometric_center",
        "units": {"position": "m", "heading": "rad"},
        "total_predictions": count,
        "class_counts": dict(sorted(Counter(p["class_name"] for p in predictions).items())),
        "predictions": predictions,
    }


def plain_config(value):
    if isinstance(value, Mapping):
        return {str(key): plain_config(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain_config(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(f"Unsupported configuration value: {type(value).__name__}")


@contextmanager
def working_directory(path: Path):
    previous = Path.cwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(previous)


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def command_output(arguments):
    try:
        result = subprocess.run(arguments, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def run_inference(args, cloud):
    """Upstream prepare/collate/build/forward flow, without importing GUI demos.

    Reconstructed from the pinned upstream interfaces, not recovered from the
    successful notebook. Requires a fresh GPU process for validation.
    """
    for name in ("CUMM_DISABLE_JIT", "SPCONV_DISABLE_JIT"):
        if os.environ.setdefault(name, "1") != "1":
            raise ValueError(f"{name}=1 must be set before launching this process")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.5")
    sys.path.insert(0, str(args.openpcdet_root))
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "An external CUDA GPU is required; CPU/macOS inference is unsupported"
            )
        import pcdet
        from pcdet.config import cfg, cfg_from_yaml_file
        from pcdet.datasets import DatasetTemplate
        from pcdet.models import build_network, load_data_to_gpu
        from pcdet.utils import common_utils
    except ImportError as exc:
        raise RuntimeError(
            "External GPU dependencies are unavailable. See docs/pointpillars-feasibility.md; "
            "for missing av2, review the documented optional Argoverse workaround. "
            f"Original error: {exc}"
        ) from exc
    if Path(pcdet.__file__).resolve().parent != args.openpcdet_root / "pcdet":
        raise ValueError("Imported pcdet does not match --openpcdet-root")
    revision = subprocess.run(
        ["git", "-C", str(args.openpcdet_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(args.openpcdet_root), "diff", "--name-only", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    # Upstream base YAML paths are relative to tools/, not to the top-level YAML.
    with working_directory(args.openpcdet_root / "tools"):
        cfg_from_yaml_file(str(args.config), cfg)
        if cfg.MODEL.NAME != "PointPillar" or cfg.MODEL.POST_PROCESSING.OUTPUT_RAW_SCORE:
            raise ValueError("This runner requires PointPillar with OUTPUT_RAW_SCORE=False")
        logger = common_utils.create_logger()
        dataset = DatasetTemplate(
            dataset_cfg=cfg.DATA_CONFIG,
            class_names=cfg.CLASS_NAMES,
            training=False,
            root_path=args.input.parent,
            logger=logger,
        )
        prepared = dataset.prepare_data({"points": cloud.points.copy(), "frame_id": cloud.frame_id})
        model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
        # Explicitly strict: never claim inference from a silently partial weight load.
        try:
            checkpoint = torch.load(
                args.checkpoint, map_location="cpu", weights_only=not args.trust_checkpoint
            )
        except pickle.UnpicklingError as exc:
            raise RuntimeError(
                "Checkpoint loading failed. Review provenance and the original error; "
                "only a trusted checkpoint may be retried explicitly with --trust-checkpoint. "
                f"Original error: {exc}"
            ) from exc
        if not isinstance(checkpoint, Mapping) or not isinstance(
            checkpoint.get("model_state"), Mapping
        ):
            raise TypeError("Expected an OpenPCDet checkpoint containing a model_state mapping")
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.cuda().eval()
        batch = dataset.collate_batch([prepared])
        load_data_to_gpu(batch)
        with torch.no_grad():
            result, _ = model.forward(batch)
        prediction = result[0]
        arrays = [
            prediction[key].detach().cpu().numpy()
            for key in ("pred_boxes", "pred_scores", "pred_labels")
        ]
        environment = {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "torch_cuda_runtime": torch.version.cuda,
            "cuda_toolkit_nvcc": command_output(["nvcc", "--version"]),
            "gpu_driver": command_output(
                ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]
            ),
            "spconv_cu126": package_version("spconv-cu126"),
            "cumm_cu126": package_version("cumm-cu126"),
            "numpy": np.__version__,
            "openpcdet": pcdet.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "visible_gpu_count": torch.cuda.device_count(),
            "device_index": 0,
            "environment_flags": {
                key: os.environ[key]
                for key in ("CUMM_DISABLE_JIT", "SPCONV_DISABLE_JIT", "TORCH_CUDA_ARCH_LIST")
            },
        }
        return (
            arrays,
            list(cfg.CLASS_NAMES),
            {
                "environment": environment,
                "openpcdet_commit": revision,
                "openpcdet_modified_files": dirty,
                "effective_config": plain_config(cfg),
            },
        )


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
