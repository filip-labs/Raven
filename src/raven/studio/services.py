"""CPU-only Studio adapters over Raven's existing data, pipeline and evidence APIs."""

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from raven.evaluation.inputs import load_completed_run, strict_json
from raven.experiments.artifacts import file_sha256, write_json_exclusive
from raven.experiments.config import DatasetConfig
from raven.experiments.splits import split_membership
from raven.exporting.evidence import Snapshot, load_evidence
from raven.perception.data.kitti_annotations import load_kitti_labels
from raven.perception.data.kitti_calibration import load_kitti_calibration
from raven.perception.data.kitti_loader import load_kitti_bin
from raven.perception.detection.predictions import detections_from_document
from raven.perception.preprocessing.pipeline import validate_preprocessing_stages

STAGES = {
    "Range filter": {
        "name": "range_filter",
        "x_range": [0.0, 70.0],
        "y_range": [-40.0, 40.0],
        "z_range": [-3.0, 1.0],
    },
    "Voxel downsampling": {"name": "voxel_downsample", "voxel_size": 0.2},
    "Statistical outliers": {
        "name": "statistical_outlier_removal",
        "nb_neighbors": 20,
        "std_ratio": 2.0,
    },
    "Gaussian XYZ noise": {"name": "noise", "position_std": 0.01, "seed": 42},
}


@dataclass(frozen=True)
class DatasetIndex:
    config: DatasetConfig
    paths: tuple[Path, ...]
    membership: str


def discover_dataset(root, split):
    config = DatasetConfig(Path(root).expanduser().resolve(), split)
    if not config.scan_directory.is_dir():
        raise ValueError(f"Missing KITTI directory: {config.scan_directory}")
    paths = tuple(
        sorted(
            p
            for p in config.scan_directory.glob("*.bin")
            if re.fullmatch(r"[0-9]{6}", p.stem) and p.is_file()
        )
    )
    if split != "test":
        members, _ = split_membership(split)
        paths = tuple(p for p in paths if p.stem in members)
        membership = "Canonical split membership verified; train and val share training/."
    else:
        membership = "Unlabeled testing/ directory; authoritative test membership unavailable."
    if not paths:
        raise ValueError(f"No available six-digit KITTI scans in split {split!r}.")
    return DatasetIndex(config, paths, membership)


@dataclass(frozen=True)
class FrameInfo:
    path: Path
    cloud: object
    bounds: tuple | None
    ground_truth: dict | None
    calibration_available: bool


def inspect_frame(path):
    path = Path(path).resolve()
    cloud = load_kitti_bin(path)
    base = path.parent.parent
    labels = base / "label_2" / f"{cloud.frame_id}.txt"
    calibration = base / "calib" / f"{cloud.frame_id}.txt"
    counts = (
        dict(Counter(r.object_type for r in load_kitti_labels(labels).records))
        if labels.is_file()
        else None
    )
    if calibration.is_file():
        load_kitti_calibration(calibration)
    bounds = (
        (
            tuple(map(float, cloud.points[:, :3].min(axis=0))),
            tuple(map(float, cloud.points[:, :3].max(axis=0))),
        )
        if cloud.count
        else None
    )
    return FrameInfo(path, cloud, bounds, counts, calibration.is_file())


def pipeline_document(stages):
    """Reuse validation and return an ordinary detached existing stage-array format."""
    normalized = validate_preprocessing_stages(stages)
    return [
        {k: list(v) if isinstance(v, tuple) else v for k, v in row.items()} for row in normalized
    ]


def load_pipeline(path):
    return pipeline_document(strict_json(Path(path).read_bytes()))


@dataclass(frozen=True)
class SavedRun:
    run: object
    evidence_kind: str

    @property
    def frames(self):
        return self.run.results["frames"]

    def frame(self, frame_id):
        return next(row for row in self.frames if row["frame_id"] == frame_id)


def load_saved_run(path):
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        matches = list(path.glob("*.manifest.json"))
        if len(matches) != 1:
            raise ValueError(
                "Choose a run with exactly one *.manifest.json, or select its manifest explicitly."
            )
        path = matches[0]
    run = load_completed_run(path)
    evidence = load_evidence(path, Snapshot())  # Includes exact remote envelope/binding validation.
    return SavedRun(run, evidence.evidence_kind)


def prediction_input(saved, frame_id):
    """Re-read all identities immediately before making a new viewer copy."""
    saved = load_saved_run(saved.run.manifest_path)
    row = saved.frame(frame_id)
    document = row["predictions"]
    detections_from_document(document, expected_frame_id=frame_id)
    scan = Path(row["inputs"]["scan"]["path"]).resolve()
    if scan.stem != frame_id or file_sha256(scan) != row["inputs"]["scan"]["sha256"]:
        raise ValueError("Original scan is missing or no longer matches the saved input identity.")
    # Saved detections belong to the recorded processed cloud, not necessarily raw.
    if row["processed_xyzi_sha256"] != row["inputs"]["scan"]["sha256"]:
        raise ValueError(
            "This MVP overlays raw-baseline runs only; processed-run overlays need an exact replayed cloud."
        )
    return scan, document


def export_prediction_copy(saved, frame_id, output):
    scan, document = prediction_input(saved, frame_id)
    output = Path(output).resolve()
    protected = (saved.run.manifest_path.parent, saved.run.config.dataset.root)
    if any(output.is_relative_to(root) for root in protected):
        raise ValueError("Viewer copies must be outside the dataset and source run.")
    write_json_exclusive(output, document)
    return scan, output


def summarize_result(saved, frame_id):
    row = saved.frame(frame_id)
    model = row["prediction_source"].get("model_provenance", {})
    response = model.get("response", {})
    worker = response.get("worker", {})
    times = response.get("timings", {})
    return {
        "experiment": saved.run.config.experiment_id,
        "status": saved.run.results["summary"]["status"],
        "frame": frame_id,
        "dataset": str(saved.run.config.dataset.root),
        "split": saved.run.config.dataset.split,
        "backend": saved.run.config.model.source,
        "model": saved.run.config.model.name,
        "model_identity": response.get("model", {}),
        "worker_packages": worker.get("packages", {}),
        "evidence": saved.evidence_kind,
        "detections": row["prediction_count"],
        "classes": row["class_counts"],
        "gpu": worker.get("gpu"),
        "initialization": times.get("model_load_seconds"),
        "detection": times.get("detect_seconds"),
        "end_to_end": row.get("client_end_to_end_timing", {}).get("seconds"),
        "preprocessing": [s["name"] for s in row["preprocessing"]["stages"]],
        "config_sha256": saved.run.config.config_sha256,
    }


@dataclass(frozen=True)
class ViewOptions:
    color: str = "distance"
    point_size: float = 2.0
    screenshot_dir: Path | None = None

    def arguments(self):
        if self.color not in ("distance", "intensity", "uniform"):
            raise ValueError("Unsupported point color.")
        import math

        if (
            isinstance(self.point_size, bool)
            or not math.isfinite(self.point_size)
            or not 1 <= self.point_size <= 25
        ):
            raise ValueError("Point size must be finite, in [1, 25].")
        result = ["--color", self.color, "--point-size", str(self.point_size)]
        if self.screenshot_dir is not None:
            result += ["--screenshot-dir", str(self.screenshot_dir)]
        return result


def viewer_arguments(scan, options, *, ground_truth=False, predictions=None):
    scan = Path(scan).resolve()
    if not scan.is_file():
        raise FileNotFoundError(scan)
    arguments = ["-m", "raven.cli", "visualize", str(scan), *options.arguments()]
    if ground_truth:
        arguments += ["--ground-truth"]
    if predictions is not None:
        predictions = Path(predictions).resolve()
        detections_from_document(strict_json(predictions.read_bytes()), expected_frame_id=scan.stem)
        arguments += ["--predictions", str(predictions)]
    return arguments
