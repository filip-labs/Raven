"""Typed KITTI evaluation outcomes and minimal exclusive evidence persistence."""

import platform
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np

from raven.evaluation.backend import (
    CLASSES,
    DIFFICULTIES,
    IDENTITY,
    EvaluationBackendUnavailable,
    evaluate_annotations,
    geometry_identity,
)
from raven.evaluation.geometry import ground_truth_annotation, image_metadata, prediction_annotation
from raven.evaluation.inputs import EvaluationBlocked, load_completed_run, prepare_sources
from raven.experiments.artifacts import file_sha256, write_json_exclusive
from raven.experiments.runner import _environment as execution_environment


@dataclass(frozen=True)
class MetricResult:
    metric: str
    class_name: str
    difficulty: str
    iou_threshold: float
    eligible_ground_truth: int
    availability: str
    ap_r40_percent: float | None


@dataclass(frozen=True)
class EvaluationResult:
    status: Literal["success", "invalid", "blocked"]
    experiment_identity: dict
    config_sha256: str | None
    run_manifest_sha256: str | None
    evaluator: dict
    split: dict
    frame_ids: tuple[str, ...]
    purpose: str
    metrics: tuple[MetricResult, ...]
    diagnostics: tuple[dict, ...]
    warnings: tuple[str, ...]
    failures: tuple[str, ...]
    provenance: dict
    evaluation_schema_version: int = 1
    classes: tuple[str, ...] = CLASSES
    difficulties: tuple[str, ...] = DIFFICULTIES

    @property
    def evaluated_frame_count(self) -> int:
        return len(self.frame_ids) if self.status == "success" else 0

    def to_dict(self):
        return {**asdict(self), "evaluated_frame_count": self.evaluated_frame_count}


def evaluate_experiment(
    path: str | Path, *, allow_training_smoke: bool = False, include_bev: bool = True
) -> EvaluationResult:
    """Read-only evaluation. Invalid/blocked inputs never yield numerical metrics."""
    run = None
    purpose = "unavailable"
    warnings = []
    evidence = {}
    identity = {"manifest_path": str(Path(path).resolve())}
    frame_ids = ()
    source_metadata = []
    try:
        if type(allow_training_smoke) is not bool or type(include_bev) is not bool:
            raise ValueError("evaluation options must be explicit booleans")
        run = load_completed_run(path)
        identity.update(
            name=run.config.name,
            experiment_id=run.config.experiment_id,
            artifact_prefix=run.config.output.artifact_prefix,
        )
        frame_ids = run.config.frame_ids
        frames, purpose, evidence, warnings = prepare_sources(
            run, allow_training_smoke=allow_training_smoke
        )
        gt, dt = [], []
        for frame_id, labels, calibration, detections, image_path, files in frames:
            if detections:
                try:
                    shape, image = image_metadata(image_path)
                except FileNotFoundError as exc:
                    raise EvaluationBlocked(
                        f"genuine image dimensions unavailable for {frame_id}: {image_path}"
                    ) from exc
            else:
                shape, image = None, {"availability": "not_required_for_empty_predictions"}
            if "P2" not in calibration.matrices:
                raise EvaluationBlocked(f"genuine P2 projection unavailable for {frame_id}")
            annotation, conversion = prediction_annotation(detections, calibration, shape)
            gt.append(ground_truth_annotation(labels))
            dt.append(annotation)
            source_metadata.append(
                {"frame_id": frame_id, "image": image, "conversion": conversion, "inputs": files}
            )
        metrics, diagnostics = evaluate_annotations(gt, dt, include_bev=include_bev)
        for diagnostic in diagnostics:
            diagnostic["frame_id"] = frame_ids[diagnostic.pop("frame_index")]
        if any(m["eligible_ground_truth"] and m["eligible_ground_truth"] < 41 for m in metrics):
            warnings.append(
                "fewer than 41 eligible objects: reference fixed recall-bin sampling can underfill AP_R40"
            )
        warnings.append(
            "pinned OpenPCDet KITTI upright conversion is used, not a general full-SO(3) box evaluator"
        )
        return EvaluationResult(
            "success",
            identity,
            run.config.config_sha256,
            run.manifest_sha256,
            IDENTITY,
            {
                **evidence,
                "dataset_root": str(run.config.dataset.root),
                "split": run.config.dataset.split,
            },
            frame_ids,
            purpose,
            tuple(MetricResult(**m) for m in metrics),
            tuple(diagnostics),
            tuple(warnings),
            (),
            {
                "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
                "options": {
                    "allow_training_smoke": allow_training_smoke,
                    "include_bev": include_bev,
                },
                "python": platform.python_version(),
                "numpy": np.__version__,
                "execution_environment": execution_environment(),
                "geometry_engine": geometry_identity(),
                "adapter_sha256": {
                    name: file_sha256(Path(__file__).with_name(name))
                    for name in ("backend.py", "geometry.py", "inputs.py", "evaluate.py")
                },
                "artifact_checksums": run.artifact_identities,
                "source_execution": run.provenance,
                "evaluation_inputs": source_metadata,
            },
        )
    except (EvaluationBlocked, EvaluationBackendUnavailable, FileNotFoundError) as exc:
        status = "blocked"
        failure = str(exc)
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        IndexError,
        OverflowError,
    ) as exc:
        status = "invalid"
        failure = f"{type(exc).__name__}: {exc}"
    return EvaluationResult(
        status,
        identity,
        run.config.config_sha256 if run else None,
        run.manifest_sha256 if run else None,
        IDENTITY,
        evidence,
        frame_ids,
        purpose,
        (),
        (),
        tuple(warnings),
        (failure,),
        {"evaluation_inputs_checked": source_metadata},
    )


def save_evaluation_result(path: str | Path, result: EvaluationResult):
    """Minimal JSON evidence only; exclusive publication, no CSV/export framework."""
    write_json_exclusive(Path(path), result.to_dict())
