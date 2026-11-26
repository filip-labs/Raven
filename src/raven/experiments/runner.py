"""Preflight, bounded frame execution and genuine provenance for schema-v1 experiments."""

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import subprocess
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from raven.experiments.artifacts import (
    file_sha256,
    persist_run,
    reserve_run,
    write_json_exclusive,
)
from raven.experiments.config import ExperimentConfig, load_experiment_config
from raven.experiments.splits import split_membership
from raven.perception.data.kitti_annotations import load_kitti_labels
from raven.perception.data.kitti_calibration import load_kitti_calibration
from raven.perception.data.kitti_loader import load_kitti_bin
from raven.perception.detection.predictions import (
    CLASS_NAMES,
    detections_from_document,
    detections_to_document,
    load_detections,
)
from raven.perception.preprocessing.pipeline import run_preprocessing_pipeline


@dataclass(frozen=True)
class RunFailure:
    phase: str
    message: str
    frame_id: str | None = None


@dataclass(frozen=True)
class FileIdentity:
    path: Path
    sha256: str
    bytes: int

    def to_dict(self):
        return {"path": str(self.path), "sha256": self.sha256, "bytes": self.bytes}


@dataclass(frozen=True)
class FrameInputs:
    frame_id: str
    files: tuple[tuple[str, FileIdentity], ...]


@dataclass(frozen=True)
class PreflightReport:
    status: Literal["ready", "failed", "blocked"]
    frames: tuple[FrameInputs, ...]
    failures: tuple[RunFailure, ...]
    split_evidence: dict = field(default_factory=dict)
    model_evidence: dict = field(default_factory=dict)


@dataclass(frozen=True)
class RunSummary:
    status: Literal["ready", "success", "failed", "blocked"]
    config_sha256: str
    selected_frame_ids: tuple[str, ...]
    succeeded: int = 0
    failed: int = 0
    output_directory: Path | None = None
    failures: tuple[RunFailure, ...] = ()

    def to_dict(self) -> dict:
        return {
            "run_schema_version": 1,
            "status": self.status,
            "config_sha256": self.config_sha256,
            "selected_frame_ids": list(self.selected_frame_ids),
            "succeeded": self.succeeded,
            "failed": self.failed,
            "output_directory": str(self.output_directory) if self.output_directory else None,
            "failures": [asdict(v) for v in self.failures],
        }


def _now():
    return datetime.now(timezone.utc).isoformat()


def _observe(path, loader):
    path = Path(path).resolve()
    before = file_sha256(path)
    value = loader(path)
    after = file_sha256(path)
    if before != after:
        raise ValueError(f"input changed during read: {path}")
    return value, FileIdentity(path, after, path.stat().st_size)


def _saved(path, frame_id, scan, scan_identity):
    # Keep the RV-023 strict reader as the schema boundary. Hash checks around
    # both reads ensure the metadata and validated detections identify one file.
    detections = load_detections(path, expected_frame_id=frame_id)
    document = json.loads(path.read_text(encoding="utf-8"))
    json.dumps(document, allow_nan=False)  # Reject nonfinite values even in optional metadata.
    if detections_from_document(document, expected_frame_id=frame_id) != detections:
        raise ValueError("prediction content changed during read")
    effective = document.get("effective_config", {})
    if not isinstance(effective, dict) or not isinstance(effective.get("MODEL", {}), dict):
        raise TypeError("invalid saved effective model configuration")
    names = document.get("class_names", effective.get("CLASS_NAMES"))
    if names != list(CLASS_NAMES):
        raise ValueError("saved PointPillars class_names must be Car, Pedestrian, Cyclist")
    source = document["input"]
    checksum = source.get("sha256")
    if checksum is not None:
        if not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum):
            raise ValueError("invalid prediction input SHA256")
        if checksum != scan_identity.sha256:
            raise ValueError("prediction input SHA256 does not match actual scan")
    if "points" in source and (type(source["points"]) is not int or source["points"] != scan.count):
        raise ValueError("prediction input point count does not match actual scan")
    if document.get("preprocessing"):
        raise ValueError(
            "saved capture declares preprocessing; raw baseline binding is unsupported"
        )
    effective = document.get("effective_config", {})
    if (
        isinstance(effective, dict)
        and effective.get("MODEL", {}).get("NAME", "PointPillar") != "PointPillar"
    ):
        raise ValueError("saved capture is not a PointPillar model")
    declared = document.get("checkpoint_sha256")
    nested = document.get("checkpoint", {})
    nested_hash = nested.get("sha256") if isinstance(nested, dict) else None
    for v in (declared, nested_hash):
        if v is not None and (not isinstance(v, str) or not re.fullmatch(r"[0-9a-f]{64}", v)):
            raise ValueError("invalid declared checkpoint SHA256")
    if declared and nested_hash and declared != nested_hash:
        raise ValueError("capture checkpoint identities disagree")
    return detections, {
        "mode": "loaded_saved_predictions",
        "input_binding": "sha256_verified" if checksum is not None else "unavailable",
        "model_provenance": "declared_not_independently_verified",
        "capture_metadata": {
            k: v for k, v in document.items() if k not in ("predictions", "detections")
        },
    }


def load_experiment_frame(
    config: ExperimentConfig, frame_id: str, expected: FrameInputs | None = None
):
    """Load/hash existing experiment inputs only; never preprocess or invoke a detector."""
    if expected is not None:
        for _, identity in expected.files:
            if file_sha256(identity.path) != identity.sha256:
                raise ValueError(f"input identities changed since preflight: {identity.path}")
    root = config.dataset.scan_directory.parent
    scan_path = config.dataset.scan_directory / f"{frame_id}.bin"
    cloud, scan_id = _observe(scan_path, load_kitti_bin)
    if cloud.frame_id != frame_id or not cloud.count:
        raise ValueError("scan identity mismatch or empty scan")
    files = [("scan", scan_id)]
    labels, labels_id = _observe(root / "label_2" / f"{frame_id}.txt", load_kitti_labels)
    calibration, calibration_id = _observe(
        root / "calib" / f"{frame_id}.txt", load_kitti_calibration
    )
    if labels.frame_id != frame_id:
        raise ValueError("annotation frame identity mismatch")
    files.extend((("labels", labels_id), ("calibration", calibration_id)))
    predictions = source = None
    if config.model.source == "saved":
        path = config.model.predictions_dir / f"{frame_id}.json"
        (predictions, source), identity = _observe(
            path, lambda p: _saved(p, frame_id, cloud, scan_id)
        )
        files.append(("predictions", identity))
    frame = FrameInputs(frame_id, tuple(files))
    if expected is not None and frame != expected:
        raise ValueError("input identities changed since preflight")
    return frame, cloud, labels, calibration, predictions, source


def _inference_preflight(config):
    """No model execution/deserialization. Missing CUDA is explicitly blocked."""
    from raven.perception.detection._openpcdet import PINNED_REVISION, validate_assets

    if config.model.openpcdet_revision != PINNED_REVISION:
        raise ValueError("inference requires the adapter's pinned OpenPCDet revision")
    try:
        import torch
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            f"unsupported inference environment: PyTorch unavailable ({exc})"
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError("unsupported inference environment: CUDA unavailable")
    if any(importlib.util.find_spec(v) is None for v in ("spconv", "cumm")):
        raise RuntimeError("unsupported inference environment: spconv/cumm unavailable")
    candidates = [p for p in config.model.config.parents if (p / "pcdet/__init__.py").is_file()]
    if not candidates:
        raise ValueError(
            "cannot derive OpenPCDet root from model.config; put config in its checkout"
        )
    root = candidates[0]
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != config.model.openpcdet_revision:
        raise ValueError("actual OpenPCDet revision disagrees with experiment configuration")
    validate_assets(
        root, config.model.config, config.model.checkpoint, config.model.checkpoint_sha256
    )
    return {
        "openpcdet_root": str(root),
        "config_sha256": file_sha256(config.model.config),
        "checkpoint_sha256": config.model.checkpoint_sha256,
        "revision": revision,
        "device_index": 0,
        "pickle_loading_allowed": False,
    }


def preflight_experiment(config: ExperimentConfig) -> PreflightReport:
    """Read/validate every selected input before creating any execution output."""
    failures, frames = [], []
    evidence = {}
    try:
        # Revalidate the frozen declaration, manifest checksum and current paths.
        revalidated = ExperimentConfig.from_dict(config.to_dict(), base_dir=Path.cwd())
        if revalidated != config:
            raise ValueError("configuration paths changed since loading")
        if config.run_directory.exists() or config.run_directory.is_symlink():
            raise FileExistsError(f"refusing existing run directory: {config.run_directory}")
        ancestor = config.output.directory
        while not ancestor.exists():
            ancestor = ancestor.parent
        if not os.access(ancestor, os.W_OK | os.X_OK):
            raise PermissionError(f"output ancestor is not writable/searchable: {ancestor}")
        if not config.dataset.root.is_dir():
            raise FileNotFoundError(f"dataset root unavailable: {config.dataset.root}")
    except (OSError, ValueError, TypeError) as exc:
        return PreflightReport("failed", (), (RunFailure("preflight", str(exc)),))
    try:
        members, evidence = split_membership(config.dataset.split)
    except FileNotFoundError as exc:
        return PreflightReport("blocked", (), (RunFailure("split_evidence_unavailable", str(exc)),))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return PreflightReport("failed", (), (RunFailure("split_evidence_invalid", str(exc)),))
    for frame_id in config.frame_ids:
        try:
            if frame_id not in members:
                raise ValueError(
                    f"frame {frame_id} is not in declared {config.dataset.split} split"
                )
            frame, *_ = load_experiment_frame(config, frame_id)
            frames.append(frame)
        except (OSError, ValueError, TypeError) as exc:
            failures.append(RunFailure("preflight", str(exc), frame_id))
    if failures:
        return PreflightReport("failed", tuple(frames), tuple(failures), evidence)
    model = {"source": config.model.source}
    try:
        if (
            any(s["name"] == "statistical_outlier_removal" for s in config.preprocessing)
            and importlib.util.find_spec("scipy") is None
        ):
            raise RuntimeError("statistical preprocessing requires optional SciPy")
        if config.model.source == "inference":
            model.update(_inference_preflight(config))
        elif config.model.source == "modal":
            from raven.inference.modal_transport import modal_preflight

            model.update(modal_preflight())
    except RuntimeError as exc:
        return PreflightReport(
            "blocked",
            tuple(frames),
            (RunFailure("unsupported_environment", str(exc)),),
            evidence,
            model,
        )
    except (OSError, ValueError, TypeError, subprocess.SubprocessError) as exc:
        return PreflightReport(
            "failed", tuple(frames), (RunFailure("model_preflight", str(exc)),), evidence, model
        )
    return PreflightReport("ready", tuple(frames), (), evidence, model)


def _environment():
    packages = {}
    for name in ("raven-vision", "numpy", "scipy", "torch", "spconv-cu126", "pcdet"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    module = Path(__file__).resolve()
    source = module.parents[3]
    try:
        source_hash = file_sha256(module)
    except OSError:
        source_hash = None
    try:
        # An installed wheel inside another checkout must not inherit that
        # unrelated repository's SHA via Git's upward directory search.
        if (
            not (source / ".git").exists()
            or (source / "src/raven/experiments/runner.py").resolve() != module
        ):
            raise FileNotFoundError("Raven source checkout identity unavailable")
        commit = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        modified = subprocess.check_output(
            ["git", "-C", str(source), "status", "--short"], text=True
        ).splitlines()
    except (OSError, subprocess.SubprocessError):
        commit, modified = None, None
    return {
        "python": platform.python_version(),
        "packages": packages,
        "os": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "raven_commit": commit,
        "raven_worktree_status": modified,
        "source_path": str(source) if commit is not None else None,
        "runner_source_sha256": source_hash,
        "runner_module": str(module),
    }


def _build_detector(config, report):
    if config.model.source == "modal":
        from raven.inference.contracts import digest
        from raven.inference.detector import RemoteDetector
        from raven.inference.modal_transport import ModalTransport

        return RemoteDetector(
            ModalTransport(config.model.remote),
            experiment_id=config.experiment_id,
            config_sha256=config.config_sha256,
            preprocessing_sha256=digest(config.to_dict()["preprocessing"]),
        )
    from raven.perception.detection.pointpillars import PointPillarsDetector

    return PointPillarsDetector(
        openpcdet_root=report.model_evidence["openpcdet_root"],
        config=config.model.config,
        checkpoint=config.model.checkpoint,
        checkpoint_sha256=config.model.checkpoint_sha256,
    )


def _execute_frame(config, frame, detector):
    observed, raw, labels, calibration, predictions, source = load_experiment_frame(
        config, frame.frame_id, frame
    )
    preprocessing_started = time.perf_counter()
    processed = run_preprocessing_pipeline(raw, config.preprocessing)
    preprocessing_seconds = time.perf_counter() - preprocessing_started
    processed_hash = hashlib.sha256(processed.cloud.points.astype("<f4").tobytes()).hexdigest()
    if detector is not None:
        predictions = detector.detect(processed.cloud)
        if (
            hashlib.sha256(processed.cloud.points.astype("<f4").tobytes()).hexdigest()
            != processed_hash
        ):
            raise ValueError("detector mutated the recorded input cloud")
        source = {
            "mode": "remote_adapter" if config.model.source == "modal" else "live_adapter",
            "input_binding": "executed_on_recorded_processed_cloud",
            "model_provenance": detector.provenance,
        }
    document = detections_to_document(predictions, frame_id=frame.frame_id)
    return {
        "frame_id": frame.frame_id,
        "status": "success",
        "inputs": {k: v.to_dict() for k, v in observed.files},
        "input_count": raw.count,
        "processed_count": processed.cloud.count,
        "processed_xyzi_sha256": processed_hash,
        "processed_hash_representation": "contiguous-little-endian-float32-XYZI",
        "preprocessing": processed.to_metadata_dict(),
        **(
            {
                "client_preprocessing_timing": {
                    "seconds": preprocessing_seconds,
                    "clock": "time.perf_counter",
                    "scope": "run_preprocessing_pipeline",
                }
            }
            if config.model.source == "modal"
            else {}
        ),
        "predictions": document,
        "prediction_source": source,
        "prediction_count": len(predictions),
        "class_counts": dict(Counter(d.label for d in predictions)),
        "ground_truth": {
            "coordinate_frame": labels.coordinate_frame,
            "objects": [asdict(v) for v in labels.objects],
            "ignored_regions": [asdict(v) for v in labels.ignored_regions],
            "calibration": {k: v.tolist() for k, v in calibration.matrices.items()},
        },
    }


def run_experiment(
    configuration: ExperimentConfig | str | Path, *, dry_run: bool = False
) -> RunSummary:
    """Preflight all frames, then collect frame failures; any failure makes overall status failed."""
    config = (
        configuration
        if isinstance(configuration, ExperimentConfig)
        else load_experiment_config(configuration)
    )
    start = _now()
    report = preflight_experiment(config)
    if report.status != "ready" or dry_run:
        return RunSummary(
            report.status, config.config_sha256, config.frame_ids, failures=report.failures
        )
    environment = _environment()
    detector = None
    if config.model.source in ("inference", "modal"):
        try:
            if config.model.source == "inference" and file_sha256(
                config.model.config
            ) != report.model_evidence.get("config_sha256"):
                raise ValueError("model config changed since preflight")
            detector = _build_detector(config, report)
            detector.initialize()
        except Exception as exc:  # noqa: BLE001 -- Serialize external execution failures, never fallback.
            if detector is not None:
                try:
                    detector.close()
                except Exception as cleanup:  # noqa: BLE001 -- Preserve initialization and cleanup errors.
                    exc = RuntimeError(f"{exc}; cleanup failed: {cleanup}")
            return RunSummary(
                "blocked",
                config.config_sha256,
                config.frame_ids,
                failures=(RunFailure("backend_initialization", str(exc)),),
            )
    directory = None
    marker = config.run_directory / f"{config.output.artifact_prefix}.started.json"
    frames, failures = [], []
    try:
        directory = reserve_run(config)
        write_json_exclusive(
            marker,
            {
                "run_schema_version": 1,
                "status": "started",
                "config_sha256": config.config_sha256,
                "started_at_utc": start,
            },
        )
        for index, frame in enumerate(report.frames):
            try:
                frames.append(_execute_frame(config, frame, detector))
            except Exception as exc:  # noqa: BLE001 -- Serialize external execution failures, never fallback.
                failure = RunFailure("frame_execution", str(exc), frame.frame_id)
                failures.append(failure)
                frames.append(
                    {
                        "frame_id": frame.frame_id,
                        "status": "failed",
                        "failure": asdict(failure),
                        "inputs": {k: v.to_dict() for k, v in frame.files},
                    }
                )
                if config.model.source == "modal":
                    # An uncertain remote outcome must not trigger further chargeable frames.
                    for remaining in report.frames[index + 1 :]:
                        skipped = RunFailure(
                            "remote_aborted",
                            "not submitted after remote failure",
                            remaining.frame_id,
                        )
                        failures.append(skipped)
                        frames.append(
                            {
                                "frame_id": remaining.frame_id,
                                "status": "failed",
                                "failure": asdict(skipped),
                                "inputs": {k: v.to_dict() for k, v in remaining.files},
                            }
                        )
                    break
    except (OSError, ValueError, TypeError) as exc:
        failures.append(RunFailure("output_creation", str(exc)))
    finally:
        if detector is not None:
            try:
                detector.close()
            except Exception as exc:  # noqa: BLE001 -- Serialize external execution failures, never fallback.
                failures.append(RunFailure("backend_cleanup", str(exc)))
    succeeded = sum(f["status"] == "success" for f in frames)
    failed = sum(f["status"] == "failed" for f in frames)
    summary = RunSummary(
        "failed" if failures else "success",
        config.config_sha256,
        config.frame_ids,
        succeeded,
        failed,
        directory,
        tuple(failures),
    )
    if directory is None:
        return summary
    try:
        if any(f.phase == "output_creation" for f in failures):
            raise OSError("run output initialization failed")
        provenance = {
            "run_schema_version": 1,
            "started_at_utc": start,
            "ended_at_utc": _now(),
            "config_sha256": config.config_sha256,
            "configuration": config.to_dict(),
            "selection": config.selection_provenance,
            "split": report.split_evidence,
            "model_preflight": report.model_evidence,
            "environment": environment,
            "inputs": [
                {"frame_id": f.frame_id, "files": {k: v.to_dict() for k, v in f.files}}
                for f in report.frames
            ],
            "failure_policy": "collect_frames_any_failure_fails_run",
        }
        persist_run(config, summary, frames, provenance)
        return summary
    except (OSError, ValueError, TypeError) as exc:
        failure = RunFailure("artifact_persistence", str(exc))
        summary = RunSummary(
            "failed",
            config.config_sha256,
            config.frame_ids,
            succeeded,
            failed,
            directory,
            (*failures, failure),
        )
        try:
            write_json_exclusive(
                directory / f"{config.output.artifact_prefix}.failure.json", summary.to_dict()
            )
        except (OSError, ValueError, TypeError):
            pass  # The typed/CLI failure is still returned when the filesystem cannot record it.
        return summary
