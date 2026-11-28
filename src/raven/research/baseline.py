"""Offline M6 planning using existing experiment, input and evaluator boundaries.

Nothing here creates a transport, imports the Modal SDK or executes inference.
Planning readiness is never execution authorization or measured performance.
"""

import hashlib
import importlib.metadata
import os
from dataclasses import asdict
from pathlib import Path

from raven.evaluation.backend import CLASSES, DIFFICULTIES, IDENTITY, metric_reference
from raven.evaluation.geometry import (
    CalibrationAdapter,
    ground_truth_annotation,
    image_metadata,
)
from raven.evaluation.inputs import strict_json
from raven.experiments.artifacts import write_json_exclusive, write_text_exclusive
from raven.experiments.config import ExperimentConfig, load_experiment_config
from raven.experiments.runner import load_experiment_frame, preflight_experiment
from raven.experiments.splits import split_membership
from raven.inference.contracts import MAX_POINTS, digest, model_identity
from raven.perception.data.frame_manifest import read_frame_manifest, validate_frame_ids

SUBSET_SHA256 = "1e974d53fd9a93f3ad0cef16d1d1b5bfc02f21461a098b2ee0236449c44727a1"


def frozen_selection():
    """Load exact committed bytes, never resample or replace unavailable frames."""
    packaged = Path(__file__).with_name("baseline_selection")
    directory = (
        packaged if packaged.is_dir() else Path(__file__).resolve().parents[3] / "configs/research"
    )
    ids, checksum = read_frame_manifest(directory / "m6-baseline-256.frames.txt")
    record = strict_json((directory / "m6-baseline-256.selection.json").read_bytes())
    members, evidence = split_membership("val")
    if (
        checksum != SUBSET_SHA256
        or checksum != record["manifest_sha256"]
        or len(ids) != record["count"]
        or len(ids) != 256
        or record["canonical_count"] != len(members)
        or not set(ids) <= members
        or tuple(sorted(ids)) != ids
        or record["split"] != evidence
        or record["seed"] != 42
        or record["selection_version"] != 1
        or record["algorithm"] != "PCG64.choice-without-replacement/input-order/v1"
        or record["candidate_order"] != "ascending-six-digit-ID"
        or record["model"] != model_identity()
        or record["preprocessing"] != []
        or record["preprocessing_sha256"] != digest([])
    ):
        raise ValueError("frozen baseline selection identity mismatch")
    return ids, record


def validate_baseline_config(config):
    """Require the fixed val/raw/Modal condition, without weakening schema-v2 guards."""
    frozen, _ = frozen_selection()
    if (
        config.schema_version != 2
        or config.dataset.split != "val"
        or config.model.source != "modal"
        or config.preprocessing
        or config.seed != 42
    ):
        raise ValueError("baseline requires schema 2, val, modal, seed 42 and preprocessing []")
    if not set(config.frame_ids) <= set(frozen):
        raise ValueError("baseline frames must belong to the frozen canonical validation subset")
    if (config.model.remote.app, config.model.remote.environment) != ("raven-pointpillars", "main"):
        raise ValueError("baseline requires the reviewed raven-pointpillars/main deployment target")


def baseline_configuration(
    template, *, dataset_root, output_directory, experiment_id, frame_count=1
):
    """Adapt the existing template into ONE bounded run, not a multi-run batch."""
    if type(frame_count) is not int or not 1 <= frame_count <= 8:
        raise ValueError("frame_count must be an integer in [1,8]")
    template = load_experiment_config(template)
    if template.schema_version != 2 or template.model.source != "modal" or template.preprocessing:
        raise ValueError("template must be a schema-v2 raw Modal experiment")
    ids, _ = frozen_selection()
    document = template.to_dict()
    document.update(
        name="M6 raw validation baseline pilot (not representative accuracy)",
        experiment_id=experiment_id,
        dataset={"root": str(Path(dataset_root).resolve()), "split": "val"},
        frames={"ids": list(ids[:frame_count])},
        seed=42,
        output={"directory": str(Path(output_directory).resolve()), "artifact_prefix": "baseline"},
    )
    document["model"]["remote"]["max_frames"] = frame_count
    config = ExperimentConfig.from_dict(document, base_dir=Path.cwd())
    validate_baseline_config(config)
    return config


def inspect_baseline_inputs(config, frame_ids=None):
    """Read and fingerprint selected inputs; count eligibility via the pinned evaluator.

    All missing/invalid files are retained as failures, never silently substituted.
    Images are validated through RV-032's existing PNG metadata reader, not guessed.
    """
    validate_baseline_config(config)
    ids = config.frame_ids if frame_ids is None else validate_frame_ids(frame_ids)
    frozen, _ = frozen_selection()
    if not set(ids) <= set(frozen):
        raise ValueError("input audit frames must belong to the frozen validation subset")
    coverage = {name: dict.fromkeys(DIFFICULTIES, 0) for name in CLASSES}
    rows, labels_verified = [], 0
    reference = metric_reference()
    for frame_id in ids:
        row = {"frame_id": frame_id, "files": {}, "failures": []}
        try:
            inputs, cloud, labels, calibration, _, _ = load_experiment_frame(config, frame_id)
            if cloud.coordinate_frame != "kitti_velodyne" or cloud.count > MAX_POINTS:
                raise ValueError("scan violates Modal KITTI Velodyne/point-count contract")
            CalibrationAdapter(calibration)
            for r in labels.records:
                if r.object_type in CLASSES and (
                    r.occlusion < 0 or r.truncation < 0 or all(v == -1 for v in r.bbox_xyxy)
                ):
                    raise ValueError("ground-truth visibility/image-box metadata unavailable")
            row["files"] = {name: identity.to_dict() for name, identity in inputs.files}
            row["points"] = cloud.count
            row["coordinate_frame"] = cloud.coordinate_frame
            annotation = ground_truth_annotation(labels)
            for c, name in enumerate(CLASSES):
                for d, difficulty in enumerate(DIFFICULTIES):
                    eligible, *_ = reference.clean_data(annotation, {"name": []}, c, d)
                    coverage[name][difficulty] += int(eligible)
            labels_verified += 1
        except (OSError, ValueError, TypeError) as exc:
            row["failures"].append({"input": "scan_labels_calibration", "message": str(exc)})
        image = config.dataset.scan_directory.parent / "image_2" / f"{frame_id}.png"
        try:
            shape, metadata = image_metadata(image)
            row["files"]["image"] = {**metadata, "shape": list(shape)}
        except (OSError, ValueError, TypeError) as exc:
            row["failures"].append({"input": "image", "message": str(exc)})
        rows.append(row)
    warnings = [
        "Eligibility is cumulative by difficulty, not disjoint counts or measured AP.",
        "AP_R40 can be unstable with sparse eligible objects; >=41 is not a statistical sufficiency guarantee.",
    ]
    for cls, counts in coverage.items():
        for difficulty, count in counts.items():
            if count == 0:
                warnings.append(f"{cls}/{difficulty}: no verified eligible objects; AP unsupported")
            elif count < 41:
                warnings.append(
                    f"{cls}/{difficulty}: only {count} eligible objects; sparse AP_R40 sampling"
                )
    return {
        "status": "ready" if all(not r["failures"] for r in rows) else "blocked",
        "frame_ids": list(ids),
        "frame_count": len(ids),
        "frames": rows,
        "eligible_ground_truth": coverage,
        "coverage_complete": labels_verified == len(ids),
        "verified_label_frames": labels_verified,
        "warnings": warnings,
    }


def baseline_preflight(config, *, inspect_subset=False):
    """Report local readiness and unresolved live checks, even when env gates are set."""
    if type(inspect_subset) is not bool:
        raise TypeError("inspect_subset must be a boolean")
    validate_baseline_config(config)
    ids, selection = frozen_selection()
    inputs = inspect_baseline_inputs(config, ids if inspect_subset else None)
    preflight = preflight_experiment(config)  # Existing offline dry-run checks only.
    from raven.inference.modal_image import recipe_sha256
    from raven.inference.worker import runtime_source_sha256

    try:
        modal_version = importlib.metadata.version("modal")
    except importlib.metadata.PackageNotFoundError:
        modal_version = None
    return {
        "planning_version": 1,
        "purpose": "offline_preparation_not_measurements",
        "configuration_sha256": config.config_sha256,
        "selection": selection,
        "run_frame_ids": list(config.frame_ids),
        "inputs": inputs,
        "runner_preflight": {
            "status": preflight.status,
            "failures": [asdict(v) for v in preflight.failures],
        },
        "evaluation_readiness": "inputs_ready_predictions_pending"
        if inputs["status"] == "ready"
        else "blocked_inputs",
        "metrics": None,
        "evaluator": IDENTITY,
        "model": model_identity(),
        "preprocessing_sha256": digest([]),
        "deployment_requirements": {
            "remote": config.model.remote.to_dict(),
            "gpu": "T4",
            "worker_source_sha256": runtime_source_sha256(),
            "image_recipe_sha256": recipe_sha256(),
            "live_deployment_identity": "not_checked_offline",
            "account_billing_budget": "not_checked_offline",
        },
        "cloud": {
            "authorization": "not_authorized_for_this_plan",
            "operations_performed": 0,
            "installed_modal_version": modal_version,
            "environment_gates": {
                key: os.environ.get(key) == "1"
                for key in ("RAVEN_ALLOW_MODAL_GPU", "RAVEN_MODAL_ASSET_RIGHTS_CONFIRMED")
            },
            "new_scope_approval_required": True,
        },
    }


def prepare_baseline(
    template,
    *,
    dataset_root,
    output_directory,
    experiment_id,
    plan_directory,
    frame_count=1,
    inspect_subset=False,
):
    """Publish a reviewable configuration/report into a new exclusive plan directory.

    Failed readiness is still saved as a blocked plan, never as a successful run.
    No run directory is reserved. Failure during persistence leaves partial planning
    evidence for diagnosis and cannot overwrite a prior plan on retry.
    """
    config = baseline_configuration(
        template,
        dataset_root=dataset_root,
        output_directory=output_directory,
        experiment_id=experiment_id,
        frame_count=frame_count,
    )
    target = Path(plan_directory).absolute()
    if target != target.resolve() or ".." in target.parts:
        raise ValueError("plan directory cannot redirect through symlinks or traversal")
    # Disjoint from source data, future outputs, and template; don't contaminate input roots.
    for protected in (config.dataset.root, config.output.directory, Path(template).resolve()):
        if target == protected or target in protected.parents or protected in target.parents:
            raise ValueError("plan directory must be disjoint from dataset, outputs and template")
    if target.exists():
        raise FileExistsError(f"refusing existing plan directory: {target}")
    if config.run_directory.exists():
        raise FileExistsError(f"refusing existing run directory: {config.run_directory}")
    report = baseline_preflight(config, inspect_subset=inspect_subset)
    target.mkdir(parents=True, exist_ok=False)
    write_json_exclusive(target / "experiment.json", config.to_dict())
    write_json_exclusive(target / "preflight.json", report)
    ids, _ = frozen_selection()
    raw = "".join(f"{i}\n" for i in ids)
    assert hashlib.sha256(raw.encode()).hexdigest() == SUBSET_SHA256
    write_text_exclusive(target / "subset.frames.txt", raw)
    return {
        "plan_directory": str(target),
        "configuration": str(target / "experiment.json"),
        "report": str(target / "preflight.json"),
        "evaluation_readiness": report["evaluation_readiness"],
        "run_frame_ids": list(config.frame_ids),
        "cloud_authorization": report["cloud"]["authorization"],
        "runner_preflight_status": report["runner_preflight"]["status"],
    }
