"""Scoped read-only validation, optional CPU replay and controlled artifact comparisons."""

from dataclasses import asdict, dataclass
from pathlib import Path

from raven.benchmark.experiment import cloud_digest, digest
from raven.experiments.artifacts import file_sha256
from raven.exporting.evidence import Snapshot, load_evidence, require
from raven.perception.data.kitti_loader import load_kitti_bin
from raven.perception.preprocessing.pipeline import run_preprocessing_pipeline
from raven.reproducibility.bundle import load_set
from raven.reproducibility.policy import POLICY, differences, projection


@dataclass(frozen=True)
class VerificationReport:
    status: str
    sources: tuple[dict, ...]
    checks: dict
    tolerance_policy: dict
    scientific_projection_sha256: tuple[str, ...]
    environment_differences: tuple[dict, ...]
    failures: tuple[str, ...]
    verification_schema_version: int = 1

    def to_dict(self):
        return asdict(self)


def _check(status, **detail):
    return {"status": status, **detail}


def _primary(record):
    return record.source_artifacts[0]["path"]


def _references(evidence):
    if evidence.bundle is None:
        return evidence.records
    snapshot = Snapshot()
    checked = []
    for record in evidence.records:
        selection = (record.configuration or {}).get("frames", {})
        if selection.get("manifest") is not None:
            manifest_path = Path(selection["manifest"])
            if not manifest_path.exists():
                raise FileNotFoundError(f"selection manifest unavailable: {manifest_path}")
            snapshot.identity(manifest_path, "selection_manifest", selection["sha256"])
        for artifact in record.source_artifacts:
            if not Path(artifact["path"]).exists():
                raise FileNotFoundError(
                    f"original source reference unavailable: {artifact['path']}"
                )
            snapshot.identity(artifact["path"], artifact["role"], artifact["sha256"])
        loaded = load_evidence(_primary(record), snapshot)
        require(
            digest(loaded.to_dict()) == digest(record.to_dict()),
            "original source and embedded export content disagree",
        )
        checked.append(loaded)
    snapshot.verify_unchanged()
    return tuple(checked)


def _input_records(record):
    if record.source_type == "experiment":
        return [
            (row["frame_id"], row["inputs"], row) for row in record.payload["results"]["frames"]
        ]
    if record.source_type == "benchmark":
        return [
            (
                f["frame_id"],
                {
                    "scan": {
                        "path": f["source"]["path"],
                        "sha256": f["source"]["sha256"],
                        "bytes": f["source"]["bytes"],
                    }
                },
                f,
            )
            for f in record.payload["frames"]
        ]
    if record.source_type == "evaluation" and record.run_manifest_sha256 is not None:
        path = record.payload["experiment_identity"]["manifest_path"]
        snapshot = Snapshot()
        return _input_records(load_evidence(path, snapshot))
    return []


def _declared_inputs(record):
    if record.source_type == "evaluation":
        data = record.payload["provenance"].get(
            "evaluation_inputs", record.payload["provenance"].get("evaluation_inputs_checked", [])
        )
        return [{"frame_id": r["frame_id"], "files": r["inputs"]} for r in data]
    return [{"frame_id": r[0], "files": r[1]} for r in _input_records(record)]


def _verify_eval_link(record):
    if record.source_type != "evaluation" or record.status != "success":
        return
    snapshot = Snapshot()
    original = load_evidence(record.payload["experiment_identity"]["manifest_path"], snapshot)
    require(
        record.payload["provenance"].get("source_execution") == original.payload["provenance"],
        "evaluation source_execution provenance disagrees with referenced run",
    )
    snapshot.verify_unchanged()


def _inputs(records, replay):
    rows = []
    total = sum(len(_input_records(r)) for r in records)
    require(
        not replay or total <= 32, "CPU replay limited to 32 frames; no implicit subset selection"
    )
    for record in records:
        require(record.configuration is not None, "source has no validated configuration")
        require(record.status == "success", "replay/input claims require successful source results")
        frame_inputs = _input_records(record)
        require(bool(frame_inputs), "no original frame input identities available")
        stages = record.configuration["preprocessing"]
        for frame_id, files, original in frame_inputs:
            for role, identity in files.items():
                path = Path(identity["path"])
                if not path.exists() and not path.is_symlink():
                    raise FileNotFoundError(f"original input unavailable: {path}")
                require(
                    not path.is_symlink() and path.is_file(),
                    f"original {role} unavailable or redirected: {path}",
                )
                require(
                    path.stat().st_size == identity["bytes"]
                    and file_sha256(path) == identity["sha256"],
                    f"original input checksum/size mismatch: {path}",
                )
            entry = {"source_id": record.source_id, "frame_id": frame_id, "inputs": files}
            if replay:
                scan = Path(files["scan"]["path"])
                require(scan.stat().st_size <= 128_000_000, "replay scan exceeds 128 MB bound")
                cloud = load_kitti_bin(scan)
                require(
                    cloud.frame_id == frame_id and cloud_digest(cloud) == files["scan"]["sha256"],
                    "replay frame/input identity mismatch",
                )
                before = cloud.points.copy()
                processed = run_preprocessing_pipeline(cloud, stages).cloud
                require(
                    (cloud.points == before).all(), "preprocessing mutated original replay cloud"
                )
                if record.source_type == "benchmark":
                    observed = [o for o in original["observations"] if o["status"] == "success"]
                    require(bool(observed), "benchmark lacks completed observations")
                    expected_hashes = {o["processed_xyzi_sha256"] for o in observed}
                    expected_counts = {o["output_count"] for o in observed}
                else:
                    expected_hashes = {original["processed_xyzi_sha256"]}
                    expected_counts = {original["processed_count"]}
                require(
                    {cloud_digest(processed)} == expected_hashes
                    and {processed.count} == expected_counts,
                    "CPU replay processed hash/count mismatch",
                )
                require(
                    processed.frame_id == frame_id
                    and processed.coordinate_frame == cloud.coordinate_frame,
                    "CPU replay changed coordinate metadata",
                )
                require(
                    file_sha256(scan) == files["scan"]["sha256"], "source changed during replay"
                )
                entry.update(
                    processed_xyzi_sha256=cloud_digest(processed), output_count=processed.count
                )
            rows.append(entry)
    return rows


def verify_results(
    path, *, compare=None, check_sources=False, check_inputs=False, replay_preprocessing=False
):
    """No GPU/model execution; CPU preprocessing replay occurs ONLY on explicit request."""
    for flag in (check_sources, check_inputs, replay_preprocessing):
        if type(flag) is not bool:
            raise TypeError("verification flags must be explicit booleans")
    checks = {
        "artifact_integrity": _check("not_checked"),
        "original_source_references": _check("not_requested"),
        "original_input_integrity": _check("not_requested"),
        "preprocessing_replay": _check("not_requested"),
        "deterministic_comparison": _check("not_requested"),
        "numerical_evaluation_comparison": _check("not_requested"),
        "actual_model_inference_reproduction": _check(
            "not_performed",
            reason="No model initialization, weights, GPU or remote calls are performed.",
        ),
        "cross_environment_reproducibility": _check(
            "not_established",
            reason="Matching declared evidence is not proof across hardware/software environments.",
        ),
    }
    summaries = []
    hashes = []
    environment_changes = []
    failures = []
    status = "integrity_only"
    try:
        left = load_set(path)
        for record in left.records:
            if left.bundle is None:
                _verify_eval_link(record)
            science, _variable, _ = projection(record)
            hashes.append(digest(science))
            summaries.append(
                {
                    "source_id": record.source_id,
                    "source_type": record.source_type,
                    "source_status": record.status,
                    "evidence_kind": record.evidence_kind,
                    "config_sha256": record.config_sha256,
                    "frame_ids": list(record.frame_ids),
                    "canonical_split": record.canonical_split,
                    "configuration": record.configuration,
                    "input_identities": _declared_inputs(record),
                    "source_artifacts": list(record.source_artifacts),
                    "declared_environment": record.environment,
                }
            )
        checks["artifact_integrity"] = _check(
            "verified",
            scope="source schema/checksums/consistency"
            if left.bundle is None
            else left.bundle["scope"],
            exact_export_content_sha256=left.bundle["content_sha256"] if left.bundle else None,
            authenticity="not independently established",
        )
        original = left.records
        if check_sources or check_inputs or replay_preprocessing:
            try:
                original = _references(left)
                for record in original:
                    _verify_eval_link(record)
                checks["original_source_references"] = _check("verified")
            except FileNotFoundError as exc:
                checks["original_source_references"] = _check("unavailable", reason=str(exc))
                status = "unavailable"
                original = None
            except (OSError, ValueError, TypeError) as exc:
                checks["original_source_references"] = _check("failed", reason=str(exc))
                raise
        if check_inputs or replay_preprocessing:
            if original is None:
                checks["original_input_integrity"] = _check(
                    "unavailable", reason="Original source references unavailable."
                )
                if replay_preprocessing:
                    checks["preprocessing_replay"] = _check("unavailable")
            else:
                try:
                    inputs = _inputs(original, replay_preprocessing)
                    checks["original_input_integrity"] = _check("verified", frames=inputs)
                    if replay_preprocessing:
                        checks["preprocessing_replay"] = _check(
                            "verified",
                            scope="CPU preprocessing only; bitwise processed hash/count and metadata",
                            frames=len(inputs),
                        )
                    status = "checks_passed"
                except FileNotFoundError as exc:
                    checks["original_input_integrity"] = _check("unavailable", reason=str(exc))
                    status = "unavailable"
                    if replay_preprocessing:
                        checks["preprocessing_replay"] = _check("unavailable")
                except (OSError, ValueError, TypeError) as exc:
                    checks["original_input_integrity"] = _check("failed", reason=str(exc))
                    if replay_preprocessing:
                        checks["preprocessing_replay"] = _check("failed", reason=str(exc))
                    raise
        if compare is not None:
            right = load_set(compare)
            if right.bundle is None:
                for record in right.records:
                    _verify_eval_link(record)
            require(len(left.records) == len(right.records), "comparison source counts differ")
            require(
                all(
                    a.source_type == b.source_type
                    for a, b in zip(left.records, right.records, strict=True)
                ),
                "comparison source types/order differ",
            )
            if any(r.status != "success" for r in (*left.records, *right.records)):
                checks["deterministic_comparison"] = _check(
                    "unavailable",
                    reason="Successful scientific outputs required; blocked/failed outcomes cannot prove replay.",
                )
                status = "unavailable"
            else:
                science_changes = []
                numeric_changes = []
                numeric_count = 0
                null_count = 0
                for i, (a, b) in enumerate(zip(left.records, right.records, strict=True)):
                    sa, ea, na = projection(a)
                    sb, eb, nb = projection(b)
                    science_changes.extend(differences(sa, sb, path=f"$[{i}]"))
                    environment_changes.extend(differences(ea, eb, path=f"$[{i}]"))
                    require(len(na) == len(nb), "evaluation metric-vector lengths differ")
                    for x, y in zip(na, nb, strict=True):
                        require(
                            x["identity"] == y["identity"],
                            "numerical comparison metric identities differ",
                        )
                        u, v = x["ap_r40_percent"], y["ap_r40_percent"]
                        if u is None or v is None:
                            if u is not None or v is not None:
                                numeric_changes.append(
                                    {"metric": x["identity"], "left": u, "right": v}
                                )
                            else:
                                null_count += 1
                        else:
                            numeric_count += 1
                            if abs(u - v) > POLICY["ap_r40_absolute_tolerance_percent"]:
                                numeric_changes.append(
                                    {
                                        "metric": x["identity"],
                                        "left": u,
                                        "right": v,
                                        "absolute_error_percent": abs(u - v),
                                    }
                                )
                checks["deterministic_comparison"] = _check(
                    "matched" if not science_changes else "mismatch",
                    differences=science_changes,
                    scope="Explicit scientific projection; not raw run/artifact byte equality",
                    exact_export_content_sha256_equal=(
                        left.bundle["content_sha256"] == right.bundle["content_sha256"]
                    )
                    if left.bundle and right.bundle
                    else None,
                )
                checks["numerical_evaluation_comparison"] = _check(
                    "mismatch"
                    if numeric_changes
                    else "matched"
                    if numeric_count
                    else "unavailable",
                    compared_AP_values=numeric_count,
                    matching_null_metrics=null_count,
                    differences=numeric_changes,
                )
                if science_changes or numeric_changes:
                    status = "failed"
                    failures.append("Declared scientific or numerical comparison mismatch.")
                elif status != "unavailable":
                    status = "checks_passed"
            right.snapshot.verify_unchanged()
        left.snapshot.verify_unchanged()
    except (OSError, ValueError, TypeError, KeyError, AttributeError, IndexError) as exc:
        status = "failed"
        failures.append(f"{type(exc).__name__}: {exc}")
        if checks["artifact_integrity"]["status"] == "not_checked":
            checks["artifact_integrity"] = _check("failed", reason=str(exc))
    return VerificationReport(
        status,
        tuple(summaries),
        checks,
        POLICY,
        tuple(hashes),
        tuple(environment_changes),
        tuple(failures),
    )
