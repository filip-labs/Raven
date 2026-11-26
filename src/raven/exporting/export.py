"""Prepare a deterministic evidence projection, then exclusively publish a complete bundle."""

import hashlib
import json
import os
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

from raven.experiments.artifacts import write_text_exclusive
from raven.experiments.runner import _environment
from raven.exporting.csv_tables import compact, tables
from raven.exporting.evidence import Snapshot, load_evidence, require


class ExportWriteError(OSError):
    """Publication failed; no final successful manifest is retained."""


def dimensions(record):
    value = record.payload
    return {
        "dataset": record.dataset,
        "canonical_split": record.canonical_split,
        "frame_ids": list(record.frame_ids),
        "model": record.model,
        "preprocessing": record.preprocessing,
        "evidence_kind": record.evidence_kind,
        "source_status": record.status,
        "input_sizes": value.get("experiment", {}).get("inputs")
        if record.source_type == "benchmark"
        else value.get("provenance", {}).get("inputs"),
        "evaluation_protocol": value.get("evaluator")
        if record.source_type == "evaluation"
        else None,
        "hardware": {
            k: (record.environment or {}).get(k)
            for k in ("os", "machine", "cpu_model", "logical_cpu_count", "packages")
        },
        "timing_scope": value.get("methodology") if record.source_type == "benchmark" else None,
        "warmup_repetitions": value.get("specification")
        if record.source_type == "benchmark"
        else None,
        "source_type": record.source_type,
    }


def comparison(records):
    profiles = {r.source_id: dimensions(r) for r in records}
    pairs = []
    for left, right in combinations(records, 2):
        a, b = profiles[left.source_id], profiles[right.source_id]
        differences = [k for k in a if a[k] != b[k]]
        missing = [k for k in a if a[k] is None or b[k] is None]
        pairs.append(
            {
                "left": left.source_id,
                "right": right.source_id,
                "comparability": "not_assumed",
                "different_dimensions": differences,
                "missing_dimensions": missing,
                "warnings": [
                    "No performance/accuracy improvement is calculated; dimension agreement is not experimental comparability."
                ],
            }
        )
    return {
        "dimensions": profiles,
        "pairs": pairs,
        "policy": "Explicit dimensions only; no automatic pooling, rankings or improvement percentages.",
    }


def _destination(path, snapshot, records):
    destination = Path(path).absolute()
    require(".." not in destination.parts, "destination path traversal rejected")
    require(
        destination.resolve() == destination and not destination.is_symlink(),
        "destination symlink redirection rejected",
    )
    require(not destination.exists(), "export destination already exists")
    require(destination.parent.is_dir(), "export destination parent must already exist")
    require(
        destination not in (Path(destination.anchor), Path.home().resolve()),
        "unsafe export destination",
    )
    protected = {
        Path(a["path"]).parent
        for r in records
        for a in r.source_artifacts
        if a["role"] in ("run_manifest", "referenced_run_manifest", "benchmark_manifest")
    }
    for directory in protected:
        require(
            destination != directory and directory not in destination.parents,
            "export directory must be separate from source run/benchmark directories",
        )
    for source in snapshot.files:
        require(destination not in source.parents, "export destination contains source evidence")
    return destination


def export_results(sources, destination, *, format="json", raw_timings=False):
    """Export validated source snapshots only. Existing source evidence is never executed."""
    require(format in ("json", "csv", "all"), "format must be json, csv or all")
    require(
        type(raw_timings) is bool and (not raw_timings or format != "json"),
        "raw timings require CSV output",
    )
    require(
        isinstance(sources, (list, tuple)) and 1 <= len(sources) <= 32,
        "provide 1..32 explicit sources",
    )
    snapshot = Snapshot()
    records = tuple(load_evidence(path, snapshot) for path in sources)
    require(len({r.source_id for r in records}) == len(records), "duplicate source evidence")
    destination = _destination(destination, snapshot, records)
    snapshot.verify_unchanged()
    content = {"sources": [r.to_dict() for r in records], "comparison": comparison(records)}
    content_sha = hashlib.sha256(compact(content).encode("utf-8")).hexdigest()
    source_environment = _environment()
    envelope = {
        "export_schema_version": 1,
        "content_sha256": content_sha,
        "content": content,
        "generation": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "exporter": "raven-results-export",
            "format": format,
            "raven_commit": source_environment["raven_commit"],
            "raven_worktree_status": source_environment["raven_worktree_status"],
            "python": source_environment["python"],
            "source_sha256": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in Path(__file__).parent.glob("*.py")
            },
        },
        "validation": {
            "artifact_integrity": "recorded checksums and internal schema consistency",
            "gpu_authenticity": "not independently established",
            "statistics_validation_tolerance": {"relative": 1e-12, "absolute_ms": 1e-12},
            "csv_text_encoding": "prefix one apostrophe for formula-prone or apostrophe-leading text; use restore_text to reverse",
        },
    }
    payloads = {
        "metadata.json" if format == "csv" else "results.json": json.dumps(
            envelope, indent=2, ensure_ascii=False, allow_nan=False
        )
        + "\n"
    }
    if format in ("csv", "all"):
        payloads.update(tables(records, raw_timings))
    checksums = {
        name: hashlib.sha256(text.encode("utf-8")).hexdigest() for name, text in payloads.items()
    }
    manifest = {
        "export_schema_version": 1,
        "status": "success",
        "content_sha256": content_sha,
        "sources": [
            {
                "source_id": r.source_id,
                "source_type": r.source_type,
                "artifacts": list(r.source_artifacts),
            }
            for r in records
        ],
        "artifacts": {
            name: {"file": name, "sha256": checksum, "bytes": len(payloads[name].encode("utf-8"))}
            for name, checksum in checksums.items()
        },
    }
    payloads["manifest.json"] = (
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    # mkdir is the exclusive reservation; rename cannot safely reject existing empty directories on POSIX.
    reserved = False
    try:
        os.mkdir(destination, mode=0o700)
        reserved = True
        for name, text in payloads.items():
            if name == "manifest.json":
                snapshot.verify_unchanged()
            require(
                destination.resolve() == destination and not destination.is_symlink(),
                "destination changed during export",
            )
            write_text_exclusive(destination / name, text)
    except Exception as exc:
        if reserved and destination.is_dir() and not destination.is_symlink():
            for name, text in reversed(list(payloads.items())):
                path = destination / name
                if (
                    path.is_file()
                    and not path.is_symlink()
                    and path.read_bytes() == text.encode("utf-8")
                ):
                    path.unlink()
            try:
                destination.rmdir()
            except OSError:
                pass  # Unrelated/concurrently added files must never be removed.
        raise ExportWriteError(f"export publication failed: {exc}") from exc
    return {
        "export_schema_version": 1,
        "status": "success",
        "output_directory": str(destination),
        "source_count": len(records),
        "content_sha256": content_sha,
        "files": list(payloads),
        "warnings": [warning for r in records for warning in r.warnings],
    }
