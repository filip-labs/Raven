"""Offline export bundle integrity/CSV-content checks; original references are optional."""

from dataclasses import fields
from pathlib import Path

from raven.exporting.csv_tables import compact, tables
from raven.exporting.evidence import Evidence, Snapshot, require, sha


class EvidenceSet:
    def __init__(self, records, snapshot, *, bundle=None):
        self.records = tuple(records)
        self.snapshot = snapshot
        self.bundle = bundle


def load_bundle(path, snapshot):
    manifest = snapshot.json(path)
    require(
        type(manifest.get("export_schema_version")) is int
        and manifest["export_schema_version"] == 1,
        "unsupported export manifest schema",
    )
    require(manifest.get("status") == "success", "incomplete export bundle")
    path = Path(path).resolve()
    artifacts = manifest.get("artifacts")
    require(isinstance(artifacts, dict), "export artifact inventory missing")
    json_names = set(artifacts) & {"results.json", "metadata.json"}
    require(len(json_names) == 1, "export requires one JSON envelope")
    allowed = {
        "results.json",
        "metadata.json",
        "experiment_summary.csv",
        "evaluation_metrics.csv",
        "benchmark_summaries.csv",
        "raw_timing_samples.csv",
    }
    require(
        set(artifacts) <= allowed
        and len({r["file"] for r in artifacts.values()}) == len(artifacts),
        "unsupported/duplicate export files",
    )
    data = {}
    for name, record in artifacts.items():
        require(
            record.get("file") == name and "/" not in name and "\\" not in name,
            "unsafe export artifact name",
        )
        target = path.parent / name
        require(
            not target.is_symlink() and target.resolve().parent == path.parent,
            "export artifact symlink/redirection",
        )
        snapshot.identity(target, name, record.get("sha256"))
        raw = snapshot.read(target)
        require(
            type(record.get("bytes")) is int and len(raw) == record["bytes"],
            "export artifact size mismatch",
        )
        data[name] = raw
    envelope = snapshot.json(path.parent / next(iter(json_names)))
    require(
        type(envelope.get("export_schema_version")) is int
        and envelope["export_schema_version"] == 1,
        "unsupported export envelope schema",
    )
    content = envelope["content"]
    # Export content uses UTF-8 ensure_ascii=False canonicalization, unlike some older helpers.
    import hashlib

    content_hash = hashlib.sha256(compact(content).encode()).hexdigest()
    require(
        content_hash == sha(envelope["content_sha256"]) == sha(manifest["content_sha256"]),
        "export deterministic content checksum mismatch",
    )
    require(
        isinstance(content.get("sources"), list) and 1 <= len(content["sources"]) <= 32,
        "export source inventory missing",
    )
    records = []
    expected = {f.name for f in fields(Evidence)}
    for source in content["sources"]:
        require(set(source) == expected, "unsupported exported Evidence fields")
        require(
            source["source_schema_version"] == 1 and type(source["source_schema_version"]) is int,
            "unsupported embedded source schema",
        )
        require(
            source["source_type"] in ("experiment", "evaluation", "benchmark"),
            "unsupported embedded source type",
        )
        primary = source["source_artifacts"][0]
        require(
            source["source_id"] == source["source_type"] + "-" + sha(primary["sha256"]),
            "embedded source identity mismatch",
        )
        record = Evidence(**source)
        from raven.inference.evidence import validate_archived_remote_run

        validate_archived_remote_run(record)
        records.append(record)
    require(len({r.source_id for r in records}) == len(records), "duplicate embedded evidence")
    require(
        manifest.get("sources")
        == [
            {
                "source_id": r.source_id,
                "source_type": r.source_type,
                "artifacts": list(r.source_artifacts),
            }
            for r in records
        ],
        "export source inventory disagreement",
    )
    csv_names = set(data) - json_names
    if csv_names:
        expected_csv = tables(records, raw_timings="raw_timing_samples.csv" in data)
        require(set(expected_csv) == csv_names, "incomplete CSV set")
        for name, text in expected_csv.items():
            require(
                data[name] == text.encode("utf-8"), f"CSV disagrees with embedded evidence: {name}"
            )
    from raven.exporting.export import comparison

    require(
        content["comparison"] == comparison(records),
        "comparison dimensions disagree with embedded evidence",
    )
    snapshot.verify_unchanged()
    return EvidenceSet(
        records,
        snapshot,
        bundle={
            "content_sha256": content_hash,
            "generation": envelope["generation"],
            "scope": "Self-contained file hashes/content/CSV projection; original source references not yet checked.",
        },
    )


def load_set(path):
    snapshot = Snapshot()
    document = snapshot.json(path)
    if "export_schema_version" in document:
        return load_bundle(path, snapshot)
    from raven.exporting.evidence import load_evidence

    record = load_evidence(path, snapshot)
    snapshot.verify_unchanged()
    return EvidenceSet([record], snapshot)
