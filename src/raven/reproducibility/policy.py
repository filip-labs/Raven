"""Explicit scientific/environment partitions; unknown fields remain scientific."""

from copy import deepcopy

from raven.benchmark.experiment import digest

POLICY = {
    "projection_version": 1,
    "ap_r40_absolute_tolerance_percent": 1e-8,
    "ap_r40_relative_tolerance": 0.0,
    "benchmark_statistics_validation": {"absolute_ms": 1e-12, "relative": 1e-12},
    "processed_xyzi": "bitwise little-endian float32; same inputs/seeds/compatible environment",
    "environment_dependent": [
        "configuration.output",
        "input reference paths",
        "pipeline duration_seconds",
        "pipeline reproducibility software versions",
        "run environment/start/end timestamps",
        "benchmark raw durations/statistical values and audit/output locations",
        "evaluation execution provenance/timestamp and original run hash",
        "export generation metadata",
    ],
    "rule": "Only explicitly named fields are partitioned; unknown scientific fields remain comparable.",
    "latency_equality": "never required for real hardware; statistics are validated against each source's own raw samples",
}


def _take(value, keys):
    return {k: value.pop(k) for k in keys if k in value}


def _pipeline(value):
    science = deepcopy(value)
    variability = _take(science, ["duration_seconds"])
    versions = science.get("reproducibility", {})
    variability["software_versions"] = _take(versions, ["python", "numpy", "scipy"])
    variability["stages"] = []
    for stage in science.get("stages", []):
        variability["stages"].append(_take(stage, ["duration_seconds"]))
    return science, variability


def _inputs(value):
    scientific, locations = {}, {}
    for role, item in value.items():
        scientific[role] = deepcopy(item)
        locations[role] = _take(scientific[role], ["path"])
    return scientific, locations


def projection(record):
    """Return scientific fields, explicitly variable fields and AP-only numeric vector."""
    config = deepcopy(record.configuration)
    variable = {
        "source_artifacts": deepcopy(record.source_artifacts),
        "config_sha256": record.config_sha256,
        "environment": deepcopy(record.environment),
        "run_manifest_sha256": record.run_manifest_sha256,
    }
    if config is not None:
        variable["configuration_output"] = config.pop("output")
    scientific = {
        "source_type": record.source_type,
        "status": record.status,
        "evidence_kind": record.evidence_kind,
        "configuration": config,
        "canonical_split": deepcopy(record.canonical_split),
        "frame_ids": list(record.frame_ids),
        "unavailable": deepcopy(record.unavailable),
    }
    value = deepcopy(record.payload)
    numerical = []
    if record.source_type == "experiment":
        results = value["results"]
        scientific["run_schema_version"] = results["run_schema_version"]
        scientific["summary"] = deepcopy(results["summary"])
        variable["run_summary"] = _take(
            scientific["summary"], ["output_directory", "config_sha256"]
        )
        provenance = deepcopy(value["provenance"])
        variable["provenance"] = _take(
            provenance,
            ["started_at_utc", "ended_at_utc", "environment", "configuration", "config_sha256"],
        )
        # Frame identities below bind input paths independently of source-specific provenance paths.
        variable["provenance_inputs"] = provenance.pop("inputs", None)
        scientific["provenance"] = provenance
        scientific["frames"] = []
        variable["frames"] = []
        for row in results["frames"]:
            row["inputs"], paths = _inputs(row["inputs"])
            row["preprocessing"], timings = _pipeline(row["preprocessing"])
            extra = {"input_paths": paths, "preprocessing": timings}
            source = row.get("prediction_source", {})
            if source.get("mode") == "live_adapter" and isinstance(
                source.get("model_provenance"), dict
            ):
                metadata = source["model_provenance"]
                extra["model_environment"] = _take(metadata, ["environment", "raven_version"])
                if isinstance(metadata.get("config"), dict):
                    extra["model_config_path"] = _take(metadata["config"], ["path"])
            scientific["frames"].append(row)
            variable["frames"].append(extra)
        # Top-level extras are not silently discarded.
        scientific["extra_payload"] = {
            k: v
            for k, v in value.items()
            if k
            not in {
                "status",
                "manifest",
                "configuration",
                "frames",
                "results",
                "provenance",
                "started",
            }
        }
    elif record.source_type == "benchmark":
        scientific["experiment_extra"] = {
            k: v
            for k, v in value["experiment"].items()
            if k not in {"configuration", "config_sha256", "selection", "split", "inputs"}
        }
        scientific["selection"] = value["experiment"]["selection"]
        scientific["specification"] = value["specification"]
        scientific["methodology"] = value["methodology"]
        scientific["model_provenance"] = value["model_provenance"]
        variable["audit"] = _take(value, ["started_at_utc", "ended_at_utc", "output_directory"])
        scientific["frames"] = []
        variable["frames"] = []
        for frame in value["frames"]:
            source = frame["source"]
            source_path = _take(source, ["path"])
            counts = []
            timings = []
            for row in frame["observations"]:
                times = _take(row, ["durations_ns"])
                for op in row["operations"]:
                    times.setdefault("operations", []).append(_take(op, ["duration_ns"]))
                counts.append(row)
                timings.append(times)
            summaries = []
            raw_series = []
            for series in frame["series"]:
                raw_series.append(_take(series, ["raw_ns"]))
                summary = series.pop("summary")
                series["summary_contract"] = {
                    k: summary.pop(k) for k in ("count", "percentile_method", "std_ddof")
                }
                raw_series[-1]["statistics"] = summary
                summaries.append(series)
            scientific["frames"].append(
                {
                    "frame_id": frame["frame_id"],
                    "status": frame["status"],
                    "source": source,
                    "observations": counts,
                    "series": summaries,
                }
            )
            variable["frames"].append(
                {"source_path": source_path, "durations": timings, "series": raw_series}
            )
        scientific["initialization_contract"] = None
        if value["initialization"] is not None:
            init = value["initialization"]
            variable["initialization"] = _take(init, ["raw_ns", "summary"])
            scientific["initialization_contract"] = init
        known = {
            "status",
            "evidence_kind",
            "experiment",
            "specification",
            "environment",
            "frames",
            "initialization",
            "model_provenance",
            "unavailable",
            "failures",
            "benchmark_schema_version",
            "methodology",
            "specification_sha256",
        }
        scientific["extra_payload"] = {k: v for k, v in value.items() if k not in known}
        scientific["failures"] = value["failures"]
    elif record.source_type == "evaluation":
        variable["evaluation_provenance"] = value.pop("provenance")
        variable["identity"] = _take(value["experiment_identity"], ["manifest_path"])
        variable.update(_take(value, ["config_sha256", "run_manifest_sha256"]))
        # Paths here are declarations bound by configuration/verified source identities.
        variable["dataset_root"] = value["split"].pop("dataset_root", None)
        value["metrics"].sort(key=lambda m: (m["metric"], m["class_name"], m["difficulty"]))
        value["diagnostics"].sort(
            key=lambda m: (m["metric"], m["class_name"], m["difficulty"], m["frame_id"])
        )
        for metric in value["metrics"]:
            numerical.append(
                {
                    "identity": {k: metric[k] for k in ("metric", "class_name", "difficulty")},
                    "ap_r40_percent": metric.pop("ap_r40_percent"),
                }
            )
        scientific["evaluation"] = value
    return scientific, variable, numerical


def scientific_hash(record):
    return digest(projection(record)[0])


def differences(left, right, path="$", limit=100):
    """Bounded structural differences, not blanket field deletion or float coercion."""
    output = []

    def visit(a, b, p):
        if len(output) >= limit:
            return
        if type(a) is not type(b):
            output.append({"path": p, "left": a, "right": b})
            return
        if isinstance(a, dict):
            for key in sorted(a.keys() | b.keys()):
                if key not in a or key not in b:
                    output.append(
                        {
                            "path": p + "." + key,
                            "left": a.get(key),
                            "right": b.get(key),
                            "missing": True,
                        }
                    )
                else:
                    visit(a[key], b[key], p + "." + key)
                if len(output) >= limit:
                    break
        elif isinstance(a, (list, tuple)):
            if len(a) != len(b):
                output.append({"path": p + ".length", "left": len(a), "right": len(b)})
            for i, (x, y) in enumerate(zip(a, b, strict=False)):
                visit(x, y, f"{p}[{i}]")
        elif a != b:
            output.append({"path": p, "left": a, "right": b})

    visit(left, right, path)
    return output
