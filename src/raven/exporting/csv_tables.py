"""Stable normalized UTF-8 CSV tables; reversible text-only formula neutralization."""

import csv
import io
import json
from pathlib import Path

from raven.benchmark.experiment import digest

EXPORT_VERSION = 1
COMMON = ("export_schema_version", "source_id", "source_type", "source_schema_version")
SUMMARY_COLUMNS = COMMON + (
    "experiment_id",
    "experiment_name",
    "config_sha256",
    "dataset_root",
    "dataset_split",
    "canonical_split_json",
    "frame_ids_json",
    "preprocessing_json",
    "model_json",
    "evidence_kind",
    "source_status",
    "run_manifest_sha256",
    "source_artifacts_json",
    "warnings_json",
    "unavailable_json",
)
EVALUATION_COLUMNS = COMMON + (
    "experiment_id",
    "run_manifest_sha256",
    "dataset_split",
    "evaluation_purpose",
    "evaluator_name",
    "evaluator_version",
    "protocol",
    "metric",
    "class_name",
    "difficulty",
    "iou_threshold",
    "eligible_ground_truth",
    "ap_r40_percent",
    "metric_availability",
    "evidence_kind",
    "source_status",
)
BENCHMARK_COLUMNS = COMMON + (
    "benchmark_id",
    "experiment_id",
    "frame_id",
    "input_count",
    "output_count",
    "input_sha256",
    "hardware_json",
    "evidence_kind",
    "preprocessing_sha256",
    "stage",
    "clock_domain",
    "relationship",
    "timing_unit",
    "warmup",
    "repetitions",
    "count",
    "mean_ms",
    "median_ms",
    "minimum_ms",
    "maximum_ms",
    "std_ms",
    "p50_ms",
    "p90_ms",
    "p95_ms",
    "p99_ms",
    "percentile_method",
    "std_ddof",
    "frame_status",
    "benchmark_status",
)
RAW_COLUMNS = COMMON + (
    "benchmark_id",
    "frame_id",
    "iteration_index",
    "phase",
    "stage",
    "duration",
    "timing_unit",
    "evidence_kind",
    "observation_status",
    "benchmark_status",
)
NUMERIC = {
    "export_schema_version",
    "source_schema_version",
    "iou_threshold",
    "eligible_ground_truth",
    "ap_r40_percent",
    "input_count",
    "output_count",
    "warmup",
    "repetitions",
    "count",
    "mean_ms",
    "median_ms",
    "minimum_ms",
    "maximum_ms",
    "std_ms",
    "p50_ms",
    "p90_ms",
    "p95_ms",
    "p99_ms",
    "std_ddof",
    "iteration_index",
    "duration",
}


def compact(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def _dangerous(value):
    return value.lstrip().startswith(("=", "+", "-", "@", "＝", "＋", "－", "＠")) or (
        bool(value) and ord(value[0]) < 32
    )


def protect_text(value):
    return "'" + value if value.startswith("'") or _dangerous(value) else value


def restore_text(value):
    if value.startswith("''") or (value.startswith("'") and _dangerous(value[1:])):
        return value[1:]
    return value


def render_csv(columns, rows):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, quoting=csv.QUOTE_ALL, lineterminator="\r\n")
    writer.writerow(columns)
    for row in rows:
        writer.writerow(
            [
                ""
                if row.get(k) is None
                else str(row[k])
                if k in NUMERIC
                else protect_text(str(row[k]))
                for k in columns
            ]
        )
    return stream.getvalue()


def _common(record):
    return {
        k: getattr(record, k) for k in ("source_id", "source_type", "source_schema_version")
    } | {"export_schema_version": EXPORT_VERSION}


def tables(records, raw_timings=False):
    summaries, metrics, benchmarks, raw = [], [], [], []
    for r in records:
        common = _common(r)
        summaries.append(
            common
            | {
                "experiment_id": r.experiment_id,
                "experiment_name": r.experiment_name,
                "config_sha256": r.config_sha256,
                "dataset_root": (r.dataset or {}).get("root"),
                "dataset_split": (r.dataset or {}).get("split"),
                "canonical_split_json": compact(r.canonical_split),
                "frame_ids_json": compact(r.frame_ids),
                "preprocessing_json": compact(r.preprocessing),
                "model_json": compact(r.model),
                "evidence_kind": r.evidence_kind,
                "source_status": r.status,
                "run_manifest_sha256": r.run_manifest_sha256,
                "source_artifacts_json": compact(r.source_artifacts),
                "warnings_json": compact(r.warnings),
                "unavailable_json": compact(r.unavailable),
            }
        )
        if r.source_type == "evaluation":
            for metric in r.payload["metrics"]:
                metrics.append(
                    common
                    | {
                        "experiment_id": r.experiment_id,
                        "run_manifest_sha256": r.run_manifest_sha256,
                        "dataset_split": (r.dataset or {}).get("split"),
                        "evaluation_purpose": r.payload["purpose"],
                        "evaluator_name": r.payload["evaluator"]["name"],
                        "evaluator_version": r.payload["evaluator"]["version"],
                        "protocol": r.payload["evaluator"]["protocol"],
                        "evidence_kind": r.evidence_kind,
                        "source_status": r.status,
                        **metric,
                        "metric_availability": metric["availability"],
                    }
                )
        if r.source_type == "benchmark":
            value = r.payload
            spec = value["specification"]
            bid = (
                Path(value["output_directory"]).name.removesuffix(".benchmark")
                if isinstance(value["output_directory"], str)
                else None
            )
            for frame in value["frames"]:
                for series in frame["series"]:
                    benchmarks.append(
                        common
                        | {
                            "benchmark_id": bid,
                            "experiment_id": r.experiment_id,
                            "frame_id": frame["frame_id"],
                            "input_count": frame["source"]["input_count"],
                            "output_count": next(
                                (
                                    o.get("output_count")
                                    for o in reversed(frame["observations"])
                                    if o.get("status") == "success"
                                ),
                                None,
                            ),
                            "input_sha256": frame["source"]["sha256"],
                            "hardware_json": compact(r.environment),
                            "evidence_kind": r.evidence_kind,
                            "preprocessing_sha256": digest(r.preprocessing),
                            "stage": series["stage"],
                            "clock_domain": series["clock_domain"],
                            "relationship": series["relationship"],
                            "timing_unit": "ms",
                            "warmup": spec["warmup"],
                            "repetitions": spec["repetitions"],
                            **series["summary"],
                            "frame_status": frame["status"],
                            "benchmark_status": r.status,
                        }
                    )
                if raw_timings:
                    for obs in frame["observations"]:
                        for stage, elapsed in obs["durations_ns"].items():
                            raw.append(
                                common
                                | {
                                    "benchmark_id": bid,
                                    "frame_id": frame["frame_id"],
                                    "iteration_index": obs["iteration"],
                                    "phase": obs["phase"],
                                    "stage": stage,
                                    "duration": elapsed,
                                    "timing_unit": "ns",
                                    "evidence_kind": r.evidence_kind,
                                    "observation_status": obs["status"],
                                    "benchmark_status": r.status,
                                }
                            )
            if value["initialization"] is not None:
                series = value["initialization"]
                benchmarks.append(
                    common
                    | {
                        "benchmark_id": bid,
                        "experiment_id": r.experiment_id,
                        "frame_id": None,
                        "hardware_json": compact(r.environment),
                        "evidence_kind": r.evidence_kind,
                        "stage": series["stage"],
                        "clock_domain": series["clock_domain"],
                        "relationship": series["relationship"],
                        "timing_unit": "ms",
                        "warmup": 0,
                        "repetitions": 1,
                        **series["summary"],
                        "frame_status": "lifecycle",
                        "benchmark_status": r.status,
                    }
                )
                if raw_timings:
                    raw.append(
                        common
                        | {
                            "benchmark_id": bid,
                            "frame_id": None,
                            "iteration_index": None,
                            "phase": "initialization",
                            "stage": series["stage"],
                            "duration": series["raw_ns"][0],
                            "timing_unit": "ns",
                            "evidence_kind": r.evidence_kind,
                            "observation_status": "success",
                            "benchmark_status": r.status,
                        }
                    )
    result = {
        "experiment_summary.csv": render_csv(SUMMARY_COLUMNS, summaries),
        "evaluation_metrics.csv": render_csv(EVALUATION_COLUMNS, metrics),
        "benchmark_summaries.csv": render_csv(BENCHMARK_COLUMNS, benchmarks),
    }
    if raw_timings:
        result["raw_timing_samples.csv"] = render_csv(RAW_COLUMNS, raw)
    return result
