"""Validate archived source evidence only: no dataset loading, inference or metric execution."""

import hashlib
import math
from collections import Counter
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from raven.benchmark.experiment import BenchmarkResult, FrameBenchmark, TimingSeries, digest
from raven.benchmark.spec import BenchmarkSpec
from raven.benchmark.timing import TimingSummary, duration_ns, summarize
from raven.evaluation.evaluate import EvaluationResult, MetricResult
from raven.evaluation.inputs import load_completed_run, strict_json
from raven.experiments.config import ExperimentConfig
from raven.experiments.splits import split_membership
from raven.perception.data.frame_manifest import validate_frame_ids
from raven.perception.detection.predictions import detections_from_document


class ExportValidationError(ValueError):
    """Unsupported, corrupt, incomplete or internally inconsistent evidence."""


def require(condition, message):
    if not condition:
        raise ExportValidationError(message)


def sha(value):
    require(
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value),
        "expected lowercase SHA256",
    )
    return value


def integer(value, name):
    require(type(value) is int and value >= 0, f"{name} must be a nonnegative integer")
    return value


def number(value, name, upper=None):
    require(
        type(value) in (int, float)
        and math.isfinite(value)
        and value >= 0
        and (upper is None or value <= upper),
        f"{name} must be finite and in range",
    )
    return value


def keys(value, names, context):
    require(isinstance(value, dict) and set(value) == set(names), f"unsupported {context} fields")


def model_fields(cls):
    return {f.name for f in fields(cls)}


class Snapshot:
    """Bounded snapshots; recorded hashes are checked before any destination is created."""

    def __init__(self):
        self.files = {}
        self.total_bytes = 0

    def read(self, path):
        original = Path(path).absolute()
        require(not original.is_symlink(), "source file cannot be a symlink")
        path = original.resolve()
        if path in self.files:
            return self.files[path][0]
        require(path.is_file(), f"required source unavailable: {path}")
        require(path.stat().st_size <= 128_000_000, "source file exceeds 128 MB limit")
        raw = path.read_bytes()
        checksum = hashlib.sha256(raw).hexdigest()
        self.total_bytes += len(raw)
        require(self.total_bytes <= 256_000_000, "source bundle exceeds 256 MB limit")
        self.files[path] = (raw, checksum)
        return raw

    def json(self, path):
        value = strict_json(self.read(path))
        require(isinstance(value, dict), "source JSON must be an object")
        return value

    def identity(self, path, role, recorded=None):
        path = Path(path).resolve()
        raw = self.read(path)
        checksum = hashlib.sha256(raw).hexdigest()
        if recorded is not None:
            require(checksum == sha(recorded), f"source checksum mismatch: {path.name}")
        return {
            "role": role,
            "path": str(path),
            "sha256": checksum,
            "bytes": len(raw),
            "recorded_sha256": recorded,
        }

    def manifest(self, path, document):
        path = Path(path).resolve()
        records = document.get("artifacts")
        require(isinstance(records, dict) and records, "manifest requires artifacts")
        loaded, identities, seen = {}, [], set()
        for role, record in records.items():
            keys(record, {"file", "sha256"}, "artifact")
            name = record["file"]
            require(
                isinstance(name, str)
                and Path(name).name == name
                and name not in (".", "..", "")
                and name not in seen,
                "unsafe/duplicate artifact filename",
            )
            seen.add(name)
            target = path.parent / name
            require(
                not target.is_symlink() and target.resolve().parent == path.parent,
                "artifact symlink/redirection rejected",
            )
            identities.append(self.identity(target, role, record["sha256"]))
            loaded[role] = (
                self.read(target).decode("utf-8").splitlines()
                if role == "frames"
                else self.json(target)
            )
        return loaded, identities

    def verify_unchanged(self):
        from raven.experiments.artifacts import file_sha256

        for path, (_, checksum) in self.files.items():
            require(
                not path.is_symlink() and file_sha256(path) == checksum,
                f"source changed during export: {path}",
            )


def config_identity(document, checksum, ids):
    config = ExperimentConfig.from_dict(document, base_dir=Path.cwd())
    require(config.config_sha256 == sha(checksum), "configuration SHA256 mismatch")
    require(config.frame_ids == tuple(ids), "configuration frame selection mismatch")
    members, evidence = split_membership(config.dataset.split)
    require(set(ids) <= members, "frame outside declared canonical split")
    return config, evidence


@dataclass(frozen=True)
class Evidence:
    source_id: str
    source_type: str
    source_schema_version: int
    status: str
    evidence_kind: str
    configuration: dict | None
    config_sha256: str | None
    experiment_id: str | None
    experiment_name: str | None
    dataset: dict | None
    canonical_split: dict | None
    frame_ids: tuple[str, ...]
    preprocessing: list | None
    model: dict | None
    environment: dict | None
    run_manifest_sha256: str | None
    source_artifacts: tuple[dict, ...]
    payload: dict
    warnings: tuple[str, ...]
    unavailable: dict

    def to_dict(self):
        return asdict(self)


def _record(
    path,
    typ,
    document,
    config,
    split,
    artifacts,
    kind,
    warnings=(),
    unavailable=None,
    environment=None,
    run_hash=None,
    observed_model=None,
):
    rawsha = artifacts[0]["sha256"]
    identity = document.get("experiment_identity", {})
    ids = config.frame_ids if config else tuple(document.get("frame_ids", []))
    return Evidence(
        f"{typ}-{rawsha}",
        typ,
        1,
        document.get("status", "success"),
        kind,
        config.to_dict() if config else None,
        config.config_sha256 if config else document.get("config_sha256"),
        config.experiment_id if config else identity.get("experiment_id"),
        config.name if config else identity.get("name"),
        {"root": str(config.dataset.root), "split": config.dataset.split} if config else None,
        split,
        ids,
        [dict(s) for s in config.to_dict()["preprocessing"]] if config else None,
        {"declared": config.model.to_dict(), "observed": observed_model} if config else None,
        environment,
        run_hash,
        tuple(artifacts),
        document,
        tuple(warnings),
        unavailable or {},
    )


def _run(path, doc, snapshot):
    require(
        type(doc.get("run_schema_version")) is int and doc["run_schema_version"] == 1,
        "unsupported run schema",
    )
    require(
        doc.get("status") == "success", "run export requires a completed successful run manifest"
    )
    artifacts = [snapshot.identity(path, "run_manifest")]
    loaded, entries = snapshot.manifest(path, doc)
    artifacts.extend(entries)
    # Existing archive loader verifies normalized config, final status, selection and per-frame set.
    run = load_completed_run(path)
    config, split = config_identity(loaded["configuration"], doc["config_sha256"], doc["frame_ids"])
    require(run.provenance.get("split") == split, "run split provenance mismatch")
    require(loaded.get("started", {}).get("status") == "started", "run start marker unavailable")
    require(
        loaded["started"].get("config_sha256") == config.config_sha256, "run start config mismatch"
    )
    kinds = set()
    for row in run.results["frames"]:
        detections = detections_from_document(
            row.get("predictions"), expected_frame_id=row["frame_id"]
        )
        require(
            integer(row.get("prediction_count"), "prediction_count") == len(detections),
            "prediction count mismatch",
        )
        require(
            row.get("class_counts") == dict(Counter(d.label for d in detections)),
            "prediction class counts mismatch",
        )
        integer(row.get("input_count"), "input_count")
        integer(row.get("processed_count"), "processed_count")
        sha(row.get("processed_xyzi_sha256"))
        source = row.get("prediction_source", {})
        if config.model.source == "saved":
            require(
                source.get("mode") == "loaded_saved_predictions",
                "saved source mislabeled as inference",
            )
            kinds.add("saved_predictions")
        else:
            require(source.get("mode") == "live_adapter", "local inference source binding missing")
            metadata = source.get("model_provenance")
            require(isinstance(metadata, dict) and metadata, "model provenance missing")
            kinds.add(
                "synthetic_test"
                if metadata.get("synthetic_test") or metadata.get("not_real_gpu_inference")
                else "declared_local_inference"
            )
    kind = "synthetic_test" if "synthetic_test" in kinds else next(iter(kinds))
    payload = {"status": "success", "manifest": doc, **loaded}
    return _record(
        path,
        "experiment",
        payload,
        config,
        split,
        artifacts,
        kind,
        ["Checksums establish consistency, not independent GPU authenticity."],
        {"evaluation": "not provided", "benchmark": "not provided"},
        run.provenance.get("environment"),
        run.manifest_sha256,
        observed_model=[
            row["prediction_source"].get(
                "model_provenance", row["prediction_source"].get("capture_metadata")
            )
            for row in run.results["frames"]
        ],
    )


def _timing_series(value, expected_samples):
    keys(value, model_fields(TimingSeries), "timing series")
    require(isinstance(value["stage"], str) and value["stage"], "timing stage missing")
    require(value["clock_domain"] == "local_process", "unsupported benchmark clock domain")
    require(isinstance(value["relationship"], str), "timing relationship missing")
    require(isinstance(value["raw_ns"], list), "raw timing samples must be a list")
    for sample in value["raw_ns"]:
        duration_ns(sample)
    require(
        value["raw_ns"] == expected_samples,
        "raw timing series disagrees with measured observations",
    )
    for n in expected_samples:
        duration_ns(n)
    keys(value["summary"], model_fields(TimingSummary), "timing summary")
    expected = summarize(
        expected_samples
    ).to_dict()  # Validation only; no operation/measurement is rerun.
    for key, number_value in expected.items():
        actual = value["summary"][key]
        if key in ("count", "std_ddof"):
            require(type(actual) is int and actual == number_value, "timing count/ddof mismatch")
        elif isinstance(number_value, str):
            require(actual == number_value, "unsupported aggregation method")
        else:
            number(actual, key)
            require(
                math.isclose(actual, number_value, rel_tol=1e-12, abs_tol=1e-12),
                "summary differs from measured raw samples (warmup excluded)",
            )


def _benchmark(path, doc, snapshot):
    require(
        type(doc.get("benchmark_schema_version")) is int and doc["benchmark_schema_version"] == 1,
        "unsupported benchmark schema",
    )
    artifacts = [snapshot.identity(path, "benchmark_manifest")]
    loaded, entries = snapshot.manifest(path, doc)
    artifacts.extend(entries)
    require(
        set(loaded) == {"started.json", "benchmark.json"},
        "benchmark manifest artifact set mismatch",
    )
    value = loaded["benchmark.json"]
    keys(
        value,
        model_fields(BenchmarkResult) | {"methodology", "specification_sha256"},
        "benchmark result",
    )
    require(
        value["benchmark_schema_version"] == 1 and type(value["benchmark_schema_version"]) is int,
        "unsupported result schema",
    )
    require(
        value["status"] in ("success", "failed") and doc.get("status") == value["status"],
        "benchmark final status mismatch",
    )
    keys(value["specification"], model_fields(BenchmarkSpec), "benchmark specification")
    spec = BenchmarkSpec(**value["specification"])
    require(
        digest(value["specification"]) == sha(value["specification_sha256"]),
        "specification digest mismatch",
    )
    experiment = value["experiment"]
    config, split = config_identity(
        experiment["configuration"],
        experiment["config_sha256"],
        experiment["selection"]["selected_frame_ids"],
    )
    spec.validate_size(config.frame_ids)
    require(
        experiment.get("split") == split and doc["config_sha256"] == config.config_sha256,
        "benchmark split/config mismatch",
    )
    require(
        experiment.get("selection") == config.selection_provenance,
        "benchmark selection provenance mismatch",
    )
    require(
        loaded["started.json"].get("config_sha256") == config.config_sha256
        and loaded["started.json"].get("status") == "started",
        "benchmark start marker mismatch",
    )
    require(
        value["evidence_kind"] in ("local_cpu", "local_cuda", "synthetic_test"),
        "unsupported measured evidence kind",
    )
    env = value["environment"]
    require(isinstance(env, dict), "environment missing")
    if value["evidence_kind"] != "synthetic_test":
        require(
            env.get("effective_clock") == "time.perf_counter_ns",
            "synthetic/unavailable clock mislabeled as hardware measurement",
        )
    require(
        isinstance(value["output_directory"], str)
        and Path(value["output_directory"]).name.endswith(".benchmark"),
        "benchmark output identity missing",
    )
    if value["evidence_kind"] == "local_cpu":
        require(
            spec.mode == "preprocessing" and env.get("cuda_used") is False,
            "CPU evidence claims CUDA usage",
        )
    if value["evidence_kind"] == "local_cuda":
        require(spec.mode == "inference", "CUDA source requires inference mode")
        if value["status"] == "success" or value["initialization"] is not None:
            require(
                env.get("cuda_used") is True and isinstance(value["model_provenance"], dict),
                "CUDA provenance unavailable",
            )
            require(
                not value["model_provenance"].get("synthetic_test")
                and not value["model_provenance"].get("not_real_gpu_inference"),
                "synthetic evidence mislabeled CUDA",
            )
        else:
            require(
                env.get("cuda_used") is False and value["model_provenance"] is None,
                "failed initialization has inconsistent GPU evidence",
            )
    require(
        isinstance(value["methodology"], dict)
        and value["methodology"].get("raw_unit") == "ns"
        and value["methodology"].get("summary_unit") == "ms",
        "timing units missing/unsupported",
    )
    require(
        value["methodology"].get("std") == "population, ddof=0"
        and value["methodology"].get("percentiles") == "NumPy linear (Hyndman-Fan type 7)",
        "unsupported statistical methodology",
    )
    sources = experiment.get("inputs")
    require(
        isinstance(sources, list) and len(sources) == len(config.frame_ids),
        "benchmark source identities missing",
    )
    require(
        [s.get("frame_id") for s in sources] == list(config.frame_ids),
        "benchmark source frame order mismatch",
    )
    for source in sources:
        sha(source.get("sha256"))
        integer(source.get("input_count"), "input_count")
        integer(source.get("bytes"), "source bytes")
        require(
            source["coordinate_frame"] == "kitti_velodyne"
            and source["bytes"] == source["input_count"] * 16,
            "input shape/convention mismatch",
        )
    frames = value["frames"]
    require(isinstance(frames, list), "benchmark frames must be a list")
    require(
        [f.get("frame_id") for f in frames] == list(config.frame_ids[: len(frames)]),
        "benchmark frame order/set mismatch",
    )
    if value["status"] == "success":
        require(
            len(frames) == len(config.frame_ids) and not value["failures"],
            "incomplete successful benchmark",
        )
    else:
        require(
            isinstance(value["failures"], list) and value["failures"],
            "failed benchmark must retain diagnostics",
        )
    for frame, source in zip(frames, sources, strict=False):
        keys(frame, model_fields(FrameBenchmark), "frame benchmark")
        require(
            frame["source"] == source and frame["status"] in ("success", "failed"),
            "frame identity/status mismatch",
        )
        require(
            value["status"] != "success" or frame["status"] == "success",
            "failed frame in successful benchmark",
        )
        observations = frame["observations"]
        require(isinstance(observations, list), "observations missing")
        require(len(observations) <= spec.warmup + spec.repetitions, "excess observations")
        if frame["status"] == "success":
            require(len(observations) == spec.warmup + spec.repetitions, "missing iterations")
        measured = {}
        successful = []
        for index, row in enumerate(observations):
            require(
                type(row.get("iteration")) is int and row["iteration"] == index,
                "iteration index mismatch",
            )
            phase = "warmup" if index < spec.warmup else "measured"
            require(row.get("phase") == phase, "warmup/measured phase mismatch")
            require(row.get("status") in ("success", "failed"), "invalid observation status")
            times = row.get("durations_ns")
            require(isinstance(times, dict), "durations missing")
            for timing in times.values():
                duration_ns(timing)
            if row["status"] == "failed":
                require(
                    frame["status"] == "failed"
                    and index == len(observations) - 1
                    and isinstance(row.get("failure"), str),
                    "failed observation not terminal",
                )
                continue
            require(
                integer(row.get("input_count"), "input_count") == source["input_count"],
                "observation input count mismatch",
            )
            integer(row.get("output_count"), "output_count")
            sha(row.get("processed_xyzi_sha256"))
            successful.append(row)
            ops = row.get("operations")
            require(
                isinstance(ops, list) and len(ops) == len(config.preprocessing),
                "operation records missing",
            )
            expected_keys = {"preprocessing", "end_to_end"} | (
                {"loading"} if spec.include_loading else set()
            )
            if spec.mode == "inference":
                expected_keys |= {"inference_adapter", "input_integrity_guard"}
            previous_count = source["input_count"]
            for position, (operation, stage) in enumerate(
                zip(ops, config.preprocessing, strict=True), 1
            ):
                require(
                    operation.get("position") == position
                    and operation.get("name") == stage["name"],
                    "operation order mismatch",
                )
                require(
                    operation.get("input_count") == previous_count,
                    "operation point-count chain mismatch",
                )
                previous_count = integer(operation.get("output_count"), "operation output_count")
                key = f"operation/{position}:{stage['name']}"
                expected_keys.add(key)
                require(
                    times.get(key) == operation.get("duration_ns"), "operation duration mismatch"
                )
            require(
                previous_count == row["output_count"] and set(times) == expected_keys,
                "timing series/count mismatch",
            )
            require(
                times["preprocessing"] >= sum(o["duration_ns"] for o in ops),
                "nested preprocessing interval inconsistent",
            )
            disjoint = sum(
                times.get(name, 0)
                for name in (
                    "loading",
                    "preprocessing",
                    "input_integrity_guard",
                    "inference_adapter",
                )
            )
            require(
                times["end_to_end"] >= disjoint,
                "end-to-end interval shorter than disjoint measured stages",
            )
            if spec.mode == "inference":
                detections_from_document(
                    row.get("predictions"), expected_frame_id=frame["frame_id"]
                )
            else:
                require(
                    row.get("predictions") is None, "CPU observation contains inference predictions"
                )
            if phase == "measured":
                for key, elapsed in times.items():
                    measured.setdefault(key, []).append(elapsed)
        require(
            len({o["processed_xyzi_sha256"] for o in successful}) <= 1,
            "non-equivalent processed inputs",
        )
        require(isinstance(frame["series"], list), "summary series missing")
        if frame["status"] == "failed":
            require(not frame["series"], "failed frame contains steady-state summaries")
        else:
            require(
                len(frame["series"]) == len(measured)
                and {s["stage"] for s in frame["series"]} == set(measured),
                "missing/duplicate summary stages",
            )
            for series in frame["series"]:
                _timing_series(series, measured[series["stage"]])
    if value["initialization"] is not None:
        require(spec.mode == "inference", "CPU benchmark contains model initialization")
        _timing_series(value["initialization"], value["initialization"]["raw_ns"])
        require(
            value["initialization"]["stage"] == "detector_initialization"
            and len(value["initialization"]["raw_ns"]) == 1,
            "initialization must be a single lifecycle observation",
        )
    elif spec.mode == "inference" and value["status"] == "success":
        raise ExportValidationError("successful inference benchmark lacks initialization")
    warnings = [
        "Benchmark environment/execution labels are source declarations, not independently authenticated hardware evidence."
    ]
    if value["evidence_kind"] == "synthetic_test":
        warnings.append("Synthetic timings: do not interpret as hardware performance.")
    if value["status"] != "success":
        warnings.append(
            "Failed benchmark: retained values are diagnostic, not a complete successful measurement."
        )
    return _record(
        path,
        "benchmark",
        value,
        config,
        split,
        artifacts,
        value["evidence_kind"],
        warnings,
        value["unavailable"],
        env,
        observed_model=value["model_provenance"],
    )


def _evaluation(path, doc, snapshot):
    keys(doc, model_fields(EvaluationResult) | {"evaluated_frame_count"}, "evaluation result")
    require(
        type(doc["evaluation_schema_version"]) is int and doc["evaluation_schema_version"] == 1,
        "unsupported evaluation schema",
    )
    require(doc["status"] in ("success", "blocked", "invalid"), "unsupported evaluation status")
    require(isinstance(doc["frame_ids"], list), "evaluation frame IDs must be a list")
    if doc["frame_ids"]:
        validate_frame_ids(doc["frame_ids"])
    require(
        doc["classes"] == ["Car", "Pedestrian", "Cyclist"]
        and doc["difficulties"] == ["easy", "moderate", "hard"],
        "unsupported evaluation classes/difficulties",
    )
    require(
        isinstance(doc["warnings"], list) and all(isinstance(s, str) for s in doc["warnings"]),
        "invalid warnings",
    )
    require(
        isinstance(doc["failures"], list) and all(isinstance(s, str) for s in doc["failures"]),
        "invalid failures",
    )
    evaluator = doc["evaluator"]
    require(
        isinstance(evaluator, dict)
        and evaluator.get("protocol") == "AP_R40"
        and isinstance(evaluator.get("name"), str)
        and isinstance(evaluator.get("version"), str),
        "evaluator identity/protocol unavailable",
    )
    require(
        isinstance(doc["provenance"], dict) and isinstance(doc["split"], dict),
        "evaluation provenance missing",
    )
    artifacts = [snapshot.identity(path, "evaluation_result")]
    config, split, run_record = None, None, None
    if doc["run_manifest_sha256"] is not None:
        manifest_path = doc["experiment_identity"].get("manifest_path")
        require(isinstance(manifest_path, str), "referenced run location missing")
        artifacts.append(
            snapshot.identity(manifest_path, "referenced_run_manifest", doc["run_manifest_sha256"])
        )
        run_doc = snapshot.json(manifest_path)
        run_record = _run(manifest_path, run_doc, snapshot)
        artifacts.extend(run_record.source_artifacts)
        config, split = config_identity(
            run_record.configuration, doc["config_sha256"], doc["frame_ids"]
        )
        require(
            doc["experiment_identity"].get("experiment_id") == config.experiment_id
            and doc["experiment_identity"].get("name") == config.name,
            "evaluation experiment identity mismatch",
        )
    else:
        require(
            doc["status"] != "success" and doc["config_sha256"] is None,
            "successful evaluation requires verified source run identity",
        )
    metrics = doc["metrics"]
    require(isinstance(metrics, list), "evaluation metrics missing")
    if doc["status"] != "success":
        require(
            doc["evaluated_frame_count"] == 0
            and type(doc["evaluated_frame_count"]) is int
            and not metrics
            and not doc["diagnostics"]
            and doc["failures"],
            "blocked/invalid evaluation cannot contain measurements",
        )
    else:
        require(
            config is not None and not doc["failures"],
            "successful evaluation lacks verified completion",
        )
        require(
            doc["evaluated_frame_count"] == len(config.frame_ids)
            and type(doc["evaluated_frame_count"]) is int,
            "evaluated frame count mismatch",
        )
        expected_split = {
            **split,
            "dataset_root": str(config.dataset.root),
            "split": config.dataset.split,
        }
        require(doc["split"] == expected_split, "evaluation dataset/split identity mismatch")
        members, _ = split_membership(config.dataset.split)
        purpose = (
            "training_smoke"
            if config.dataset.split == "train"
            else ("validation_full" if set(config.frame_ids) == members else "validation_subset")
        )
        require(
            config.dataset.split != "test" and doc["purpose"] == purpose,
            "evaluation purpose/split mismatch",
        )
        require(
            doc["provenance"].get("artifact_checksums")
            == load_completed_run(doc["experiment_identity"]["manifest_path"]).artifact_identities,
            "evaluation source artifact identities mismatch",
        )
        thresholds = evaluator.get("parameters", {}).get("iou_thresholds")
        require(
            thresholds == {"Car": 0.7, "Pedestrian": 0.5, "Cyclist": 0.5},
            "unsupported source IoU thresholds",
        )
        seen = set()
        for metric in metrics:
            keys(metric, model_fields(MetricResult), "evaluation metric")
            key = (metric["metric"], metric["class_name"], metric["difficulty"])
            require(
                metric["metric"] in ("3d", "bev")
                and metric["class_name"] in doc["classes"]
                and metric["difficulty"] in doc["difficulties"]
                and key not in seen,
                "invalid/duplicate metric identity",
            )
            seen.add(key)
            require(
                number(metric["iou_threshold"], "iou_threshold", 1)
                == thresholds[metric["class_name"]],
                "metric IoU threshold mismatch",
            )
            eligible = integer(metric["eligible_ground_truth"], "eligible_ground_truth")
            if eligible == 0:
                require(
                    metric["availability"] == "no_eligible_ground_truth"
                    and metric["ap_r40_percent"] is None,
                    "undefined AP must remain null",
                )
            else:
                require(metric["availability"] == "available", "eligible metric unavailable")
                number(metric["ap_r40_percent"], "AP_R40 percent", 100)
        expected = {
            (m, c, d)
            for m in {x["metric"] for x in metrics}
            for c in doc["classes"]
            for d in doc["difficulties"]
        }
        require(
            seen == expected and any(m["metric"] == "3d" for m in metrics),
            "incomplete evaluation metric grid",
        )
        diagnostics = doc["diagnostics"]
        require(isinstance(diagnostics, list), "diagnostics missing")
        diagnostic_keys = set()
        for row in diagnostics:
            key = (row["metric"], row["class_name"], row["difficulty"])
            full = (*key, row["frame_id"])
            require(
                key in seen and row["frame_id"] in config.frame_ids and full not in diagnostic_keys,
                "invalid/duplicate diagnostic identity",
            )
            diagnostic_keys.add(full)
            for name in ("tp", "fp", "fn"):
                integer(row[name], name)
            number(row["score_threshold"], "score threshold", 1)
        require(
            len(diagnostic_keys) == len(seen) * len(config.frame_ids), "incomplete diagnostic grid"
        )
        for metric in metrics:
            eligible = sum(
                r["tp"] + r["fn"]
                for r in diagnostics
                if (r["metric"], r["class_name"], r["difficulty"])
                == (metric["metric"], metric["class_name"], metric["difficulty"])
            )
            require(
                eligible <= metric["eligible_ground_truth"],
                "diagnostic TP/FN exceed eligible ground truth",
            )
    unavailable = {"benchmark": "not provided"}
    if doc["status"] != "success":
        unavailable["evaluation"] = "; ".join(doc["failures"])
    if config is None:
        unavailable.update(
            configuration="source run unavailable",
            model="not established",
            dataset="not established",
        )
    warnings = [
        *doc["warnings"],
        "Export validates source consistency, not accuracy recomputation or GPU authenticity.",
    ]
    return _record(
        path,
        "evaluation",
        doc,
        config,
        split or doc["split"],
        artifacts,
        run_record.evidence_kind if run_record else "unavailable",
        warnings,
        unavailable,
        doc["provenance"].get("execution_environment"),
        doc["run_manifest_sha256"],
        observed_model=run_record.model["observed"] if run_record else None,
    )


def _benchmark_outcome(path, doc, snapshot):
    keys(
        doc,
        model_fields(BenchmarkResult) | {"methodology", "specification_sha256"},
        "benchmark outcome",
    )
    require(
        type(doc["benchmark_schema_version"]) is int and doc["benchmark_schema_version"] == 1,
        "unsupported benchmark outcome schema",
    )
    require(
        doc["status"] in ("blocked", "invalid")
        and doc["evidence_kind"] in ("unavailable", "synthetic_test"),
        "standalone benchmark sources must be unavailable outcomes",
    )
    require(
        doc["frames"] == []
        and doc["initialization"] is None
        and doc["model_provenance"] is None
        and doc["output_directory"] is None,
        "unavailable outcome contains measurements",
    )
    require(
        isinstance(doc["failures"], list) and doc["failures"],
        "unavailable outcome needs diagnostics",
    )
    keys(doc["specification"], model_fields(BenchmarkSpec), "benchmark specification")
    BenchmarkSpec(**doc["specification"])
    require(
        digest(doc["specification"]) == sha(doc["specification_sha256"]),
        "specification checksum mismatch",
    )
    config = ExperimentConfig.from_dict(doc["experiment"]["configuration"], base_dir=Path.cwd())
    require(
        config.config_sha256 == sha(doc["experiment"]["config_sha256"]),
        "outcome configuration checksum mismatch",
    )
    require(
        doc["experiment"].get("selection") == config.selection_provenance,
        "outcome frame selection mismatch",
    )
    require(
        isinstance(doc["environment"], dict) and doc["environment"].get("cuda_used") is False,
        "unavailable outcome claims GPU use",
    )
    require(isinstance(doc["unavailable"], dict), "unavailable states missing")
    return _record(
        path,
        "benchmark",
        doc,
        config,
        doc["experiment"].get("split"),
        [snapshot.identity(path, "benchmark_outcome")],
        doc["evidence_kind"],
        [
            "Standalone blocked/invalid benchmark outcome; no final measurement manifest or latency values."
        ],
        doc["unavailable"],
        doc["environment"],
    )


def load_evidence(path, snapshot):
    try:
        doc = snapshot.json(path)
        flags = [
            name
            for name in (
                "run_schema_version",
                "evaluation_schema_version",
                "benchmark_schema_version",
            )
            if name in doc
        ]
        require(len(flags) == 1, "ambiguous or unsupported source schema metadata")
        if flags[0] == "run_schema_version":
            return _run(path, doc, snapshot)
        if flags[0] == "evaluation_schema_version":
            return _evaluation(path, doc, snapshot)
        return (
            _benchmark(path, doc, snapshot)
            if "artifacts" in doc
            else _benchmark_outcome(path, doc, snapshot)
        )
    except (KeyError, TypeError, AttributeError, IndexError, OverflowError, RecursionError) as exc:
        raise ExportValidationError(
            f"malformed source evidence: {type(exc).__name__}: {exc}"
        ) from exc
