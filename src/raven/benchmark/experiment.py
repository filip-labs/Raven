"""Bounded benchmark orchestration using existing config, loader, pipeline and detector.

This measures operations; it does not execute/persist a second detection experiment.
"""

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from raven.benchmark.environment import environment
from raven.benchmark.gpu import LocalCudaBackend
from raven.benchmark.spec import BenchmarkSpec, benchmark_directory
from raven.benchmark.timing import Timer, summarize
from raven.core.point_cloud import PointCloud
from raven.experiments.artifacts import file_sha256, write_json_exclusive
from raven.experiments.config import ExperimentConfig, load_experiment_config
from raven.experiments.splits import split_membership
from raven.perception.data.kitti_loader import load_kitti_bin
from raven.perception.detection.predictions import detections_to_document
from raven.perception.preprocessing.pipeline import run_preprocessing_pipeline


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def cloud_digest(cloud):
    return hashlib.sha256(cloud.points.astype("<f4").tobytes(order="C")).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class TimingSeries:
    stage: str
    raw_ns: tuple[int, ...]
    summary: dict
    clock_domain: str = "local_process"
    relationship: str = "inclusive; do not sum with nested series"


@dataclass(frozen=True)
class FrameBenchmark:
    frame_id: str
    status: str
    source: dict
    observations: tuple[dict, ...]
    series: tuple[TimingSeries, ...]


@dataclass(frozen=True)
class BenchmarkResult:
    status: str
    evidence_kind: str
    experiment: dict
    specification: dict
    environment: dict
    frames: tuple[FrameBenchmark, ...]
    initialization: TimingSeries | None
    model_provenance: dict | None
    unavailable: dict
    failures: tuple[dict, ...]
    started_at_utc: str
    ended_at_utc: str
    output_directory: str | None = None
    benchmark_schema_version: int = 1

    def to_dict(self):
        return {
            **asdict(self),
            "methodology": dict(METHODOLOGY),
            "specification_sha256": digest(self.specification),
        }


def _series(stage, values):
    return TimingSeries(stage, tuple(values), summarize(values).to_dict())


class StageObserver:
    def __init__(self, timer):
        self.timer = timer
        self.pending = None
        self.records = []

    def __call__(self, event, position, name, count):
        if event == "start":
            if self.pending is not None:
                raise ValueError("overlapping stage observer events")
            self.pending = (position, name, count, self.timer.start())
        elif event == "finish":
            if self.pending is None or self.pending[:2] != (position, name):
                raise ValueError("mismatched stage observer event")
            p, n, before, start = self.pending
            elapsed = self.timer.stop(start)
            self.records.append(
                {
                    "position": p,
                    "name": n,
                    "input_count": before,
                    "output_count": count,
                    "duration_ns": elapsed,
                }
            )
            self.pending = None
        else:
            raise ValueError("unsupported stage observer event")


def _preflight(config, spec, benchmark_id):
    # Reuse schema/path/split/input primitives without local GPU checks or saved-IO in CPU mode.
    checked = ExperimentConfig.from_dict(config.to_dict(), base_dir=Path.cwd())
    if checked != config:
        raise ValueError("experiment changed during benchmark preflight")
    spec.validate_size(config.frame_ids)
    directory = benchmark_directory(config, benchmark_id)
    members, split = split_membership(config.dataset.split)
    if not set(config.frame_ids) <= members:
        raise ValueError("benchmark frame is outside declared canonical split")
    identities = []
    for frame_id in config.frame_ids:
        path = config.dataset.scan_directory / f"{frame_id}.bin"
        checksum = file_sha256(path)
        cloud = load_kitti_bin(path)
        if (
            cloud.frame_id != frame_id
            or cloud_digest(cloud) != checksum
            or file_sha256(path) != checksum
        ):
            raise ValueError("scan changed or frame/XYZI identity mismatch during preflight")
        identities.append(
            {
                "frame_id": frame_id,
                "path": str(path.resolve()),
                "sha256": checksum,
                "bytes": path.stat().st_size,
                "input_count": cloud.count,
                "coordinate_frame": cloud.coordinate_frame,
            }
        )
    return directory, split, identities


def _reserve(directory):
    base = directory.parent
    if directory.resolve() != directory or base.resolve() != base:
        raise ValueError("benchmark output redirected through a symlink")
    base.mkdir(parents=True, exist_ok=True)
    directory.mkdir(mode=0o700)
    if directory.resolve() != directory or base.resolve() != base:
        raise ValueError("benchmark output changed during reservation")


def _frame(config, spec, identity, timer, backend):
    source = load_kitti_bin(identity["path"])
    if cloud_digest(source) != identity["sha256"]:
        raise ValueError("source cloud changed since preflight")
    observations = []
    collected = defaultdict(list)
    for index in range(spec.warmup + spec.repetitions):
        warmup = index < spec.warmup
        observer, times = StageObserver(timer), {}
        try:
            # Fresh independent source preparation excluded for preload mode; pipeline copies remain timed.
            cloud = (
                None
                if spec.include_loading
                else PointCloud(source.points, source.frame_id, source.coordinate_frame)
            )
            if backend is not None:
                backend.synchronize()  # Exclude earlier GPU work from this end-to-end interval.
            start = timer.start()
            if spec.include_loading:
                cloud, times["loading"] = timer.call(lambda: load_kitti_bin(identity["path"]))
            processed, times["preprocessing"] = timer.call(
                lambda cloud=cloud, observer=observer: run_preprocessing_pipeline(
                    cloud, config.preprocessing, stage_observer=observer
                )
            )
            detections = None
            if backend is not None:
                before_inference, times["input_integrity_guard"] = timer.call(
                    lambda processed=processed: cloud_digest(processed.cloud)
                )
                detections, times["inference_adapter"] = timer.call(
                    lambda processed=processed: backend.detect(processed.cloud), backend.synchronize
                )
            times["end_to_end"] = timer.stop(start)
            # Post-call hashing, mutation checks and serialization are outside measured intervals.
            if (
                cloud_digest(cloud) != identity["sha256"]
                or file_sha256(identity["path"]) != identity["sha256"]
            ):
                raise ValueError("source/input mutation detected during benchmark")
            output_hash = cloud_digest(processed.cloud)
            if observations and output_hash != observations[0]["processed_xyzi_sha256"]:
                raise ValueError("non-equivalent processed inputs across repetitions")
            if observer.pending is not None or len(observer.records) != len(config.preprocessing):
                raise ValueError("incomplete operation timing observations")
            expected = [(i, stage["name"]) for i, stage in enumerate(config.preprocessing, 1)]
            if [(s["position"], s["name"]) for s in observer.records] != expected:
                raise ValueError("operation timing order mismatch")
            if (
                processed.cloud.frame_id != cloud.frame_id
                or processed.cloud.coordinate_frame != cloud.coordinate_frame
            ):
                raise ValueError("preprocessing changed source metadata")
            document = None
            if backend is not None:
                if output_hash != before_inference:
                    raise ValueError("detector mutated processed input")
                document = detections_to_document(detections, frame_id=cloud.frame_id)
            for s in observer.records:
                times[f"operation/{s['position']}:{s['name']}"] = s["duration_ns"]
            record = {
                "iteration": index,
                "phase": "warmup" if warmup else "measured",
                "status": "success",
                "input_count": cloud.count,
                "output_count": processed.cloud.count,
                "processed_xyzi_sha256": output_hash,
                "durations_ns": times,
                "operations": observer.records,
                "predictions": document,
            }
            observations.append(record)
            if not warmup:
                for name, value in times.items():
                    collected[name].append(value)
        except Exception as exc:  # noqa: BLE001 -- Retain failed repetition diagnostics, never aggregate them.
            observations.append(
                {
                    "iteration": index,
                    "phase": "warmup" if warmup else "measured",
                    "status": "failed",
                    "durations_ns": times,
                    "operations": observer.records,
                    "failure": f"{type(exc).__name__}: {exc}",
                }
            )
            return FrameBenchmark(source.frame_id, "failed", identity, tuple(observations), ())
    if len({v["processed_xyzi_sha256"] for v in observations}) != 1:
        raise ValueError("non-equivalent processed inputs across repetitions")
    return FrameBenchmark(
        source.frame_id,
        "success",
        identity,
        tuple(observations),
        tuple(_series(k, v) for k, v in collected.items()),
    )


def benchmark_experiment(
    configuration, *, specification=None, benchmark_id=None, _clock=None, _backend_factory=None
) -> BenchmarkResult:
    """Exclusive new benchmark artifacts. Test clocks/backends ALWAYS mark evidence synthetic."""
    config = (
        configuration
        if isinstance(configuration, ExperimentConfig)
        else load_experiment_config(configuration)
    )
    spec = BenchmarkSpec() if specification is None else specification
    if not isinstance(spec, BenchmarkSpec):
        raise TypeError("expected validated BenchmarkSpec")
    name = config.experiment_id if benchmark_id is None else benchmark_id
    timer = Timer(_clock)
    started = now()
    synthetic = _clock is not None or _backend_factory is not None
    kind = (
        "synthetic_test"
        if synthetic
        else ("local_cpu" if spec.mode == "preprocessing" else "local_cuda")
    )
    experiment = {
        "config_sha256": config.config_sha256,
        "configuration": config.to_dict(),
        "selection": config.selection_provenance,
    }
    env = environment()
    env["effective_clock"] = (
        "synthetic injected clock" if _clock is not None else "time.perf_counter_ns"
    )
    unavailable = {
        "gpu_substages": "preparation/H2D/forward/NMS boundaries not exposed by adapter",
        "remote_stages": "not implemented; no remote calls or cross-host clock subtraction",
        "saved_prediction_io": "not measured; saved results are never model inference",
    }
    if spec.mode == "preprocessing":
        unavailable["inference_adapter"] = "not requested; CPU preprocessing only"
        unavailable["detector_initialization"] = "not requested"
    if not spec.include_loading:
        unavailable["loading"] = "preloaded input; loading/copying policy recorded in methodology"
    frames, failures, initialization, model, backend = [], [], None, None, None
    directory = None
    status = "success"
    try:
        directory, split, identities = _preflight(config, spec, name)
        experiment.update(split=split, inputs=identities)
    except (OSError, ValueError, TypeError) as exc:
        status = "invalid"
        failures.append({"phase": "preflight", "message": str(exc)})
    if not failures and spec.mode == "inference":
        try:
            backend = (_backend_factory or LocalCudaBackend)(config)
        except (OSError, RuntimeError, ImportError) as exc:
            status = "blocked"
            failures.append({"phase": "cuda_availability", "message": str(exc)})
            unavailable["inference_adapter"] = str(exc)
        except (ValueError, TypeError) as exc:
            status = "invalid"
            failures.append({"phase": "model_configuration", "message": str(exc)})
    reserved = False
    if not failures:
        try:
            _reserve(directory)
            reserved = True
            write_json_exclusive(
                directory / "started.json",
                {
                    "benchmark_schema_version": 1,
                    "status": "started",
                    "started_at_utc": started,
                    "config_sha256": config.config_sha256,
                },
            )
            if backend is not None:
                _, elapsed = timer.call(backend.initialize, backend.synchronize)
                initialization = _series("detector_initialization", [elapsed])
                model = backend.provenance
            for identity in identities:
                frame = _frame(config, spec, identity, timer, backend)
                frames.append(frame)
                if frame.status != "success":
                    status = "failed"
                    failures.append(
                        {
                            "phase": "repetition",
                            "frame_id": frame.frame_id,
                            "message": frame.observations[-1]["failure"],
                        }
                    )
                    break
        except Exception as exc:  # noqa: BLE001 -- External operation failure invalidates the benchmark, no retry.
            status = "failed"
            failures.append({"phase": "measurement", "message": f"{type(exc).__name__}: {exc}"})
        finally:
            if backend is not None:
                try:
                    backend.close()
                except Exception as exc:  # noqa: BLE001 -- Retain cleanup failures explicitly.
                    status = "failed"
                    failures.append({"phase": "cleanup", "message": str(exc)})
    if not synthetic and status in ("blocked", "invalid"):
        kind = "unavailable"
    env["cuda_used"] = bool(
        initialization is not None and not synthetic and spec.mode == "inference"
    )
    result = BenchmarkResult(
        status,
        kind,
        experiment,
        spec.to_dict(),
        env,
        tuple(frames),
        initialization,
        model,
        unavailable,
        tuple(failures),
        started,
        now(),
        str(directory) if reserved else None,
    )
    if reserved:
        try:
            _persist(directory, result)
        except (OSError, ValueError, TypeError) as exc:
            result = BenchmarkResult(
                "failed",
                kind,
                experiment,
                spec.to_dict(),
                env,
                tuple(frames),
                initialization,
                model,
                unavailable,
                (*failures, {"phase": "persistence", "message": str(exc)}),
                started,
                now(),
                str(directory),
            )
    return result


METHODOLOGY = {
    "clock": "time.perf_counter_ns",
    "raw_unit": "ns",
    "summary_unit": "ms",
    "std": "population, ddof=0",
    "percentiles": "NumPy linear (Hyndman-Fan type 7)",
    "order": "configured frame order; per frame warmup then measured repetitions; stop on failure",
    "input_policy": "same scan; fresh isolated source copy per preload iteration outside timers; pipeline's own copy inside",
    "loading_policy": "optional loader+parse+PointCloud copy; checksums/preflight outside; OS page cache uncontrolled",
    "nested_stages": "operation series are inside preprocessing, inside end_to_end; never add nested totals",
    "end_to_end": "optional loading + pipeline + optional synchronized adapter and pre-inference input integrity guard; excludes post-validation/hashes/serialization/persistence and initialization",
    "initialization": "single observed construction+initialize with GPU synchronization; preflight/context/cache effects not process cold start",
    "synchronization": "GPU barrier before clock start, completion barrier before clock stop; E2E includes adapter barriers",
    "warmup": "observations retained but excluded from every steady-state summary",
    "noise": "existing per-stage explicit PCG64 seeds reused; output digest equality verified",
    "failure_policy": "abort on first failure; partial successful frames are diagnostic, not complete benchmark evidence",
}


def _persist(directory, result):
    document = result.to_dict()
    document["methodology"] = METHODOLOGY
    document["specification_sha256"] = digest(result.specification)
    write_json_exclusive(directory / "benchmark.json", document)
    write_json_exclusive(
        directory / "manifest.json",
        {
            "benchmark_schema_version": 1,
            "status": result.status,
            "config_sha256": result.experiment["config_sha256"],
            "artifacts": {
                name: {"file": name, "sha256": file_sha256(directory / name)}
                for name in ("started.json", "benchmark.json")
            },
        },
    )
