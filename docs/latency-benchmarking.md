# Latency benchmarking (RV-033)

`raven benchmark-experiment` measures existing Raven operations with explicit
warmup, repeat counts, integer monotonic intervals, raw observations, statistical
summaries and hardware/source provenance. It consumes RV-030's validated experiment
configuration and RV-031's existing loading/split/artifact/detector primitives.
It does not execute another inference runner, change RV-032 metrics, retrofit old
run artifacts or depend on the open Modal PR #77.

CPU preprocessing works on Apple Silicon with NumPy; statistical removal uses
its existing optional SciPy backend. No torch, CUDA, spconv, OpenPCDet, Modal or
Open3D import is needed for CPU execution. Inference mode supports only the
existing **local CUDA** PointPillars adapter. On macOS it blocks before importing
torch. It never silently substitutes saved predictions, CPU inference or a remote
service. This task initiates no cloud operations.

## API, configuration and CLI

```python
from raven.benchmark.experiment import benchmark_experiment
from raven.benchmark.spec import BenchmarkSpec

result = benchmark_experiment(
    "/your/experiment.json",
    specification=BenchmarkSpec(warmup=5, repetitions=30, include_loading=False),
    benchmark_id="cpu-variant-001",
)
```

The experiment remains schema v1. A separate strict benchmark spec avoids changing
existing runner/config semantics:

```json
{
  "schema_version": 1,
  "mode": "preprocessing",
  "warmup": 5,
  "repetitions": 30,
  "include_loading": false
}
```

All five keys are required in a spec file; duplicate/unknown keys and nonfinite
JSON constants are rejected. Defaults without --spec are the same CPU settings.
`mode` is preprocessing or inference; warmup is integer 0..20, measured repetitions
integer 1..100, and include_loading is boolean. Booleans/floats are not accepted
as counts. A benchmark is bounded to 32 selected frames and 1,000 total iterations
including warmup; this is a guard against accidental large runs, not a runtime cap.
Invalid configuration/stages fail before processing.

```bash
# Validate/copy the template, set your real dataset/output paths first.
raven inspect-experiment /your/preprocessing-experiment.json
raven benchmark-experiment /your/preprocessing-experiment.json

# Explicit CPU spec and a new output identity:
raven benchmark-experiment /your/preprocessing-experiment.json \
  --spec configs/benchmarks/cpu.json --benchmark-id cpu-full-001

# Compatible local NVIDIA CUDA host and genuine model assets required:
raven benchmark-experiment /your/inference-experiment.json \
  --spec configs/benchmarks/local-cuda.json --benchmark-id local-gpu-001
```

`configs/experiments/preprocessing-benchmark.json` is a four-stage example with
placeholder paths. CPU mode reads only the required selected scans: it does not
validate/load declared model assets, predictions, labels or calibration. Existing
schema v1 requires an inference model declaration when preprocessing is nonempty;
that declaration is **not a claim of GPU execution**. For CPU runs, set dataset
and output paths; model placeholders are not accessed. Local CUDA mode must use
real config/checkpoint/revision and the existing adapter's safe loading policy.
Saved-source configurations may benchmark their identity pipeline, but cannot
claim new preprocessing/inference from cached detections. Saved deserialization
is explicitly unmeasured, never relabeled model inference.

Exit codes: **0 success, 1 measurement/persistence failure, 2 invalid config/input
or output collision, 3 unavailable/blocked GPU**. The console prints the typed
result as JSON. A blocked GPU request has no timing samples or initialization
value, `evidence_kind: unavailable`, `cuda_used: false`, and an explicit reason.
It creates no successful benchmark evidence or output directory.

## Measurement architecture and boundaries

`benchmark/timing.py` uses [time.perf_counter_ns](https://docs.python.org/3/library/time.html#time.perf_counter_ns).
Elapsed intervals are integer nanoseconds; wall-clock UTC stamps are audit labels
only. Nanosecond representation does not imply nanosecond clock resolution; actual
clock implementation/resolution is recorded. Negative, noninteger, nonfinite,
boolean or over-24-hour sample intervals are rejected.

`benchmark/experiment.py` orchestrates bounded measurement and provenance;
`benchmark/gpu.py` owns lazy local CUDA checks and synchronization. An optional
`stage_observer(event, position, name, count)` keyword on the existing preprocessing
pipeline brackets actual operation calls without reimplementing any algorithm.
Normal pipeline callers retain their existing result/metadata schema and behavior.
Observer callbacks receive counts, not mutable point arrays. The benchmark's
operation intervals use the same nanosecond timer; it does not reinterpret the
older pipeline's float-second metadata as raw nanosecond measurements.

| Series | Exact boundary / inclusions |
| --- | --- |
| loading, optional | `load_kitti_bin` including validation, read/parse and PointCloud construction/copy; file checksums excluded |
| preprocessing | Complete `run_preprocessing_pipeline`, including prevalidation, its initial independent copy, actual operations, observer instrumentation and pipeline provenance assembly |
| operation/N:name | Actual existing operation call, including its output copies; observer dispatch has small measurement overhead; position separates repeated operations |
| input_integrity_guard, GPU mode | Processed XYZI digest immediately before inference; outside inference_adapter, inside guarded end_to_end |
| inference_adapter | Existing `detect(processed.cloud)` plus completion synchronization; includes host preparation, transfers, model, NMS and validated Detection3D conversion |
| end_to_end | Optional timed loading + full preprocessing + optional input integrity guard and synchronized adapter, plus orchestration/barrier overhead; excludes initialization and post-call validation/persistence |
| detector_initialization | One construction/asset-validation/initialize call with completion synchronization; recorded separately, never included in steady-state repetition statistics |

Nested series overlap: **operation totals are inside preprocessing, which is
inside end_to_end**. Do not add these inclusive totals. End-to-end may exceed
the sum of disjoint series because it includes instrumentation/orchestration.
It is a guarded in-process scope, not application launch, dataset enumeration,
checkpoint preflight or remote service latency.

For CUDA intervals, a [CUDA synchronization barrier](https://docs.pytorch.org/docs/stable/generated/torch.cuda.synchronize.html)
occurs **before clock start**, then another **before clock stop**. End-to-end
excludes earlier GPU work via its own initial barrier and includes adapter barriers.
Initialization is one observed model/session startup after environment/asset
preflight, not a fresh-process cold-start distribution. Its n=1 summary/std=0 is
mechanical, not evidence of stable startup time. CUDA context, filesystem/allocator
caches, other GPU users and process imports may already be warm. First warmup
calls may still exercise lazy kernels/caches; their samples remain separate.

The adapter has no independently exposed boundaries for host preparation, H2D,
forward-only or postprocessing/NMS. These substages are **unavailable**, not zeros
or guessed splits. Full adapter latency is not pure GPU kernel execution time.
Real outputs and model/config/checkpoint/device/runtime provenance are retained
for all successful inference iterations, including warmup. Malformed outputs,
input mutation or a failed repetition fail the benchmark; there is no retry.
Mocked GPU tests are always labeled `synthetic_test`, never NVIDIA performance.

## Input, repetition and ordering policy

All selected IDs must belong to the existing pinned canonical split. Preflight
loads/checks actual scans, hashes and frame identities before measurement and
output reservation. Ground truth/images are irrelevant to preprocessing timing;
accuracy evaluation remains RV-032's separate responsibility.

Frames run in explicit selected order. Each frame completes its warmup sequence,
then its measured repetitions, before the next frame. No variants are reordered,
parallelized or automatically cached by this framework. CPU preprocessing supports
baseline `[]`, range_filter, voxel_downsample, statistical_outlier_removal, noise
and arbitrary validated ordered/repeated combinations from the existing API.

In preload mode, a fresh independent source PointCloud is prepared **outside**
all timers for each iteration. The pipeline's own initial copy and operation
allocations remain **inside** preprocessing timing. With loading enabled, a fresh
scan is loaded inside the optional loading/end-to-end scopes instead. Source
checks, output hashes, prediction serialization and artifact writes happen after
timed work. The explicit pre-inference integrity guard is the exception shown
above. File/content mutation checks run after every iteration.

Each repetition must produce the same processed XYZI hash; mismatches fail rather
than silently mix non-equivalent inputs. Noise retains the existing **per-stage
explicit PCG64 seed**; experiment.seed controls frame selection, not a hidden
replacement for noise seeds. Input count, output count, frame/convention, stage
parameters/counts and processed hashes are recorded without raw point data.

Preflight and integrity hashing warm the filesystem cache. Loading-enabled
results are therefore **not cold-disk throughput**. No page-cache flushing,
CPU-frequency pinning, thermal stabilization, scheduler isolation or GPU exclusivity
is asserted. Keep input/copy/loading/warmup/order settings identical when comparing
variants and record external workload/power/thermal conditions separately.

## Statistical method and failure policy

Raw observations retain iteration index, warmup/measured phase, integer intervals,
operation/count metadata and output identity. Warmup samples **never** contribute
to steady-state summaries. Each frame/stage has its own summary; unlike frames
are not silently pooled.

For N valid measured durations x_i in milliseconds:

- Mean = sum(x_i)/N; median, minimum and maximum use the same sample set.
- Population standard deviation = sqrt(sum((x_i - mean)^2)/N), `ddof=0`.
- p50/p90/p95/p99 use [NumPy percentile](https://numpy.org/doc/stable/reference/generated/numpy.percentile.html)
  `method="linear"` (Hyndman-Fan type 7): sorted index h=(N-1)p, interpolate between
  floor(h) and ceil(h). p50 equals median.

Zero durations are valid clock-resolution observations. Empty/invalid sample
sets produce no summary. A zero-operation pipeline has its actual copy/provenance
cost measured; it does not invent zero-valued operation series. On a repetition
failure, previous raw observations and completed stage intervals are retained,
the failed frame has **no aggregate series**, remaining work stops, and overall
status is failed. Completed earlier frames remain diagnostic records, not evidence
that the full benchmark succeeded. No failed repetition is dropped to improve
reported statistics. A missing final manifest is incomplete evidence.

## Results and safe artifacts

Output is a new sibling directory `output.directory/BENCHMARK_ID.benchmark`;
default benchmark ID is experiment_id. Existing directories, files or symlinks
are rejected before model execution. Existing experiment run artifacts are never
edited, even when benchmarking the same configuration. Exclusive atomic JSON
publication reuses RV-031's publisher:

- started.json: schema, source config identity and audit start timestamp.
- benchmark.json: versioned typed result, complete config/spec, method, raw
  observations, summaries, outputs, failures and environment/model provenance.
- manifest.json: final status/config identity and exact artifact checksums,
  published last. Only status success with verified files is complete evidence.

Result fields include benchmark_schema_version=1; status; evidence_kind
(local_cpu/local_cuda/synthetic_test/unavailable); experiment/config/split/frame
identities; specification and its SHA256; frames/observations/TimingSeries;
initialization; unavailable stages; failures; hardware/environment and UTC audit
stamps. All raw measurements are `ns`; summary field names end in `_ms`.
`TimingSummary`, `TimingSeries`, `FrameBenchmark` and `BenchmarkResult` are typed
Python dataclasses with detached JSON serialization. No full RV-034 export system
or result-import framework is introduced.

Hardware evidence includes OS-reported CPU model/source, architecture, logical
CPU count, Python/NumPy/SciPy/package metadata, selected thread environment
variables, clock metadata, Raven revision/dirty state and relevant benchmark,
preprocessing, loader, config and detector source hashes. Installed GPU package
metadata is not proof of use. Actual GPU/device/config/effective model evidence
comes only from an initialized local adapter; `cuda_used` stays false for CPU,
blocked and synthetic runs. CPU identity is OS-reported, not independent hardware
attestation. A wheel may lack a Git revision; source hashes remain available.

## Genuine local CPU smoke evidence

Observed on **2026-11-26**, Apple M1 Pro / arm64, OS-reported
macOS-26.6.2-arm64-arm-64bit, Python 3.11.12, NumPy 2.4.6, SciPy 1.17.1.
Source revision: **84d321fb6f95c1c2524d503e348c43ebb4035d74**, clean worktree for
all six variants. Later changes add documentation and failure-path handling;
the measured timing boundaries and preprocessing algorithms are unchanged.
Input: training frame **000000**, **115,384 points**, SHA256
`0e09c85e3f6078ecbdd1e706ee9624519f1bd29417437167a9ed7fbe6f54b4b1`.
Defaults: **5 warmups + 30 measured repetitions**, preload mode, fixed order shown
below. Values are genuine measured CPU preprocessing milliseconds, rounded here;
raw observations and full statistics remain in ignored local artifacts.

| Variant | Output points | Mean ms | Median ms | p95 ms |
| --- | ---: | ---: | ---: | ---: |
| Baseline, identity pipeline | 115,384 | 0.076 | 0.073 | 0.090 |
| Range X[0,70], Y[-40,40], Z[-3,1] m | 62,853 | 1.570 | 1.541 | 1.776 |
| Voxel 0.2 m | 22,595 | 52.187 | 51.960 | 54.070 |
| Statistical K=16, ratio=1 | 111,261 | 302.279 | 301.227 | 314.762 |
| Noise std=0.01 m, seed=42 | 115,384 | 4.726 | 4.716 | 4.883 |
| Range → voxel → statistical → noise, above parameters | 12,232 | 61.730 | 61.534 | 63.164 |

A separate baseline loading-enabled check measured mean warm-cache loader time
**0.230 ms**, also 5+30. The optional real-data CPU pytest passed. All repetitions
had stable processed hashes; the original scan remained unchanged. Local evidence:
`outputs/rv033/real-cpu-summary.json`, `real-cpu-log.txt`, `runs/*/benchmark.json`
and their manifests. Generated files and dataset content are not committed.

These are one-frame observational smoke results, **not a representative KITTI
benchmark, an accuracy result or detector inference latency**. Variant order was
fixed, not randomized, and load/thermals/power state were not controlled. Baseline
is an identity-copy overhead, not zero CPU work. A smaller processed cloud does
not automatically mean lower complete detector latency or better accuracy.

A genuine local inference availability check returned **blocked**: local CUDA
inference is unavailable on macOS. No initialization or inference durations were
fabricated; no weights were loaded. CPU-only tests use fake GPU barriers/clocks,
mark synthetic evidence and make no PointPillars performance claim.

## Comparisons, validation and future integration

Compare only successful results with matching source frame/hash, hardware,
software/source version, loading/copy policy, seeds, warmup/repetitions and stage
boundaries. For comparable positive baseline B and candidate C, percentage change
may be calculated as 100*(C-B)/B; B=0, unavailable or mismatched evidence cannot
support it. This framework preserves the needed evidence but does not automatically
claim statistical significance or causal improvement. Cross-frame distributions,
randomized/interleaved variant trials, confidence intervals and controlled thermal
experiments require a broader research design.

Apple Silicon CPU preprocessing and NVIDIA GPU adapter measurements are different
stages/hardware. Do not compare them as equivalent compute benchmarks or infer
end-to-end improvements from preprocessing point counts alone. Accuracy stays
with RV-032. RV-034 can consume the versioned raw/summary JSON without treating
unavailable values as zero. RV-035 should retain input/model/config checksums,
clean source identity, dependency closure, warmup policy, host contention/power
conditions and full artifacts; the recorded subset here is not that full milestone.

Future RV-049 integration may add serialization, SDK round trip, worker model
loading/detection/total and client end-to-end series with **separate clock domains**.
Do not subtract monotonic readings or unsynchronized UTC stamps across hosts.
Queue/upload/download remain unavailable unless independently measured. This PR
implements none of Modal, deployment, asset upload or remote measurement; it does
not copy unfinished PR #77 code or require it to merge.

Standard CI uses synthetic fixtures, no datasets/weights/GPU/service credentials.
Local quality checks: **1274 passed, 4 skipped** (64 additional CPU cases and one
optional KITTI skip), Ruff, dependency and diff checks pass. Existing runner,
evaluator and pipeline regressions remain intact. Optional actual data test:

```bash
RAVEN_KITTI_BENCHMARK_ROOT=/your/kitti python -m pytest -q \
  tests/test_benchmark.py::test_optional_real_kitti_cpu_smoke
```
