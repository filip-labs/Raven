# Configurable preprocessing pipeline (RV-017)

```python
from raven.perception.preprocessing.pipeline import run_preprocessing_pipeline

stages = [
    {'name': 'range_filter', 'x_range': [0, 70], 'y_range': [-40, 40], 'z_range': [-3, 1]},
    {'name': 'voxel_downsample', 'voxel_size': 0.2},
    {'name': 'statistical_outlier_removal', 'nb_neighbors': 16, 'std_ratio': 1.0},
    {'name': 'noise', 'position_std': 0.01, 'seed': 42},
]
result = run_preprocessing_pipeline(cloud, stages)
processed_cloud = result.cloud
metadata = result.to_metadata_dict()  # JSON-compatible; contains no point data
```

The pipeline calls the existing tested operations; it does not reimplement their
algorithms. It adds no inference, GPU dependencies, file writes or CLI command.
The caller remains responsible for selecting meaningful preprocessing settings.

## Configuration contract

`run_preprocessing_pipeline(cloud, stages) -> PipelineResult` accepts an ordered,
reusable sequence (normally a list or tuple) of mappings. Each mapping has `name`
and the operation's explicit parameters **at the same level**:

| Canonical name | Required parameters | Existing semantics |
| --- | --- | --- |
| `range_filter` | `x_range`, `y_range`, `z_range` | Inclusive XYZ bounds; lower < upper |
| `voxel_downsample` | `voxel_size` | Positive scalar or three per-axis sizes; zero-origin floor cells, mean XYZI |
| `statistical_outlier_removal` | `nb_neighbors`, `std_ratio` | Positive integer K, finite nonnegative ratio; inclusive population-statistics threshold |
| `noise` | `position_std`, `seed` | Finite nonnegative scalar std, explicit nonnegative integer seed; Gaussian XYZ only |

Names are case-sensitive, without aliases. There are no hidden parameter
defaults or nested `params` objects. Unknown names, missing parameters and extra
keys are errors. Every noise occurrence needs its own explicit seed. Repeated
operations are valid and receive their own configuration and metadata record.
Their order is exactly the order supplied; no sorting, cropping or inferred
stage is added. Each stage can be used on its own. `stages=[]` returns an independent
identity PointCloud and an empty record tuple.

Range pairs and per-axis voxel sizes accept lists, tuples or NumPy arrays;
NumPy scalar parameters follow the standalone validators. Iterators/generators
are rejected in both the outer configuration and sequence-valued parameters to
avoid consuming the caller's configuration. Normalized parameter snapshots use
Python numbers and immutable tuples. Input mappings/arrays are not changed.

All names, key sets and parameter values are validated before the first operation
runs, using the existing operations' validators. For example, a missing seed in
stage 4 prevents stages 1–3 from running. The rules remain those documented in
[range filtering](range-filtering.md), [voxel downsampling](voxel-downsampling.md),
[statistical filtering](statistical-outlier-removal.md) and [noise](controlled-noise.md).

## Results and provenance

`PipelineResult` is a frozen dataclass with:

- `cloud`: processed PointCloud with independent, read-only float32 XYZI storage;
- `stages`: immutable tuple of frozen `StageExecution` records;
- `duration_seconds`: separate total preprocessing duration;
- `reproducibility`: read-only mapping of Python and NumPy versions, plus
  `noise_rng='numpy.random.Generator(PCG64)'` when noise is configured and the
  installed SciPy version when statistical filtering is configured. SciPy may
  be null if unavailable and the configured statistical stages only process
  empty/singleton inputs, which do not need that backend.

Each stage record contains:

| Field | Meaning |
| --- | --- |
| `position` | 1-based position in the supplied sequence |
| `name` | Canonical operation name |
| `parameters` | Detached, read-only normalized explicit parameters, including noise seed |
| `input_count`, `output_count` | Counts immediately before and after this occurrence |
| `points_removed` | Input minus output count for range/voxel/statistical stages; null for noise |
| `duration_seconds` | Observed operation-call duration from monotonic `perf_counter` |
| `frame_id`, `coordinate_frame` | Preserved source metadata |

For voxel downsampling, `points_removed` measures count reduction from
aggregation; it does not mean those returns were identified as invalid.
Noise changes XYZ, never point count. Range/statistical filters preserve retained
rows and intensity, voxel downsampling averages XYZI and sorts cells, and noise
preserves intensity and current row correspondence. Frame metadata remains
unchanged across all stages; there is no implicit coordinate transform.

`to_metadata_dict()` returns fresh mutable JSON-compatible containers, with
parameter tuples exported as lists and no point arrays. Its top level contains
`input_count`, `output_count`, `frame_id`, `coordinate_frame`, `duration_seconds`,
`reproducibility`, and the ordered `stages` list. Editing this export cannot
change the configuration or the result's stored provenance. For example:

```python
import json
text = json.dumps(result.to_metadata_dict(), allow_nan=False, indent=2)
```

Stage durations include the complete function call, validation, allocation and
any first-use optional-backend import. Total duration starts before configuration
validation and includes the initial independent cloud copy, all stages, record
construction and version lookup. It excludes dataset I/O, metadata export and
inference. Total need not equal the sum of stage durations. Durations vary across
runs and are observations, not formal benchmarks or latency guarantees.

## Errors, empty clouds and dependencies

Invalid configuration raises `PipelineConfigurationError` before any stage runs.
Stage-specific errors identify the 1-based position and name, preserving the
original exception as `__cause__`. Data-dependent failures cannot all be decided
by configuration validation: unresolvable voxel indices, overflowing noisy XYZ,
a nonfinite statistical threshold or a missing SciPy backend can fail at runtime.
These raise `PipelineExecutionError` with the failing stage's position/name and
underlying message. No successful partial result is returned. The original cloud
and configuration remain unchanged on failure; execution has no external writes.

Empty clouds flow through every stage using its existing rules. Statistical
filtering copies empty/singleton clouds and caps K at N-1 for larger inputs.
Zero noise copies its input. Metadata records zero counts normally. No stage is
silently skipped, even when the input is empty.

The standard installation remains NumPy-only. Install `.[preprocessing]` for
SciPy when statistical filtering will process two or more points. Merely
importing the pipeline or executing the other operations does not import SciPy.
No Open3D, PyTorch, CUDA, OpenPCDet, spconv or cumm is imported. Development tests
use the existing `.[dev]` extra, which includes SciPy.

## Determinism and future experiment integration

Processed points are deterministic for the same input/order, configurations,
seeds and environment; timing metadata is intentionally variable. Each noise
occurrence restarts its local PCG64 stream. Reusing a seed across repeated noise
stages creates correlated perturbations; explicit different seeds are appropriate
when independent experimental repetitions are intended. Float32 storage and
platform/library numerical differences retain the limits of each operation.
The pipeline does not promise cross-version/platform bitwise identity.

Stage order matters. Noise can move points across crop/voxel boundaries or
change neighbor distances; voxel means change density and intensity. A reduced
point count is neither an accuracy measurement nor proof of lower total inference
latency. Filtering may discard valid sparse structures. Gaussian Cartesian noise
is a simplified error model, not a physical LiDAR/weather simulator.

A future experiment runner can pass `result.cloud` to the existing ObjectDetector
interface and save `result.to_metadata_dict()` alongside measurements. Callers
should also record input hashes, Raven commit, full environment, experimental
split and separate inference/evaluation timing. This change does not invoke
PointPillars, choose a pipeline or establish any detector accuracy result.

## Read-only KITTI demonstration

Observed on 2026-11-14, frame 000000, 115,384 input points, SHA256
`0e09c85e3f6078ecbdd1e706ee9624519f1bd29417437167a9ed7fbe6f54b4b1`;
Python 3.11.12, NumPy 2.4.6, SciPy 1.17.1, macOS 26.6.2 arm64.
Use your own licensed scan with `load_kitti_bin('/path/to/000000.bin')` and the
API example above. Configurations were:

- A: `[]` (identity baseline).
- B: range only: X [0,70], Y [-40,40], Z [-3,1] meters.
- C: voxel only: scalar 0.2 m.
- D: B followed by C.
- E: D, then statistical filtering K=16 / ratio=1.0, then noise std=0.01 m / seed=42.

| Run | Output count | Observed stage durations, seconds (in order) | Total seconds |
| --- | --- | --- | --- |
| A | 115,384 | none | 0.000389 |
| B | 62,853 | range 0.001840 | 0.002031 |
| C | 22,595 | voxel 0.052606 | 0.052734 |
| D | 12,565 | range 0.001505; voxel 0.026447 | 0.028107 |
| E | 12,232 | range 0.001447; voxel 0.026106; statistical 0.185791; noise 0.000583 | 0.216619 |

E's intermediate counts were 115,384 → 62,853 → 12,565 → 12,232 → 12,232.
All five processed outputs were bitwise identical when repeated, ignoring timing
metadata. The input array and file SHA256 were unchanged. Times are single-call
observations and include any first-use SciPy spatial import, not a controlled
performance comparison. No inference, generated scans or dataset commits occurred.

## Verification

`tests/test_pipeline.py` covers individual stages, four-stage/multiple-order and
repeated compositions, independently known small geometry, identity and
empty/singleton clouds, prevalidation before any operation, immutable inputs and
provenance, JSON export, seed repeatability, counts/order/intensity/frames,
controlled monotonic clock readings, later runtime failure and optional-backend
behavior. CI uses synthetic data and mocked import/timing boundaries.

```bash
python -m pytest -q
python -m ruff check .
python -m pip check
git diff --check
```
