# M2 preprocessing regression verification (RV-018)

## Scope and audit

M2 consists of range filtering (RV-013), voxel downsampling (RV-014), statistical
outlier removal (RV-015), controlled noise (RV-016), and their ordered pipeline
(RV-017). PR #67 was verified merged before this audit. Issues RV-013–017 were
closed and their implementations were present on main at `7f859a3`.

The existing suite was reviewed before adding tests:

| Existing module | Behavior already protected |
| --- | --- |
| `tests/test_range_filter.py` | All-axis inclusive/adjacent boundaries, precise float64 bounds, empty results, invalid bounds, original order/intensity, immutable independent storage, synthetic KITTI loading |
| `tests/test_voxel_downsample.py` | Positive/negative floor membership, zero origin, anisotropic sizes, boundary neighbors, lexicographic ordering, deterministic mean XYZI, float64 aggregation/float32 storage, invalid/extreme sizes |
| `tests/test_statistical_outlier_removal.py` | Independent tiny exhaustive nearest-neighbor reference, isolated outliers, duplicate/equal distances, self-index exclusion, inclusive population threshold, K cap/small clouds, bounded exact single-worker queries, optional SciPy errors |
| `tests/test_noise.py` | Explicit PCG64 normal parameterization, independent-axis sample statistics, seed repeatability, zero noise, unchanged intensity bits and RNG state, invalid/extreme parameters and finite float32 output |
| `tests/test_pipeline.py` | Individual stages, multiple orders/repeats, identity/empty/singleton cases, invalid configuration, frozen/detached provenance, JSON export, clock metadata, late noise overflow, missing SciPy and CPU-only imports |

These unit tests remain the primary protection for their individual numerical
contracts. Existing end-to-end composition tests mostly obtained expected output
by chaining the same Raven operations as the pipeline. This checks wiring but
cannot independently detect a shared operation regression. Other gaps were
inter-stage float32 rounding, a nonempty input becoming empty before a statistical
stage without SciPy, invalid trailing configurations preventing *every* operation,
strict provenance field sets, and additional data-dependent failure diagnostics.

No production correctness bug was found. This change modifies tests and
documentation only; it does not change preprocessing algorithms or interfaces.

## New independent integration coverage

`tests/test_preprocessing_regression.py` adds 37 synthetic test cases:

- All **24 permutations** of the four operations, plus two explicit repeated-noise
  compositions with separate seed parameters. A reference that calls no Raven
  operation or validator calculates expected point bytes and every stage count.
- A hand-derived four-stage case: seven rows become six after range filtering,
  four voxel representatives, three after statistical filtering and three after
  noise. Voxel X/intensity pairs are `(-0.5,4)`, `(0.5,4)`, `(2.25,80)`,
  `(20.25,100)`; K=1 distances are `[1,1,1.75,18]`, population mean `5.4375`,
  variance `52.69921875`. Noise uses PCG64 seed 42 and std 0.125.
- An adjacent-float32 midpoint regression: voxel averaging rounds to X=0.5
  before the following inclusive crop. Voxel→crop retains mean intensity 4;
  crop→voxel retains intensity 2. This distinguishes correctly staged storage
  and order effects even though both final counts are one.
- Nonempty→empty, singleton and initially empty flows through all four stages,
  with the SciPy spatial import unavailable. Every stage still records counts.
- Invalid fifth stages of each operation type, with all registered operations
  guarded so that any premature execution is detected.
- A later unresolvable voxel-index failure and a later overflowing statistical
  threshold. The error identifies position/name, chains the original ValueError,
  and leaves source bytes unchanged.

The reference uses Python floor-based cells and sorted cell keys, `math.fsum`
centroids, exhaustive `math.dist` neighbors and population statistics. Gaussian
samples come from explicit PCG64 **standard normals multiplied by std**, rather
than calling Raven or its `Generator.normal` expression. NumPy remains the shared
random-number primitive, not an independently implemented RNG; existing noise
unit tests additionally check empirical axis means/spreads/correlations.

Each reference stage stores float32 before the next stage, matching the public
PointCloud boundary. Fixtures avoid cancellation-sensitive or near-threshold
statistical ambiguity; the separate midpoint test deliberately targets a known
rounding boundary. Exact point bytes are checked against the reference and a
second execution in the same environment. Configuration/source immutability,
nonaliasing, frames, parameters/seeds, stage counts/removals and finite nonnegative
stage/total timings are checked along with the mathematical output. JSON export
uses `allow_nan=False`, a round trip and explicit field allowlists, including
reproducibility fields, to exclude raw point arrays from provenance.

During this audit, isolated in-memory fault injections confirmed the hand-derived
test fails if statistical filtering is skipped or noise std is incorrectly
squared. Those intentional faults were not source edits and are not shipped.
They are evidence of regression sensitivity, not a complete mutation analysis.

## CPU-only execution and optional real data

From the repository root in your environment:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check .
python -m pip check
git diff --check
```

For just the new regression module:

```bash
python -m pytest -q tests/test_preprocessing_regression.py
```

SciPy is included in the existing optional development extra. No CUDA, NVIDIA
GPU, OpenPCDet, checkpoints, downloads or Open3D display is required. Existing
fresh-process import guards cover the optional framework boundary; missing SciPy
is simulated for applicable nontrivial-filter failure and empty/singleton
success paths. Other operations and pipeline imports remain NumPy-only.

The new module uses only generated in-memory arrays. Existing optional tests in
`test_range_filter.py`, `test_voxel_downsample.py` and `test_viewer.py` look for
`data/kitti/training/velodyne/000000.bin` relative to the checkout and skip cleanly
when absent. The range test checks actual membership, voxel tests check finite,
repeatable output at three sizes, and viewer tests check loading/coloring without
opening a graphical window. No new real-scan test or dataset is added here.
Previous read-only real-scan observations remain documented in the individual
[range](range-filtering.md), [voxel](voxel-downsampling.md),
[statistical](statistical-outlier-removal.md), [noise](controlled-noise.md) and
[pipeline](preprocessing-pipeline.md) guides. They are smoke evidence, not an
independent full-scan numerical ground truth or a detection evaluation.

## Runtime, memory and reproducibility limits

The new exhaustive reference asserts at most **10 points** and uses seven in the
order tests. Its quadratic distance calculation must never process real KITTI
scans. There are no large synthetic matrices, runtime thresholds or new package
dependencies. Production neighbor search remains the existing chunked KD-tree.

Observed local validation on 2026-11-14: **955 passed, 3 skipped**, with the new
module contributing 37 passing cases. Python 3.11.12, NumPy 2.4.6, SciPy 1.17.1,
macOS 26.6.2 arm64; Torch, OpenPCDet, spconv, cumm and Open3D were not installed.
One full run reported 2.54 seconds in pytest, 2.91 seconds wall time and
87,293,952 bytes (~83.3 MiB) maximum RSS from `/usr/bin/time -l`. These are local
observations, not resource ceilings or benchmark comparisons. The three optional
real-data tests skipped because their checkout-relative scan was absent.

Point output is deterministic for fixed input/order, parameters, seed and
software environment. Timing fields intentionally vary and are not compared for
bitwise identity. Gaussian streams and floating-point threshold behavior are not
promised bitwise across all NumPy/SciPy versions, builds or platforms. CI currently
uses Python 3.11 on Ubuntu; this is not exhaustive support-matrix testing.

Synthetic correctness does not establish detector accuracy or inference latency,
sensor realism, robust behavior for every possible spatial distribution, or a
complete performance/memory bound. Sparse valid objects may still be removed by
statistical filtering, downsampling is lossy, and operation order changes inputs.
M2 is ready for review after these checks; milestone/issue closure remains a
review-and-merge decision.
