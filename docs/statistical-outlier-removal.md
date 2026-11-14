# Statistical outlier removal (RV-015)

Raven provides an explicit, deterministic CPU preprocessing operation:

```python
from raven.perception.preprocessing.statistical_outlier_removal import statistical_outlier_removal
filtered = statistical_outlier_removal(cloud, nb_neighbors=16, std_ratio=1.0)
```

Install the optional CPU backend from the repository root:

```bash
python -m pip install -e ".[preprocessing]"
```

The extra declares `scipy>=1.10,<2`; `.[dev]` also installs SciPy for tests.
NumPy remains the only mandatory runtime dependency. The module imports SciPy
lazily when processing two or more points; missing SciPy produces an actionable
error. Empty/singleton copies and imports work without SciPy. No PyTorch, CUDA,
OpenPCDet, spconv, cumm or Open3D is imported by this preprocessing path.
There is no implicit inference call or new CLI command.

## Definition and boundary policy

For N >= 2, let K = min(nb_neighbors, N-1). For each row i:

1. Find the K nearest other rows in XYZ by Euclidean distance.
2. Compute their arithmetic mean distance m_i.
3. Compute mu = sum(m_i)/N and population sigma = sqrt(sum((m_i-mu)^2)/N).
4. Keep the entire original XYZI row iff **m_i <= mu + std_ratio * sigma**.

The threshold is inclusive, without an epsilon or hidden score threshold.
`nb_neighbors` must be a positive integer (including NumPy integer scalars).
`std_ratio` must be a finite nonnegative real number; zero is valid. Booleans,
strings, complex numbers and non-scalar arrays are rejected. Invalid types raise
TypeError; invalid values raise ValueError, even for empty inputs. A finite ratio
whose computed threshold overflows is rejected with ValueError.

An empty input produces an independent `(0, 4)` cloud. A singleton is returned
as an independent copy because no other neighbor exists. For other small clouds,
K is capped at N-1 rather than rejecting the cloud or inventing neighbors. With
N=2 both rows have the same mean distance and survive. Valid nonempty clouds
cannot produce an empty result under this nonnegative-ratio rule: at least one
mean is <= mu. Constant-distance clouds retain all rows even when ratio=0.

Only the query's own **row index** is excluded. Distinct rows at identical XYZ
remain valid zero-distance neighbors; intensity does not affect neighbor search.
No duplicate removal or resampling occurs. We query K+1 candidates, exclude self
if present, sort distances, and take K. Self may be absent from a tied set of
K+1 zero distances; those candidates are already other rows. Choosing different
identities at an exactly equal distance does not change the distance mean.

## Data and coordinate conventions

Output is a new PointCloud with independent, read-only float32 storage. Retained
XYZI rows are copied exactly, in original order; intensity is neither normalized
nor averaged. The original cloud is never modified. Frame ID and coordinate
frame are preserved verbatim. The caller supplies the coordinate system and its
units. KITTI Velodyne uses meters, X forward, Y left, Z up; distances and the
threshold are in meters. There is no coordinate transform, normalization,
automatic range crop or OpenPCDet range/voxel processing.

PointCloud already rejects nonfinite XYZI data. Distances, per-point means and
global statistics use float64 over the existing float32 coordinates, which cannot
recover precision lost before filtering. Neighbor distances and global means are
sorted before reduction. Global statistics are computed after subtracting the
minimum mean, then the offset is restored to mu; this equivalent formulation
preserves the constant-distance case. Population variance uses ddof=0.
Single-worker exact queries and fixed reduction ordering make repeated calls
deterministic in a fixed environment, including duplicate/equal-distance cases.
Points extremely close to a threshold may differ across SciPy/NumPy versions or
platform floating-point implementations; cross-platform bitwise identity is not
promised. There is no tolerance that silently changes the selection rule.

## Backend and resource costs

The backend is [SciPy KDTree](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html),
with balanced/compact nodes and a private data copy. Its
[query API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.query.html)
is called with `eps=0`, Euclidean `p=2`, and `workers=1`. This provides exact
three-dimensional neighbor searches on CPU and avoids a full pairwise matrix.
SciPy is a compiled dependency: use a compatible wheel for your Python/platform
or provide a suitable local build toolchain when no wheel is available.

For typical 3D scans and modest K, tree construction is approximately O(N log N),
queries roughly O(N log N + NK), and distance sorting O(NK log K). Global sorting
adds O(N log N). These are practical expectations, not worst-case guarantees:
degenerate spatial distributions can degrade tree queries, and K near N requires
quadratic neighbor work even though buffers are bounded. Use modest K for real
scans; the API does not hide or silently replace expensive requested values.

Tree/data/statistical storage is O(N). Queries run in chunks of at most 4,096
rows and normally at most 262,144 neighbor entries. Float64 distances plus
64-bit indices use about 4 MiB per such query buffer pair; masks and SciPy's
internal buffers add overhead. If K+1 exceeds the entry budget, one row is
queried at a time, requiring O(K) query storage. Owned float64 XYZ and its tree
copy together require about 48N bytes, in addition to tree nodes, O(N) means,
sorted/statistical temporaries, source data and output copies. This is not a
measured peak-RSS bound. No N-by-N matrix is allocated for a complete scan.

## KITTI example and observed smoke test

```python
from raven.perception.data.kitti_loader import load_kitti_bin
from raven.perception.preprocessing.statistical_outlier_removal import statistical_outlier_removal

cloud = load_kitti_bin("/path/to/your/kitti/training/velodyne/000000.bin")
filtered = statistical_outlier_removal(cloud, nb_neighbors=16, std_ratio=1.0)
print(cloud.count, filtered.count, cloud.count - filtered.count)
assert filtered.frame_id == cloud.frame_id
assert filtered.coordinate_frame == "kitti_velodyne"
```

Read-only local smoke test on 2026-11-14, unfiltered frame 000000: 115,384 points,
SHA256 `0e09c85e3f6078ecbdd1e706ee9624519f1bd29417437167a9ed7fbe6f54b4b1`.
Python 3.11.12, NumPy 2.4.6, SciPy 1.17.1, macOS 26.6.2 arm64,
`scipy.spatial.KDTree`, one query worker:

| nb_neighbors | std_ratio | Retained | Removed | Observed seconds |
| --- | --- | --- | --- | --- |
| 16 | 1.0 | 111,261 | 4,123 | 0.415 |
| 32 | 2.0 | 113,392 | 1,992 | 0.497 |

Each time is one `perf_counter` observation around the complete function call,
including tree construction, queries, statistics and output allocation. File I/O
and SciPy import are excluded. Repeated calls produced identical rows; the input
array and file SHA256 were unchanged. These are exploratory configurations and
observational smoke-test timings, not a formal benchmark or optimal settings.
No scan or generated point data is distributed with this change.

## Research limits and composition

Statistically distant measurements are not necessarily erroneous sensor returns.
Sparse distant objects, boundaries and valid isolated structures may be removed.
A global threshold cannot represent every local density. Removal is lossy:
original individual returns cannot be recovered from the output alone. No better
detection accuracy or reduced end-to-end inference latency is established here;
CPU preprocessing itself adds work and alters the detector input distribution.

[Range filtering](range-filtering.md) crops spatial bounds;
[voxel downsampling](voxel-downsampling.md) replaces occupied cells with mean
representatives. This filter selects original rows using neighborhood statistics.
They can be composed explicitly by the caller, but changing stage order changes
densities, neighborhoods and thresholds. RV-017 will address configurable pipeline
composition; this implementation does not select or repeat stages automatically.

## Validation

`tests/test_statistical_outlier_removal.py` compares small synthetic examples with
an independent exhaustive Euclidean-distance oracle (only in tests). It covers
isolated points, dense clusters, duplicate/equal distances, population statistics,
threshold equality, K near/above N, empty/singleton inputs, invalid parameters,
order/intensity/metadata preservation, repeat/permutation determinism, float32
retention, memory independence, chunked exact queries, optional-backend errors,
and GPU-free execution. Real data is not required by the tests or CI.

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check .
python -m pip check
git diff --check
```
