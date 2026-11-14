# Deterministic voxel downsampling (RV-014)

The CPU-only API is:

```python
from raven.perception.data.kitti_loader import load_kitti_bin
from raven.perception.preprocessing.voxel_downsample import voxel_downsample

cloud = load_kitti_bin("data/kitti/training/velodyne/000000.bin")
reduced = voxel_downsample(cloud, 0.2)              # Cubic cells.
anisotropic = voxel_downsample(cloud, (0.2, 0.2, 0.4))
print(cloud.count, reduced.count)
assert reduced.frame_id == cloud.frame_id
assert reduced.coordinate_frame == cloud.coordinate_frame
```

These sizes are exploratory examples, not tuned PointPillars settings. NumPy is
the only runtime dependency beyond the standard library/Raven core. No Open3D,
torch, CUDA, spconv or OpenPCDet import, installation or execution is required.

## Grid and membership

The fixed origin is **(0,0,0) of the caller's coordinate system**, not the minimum
point or the center of the current cloud. Each axis uses:

```text
index = floor(float64(coordinate) / size)
cell k = [k * size, (k + 1) * size)
```

Cells are half-open: the lower boundary belongs to the cell, and the upper
boundary belongs to the next cell. Negative coordinates use floor, not truncation
toward zero: with size 0.5, -0.25 maps to -1, -0.5 maps to -1, and 0 maps to 0.
Changing the cloud minimum or applying a crop does not re-anchor the grid.

Scalar sizes apply to all three axes. Ordered three-element inputs supply X/Y/Z
sizes separately, including tuples, lists, one-dimensional NumPy arrays and
ordered iterables. Sizes must be positive, finite real numbers convertible to
finite Python floats. Booleans, strings, complex values, mappings/sets, nested or
wrong-length vectors, nonpositive/nonfinite values and zero-dimensional arrays
are rejected. Use a Python/NumPy numeric scalar rather than a zero-dimensional
array. Validation applies even when the input is empty.

Coordinates and sizes are interpreted in the caller's units/frame. KITTI data
uses meters in `kitti_velodyne` (X forward, Y left, Z up), but this operation does
not impose Velodyne on another frame or perform a coordinate transform.

## Aggregation and determinism

Each occupied cell emits **one representative XYZI row**:

- XYZ is the arithmetic mean of all original XYZ rows in the cell.
- Intensity is the arithmetic mean of the corresponding intensities, with no
  clipping or normalization. Duplicate returns contribute with their multiplicity.

Cells sort lexicographically by integer `(ix, iy, iz)` (X first, then Y, then Z).
Within each cell, rows sort lexicographically by original `(x,y,z,intensity)`
before float64 accumulation/division. Thus repeated calls and permutations of
the same input multiset use the same cell and summation order, independent of
hash iteration. This is a defined reduction order, not a promise of bit-identical
results across all NumPy versions/platforms or correctly rounded arbitrary-
precision summation in ill-conditioned cases.

The returned PointCloud has independent float32 storage, read-only by default,
and retains the exact frame ID and coordinate-frame metadata. The source array,
intensity and writeability flag remain untouched. Empty input returns a new valid
`(0,4)` float32 cloud with the same metadata. A single return or identical duplicate
returns reduce to the same stored XYZI value. There is no hidden crop, sampling
limit, random selection or OpenPCDet voxelization.

This operation is **lossy**: centroids and mean intensities replace individual
returns. Original order and individual reflectances cannot be recovered from the
representatives. A centroid may not coincide with an actual laser return.

## Precision and index limits

PointCloud already stores float32. Indexing promotes those stored values to
float64 and performs division followed by floor, without epsilon/snapping to
nominal decimal boundaries. For example, stored float32 `0.7` is slightly below
Python `0.7`; division by Python `0.1` can place it in cell 6, rather than a nominal
decimal cell 7. Float64 division itself is finite-precision arithmetic.

Means accumulate/divide in float64, then round once to PointCloud's float32 output
storage. Promotion does not recover information lost in source float32 values,
and float64 summation can still lose small contributions. Final centroid rounding
can land on a cell boundary; repeated downsampling is not promised to be
idempotent. No coordinate offset or intensity rescaling is used to hide rounding.

Nonfinite scaled coordinates or `abs(coordinate/size) >= 2**53` are rejected
explicitly. This conservative limit avoids relying on float64 to resolve unit
integer indices beyond its consecutive-integer range and prevents unsafe int64
casts. Tiny positive sizes can therefore be invalid for a particular cloud even
though they pass the size-only check. The occupied-cell method allocates no dense
3D grid; canonical sorting costs O(N log N), with O(N) temporary storage.

## Future preprocessing pipelines and experiments

The [range filter](range-filtering.md) and voxel operation are separate stages:

```python
from raven.perception.preprocessing.range_filter import filter_range
cropped = filter_range(cloud, (0, 70), (-40, 40), (-3, 1))
reduced = voxel_downsample(cropped, 0.2)
```

RV-017 will compose stages with explicit configuration/order; it is not implemented
here. Crop-then-downsample and downsample-then-crop need not be equivalent. Record
order, bounds, voxel sizes/origin, input/output counts and software versions.

PointPillars still applies its own configured feature/voxel processing when a
caller supplies the reduced cloud. Raven preprocessing does not replace or disable
that stage. Fewer input points do not guarantee lower total latency: CPU sorting,
aggregation, transfers, fixed network work and upstream voxel processing all
contribute. Lost geometric/intensity information may change detection quality in
either direction. No accuracy, AP/mAP, latency gain or optimal voxel size is claimed.
Use a fixed [validation population](kitti-dataset.md), checkpoint/config and separate
preprocessing/inference timing in later experiments.

## Verification and local KITTI smoke evidence

```bash
python -m pytest -q tests/test_voxel_downsample.py
python -m pytest -q
python -m ruff check .
python -m pip check
git diff --check
```

Synthetic tests cover positive/negative membership and exact/adjacent boundaries,
scalar/anisotropic sizes, XYZ/intensity means, lexicographic ordering, input
permutations, independent storage, metadata, empty/single/duplicate inputs,
invalid sizes/index overflow, float32 precision, explicit range composition,
KITTI loader compatibility and GPU-free execution. An optional dataset test skips
in CI when local data is absent; no downloads are required.

A separate read-only check used the original checkout's complete KITTI training
frame 000000, with **115,384 points**, before any range crop. Input SHA256:
`0e09c85e3f6078ecbdd1e706ee9624519f1bd29417437167a9ed7fbe6f54b4b1`.

| Cubic voxel size | Output points |
| --- | ---: |
| 0.1 m | 47,758 |
| 0.2 m | 22,595 |
| 0.4 m | 9,117 |

Each count matched independently enumerated unique floor-based cell IDs. Up to
25 multi-point cells per size were checked against separate `math.fsum` means
at float32 precision, and repeated outputs were identical. Input file hash/array
and metadata remained unchanged; outputs had independent read-only storage.
These are compatibility/count observations on one scan, not inference or latency
measurements. No dataset content or generated capture is committed.
