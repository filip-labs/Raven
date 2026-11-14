# Deterministic XYZ range filtering (RV-013)

The existing CPU-only API is retained:

```python
from raven.perception.data.kitti_loader import load_kitti_bin
from raven.perception.preprocessing.range_filter import filter_range

cloud = load_kitti_bin("data/kitti/training/velodyne/000000.bin")
filtered = filter_range(cloud, x_range=(0, 70), y_range=(-40, 40), z_range=(-3, 1))
print(cloud.count, filtered.count)
assert filtered.frame_id == cloud.frame_id
assert filtered.coordinate_frame == cloud.coordinate_frame
```

The shown ROI is illustrative, not a declaration of the pretrained model's
configured range. Positional range arguments remain supported. No dataset is
bundled or downloaded by the filter.

## Semantics and validation

A point survives only when **all three** conditions hold:

```text
x_min <= x <= x_max
y_min <= y <= y_max
z_min <= z <= z_max
```

Both endpoints are inclusive. Each interval must have two finite real numeric
bounds with **minimum < maximum**. Equal bounds remain invalid, preserving the
existing contract; zero-width slices are not introduced. Reversed/nonfinite
bounds, booleans, strings, complex values, malformed pairs, mappings and unordered
sets are rejected. Errors identify X, Y or Z. Unsupported collections/endpoint
types raise TypeError; malformed pairs, nonfinite values and invalid ordering
raise ValueError. Ordered iterable pairs, including
lists/one-dimensional NumPy arrays, are accepted and consumed once. Validation
also runs for an empty input cloud.

Bounds are normalized to finite Python floats. Stored float32 coordinates are
promoted to float64 for comparison so a more precise threshold is not implicitly
rounded down/up to float32. This does not restore precision already lost when
PointCloud stored its data. For example, `np.float32(0.1)` is slightly greater
than Python `0.1`; it is outside an upper bound of exactly Python `0.1`. Supply
the actual stored value as a bound when selecting that exact boundary.

The implementation uses a vectorized NumPy mask over XYZ. Retained **entire XYZI
rows** stay in original order with bit-identical stored values. Intensities are
neither clipped nor normalized, and duplicate rows are not merged. The function
returns a new PointCloud with an independent float32 array, read-only by default;
input data, writeability and metadata are unchanged even when every point survives.

No survivors is a valid `(0,4)` float32 cloud with count 0 and unchanged frame ID
and coordinate-frame name. Empty input produces the same valid empty shape.
Downstream components retain their own policies: the viewer and PointPillars
adapter currently reject empty clouds; callers should inspect `filtered.count`
before invoking them. No placeholder points are invented.

## Units and coordinate frame

Bounds use the **same units and XYZ axes as the caller's cloud**. Raven's KITTI
loader supplies meters in `kitti_velodyne`: X forward, Y left, Z up. Other caller
frames are preserved; filtering does not silently transform them into Velodyne,
change axis order, or introduce a new coordinate convention. Choose bounds for
the actual source frame. No intensity, yaw or box transformation is performed.

## Later preprocessing and PointPillars experiments

This is an explicit caller-operated crop. It does not execute OpenPCDet, alter
its configuration, or automatically apply another range filter inside the adapter.
If a caller passes the result to PointPillars, the model's configured preprocessing
still runs. Do not assume its range/boundary policy equals Raven's inclusive ROI.

Range filtering removes information outside the ROI irreversibly from the result;
keep the original cloud for baseline comparisons. A smaller point count may change
voxelization and computation, but neither latency improvements nor detection
accuracy gains are established here. Objects may lose supporting points or leave
the ROI entirely. Future voxel/outlier/noise experiments should record bounds,
frame IDs, before/after point counts, preprocessing order and timing, while keeping
the same [validation population](kitti-dataset.md), model/checkpoint and evaluation
policy. Voxelization and experiment orchestration are separate work.

The mask is O(N) and uses a temporary float64 XYZ buffer plus the boolean mask;
the result is copied into PointCloud's existing storage contract. This is a
correctness-oriented single-cloud operation, not a benchmark result or streaming
implementation.

## Verification

```bash
python -m pytest -q tests/test_range_filter.py
python -m pytest -q
python -m ruff check .
python -m pip check
git diff --check
```

Synthetic tests cover each axis, exact endpoints/adjacent float32 values,
float64-threshold behavior, empty/all-kept results, malformed/nonfinite/reversed
bounds, intensity alignment, order, independent output storage, deterministic
results, metadata, KITTI loader compatibility, and GPU/GUI-free import/execution.
An optional real-data smoke test skips when the local dataset is absent; no KITTI
download is required by CI.

An additional read-only check used the original checkout's local KITTI training
frame 000000 via an explicit external path from the clean worktree:

- Input: **115,384 points**, SHA256
  `0e09c85e3f6078ecbdd1e706ee9624519f1bd29417437167a9ed7fbe6f54b4b1`.
- Illustrative inclusive ROI: X `[0,70]`, Y `[-40,40]`, Z `[-3,1]` meters.
- Retained **62,853** points; removed **52,531**.
- Full retained XYZI array matched an independently computed per-row predicate.
  Input array and file hash were unchanged; `frame_id="000000"` and
  `coordinate_frame="kitti_velodyne"` were preserved.

No point data or generated artifacts are committed, and no inference, latency or
accuracy measurement was performed for RV-013.
