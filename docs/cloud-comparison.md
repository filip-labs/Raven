# Raw vs processed LiDAR comparison (RV-027)

Install from the repository root in a graphical desktop environment:

```bash
python -m pip install -e ".[visualization,preprocessing]"
```

Open3D remains optional; the normal installation, inspection, saved prediction
readers and CPU tests need no renderer. SciPy is needed only when a configured
statistical stage processes at least two points. Preprocessing runs on CPU and
does not import or invoke PointPillars, CUDA, torch, OpenPCDet or spconv.

## Navigation and export

Add `--navigate` when the input is a Velodyne directory, optionally selecting
`--start-frame 000000`. RAW and PROCESSED update together; processing occurs once
per selected frame. F10/F11 navigate, F12 fits the current pair, and Shift+F12
exports a numbered RAW/PROCESSED PNG pair. Navigation defaults to
`outputs/screenshots`; use `--screenshot-dir` to choose another directory or to
enable export for a single frame. F1/F2/F3 retain comparison behavior.
See [scene navigation and export](scene-navigation.md) for frame matching, camera
persistence, limitations and native macOS checks. Comparison plus overlays remains
explicitly unsupported.

## CLI and configurations

```bash
# A: original vs inclusive range crop.
raven visualize /path/to/kitti/training/velodyne/000000.bin \
  --compare configs/preprocessing/range.json

# B: original vs 0.2 m voxel means, with intensity coloring.
raven visualize /path/to/kitti/training/velodyne/000000.bin \
  --compare configs/preprocessing/voxel.json --color intensity

# C: original vs range → voxel → statistical → seeded noise.
raven visualize /path/to/kitti/training/velodyne/000000.bin \
  --compare configs/preprocessing/full.json --color distance --point-size 3
```

Use your own licensed scan; no KITTI data or generated processed scans accompany
these configs. A config file is exactly the ordered stage list accepted by the
[existing pipeline](preprocessing-pipeline.md), encoded as JSON. There is no new
wrapper or second parameter schema. For example:

```json
[
  {"name": "voxel_downsample", "voxel_size": 0.2},
  {"name": "noise", "position_std": 0.01, "seed": 42}
]
```

`[]` compares the scan with an independent identity copy. Any supported stage
order/repetition is allowed, with explicit parameters and a seed for every noise
occurrence. Unknown names, missing/extra keys, invalid values, duplicate JSON keys,
nonstandard NaN/Infinity tokens, malformed JSON and missing files are errors.
All stages are prevalidated before processing and before opening windows.
Data-dependent/backend failures retain the pipeline's stage diagnostics.
No preprocessing is recomputed during camera movement, matching or reset.
Frame selection in navigation mode runs the pipeline once for the new frame.

`--color distance|intensity|uniform` and `--point-size 1..25` are supported.
`--compare` cannot be combined with `--ground-truth` or `--predictions`; these
remain available through the existing [overlay viewer](visualization.md). Labels
and calibration flags still require `--ground-truth`. Comparison displays point
clouds rather than pretending processed points are new verified detections.

## Public API and input contract

```python
from raven.perception.preprocessing.pipeline import run_preprocessing_pipeline
from raven.perception.visualization.comparison import prepare_comparison, view_cloud_comparison

result = run_preprocessing_pipeline(cloud, stages)
summary = prepare_comparison(cloud, result.cloud, color='distance')  # no Open3D import
print(summary.raw_count, summary.processed_count, summary.reduction_percent)
view_cloud_comparison(cloud, result.cloud, color='distance', point_size=3)
```

The reusable viewer accepts two PointCloud objects with identical frame IDs and
coordinate-frame strings. It rejects a missing/empty original cloud. An empty
processed cloud is valid and shows axes with an EMPTY title. Input XYZ, intensity,
order and metadata are never modified or saved by comparison. Geometry passed to
native Open3D owns writable float64 buffers, separate from read-only source data.

`prepare_comparison` provides immutable fields/read-only arrays for counts,
percentage reduction, both RGB arrays and the shared camera bounds. Reduction is
`100 * (1 - processed_count/raw_count)`. The viewer API also accepts a larger
processed cloud, yielding a negative reduction; the current pipeline operations
do not increase point count. The processed title shows signed percentage change
(e.g. `-45.53%` means 45.53% fewer points); the terminal reports reduction explicitly.
Frame-ID agreement is a consistency check, not proof of scan provenance: pass the
corresponding pipeline result and retain input hashes/experiment records.

## Layout, cameras and controls

Two legacy Open3D/GLFW windows are initially positioned side by side, RAW left and
PROCESSED right, each with a 640×720 content viewport. Window titles identify the
clouds and counts; processed also shows percentage change. The launching terminal
prints both counts, reduction and the controls. OS display/Retina/window-manager
placement can affect apparent size/position; move the windows if necessary.

Both views use the **union of raw and processed XYZ bounds**, so noise outside
the raw extent is included. Neither view is independently fitted to its own
cloud. A coincident/singleton cloud gets camera-only nonzero padding; the points
are never translated or scaled to create the layout. Both coordinate axes are
at the actual origin, with 2-unit size (meters for KITTI).

The complete legacy ViewTrajectory state is copied, including look-at, front,
up, zoom, field of view and common bounding box. Camera changes from either
window are detected in the shared event loop and copied to its partner. Point
size changes are linked too. If both cameras change between polls, RAW wins.
Updates can lag by one event-loop iteration; this is not frame-locked rendering.
The implementation uses Open3D's supported
[Visualizer view-state API](https://www.open3d.org/docs/release/python_api/open3d.visualization.Visualizer.html).
The [upstream ViewControl implementation](https://github.com/isl-org/Open3D/blob/main/cpp/open3d/visualization/visualizer/ViewControl.cpp)
shows that the view state retains the bounding box, FOV and zoom used for projection.

| Control | Action |
| --- | --- |
| Left drag | Standard Open3D orbit, copied to the partner |
| Ctrl + left drag | Standard Open3D pan, copied to the partner |
| Mouse wheel | Standard Open3D zoom, copied to the partner |
| F1 | Print comparison help |
| F2 | Explicitly match both cameras from the focused window |
| F3 | Restore equal window sizes, initial common view and CLI point size |
| Q / Esc or closing either window | Finish the pair; clean up both windows |

Focus a window before pressing keys. macOS may require Fn with F1–F3. These
comparison-specific F-keys differ from the overlay viewer's layer controls.
Keep both window dimensions equal for identical projection/aspect ratio. Manual
independent resizing changes aspect ratio; F3 restores matched dimensions and
position. No independent autoscaling is performed on navigation or reset.

Distance colors share one maximum Euclidean distance from the coordinate origin
across both clouds: zero green, maximum red, blue 0.2. Thus an unchanged point
keeps the same distance color after a crop. Intensity retains the existing clipped
[0,1] grayscale semantics; uniform remains gray 0.7. Voxel means and noise can
change geometry/intensity according to their documented preprocessing rules.

## Real KITTI and macOS validation

Read-only frame 000000, 115,384 points, SHA256:
`0e09c85e3f6078ecbdd1e706ee9624519f1bd29417437167a9ed7fbe6f54b4b1`.
On 2026-11-14, both computations and actual GLFW rendering were exercised on
macOS 26.6.2 arm64, Python 3.11.12, NumPy 2.4.6, Open3D 0.20.0;
SciPy 1.17.1 was installed in the GUI environment for the full configuration.

| Config | Processed count | Reduction |
| --- | --- | --- |
| `range.json` | 62,853 | 45.5271% |
| `voxel.json` | 22,595 | 80.4176% |
| `full.json` | 12,232 | 89.3989% |

Range settings are X [0,70], Y [-40,40], Z [-3,1] m. Voxel size is 0.2 m.
Full adds statistical K=16 / std_ratio=1.0, then noise std=0.01 m / seed=42.
Repeated processing was bitwise identical and source file SHA256 unchanged.
All three pairs were opened and their raw/processed geometry inspected. Distance
and intensity colors, titles/counts, help, explicit matching, reset and clean Q
closing were exercised. No screenshots or generated clouds are committed.

A temporary, unshipped verification harness invoked real Open3D orbit/zoom/pan
methods from keyboard callbacks. Native camera intrinsic/extrinsic matrices and
full view states matched after synchronization in both directions; the changed
views were also inspected visually. This verifies actual renderer camera copying,
not native mouse-gesture delivery: automated mouse drags failed window routing in
the multi-display/Retina session and are not claimed as visually confirmed.

An initial attempt using the newer Open3D GUI/Metal renderer displayed a black
window; an independent minimal GUI/label probe did the same on this machine.
The shipped implementation uses the existing GLFW/OpenGL renderer instead.
This observation is not a claim that all macOS GUI/Metal builds fail. Other
Open3D versions, platforms, drivers and display arrangements remain unverified.

## Headless behavior, tests and research use

Run on the main thread in a graphical desktop with OpenGL support. Missing
Open3D and window/geometry/camera/event/cleanup failures produce actionable errors;
closing one window cleans up the pair. Linux without DISPLAY/WAYLAND_DISPLAY is
rejected before creating windows. Native driver crashes cannot be recovered by
Python error handling. No offscreen/headless-rendering fallback is added.

CPU CI mocks the native boundary and covers identical/reduced/empty results,
frames, immutable inputs, shared bounds/color scale, linked pose/FOV/zoom/point
size from either window, callbacks, cleanup failures, CLI JSON validation,
one-time processing, missing Open3D, worker-thread rejection and dependency-light
imports. The full existing suite also covers overlays and M2 regressions. No
renderer, dataset downloads, checkpoints or CUDA are needed for those tests.

The comparison supports research by exposing how an explicit pipeline changes
point count, density and geometry while holding the view constant. It is not a
detection-quality metric or an inference/end-to-end latency experiment. Use the
pipeline's structured provenance and separate inference/evaluation measurements
in the future experiment runner; viewing a sparser cloud proves no accuracy or
performance improvement.
