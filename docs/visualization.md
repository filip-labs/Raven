# LiDAR, ground-truth and prediction overlays (RV-028)

Install the optional desktop visualization extra:

```bash
python -m pip install -e ".[dev,visualization]"
```

The existing Open3D viewer displays one LiDAR scan, calibrated KITTI ground truth,
and saved schema-v1 predictions. No inference is run: torch, CUDA, spconv and
OpenPCDet are not required to read or display the JSON. Open3D remains optional
for the standard Raven installation, CLI inspection, JSON validation and CPU CI.

## Raw vs processed comparison

Use `raven visualize SCAN.bin --compare configs/preprocessing/full.json` for a
side-by-side comparison using the existing preprocessing pipeline. This mode has
its own [comparison guide](cloud-comparison.md), matched-camera controls and
shared color scale. `--compare` cannot be combined with `--ground-truth` or
`--predictions`; select the overlay modes below to inspect those layers. No
inference is performed by either path.

## Navigation and screenshot export

`raven visualize SPLIT/velodyne --navigate --ground-truth --screenshot-dir outputs/screenshots`
loads one frame at a time. F10/F11 navigate, F12 fits the current scene and
Shift+F12 exports the real framebuffer without overwriting existing files.
Navigation overlays use frame-matched sibling labels/calibration and optionally
`--predictions-dir`. Explicit single-frame overlay paths are rejected in navigation.
See the [scene guide](scene-navigation.md) for complete commands, supported
comparison navigation, filenames, error behavior and native macOS validation.

## Four single-frame overlay modes

Run from the repository root with your own licensed files. Prediction paths below
are examples; the capture is not included in the repository.

```bash
# LiDAR only (existing behavior).
raven visualize data/kitti/training/velodyne/000000.bin

# LiDAR + ground truth.
raven visualize data/kitti/training/velodyne/000000.bin --ground-truth

# LiDAR + saved predictions.
raven visualize data/kitti/training/velodyne/000000.bin \
  --predictions /path/to/raven_adapter_predictions_000000.json

# LiDAR + both overlays.
raven visualize data/kitti/training/velodyne/000000.bin --ground-truth \
  --predictions /path/to/raven_adapter_predictions_000000.json \
  --color intensity --point-size 3.0
```

`--predictions PATH` uses the [RV-023 validated reader](prediction-conversion.md).
The JSON's `input.frame_id` must match the scan stem, and all coordinate/box
conventions must be supported. Its filename can differ from the scan stem.
Missing files, malformed JSON, incompatible schema/conventions and mismatched
frame IDs fail before a GUI opens. Schema-v1 adapter and raw-only captures are
supported; malformed predictions are never silently omitted. Empty captures
produce no prediction lines. Reading/rendering never modifies the capture.

For standard `SPLIT/velodyne/FRAME.bin` paths, `--ground-truth` resolves matching
`SPLIT/label_2/FRAME.txt` and `SPLIT/calib/FRAME.txt`. Nonstandard example:

```bash
raven visualize /path/to/scans/000000.bin --ground-truth \
  --labels /path/to/annotations/000000.txt \
  --calib /path/to/calibration/000000.txt \
  --predictions /path/to/raven_adapter_predictions_000000.json
```

`--labels` and `--calib` still require `--ground-truth`. Standard layouts allow
one inferred sibling to be overridden; otherwise provide both. All scan/label/
calibration stems must match. Missing files are errors even when labels are empty;
DontCare records create no physical box. Testing labels are not fabricated.

Stem/JSON frame agreement does not authenticate scan bytes, sequence, partition
or calibration identity. Use the correct associated data and retain provenance.
The viewer performs no reference matching, AP measurement or accuracy evaluation.

## Interactive controls and legend

With at least one box supplied, the viewer prints the legend and current layer/
class counts to its launching terminal. Focus the window before pressing keys;
on macOS the keyboard may require **Fn + F1–F9**.

| Key | Action |
| --- | --- |
| F1 | Print controls and legend again |
| F2 | Toggle ground truth independently |
| F3 | Toggle predictions independently |
| F4 | Both off when both enabled; otherwise enable both |
| F5 | Car only, on both layers |
| F6 | Pedestrian only, on both layers |
| F7 | Cyclist only, on both layers |
| F8 | Restore all classes, including other/unknown annotation classes |
| F9 | Print currently visible predictions' class, confidence, center, LWH and yaw |

Class filtering preserves each layer's on/off state. F4 preserves the class
filter. Controls operate on already supplied boxes; they do not load missing
layers or execute inference. With no boxes at all, the point-only viewer remains
available. F9 is terminal output, not a 3D text label; displayed numbers use nine
significant digits, without modifying stored values or applying score thresholds.

| Layer/class | Appearance |
| --- | --- |
| GT Car | Solid green |
| GT Pedestrian | Solid red |
| GT Cyclist | Solid blue |
| GT other classes | Solid yellow |
| All predictions | Cyan dashed edges, interior heading arrow and center crossbar |

Each prediction edge has four collinear dash segments; the interior glyph helps
identify the prediction layer even when its bounding edges coincide with GT.
Box centers, corners and dimensions are never offset or inflated for styling.
Coincident rasterized edges can occlude each other, so use the independent toggles
for close inspection. The cyan arrow/crossbar does not represent extra detections.

Point colors remain distance (default), intensity or uniform; point size remains
1–25 pixels, default 2. Visible boxes dim the selected point RGB values to 35%.
Hiding/filtering all boxes restores full brightness. Layer changes preserve the
camera via `reset_bounding_box=False`; mouse navigation and other native Open3D
shortcuts remain unchanged. No numeric keys are intercepted, preserving Open3D's
native color controls. Close the window or use native Q/Escape.

The implementation uses Open3D's
[key-action callback API](https://www.open3d.org/docs/release/python_api/open3d.visualization.VisualizerWithKeyCallback.html#open3d.visualization.VisualizerWithKeyCallback.register_key_action_callback)
and [GLFW function-key tokens](https://www.glfw.org/docs/latest/group__keys.html).
Only unmodified press events act; release/repeat events are ignored. Function
keys avoid the built-in letter/numeric shortcuts; OS shortcuts may still intercept
keys before they reach the window.

## Geometry and reusable API

```python
from raven.perception.data.kitti_loader import load_kitti_bin
from raven.perception.detection.predictions import load_detections
from raven.perception.visualization.boxes import load_ground_truth_boxes
from raven.perception.visualization.viewer import view_point_cloud

path = "data/kitti/training/velodyne/000000.bin"
cloud = load_kitti_bin(path)
ground_truth = load_ground_truth_boxes(path)
predictions = load_detections("/path/to/raven_adapter_predictions_000000.json",
                              expected_frame_id=cloud.frame_id)
view_point_cloud(cloud, boxes=ground_truth, predictions=predictions, color="uniform")
```

`boxes=` retains its existing OrientedBox3D interface. `predictions=` accepts
validated Detection3D values. Ground truth reuses calibration and
`transform_annotation`: rectified-camera bottom center is shifted to geometric
center, then the **full pose** is transformed into `kitti_velodyne` (X forward,
Y left, Z up; meters). Calibration roll/pitch are preserved; no yaw approximation,
manual axis swap or second transform is added. Original annotation references and
calibration arrays remain untouched.

Predictions already have geometric-center Velodyne coordinates, LWH sizes and
heading about +Z in radians. Rendering constructs the exact Z rotation and uses
the shared `box_corners` geometry used by OrientedBox3D. Generic `lidar`/camera
frames and non-Z prediction axes fail explicitly, without relabeling/transforming
the box. Source coordinates, confidence and point intensity remain unchanged.
The JSON reader checks frame IDs before the renderer receives the predictions.
Direct API callers are responsible for associating predictions with the correct
scan because Detection3D itself has no per-scan frame ID.

The existing `wireframe_from_corners` validates eight distinct finite corners.
GT batching uses the existing twelve-edge topology (lower face 0–3, upper 4–7,
corresponding vertical edges). Prediction dash endpoints lie on those same edges;
its direction glyph lies inside the box. No axis-aligned or minimum-volume fit
is used. See [calibration conventions](kitti-calibration.md).

## Using saved RV-024 captures

Download `raven_adapter_predictions_000000.json` from the external session to a
private local path; pass it with `--predictions` and your matching local scan.
The RV-024 comparison report itself is not a prediction capture. Keep the
reference/candidate/report and their hashes together for provenance, without
committing them or the dataset/checkpoint.

[PR #60's experimenter report](https://github.com/filip-labs/Raven/pull/60#issuecomment-6868499971)
records all 29 reference/candidate detections matching with zero reported errors
for frame 000000 on a T4. That supports numerical agreement for that case, not
KITTI AP/mAP or generalization. The actual captured prediction JSON was not found
in local repository/Downloads/attachments searches for RV-028 and was not fabricated.

## Validation and limitations

CPU tests cover exact prediction poses/edge lengths, dashed/glyph geometry,
independently calculated camera-to-LiDAR alignment, unsupported frames/axes,
class filters and layer visibility, key press/repeat behavior, confidence output,
camera-preserving Open3D calls, CLI modes/errors and GPU-free capture loading.
Existing annotation/calibration and point-only CLI tests remain in place.

Actual graphical checks ran on the local macOS desktop with **Open3D 0.20.0**:

- A clearly labeled synthetic three-class scene used exactly coincident GT and
  prediction boxes. Native window inspection verified both styles, F1–F9,
  independent/both toggles, all class filters, restored point brightness,
  confidence/help output, mouse rotation/zoom, and clean Q closing.
- The real KITTI frame 000000 opened through the CLI with its real calibration/
  labels plus one explicitly synthetic prediction JSON under `/tmp`. Both layers
  loaded; F9 printed the synthetic record and window closing returned exit 0.
  This tests the real scan/GT and saved-JSON path, not model prediction correctness.

Synthetic predictions are not the unavailable Kaggle capture. Pan behavior and
other platforms/display drivers have not been separately exercised in this update.
No screenshots, datasets, checkpoints or prediction captures are committed.

This remains a legacy GLFW/OpenGL desktop viewer. Headless rendering, inference, live detection, AP metrics and 3D confidence
text remain unsupported. Frame navigation and PNG export are documented in the
[scene guide](scene-navigation.md). Line visibility depends on zoom, occlusion, resolution and driver;
the initial camera fits the LiDAR scene, so zoom in for small/distant boxes.
