# Point-cloud and KITTI ground-truth visualization

Install the existing optional Open3D extra in a graphical desktop environment:

```bash
python -m pip install -e ".[dev,visualization]"
raven visualize data/kitti/training/velodyne/000000.bin --ground-truth
raven visualize data/kitti/training/velodyne/000001.bin --ground-truth --color uniform
```

`--ground-truth` adds dataset annotations, not detector predictions. It does not
invoke PointPillars, read prediction JSON, perform matching, or assign confidence.
The original point-only command works unchanged when this flag is absent.

## Input paths and CLI options

For a standard KITTI Object Detection layout, a scan at
`SPLIT/velodyne/FRAME.bin` resolves labels at `SPLIT/label_2/FRAME.txt` and
calibration at `SPLIT/calib/FRAME.txt`. Paths are inferred from the supplied scan,
not the current working directory. Matching files are required even for empty
label files; testing-split labels are not fabricated.

Nonstandard layout example:

```bash
raven visualize /path/to/scans/000000.bin --ground-truth \
  --labels /path/to/annotations/000000.txt \
  --calib /path/to/calibration/000000.txt \
  --color intensity --point-size 3.0
```

`--labels` and `--calib` require `--ground-truth`. In a standard layout, either
path can override its inferred sibling; otherwise both explicit paths are needed.
All three filename stems must match exactly. This catches obvious frame mistakes
but does not prove files from different sequences with equal names belong together;
the user must supply the correct matching calibration and labels.

Missing files, malformed annotations/calibration, mismatched frame IDs, and invalid
CLI options fail before a GUI opens. Empty or DontCare-only annotation files show
the point cloud with no box lines. An empty cloud remains invalid for visualization.

The point-color modes remain distance (default), intensity, and uniform. Point
size accepts finite values from 1 to 25 pixels, default 2. Box colors are fixed:

| Annotation class | Box RGB | Appearance |
| --- | --- | --- |
| Car | (0, 1, 0) | Green |
| Pedestrian | (1, 0, 0) | Red |
| Cyclist | (0, 0, 1) | Blue |
| Van, Truck, Person_sitting, Tram, Misc, unknown names | (1, 1, 0) | Yellow fallback |

The wireframes use saturated RGB values. Whenever at least one box is present,
point RGB values are multiplied by 0.35, preserving the selected mapping while
making box lines brighter. With no boxes, point coloring is unchanged. There are
no labels, legends, class filters, confidence labels, or independent layer toggles.

## Geometry and reusable API

```python
from raven.perception.data.kitti_loader import load_kitti_bin
from raven.perception.visualization.boxes import load_ground_truth_boxes
from raven.perception.visualization.viewer import view_point_cloud

path = "data/kitti/training/velodyne/000000.bin"
cloud = load_kitti_bin(path)
boxes = load_ground_truth_boxes(path)
view_point_cloud(cloud, boxes=boxes, color="uniform")
```

The existing annotation parser preserves KITTI rectified-camera bottom centers
and rotation about camera Y. `load_ground_truth_boxes` reuses
`load_kitti_calibration` and `transform_annotation` to produce `OrientedBox3D`
geometry in `kitti_velodyne` (X forward, Y left, Z up; meters). The calibration
module performs the bottom-to-geometric-center offset and full matrix rotation.
No yaw approximation, manual axis swap, or second transformation is added by the
viewer. Source annotation references and immutable calibration arrays are preserved.
See [calibration conventions](kitti-calibration.md) for the equations and tolerances.

`wireframe_from_corners(corners, label)` validates and copies eight distinct finite
ordered vertices. `box_wireframes(boxes, coordinate_frame)` rejects mismatched
frames and batches all boxes into one points/edges/colors tuple without Open3D.
The viewer converts it into one Open3D LineSet. Lower local-height corners are
0–3, upper corners 4–7; edges are:

```text
0–1, 1–2, 2–3, 3–0
4–5, 5–6, 6–7, 7–4
0–4, 1–5, 2–6, 3–7
```

Each box has eight vertices and twelve edges. Corners come directly from the full
source geometry, including calibration roll and pitch. No minimum-volume or
axis-aligned box is fitted. Unknown classes retain their original names and use
the fallback color; DontCare is ignore metadata and produces no physical box.

## Validation and limitations

CPU tests verify geometry, vertex/edge counts, edge lengths, independent camera to
LiDAR corner calculations, full rotation, color selection, file resolution,
missing/mismatched inputs, ignored/empty annotations, and the mocked Open3D
boundary. They require neither Open3D nor a display, real dataset, CUDA, or network.

A real-scene smoke check should load matching scan/label/calibration files, compare
wireframe corners to calibrated geometry numerically, then open the scene. Render
captures provide additional visual evidence; screenshots alone do not establish
mathematical correctness. Manual mouse rotation, zoom, pan, and window closing
remain user validation when desktop automation cannot exercise them reliably.
The viewer uses the existing Open3D event loop and does not override those controls.

This remains a single-frame desktop viewer requiring graphical OpenGL support.
Headless rendering, model prediction overlays, detector inference, and frame
navigation are outside scope. Thin lines and apparent color depend on display
resolution, viewing distance, occlusion, and OpenGL rendering. The initial view
continues to fit the point cloud; zoom in to inspect small or distant boxes.
Point clouds and all source data remain unmodified, and no screenshots or datasets
are checked into the repository.
