# KITTI annotations and 3D box conventions

The parser targets KITTI 3D Object Detection `label_2/*.txt` files. Refer to the
[official benchmark and object development kit](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)
for source data specifications. No calibration, coordinate transformations,
rendering, detection inference, or evaluation is implemented by this parser.

## Reusable API

```python
from raven.perception.data.kitti_annotations import load_kitti_labels

labels = load_kitti_labels("data/kitti/training/label_2/000000.txt")
for record in labels.records:
    if record.is_ignored:
        # Preserve the image-space ignore region; there is no valid 3D box.
        print(record.bbox_xyxy)
    else:
        box = record.ground_truth
        print(box.label, box.position_xyz, box.coordinate_frame)
```

`parse_kitti_annotation(line)` parses a single record. `load_kitti_labels(path)`
returns a frozen `KittiLabels` frame container. Records retain file order;
`objects` contains only validated non-ignore boxes, and `ignored_regions` retains
`DontCare` records. Unknown class names are retained verbatim with
`is_known_class=False`; they are neither discarded nor mapped to another class.
The recognized classes are Car, Van, Truck, Pedestrian, Person_sitting, Cyclist,
Tram, Misc, plus the special DontCare ignore marker.

## Original fields

Exactly 15 whitespace-separated fields are required, with an optional 16th score.
Blank lines in a file are skipped. Malformed records fail the entire load with a
filename and physical line number; they are not silently dropped.

| Source fields | Annotation attribute | Meaning |
| --- | --- | --- |
| type | `object_type` | Original case-sensitive class name |
| truncated | `truncation` | Image truncation fraction, or unavailable sentinel -1 |
| occluded | `occlusion` | Integer 0 visible, 1 partial, 2 largely occluded, 3 unknown; -1 unavailable |
| alpha | `alpha` | Observation angle in radians; -10 unavailable |
| left, top, right, bottom | `bbox_xyxy` | Zero-based image coordinates in pixels |
| height, width, length | `dimensions_hwl` | Original local box dimensions in meters |
| x, y, z | `location_xyz` | Original bottom-center location in rectified camera coordinates, meters |
| rotation_y | `rotation_y` | Original right-handed rotation about camera Y, radians |
| optional score | `source_score` | Finite raw ranking score, or None when absent |

All numeric values must be finite. Non-ignore boxes require positive dimensions,
ordered 2D bounds, valid truncation/occlusion values, and angles within [-pi, pi]
except the documented unavailable alpha sentinel. DontCare records retain their
finite placeholder dimensions, locations, angles, and metadata without treating
these placeholders as physical geometry. Their 2D bounds must still be ordered.
Zero-area 2D bounds are retained; no image-size clipping or difficulty filtering
is performed.

The optional source score is not assumed to be a calibrated probability; finite
values outside [0, 1] are preserved. No confidence is invented for ground truth.
This loader is intended for trusted annotation files: an optional score neither
converts a record into a `Detection3D` prediction nor proves its ground-truth
provenance. Results-to-detection conversion is outside this implementation.

## Coordinate and box semantics

KITTI annotations use `kitti_rectified_camera`: X points right, Y down, Z forward.
The source location is at the bottom-center of the box. It is **not** its geometric
center. The parser does not apply the potential geometric-center offset
`y - height / 2` and does not perform any camera-to-LiDAR transformation.

`GroundTruth3D` explicitly stores `position_xyz`, `position_reference`,
`size_lwh`, `rotation_rad`, `rotation_axis`, `label`, and `coordinate_frame`.
For KITTI, position_reference is bottom_center and rotation_axis is y. The only
reordering is dimensions `(h, w, l)` to `(l, w, h)`; raw dimensions remain on the
annotation. Camera rotation_y is never silently interpreted as LiDAR Z-axis yaw.
Dataset-specific metadata remains on the associated annotation record.

`Detection3D` remains at its existing import path and accepts the original five
positional arguments: geometric `center_xyz`, `size_lwh`, `yaw_rad`, `label`,
`score`. It now validates shapes, finite numbers, positive dimensions, and a
confidence probability in [0, 1]. The backwards-compatible defaults are
`coordinate_frame="lidar"`, `rotation_axis="z"`; backends using other conventions
must explicitly supply the frame and axis. These defaults do not perform a
transformation. Generic ground-truth boxes require a coordinate frame and can
explicitly select geometric_center or bottom_center position semantics.

Both models use meters and radians, preserve angle values without normalization,
and defensively store vectors as immutable numeric tuples. Ground truth has no
prediction score field. The [calibration module](kitti-calibration.md) converts these boxes to full-pose
geometry while retaining their original records. Future bounding-box visualization
must honor the frame, reference point, dimension ordering, and rotation axes.

## CLI and validation

```bash
raven inspect-labels data/kitti/training/label_2/000000.txt
python -m pytest -q
python -m ruff check .
```

The JSON summary includes frame_id, coordinate_frame, objects, class_counts,
ignored_regions, and unknown_classes. Unknown classes count as labeled objects;
DontCare regions do not. No datasets, Open3D, GPU, display, or network access are
required by annotation tests, which use synthetic records. Local smoke checks
can load existing training labels; testing-split labels are not supplied locally
and must not be fabricated.
