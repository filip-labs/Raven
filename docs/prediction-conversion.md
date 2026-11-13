# PointPillars prediction conversion and JSON (RV-023)

`raven.perception.detection.predictions` provides CPU-only conversion, export and
import of the existing RV-022 schema-v1 format. It imports NumPy and Raven models;
reading results never imports the OpenPCDet backend, torch, CUDA, spconv or Open3D.
The existing `pointpillars.convert_predictions` and `_openpcdet.serialize_predictions`
import paths remain available. Inference and adapter signatures are unchanged.

## Geometry and validation

OpenPCDet supplies seven-column `pred_boxes=(x,y,z,dx,dy,dz,heading)`, one-dimensional
`pred_scores`, and one-dimensional, one-based `pred_labels`. The class vocabulary
is supplied in model order; the pinned KITTI model uses Car, Pedestrian, Cyclist.
The conversion produces the existing `Detection3D` model:

- `center_xyz=(x,y,z)`: geometric center in KITTI Velodyne meters;
- `size_lwh=(dx,dy,dz)`: local length, width, height, in meters;
- `yaw_rad=heading`: radians about +Z; no wrapping or normalization;
- `label`: the configured class at `pred_labels - 1`;
- `score`: unchanged confidence, including valid endpoints 0 and 1;
- `coordinate_frame="kitti_velodyne"`, `rotation_axis="z"`.

No camera transform, dimension permutation, bottom-center correction or confidence
threshold is applied. Empty arrays shaped `(0,7)`, `(0,)`, `(0,)` return `[]`.
Shapes/counts must agree. All values must be real, finite numbers representable
as Python floats; dimensions must be positive, scores in `[0,1]`, and class
indices integral and within the vocabulary. Booleans (including mixed Python
sequences), strings, complex/object arrays, malformed/ragged arrays and invalid
class names fail with descriptive errors. Integral floating-point label arrays
remain accepted, as in the existing API. JSON `label_id` fields require integers.
Inputs are read without mutation; output vectors are independent immutable tuples.

### Precision

Values are stored as Python binary64 floats by `Detection3D`. A finite float32
value promotes exactly, including its already-rounded binary value. For example,
`float(np.float32(0.1))` is approximately `0.10000000149011612`, not ideal decimal
`0.1`. JSON round trips preserve that Python-float value; they do not recover lost
float32 precision. Higher-precision floats and integer geometry also follow the
existing model's Python-float conversion, not arbitrary-precision arithmetic;
values outside its finite range are rejected. No decimal rounding, epsilon
comparison, angle normalization or threshold adjustment is introduced here.
RV-024 must define tolerances and compare actual per-box outputs separately.

## Existing JSON envelope

Schema version remains **integer 1**. Existing fields are retained:

- `input.frame_id` and `input.coordinate_frame` identify the source frame;
- top-level coordinate frame, rotation axis, geometric-center reference,
  seven-column box format and meter/radian units describe the geometry;
- `predictions` contains raw-format `box`, `score`, `label_id`, `class_name` rows;
- adapter-mode `detections` contains all seven named `Detection3D` fields;
- `total_predictions` and `class_counts` summarize the records;
- existing full runner captures retain their runtime/model/input provenance.

`class_names` is additive metadata written by the shared serializer. Older RV-022
captures obtain the same vocabulary from `effective_config.CLASS_NAMES`. If both
are present they must agree. No implicit vocabulary is invented when reading a
capture without either source. The export helper defaults to the pinned KITTI
class order; pass `class_names=` explicitly for another supported class mapping.

Import validates the envelope, all raw rows, and every named detection. When both
representations are present they must agree exactly in order, geometry, label,
confidence and frame; mismatches fail rather than choosing one silently. Raw-only
standalone schema-v1 captures can also be imported. Counts and class-name/index
mapping must agree. Missing conventions or model fields never fall back to the
legacy `Detection3D` constructor defaults. Unknown top-level provenance fields are
allowed; they are not authenticated by this geometry reader. Per-record fields
are strict. Unsupported versions, duplicate JSON keys and nonstandard NaN/Infinity
constants are rejected. Later incompatible layouts require a new schema version.

Frame IDs must be six ASCII digits and, when supplied, match `expected_frame_id`.
This is format/association validation, not proof of dataset membership, source
checksum, training/testing partition, or calibration identity. Those require the
[dataset provenance policy](kitti-dataset.md). Equal IDs in different KITTI
partitions do not establish that they refer to the same scan. A frame string is
not a coordinate transformation; this format supports KITTI Velodyne +Z boxes.

## CPU-only use

This example uses synthetic values, not a recovered GPU capture:

```python
from pathlib import Path
import numpy as np
from raven.perception.detection.predictions import (
    convert_predictions, save_detections, load_detections,
)

boxes = np.array([[1, 2, 3, 4, 2, 1, 0.25]], dtype=np.float32)
detections = convert_predictions(boxes, np.array([0.9]), np.array([1]),
                                 ["Car", "Pedestrian", "Cyclist"])
Path("outputs").mkdir(exist_ok=True)
save_detections("outputs/synthetic_predictions.json", detections, frame_id="000000")
restored = load_detections("outputs/synthetic_predictions.json", expected_frame_id="000000")
assert restored == detections
```

`save_detections` creates a new file and refuses to overwrite an existing capture.
It reuses the existing envelope without fabricating environment/model provenance.
For an already parsed mapping use `detections_from_document`; for a JSON-ready
mapping use `detections_to_document`. The loader returns `list[Detection3D]`, ready
for future evaluation/overlay code. Keep the original capture if runtime evidence
is needed; exporting boxes into a fresh document does not recreate that evidence.
The Kaggle runner now validates its assembled document before writing and keeps
its existing CLI and adapter behavior.

## Real-capture evidence and remaining work

[PR #58's experimenter report](https://github.com/filip-labs/Raven/pull/58#issuecomment-6839286530)
records an actual `PointPillarsDetector.detect` run on a Kaggle Tesla T4: exit 0,
115,384 input points for frame 000000, **29** detections — Car **4**, Pedestrian
**19**, Cyclist **6** — in `kitti_velodyne`. This establishes the reported adapter
execution, not detection accuracy or numerical equivalence.

On 2026-11-14, `raven_adapter_predictions_000000.json` was not found in the local
repository/Downloads/attachments searches or the available Spotlight name search.
Its contents were not fabricated, loaded or committed. Local validation of that
capture remains unavailable. The new conversion/JSON boundary is CPU-tested with
synthetic fixtures; no new GPU inference was run for RV-023. If you have the capture,
validate it read-only from an environment with the standard Raven core installed:

```bash
python - <<'PY'
from collections import Counter
from hashlib import sha256
from pathlib import Path
from raven.perception.detection.predictions import load_detections
path = Path('/path/to/raven_adapter_predictions_000000.json')
before = sha256(path.read_bytes()).hexdigest()
boxes = load_detections(path, expected_frame_id='000000')
print(len(boxes), dict(Counter(box.label for box in boxes)))
print({box.coordinate_frame for box in boxes})
assert sha256(path.read_bytes()).hexdigest() == before
PY
```

Compare printed counts with the historical report without editing the capture.
Count equality is not per-box equivalence. RV-024 owns reference matching/tolerances;
RV-028 owns prediction/ground-truth overlays. This change implements neither AP
metrics nor rendering, and makes no accuracy or independent-generalization claim.

## Tests

```bash
python -m pytest -q tests/test_prediction_conversion.py
python -m pytest -q
python -m ruff check .
python -m pip check
git diff --check
```

Synthetic tests cover empty/single/multiple outputs, all three classes, shapes,
nonfinite values, score/dimension/index/type validation, immutable arrays,
float32 precision, legacy/current schema round trips, frame/convention checks,
duplicate-representation disagreement and strict JSON. A fresh interpreter blocks
all GPU/GUI imports while exporting and importing validated detections.
