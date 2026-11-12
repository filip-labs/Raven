# ObjectDetector contract (RV-021)

`raven.perception.detection.base` provides the existing model-independent API:

```python
class ObjectDetector(Protocol):
    def detect(self, cloud: PointCloud) -> list[Detection3D]: ...
```

An implementation accepts a validated `PointCloud` and returns a list of validated
`Detection3D` values. Zero detections is `[]`, not `None`; multiple detections are
ordinary list entries. Preserve the cloud's XYZI array, intensity, frame ID,
coordinate frame and writeability state. Backend failures must be reported as
errors rather than disguised as an empty successful result. Empty clouds are valid
`PointCloud` values; backends should document their handling (the empty test double
accepts them and returns `[]`). The protocol defines no confidence threshold,
ordering, batching, GPU lifecycle, or framework-specific API.

## Detection semantics and coordinate ownership

- `center_xyz`: three finite coordinates of the box's **geometric center**, in meters.
- `size_lwh`: strictly positive finite length, width, height in object-local axes,
  in meters; not KITTI's source height/width/length order.
- `yaw_rad`: finite rotation in radians about `rotation_axis`, following the
  right-hand rule. `rotation_axis` is `x`, `y`, or `z`; the historical name does
  not force every frame to use Z. Angles are not normalized to a fixed interval.
- `label`: nonempty class text without surrounding whitespace; no fixed class
  enumeration or numeric class-to-label mapping is imposed by this contract.
- `score`: finite confidence in the inclusive range `[0, 1]`.
- `coordinate_frame`: explicit frame name without surrounding whitespace.

The original five positional constructor arguments remain supported, defaulting
to `coordinate_frame="lidar"`, `rotation_axis="z"`. This is a compatibility
default, not an automatic alias for `kitti_velodyne` or a camera frame. Model
construction validates values but cannot infer units or verify physical geometry.
Boxes are frozen, and vector inputs are copied into tuples.

**Every returned box must use the input cloud's coordinate frame.** A backend
owns any conversion of model-native coordinates, units, center reference, axis
conventions, dimensions and class IDs before constructing detections. Setting a
frame string alone is not a coordinate transform. If it cannot produce a valid
representation in that frame, it must raise an explicit error. The single-axis
box model does not represent an arbitrary full orientation matrix; an adapter
must not silently discard unsupported orientation components. The protocol itself
does not enforce these obligations at runtime.

`GroundTruth3D` remains a distinct source-data model: it has no prediction score
and can retain bottom-center coordinates with an explicit position reference.
Do not treat KITTI camera annotations as detector output without conversion. See
[annotation](kitti-annotations.md) and [calibration](kitti-calibration.md) conventions.

## CPU-only testing

[`tests/detector_doubles.py`](../tests/detector_doubles.py) contains `FixedDetector`.
It uses structural typing, without inheriting from the protocol. It stores a
predefined iterable of validated boxes as a tuple, returns a fresh list on every
call, supports empty results, and rejects frame mismatches. Mutating a caller's
configuration/result list does not alter subsequent results. It neither examines
the points to infer objects nor transforms/relabels boxes; it is testing
infrastructure, not a production detector.

For pytest modules in `tests/` (using the repository's default pytest import mode):

```python
from detector_doubles import FixedDetector
from raven.perception.detection.base import Detection3D, ObjectDetector

box = Detection3D((1, 2, 3), (4, 2, 1), 0, "Car", 0.9, "kitti_velodyne")
detector: ObjectDetector = FixedDetector([box])
# Given a PointCloud named cloud with coordinate_frame="kitti_velodyne":
# predictions = detector.detect(cloud)
```

No `@runtime_checkable` decorator is needed: consumers call `detect()` through
the typed protocol. The tests exercise behavior through that API. Python does
not enforce the annotations; this is not a static type-checker run, and runtime
structural membership tests would not validate signatures or geometry anyway.

```bash
python -m pytest -q tests/test_detector_contract.py tests/test_detection_models.py
python -m pytest -q
python -m ruff check .
python -m pip check
git diff --check
```

The new contract tests cover multiple/empty results, confidence endpoints,
metadata, input preservation, repeatable independent result lists, frame errors,
legacy positional construction and CPU-only imports. Existing model tests cover
invalid confidence, nonfinite values, malformed coordinates, invalid dimensions,
text validation, immutability and ground-truth conventions. A fresh interpreter
blocks imports of torch, OpenPCDet/pcdet, spconv/cumm and Open3D, then imports the
normal Raven package/models and runs the test double. No KITTI downloads, GPU,
display, new runtime dependencies or CI changes are required.

## Future PointPillars implementation

RV-022 will implement this same `detect(cloud)` method around an external backend.
It will own model preprocessing/inference and return standardized boxes in the
input frame; RV-023 prediction conversion and RV-024 reference comparison remain
separate work. No PointPillars adapter, prediction overlay or new GPU inference
code is implemented by RV-021. The standalone
[Kaggle feasibility runner](pointpillars-feasibility.md) is not this adapter.

CUDA, torch, spconv and OpenPCDet belong in a separately provisioned optional GPU
environment. Importing the core contract or running CPU tests must never import
those frameworks or Open3D. Backend-specific installation and device management
must stay outside the shared protocol and standard CPU installation.
