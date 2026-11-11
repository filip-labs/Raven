# KITTI calibration and full-pose box geometry

This module handles the KITTI **object-detection** calibration format described by
its [object development kit](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d).
It does not implement image projection, raw/odometry calibration formats, inference,
or bounding-box rendering. Existing point-cloud visualization is unchanged.

## Frames and calibration fields

- `kitti_velodyne`: X forward, Y left, Z up.
- `kitti_camera_rect`: rectified reference camera, X right, Y down, Z forward.
- `kitti_rectified_camera`: accepted alias for the existing annotation parser's
  frame name. Source annotations keep this original name; transformed geometry
  uses the canonical names above.

All positions and dimensions are in meters. Calibration is specific to a frame;
load the matching calibration/label/scan filenames. Dataset-specific camera frames
are never treated as already being in Velodyne coordinates.

`load_kitti_calibration(path)` returns `KittiCalibration`. Its read-only `matrices`
mapping retains original matrix shapes and any additional numeric fields:

| Field | Shape | Use |
| --- | --- | --- |
| P0, P1, P2, P3 | 3 × 4 | Optional projection matrices, retained but not used for rigid transforms |
| R0_rect | 3 × 3 | Required reference-camera rectification rotation |
| Tr_velo_to_cam | 3 × 4 | Required Velodyne-to-unrectified-reference-camera rigid transform |
| Tr_imu_to_velo | 3 × 4 | Optional IMU extrinsic, validated and retained; no IMU transform API |
| Additional fields | N | Finite numeric vectors retained under their original keys |

Blank lines are ignored. Missing required matrices, duplicates, malformed names,
wrong dimensions, nonfinite values, reflections, singular rotations, and nonrigid
scaling are rejected. Invalid entries report the filename and physical line.
Projection matrices do not need to be invertible: they are not rigid transforms.
The 3 × 4 extrinsics acquire the homogeneous final row `[0, 0, 0, 1]`; arbitrary
4 × 4 extrinsic fields are rejected rather than silently reinterpreted.

## Point API and equations

```python
from raven.perception.data.kitti_calibration import (
    load_kitti_calibration, transform_points,
)

calibration = load_kitti_calibration("data/kitti/training/calib/000000.txt")
camera_xyz = transform_points(
    lidar_xyz, calibration,
    source_frame="kitti_velodyne", target_frame="kitti_camera_rect",
)
recovered_xyz = transform_points(
    camera_xyz, calibration,
    source_frame="kitti_camera_rect", target_frame="kitti_velodyne",
)
```

For column-vector homogeneous coordinates:

```text
T_velo_to_rect = R0_rect_h @ Tr_velo_to_cam_h
X_rect_h = T_velo_to_rect @ X_velo_h
X_velo_h = inverse(T_velo_to_rect) @ X_rect_h
```

`calibration.velo_to_rect`, `rect_to_velo`, and
`calibration.transform(source_frame, target_frame)` expose the 4 × 4 matrices.
`transform_points` accepts `(3,)` or `(N, 3)`, including `(0, 3)`, preserving
shape and inputs while returning float64 XYZ. The implementation evaluates
`points @ R.T + t`, algebraically equivalent to homogeneous multiplication,
without allocating an extra N × 4 array. A same-frame request returns a copy;
unsupported frame names are rejected even for empty or same-frame inputs.
Pass `cloud.points[:, :3]` for XYZ; intensity is not a coordinate and is neither
transformed nor modified. P2 is never substituted for the rigid transform.

For example, with identity rectification, rotation
`[[0,-1,0], [0,0,-1], [1,0,0]]` and translation `[1,2,3]`, Velodyne
`[10,2,1]` becomes camera `[-1,1,13]`. Applying the inverse restores `[10,2,1]`.
This is an explicit axis mapping encoded in a proper rotation, not a heuristic
coordinate swap performed outside the calibration.

## Box API and geometry

```python
from raven.perception.data.kitti_annotations import load_kitti_labels
from raven.perception.data.kitti_calibration import transform_annotation, transform_box

labels = load_kitti_labels("data/kitti/training/label_2/000000.txt")
for record in labels.records:
    geometry = transform_annotation(record, calibration, target_frame="kitti_velodyne")
    if geometry is None:  # DontCare: retain the original image-space ignore metadata.
        continue
    print(geometry.center_xyz, geometry.rotation, geometry.corners)
    camera_geometry = transform_box(geometry, calibration, target_frame="kitti_camera_rect")
```

`transform_box` also accepts the existing `GroundTruth3D`. It returns
`OrientedBox3D`: a geometric center, length/width/height dimensions, a full 3 × 3
rotation matrix, canonical coordinate frame, and original `source` ground truth.
`transform_annotation` additionally retains the original immutable record in
`geometry.annotation`, including class, source score, truncation, occlusion,
observation angle, image bounds, and raw dimensions/location. The source label is
available as `geometry.label`. DontCare records produce None, never fake boxes.

For KITTI camera boxes, bottom center `(x,y,z)` becomes geometric center
`(x,y-h/2,z)`. A geometric-center input is not shifted. Local length/width/height
axes before rotation are camera `(+X,+Z,-Y)`, a right-handed basis. The complete
camera box orientation is `R_y(rotation_y) @ basis`; it is then left-multiplied
by the calibration rotation into the destination frame. Velodyne ground-truth
boxes use the normal XYZ basis, rotation about Z, and a bottom-to-center offset
of `+h/2` along Z. Other source-axis conventions are rejected explicitly.

Calibration can introduce roll and pitch. The transformed orientation is retained
as a matrix, with columns giving the length, width, and height directions;
there is no lossy conversion to a single destination yaw. `size_lwh` stays in
object-local axes and is not replaced by axis-aligned extents.

Corners are generated from local signs `---, +--, ++-, -+-, --+, +-+, +++, -++`.
Indices 0–3 form the lower local-height face and 4–7 the matching upper face.
Edges from corner 0 to corners 1, 3, and 4 correspond to length, width, and height.
Corner generation and direct point transformation therefore agree, including
under arbitrary calibration rotation and translation.

## Precision and limitations

Calculations use float64 even for float32 point inputs. Rotation validation uses
absolute tolerance 1e-5 for orthonormality and determinant +1, allowing rounded
KITTI text calibrations. Input matrices are not silently normalized or repaired.
The inverse uses a numerical matrix inverse rather than assuming that the
transpose of a rounded rotation is its exact inverse. Rounded calibrations can
produce tiny deviations in rotation determinant or edge lengths; use roughly
1e-5 tolerances when checking geometry. Near-zero synthetic round-trip errors
verify the math, not the physical accuracy of sensor calibration.

Geometry arrays and calibration matrices are immutable defensive copies. Returned
point/corner arrays are independent results. The API does not infer missing
matrices, convert other KITTI formats, rectify images, render boxes, or validate
sensor calibration accuracy. Matching test-split annotations are unavailable
locally and are not fabricated. Tests use small synthetic CPU-only fixtures;
real-data checks can be run separately against the existing local dataset.
