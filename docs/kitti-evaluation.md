# KITTI 3D evaluation (RV-032)

This is a CPU integration of the **pinned OpenPCDet KITTI protocol**, consuming
completed RV-031 runs. It does not rerun inference or introduce a second runner.
It reports 3D AP_R40 and optional BEV AP_R40 for Car, Pedestrian and Cyclist at
Easy, Moderate and Hard difficulty. Values are percentages. AP11, 2D AP, AOS,
training, latency benchmarks and general CSV export are not implemented here.

## Backend, provenance and licenses

Matching, cleaning/ignore logic, threshold sampling, precision envelopes, 3D
vertical intersection and AP_R40 come from
[OpenPCDet kitti_object_eval_python](https://github.com/open-mmlab/OpenPCDet/tree/233f849829b6ac19afb8af8837a0246890908755/pcdet/datasets/kitti/kitti_object_eval_python),
revision **233f849829b6ac19afb8af8837a0246890908755**. Three camera-conversion/
projection functions come from that revision's `pcdet/utils/box_utils.py`.
Bundled source files remain byte-for-byte unchanged, with SHA256 checks and
original MIT/Apache-2.0 license/attribution files under `evaluation/_vendor`.
Only checksum-pinned function ASTs are loaded; imports and JIT decorators are
omitted. No caller-provided source is executed and no source is fetched at runtime.
The adapter version is **1**; results contain evaluator name, revision, source
checksums, protocol parameters, adapter code hashes and execution environment.

The overlap backend is **Shapely 2.1.2 / GEOS**, using float64 oriented rectangle
polygon intersection, followed by the reference's vertical-intersection formula.
This is not axis-aligned, voxel or raster IoU. Shapely is BSD-3-Clause; its GEOS
dependency is LGPL-2.1 ([package license information](https://pypi.org/project/shapely/2.1.2/)).
The local verified wheel uses GEOS 3.13.1; actual Shapely/GEOS versions are recorded.
Shapely is pinned; other installed versions or its absence produce blocked
backend availability rather than silently selecting another algorithm.

```bash
python -m pip install -e '.[evaluation]'
# Development tests include the same pinned CPU geometry package:
python -m pip install -e '.[dev]'
```

NumPy remains the sole core dependency. Evaluation needs no CUDA, numba, torch,
OpenPCDet installation, spconv, SciPy or Open3D. The vendored rotate_iou.py is an
unchanged audit/reference source, not a runtime dependency. Direct CPU execution
of its CUDA-device helper bodies failed coincident rotated-box invariance tests
and was rejected; no such fallback remains. Float64 GEOS results are **not
claimed CUDA-bitwise-equivalent**, especially near strict IoU thresholds.

These are reference-implementation metrics, **not certification of official KITTI
server equivalence**. The [official benchmark description](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)
defines class overlaps, difficulty categories and the 40-recall-position protocol.
The selected implementation's exact edge/ignore behavior is specified below;
full server/GPU reference equivalence and a full real validation benchmark have
not been executed in this task.

## Exact selected protocol

| Parameter | Car | Pedestrian | Cyclist |
| --- | ---: | ---: | ---: |
| Strict 3D IoU threshold | 0.7 | 0.5 | 0.5 |
| Strict BEV IoU threshold | 0.7 | 0.5 | 0.5 |

Matching requires **IoU > threshold**, not >=. No epsilon is added to the matching
gate. The reference selects confidence thresholds from true-positive scores,
computes TP/FP/FN at those thresholds, applies the rightward maximum precision
envelope, and averages precision bins **1 through 40**, excluding recall bin 0.
The loose alternative overlap set and COCO-style averaging are not requested.
All detections retain their original confidence; no extra score filter is added.

| Difficulty | GT image-box height | DT image-box height | Max occlusion | Max truncation |
| --- | --- | --- | ---: | ---: |
| Easy | >40 px | >=40 px | 0 | 0.15 |
| Moderate | >25 px | >=25 px | 1 | 0.30 |
| Hard | >25 px | >=25 px | 2 | 0.50 |

The GT/DT boundary asymmetry above is the pinned implementation's actual code.
Unknown required GT visibility (`-1` occlusion/truncation or unavailable bbox)
blocks evaluation; it is not fabricated as fully visible. Alpha is not needed
because AOS is not reported. Class/difficulty-ignored GT can consume a matching
DT without counting TP/FP. Van is ignored when evaluating Car; Person_sitting is
ignored when evaluating Pedestrian. Other classes are excluded as in the source.

**DontCare nuance:** the pinned `compute_statistics_jit` suppresses detections
against 2D DontCare boxes only for metric 0 (2D). It does **not** apply that
suppression to metric 1/2 (BEV/3D). This differs from the general wording of the
benchmark page; do not call this implementation an independently certified server
replica. We preserve/test the selected reference behavior rather than inventing
3D DontCare boxes or quietly applying a different ignore rule. Original DontCare
records remain intact. Nonpositive 3D sentinel dimensions have unused zero overlap
cells; those GT are excluded by the reference class/ignore rules.

Classes/difficulties with **no eligible GT** return null AP and
`availability: no_eligible_ground_truth`, not an invented zero. Valid empty DT
with eligible GT gives zero. With fewer than 41 eligible objects the reference
threshold sampling can leave recall bins unfilled: even a single perfect match
can have zero AP_R40. Such outcomes receive a warning and must not be promoted
as representative accuracy. Synthetic reference fixtures use 41 GT to populate
all bins.

## Coordinates and real image metadata

Raven detections are geometric-center `(x,y,z,dx,dy,dz,heading)` in KITTI Velodyne,
L/W/H metres and +Z radians. The adapter uses the **exact pinned KITTI upright
parameter conversion**, not arbitrary SO(3) box compression:

1. Subtract half height along LiDAR Z to obtain its bottom center.
2. Transform that point using Raven's complete `R0_rect_h @ Tr_velo_to_cam_h`,
   including rotation and translation.
3. Reorder dimensions to camera **L/H/W**, and use reference
   `rotation_y = -heading - pi/2`.
4. Project all eight benchmark-upright camera corners through genuine P2, using
   the pinned projection function, and clip to the actual image's width/height.

KITTI camera GT already uses bottom centers, H/W/L in the source annotation and
rotation about camera Y. Its original location/yaw are not rewritten; only array
dimension ordering changes for the evaluator. Existing calibration loaders and
`transform_points` are reused. Source detections/annotations remain unchanged.

The reference's upright yaw convention does not represent a general tilted box.
Full SO(3) geometry is still available in Raven's existing calibration utilities,
but it is not the KITTI evaluator's seven-parameter model. We record this
limitation and actual calibration-axis deviation. Calibrations more than 0.05 rad
from the expected KITTI sensor/camera axis convention are rejected. There is no
silent general-orientation approximation or claim that all transformed corners
of a tilted physical box equal an upright benchmark box.

P2 must be genuine and use the KITTI depth convention. Nonzero P2[2,3] is retained;
the pinned OpenPCDet projection divides image coordinates by rectified point Z
and subtracts P2[2,3] for the returned depth. It must not be silently replaced by
a different homogeneous-depth formula. Undefined projections fail explicitly.
Calibration calculations use Raven float64 matrices; reference corner construction
retains the pinned function's float32 operations. Fixture tolerances are 1e-12 m
for the independent center calculation, 1e-4 px for projected corner bounds, and
1e-10 for float64 oriented IoU; AP fixture tolerance is 1e-8 percentage points.

Nonempty predictions require an actual matching
`SPLIT/image_2/FRAME.png`. PNG dimensions are read from its IHDR; chunk structure,
checksums and IDAT/IEND presence are checked, and the complete image file is
hashed. This reads metadata, not image pixels or semantic visibility. It is not
proof of dataset authenticity. No fixed KITTI resolution, inferred bbox or
invented image metadata is substituted. For an empty DT set, dimensions are
explicitly not required; no image shape is invented. Genuine P2 remains required.

## Completed-run validation and data policy

`load_completed_run()` requires successful run-schema-v1 completion, verifies all
referenced artifact checksums and safe local filenames, and checks config digest,
selected IDs, summary, result/provenance schemas and frame sets. Failed,
incomplete, duplicate, mismatched or corrupted evidence is rejected.

Evaluation then uses the same public `load_experiment_frame()` input-loading
boundary as RV-031, **without executing its preprocessing/model stages**. It
rechecks actual scans, GT, calibration and saved source captures against recorded
hashes/paths/sizes, verifies persisted GT/calibration/detection consistency, and
joins records by exact frame_id. Reordering the result array cannot cross-match
frames. The saved-source no-new-preprocessing rule remains enforced, including
identity processed-data hashes and counts.

Canonical split membership is reverified using existing pinned train/val manifests
and must agree with the run's split evidence. Normal evaluation requires val.
A subset is labeled `validation_subset`; only all canonical val IDs yield
`validation_full`. This scope label does not authenticate the dataset distribution
or external model. Exact frame IDs/counts and eligible-object counts are recorded.
Unlabeled test cannot provide measured AP and is blocked. All mandatory inputs
must be present; no missing prediction becomes an empty result.

Saved predictions must have a verified source scan-hash binding. Missing binding
is blocked, not silently accepted as matching data. Their GPU execution/model
metadata remains declared, not independently authenticated, and is warned about.
Live results use the recorded processed-cloud execution binding; explicitly mocked
backend evidence is blocked. Artifact hashes ensure consistency, not signatures
or independent proof of who ran a model.

## CLI and persistence

```bash
# Completed validation run; default emits PREFIX.evaluation.json next to manifest.
raven evaluate-experiment /path/to/baseline.manifest.json

# Alternative new evidence path; its parent must exist.
raven evaluate-experiment /path/to/baseline.manifest.json --output /path/to/evaluation.json

# Omit optional BEV rows.
raven evaluate-experiment /path/to/baseline.manifest.json --no-bev

# Explicit development-only training scope; still requires genuine metadata.
raven evaluate-experiment /path/to/training.manifest.json --allow-training-smoke
```

Exit 0 means successful computation, 2 means invalid input/output, 3 means blocked
metadata/scope/backend availability. The JSON console summary prints purpose,
frame IDs/count, metrics, warnings, failures and output location. Default evidence
is written only on success. An explicit --output can also retain a blocked/invalid
outcome with **no metrics**. Existing files/symlinks are never overwritten; the
existing exclusive atomic publisher is reused. Source artifacts/manifests are
not rewritten or added to a second runner. This is minimal JSON evidence, not
RV-034's export framework. Filesystem hard-link support is required.

## Typed result schema

`EvaluationResult` and `MetricResult` provide detached `to_dict()` serialization.

| Field | Content |
| --- | --- |
| evaluation_schema_version | 1 |
| status / purpose | success, invalid or blocked; validation_full/subset, training_smoke or unavailable |
| experiment_identity | Source experiment name/ID, artifact prefix and manifest location |
| config_sha256 / run_manifest_sha256 | Original configuration identity and exact source manifest checksum |
| evaluator | Reference revision/source hashes, adapter version, explicit protocol/difficulty/IoU parameters and tolerances |
| split / frame_ids / evaluated_frame_count | Dataset/split evidence, exact joined IDs, number actually evaluated; blocked means zero |
| classes / difficulties | Car/Pedestrian/Cyclist and easy/moderate/hard |
| metrics | Metric/class/difficulty, threshold, eligible GT, availability, nullable AP_R40 percent |
| diagnostics | Per-frame TP/FP/FN at score >=0, separate from confidence-sampled AP |
| warnings / failures | Scope, data/backend availability, unverified provenance and validation failures |
| provenance | Exact evaluated input/image identities, conversion policy, source-run provenance, actual geometry engine versions, adapter/module hashes, source environment and flags |

The reference does not expose assigned detection indices or per-assignment IoU.
Diagnostics explicitly state `matching_details: not_exposed_by_reference`; no
independent lookalike matcher is presented as the AP assignment. Original scores
and boxes remain available through the verified source RV-023 documents.
Diagnostics at score zero can count low-ranked duplicate FP even when AP stays
100%; these counts are not substituted for aggregate AP calculations.

## Independent verification and reproducible fixtures

`tests/fixtures/kitti_evaluation_reference.json` contains **synthetic**, independently
derived expectations, not real KITTI accuracy. `tests/reference/generate_kitti_reference.py`
loads the pinned upstream Python evaluator directly, omits decorators and supplies
closed-form axis-aligned 3D overlaps. It does not import Raven's evaluator, GEOS,
numba or CUDA. Its frozen independent reference execution is reproducible:

```bash
python tests/reference/generate_kitti_reference.py /new/path/reference.json
```

| 41-frame controlled case | Upstream reference / Raven CPU AP_R40 (%) |
| --- | ---: |
| One exact matched GT/DT per frame | 100 |
| One higher-ranked nonoverlapping FP plus TP per frame | 50 |
| Wrong class only | 0 for the class with GT |
| Empty predictions | 0 for the class with GT |

These follow independent precision/recall calculations (all 40 bins populated,
precision 1 or 1/2). Other tests cover all class-specific thresholds, duplicates,
strict IoU/difficulty boundaries, Van/Person_sitting/DontCare policies, empty GT,
multiple frames, reversed result ordering, unavailable visibility/P2/images,
corrupted artifacts and output collisions. Rotated IoU is checked against closed
forms and an independent float64 Sutherland-Hodgman/shoelace oracle over 100 seeded
pairs, plus coincident rotation/translation invariance. No geometry test merely
compares GEOS against GEOS. Coordinate tests use independent full matrix and
corner projection calculations. A fresh subprocess blocks all GPU/JIT/GUI imports.

These validate selected numerical/protocol cases. The independent reference run
uses analytic overlaps, **not the original CUDA kernel**, and does not establish
all-case GPU/server bitwise agreement. Near strict thresholds, float64 GEOS and
float32 CUDA arithmetic can differ; thresholds are never relaxed to force agreement.

## Bounded real-data check and remaining blockers

The existing completed training run for frame 000000 was integrity-checked:
115,384 scan points, 29 saved predictions and one GT object. All 29 benchmark
camera conversions were finite using full calibration; actual axis deviation was
about 0.01398 rad. Scan, labels, calibration, capture and original run artifacts
were preserved. This is a real-data **input/conversion smoke check**, not new
inference, a GPU comparison or representative validation accuracy.

Actual evaluation with --allow-training-smoke returned **blocked** because the
local KITTI root lacks image_2/000000.png. No image size or AP value was invented.
The full real validation benchmark is also blocked: a complete authentic validation
prediction set and required image evidence were not available. Only the existing
training-frame capture was used. No images, datasets or weights were downloaded.
Ignored local evidence is under `outputs/rv032`, including real-smoke.json and the
explicit blocked result; none of these generated outputs is committed.

## Future GPU predictions, RV-033 and RV-034

Approved cloud inference should export authentic schema-v1 captures with input
hashes, frame IDs and full model/execution provenance. On the local dataset,
create an existing RV-031 **saved** validation run using those frame-matched files
and the same canonical split. Then evaluate its completed manifest with this exact
protocol. This neither reruns inference nor claims the local saved baseline was
a new GPU run. Keep the original remote logs/run artifacts for provenance.
That local saved-import workflow applies to raw scan-bound baseline captures.
Preprocessed live experiments must use their genuine completed-run evidence; they
cannot be relabeled as raw saved predictions. Evaluate those runs where their
recorded source inputs are accessible. No worker/API integration or remote-path
remapping is added here. Relocating an archived live run to a different dataset
root is a future input-location integration step, not a change to metrics logic.

RV-033 can join timing evidence by original config digest, source manifest hash
and exact IDs; evaluator wall time is not detector latency. RV-034 can consume the
typed schema without turning null/unavailable metrics into zeros or mixing training
smoke with validation results. Changes to protocols/dependencies require new
reference validation/versioning, not silently reusing this adapter's identity.

## Remote execution evidence (RV-049)

Schema-v2 Modal runs use this same evaluator and unchanged protocol. The input
boundary additionally verifies the remote response, worker/source/model identity
and binding to the exact experiment configuration, preprocessing digest, frame,
processed point count and XYZI hash. Evaluation makes no remote call and does not
rerun preprocessing. Hashes are consistency evidence, not independent hardware
attestation. Required genuine calibration/images/splits and all availability rules
above still apply. See the [Modal guide](modal-inference.md).
