# Optional PointPillars detector (RV-022)

`PointPillarsDetector` implements the existing
[`ObjectDetector`](object-detector.md) API using the OpenPCDet headless workflow.
It is a reusable backend, independent of the single-frame experiment CLI.

**Validation status:** the experimenter reported successful adapter execution on
Kaggle T4 in [PR #58](https://github.com/filip-labs/Raven/pull/58#issuecomment-6839286530):
29 validated detections (Car 4, Pedestrian 19, Cyclist 6), `kitti_velodyne`, exit 0.
The historical standalone runner execution is also recorded in PR #54. Individual
box/score equivalence remains RV-024; accuracy was not measured. RV-023 adds
[CPU-tested conversion and JSON import/export](prediction-conversion.md); the
original GPU capture was unavailable locally and no new GPU run is claimed.

## Environment and assets

The initial external target is the previously reported Kaggle NVIDIA Tesla T4
(compute capability 7.5) stack: Python 3.13.16, PyTorch 2.11.0+cu130, CUDA toolkit
13.0, spconv-cu126 2.3.8 and OpenPCDet 0.6.0+233f849. This is evidence for the
historical run, not a portable installation lockfile or a promise of compatibility
with future Kaggle images. The local CPU suite targets Python 3.11. CPU/macOS
inference, Apple MPS, distributed inference and other model architectures are
unsupported. `device_index` selects one GPU from the visible CUDA devices.

Keep GPU packages in a separate external environment. Raven's normal
`python -m pip install -e '.[dev]'` installs no torch, CUDA, spconv, OpenPCDet or
Open3D. Provision the GPU stack separately using the pinned
[upstream installation guide](https://github.com/open-mmlab/OpenPCDet/blob/233f849829b6ac19afb8af8837a0246890908755/docs/INSTALL.md)
and the [feasibility notes](pointpillars-feasibility.md). Exact successful historical
installation commands and the Kaggle image lock are unavailable; do not treat a
blind version install as independently verified reproduction.

Requirements:

- Trusted OpenPCDet Git checkout at
  `233f849829b6ac19afb8af8837a0246890908755`; other revisions are rejected.
- Matching `tools/cfgs/kitti_models/pointpillar.yaml` and its base YAML files.
  The [pinned model YAML](https://github.com/open-mmlab/OpenPCDet/blob/233f849829b6ac19afb8af8837a0246890908755/tools/cfgs/kitti_models/pointpillar.yaml)
  and [KITTI dataset YAML](https://github.com/open-mmlab/OpenPCDet/blob/233f849829b6ac19afb8af8837a0246890908755/tools/cfgs/dataset_configs/kitti_dataset.yaml)
  define the supported KITTI XYZI setup and class order Car/Pedestrian/Cyclist.
  Incompatible architecture, dataset, feature encoding, processors, class order
  and raw-score configurations fail explicitly. Config contents are checked at
  initialization, when the optional upstream YAML loader is available.
- User-acquired checkpoint, matching that model, with independently checked
  source and SHA256. Reference `pointpillar.pth` SHA256:
  `4c83fc0fa02575b9b3e9dec676f698e7a70bb5a795e89f91df8a96b916fa19e2`.
  [Official model-zoo association and checksum evidence](pointpillars-feasibility.md#evidence-and-its-boundaries)
  do not establish unrestricted weight-use rights. RV-020 remains open on terms.
- Your own licensed KITTI input; see [access and split policy](kitti-dataset.md).
  Do not commit or redistribute weights, scans or generated results.

Set these flags **before imports in a fresh process**:

```bash
export CUMM_DISABLE_JIT=1
export SPCONV_DISABLE_JIT=1
export TORCH_CUDA_ARCH_LIST=7.5
```

Initialization supplies those defaults, rejects conflicting JIT-disable values,
and refuses to set missing flags after spconv/cumm have already been imported.
Restart the process after changing flags. They do not guarantee compatible
prebuilt kernels. A missing `av2` import may require the previously documented,
explicitly reviewed [Argoverse workaround](pointpillars-feasibility.md#optional-argoverse-import)
in your own external OpenPCDet copy. Raven never patches upstream automatically.
Tracked upstream modifications are recorded; retain the full diff and local
installation evidence separately. Do not run untrusted upstream Python code or YAML.

## Reusable API and lifecycle

```python
from raven.perception.data.kitti_loader import load_kitti_bin
from raven.perception.detection.base import ObjectDetector
from raven.perception.detection.pointpillars import PointPillarsDetector

backend = PointPillarsDetector(
    openpcdet_root="/kaggle/working/OpenPCDet",
    config="/kaggle/working/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml",
    checkpoint="/kaggle/working/pointpillar.pth",
    checkpoint_sha256="4c83fc0fa02575b9b3e9dec676f698e7a70bb5a795e89f91df8a96b916fa19e2",
    device_index=0,
    # trust_checkpoint=False is the safe default.
)
detector: ObjectDetector = backend
try:
    # Optional: backend.initialize() loads the model now instead of on first detect.
    cloud = load_kitti_bin("/kaggle/working/000000.bin")  # use your actual input path
    detections = detector.detect(cloud)
    # Reuse the same instance for additional supported clouds.
    evidence = backend.provenance
finally:
    backend.close()
```

Import and construction are CPU-only; construction validates files and hashes.
The first `detect()` (or explicit `initialize()`) rechecks the checkpoint digest,
lazily imports GPU frameworks, validates CUDA/checkout/config, copies upstream
config globals, builds one evaluation model, and strictly loads its full state.
Failed initialization never caches a partial model. Future calls reuse the loaded
model, prepare a copy of XYZI, collate one scan, transfer the batch within the
selected CUDA-device context, and execute under `torch.no_grad()`.

Checkpoint loading uses `weights_only=True` by default, with no unsafe fallback.
If it fails on trusted checkpoint metadata, inspect the error, source and checksum,
then explicitly opt in with `trust_checkpoint=True` / `--trust-checkpoint` only if
appropriate. The digest proves byte identity, not safety or licensing rights.
There is no torch.load monkeypatch, automatic download, or permissive partial load.

`close()` releases owned model/dataset references and makes that instance unusable.
It does not reset CUDA, clear process-wide caches or unload imported modules.
`provenance` returns a defensive copy after successful initialization (also retained
after close). Device contexts and temporary import-path/cwd changes are restored
on success or failure. Raven session operations are serialized because the upstream
YAML loader resolves base paths relative to `tools/`; other threads must not change
process cwd during initialization. This is a synchronous single-scan API.

## Output and conversion boundary

Only nonempty `PointCloud` values explicitly marked `kitti_velodyne` are accepted.
A generic `lidar` frame name is insufficient proof of the coordinate convention.
Wrong frames and empty clouds fail before GPU initialization. The caller's points,
intensity, frame ID and writeability are preserved.

The pinned OpenPCDet outputs `pred_boxes=(x,y,z,dx,dy,dz,heading)`, `pred_scores`
and one-based `pred_labels`. Minimal isolated conversion validates shape, finite
numbers, positive dimensions, probability scores and class indices, then constructs
`Detection3D` with:

- `center_xyz=(x,y,z)`, geometric center in KITTI Velodyne meters;
- `size_lwh=(dx,dy,dz)` in object-local axes;
- `yaw_rad=heading`, radians about LiDAR +Z, `rotation_axis="z"`;
- class label from the configured one-based mapping and unchanged confidence;
- `coordinate_frame="kitti_velodyne"`.

No camera-axis, bottom-center, dimension-order or angle conversion is silently
applied. No extra confidence cutoff is imposed; the configured range filtering,
voxelization, threshold and NMS remain upstream-owned. Zero predictions returns
`[]`. Errors propagate, rather than pretending an inference failure is no detections.

The shared backend intentionally follows `DatasetTemplate.prepare_data`, as the
validated standalone workflow did. It does not invoke `KittiDataset.__getitem__`,
read image/calibration files, or apply image-FOV cropping implied by that dataset's
`FOV_POINTS_ONLY` setting. This is full-scan LiDAR inference with configured range
processing, not a claim of exact KITTI benchmark data preparation. See the
[RV-023 conversion/JSON guide](prediction-conversion.md); RV-024 retains reference
numerical comparisons.

## Kaggle validation procedure

In a GPU session provisioned as above, check out this PR's Raven branch, install
its core, and use your actual licensed input location. The paths below assume
`/kaggle/working/Raven`, `/kaggle/working/OpenPCDet`, `pointpillar.pth` and
`000000.bin`; the scan path is an example, not a recovered notebook path.

```bash
cd /kaggle/working/Raven
git fetch origin
git switch feat/rv-023-prediction-conversion
python -m pip install -e .
git -C /kaggle/working/OpenPCDet rev-parse HEAD
export CUMM_DISABLE_JIT=1
export SPCONV_DISABLE_JIT=1
export TORCH_CUDA_ARCH_LIST=7.5

# Fresh subprocess; validates the exact input and invokes PointPillarsDetector.detect().
python scripts/kaggle_pointpillars.py --adapter \
  --input /kaggle/working/000000.bin \
  --input-sha256 0e09c85e3f6078ecbdd1e706ee9624519f1bd29417437167a9ed7fbe6f54b4b1 \
  --openpcdet-root /kaggle/working/OpenPCDet \
  --config /kaggle/working/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml \
  --checkpoint /kaggle/working/pointpillar.pth \
  --checkpoint-sha256 4c83fc0fa02575b9b3e9dec676f698e7a70bb5a795e89f91df8a96b916fa19e2 \
  --checkpoint-source https://drive.google.com/file/d/1wMxWTpU1qUoY3DsCH31WJmvJxcjFXKlm/view \
  --device-index 0 \
  --output /kaggle/working/raven_adapter_predictions_000000.json
```

Only if the reviewed checkpoint needs trusted pickle metadata, repeat with
`--trust-checkpoint`; do not use it just to suppress an unexplained error.
The runner refuses to overwrite outputs, and writes nothing on inference failure.
Existing CLI flags and raw prediction schema remain available without `--adapter`,
and that path shares the same inference backend.

The adapter-mode JSON records `inference_entrypoint="PointPillarsDetector.detect"`,
serialized `Detection3D` records under `detections`, raw-format `predictions` for
later comparisons, class counts, input/checkpoint/config hashes, effective resolved
config, OpenPCDet revision/tracked modifications, device, versions and JIT flags.
It does not prove that user-supplied source URLs confer usage rights.
Inspect the actual dataclass records and runtime evidence:

```bash
python - <<'PY'
import json
from collections import Counter
from raven.perception.detection.base import Detection3D
p = '/kaggle/working/raven_adapter_predictions_000000.json'
d = json.load(open(p))
assert d['inference_entrypoint'] == 'PointPillarsDetector.detect'
boxes = [Detection3D(**row) for row in d['detections']]
assert all(b.coordinate_frame == 'kitti_velodyne' and b.rotation_axis == 'z' for b in boxes)
print('detections:', len(boxes), 'classes:', dict(Counter(b.label for b in boxes)))
print('runtime:', d['environment'])
print('checkpoint:', d['checkpoint'], 'revision:', d['openpcdet_commit'])
PY
```

Historical reference observation: **29**, Car **4**, Cyclist **6**, Pedestrian
**19** for frame 000000 with 115,384 points. Record actual results, including any
mismatch; do not edit thresholds or hardcode counts to force agreement. Matching
counts alone does not establish box/score equivalence. Frame 000000 is a training
split member, not held-out accuracy evidence. Archive stdout/stderr, JSON, Raven
commit, `pip freeze`, `nvidia-smi`, `nvcc --version`, actual YAMLs/upstream diff and
notebook export privately. The reported PR #58 execution is historical evidence;
no new GPU execution is claimed by the RV-023 conversion update.

## CPU verification

```bash
python -m pytest -q tests/test_pointpillars_adapter.py tests/test_pointpillars_script.py
python -m pytest -q
python -m ruff check .
python -m pip check
git diff --check
```

Synthetic tests exercise construction, lazy imports, typed protocol use, invalid
assets/config/frames, missing dependencies, selected device, strict/trusted loading,
initialization reuse/failure, immutable input, multiple/empty predictions,
serialization through an actual adapter call, JIT flags, and resource handling.
A fresh interpreter blocks GPU/GUI imports during adapter import/construction.
CPU tests do not substitute for per-box reference comparison or a new GPU run.
