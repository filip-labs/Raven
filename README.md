# Raven

Raven is a modular cooperative LiDAR perception research project for connected vehicles. Its long-term direction combines deep learning, sensor fusion, and distributed communication.

## Current phase: Raven Vision

Phase 1 supports the academic module *Napredni inteligentni sistemi* (Advanced Intelligent Systems). The current Python starter implements:

- KITTI Velodyne `.bin` loading with file-size validation and explicit coordinate-frame metadata.
- A dataset-neutral `PointCloud` model with validated finite XYZ/intensity values and a read-only array copy.
- Spatial range filtering that preserves intensity and frame metadata.
- Deterministic CPU voxel downsampling with mean XYZ/intensity and fixed-origin cells.
- Statistical outlier removal with an optional CPU SciPy KD-tree backend.
- Seeded, NumPy-only Gaussian XYZ noise with exact intensity preservation.
- Ordered preprocessing pipelines with full prevalidation and per-stage provenance.
- The `raven inspect` CLI for frame summaries.
- KITTI calibration parsing and reversible point/full-pose box transformations.
- Validated detection/ground-truth boxes and KITTI `label_2` parsing with source metadata.
- An optional interactive Open3D viewer with selectable point coloring, calibrated KITTI ground truth and saved-prediction overlays.
- An optional reusable PointPillars detector adapter with lazy external GPU dependencies (Kaggle T4 execution reported).
- Automated unit tests and CPU-only Python CI.

A user-verified Kaggle experiment executed pretrained PointPillars inference on
real KITTI data: 29 predictions from frame 000000. This demonstrates external GPU
feasibility, not measured accuracy. The reusable adapter also has a
[successful Kaggle T4 execution report](https://github.com/filip-labs/Raven/pull/58#issuecomment-6839286530).
See [PointPillars feasibility and reproduction](docs/pointpillars-feasibility.md)
for the reported environment, independently checked input/checkpoint hashes, and
remaining reproduction gaps. The headless experiment runner also completed a
[successful Kaggle T4 run](https://github.com/filip-labs/Raven/pull/54#issuecomment-6808816795)
with matching prediction counts. The reusable adapter now has CPU-tested lifecycle and contract
behavior and reported T4 execution. [PR #60](https://github.com/filip-labs/Raven/pull/60#issuecomment-6868499971)
records zero numerical differences for all 29 predictions on the tested frame.
Detection accuracy remains unevaluated. Broader validation, scientific benchmarking
and a release remain pending.

## Installation

Python 3.11 is the development and CI target. The package declares Python 3.10 or newer; other versions have not been validated here. From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

NumPy is the only required runtime dependency. Open3D is optional and is not needed for help, inspection, tests, or CI. To install the viewer together with development tools:

```bash
python -m pip install -e ".[dev,visualization]"
```

The local target is Apple Silicon. The M1 Pro has no CUDA support. GPU-specific
packages stay in a separate external environment: no CUDA, torch, spconv, or
OpenPCDet dependency is added to the standard Raven installation or CPU CI.
The reported Kaggle T4/OpenPCDet combination is evidence for that experiment;
compatibility with other GPU images is not guaranteed.

## CLI usage

Obtain a KITTI Velodyne frame through the official [KITTI 3D Object Detection benchmark](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) and store it locally under the ignored `data/` directory:

```bash
raven --help
raven inspect data/kitti/training/velodyne/000000.bin
```

The command prints JSON containing the frame ID, coordinate frame, point count, and minimum/maximum XYZ bounds in the source sensor frame. Empty frames report null bounds. It does not run inference or display a 3D scene.

See the [KITTI dataset policy](docs/kitti-dataset.md) for official access and
license terms, pinned 3,712/3,769 train/validation memberships, read-only integrity
checks, and pretrained-model leakage limitations.

## KITTI annotation inspection

```bash
raven inspect-labels data/kitti/training/label_2/000000.txt
```

The JSON summary reports the frame, object count, counts by class, ignored-region
count, unknown classes, and `kitti_rectified_camera` coordinate frame. No Open3D
or graphical display is required. Empty files produce zero counts.

Annotations retain KITTI's **bottom-center** camera position, dimensions in
height/width/length order, and rotation about camera Y. The associated ground-truth
box explicitly stores `position_reference="bottom_center"`, `rotation_axis="y"`,
and dimensions reordered to length/width/height. It does not shift positions,
convert to LiDAR, or reinterpret camera rotation as LiDAR yaw. `DontCare` records
remain ignore metadata without a valid 3D box; unknown classes are preserved.

See [annotation format and model conventions](docs/kitti-annotations.md) for the
reusable API, validation rules, optional score handling, and current limitations.

## KITTI calibration and coordinate transformations

```bash
raven inspect-calib data/kitti/training/calib/000000.txt
```

Calibration parsing and point/box conversions use NumPy, without Open3D or a GUI.
The command reports available matrix shapes, the Velodyne-to-rectified-camera
homogeneous transform, frame names, and invertibility. Transformed box geometry
preserves all three orientation axes and source metadata. The viewer can display
this geometry with `--ground-truth`. The existing annotation frame name `kitti_rectified_camera`
is accepted as an alias for `kitti_camera_rect`, without changing source records.

See [calibration and transform conventions](docs/kitti-calibration.md) for the
reusable API, numerical example, corner ordering, and precision limits.

## Range filtering

`filter_range(cloud, x_range, y_range, z_range)` performs an explicit CPU-only,
inclusive XYZ crop, preserving point order, intensity and frame metadata in an
independent PointCloud. Empty results are valid; malformed/nonfinite/reversed
bounds are rejected. See [range filtering semantics and examples](docs/range-filtering.md)
for precision, coordinate conventions and future preprocessing experiment limits.

## Voxel downsampling

`voxel_downsample(cloud, voxel_size)` supports cubic or per-axis sizes, using
zero-origin floor-based cells and mean XYZI representatives. Output cells and
aggregation order are deterministic; metadata is preserved in an independent
PointCloud. See [voxel semantics, precision and KITTI examples](docs/voxel-downsampling.md)
for lossy behavior and limits. This is explicit Raven preprocessing, separate
from OpenPCDet's internal voxel processing and future RV-017 pipelines.

## Statistical outlier removal

`statistical_outlier_removal(cloud, nb_neighbors, std_ratio)` retains original
XYZI rows whose mean K-neighbor distance is at or below the global mean plus
`std_ratio` population standard deviations. It preserves order and metadata;
K excludes self and is capped at N-1. Install `python -m pip install -e ".[preprocessing]"`
for the optional CPU SciPy backend. See [the statistical filtering guide](docs/statistical-outlier-removal.md)
for small-cloud policy, duplicate handling, precision, resource costs, KITTI
examples and observed smoke-test counts. Sparse valid objects can be removed;
no accuracy or inference-latency improvement is claimed.

## Controlled noise injection

`inject_noise(cloud, position_std, seed)` adds independent Gaussian XYZ noise
using a local, explicitly seeded PCG64 generator. Standard deviation is in the
caller's spatial units (meters for KITTI); intensity, count, row order and metadata
are preserved. Zero noise returns an exact independent copy. Float64 addition
and strict float32 range validation prevent silent overflow or clipping.
See [the controlled-noise guide](docs/controlled-noise.md) for parameter/seed
policy, reproducibility limits, KITTI smoke statistics and stage composition.
This is a simplified synthetic error model, without demonstrated accuracy or
latency effects.

## Preprocessing pipeline

Compose existing operations in an explicit order, with no hidden defaults:

```python
from raven.perception.preprocessing.pipeline import run_preprocessing_pipeline

result = run_preprocessing_pipeline(cloud, [
    {"name": "voxel_downsample", "voxel_size": 0.2},
    {"name": "noise", "position_std": 0.01, "seed": 42},
])
processed_cloud = result.cloud
metadata = result.to_metadata_dict()
```

All configurations are validated before the first stage. Repeated stages and an
empty identity pipeline are supported. Metadata records ordered parameters,
point counts, frames, stage/total durations and software versions. See the
[pipeline guide](docs/preprocessing-pipeline.md) for all four canonical stage
names, error behavior, optional SciPy dependency and read-only KITTI examples.
No inference runs automatically, and timings establish no accuracy/latency gains.

## Detector interface

The model-independent `ObjectDetector.detect(cloud) -> list[Detection3D]` contract
and a reusable CPU-only test double are documented in the
[detector developer guide](docs/object-detector.md). Backends must preserve the
input and return geometric-center detections in its coordinate frame. The
[optional PointPillars adapter](docs/pointpillars-adapter.md) implements this API
with a model reused across scans and reported T4 execution. Standard Raven imports
require no GPU packages. See [prediction conversion and JSON](docs/prediction-conversion.md)
for CPU-only export/import of validated detections, schema checks and precision limits.
The [numerical reference validation guide](docs/reference-validation.md) provides
independent OpenPCDet/Raven capture commands and a CPU-only, provenance-gated
comparison CLI. The real T4 comparison passed for the reported frame 000000; other samples remain unverified.

## Interactive visualization

With the visualization extra installed, run:

```bash
raven visualize data/kitti/training/velodyne/000000.bin
raven visualize data/kitti/training/velodyne/000000.bin --color intensity
raven visualize data/kitti/training/velodyne/000000.bin --color uniform --point-size 3.0
raven visualize data/kitti/training/velodyne/000000.bin --ground-truth
raven visualize data/kitti/training/velodyne/000001.bin --ground-truth --color uniform
raven visualize data/kitti/training/velodyne/000000.bin --predictions /path/to/raven_adapter_predictions_000000.json
raven visualize data/kitti/training/velodyne/000000.bin --ground-truth --predictions /path/to/raven_adapter_predictions_000000.json
raven visualize --help
```

The default is `--color distance --point-size 2.0`. Point size accepts finite values
from 1 to 25 pixels. The viewer shows a two-meter coordinate frame at the sensor
origin and includes the frame ID and coordinate frame in the window title.

- **Distance:** Euclidean distance from the sensor, scaled from zero to the current
  frame's farthest point. Near points are green, far points red, and blue is fixed
  at 0.2. The scale is per frame, so colors are not absolute distances across scans.
- **Intensity:** grayscale using the original reflectance, clipped to [0, 1].
  Constant intensities retain their brightness without normalization or division.
- **Uniform:** neutral gray for every point.

With `--ground-truth`, the viewer loads matching sibling `label_2/FRAME.txt` and
`calib/FRAME.txt` files, transforms camera bottom-center boxes into the cloud's
Velodyne frame, and draws their full-orientation corners with twelve edges each.
Car is green, Pedestrian red, Cyclist blue, and all other classes yellow.
`DontCare` regions are skipped. Point colors are dimmed to 35% while either
overlay has visible boxes; hiding or filtering all boxes restores the selected
point colors.

For nonstandard layouts, supply both `--labels PATH` and `--calib PATH` together
with `--ground-truth`. Scan, label, and calibration filename stems must match;
missing files or mismatches produce errors before opening a window. See the
[visualization guide](docs/visualization.md) for explicit-path examples and the
reusable wireframe API. These boxes are dataset ground truth, not model predictions.

Coloring does not alter the original XYZ, intensity, or coordinate metadata.
Open3D is imported only when a nonempty cloud is displayed; help, inspection, and
unit tests work without it. Missing dependencies, invalid files, empty clouds,
and failed window creation produce CLI errors. Empty frames remain valid for
`raven inspect`.

This is a single-frame desktop viewer requiring a graphical session and OpenGL.
Headless rendering is not supported. Rendering requires Open3D geometry and RGB
buffers in addition to the original cloud. Saved schema-v1 predictions can be
loaded with `--predictions PATH` without GPU inference packages. Predictions are cyan dashed boxes with heading glyphs; GT
retains its class colors. F1 shows help, F2/F3 toggle layers, F4 toggles both,
F5/F6/F7 select Car/Pedestrian/Cyclist, F8 restores all classes, and F9 prints
visible confidence values to the terminal. See the [overlay guide](docs/visualization.md)
for frame validation and graphical-test limits. Live detection, 3D text labels
and frame navigation are not implemented. Use the window close control
to finish the command; Open3D's built-in navigation controls are not customized.

## Testing

```bash
python -m pytest -q
python -m ruff check .
```

Tests generate small synthetic point clouds in temporary directories and require no dataset downloads, pretrained weights, Open3D, display, or GPU. Rendering is mocked in unit tests; an optional smoke test checks loading and coloring a real local KITTI frame when available and skips when absent. GitHub Actions runs installation, Ruff, and pytest with Python 3.11 on pull requests and pushes to `main`.

See [M2 preprocessing regression coverage](docs/preprocessing-testing.md) for the
coverage audit, independent four-stage references across all 24 stage orders,
float32 boundary regressions, optional KITTI checks and reproducibility limits.

## Academic research objective

**Research question:** How do point-cloud preprocessing and sensor degradation affect 3D detection accuracy and end-to-end latency?

Planned comparisons include unmodified point clouds, intensity-preserving voxel downsampling, statistical outlier removal, and synthetic degradation. The intended evaluation uses held-out labeled KITTI validation frames, a compatible 3D AP implementation, fixed pretrained weights, and separate preprocessing/inference timings. This methodology and the resulting experiments remain future work.

## Roadmap and GitHub planning

See the checked-in [Raven Vision roadmap](docs/ROADMAP.md) for milestone scope and feasibility criteria. Track detailed work in [GitHub issues](https://github.com/filip-labs/Raven/issues) and the proposed [development plan](https://github.com/filip-labs/Raven/blob/docs/vision-development-plan/docs/development-plan.md) on its documentation branch. RV identifiers are stable planning IDs, not GitHub issue numbers. Milestones describe intended outcomes, not completed functionality.

| Milestone | Focus |
| --- | --- |
| M0 | Project foundation and feasibility |
| M1 | KITTI data integration |
| M2 | Point cloud processing |
| M3 | Pretrained 3D object detection |
| M4 | 3D visualization |
| M5 | Reproducible experiment framework |
| M6 | Research and benchmarking |
| M7 | Academic delivery and release |

Long-term phases are planned beyond Raven Vision:

- **Raven Cooperative Intelligence:** Applied Machine Learning work using OPV2V, coordinate alignment, late fusion, Logistic Regression, Random Forest, and MLP.
- **Raven Connected:** IoT work with simulated vehicles, MQTT detection exchange, message validation, authentication, latency, and packet-loss experiments.
- **Master's thesis:** integrated, communication-aware cooperative perception and adaptive exchange under network constraints.

The modular `src/raven/core` and `src/raven/perception` architecture supports later extensions. Cooperative fusion and IoT communication are outside the current phase.

## Data and licensing

Keep datasets, downloaded data, pretrained weights, checkpoints, virtual environments, credentials, and generated experiment artifacts outside version control. The root `data/`, `datasets/`, `weights/`, `checkpoints/`, and output directories are ignored; Python data-loader source remains tracked.

Follow each dataset and pretrained-model provider's license and access terms. Before experiments, record the dataset source, model/weight source, versions, and applicable terms; do not redistribute assets without permission. No project license has been selected. Public visibility alone does not grant an open-source license.

See the [PointPillars checkpoint provenance and rights audit](docs/pointpillars-checkpoint-provenance.md)
for the official source, verified checksum/configuration, and unresolved weight
license scope. Apache-2.0 code and KITTI dataset terms are reviewed separately.

## Author

Filip Cvetković · [Portfolio](https://www.filipzcvetkovic.com)
