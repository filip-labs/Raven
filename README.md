# Raven

Raven is a modular cooperative LiDAR perception research project for connected vehicles. Its long-term direction combines deep learning, sensor fusion, and distributed communication.

## Current phase: Raven Vision

Phase 1 supports the academic module *Napredni inteligentni sistemi* (Advanced Intelligent Systems). The current Python starter implements:

- KITTI Velodyne `.bin` loading with file-size validation and explicit coordinate-frame metadata.
- A dataset-neutral `PointCloud` model with validated finite XYZ/intensity values and a read-only array copy.
- Spatial range filtering that preserves intensity and frame metadata.
- The `raven inspect` CLI for frame summaries.
- KITTI calibration parsing and reversible point/full-pose box transformations.
- Validated detection/ground-truth boxes and KITTI `label_2` parsing with source metadata.
- An optional interactive Open3D viewer with distance, intensity, and uniform coloring.
- Automated unit tests and CPU-only Python CI.

PointPillars inference, object detection, GPU processing and scientific benchmarking are planned. The detection models and protocol define the backend interface; no detection backend is implemented. No inference results or release are available.

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

The local target is Apple Silicon. The M1 Pro has no CUDA support. Validate an external GPU environment, such as Kaggle, and its framework/weight compatibility before selecting a PointPillars backend or adding CUDA dependencies.

## CLI usage

Obtain a KITTI Velodyne frame through the official [KITTI 3D Object Detection benchmark](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) and store it locally under the ignored `data/` directory:

```bash
raven --help
raven inspect data/kitti/training/velodyne/000000.bin
```

The command prints JSON containing the frame ID, coordinate frame, point count, and minimum/maximum XYZ bounds in the source sensor frame. Empty frames report null bounds. It does not run inference or display a 3D scene.

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
preserves all three orientation axes and source metadata; bounding-box rendering
is not implemented. The existing annotation frame name `kitti_rectified_camera`
is accepted as an alias for `kitti_camera_rect`, without changing source records.

See [calibration and transform conventions](docs/kitti-calibration.md) for the
reusable API, numerical example, corner ordering, and precision limits.

## Interactive visualization

With the visualization extra installed, run:

```bash
raven visualize data/kitti/training/velodyne/000000.bin
raven visualize data/kitti/training/velodyne/000000.bin --color intensity
raven visualize data/kitti/training/velodyne/000000.bin --color uniform --point-size 3.0
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

Coloring does not alter the original XYZ, intensity, or coordinate metadata.
Open3D is imported only when a nonempty cloud is displayed; help, inspection, and
unit tests work without it. Missing dependencies, invalid files, empty clouds,
and failed window creation produce CLI errors. Empty frames remain valid for
`raven inspect`.

This is a single-frame desktop viewer requiring a graphical session and OpenGL.
Headless rendering is not supported. Rendering requires Open3D geometry and RGB
buffers in addition to the original cloud. Detection, bounding boxes, ground-truth
overlays, and frame navigation are not implemented. Use the window close control
to finish the command; Open3D's built-in navigation controls are not customized.

## Testing

```bash
python -m pytest -q
python -m ruff check .
```

Tests generate small synthetic point clouds in temporary directories and require no dataset downloads, pretrained weights, Open3D, display, or GPU. Rendering is mocked in unit tests; an optional smoke test checks loading and coloring a real local KITTI frame when available and skips when absent. GitHub Actions runs installation, Ruff, and pytest with Python 3.11 on pull requests and pushes to `main`.

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

## Author

Filip Cvetković · [Portfolio](https://www.filipzcvetkovic.com)
