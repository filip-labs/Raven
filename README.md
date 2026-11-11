# Raven

Raven is a modular cooperative LiDAR perception research project for connected vehicles. Its long-term direction combines deep learning, sensor fusion, and distributed communication.

## Current phase: Raven Vision

Phase 1 supports the academic module *Napredni inteligentni sistemi* (Advanced Intelligent Systems). The current Python starter implements:

- KITTI Velodyne `.bin` loading with file-size validation and explicit coordinate-frame metadata.
- A dataset-neutral `PointCloud` model with validated finite XYZ/intensity values and a read-only array copy.
- Spatial range filtering that preserves intensity and frame metadata.
- The `raven inspect` CLI for frame summaries.
- Automated unit tests and CPU-only Python CI.

PointPillars inference, object detection, GPU processing, 3D visualization, KITTI annotation parsing, calibration transforms, and scientific benchmarking are planned. The existing detection data model and protocol only define an interface; no detection backend is implemented. No inference results or release are available.

## Installation

Python 3.11 is the development and CI target. The package declares Python 3.10 or newer; other versions have not been validated here. From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

NumPy is the only required runtime dependency. Open3D is optional and is not needed for the CLI, tests, or CI. To install it for future visualization work:

```bash
python -m pip install -e ".[visualization]"
```

The local target is Apple Silicon. The M1 Pro has no CUDA support. Validate an external GPU environment, such as Kaggle, and its framework/weight compatibility before selecting a PointPillars backend or adding CUDA dependencies.

## CLI usage

Obtain a KITTI Velodyne frame through the official [KITTI 3D Object Detection benchmark](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) and store it locally under the ignored `data/` directory:

```bash
raven --help
raven inspect data/kitti/training/velodyne/000000.bin
```

The command prints JSON containing the frame ID, coordinate frame, point count, and minimum/maximum XYZ bounds in the source sensor frame. Empty frames report null bounds. It does not run inference or display a 3D scene.

## Testing

```bash
python -m pytest -q
python -m ruff check .
```

Tests generate small synthetic point clouds in temporary directories and require no dataset downloads, pretrained weights, Open3D, or GPU. GitHub Actions runs installation, Ruff, and pytest with Python 3.11 on pull requests and pushes to `main`.

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
