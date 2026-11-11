# Raven

Modular cooperative LiDAR perception system for connected vehicles, combining deep learning, sensor fusion, and distributed communication.

## Current focus: Raven Vision

Phase 1 explores how LiDAR preprocessing affects pretrained 3D object detection on KITTI. The roadmap covers data ingestion, preprocessing, PointPillars integration, visualization, reproducible experiments, and academic delivery.

This repository currently contains project planning documentation. The existing `raven_vision_starter` implementation on the developer's MacBook has not yet been imported or verified here. No inference results or release are available.

## Roadmap

| Milestone | Focus |
| --- | --- |
| M0 | Project foundation and feasibility |
| M1 | KITTI data integration |
| M2 | Point cloud processing |
| M3 | AI object detection |
| M4 | 3D visualization |
| M5 | Experiment framework |
| M6 | Research and benchmarking |
| M7 | Academic delivery and release |

See [development plan](docs/development-plan.md) and [issues](https://github.com/filip-labs/Raven/issues). RV identifiers are stable planning IDs, not GitHub issue numbers.

## Development environment

The initial local target is Python 3.11 on Apple Silicon. The reported local toolset includes NumPy, pytest, Ruff, and Open3D 0.20.0. Exact dependency versions and install commands will be taken from the actual package configuration after import.

The M1 Pro has no CUDA support. Validate an external GPU inference environment, such as Kaggle, before choosing a PointPillars implementation or adding CUDA dependencies. Ordinary pull-request checks must run without GPU inference.

## Future direction

- **Raven Cooperative Intelligence:** Applied Machine Learning project using OPV2V, coordinate alignment, late fusion, Logistic Regression, Random Forest, and MLP.
- **Raven Connected:** IoT project with simulated vehicles, MQTT detection exchange, validation, authentication, latency, and packet-loss experiments.
- **Master thesis:** integrated, communication-aware cooperative perception and adaptive exchange under network constraints.

These phases are future work; current development is limited to Raven Vision.

## Data and licensing

Datasets, pretrained weights, virtual environments, credentials, and generated experiment artifacts must remain outside version control. Follow each dataset and model provider's terms. No project license has been selected; public visibility alone does not grant an open-source license.

## Author

Filip Cvetković · [Portfolio](https://www.filipzcvetkovic.com)
