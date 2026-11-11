# Raven Vision development plan

Current scope is Phase 1 only. All tasks remain open until repository evidence establishes completion. RV identifiers are planning IDs, not issue numbers.

## GitHub setup pending

The repository is public with main as default and the requested description. Configure homepage https://www.filipzcvetkovic.com and topics: lidar, autonomous-driving, computer-vision, deep-learning, point-cloud, pointpillars, cooperative-perception, v2v, iot, python, pytorch, machine-learning.

Create the eight milestones below and assign issues by RV ID. Intended labels are recorded in each issue; create phase:vision, type:feature, type:bug, type:research, type:test, type:documentation, type:infrastructure, priority:critical, priority:high, priority:medium, priority:low, status:blocked, status:ready as needed. Do not mark work ready until prerequisites are verified.

Create a Project named **Raven — Development Roadmap**, add Phase 1 issues, and configure Backlog → Ready → In Progress → In Review → Done. These metadata operations are pending; text references are not milestone assignments.

## M0 — Project Foundation & Feasibility

Goal: Establish a reproducible Python project and verify the technical feasibility of the development environment.

Issues:

- RV-001 — Initialize Raven repository.
- RV-002 — Configure Python development environment.
- RV-003 — Validate Python, NumPy and Open3D on Apple Silicon.
- RV-004 — Investigate and validate KITTI dataset.
- RV-005 — Validate PointPillars inference environment.
- RV-006 — Configure GitHub Actions and code quality.

Definition of Done:

- Repository initialized.
- Python package installable.
- Tests passing.
- Dataset access validated.
- PointPillars execution strategy verified.
- CI checks configured.



## M1 — KITTI Data Integration

Goal: Implement robust LiDAR dataset ingestion and standardized data representations.

Issues:

- RV-007 — Implement PointCloud data model.
- RV-008 — Implement Detection3D and GroundTruth3D models.
- RV-009 — Implement KITTI point cloud loader.
- RV-010 — Implement KITTI annotation parser.
- RV-011 — Implement calibration and coordinate transformations.
- RV-012 — Add dataset validation tests.

Definition of Done:

- KITTI LiDAR files load correctly.
- Ground-truth annotations are parsed.
- Coordinate transformations are validated.
- Dataset loading has automated tests.



## M2 — Point Cloud Processing

Goal: Implement reusable preprocessing algorithms for LiDAR data.

Issues:

- RV-013 — Implement range filtering.
- RV-014 — Implement voxel downsampling.
- RV-015 — Implement statistical outlier removal.
- RV-016 — Implement controlled noise injection.
- RV-017 — Implement configurable preprocessing pipeline.
- RV-018 — Add preprocessing unit tests.

Definition of Done:

- Processing methods work independently and together.
- LiDAR intensity attributes are handled correctly.
- Algorithms support configurable parameters.
- Results are reproducible.



## M3 — AI Object Detection

Goal: Integrate a pretrained 3D object detection model.

Issues:

- RV-019 — Select a compatible PointPillars implementation.
- RV-020 — Download and verify pretrained checkpoint.
- RV-021 — Implement ObjectDetector interface.
- RV-022 — Implement PointPillars adapter.
- RV-023 — Implement prediction conversion to Raven data models.
- RV-024 — Validate inference against reference examples.

Definition of Done:

- Pretrained model loads successfully.
- KITTI inference works.
- Predictions use standardized Raven structures.
- Model configuration and execution environment are documented.

Important: The local MacBook M1 Pro does not provide CUDA. Investigate a compatible external GPU environment, such as Kaggle, before committing to CUDA-specific dependencies.



## M4 — 3D Visualization

Goal: Implement interactive visualization of LiDAR scenes and 3D detections.

Issues:

- RV-025 — Implement Open3D point cloud viewer.
- RV-026 — Add 3D bounding box visualization.
- RV-027 — Implement raw versus processed cloud comparison.
- RV-028 — Add ground-truth and prediction overlays.
- RV-029 — Add scene navigation and visualization CLI.

Definition of Done:

- Interactive point cloud viewing.
- Bounding box visualization.
- Ground-truth and prediction comparison.
- Screenshots can be generated for academic documentation.



## M5 — Experiment Framework

Goal: Build a reproducible framework for evaluating LiDAR preprocessing and object detection.

Issues:

- RV-030 — Implement experiment configuration.
- RV-031 — Implement experiment runner.
- RV-032 — Integrate KITTI 3D detection evaluation.
- RV-033 — Implement preprocessing and inference latency measurements.
- RV-034 — Implement CSV/JSON result export.
- RV-035 — Add experiment reproducibility tests.

Definition of Done:

- Experiments run from configuration files.
- Detection metrics are evaluated consistently.
- Latency is measured separately for preprocessing and inference.
- Results and configurations are saved.



## M6 — Research & Benchmarking

Goal: Evaluate the impact of preprocessing on LiDAR object detection.

Issues:

- RV-036 — Establish baseline detection performance.
- RV-037 — Evaluate voxel downsampling.
- RV-038 — Evaluate statistical outlier removal.
- RV-039 — Evaluate robustness under synthetic noise.
- RV-040 — Generate experimental comparison charts.
- RV-041 — Analyze experimental results and limitations.

Definition of Done:

- Baseline results available.
- Preprocessing experiments completed.
- Charts generated.
- Findings and limitations documented.



## M7 — Academic Delivery & Release

Goal: Deliver the academic project and publish the first stable Raven Vision release.

Issues:

- RV-042 — Write seminar introduction and theoretical background.
- RV-043 — Document system architecture and implementation.
- RV-044 — Document experimental methodology.
- RV-045 — Present and analyze experimental results.
- RV-046 — Prepare project demonstration.
- RV-047 — Finalize README and reproducibility documentation.
- RV-048 — Prepare Raven Vision v0.1.0 release.

Definition of Done:

- Functional application.
- Passing automated tests.
- Reproducible experiments.
- Seminar paper of approximately 20–30 pages.
- Clear documentation and demonstration.
- Release `v0.1.0` created only after the implementation is actually complete.


## Importing the existing starter safely

The cloud setup has no access to raven_vision_starter on the MacBook. Do not recreate its source or claim it is committed. Inspect local git status, existing remotes, history, untracked files, and ignore rules before importing. Fetch origin/main and reconcile the documentation commit with local history on a feature branch. If histories are independent, review and reconcile deliberately; never force-push or reset away local work. Preserve local pyproject.toml, package code, tests, and documentation. Review the staged diff for datasets, weights, secrets, and virtual environments, then open a pull request.

Do not copy an additional repository's .git directory into the starter. Do not run blanket git add before reviewing exclusions. Avoid placeholder source trees or dependency files.

## CI after code import

RV-006 must use the actual pyproject.toml and lock/install strategy. Configure checkout, Python 3.11, dependency installation, Ruff, and pytest. Verify on a clean CPU runner. Keep Open3D headless/system-library requirements explicit if tests import it. Exclude model downloads, dataset downloads, and GPU inference from ordinary PR checks. No CI workflow is added before the package exists.

## Contribution workflow

Use main as default. Examples: feat/rv-007-point-cloud-model, feat/rv-009-kitti-loader, feat/rv-014-voxel-downsampling, feat/rv-022-pointpillars-adapter.

Use conventional commits, such as feat: implement KITTI point cloud loader, test: add point cloud validation tests, docs: document Raven Vision architecture, and chore: configure Python CI. Open pull requests for substantive changes. Do not auto-merge, change branch protections, or publish v0.1.0 until authorized conditions are met.

## Architecture after source import

Grow the actual Python package under src/raven/core and src/raven/perception/{data,preprocessing,detection,evaluation,visualization}; use configs, experiments, tests, and docs/advanced_intelligent_systems as implementation requires. Do not create empty placeholder modules. Keep data local and untracked.
