# Raven Vision roadmap — Advanced Intelligent Systems

- M0 — Project foundation, KITTI source verification and PointPillars feasibility (KITTI policy and pinned split documented; Kaggle runner GPU-validated)
- M1 — KITTI annotations, calibration transforms and dataset validation
- M2 — Range filter, intensity-preserving voxel downsampling, outlier removal, noise injection
- M3 — PointPillars pretrained inference via verified CUDA GPU environment; model adapter
- M4 — Open3D point cloud / rotated 3D bounding box visualization
- M5 — KITTI-compatible evaluation and reproducible experiment runner
- M6 — Baseline, voxel, outlier and noise robustness experiments
- M7 — 20–30 page seminar, reproducibility guide, v0.1.0 release

## Scope boundaries

No cooperative fusion, V2V communications, database, or cloud services in Raven Vision.
Future: OPV2V and ML late fusion (Applied ML); MQTT-based simulated exchange (IoT).

## M0 exit criteria

1. KITTI sample loads and visualization verified on a real M1 Pro.
2. A compatible pretrained 3D detector runs on a suitable GPU. The standalone runner has a successful Kaggle T4/OpenPCDet execution report; notebook archival and per-box equivalence remain open. See [feasibility evidence](pointpillars-feasibility.md).
3. [Dataset source, terms and pinned validation split](kitti-dataset.md) documented. Model source/version/hash are recorded; checkpoint usage terms remain unresolved.
4. CI green.

The checked-in starter only implements part of M0/M1 and is not yet a detector.
