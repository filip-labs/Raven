# KITTI Object Detection dataset policy (RV-004)

## Source, access, and permitted use

Raven Vision uses **KITTI Object Detection**, not KITTI Raw or KITTI tracking.
KITTI is a KIT/TTIC autonomous-driving benchmark captured around Karlsruhe.
The [official 3D benchmark](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)
provides 7,481 training and 7,518 testing frames, corresponding point clouds,
calibration, training annotations, and a format/evaluation development kit.
Public local ground truth is provided for training, not the benchmark test set.

Sources checked on 2026-11-12:

- [KITTI homepage, copyright and citation](https://www.cvlibs.net/datasets/kitti/).
- [Official download/submission policy](https://www.cvlibs.net/datasets/kitti/user_login.php).
- [CC BY-NC-SA 3.0 deed](https://creativecommons.org/licenses/by-nc-sa/3.0/)
  and its linked legal code.
- [OpenPCDet preparation instructions at the pinned revision](https://github.com/open-mmlab/OpenPCDet/blob/233f849829b6ac19afb8af8837a0246890908755/docs/GETTING_STARTED.md#kitti-dataset).

The KITTI homepage specifies **Creative Commons Attribution-NonCommercial-ShareAlike
3.0** for its datasets/benchmarks: attribution, noncommercial use, and same-license
sharing of adaptations apply. Conditional sharing permission is not unrestricted
redistribution permission. The license deed also cautions that other rights can
limit use. Raven does not redistribute scans, labels, calibration files, archives,
or sample-derived artifacts. Obtain your own copy and check current terms before
sharing derived material or using it commercially. Dataset terms do not establish
checkpoint usage rights; those remain separately unresolved in the
[PointPillars report](pointpillars-feasibility.md#implementation-choice-and-licensing-boundaries).

Acquisition procedure:

1. Start at the official 3D benchmark page and follow its download links.
2. Register/login yourself and specify the intended purpose. The public policy
   separates download registration from the additional submission registration.
3. Review any account/download-specific terms shown to you. An authenticated
   registration agreement was not independently inspected for this update.
4. Download Velodyne point clouds, calibration, training labels, and the object
   development kit. Images are optional for Raven's present LiDAR inspection;
   external preparation/evaluation tools may require `image_2` as well.
5. Extract locally into the layout below. Keep original archives and retrieval
   date/source URLs/checksums in a private provenance record. No archive retrieval
   date or original download log is available for the existing local copy.

Do not bypass registration or use a third-party mirror as evidence of official
provenance. No dataset download is needed for standard CI.

For this object benchmark, the official requested research citation is:
Andreas Geiger, Philip Lenz, and Raquel Urtasun, *Are we ready for Autonomous
Driving? The KITTI Vision Benchmark Suite*, CVPR, 2012. Retain the dataset title,
authors, source and license links when attributing material; a paper citation
alone does not establish compliance with every license condition.

## Local layout and coordinate conventions

```text
data/kitti/                    # ignored; user-acquired data only
├── training/
│   ├── velodyne/000000.bin     # through 007480.bin
│   ├── calib/000000.txt        # matching six-digit IDs
│   └── label_2/000000.txt
└── testing/
    ├── velodyne/000000.bin     # through 007517.bin
    └── calib/000000.txt        # no public label_2 expected
```

IDs are scoped by `training`/`testing`; equal stems across those directories do
not mean the same frame. A LiDAR record is four little-endian float32 values
(X, Y, Z in meters, reflectance), 16 bytes per point. Velodyne axes are forward,
left, up; rectified camera axes are right, down, forward. Labels use camera
bottom-center locations, height/width/length, and rotation about camera Y.
`DontCare` regions are ignore metadata, not ordinary target boxes.

Reuse [annotation parsing](kitti-annotations.md) and
[calibration transformations](kitti-calibration.md). Camera conversion uses
`R0_rect_h @ Tr_velo_to_cam_h`; projection matrices are not rigid transforms.
Match scan, calibration, and label stems before conversion. Raven preserves
source coordinates and converts full box orientation without replacing it with
a camera-to-LiDAR yaw shortcut. The official development kit is the format reference;
these descriptions also state the conventions enforced by Raven's existing APIs.

## Frozen internal validation policy

Use the checked-in **frame-ID metadata only** in
[`configs/kitti_splits`](../configs/kitti_splits/):

| Membership | Frames | Source partition |
| --- | ---: | --- |
| `train.txt` | 3,712 | labeled `training/` |
| `val.txt` | 3,769 | labeled `training/` |
| Official benchmark test | 7,518 | unlabeled `testing/`; excluded from local tuning |

The lists are byte-for-byte copies from OpenPCDet commit
`233f849829b6ac19afb8af8837a0246890908755`, matching the upstream revision selected
for Raven's PointPillars feasibility work. Exact source URLs, counts and SHA256
hashes are in [`provenance.json`](../configs/kitti_splits/provenance.json).
They are retained under upstream OpenPCDet's
[Apache-2.0 license](https://github.com/open-mmlab/OpenPCDet/blob/233f849829b6ac19afb8af8837a0246890908755/LICENSE);
see the adjacent attribution/license notice. These IDs contain no scan or label content.
This is a conventional OpenPCDet split, **not an official KITTI validation split**.

Selection algorithm: read the pinned lists in file order, retain six-digit IDs,
reject malformed/duplicate IDs, verify recorded byte hashes/counts, require no
overlap, and require their union to be exactly `000000` through `007480`.
Resolve every ID under `training/`, never `testing/`.
**Split seed: not applicable.** No random sampling is performed and the original
upstream generation seed is not established. A new seeded random split would
change the comparison population; it is deliberately not invented here. The
recorded memberships are the reproducibility authority, stronger than a seed alone.

Use the full same `val.txt` for baseline, voxel, outlier and noise comparisons.
Do not silently drop missing frames or choose different successful subsets per
condition. Fail the run, or predeclare a separate reduced protocol and archive
its exact IDs, rationale and limitations before looking at results. For future
stochastic degradation, use fixed seed **42**, and archive RNG implementation,
version and per-frame draws/parameters; this seed does not redefine membership.
Freeze preprocessing, thresholds, checkpoint/config hashes, class/difficulty/IoU
rules, evaluation implementation/version and timing boundaries for comparisons.
The experiment runner and KITTI AP evaluation are future work, not implemented
by this dataset policy.

### Leakage and claims

The upstream split is suitable for comparing with the selected implementation,
but split disjointness alone does not prove an independent evaluation. The exact
checkpoint training IDs, augmentation database membership and historical effective
training configuration have not been independently recovered. Do not claim the
checkpoint never saw these validation frames. Closely related KITTI scenes may
also remain correlated; this is not a proven sequence-disjoint split.

Frame `000000` is in **train.txt**. Its successful GPU run demonstrates inference
feasibility, not held-out accuracy. No existing prediction counts establish AP,
generalization, or a match to ground truth. Until training provenance is verified,
label future results as fixed-checkpoint internal robustness comparisons with
potential training-data overlap. Validation used for tuning is not an untouched
final holdout. Official test data must not be used for parameter selection;
follow the official submission policy rather than using the server for tuning.

## Read-only validation and reproduction

After the normal CPU installation (`python -m pip install -e '.[dev]'`), run from
the repository root:

```bash
# Always works without KITTI: verifies pinned membership, hashes and counts.
python scripts/validate_kitti_dataset.py

# Complete filename inventory; content checks for two explicit training samples.
python scripts/validate_kitti_dataset.py --root data/kitti \
  --sample 000000 --sample 000001

# Existing focused inspection commands; no GPU or GUI required.
raven inspect data/kitti/training/velodyne/000000.bin
raven inspect-labels data/kitti/training/label_2/000000.txt
raven inspect-calib data/kitti/training/calib/000000.txt
```

For externally stored data, pass an absolute `--root`; do not copy or symlink data
into tracked paths. Inventory checks require all expected `.bin`/`.txt` stems in
the five directories; wrong/missing IDs fail clearly. It reads content only for
selected samples, using existing Raven finite-XYZI, binary-length, label and
calibration validation. It hashes selected files before/after parsing to detect
byte changes during inspection. It never writes source data. This is not a
complete dataset checksum audit or a guarantee that uninspected contents are valid.
Unrelated files/extensions are ignored. An empty cloud remains valid under the
existing loader contract; inspect point counts before an inference experiment.

Locally checked on 2026-11-12 using the original checkout's ignored `data/kitti`
as an explicit external root from a clean worktree:

- All five inventories matched: training scans/calibration/labels **7,481 each**;
  testing scans/calibration **7,518 each**.
- Frame `000000`: **115,384 points**, one non-ignored object; scan SHA256
  `0e09c85e3f6078ecbdd1e706ee9624519f1bd29417437167a9ed7fbe6f54b4b1`.
- Frame `000001`: **120,268 points**, three non-ignored objects; scan SHA256
  `59a02fdaaab3b7e903713cb618e8f53efcaf71c144436ddfcdf4f28bdbd73d20`.
- Both samples passed existing parsers; all six inspected source hashes were
  unchanged before/after. No scans, weights, labels or prediction artifacts were
  copied, committed or modified. Matching structure/sample hashes do not prove
  the original acquisition path or whole-archive integrity.

The [PR #54 GPU validation report](https://github.com/filip-labs/Raven/pull/54#issuecomment-6808816795)
confirms the standalone runner executed on a Kaggle T4 and reproduced 29 predictions
(Car 4, Pedestrian 19, Cyclist 6). Individual boxes/scores have not been compared
numerically and detection accuracy remains unevaluated. See the
[feasibility guide](pointpillars-feasibility.md) for isolated GPU reproduction.

CPU validation for this update: `python -m pytest -q` (288 passed, one optional
local-data test skipped in the clean worktree), `python -m ruff check .`,
`python -m pip check`, and `git diff --check`. Tests use synthetic temporary
samples and frozen metadata, covering determinism, disjointness, malformed and
duplicate IDs, tampering, missing files, parser errors, read-only repeatability,
and CLI behavior. No CUDA dependencies or dataset downloads are added to CI.
