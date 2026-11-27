# Experiment runner (RV-031)

`raven.experiments.runner` executes the existing schema-v1
[experiment configuration](experiment-configuration.md). It reuses Raven's KITTI
loaders, RV-017 pipeline, RV-023 prediction documents and PointPillarsDetector.
No parallel detector, configuration format, evaluator or rendering path is added.

## CLI

```bash
# Read/validate all selected inputs and dependencies; write nothing.
raven run-experiment /path/to/my-experiment.json --dry-run

# Reserve a new experiment-ID directory and execute.
raven run-experiment /path/to/my-experiment.json
```

The command prints a JSON summary with status, config digest, selected IDs,
succeeded/failed executed-frame counts, output directory and failure details.
Exit 0 means successful execution or a `ready` dry-run; exit 1 means failed/blocked
execution/preflight; exit 2 means invalid configuration/CLI syntax. A dry-run's
zero processed counts do not claim execution. Preflight failure writes no run
artifacts; retain the CLI/API errors externally if needed. Reusing an experiment
ID in an existing output base fails instead of silently overwriting results.

The example below consumes your existing frame-matched saved predictions:

```json
{
  "schema_version": 1,
  "name": "Previously generated PointPillars baseline",
  "experiment_id": "saved-baseline-001",
  "dataset": {"root": "/path/to/kitti", "split": "train"},
  "frames": {"ids": ["000000"]},
  "model": {
    "name": "pointpillars", "source": "saved",
    "predictions_dir": "/path/to/saved-predictions"
  },
  "preprocessing": [],
  "seed": 42,
  "output": {"directory": "/path/to/outputs", "artifact_prefix": "baseline"}
}
```

Saved prediction files must be named exactly `FRAME.json`. Copy an existing
capture into a private directory with that name if necessary; never rename or
modify your sole original evidence. Frame 000000 belongs to canonical **train**,
not val. Relative paths retain RV-030's config-file-relative semantics.

## Architecture and execution flow

1. `load_experiment_config()` / existing immutable `ExperimentConfig`: normalized
   declarations, seed/selection identity and compatibility validation.
2. `preflight_experiment(config) -> PreflightReport`: revalidate paths/manifest,
   reject existing run directories, check writable output ancestry, verify dataset
   and split evidence, then load/hash/validate every selected scan, labels,
   calibration and saved capture. No outputs are created. Inputs are held one
   frame at a time; no whole-dataset point arrays are retained.
3. For inference, check the pinned revision, CUDA/PyTorch and required modules,
   derive OpenPCDet root from the configuration inside its checkout, verify actual
   revision/assets/checkpoint checksum, and construct/initialize the **existing**
   PointPillarsDetector before reserving outputs. Model initialization failure
   returns blocked, closes the adapter when possible and creates no results.
4. Exclusively reserve `output.directory/experiment_id` (private directory), record
   a started event, then re-read/check each input against preflight hashes.
   Invoke `run_preprocessing_pipeline` in configured order. Saved mode allows
   only the identity pipeline; inference calls `detector.detect(processed.cloud)`.
5. Collect genuine per-frame predictions/failures and ground-truth/calibration
   data. Close an initialized detector even after failures. Record actual source,
   environment, identities and source-specific provenance.
6. Publish each artifact atomically and exclusively. Publish the final manifest
   **last** as the completion marker. Return typed `RunSummary`.

The public API supports a configuration path or validated configuration:

```python
from raven.experiments.runner import preflight_experiment, run_experiment

# See the configuration guide for loading an immutable config object.
summary = run_experiment("my-experiment.json", dry_run=True)
if summary.status == "ready":
    summary = run_experiment("my-experiment.json")
print(summary.to_dict())
```

`FileIdentity`, `FrameInputs`, `PreflightReport`, `RunFailure` and `RunSummary`
carry structured execution state, not new detector interfaces. Tests may mock the
existing adapter boundary; such tests are explicitly synthetic, not GPU evidence.

## Split membership and applicability

Offline train/val membership comes from the existing pinned OpenPCDet split
manifests and provenance metadata in `configs/kitti_splits`. The runner verifies
both SHA256s/counts, disjointness and the labeled KITTI ID union through the
same shared verifier as the existing dataset-validation script before accepting
selected membership. The same resources, notices and license ship in the wheel;
source/editable checkouts use the existing files. A training-directory filename
alone is never evidence that a frame belongs to val.

Missing authoritative split evidence yields **blocked**; corrupt split evidence
or an ID outside the declared split yields **failed**. No authoritative test
manifest is currently bundled, so test execution is explicitly **blocked**.
Testing samples are not assigned labels or presented as labeled evaluation data.
Supporting a reviewed unlabeled test policy later must preserve that distinction.
Train/val currently require matching `label_2` and `calib` files even for baseline
execution. Ground truth keeps the existing rectified-camera bottom-center/Y-axis
conventions; saved predictions stay Velodyne geometric-center/+Z. No approximate
coordinate conversion or detection matching is performed here.

## Saved predictions and live inference

Saved mode loads the existing RV-023 schema reader and verifies expected frame,
coordinate/units/box conventions and KITTI class names. When present, capture
input SHA256 and point count must match the actual raw scan. Declared checkpoint
hashes must be well formed/consistent, and a declared effective model must be
PointPillar. Optional metadata must be finite JSON. Full capture metadata is
retained alongside the actual capture-file hash. Invalid or mismatched inputs
fail preflight. No threshold or prediction geometry is changed.

Older valid schema-v1 files without scan hashes can be consumed, but the frame's
`prediction_source.input_binding` is explicitly **unavailable**. With a matching
hash it is `sha256_verified`; neither status independently authenticates GPU
execution, checkpoint rights, model quality or the capture's declared metadata.
`model_provenance` is marked declared/not independently verified. Do not turn
missing provenance into a claim of full reproducibility.

Saved predictions plus any new preprocessing remain forbidden by RV-030. Captures
that themselves declare nonempty preprocessing are rejected for this raw baseline
path. There is no silent inference-to-saved fallback. Successful saved-mode
execution means **loaded_saved_predictions**, never a new inference run.

Live mode uses the existing adapter's strict loading path, device 0 and
`trust_checkpoint=False`; there is no unsafe pickle retry. Model configuration
must reside in the declared pinned OpenPCDet checkout so the root can be derived
without a parallel configuration source. Existing adapter JIT/arch setup and
runtime provenance are reused. CPU/macOS or absent CUDA/dependencies returns
**blocked**, not fabricated predictions. Dry-run checks availability/assets when
possible but does not initialize a model or run inference; transitive dependency,
model-state and actual runtime failures can still block initialization afterward.
No weights are downloaded or GPU infrastructure provisioned. Asset hashes identify
bytes, not licensing permissions: external users must supply approved assets and
follow [checkpoint rights](pointpillars-checkpoint-provenance.md) and
[reference reproduction](reference-validation.md).

## Artifacts and transaction semantics

The existing RV-030 planned names are reused, with two minimal runner additions:

| Artifact | Version / content |
| --- | --- |
| `PREFIX.config.json` | Existing configuration schema v1; normalized effective declaration |
| `PREFIX.frames.txt` | Exact selected IDs, in execution order |
| `PREFIX.results.json` | `run_schema_version: 1`; typed-summary representation and per-frame records |
| `PREFIX.provenance.json` | `run_schema_version: 1`; UTC start/end, source/environment, split, selection, model preflight and input identities |
| `PREFIX.started.json` | Historical start event, not a claim of completion |
| `PREFIX.manifest.json` | `run_schema_version: 1`; final status, config digest, frame IDs, artifact filenames/SHA256s; published last |
| `PREFIX.failure.json` | Best-effort failure summary if persistence fails; may itself be unwritable |

Per-frame success records include actual input identities (scan, labels,
calibration, capture where applicable), input/processed counts, little-endian
float32 XYZI processed-data hash, existing pipeline metadata, RV-023 prediction
document, prediction source/capture metadata, counts, ground-truth objects,
DontCare regions and calibration matrices. Raw/processed point arrays and weights
are not exported. Position/axis conventions remain explicitly present in existing
Raven models/documents.

Provenance records normalized configuration/hash, bound frame selection, seed,
actual file byte hashes/sizes/resolved paths, canonical split evidence, Raven
Git SHA/worktree status (or null outside its actual source checkout), runner
module hash/path, Python/package versions,
OS/CPU architecture and actual backend evidence where applicable. Backend per-frame
provenance includes its effective model/environment details. The processed hash
is over canonical contiguous little-endian float32 XYZI bytes, not an invented
source filename. Config hashes alone do not establish reproducible execution. The existing experiment
seed controls frame selection and explicit preprocessing seeds control their own
operations. The runner does not reset global RNGs or force deterministic CUDA
kernels; live cross-run numerical reproducibility still requires external validation.

Each new file is flushed/fsynced to a private temporary file, then atomically
hard-linked to an absent destination; existing files/symlinks are never replaced.
Hard-link support is required. Temporary files are cleaned on ordinary failure.
The final manifest excludes its own hash. Consumers must require its presence,
check its status and verify referenced hashes before reading results. A failed
or interrupted persistence may leave a reserved directory and started/intermediate
files, **without a final manifest**. Those are not a committed successful run.
Do not delete/reuse that directory automatically; choose a new ID after inspection.
This is minimal persistence, not RV-034's general export framework. It is not a
security boundary against hostile concurrent filesystem mutation or a full
power-loss/network-filesystem durability guarantee.

## Failure policy

Preflight failures collect contextual frame errors and prevent all output
creation. An unavailable required execution environment/split policy is blocked.
After successful preflight/reservation, individual frame failures are recorded
and remaining selected frames are attempted. Any mandatory frame, cleanup or
persistence failure makes overall status **failed**, even if other frames succeed.
Failed frame records contain failures and preflight input identities, not invented
predictions; those inputs must not be read as successfully processed data.
Executed-frame succeeded/failed counts are separate from preflight/global errors;
no executed frames means both counts are zero. Config errors still raise the
existing configuration exception through the API. KeyboardInterrupt/process death
is not swallowed; absence of the final manifest identifies an incomplete run.

Inputs are hashed around loading and checked again at execution. Changed/corrupt
files fail instead of quietly using a new input identity. This detects ordinary
changes, not adversarial rewrite-and-restore attacks or cryptographic attestation
of an external capture. Files are not overwritten. Future RV-032 must consider
saved input_binding/unverified model provenance when making reproducibility claims.

## Executed validation and real saved-capture evidence

The standard CPU tests use controlled synthetic scans, labels, calibration,
saved detections and explicit mocked GPU boundaries. They test positive/negative
preflight, deterministic selection/identities, preprocessing, partial failures,
output permissions/collisions/atomic publication, source changes and a fresh
baseline process with SciPy/Open3D/GPU imports actively blocked.

Separately, a bounded local integration loaded genuine KITTI train frame 000000:
115,384 points, one ground-truth object, and the existing saved capture's **29**
predictions (Car 4, Pedestrian 19, Cyclist 6). Source capture SHA256:
`cbcca87d066c66843774cce94bb3129254245a4498290d5a9d25e4a45aeff5d3`;
scan SHA256:
`0e09c85e3f6078ecbdd1e706ee9624519f1bd29417437167a9ed7fbe6f54b4b1`.
Both counts and input binding were checked; source scan, labels, calibration and
original capture hashes remained unchanged. A private byte-identical capture copy
supplied the required `000000.json` name. This is a **real saved-baseline run**,
not new inference, independent reference validation, accuracy or timing evidence.
An actual live-mode attempt on this Mac returned blocked (PyTorch unavailable)
and created no run directory. No trusted NVIDIA execution occurred here.

Example **generated** manifest from the local development smoke run
`real-saved-000000-v1` (development worktree state is recorded in its provenance):

```json
{
  "run_schema_version": 1,
  "status": "success",
  "config_sha256": "355461aa999742c0960dfd37af2971548b181b86d9ebcf4e4fc4e5c485fad621",
  "frame_ids": [
    "000000"
  ],
  "artifacts": {
    "configuration": {
      "file": "baseline.config.json",
      "sha256": "3fccbdd6fda04e00fc43e30481e3dfb5bef7c37c99ef7983746a6489991027c2"
    },
    "frames": {
      "file": "baseline.frames.txt",
      "sha256": "2f1ab4cf0adb400efabfa05a6c1ca3791de7c1d2edadfba882dc9c8dba1992bb"
    },
    "provenance": {
      "file": "baseline.provenance.json",
      "sha256": "0ff09ce3b2fe93512b6f1f0734540a53528c59b02c9beb2b28714cab08dc8820"
    },
    "results": {
      "file": "baseline.results.json",
      "sha256": "49292f392ea4217735a58a5e3aa78ba1538a43ffaddc3691080e273915533f8f"
    },
    "started": {
      "file": "baseline.started.json",
      "sha256": "d6821c7027d343094895ed8cacf10916243d81e060ef0130177372ab2342e45e"
    }
  }
}
```

Local artifacts/logs are ignored under
`/Users/filip_macbook/Downloads/Raven/outputs/rv031`; no scans, capture files,
weights, generated runs or wheels are committed. The wheel resource smoke check
also verified all 3,712 train and 3,769 val memberships from the built package.

## RV-032 integration

Require a completed final manifest and verify file hashes/status first. Load
`results.json` frame records and use RV-023 `detections_from_document` for the
prediction objects. Ground truth, ignore metadata and calibration preserve existing
Raven conventions for the evaluator's coordinate transforms. Join by explicit
frame_id, never array position from another run. Inspect input identities and
source binding before interpreting comparisons; failed/uncommitted runs cannot
be represented as successful evaluations. Test is currently blocked/unlabeled.

RV-032 adds metrics and matching/ignore policy; RV-033 adds proper timing;
RV-034 adds broader export; RV-035 adds dedicated reproducibility coverage.
RV-031 does not compute AP/mAP, benchmark latency, train models or redesign views.

## Optional remote source (RV-049)

The same runner also accepts the explicit schema-v2 Modal configuration described
in [Modal inference](modal-inference.md). Local preprocessing precedes a validated
remote ObjectDetector call, with no saved/local fallback. Run artifacts remain
schema v1 and embed configuration v2, processed-cloud identity and remote
request/response provenance. Any remote frame failure aborts remaining submissions
and fails the run. Legacy v1 sources retain their existing behavior.
