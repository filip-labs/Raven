# M6 raw PointPillars baseline protocol (RV-036)

Status: **offline preparation; validation inference and AP_R40 pending**.
This methodology supports the AIS seminar research. It is not a results paper.
No Modal build, deployment, upload, asset update or inference is authorized by
this preparation task. RV-036 remains open until real baseline evidence is reviewed.

## Research question and hypothesis

What 3D detection performance and explicitly scoped latency does Raven's pinned
PointPillars adapter achieve on unmodified canonical KITTI validation scans?
The baseline fixes the inputs/model/evaluator for subsequent preprocessing
comparisons. Information loss or perturbation may change detection quality and
execution cost; neither improvement nor its magnitude/direction is assumed.
Freeze selection before inspecting performance.

Fixed condition: KITTI Object Detection `val`, PointPillars through the existing
Modal adapter (`raven-pointpillars`, environment `main`, `PointPillarsWorker`, T4),
Raven preprocessing **`[]`**, seed **42**, `kitti_velodyne`, meters, original XYZI
and ordering. OpenPCDet's internal processing is unchanged. Do not edit scores,
confidence/postprocessing thresholds or KITTI evaluation rules.

## Dataset provenance and frozen selection

The primary long-term target is the complete **3,769-frame** canonical validation
split. `configs/kitti_splits/` records OpenPCDet revision
`233f849829b6ac19afb8af8837a0246890908755`; the [source validation manifest](https://github.com/open-mmlab/OpenPCDet/blob/233f849829b6ac19afb8af8837a0246890908755/data/kitti/ImageSets/val.txt)
has SHA256 `657ac4bcc1e156e5b106a4ca18e1f88e012787ea1d2b5d0adeea97fee903fa86`.
The 3,712 training IDs are disjoint. **`000000` is training**, never validation evidence.

The intermediate target is a frozen **256-frame subset** selected with existing
`FrameSelection`: sort all canonical validation IDs ascending, use local NumPy
`Generator(PCG64(42)).choice(3769, 256, replace=False)`, sort selected candidate
indexes to retain input order. Algorithm:
`PCG64.choice-without-replacement/input-order/v1`, NumPy 2.4.6 at selection.

Exact IDs are in [`m6-baseline-256.frames.txt`](../configs/research/m6-baseline-256.frames.txt),
UTF-8, one ID per line, final LF. SHA256:
`1e974d53fd9a93f3ad0cef16d1d1b5bfc02f21461a098b2ee0236449c44727a1`.
The adjacent `.selection.json` retains count, algorithm, seed, order, NumPy version,
canonical metadata identity, fixed model and empty-pipeline digest. Installed
wheels include both files. Runtime planning verifies frozen bytes, not resampling.
NumPy changes may affect regenerated samples; the committed manifest is authoritative.

First eight in order: **000053, 000173, 000191, 000247, 000272, 000291, 000332,
000420**. The one-frame pilot uses **000053**. Use the identical ordered 256 IDs
for later intermediate preprocessing comparisons. Do not substitute unavailable
frames or change selection after seeing predictions. A new selection needs an
explicit protocol revision.

Canonical split membership does **not** prove independence from the checkpoint's
original training data. The Model Zoo association does not archive its exact full
training sample history/overrides. Retain this generalization limitation.

## Fixed model and deployment requirements

Use the same independently obtained weights as RV-049 under applicable permissions.
The existing native contracts pin the following; no payload is bundled here.

| Identity | SHA256 / revision |
| --- | --- |
| OpenPCDet | `233f849829b6ac19afb8af8837a0246890908755` |
| Checkpoint | `4c83fc0fa02575b9b3e9dec676f698e7a70bb5a795e89f91df8a96b916fa19e2` |
| `tools/cfgs/kitti_models/pointpillar.yaml` | `170a9ffe76cfd8509d1044cfbcf1cbd44c5d320fda81bf0089a8d5efaf1c91c8` |
| Base KITTI YAML | `a6c44a7b0f15b94e135946cecbd8b5541906a69d37811705e9884a411e7b1797` |
| Empty preprocessing `[]` | `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945` |
| Worker runtime source inventory | `c1d588a3984caac75081b7ab6c3b2ac3fc22755582ce7e6c98123ee7a9d81531` |
| Image recipe | `7cc7e1032f3e38df1624c11d84d13dcfc4c951920ccdf02b3c669d489b29dd20` |

The last two are local fingerprints, not a fresh live deployment inspection. The
runtime source matches the historical successful worker record. Before execution,
verify the actual deployment/runtime against reviewed code/image/model; app name
alone is insufficient. Preserve strict request/processed-cloud/model/runtime/hash
bindings. The image-spec descriptive status is historical; actual RV-049 success
and environment are documented in [Modal inference](modal-inference.md).

## Offline API, configuration and CLI

`raven.research.baseline` provides `frozen_selection()`, `baseline_configuration()`,
`inspect_baseline_inputs()`, `baseline_preflight()` and `prepare_baseline()`.
It reuses experiment schema-v2, runner inputs/preflight, RV-032 PNG/calibration
validation and the pinned evaluator's `clean_data()` eligibility rules.
It never constructs a transport, invokes a detector or imports the Modal SDK.

The planner adapts `configs/experiments/modal-pointpillars.json` into **one** bounded
run. `--frame-count` accepts 1..8, default 1, and sets `remote.max_frames` accordingly.
It cannot generate a 256-frame config or run a batch. The frozen selection/report
are planning metadata, not another experiment schema. A portable review example
is [`m6-baseline-pilot.json`](../configs/experiments/m6-baseline-pilot.json); replace
its placeholder paths and ID. No personal absolute paths or credentials are committed.

```bash
# Intentionally OFF throughout offline preparation.
export RAVEN_ALLOW_MODAL_GPU=0
export RAVEN_MODAL_ASSET_RIGHTS_CONFIRMED=0
raven prepare-baseline configs/experiments/modal-pointpillars.json \
  --dataset-root /your/kitti --output-directory /your/research-runs \
  --experiment-id m6-baseline-val-000053-UNIQUE \
  --plan-directory /your/research-plans/NEW-PLAN --frame-count 1 --inspect-subset
raven inspect-experiment /your/research-plans/NEW-PLAN/experiment.json
raven run-experiment /your/research-plans/NEW-PLAN/experiment.json --dry-run
```

`--inspect-subset` reads all 256 selected source frames offline; otherwise only
bounded-run inputs are audited. The new exclusive plan directory contains
`experiment.json`, `preflight.json`, and exact `subset.frames.txt` bytes. It must be
disjoint from dataset/future outputs/template. Existing plans/runs are not replaced.
If persistence fails, preserve any partial plan for diagnosis and choose a new
path. No run directory is reserved. Generated private plans/reports stay untracked.

The report distinguishes input/evaluation readiness, runner preflight, expected
local model/worker identities, environment gates and unverified account/deployment/
budget checks. `metrics: null` means no measurements. It always states
`not_authorized_for_this_plan`, even if environment gates were previously enabled.
Preparation exits 3 for blocked readiness after saving the report, 2 for invalid
input/configuration under existing CLI policy, 0 for prepared ready offline checks.
Zero is never cloud authorization. Existing `run-experiment --dry-run` exits 1 for
blocked status; do not enable cloud flags to obtain green output.

## Actual local input audit

The preparation session inspected the established KITTI root read-only. Inventory:
7,481 filenames each in `training/velodyne`, `training/label_2`, `training/calib`;
`training/image_2` absent. Inventory is not a claim that all 7,481 files were parsed.
All **256 selected** scans, labels and calibrations were actually parsed and hashed,
including P2/upright convention, finite nonempty XYZI, point-count cap and target
visibility metadata. They contained 30,429,442 points; 487,431,669 source bytes were
fingerprinted. **0/256 matching images** were available. No sources were changed.

The private report is under `outputs/rv036/plans/val-000053-20261127-01/` in the
operator's local checkout. Its unique experiment selects only `000053`, `val`,
empty pipeline, seed 42, max_frames 1, timeout 600. Configuration inspection passed;
existing dry-run returned **blocked** with zero submissions/no run directory because
cloud opt-in was disabled. Evaluation separately remains **blocked_inputs** until
authentic corresponding images are supplied. The audit was made before inspecting
any model performance on these IDs.

| Eligible ground-truth class | Easy | Moderate | Hard |
| --- | ---: | ---: | ---: |
| Car | 200 | 567 | 750 |
| Pedestrian | 73 | 122 | 146 |
| Cyclist | 18 | 33 | 38 |

Difficulties are cumulative, not disjoint object counts. These are annotation
eligibility counts, **not detections, true positives or AP**. Zero eligible objects
makes AP unsupported; fewer than 41 triggers a sparse AP_R40 warning. Larger counts
do not guarantee statistical adequacy. Cyclist AP on this subset will be unstable;
retain the caveat rather than altering selection. Missing labels make coverage
explicitly partial. Full validation remains the primary research target.

## Evaluation, exports and reproducibility

After an approved genuine run, use the existing evaluator/readers, without a new AP
implementation or score changes:

```bash
# FUTURE offline postprocessing after a successful authorized live run:
raven evaluate-experiment /your/completed/baseline.manifest.json
raven verify-results /your/completed/baseline.manifest.json \
  --check-inputs --replay-preprocessing
raven export-results /your/completed/baseline.manifest.json \
  /your/completed/baseline.evaluation.json --format all --output /your/NEW-export
```

RV-032 validates completed run/prediction/config/model/source identities and exact
frames, retains actual confidence, and requires genuine matching PNG metadata for
nonempty prediction projection. Never guess image dimensions. Structural PNG/hash
checks are not independent acquisition authentication: obtain matching official
images. Retain image identities and evaluator/dependency versions in evaluation.
Missing metadata blocks evaluation, not produces zero AP.

Report **3D AP_R40**, optionally BEV AP_R40, separately for Easy/Moderate/Hard:
Car IoU **0.7**, Pedestrian **0.5**, Cyclist **0.5**. Preserve reference strict
`IoU > threshold`, eligibility/ignore rules, unavailable values and existing
`validation_subset`/`validation_full` purpose. Raven's CPU adapter is not certified
as equivalent to the official server. No genuine validation predictions, AP or
placeholder metric files were produced here; synthetic tests are not research.

Original RV-049 `train/000000` evidence was independently verified offline with
input/replay checks: **checks_passed**. Its 29 detections (4 Car, 19 Pedestrian,
6 Cyclist) establish infrastructure only. It and all earlier failed/recovered
runs remain intact and are not repurposed as validation predictions.

## Latency methodology

Use RV-033 definitions and schema-v2 provenance. Do not relabel local-CUDA
`benchmark-experiment` as a Modal benchmark.

| Quantity | Boundary / limitation |
| --- | --- |
| Loading | Runner input parsing/hashing/loading scope; not pure disk throughput |
| CPU preprocessing | Existing stage/total observations; raw baseline has no optional stages |
| Serialization | Client request construction/validation scope |
| Modal round trip | Client lookup/submission/wait, not disjoint queue/upload/download measures |
| Worker initialization | Model startup observation; not all container/build/queue cold start |
| GPU-backed detection | Synchronized adapter preparation/transfers/model/postprocessing; not pure kernel time |
| Worker request | Nested worker scope, excludes initialization |
| Client end-to-end | Detector serialization/round trip/validation; excludes initial loading/final persistence |

Intervals overlap. Never add them or subtract independent cross-host clocks.
Upload/download/queue remain unavailable. Retain per-frame samples, worker identity,
failures and cold/warm context for later distributions. One/eight samples validate
infrastructure, not representative performance. Do not derive FPS from the single
historical request or compare MacBook CPU timing directly with T4 model timing.

## Explicit execution phases and financial controls

| Phase | Scope | Approval |
| --- | --- | --- |
| A | Offline preparation/audit | Authorized now; zero cloud operations |
| B | Exactly one validation frame `000053`, T4, new ID/output | New explicit scope approval |
| C | At most first eight frozen frames | Separate pilot approval; cap stays 8 |
| D | Fixed 256-frame target | Reviewed bounded multi-run protocol first; not implemented/approved here |
| E | Full 3,769-frame validation | Scientific/resource justification and separate approval |

Phase D needs missing/duplicate-frame accounting, identical model/preprocessing,
unique outputs, preserved failures and a reviewed coherent exact-frame aggregation
protocol. Current evaluator reads individual completed runs. No multi-run merger
or unrestricted batching is introduced. Averaging per-run AP is invalid as a
substitute for subset/full AP. No large batch was generated or scheduled.

Before any new live scope: confirm dataset/checkpoint cloud-use permissions,
Modal account/profile, deployed worker/identity and no active prior FunctionCall;
review current credits, billing, workspace budget/spend limits; fix a frame cap,
unique paths and acceptable worst-case exposure; obtain explicit user approval.
Historical configured limits were **$0 spend / $5 workspace usage**. Historical
credits are not current evidence. This task made no account/billing/deployment
query/change. Earlier RV-049 consent does not authorize M6 execution.

A result-wait timeout is not a hard wall-clock/billing cap: lookup, queue, build,
startup and cancellation can exceed it. Worst-case cost cannot be inferred from
one 0.765 s detection sample or frame count. Review live controls with the operator;
stop if a required limit/identity is unconfirmed. No retries, checkpoint re-upload,
unsafe loading or limit increase is implied. Live app/stopped state is unverified.

## Rights, threats to validity and closure evidence

OpenPCDet source license, external checkpoint rights and KITTI terms are separate.
Follow [dataset policy](kitti-dataset.md) and the
[checkpoint provenance/rights audit](pointpillars-checkpoint-provenance.md).
Explicit checkpoint-license scope remains unresolved. Private academic consent
for an earlier run is neither a new license grant nor new M6 authorization. Do not
distribute datasets or weights; users obtain appropriately licensed assets separately.

Threats include unknown exact training overlap, imbalance/few cyclists, pilot
selection and KITTI domain limits, reference-server equivalence limits, image
provenance, software/numerical variation, cold starts/contention/network variability
and bias from dropping failures. Later synthetic noise is not a full sensor model.
Consistent hashes/canonical membership alone do not authenticate generalization.

Before closing RV-036 require reviewed genuine val inference on a declared scope,
validated native model/input/response/run identities, authentic matching images,
actual class/difficulty AP_R40 labeled subset/full, explicit timing/sample/cold-start
limits, reproducibility checks and traceable exports. A pilot does not establish
the primary full-split baseline. This PR delivers preparation, not issue closure.

## Automated verification

CPU-only synthetic tests cover frozen IDs/hash/membership, train rejection,
deterministic sampling, 1..8 guards, incompatible conditions, independent
eligibility boundaries, absent/malformed inputs, partial coverage, read-only
behavior, config round-trip, exclusive output paths and forbidden remote/GPU
imports in a separate process. CI requires no credentials, GPU or licensed KITTI.
