# Modal remote inference (RV-049)

Raven's existing runner can submit the **actual locally processed cloud** to an
optional authenticated Modal T4 worker and consume its validated predictions.
Local OpenPCDet and saved-baseline modes retain their schema-v1 behavior. Modal
uses an explicit **experiment configuration schema v2**, request/response v1 and
the existing prediction schema v1. Run artifacts remain run-schema-v1.

**Validation status, 2026-11-26:** CPU contracts/runner/evaluator integration and
the pinned SDK's local application-definition construction were checked. No
Modal image was built, no worker deployed, no asset uploaded and no GPU inference
executed for this task. The image is a **candidate**, not a GPU-validated lockfile.
No configured Modal profile/token pair was available; actual workspace credits,
usage/spend limits, deployment and asset-use clearance could not be verified.
Mock tests below are not evidence of real model execution. RV-049's real execution
acceptance remains open until the separate bounded smoke test succeeds.

## Architecture and ownership

- `raven.inference.contracts`: strict, versioned binary request and existing
  Detection3D/prediction-document response validation, independent of the SDK.
- `RemoteDetector`: the existing ObjectDetector shape (`initialize`, `detect`,
  `provenance`, `close`) over an injectable `submit(request)` transport. One model
  abstraction, no second experiment runner.
- `ModalTransport`: optional lazy SDK import, authenticated deployed-class lookup,
  one application-level submission, bounded result wait and cancellation attempt.
- `WorkerSession`: one existing `PointPillarsDetector` initialized per worker,
  synchronized detection, actual model/environment provenance and replay handling.
- `providers/modal/app.py`: deployment policy and lifecycle hooks. Fixed T4,
  max one container, no warm/buffer containers, default sequential inputs,
  2-second idle window, 180-second execution / 300-second startup timeouts,
  application retries disabled. There is no public HTTP endpoint or schedule.
- RV-031 performs local loading/preprocessing and exclusive artifact persistence.
  RV-032 checks remote evidence at its input boundary; **metric code/protocol is
  unchanged**. It does not rerun preprocessing or inference during evaluation.

V1 has a deliberately narrow reviewed PointPillars/T4 profile. Another provider
can implement the transport and add a validated execution profile without
changing PointCloud, Detection3D or evaluation matching. Modal-specific handles
are confined to transport provenance. V2V transport, timestamp/pose alignment,
fusion and evaluation remain separate concerns; vehicles need not communicate
through Modal. Optional request context supports vehicle/sensor/session IDs,
UTC capture timestamp and an explicitly referenced finite homogeneous pose.
These are metadata only: no pose transformation, clock alignment, V2V or fusion
is performed. The current experiment runner does not invent vehicle context.

## Candidate container and model identity

The machine-readable recipe is `raven/inference/image-spec.json`; build commands
are in `modal_image.py`. The CUDA Linux/amd64 base is pinned by registry digest,
resolved from `nvidia/cuda:11.8.0-devel-ubuntu22.04`. There is no GPU fallback.

| Component | Candidate pin |
| --- | --- |
| Python | 3.10.16 |
| CUDA toolkit / Torch runtime | 11.8 / 11.8 |
| PyTorch | 2.2.2+cu118 |
| spconv / cumm | spconv-cu118 2.3.6 / cumm-cu118 0.4.11 |
| NumPy | 1.26.4 |
| OpenPCDet | 233f849829b6ac19afb8af8837a0246890908755; package 0.6.0+233f849 |
| Modal client SDK | 1.6.4 |
| GPU | T4; actual Tesla T4 / capability 7.5 required |

The [PyTorch installation matrix](https://pytorch.org/get-started/previous-versions/)
provides the 2.2.2 CUDA 11.8 build. [spconv's installation documentation](https://github.com/traveller59/spconv)
includes CUDA 11.8 wheels. PyPI metadata for
[spconv-cu118 2.3.6](https://pypi.org/project/spconv-cu118/2.3.6/) requires
cumm-cu118 >=0.4.5,<0.5.0; the selected 0.4.11 has a Linux CPython 3.10 wheel.
This verifies package availability/declared compatibility, **not** that extensions
build or inference works on Modal. All actual Python/core package/CUDA/GPU
identities are checked before accepting a worker result. Unexpected identities
fail; the historical Kaggle Python 3.13/CUDA 13 stack is not silently reused.

The base, core pins, reviewed source revision and source/recipe hashes improve
traceability. Transitive pip/apt dependencies are not a complete locked dependency
closure. After authorized build/GPU validation, retain Modal image/deployment IDs,
full package inventory/build logs and driver/toolkit evidence before calling the
environment fully reproduced. Modal runtime injection can also evolve.

Model YAML: `tools/cfgs/kitti_models/pointpillar.yaml`, SHA256
`170a9ffe76cfd8509d1044cfbcf1cbd44c5d320fda81bf0089a8d5efaf1c91c8`.
Base YAML SHA256:
`a6c44a7b0f15b94e135946cecbd8b5541906a69d37811705e9884a411e7b1797`.
Checkpoint SHA256:
`4c83fc0fa02575b9b3e9dec676f698e7a70bb5a795e89f91df8a96b916fa19e2`.
Both YAMLs and the checkpoint are verified before model use; effective runtime
configuration is retained by the existing adapter. Weights are never in the image,
Git, package or PR. The worker reads `/weights/pointpillar.pth` on a private Volume.

Before upstream imports, the image sets `CUMM_DISABLE_JIT=1`,
`SPCONV_DISABLE_JIT=1`, `TORCH_CUDA_ARCH_LIST=7.5` and CUDA_HOME. The recipe explicitly
runs the reviewed optional Argoverse workaround: only the Argo2Dataset import and
registration are removed from the **container's** pinned upstream copy, with a
preimage hash check and `__init__.py.bak` backup. Raven is not patched and the
worker never silently edits upstream at runtime. Backup/modified-file/diff hashes
are recorded; unexpected tracked upstream changes are rejected. See
`patch_argo.py` and the [historical workaround](pointpillars-feasibility.md).

Safe checkpoint loading remains default. If the known legacy checkpoint cannot
load safely, stop and review it; there is no unsafe automatic fallback. An owner
may deliberately create `raven-pointpillars-loading` with
`RAVEN_TRUST_CHECKPOINT=1`, then set `RAVEN_MODAL_TRUSTED_LOADING_SECRET=1` **during
deployment** to attach that Modal Secret. Actual pickle-loading policy is recorded.
That option is independent of cloud cost approval and asset rights.

## Rights, authentication and setup

The [checkpoint provenance audit](pointpillars-checkpoint-provenance.md) verified
the official source/hash/config association, but did not establish explicit
checkpoint-specific cloud/commercial/redistribution permissions. OpenPCDet source
licensing, checkpoint rights and KITTI terms are distinct. Public download access,
a matching checksum, research intent or a private Volume does not grant missing
rights. Confirm applicable KITTI account terms and checkpoint/provider permission
for this cloud use **before upload or execution**. This implementation supplies no
new permission and does not distribute data/weights. Only XYZI is sent for each
request; images, labels, calibration and local filesystem paths stay local.

Install the optional CPU client from the Raven checkout and authenticate through
Modal's supported SDK profile/token management (do not put credentials in JSON):

```bash
python -m pip install -e '.[modal,evaluation]'
modal setup
modal profile list
modal billing summary --json
modal billing rates --json
```

The [SDK class lookup](https://modal.com/docs/reference/modal.Cls) is lazy and uses
Modal authentication. No public endpoint, custom bearer token, hardcoded API key
or credential printing is added. Preflight checks the exact installed SDK and
opt-in flags, **not** authenticated workspace access/deployment existence; those
are checked by the SDK on submission. A wrong profile/environment or denied
access fails explicitly with no saved/local fallback. SDK exception text is
excluded from recorded transport failures to avoid leaking credential-bearing
messages. Do not enable verbose SDK logging with sensitive environment values.

After rights, budget and explicit execution approval, configure locally:

```bash
export RAVEN_ALLOW_MODAL_GPU=1
export RAVEN_MODAL_ASSET_RIGHTS_CONFIRMED=1
```

Both gates are required for deployment entry-point import and client submission;
setting flags is the operator's deliberate acknowledgement, not proof of legal
permission. Example configs contain neither flags nor credentials. Merely
validating config does not contact Modal. `--dry-run` checks local inputs, flags
and SDK only; it does not build, deploy or initialize a remote model.

## Spending controls and bounded deployment

As checked on 2026-11-26, [Modal pricing](https://modal.com/pricing) lists Starter
at $0 plus compute with $30 monthly credits. T4 is $0.000164/s (about $0.5904/h),
CPU $0.0000131/core/s and memory $0.00000222/GiB/s. These are observed rates,
not guaranteed future pricing or a statement about this user's workspace.
For illustration, 482 resource-seconds (300 startup + 180 execution + 2 idle) with
one T4, 2 cores and 8 GiB is roughly **$0.10** before build/storage/network/other
usage. A warm 60-second interval at those resources is roughly $0.0125. These are
arithmetic illustrations, **not cost caps or measured execution estimates**.
Initial source/extension builds may cost more and take longer than inference.

Credits are not a hard spending cap. [Modal budgets](https://modal.com/docs/guide/budgets)
distinguish workspace gross-usage budgets from net spend limits after credits;
environment budgets require Team/Enterprise. An owner/manager should inspect
Usage & Billing, set a suitably small workspace usage/spend limit, and retain
confirmation before this first build/upload/run. The actual workspace limit and
balance were **not accessible here**. Do not infer $30 remaining or enable a
large batch from the public Starter description.

The commands below are **manual, potentially billable operations after approval**;
they were not executed for this task. Use a permitted, locally verified checkpoint
copy; upload only that file, never a directory of datasets:

```bash
shasum -a 256 /your/permitted/pointpillar.pth
modal volume create raven-pointpillars-weights
modal volume put raven-pointpillars-weights /your/permitted/pointpillar.pth /pointpillar.pth
# Run in an installed Raven source checkout; name/environment must match your JSON.
modal deploy --env main providers/modal/app.py
```

The app requires an existing Volume (`create_if_missing=False`), starts with zero
warm containers and is not scheduled. Default request count is one; config may
explicitly raise `max_frames` only within 1..8. Exceeding the cap is invalid before
RPC. Concurrency limits are per deployment, not an account budget; other apps,
rolling deployments and platform crash/preemption replays can still incur costs.
No alternate GPU, autoscaling override or hidden warm pool is selected.

Inspect usage and stop the app after the bounded test:

```bash
modal app list --env main
modal app info raven-pointpillars --env main
modal app logs raven-pointpillars --env main
modal billing report --for today --show-resources
modal app stop raven-pointpillars --env main
```

Stopping terminates this app's workers; retain logs first. Private Volume storage
may remain after teardown. Review/remove it deliberately through Modal only when
no longer needed; Raven does not automatically delete user assets.

## Local experiment and real smoke test

Copy `configs/experiments/modal-pointpillars.json` to your own local path and set
dataset/output paths. Use a new experiment ID for each execution; existing run
outputs are never overwritten. The example selects training frame 000000 with an
identity preprocessing pipeline and a one-frame guard. Add ordered stages using
the existing pipeline format to measure a *new* processed-input inference run.
Schema v2 requires source `modal` and a `remote` object (app/environment, optional
fixed worker_class, timeout_seconds 1..600 and max_frames 1..8). Local config,
checkpoint and saved-prediction paths are forbidden in this mode. Source `saved`
and local `inference` continue to use v1 unchanged.

```bash
raven inspect-experiment /your/modal-pointpillars.json
raven run-experiment /your/modal-pointpillars.json --dry-run
# Only after explicit cloud execution approval, rights clearance and budget checks:
raven run-experiment /your/modal-pointpillars.json
```

Existing runner exit codes remain: 0 ready/success, 1 blocked/failed execution,
2 invalid config/CLI input. Results contain success/failure status and phase.
The timeout setting bounds SDK **result waiting**, not the preceding lookup or
submission RPC. Worker execution/startup have their own timeouts. No strict
client wall-time cap or provider billing cap is claimed.

The separately gated live pytest test requires an explicit config path AND both
opt-ins; regular CI skips it:

```bash
RAVEN_MODAL_SMOKE_CONFIG=/your/modal-pointpillars.json python -m pytest -q tests/test_modal_live.py
```

It checks original scan hash
`0e09c85e3f6078ecbdd1e706ee9624519f1bd29417437167a9ed7fbe6f54b4b1`, 115,384 raw
points, a genuine worker result, persisted schema-v1 prediction reload, pure CPU
visualization-corner compatibility and evaluator input compatibility. It does not
require the historical count 29 or claim accuracy. It saves a prediction capture
and evaluation evidence inside the new run directory. Graphical rendering is not
performed or claimed. Missing genuine image/GT metadata can block AP; this remains
explicit in the saved outcome. For a normal validation run use canonical val IDs
and actual labels/calibration/images. Never reuse raw baseline predictions as
processed-cloud results.

## Contracts, integrity and persistence

Requests contain version, random trace ID, experiment/config identity, exact frame
ID, KITTI Velodyne convention, preprocessing digest, reviewed model identity,
UTC submission time and optional context. Points are contiguous little-endian
float32 XYZI bytes: 1..2,000,000 rows, N×4, up to 32 MB. NaN/Inf, unsupported frames,
shape/dtype/size/hash mismatch and unknown fields are rejected. The request digest
binds all metadata and the payload SHA256. No file reference, pickle or arbitrary
array deserialization is accepted by the application contract. The authenticated
SDK has its own serialization; do not treat it as a public untrusted endpoint.

Responses must match request/frame/input/model identities and contain valid
existing schema-v1 detections, an output digest, actual worker/package/GPU/model
metadata and explicit timings. Responses over 16 MB are rejected after SDK receipt;
this application cap cannot prevent SDK-level allocation while receiving data.
Source and image-recipe hashes must match the installed client. Redeploy matching
client/worker source together; an old worker is rejected rather than silently
changing results. These are consistency checks, not hardware attestation or a
cryptographic signature proving execution origin.

RV-031 persists validated predictions, raw input identities, processed XYZI
hash/count, stage metadata and nested remote request/response/transport provenance.
Binary point payloads are **not** duplicated into JSON artifacts. No result is
published successful before validation; failed/uncertain remote outcomes abort
remaining selected frames and fail the run. RV-032 revalidates the remote envelope,
actual config/preprocessing identity and exact processed-input binding. A changed
response, source hash, frame or experiment cannot silently contribute metrics.

The client never resubmits automatically after transport/timeout/auth failures.
It attempts cancellation of an existing handle, records its safe call ID when
available, and reports uncertain outcomes. Failed cancellation does not imply the
worker stopped; inspect the app/call and stop it. The SDK/provider may internally
retry RPCs or replay inputs after container loss, even with application retries=0.
A recent 16-response in-memory worker cache returns the same result for the same
request ID/hash and rejects conflicting reuse. Cache eviction/container restart
loses that cache: **persistent exactly-once execution/charging is not guaranteed**.
Do not rerun a failed CLI run until its remote outcome is understood. For result
retrieval use the original Modal FunctionCall ID, not a new inference submission.

## Timing boundaries, validation and follow-up

All durations use each process's `time.perf_counter`; UTC stamps are identification
and audit evidence, not synchronized cross-host latency clocks. Recorded separately:

| Field | Boundary |
| --- | --- |
| client_preprocessing_timing | Local `run_preprocessing_pipeline` only |
| serialization_seconds | Client byte/envelope construction and validation |
| round_trip_seconds | SDK lookup + spawn + result wait/transfer |
| total_detector_seconds | Serialization + SDK round trip + response validation |
| model_load_seconds | Worker import/version checks + adapter initialization + CUDA synchronization; once per container, repeated as lifecycle evidence |
| detect_seconds | Existing adapter detect call between CUDA synchronizations; includes preparation/transfers/postprocessing, not a pure CUDA-kernel time |
| worker_total_seconds | Worker request validation + detection + response assembly up to timestamping; excludes return serialization/network |

Upload, download and queue delay are null with an explicit SDK visibility reason.
Container boot/image pull is not separated from the observed round trip. Never
subtract unsynchronized timestamps or sum overlapping lifecycle fields as per-frame
inference time. RV-033 can add repetitions/aggregation and finer measurement
boundaries without changing detection or evaluation semantics. RV-034 can consume
the existing artifacts; no new general export framework is added here.

CPU tests use synthetic responses and fake GPU boundaries, cover malformed
requests/responses, stale duplicates, checksum/identity/geometry errors, opt-in,
missing/wrong SDK, authentication/network/timeout failures, no automatic
resubmission, cancellation failures, processed-input runner/evaluator integration,
legacy mode regressions and imports without SDK/GPU/GUI dependencies. The pinned
SDK accepted the app/image **definition locally without RPC**; this is not an
image build. A real T4 build/boot/inference remains blocked by credentials,
explicit chargeable-execution authorization, workspace budget verification and
asset-use clearance. No new prediction count, accuracy or latency is reported.

Troubleshooting: wrong source/recipe hash requires matching redeployment; GPU or
package mismatch requires stopping and fixing the candidate image, not fallback;
checkpoint mismatch requires checking the permitted source bytes; missing av2
requires verifying the explicit patch/backup; unsafe legacy loading requires the
separate reviewed secret option; missing credentials require supported Modal
login/profile setup. Evaluation image/split errors belong to genuine dataset
metadata, not to remote inference success. Preserve original GPU logs and assets.

### Recorded local checks for this change

Full CPU suite: **1281 passed, 4 skipped** (71 additional CPU cases; the additional
skip is the opt-in live Modal test, alongside three existing optional-data skips).
Ruff, `pip check` and diff checks pass. The SDK definition check forbade socket
connections and constructed both local and simulated container-import graphs;
no App build/deploy/run/inference API was called. A built Raven wheel includes
the exact recipe JSON and imports the CPU worker/contract modules without Modal
or GPU dependencies.

The actual local training frame 000000 was read-only checked: **115,384 points**
and the expected SHA256 above. `run_experiment(..., dry_run=True)` returned
`blocked` on the missing explicit GPU opt-in, created no run directory and made
zero cloud calls; the source scan remained unchanged. Ignored local records are
under `outputs/rv049`. This is local input/preflight evidence, not remote inference,
visual rendering, accuracy, latency measurement or budget authorization.
