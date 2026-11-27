# Modal dependency/startup audit — 2026-11-27

Scope: audit and repair existing RV-049 PR #77 before another potentially billable
attempt. All validation in this audit is local/static/CPU or read-only account
inspection. No new image build, deployment, GPU call or checkpoint upload was run.
No checkpoint bytes were read locally. RV-049 is not complete based on these checks.

## Actual failures and prioritized findings

1. **Confirmed missing torchvision.** The owner-reported container trace reaches
   the pinned `pcdet/datasets/processor/data_processor.py` module-level import.
   `requirements.txt` lists torchvision, while `setup.py` does not; Raven installs
   upstream editable with `--no-deps` and had omitted it entirely. The import is
   required even for a LiDAR-only PointPillars request. The model registry also
   eagerly imports image-related backbones that import torchvision.
2. **No build-time import gate.** A successful native extension/package install
   did not exercise the actual `pcdet.datasets`/`pcdet.models` import closure.
   Missing Python packages or shared-library linkage could first fail on a GPU.
3. **Independent Git cwd defect.** Upstream `pcdet/__init__.py` tests `.git` relative
   to its own file, then invokes Git without `cwd`/`-C`. A worker imported from
   `/root` can produce the reported stderr despite a valid `/opt/OpenPCDet` clone.
   This was reproduced against the actual pinned source, without CUDA or weights.
   Importing the same source from its root produced the correct version and no
   Git warning. Raven now supplies the correct cwd, without suppressing failures
   or altering upstream source beyond the explicitly reviewed Argoverse patch.
4. **Startup retry misconception.** `retries=0` applies to ordinary method/input
   failures; it does not disable deployed-container crash rescheduling. The owner
   reported 12 containers/11 startup errors/zero completed calls. Catchable startup
   failures are now returned as ordinary method failures; process death/OOM/hard
   startup timeout remain outside this mechanism. [Modal documents this distinction](https://frontend.modal.com/docs/guide/retries).
5. **Partial runtime provenance.** Only a small package subset was checked. Runtime
   installation, build checks and returned package inventory now share the exact
   `image-spec.json` mapping, including torchvision and directly required packages.
6. **Patch-name-only acceptance.** Previously checking the one modified filename
   and backup hash did not reject extra edits in that same file. New validation
   requires the exact transformed bytes derived from the guarded original backup,
   both model YAML hashes, verified Git root/revision and the single allowed diff.
7. **Ambiguous cancellation/timeouts.** Client result wait, worker startup and
   execution are distinct scopes. The previous 300-second wait could expire while
   a 300-second startup was still possible. The example uses 600 seconds for result
   retrieval; this is not an end-to-end deadline. Uncertain handles request
   terminating cancellation with explicit lack of terminal confirmation. No client
   resubmission is introduced. Known completed errors do not trigger cancellation.
8. **Legacy candidate limitations.** Python 3.10 is selected as major.minor because
   Modal rejects a patch-level `add_python`; actual numeric patch is recorded and
   checked. The required `2025.06` environment builder remains an external setting.
   Clang is needed for the reported SharedArray compilation and `/root` PYTHONPATH
   for copied Raven build modules. Those existing owner repairs are retained.
   SDK/Python modernization is a separate migration; SDK 1.6.4 stays pinned.

The failed local run and its original manifests/configuration/results are retained
unchanged. Its persisted error corroborates client timeout and cancellation request,
not successful inference. The previous FunctionCall was inspected read-only and
reported `TERMINATED`; the app was stopped with zero active containers at preflight.
These are point-in-time observations, not future execution permission.

9. **Legacy loader security limitation.** Torch 2.2.2 is affected by
   [CVE-2025-32434 / upstream advisory](https://github.com/pytorch/pytorch/security/advisories/GHSA-53q9-r3pm-6pq6):
   `weights_only=True` is not a security boundary for malicious checkpoint content
   in affected releases (<=2.5.1; that advisory fixes 2.6.0). The existing reduced
   pickle-loading default is preserved, not advertised as a sandbox. Only the
   separately trusted, exact-hash checkpoint may be considered for this legacy
   candidate. A supported secure-stack migration needs separate compatibility
   work; Torch/SDK are not blindly upgraded in this repair.
10. **Legacy adapter package fields.** cu126 provenance keys alone were unhelpful
    for the Modal cu118 profile. Additive cu118 package fields now record observed
    values while preserving the existing cu126 fields and artifact compatibility.

## Dependency rationale and boundaries

[Official Torch compatibility](https://pytorch.org/get-started/previous-versions/#v222)
pairs Torch 2.2.2 with torchvision 0.17.2 on the CUDA 11.8 index; exact wheel pins
are `2.2.2+cu118` and `0.17.2+cu118`. The CUDA 11.8 registry exposes the CPython 3.10
Linux wheel. No torchaudio package is needed for this path.

| Required dependency | Import/use in the pinned source or direct dependency |
| --- | --- |
| Torch / torchvision | Dataset processor, model registries, neural layers and native dispatch |
| spconv / cumm | Voxel generator, tensorview, eagerly imported sparse backbones |
| SharedArray | `common_utils` and Waymo dataset registry import |
| NumPy / numba / llvmlite | XYZI/voxel arrays, model utility import; Numba's declared compiler dependency |
| SciPy | `box_utils`, Hungarian assigner imported by dense-head registry |
| scikit-image | `data_processor`/KITTI loading |
| OpenCV headless | ONCE registry's toolkit import |
| pyquaternion / Pillow | nuScenes registry and common image utilities; torchvision/skimage dependency |
| easydict / PyYAML | Model/base configuration parsing |
| tensorboardX / tqdm | Upstream requirements/common helper dependencies |
| Native CUDA extensions | Eager model registry imports and eventual PointPillars postprocessing |

Versions are recorded in the committed image specification. [Numba 0.59.1 metadata](https://pypi.org/pypi/numba/0.59.1/json)
requires NumPy <1.27 and llvmlite >=0.42,<0.43; the selected 1.26.4/0.42.0 fit.
[spconv 2.3.6 metadata](https://pypi.org/pypi/spconv-cu118/2.3.6/json) requires
cumm-cu118 >=0.4.5,<0.5; 0.4.11 fits. Core Python 3.10 Linux wheel availability and
relevant Pillow/skimage constraints were inspected in official package metadata.
This is static compatibility evidence, not execution of the corrected Linux image.
Transitive pip/apt resolution is not a complete locked closure; retain actual build
logs and inventory after approval. `pip check` and import checks must execute in
that image, not merely pass on the MacBook.

Kornia is guarded and only needed for CaDDN, torch_scatter is guarded and used by
dynamic VFE variants, av2 is removed only by the explicit patch, and other dataset
SDKs are not used for KITTI PointPillars. They are not speculatively installed.
Pandas was only needed by the excluded Argoverse path and is removed from this
profile. No model registry or preprocessing algorithm is replaced to dodge imports.

Pinned source inspected:
[requirements](https://github.com/open-mmlab/OpenPCDet/blob/233f849829b6ac19afb8af8837a0246890908755/requirements.txt),
[setup](https://github.com/open-mmlab/OpenPCDet/blob/233f849829b6ac19afb8af8837a0246890908755/setup.py),
[pcdet initialization](https://github.com/open-mmlab/OpenPCDet/blob/233f849829b6ac19afb8af8837a0246890908755/pcdet/__init__.py),
[processor](https://github.com/open-mmlab/OpenPCDet/blob/233f849829b6ac19afb8af8837a0246890908755/pcdet/datasets/processor/data_processor.py).

## Validation gates and results

- CPU suite: **1526 passed, 5 skipped**, 35.39 s; 37 new audit cases. Existing M5,
  v1/local/saved and v2/remote tests remain covered. Ruff, pip dependency and diff
  checks pass. One pre-existing third-party Dash pytest deprecation warning remains.
- New cases independently exercise real temporary Git repositories, altered
  revision/config/patch/backup content, package/Python mismatch, missing imports,
  native module linkage failures, Torch/vision ABI mismatch, CPU build-gate order,
  adapter cwd restoration, startup failure retention/no reinitialization, safe
  checkpoint refusal and timeout/cancellation identity. Runtime/SDK/GPU behavior
  in these unit tests is mocked and is not presented as GPU validation.
- Actual SDK 1.6.4 constructs corrected image and app definitions locally with socket
  connections forbidden. No hydration, build/deployment/inference was performed.
- Actual pinned OpenPCDet Git/patch/YAML validation passed on a source-only clone.
  Corrected Argoverse file SHA256:
  `b1e85ccfd71f9e0fa1e0c223f441fcf0f995e8d86b4d948bbabbfadc57b1bf0e`.
  Original backup SHA256:
  `2253ce236c5fb203db788131df2d2d348a9beb473db064a5ecd566b59f876da8`.
  The source root contained the expected `.git` metadata. No checkpoint was involved.
- CPU image gate covers all seven installed native extension imports and synthetic
  CPU voxelization. Full model construction allocates CUDA anchors and remains a
  GPU startup check. Kernel execution, driver/native ABI compatibility and safe
  loading of the real checkpoint remain unverified until authorized execution.
- The corrected ordinary Raven wheel builds using an isolated CPU build backend;
  packaged recipe, validator and worker/adapter files match the checkout exactly,
  with no datasets/weights. The non-isolated packaging attempt initially found no
  Hatchling in the client venv; isolated packaging succeeded without altering it.
- A genuine 115,384-point local frame-000000 scan also passed a fresh explicitly
  synthetic remote runner → persistence → blocked evaluation → export → offline
  verification/CPU replay smoke. The response is one synthetic fixture, not new
  model inference. The source scan is unchanged and the approval-gated live run
  directory remains unused.
- Local Docker CLI is installed but its daemon is unavailable. No local Linux
  CUDA-image build is claimed. The MacBook CPU suite cannot validate Linux `.so`
  loading; the next approved image build must genuinely pass that gate.
- Read-only account checks confirmed authentication, stopped app, no active
  containers and environment builder `2025.06`. Existing private Volume metadata
  lists `pointpillar.pth`; checkpoint bytes/hash were not reread by this audit. CLI billing totals were read, but
  current credit entitlement and usage/spend limits are not exposed by that CLI.
  Browser dashboard access failed. Operator confirmation of remaining budget
  fields and explicit execution approval is still required; no historical limit
  is assumed. Private account details remain
  only in ignored local evidence, not the repository.

Local diagnostic records live under ignored `outputs/rv049-audit/`; they include
the failed-run inventory, pinned-source checks, registry metadata, SDK graph checks,
full test log and read-only preflight records. Existing failed output is never
reused for a new run.

## Next stage requires explicit approval

Before an image build, deployment or GPU call, confirm fresh workspace budget and
asset-use rights, original call termination and stopped app state. Retain the
existing private `raven-pointpillars-weights` Volume; no re-upload or Volume
recreation is part of the proposed smoke. Do not relax limits or add payments.

Proposed bounded stage: one corrected image build/deploy, then exactly one
frame-000000 inference request with a new unique experiment/output directory,
followed by result/usage capture and stopping the app. A failed build or request
ends the stage; do not redeploy/retry automatically. The build is CPU validation;
the worker requests one T4, 2 vCPU and 8 GiB, zero warm/buffer containers, 300-second
startup and 180-second execution limits, client result wait 600 seconds. Provider
rescheduling and build/storage/network costs mean these are not a strict spend cap.
No expected detection count is imposed from the historical Kaggle capture.

Only after approval, from the corrected clean checkout and intended profile:

```bash
# Repeat read-only account/budget/app/call checks first. Do not infer limits from totals.
python -m modal profile list --json
python -m modal environment list --json
python -m modal billing summary --json
python -m modal app list --env main --json
python -m modal container list --env main --json

# Build/deploy is chargeable and is NOT authorized by this documentation:
# Set both gates only after explicit execution approval and asset-rights confirmation.
PYTHONPATH="$PWD/src" RAVEN_ALLOW_MODAL_GPU=1 RAVEN_MODAL_ASSET_RIGHTS_CONFIRMED=1 \
  RAVEN_MODAL_TRUSTED_LOADING_SECRET=0 python -m modal deploy providers/modal/app.py --env main

# Use a NEW, validated local config with frame 000000, empty preprocessing,
# source=modal, max_frames=1, result timeout=600 and a fresh output directory.
PYTHONPATH="$PWD/src" RAVEN_ALLOW_MODAL_GPU=1 RAVEN_MODAL_ASSET_RIGHTS_CONFIRMED=1 \
  python -m raven.cli run-experiment /path/to/new-approved-config.json

# Inspect/preserve actual results, export/verify only after success, then record costs:
python -m modal billing summary --json
# Stop the observed app ID, not an invented or stale ID:
python -m modal app stop ACTUAL_APP_ID --env main
```

Keep trusted loading disabled. If safe loading refuses the legacy checkpoint,
stop and report the exact exception before any separately reviewed opt-in.
The missing genuine `image_2/000000.png` still blocks full real AP evaluation;
LiDAR inference can succeed independently, and blocked AP is not zero accuracy.


Changing the reviewed recipe/runtime profile intentionally changes its recipe and
source fingerprints. Historical remote archives must retain the matching Raven
revision/profile for strict validation; they are not silently reinterpreted as
executions of this corrected image. Existing source schemas and M5 APIs are
unchanged. No old archive is rewritten or re-signed by the audit.
