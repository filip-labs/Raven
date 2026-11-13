# PointPillars feasibility: reported Kaggle execution and reproduction runner

Raven Vision has a user-verified report of pretrained PointPillars execution on
real KITTI LiDAR data using OpenPCDet on Kaggle. This establishes external GPU
inference feasibility, not detection accuracy. The newer
[reusable adapter](pointpillars-adapter.md) shares this workflow and has a
[successful T4 execution report in PR #58](https://github.com/filip-labs/Raven/pull/58#issuecomment-6839286530).
Per-box numerical comparison remains open. A subsequent
[PR #54 validation report](https://github.com/filip-labs/Raven/pull/54#issuecomment-6808816795)
confirms the standalone runner executed successfully on a Kaggle T4 and reproduced
the same 29 predictions by class. This is experimenter-reported GPU evidence, not
a new GPU rerun during the RV-004 documentation update. Individual box/score
equivalence remains unverified.

## Evidence and its boundaries

The experiment details below were supplied by the experimenter. The original
Kaggle notebook/export, complete installation log, and prediction JSON were not
available in the repository. No notebook URL, prediction boxes, scores, timings,
accuracy, or full environment lockfile has been reconstructed as historical fact.
The experiment date was not supplied.

| Reported environment | Value |
| --- | --- |
| Kaggle allocation | Two NVIDIA Tesla T4 GPUs, compute capability 7.5 |
| Python | 3.13.16 |
| PyTorch | 2.11.0+cu130 |
| CUDA toolkit | 13.0 |
| spconv package | spconv-cu126 2.3.8 |
| OpenPCDet package | 0.6.0+233f849 |
| OpenPCDet checkout | /kaggle/working/OpenPCDet |
| Configuration | tools/cfgs/kitti_models/pointpillar.yaml |
| Checkpoint | /kaggle/working/pointpillar.pth |

This allocation does not establish that both GPUs were used. The new runner uses
CUDA device 0 only; no distributed inference or scaling measurement is claimed.
CUDA toolkit, PyTorch's CUDA runtime, and the spconv wheel's CUDA suffix describe
different components. Their reported combination is empirical evidence for this
experiment, not an officially supported compatibility matrix for all machines.

Reported input:

- Frame: `000000.bin`, 115,384 float32 XYZI points.
- SHA256: `0e09c85e3f6078ecbdd1e706ee9624519f1bd29417437167a9ed7fbe6f54b4b1`.

Reported result:

- Process exit code: **0**.
- Predictions: **29** — Pedestrian **19**, Car **4**, Cyclist **6**.
- Output path: `/kaggle/working/raven_predictions_000000.json`.

These are **model predictions**, not verified ground-truth matches. No KITTI AP,
precision, recall, latency, correctness of individual detections, or threshold
sensitivity was measured by this evidence. The JSON's exact schema and contents
were not available; no fabricated substitute is checked in.

Independently checked during this repository update:

1. The existing local KITTI frame has the reported point count and input checksum.
2. The official repository resolves short revision `233f849` to full commit
   `233f849829b6ac19afb8af8837a0246890908755`.
3. The [model-zoo PointPillar row at that revision](https://github.com/open-mmlab/OpenPCDet/blob/233f849829b6ac19afb8af8837a0246890908755/README.md#kitti-3d-object-detection-baselines)
   pairs `tools/cfgs/kitti_models/pointpillar.yaml` with Google Drive file ID
   `1wMxWTpU1qUoY3DsCH31WJmvJxcjFXKlm`.
4. Downloading that publicly linked checkpoint into temporary storage yielded
   **19,383,576 bytes** and SHA256
   `4c83fc0fa02575b9b3e9dec676f698e7a70bb5a795e89f91df8a96b916fa19e2`,
   exactly matching the reported checkpoint. It was not deserialized or committed.

The resolved upstream revision is not a recovered Kaggle Git log. Local notebook
patches and effective config bytes/overrides were not archived. The upstream
model-zoo association verifies the intended checkpoint/config pair; an exact
reconstruction of the historical notebook remains incomplete.

## Implementation choice and licensing boundaries

| Candidate | Evidence for Raven | Decision |
| --- | --- | --- |
| [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) | Reported successful T4 inference; pinned revision, officially paired config/checkpoint, independently matching checkpoint hash | Retain as the feasibility implementation; adapter work remains separate |
| [MMDetection3D](https://github.com/open-mmlab/mmdetection3d) | Alternative upstream framework with [Apache-2.0 code license](https://github.com/open-mmlab/mmdetection3d/blob/main/LICENSE); no Raven runtime, checkpoint, or performance test was performed | Not selected for this experiment; no compatibility or superiority claim |

OpenPCDet's [code license](https://github.com/open-mmlab/OpenPCDet/blob/233f849829b6ac19afb8af8837a0246890908755/LICENSE)
is Apache-2.0. This is not a blanket license determination for every downloaded
checkpoint, dataset, or dependency. Before use or redistribution, confirm the
weight provider's applicable terms and KITTI's dataset terms. Keep a provenance
record with source URL, retrieval date, SHA256, model configuration/revision, and
applicable terms. The checksum proves byte identity, not licensing rights.
No project license or redistribution permission is added by this work.

Official checkpoint source:
[Google Drive model-zoo file](https://drive.google.com/file/d/1wMxWTpU1qUoY3DsCH31WJmvJxcjFXKlm/view).
Do not substitute HTML login/download-error pages for checkpoint bytes. Use your
own licensed file and verify its digest before any deserialization. A different
hash requires a separate provenance investigation, not an edited expected hash
presented as the original experiment.

## Reported workarounds

### Fresh subprocess and JIT settings

The experimenter reported an initial C++ JIT compilation failure involving
missing tensorview headers. The successful workaround launched a **fresh Python
subprocess**, with these settings present before any OpenPCDet/spconv import:

```bash
export CUMM_DISABLE_JIT=1
export SPCONV_DISABLE_JIT=1
export TORCH_CUDA_ARCH_LIST=7.5
```

The new runner sets these defaults before GPU imports and refuses conflicting
JIT-disable values. In a notebook, changing environment variables after imports
is insufficient; launch a new process. Architecture 7.5 targets the reported T4.
These flags do not supply missing prebuilt kernels, disable every compilation
step in OpenPCDet installation, or guarantee that another CUDA/Python stack works.
See the upstream [cumm instructions](https://github.com/FindDefinition/cumm) and
[spconv environment handling](https://github.com/traveller59/spconv/blob/master/spconv/constants.py).

### Optional Argoverse import

The reported failure was `ModuleNotFoundError: No module named 'av2'` while using
KITTI. At the pinned revision, `pcdet/datasets/__init__.py` eagerly imports
Argoverse 2. The experimenter modified **only the Kaggle OpenPCDet checkout**:

- Backed up the original file as `pcdet/datasets/__init__.py.bak`.
- Removed `from .argo2.argo2_dataset import Argo2Dataset`.
- Removed the corresponding `Argo2Dataset` entry from the dataset registry.

This explains the successful report; the original patch file was not available.
Raven's script **does not patch upstream code, inject fake modules, suppress an
import failure, or install av2 automatically**. First inspect the traceback.
If choosing the same workaround for a KITTI-only checkout, explicitly back up
and review those two edits yourself. Preserve `git diff` and the backup outside
version control with your run evidence. Do not use this reduced registry for
Argoverse data. Installing a compatible av2 stack is an alternative to investigate,
not a step validated here. Other missing dependencies must be diagnosed separately.

## Reproduction: reconstructed steps, not a recovered notebook

The new [headless runner](../scripts/kaggle_pointpillars.py) follows the pinned
OpenPCDet dataset preparation, batching, network construction, and forward APIs
without importing its GUI demo. Its GPU path was successfully run on the reported
Kaggle T4 stack (PR #54), producing `raven_predictions_verified.json`. Its input
validation, JSON serialization, CLI, and mocked orchestration are CPU-tested.
The script's output schema is new and does not claim to match the missing JSON.

1. In an external Kaggle GPU session, obtain a checkout of Raven and a separate
   OpenPCDet checkout at full revision
   `233f849829b6ac19afb8af8837a0246890908755`. Use the matching PointPillar config
   and its base YAML files. Check the GPU allocation and actual installed versions.
2. Provision the reported GPU stack, using the pinned upstream
   [installation guide](https://github.com/open-mmlab/OpenPCDet/blob/233f849829b6ac19afb8af8837a0246890908755/docs/INSTALL.md)
   as reference. The complete successful installation command sequence and
   Kaggle image version were not supplied; this is **not** a tested lockfile or
   an assertion that blindly installing these versions reproduces that image.
   Review extensions, compiler/driver requirements, and any local patches.
3. Install Raven's standard core with `python -m pip install -e /path/to/Raven`.
   No torch, CUDA, spconv, or OpenPCDet packages are added to Raven's dependencies
   or macOS/CPU CI. Provision those only in the external GPU environment.
4. Obtain your own licensed KITTI `.bin` scan and trusted matching checkpoint.
   Verify hashes and retain provider terms. For another input, supply its own
   digest instead of the reference digest. Export the three settings above.
5. Launch the script as a **new process**, using your actual paths:

```bash
python /path/to/Raven/scripts/kaggle_pointpillars.py \
  --input /path/to/000000.bin \
  --input-sha256 0e09c85e3f6078ecbdd1e706ee9624519f1bd29417437167a9ed7fbe6f54b4b1 \
  --openpcdet-root /kaggle/working/OpenPCDet \
  --config /kaggle/working/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml \
  --checkpoint /kaggle/working/pointpillar.pth \
  --checkpoint-sha256 4c83fc0fa02575b9b3e9dec676f698e7a70bb5a795e89f91df8a96b916fa19e2 \
  --checkpoint-source https://drive.google.com/file/d/1wMxWTpU1qUoY3DsCH31WJmvJxcjFXKlm/view \
  --output /kaggle/working/raven_predictions_000000_reproduction.json
```

Paths containing `/path/to/` are placeholders, not known Kaggle input locations.
The default loader uses PyTorch's `weights_only=True`. If a trusted checkpoint
needs pickle metadata, inspect that error and deliberately opt in using
`--trust-checkpoint`; never enable pickle for an unknown or mismatched file.
Unlike upstream's permissive partial loader, this runner uses strict state-dict
loading and will fail rather than infer from partially loaded weights. These
loading choices are intentional differences from the unarchived notebook and
were exercised by the reported successful runner execution; they do not establish
per-box equivalence with the original notebook. There is no global torch.load monkeypatch.

The script enters OpenPCDet's `tools/` directory while resolving base YAMLs,
checks the imported package against the requested checkout, and records the actual
Git revision/modified-file list. It records input/checkpoint/config hashes,
resolved config (including preprocessing and postprocessing thresholds), device
0, visible GPU count, Python/package/runtime versions, toolkit/driver output when
available, and effective flags. It does not claim the supplied source URL is
independently verified on every run. Existing output files are never overwritten.
Keep stdout/stderr, `pip freeze`, `nvcc --version`, `nvidia-smi`, config/base YAMLs,
and the reviewed upstream patch with the notebook export for a future full audit.

CPU-only validation of a real input, without any GPU imports or prediction file:

```bash
python scripts/kaggle_pointpillars.py \
  --input data/kitti/training/velodyne/000000.bin \
  --input-sha256 0e09c85e3f6078ecbdd1e706ee9624519f1bd29417437167a9ed7fbe6f54b4b1 \
  --validate-input-only
```

## Output semantics and remaining work

New output includes frame identity, checksum/provenance/environment information,
class counts, and raw OpenPCDet boxes `(x,y,z,dx,dy,dz,heading)` in Velodyne
coordinates, geometric centers, meters, and radians. Class IDs are one-based in
`cfg.CLASS_NAMES` order. Scores must be probabilities in [0,1]; raw logits and
non-KITTI seven-parameter boxes are rejected. No labels/calibration are loaded,
no ground-truth matching is performed, and no accuracy fields are invented.
The runner retains the configuration's range filtering, score threshold, and NMS;
it does not force the reported count of 29 or hardcode class predictions.

Still open: archive the original notebook/logs/JSON, establish exact effective
configuration and installation details, record checkpoint usage terms, compare
individual boxes/scores, archive the successful adapter GPU capture,
and implement scientific evaluation separately. Matching counts alone would not
establish accuracy or
byte-identical reproduction. Keep all datasets, weights, downloaded archives,
and generated outputs outside tracked files; default outputs go under ignored
`outputs/`. Do not save generated JSON under scripts/ or tests/.

## Issue acceptance evidence

- **RV-005:** The experimenter's successful real-sample execution, hardware,
  software stack, and workarounds are recorded, including successful standalone
  runner execution in PR #54. Original setup archival remains follow-up work.
- **RV-019:** OpenPCDet is the provisional, reported-working choice with a pinned
  revision, comparison boundary, code license, checkpoint pairing, and explicit
  limitations. This does not commit Raven's core to CUDA or claim another backend
  was benchmarked.
- **RV-020:** The official checkpoint source/config association and matching
  checkpoint checksum were independently checked; weights remain untracked.
  Weight usage terms and historical effective YAML bytes remain to be archived.

Issue status is tracked on GitHub. Standalone GPU execution is validated for the
reported stack; adapter integration, box/score equivalence and accuracy remain open.
See the [KITTI dataset policy](kitti-dataset.md) before interpreting future metrics.
