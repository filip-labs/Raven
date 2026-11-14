# Numerical PointPillars reference validation (RV-024)

The CPU-only comparator validates two schema-v1 prediction captures using the
[RV-023 reader](prediction-conversion.md). It reports numerical agreement and
provenance verification separately. **Real reference-versus-adapter GPU validation
is pending.** The historical T4 reports establish execution/counts, not per-box
numerical equivalence or KITTI detection accuracy. No real paired captures were
available in the repository/Downloads/attachments search for this update.

## Matching and tolerances

Each class is matched independently. Sort both sides by center, dimensions,
wrapped yaw, confidence, then original index for exact ties. Form all Euclidean
center distances, discard edges beyond `max_match_distance`, and solve a global
rectangular Hungarian assignment with dummy unmatched slots. Dummy penalties
ensure maximum cardinality first, then minimum total center distance. Ascending
traversal breaks equal-cost ties deterministically; no candidate or reference
index can be reused. No SciPy dependency is required.

This is geometric matching, not array-order matching. Dimensions, yaw and score
are checked after assignment; they do not influence center-distance cost. When
multiple objects have indistinguishable/ambiguous centers, this policy may fail
an otherwise possible alternative pairing. Inspect the reported indices/errors;
do not select whichever assignment makes an experiment pass. The matching gate
is separate from, and much wider than, the numerical acceptance tolerances.

| Setting | Default | Meaning |
| --- | ---: | --- |
| `position_atol` | 0.0001 m | Each center component |
| `dimension_atol` | 0.0001 m | Each L/W/H component |
| `yaw_atol` | 0.0001 rad | Wrapped angular difference |
| `score_atol` | 0.00001 | Confidence difference |
| `rtol` | 0.00001 | Relative allowance anchored to reference |
| `max_match_distance` | 1 m | Euclidean center gate for pairing only |

For position/dimensions/score, require `abs(candidate-reference) <= atol +
rtol*abs(reference)` per component. For yaw, wrap each angle modulo 2π and require
`abs(wrap(candidate-reference)) <= yaw_atol + rtol*abs(wrap(reference))`.
Angles equivalent modulo 2π agree; π-flipped boxes are not treated as equivalent.
These conservative defaults allow float32 representation/rounding differences;
they are not empirical guarantees for every CUDA kernel or a measure of accuracy.
Choose and record tolerances before inspecting real results. Nonfinite/negative
settings fail; matching distance must be positive. Never tune predictions or
thresholds to reproduce the reference count.

The report contains matched original indices, per-pair errors/pass status,
unmatched indices on each side, counts/classes, maximum/mean Euclidean position
error, maximum absolute dimension-component error, maximum wrapped yaw error,
and maximum confidence difference. Empty matched sets have null metrics rather
than invented zeros. Two empty prediction sets can PASS only with valid matching
provenance. Count/class mismatches, unmatched boxes, or any exceeded tolerance
produce overall FAIL.

## Provenance gate and schema compatibility

All original schema-v1 geometry checks apply: frame IDs must match, positions
are geometric centers in KITTI Velodyne meters, dimensions are LWH, heading is
radians about +Z, and both representations/counts must agree when present.
Additional top-level provenance fields are additive; schema version remains 1:

- `inference_run_id`: fresh UUID hex per invocation;
- `raven_git_commit` and `raven_modified_files`: exact checkout identity/state;
- `openpcdet_diff_sha256`: hash of the binary tracked diff against HEAD;
- reference-only `reference_dataset_sha256`: hash of upstream `tools/demo.py`.

Existing metadata supplies input frame/point count/SHA256, checkpoint SHA256,
configuration path/hash and resolved config, OpenPCDet revision/modified files,
runner source hash, environment/dependency versions and inference entrypoint.
Both runs must declare the same input/checkpoint/config hashes, point count,
OpenPCDet and Raven revisions, tracked patch identity/list, GPU name, Python,
PyTorch/CUDA runtime, OpenPCDet/NumPy/spconv-cu126 versions and effective JIT/arch
flags. Cumm/toolkit/driver/device details are also recorded where available.
Config paths may differ; their hashes and inference settings must agree.

Canonical JSON for `effective_config.MODEL`, `DATA_CONFIG` and `CLASS_NAMES` must
match, including types. This checks resolved base-YAML settings as well as the
top-level config hash. Other sections such as ROOT_DIR/optimization are recorded
but excluded from this settings comparison. Checking all DATA_CONFIG fields is
conservative even when a field affects training only. Missing critical fields,
malformed hashes, dirty/unknown Raven source state, wrong entrypoints or reused
run IDs cannot produce a verified PASS. Runner hashes must identify different
source files. Git lookup failures are recorded as null; no commit is invented.

Reference entrypoint: `OpenPCDet.DemoDataset+model.forward`.
Candidate entrypoint: `PointPillarsDetector.detect`.
Older captures remain readable by RV-023 and can yield geometry diagnostics here,
but overall verification FAILS when the new critical provenance is absent.
`numerical_pass` alone does not establish `pass`; `provenance_pass` must also hold.

These are checks on declared evidence, not cryptographic attestation of a GPU
execution. Preserve trusted logs/notebooks and review upstream patches. The diff
hash covers tracked modifications, not all possible untracked code or installed
binaries. Matching software/GPU names does not prove identical drivers/kernels.
No report authenticates checkpoint rights, measures AP/mAP, or proves independent
generalization. Frame 000000 is a training split member.

## Independent reference path

[`scripts/openpcdet_reference.py`](../scripts/openpcdet_reference.py) uses the
[pinned upstream demo](https://github.com/open-mmlab/OpenPCDet/blob/233f849829b6ac19afb8af8837a0246890908755/tools/demo.py).
It verifies the pinned `tools/demo.py` SHA256
`f75942984a03ae605862bfd29d0e664a9e9676624cf9cb17256ea4219815dcfe`,
then loads the original `DemoDataset` class body unchanged through AST extraction,
omitting top-level GUI imports and rendering. This executes trusted upstream
Python code, never JSON content; use only the reviewed pinned checkout. No
upstream source file is silently edited. The class itself reads the scan and
calls upstream preparation; the script then independently builds, loads and
forwards an OpenPCDet model. It never calls Raven's adapter or OpenPCDetSession.

Shared setup/hash/config helpers and schema serialization keep conventions
consistent without sharing an inference result. The shared CLI's input loader
performs preliminary validation/hashing; reference inference reads the scan again
through the upstream DemoDataset. Unlike the GUI demo's permissive loader, this
headless reference strictly loads the full state with `weights_only=True`; trusted
pickle requires explicit `--trust-checkpoint`, with no unsafe retry. This deliberate
safety/loading difference is recorded, not presented as byte-for-byte execution
of the unmodified demo CLI. Both paths use full-scan/range preprocessing rather
than KittiDataset image-FOV cropping.

## Kaggle workflow — not yet executed for this change

Use the already established external T4 environment and your licensed files.
See [adapter setup](pointpillars-adapter.md) for compatibility, checkpoint terms,
JIT flags and the optional explicitly reviewed Argoverse workaround. The complete
historical environment lock is unavailable. Standard CPU CI installs no GPU stack.

Use a clean Git checkout with editable installation so source commit provenance
can be resolved. Paths below assume your scan/checkpoint are in `/kaggle/working`;
replace the scan path with your actual licensed input. Do not overwrite older
captures: choose a fresh output directory, as below. Run the two scripts as
**separate fresh Python processes**; each loads and executes its own model.

```bash
cd /kaggle/working/Raven
git fetch origin
git switch feat/rv-024-reference-validation
python -m pip install -e .
git status --short
git -C /kaggle/working/OpenPCDet rev-parse HEAD
# Required upstream revision: 233f849829b6ac19afb8af8837a0246890908755
export CUMM_DISABLE_JIT=1
export SPCONV_DISABLE_JIT=1
export TORCH_CUDA_ARCH_LIST=7.5
mkdir /kaggle/working/rv024-validation

python scripts/openpcdet_reference.py \
  --input /kaggle/working/000000.bin \
  --input-sha256 0e09c85e3f6078ecbdd1e706ee9624519f1bd29417437167a9ed7fbe6f54b4b1 \
  --openpcdet-root /kaggle/working/OpenPCDet \
  --config /kaggle/working/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml \
  --checkpoint /kaggle/working/pointpillar.pth \
  --checkpoint-sha256 4c83fc0fa02575b9b3e9dec676f698e7a70bb5a795e89f91df8a96b916fa19e2 \
  --checkpoint-source https://drive.google.com/file/d/1wMxWTpU1qUoY3DsCH31WJmvJxcjFXKlm/view \
  --device-index 0 \
  --output /kaggle/working/rv024-validation/openpcdet_reference_000000.json

python scripts/kaggle_pointpillars.py --adapter \
  --input /kaggle/working/000000.bin \
  --input-sha256 0e09c85e3f6078ecbdd1e706ee9624519f1bd29417437167a9ed7fbe6f54b4b1 \
  --openpcdet-root /kaggle/working/OpenPCDet \
  --config /kaggle/working/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml \
  --checkpoint /kaggle/working/pointpillar.pth \
  --checkpoint-sha256 4c83fc0fa02575b9b3e9dec676f698e7a70bb5a795e89f91df8a96b916fa19e2 \
  --checkpoint-source https://drive.google.com/file/d/1wMxWTpU1qUoY3DsCH31WJmvJxcjFXKlm/view \
  --device-index 0 \
  --output /kaggle/working/rv024-validation/raven_adapter_predictions_000000.json

python -m raven.perception.detection.reference_validation \
  /kaggle/working/rv024-validation/openpcdet_reference_000000.json \
  /kaggle/working/rv024-validation/raven_adapter_predictions_000000.json \
  --output /kaggle/working/rv024-validation/comparison_report_000000.json
```

Only if required by reviewed, trusted checkpoint metadata, add
`--trust-checkpoint` explicitly to both inference commands. Keep the actual error
and decision with the run evidence. A successful command returns 0; comparison
FAIL returns 1, and invalid CLI settings/output handling return 2. Corrupt/missing
capture inputs generate a FAIL report. Existing capture/report files are never
overwritten. Output parent directories must already exist.

Historical count target: 29 (Car 4, Pedestrian 19, Cyclist 6). Record actual outputs
and discrepancies; count equality is insufficient. Download/persist all three
artifacts **before terminating the Kaggle session**, plus stdout/stderr, notebook,
`pip freeze`, `nvidia-smi`, `nvcc --version`, Git commits and full upstream diff.
For example, in a notebook cell:

```python
from IPython.display import FileLink, display
for name in ("openpcdet_reference_000000.json", "raven_adapter_predictions_000000.json",
             "comparison_report_000000.json"):
    display(FileLink("/kaggle/working/rv024-validation/" + name))
```

Download the linked files to your private storage and verify their SHA256 hashes
against the report's capture hashes. Keep weights, data and full prediction
captures out of Git. The comparison report can be inspected locally with only
the standard Raven core installed.

## API, report example and tests

Use `compare_files(reference_path, candidate_path, Tolerances(...))` or
`compare_documents(reference_mapping, candidate_mapping, tolerances)`.
Invalid schema/conventions/sample IDs raise clear errors through the API; the CLI
turns them into a verification FAIL report. Existing captures are read-only and
the report hashes exactly the bytes parsed.

Abbreviated output from an exact **synthetic** one-box example (not a GPU result):

```json
{
  "status": "PASS",
  "pass": true,
  "numerical_pass": true,
  "provenance_pass": true,
  "counts": {"reference": 1, "candidate": 1},
  "metrics": {
    "max_position_error_m": 0.0,
    "mean_position_error_m": 0.0,
    "max_dimension_error_m": 0.0,
    "max_yaw_error_rad": 0.0,
    "max_confidence_difference": 0.0
  }
}
```

```bash
python -m pytest -q tests/test_reference_validation.py tests/test_openpcdet_reference.py
python -m pytest -q
python -m ruff check .
python -m pip check
git diff --check
```

Synthetic tests cover exact/reordered/rounded outputs, mismatched geometry/classes,
missing/duplicate boxes, confidence, angular wrapping, tolerance boundaries,
global one-to-one assignment and deterministic ties, invalid JSON/frame/provenance,
empty sets, read-only/no-overwrite behavior, and GPU-free imports. Reference-path
mocks forbid calls to Raven inference and verify its independent dataset/forward
boundary. These tests do not satisfy the pending real GPU comparison criterion.
RV-024 must remain open until a real paired comparison passes and is accepted.
