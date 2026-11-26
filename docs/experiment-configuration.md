# Experiment configuration (RV-030)

Schema v1 declares an experiment; it does **not** run preprocessing/inference,
load weights, render a scene, create output directories, or evaluate AP/mAP.
NumPy is the only runtime dependency. Even a statistical-filter declaration
validates without SciPy; execution may later require the preprocessing extra.

## Inspect and validate

```bash
raven inspect-experiment configs/experiments/saved-baseline.json
raven inspect-experiment /path/to/my-experiment.json
```

Success prints normalized configuration, `config_sha256`, selected frame IDs,
selection provenance (algorithm/NumPy version/manifest hash), physical scan
directory and planned artifact paths; exit 0. Invalid/missing/malformed inputs
print a contextual argparse error to stderr and exit 2. Inspection writes nothing.
The shipped example uses placeholder absolute paths: successful validation is
**not proof that assets exist or that an experiment is executable**.

## JSON schema v1

All fields below are required unless explicitly optional. Unknown keys at every
object level, duplicate JSON keys, nonfinite numeric parameters and unsupported
schema versions fail rather than silently defaulting. Names/IDs are supplied by
the caller; no clock/random UUID influences configuration identity.

```json
{
  "schema_version": 1,
  "name": "Saved PointPillars baseline",
  "experiment_id": "kitti-val-baseline-001",
  "dataset": {"root": "/path/to/kitti", "split": "val"},
  "frames": {"ids": ["000001", "000002", "000004"], "sample_size": 2},
  "model": {
    "name": "pointpillars", "source": "saved",
    "predictions_dir": "/path/to/predictions"
  },
  "preprocessing": [],
  "seed": 42,
  "output": {"directory": "/path/to/outputs", "artifact_prefix": "baseline"}
}
```

| Field | Meaning / validation |
| --- | --- |
| `name` | Nonempty human-readable text |
| `experiment_id` | Caller-chosen run ID: 1–64 ASCII letters/digits/underscore/hyphen, first character alphanumeric |
| `dataset.root` | KITTI object-detection root; train/val map to `training/velodyne`, test to `testing/velodyne` |
| `dataset.split` | Exactly `train`, `val` or `test`; no implicit split |
| `frames.ids` | Nonempty ordered unique six-ASCII-digit strings; exclusive with manifest |
| `frames.manifest` | UTF-8 text with exactly one six-digit ID per line; no blanks/duplicates; exclusive with ids |
| `frames.sha256` | Optional expected lowercase SHA256 for manifest; populated automatically in normalized exports; forbidden for explicit IDs |
| `frames.sample_size` | Optional integer 1..candidate count; no replacement, preserve selected input order |
| `seed` | Explicit JSON integer 0 <= seed < 2**64; booleans/floats rejected |
| `model` | PointPillars declaration described below; assets not loaded |
| `preprocessing` | Required ordered array in the existing RV-017 flat stage format; empty array is valid |
| `output.directory` | Output base; no directories are created by validation |
| `output.artifact_prefix` | Same filename-safe grammar as experiment_id |

The split is a declaration, not an assertion that explicit IDs belong to the
canonical split. No directory glob or automatic inference selects frames. For
canonical train/val experiments use the versioned manifests documented in
[KITTI dataset policy](kitti-dataset.md), or an explicitly recorded subset.
RV-031 must verify selected membership, file availability and required labels
before execution; test has no released ground truth. Configuration validation
alone does not certify membership, file content or evaluation eligibility.

### Paths and manifests

Relative paths resolve against the **configuration file's directory**, not cwd;
`ExperimentConfig.from_dict(..., base_dir=...)` requires an explicit base.
Use absolute paths when referring to shared assets outside that directory.
Parent traversal (`..`), `~`, `$` substitutions, empty/NUL paths are rejected;
there is no environment expansion. Resolved paths account for existing symlinks.
Exports store absolute resolved paths and bind manifests to their exact byte
SHA256 (including line endings). Reloading a changed manifest fails the checksum;
an already loaded object retains its original immutable candidate snapshot.

Nonexistent dataset/model/prediction/output paths are permitted for planning or
an external GPU environment. A manifest must exist to resolve selection. Existing
dataset/prediction directories and config/checkpoint files must have appropriate
file/directory kinds. Validation never hashes or deserializes checkpoint weights.

### Model and preprocessing compatibility

Only `model.name: "pointpillars"` is currently supported. `source: "saved"`
requires `predictions_dir` and forbids inference fields. It expects future
FRAME.json schema-v1 inputs through the existing RV-023 reader; configuration
loading does not read/validate those prediction files.

`source: "inference"` instead requires all four fields and forbids predictions_dir:

```json
{
  "name": "pointpillars",
  "source": "inference",
  "config": "/path/to/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml",
  "checkpoint": "/path/to/pointpillar.pth",
  "checkpoint_sha256": "4c83fc0fa02575b9b3e9dec676f698e7a70bb5a795e89f91df8a96b916fa19e2",
  "openpcdet_revision": "233f849829b6ac19afb8af8837a0246890908755"
}
```

Hashes/revisions require lowercase 64/40-digit hex. These are declarations,
not verified checkpoint/configuration provenance or licensing permission.
The existing PointPillars adapter enforces runtime compatibility with its pinned
upstream revision; syntax acceptance here does not support other GPU revisions.
Model YAML, resolved effective config, actual hashes, hardware and source state
must be checked and recorded by RV-031 before execution. See
[adapter](pointpillars-adapter.md) and [reference evidence](reference-validation.md).

Copy the **array contents** of `configs/preprocessing/{range,voxel,full}.json`
into `preprocessing`. Names, explicit parameters, boundary semantics, repeated
stages and ordering are unchanged. The new public
`validate_preprocessing_stages()` reuses the exact validator used by
`run_preprocessing_pipeline()` and returns immutable detached stage mappings
without execution. No new preprocessing configuration format is introduced.

Saved predictions plus **any** new preprocessing stage are rejected: this issue
cannot establish that a saved prediction was produced from that processed scan.
Use inference declarations for preprocessing experiments. Explicit per-noise-stage
seeds remain required and are never replaced by the experiment seed. That seed
currently controls frame sampling only; future model randomness is not silently
controlled or promised by RV-030.

## Determinism, identity and artifact naming

Frame sampling uses a fresh NumPy `Generator(PCG64(seed))`, choice without
replacement, then sorts selected **indices** to preserve manifest/explicit order.
No global RNG is altered. No sample_size means use all IDs exactly in input order.
Fixed input, seed and environment reproduce selection; NumPy sampling algorithms
may change across versions. Inspection records the algorithm name/version,
NumPy version and exact selected IDs for future run provenance.

`config_sha256` hashes UTF-8 compact JSON with sorted object keys, no insignificant
whitespace, and normalized validated values. Array order, repetitions, source
paths, seed, name/ID, outputs and bound manifest checksum matter; JSON object-key
order/formatting do not. Integer/float stage values normalize through RV-017.
The digest identifies the **declaration**, not external dataset bytes, effective
model YAML, prediction contents, hardware or a completed run. It is intentionally
path-sensitive; relocating an asset changes identity. It is not an authentication
or independent-inference attestation mechanism.

Planned run directory is `output.directory / experiment_id`; paths are
`artifact_prefix.config.json`, `.frames.txt`, `.provenance.json`, `.results.json`.
RV-030 reserves names but creates none. Reusing an ID can target the same run
folder even when configuration identity changes; RV-031 must refuse collisions
and preserve prior runs, not overwrite them based only on the ID.

Output must be disjoint from dataset/prediction directories (including ancestors
and children) and cannot contain an input manifest/model config/checkpoint.
Root/home outputs, file-valued directory ancestors, existing file-valued run paths,
or run/artifact paths redirecting through symlinks are rejected. These checks are
not a filesystem security boundary or a promise of permissions/free space.
RV-031 must recheck paths and use exclusive/atomic creation against races.

## Python API and RV-031 handoff

```python
from raven.experiments.config import load_experiment_config, save_experiment_config

config = load_experiment_config("my-experiment.json")
print(config.config_sha256, config.frame_ids)
print(config.dataset.scan_directory, config.artifact_paths)
# Explicit export only; parent must exist, existing files are never overwritten.
save_experiment_config("normalized-experiment.json", config)
```

Frozen typed `DatasetConfig`, `FrameSelection`, `ModelConfig`, `OutputConfig` and
`ExperimentConfig` hold validated declarations. `to_dict()` returns detached
JSON-compatible values; `from_dict()` / `load_experiment_config()` raise
`ExperimentConfigurationError` for invalid documents. Direct typed constructors
also validate and can raise standard ValueError/TypeError. Read-only inspection
has no execution side effects.

The [RV-031 runner](experiment-runner.md) consumes `config.frame_ids` in order and
validates dataset/split and
saved-prediction provenance, resolves/verifies actual model settings and assets,
runs existing `run_preprocessing_pipeline(cloud, config.preprocessing)`, and
persists configuration digest, bound selection, actual per-input hashes and
runtime/source/hardware metadata. It does not infer accuracy or silently fill
missing settings. RV-030 itself remains declaration-only; execution and minimal
persistence are described in RV-031. Metrics, timing and broader export remain
RV-032–RV-034 work.

Tests use synthetic manifests/configurations, seeded golden selection, round trips,
strict errors, output collisions/aliases, existing preprocessing configurations,
and fresh-process CLI imports with SciPy/Open3D/GPU frameworks blocked. Standard
CI needs no dataset, GPU, weights or graphical desktop.

## Explicit remote configuration (RV-049)

Legacy saved/local-inference configurations remain schema v1. Schema v2 is reserved
for `model.source: "modal"` with validated `model.remote` settings; incompatible
source/version combinations are rejected. No remote credentials or local model
paths are accepted there. Frame selection, ordered preprocessing, path safety,
identity and serialization rules are reused. See the [Modal guide](modal-inference.md)
and `configs/experiments/modal-pointpillars.json` for fields and opt-in execution.
Configuration inspection remains CPU-only and nonbillable.
