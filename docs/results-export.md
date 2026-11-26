# Versioned results export (RV-034)

The exporter reads **existing evidence**, validates its schema/manifests/internal
consistency, and writes versioned JSON and normalized CSV into a new directory.
It never runs a detector, loads a dataset/checkpoint, preprocesses points,
re-evaluates AP, or repeats a latency benchmark. Existing runner, evaluator,
benchmarking algorithms and timing boundaries are unchanged. Modal PR #77 is
not a dependency and no cloud operation is added.

## API and CLI

```bash
# Completed RV-031 manifest, not a config or predictions file:
raven export-results /runs/example/baseline.manifest.json \
  --format json --output /exports/run-001

# RV-032 result; referenced run artifacts must also be retained:
raven export-results /results/evaluation.json \
  --format csv --output /exports/evaluation-001

# Several explicit RV-033 manifests; include all warmup and measured raw rows:
raven export-results /benchmarks/a/manifest.json /benchmarks/b/manifest.json \
  --format all --raw-timings --output /exports/comparison-001
```

```python
from raven.exporting.export import export_results

summary = export_results(
    ["/benchmarks/a/manifest.json"], "/exports/new-bundle",
    format="all", raw_timings=True,
)
```

`--output` is always a **new directory**, with an existing parent. It is never a
file to replace. It cannot be inside a referenced run/benchmark directory, be an
existing path, traverse `..`, or redirect through a symlink. A standalone outcome
file in your working directory can be exported to a new child directory. Explicit
source order is preserved; no directory scanning/globbing or implicit matching is
performed. Limits are 1..32 sources, 128 MB per file and 256 MB per source bundle.
Larger archives require a future streaming extension, not silently truncated data.

Exit **0** means a valid export bundle was published, **2** means invalid input,
schema/integrity/usage or destination, and **1** means publication failed. A valid
blocked evaluation can export successfully: export success is not source experiment
or GPU success. Existing CLI commands retain their behavior.

Formats:

| Option | Files |
| --- | --- |
| json (default) | results.json, manifest.json |
| csv | experiment_summary.csv, evaluation_metrics.csv, benchmark_summaries.csv, metadata.json, manifest.json |
| all | Same CSV tables plus results.json and manifest.json |
| --raw-timings with csv/all | Also raw_timing_samples.csv |

The CSV metadata sidecar contains the same complete versioned JSON evidence
projection as results.json. Nested configurations, source payloads and provenance
are retained; CSV flattening does not discard them. Tables with no available rows
still have headers. `--raw-timings` with JSON-only output is rejected.

## Accepted sources and evidence validation

Source type comes from **exactly one schema discriminator in JSON**, never the
filename or extension:

- RV-031 `run_schema_version: 1`: final successful run manifest. The existing
  completed-run loader verifies config identity, selected IDs, completion and
  artifact checksums; export also checks predictions/counts/source labels.
- RV-032 `evaluation_schema_version: 1`: validated success, blocked or invalid
  result. When a run hash exists its referenced manifest and artifacts must be
  available and match. Success additionally requires consistent config, split,
  purpose, frame count, metric/diagnostic grid and numerical ranges.
- RV-033 `benchmark_schema_version: 1`: final success/failed benchmark manifest,
  with started/result artifact checksums. A failed completed benchmark exports
  diagnostic data under **failed**, not as a successful performance trial.
- Standalone RV-033 blocked/invalid outcomes are accepted only with no frames,
  initialization, model execution or output directory, and explicit failures/
  unavailable state. They have no benchmark/raw timing CSV rows.

Configs reuse RV-030 validation and pinned canonical split identities. Frame
IDs/order and source-type/version compatibility are checked. External selection
manifests referenced by a configuration must remain available for its existing
config validator; archives with missing required references are rejected rather
than repaired or assigned a new config digest. Dataset scans, images, calibration
and checkpoints are **not reloaded**: their recorded identities are preserved,
not newly authenticated. A source file named `.txt` is acceptable if its actual
schema is valid; an arbitrary `.json` is not.

Strict JSON rejects duplicate keys, nonfinite numbers and malformed shapes.
Safe manifest basenames, symlink checks, file hashes, source snapshots and a final
unchanged-source check guard reads/publication. Unsupported schemas, missing
artifacts, corrupt checksums, impossible counts, bad numeric types and inconsistent
records are rejected **before destination creation** where possible.

Benchmark summaries are validated using RV-033's existing aggregation helper
against the **measured-phase raw samples only**. No operation, GPU call or timer
measurement is rerun. Source summary values are exported unchanged, not replaced
by recomputed values. Validation permits relative 1e-12 / absolute 1e-12 ms float
rounding differences across libraries. Counts/methods/raw sample sequences must
match exactly; durations remain nonnegative integer ns. Nested/disjoint interval
consistency, frame counts, operation order, spec hash and stable processed hashes
are also checked. Injected clocks cannot be relabeled ordinary hardware evidence.

AP is not recalculated. The current supported RV-032 schema uses AP_R40, Car /
Pedestrian / Cyclist, easy / moderate / hard, and its recorded 0.7 / 0.5 / 0.5
IoU thresholds. Numeric/grid/diagnostic consistency is checked; these checks are
not independent validation of accuracy or certification of KITTI-server equivalence.

## JSON export schema v1

```json
{
  "export_schema_version": 1,
  "content_sha256": "<SHA256 of canonical content>",
  "content": {
    "sources": ["<Evidence records described below>"],
    "comparison": {"dimensions": {}, "pairs": [], "policy": "not automatically comparable"}
  },
  "generation": {"generated_at_utc": "<UTC>", "exporter": "raven-results-export"},
  "validation": {"gpu_authenticity": "not independently established"}
}
```

This is a **structural illustration**, not numerical experiment evidence. Each
source record contains:

- source_id (type + full observed source SHA256), source_type/schema version,
  original status and evidence_kind; exact artifact paths/hashes/byte sizes and
  known recorded hashes. Root JSON hashes are observed, not signed by a third party.
- Experiment name/ID, normalized configuration/config SHA256, dataset root/split,
  canonical split identity, ordered frame IDs and preprocessing parameters.
- Model identity split into **declared** configuration and **observed** source
  metadata when available. A saved capture's metadata remains declared history,
  not evidence of new local GPU execution; absent identity is null.
- Source hardware/software environment, warnings/unavailable fields and run hash.
- Full validated source payload: run artifacts, evaluation protocol/metrics or
  benchmark method/raw observations/summaries and model execution provenance.

Numeric values use their parsed Python JSON representations without presentation
rounding. Integer ns are retained, JSON null remains null, and source units remain
explicit. This preserves stored numeric meaning, not the original JSON whitespace
or arbitrary decimal spelling. Original bytes remain identifiable by SHA256.

Generation records UTC time, exporter/source revision/dirty state, Python and
exporter module hashes. Deterministic content excludes generation timestamp,
export destination and format metadata. Repeated equivalent inputs at the same
source paths/order yield identical content hashes and CSV bytes; generation
metadata and resulting JSON file hashes may differ. Relocating source paths or
changing source bytes legitimately changes content identity.

## CSV schema v1 and exact columns

All tables share the leading columns:
`export_schema_version, source_id, source_type, source_schema_version`.
The remaining columns are listed below **in order**. These names/order are the
public v1 contract; semantic/structural incompatibilities require new versioning.

### experiment_summary.csv

Grain: **one source result snapshot**, not one unique experiment_id across every
kind. An evaluation and its run can share experiment_id/config/run hash but have
different source_id and status. Join on explicit identities; do not count both as
two independent trials. Remaining columns:

```text
experiment_id, experiment_name, config_sha256, dataset_root, dataset_split,
canonical_split_json, frame_ids_json, preprocessing_json, model_json,
evidence_kind, source_status, run_manifest_sha256, source_artifacts_json,
warnings_json, unavailable_json
```

`*_json` cells are compact JSON, including arrays, nulls and nested metadata.
Source status is independent of successful export publication.

### evaluation_metrics.csv

Grain: one source/metric/class/difficulty combination. Remaining columns:

```text
experiment_id, run_manifest_sha256, dataset_split, evaluation_purpose,
evaluator_name, evaluator_version, protocol, metric, class_name, difficulty,
iou_threshold, eligible_ground_truth, ap_r40_percent, metric_availability,
evidence_kind, source_status
```

`metric` is 3d or bev; AP is a percentage. Available 0% is numeric zero;
no eligible ground truth is an **empty AP cell** with
`metric_availability=no_eligible_ground_truth`, not 0%. Blocked/invalid evaluation
has no metric rows and retains its status/failures in the summary/JSON. Training
smoke remains `training_smoke` / train, separate from validation_full/subset / val.

### benchmark_summaries.csv

Grain: one source/frame/stage series; initialization is a separate lifecycle row
with blank frame_id, warmup=0 and count/repetitions=1. Remaining columns:

```text
benchmark_id, experiment_id, frame_id, input_count, output_count, input_sha256,
hardware_json, evidence_kind, preprocessing_sha256, stage, clock_domain,
relationship, timing_unit, warmup, repetitions, count, mean_ms, median_ms,
minimum_ms, maximum_ms, std_ms, p50_ms, p90_ms, p95_ms, p99_ms,
percentile_method, std_ddof, frame_status, benchmark_status
```

Benchmark ID is the declared output directory basename without `.benchmark`;
source_id disambiguates snapshots/repeated names. Hardware JSON preserves the
actual recorded environment. `count` is measured samples only, summaries use
milliseconds, percentiles are linear/type 7 and std is population/ddof=0.
No summary is synthesized for missing/failed frame measurements. Complete earlier
frames of a failed benchmark remain diagnostics with benchmark_status=failed.
Never use them as a complete successful trial or pool overlapping timing stages.

### raw_timing_samples.csv (optional)

Remaining columns:

```text
benchmark_id, frame_id, iteration_index, phase, stage, duration, timing_unit,
evidence_kind, observation_status, benchmark_status
```

Duration is exact integer ns. Phase is warmup, measured or initialization;
initialization has no iteration index/frame ID. Successful completed intervals
from a failed observation can remain diagnostics, explicitly marked failed.
No duration is invented for an operation that never completed. Warmup rows are
preserved but must not be included in steady-state aggregates.

## Encoding, nulls, precision and spreadsheet safety

CSV uses UTF-8 without BOM, comma separators, CRLF records, and Python's
[csv writer](https://docs.python.org/3/library/csv.html) with all fields quoted and
embedded quotes escaped. Null scalars use empty cells, never zero; availability/
status columns and the JSON sidecar distinguish absence from measured zero.
Floats use their round-trip string representation, not rounded table formatting.
Frame IDs should be read as strings to preserve leading zeros.

Text starting with formula-prone = + - @ (including full-width forms), leading
control characters, or whitespace followed by such characters receives one
apostrophe prefix. Existing apostrophe-leading text also receives one prefix,
so `raven.exporting.csv_tables.restore_text` reverses the encoding unambiguously.
Only text columns are encoded; validated numeric columns remain numeric. JSON
keeps original text. Numeric/source identifiers are never evaluated as code.
[OWASP notes](https://owasp.org/www-community/attacks/CSV_Injection) that spreadsheet
save/reopen behavior can undo mitigations; CSV has no universal safety guarantee
across all applications. Import as text where appropriate and preserve the
original bundle rather than repeatedly editing/re-saving evidence in a spreadsheet.

## Multi-source comparisons and safe persistence

Comparison metadata records dataset/split, ordered frames, declared/observed
model, preprocessing, input sizes, protocol, hardware, timing scope, repetitions,
evidence kind and source status. Every explicit pair lists different/missing
dimensions and `comparability: not_assumed`. Agreement is not a controlled
experimental design. Apple CPU preprocessing, local CUDA adapter or future remote
service intervals are not silently pooled, ranked or assigned improvement
percentages. Missing metadata remains missing. Duplicate source evidence is rejected.

All validation/rendering occurs before reserving output. A new owner-only directory
is reserved exclusively; fixed filenames use RV-031's atomic exclusive publisher.
The final export manifest is written **last**, after a repeated unchanged-source
check. It contains source identities and hashes/sizes for all generated data
files. A successful bundle requires its manifest and matching file checksums.
On failure, only transaction-owned files matching prepared bytes are removed;
concurrently added/changed unrelated files are never deleted. There is no successful
manifest for a failed transaction. Source files are never modified.

## Reproducible Python/pandas analysis

Pandas is optional analyst tooling, not a Raven dependency. Its
[read_csv](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html) supports
explicit string IDs and round-trip float parsing:

```python
from pathlib import Path
import json
import pandas as pd
from raven.exporting.csv_tables import restore_text

bundle = Path("/exports/comparison-001")
summary = pd.read_csv(
    bundle / "benchmark_summaries.csv", encoding="utf-8",
    dtype={"frame_id": "string", "source_id": "string", "benchmark_id": "string"},
    float_precision="round_trip", keep_default_na=False,
)
for column in ("frame_id", "source_id", "benchmark_id", "stage"):
    summary[column] = summary[column].map(restore_text)
valid_cpu = summary.loc[
    (summary["benchmark_status"] == "success") &
    (summary["evidence_kind"] == "local_cpu") &
    (summary["timing_unit"] == "ms")
].copy()
valid_cpu["mean_ms"] = pd.to_numeric(valid_cpu["mean_ms"], errors="raise")
hardware = valid_cpu["hardware_json"].map(json.loads)
print(valid_cpu[["benchmark_id", "frame_id", "stage", "mean_ms"]])

metrics = pd.read_csv(bundle / "evaluation_metrics.csv", keep_default_na=False,
                      float_precision="round_trip")
available = metrics.loc[metrics["metric_availability"] == "available"].copy()
available["ap_r40_percent"] = pd.to_numeric(available["ap_r40_percent"], errors="raise")
# Never fill missing AP or latency with zero, or include warmup rows in summaries.
```

## Actual read-only integration and limitations

On 2026-11-26, six existing RV-033 CPU benchmark manifests for KITTI frame 000000
were exported: baseline, range, voxel 0.2 m, statistical K16/ratio1, noise
std0.01/seed42 and the full ordered pipeline. The export contains **20 timing
summary rows and 700 raw rows: 100 warmup, 600 measured**. Every parsed CSV summary
number and raw ns interval exactly matched its source JSON; units stayed ms/ns,
frame IDs retained leading zeros, and hardware remained Apple M1 Pro / arm64
with the complete original environment. All **18 source artifact files** remained
byte-identical. No benchmarking/evaluation/inference was rerun.

A separate mixed export consumed the genuine RV-031 saved run, existing blocked
RV-032 evaluation and a CPU benchmark. All three source types exported; the blocked
evaluation produced **zero metric rows**, not invented 0% AP. Its training/saved
provenance and missing-image failure were preserved. Local generated bundles and
checks are under ignored `outputs/rv034`, including real-export-verification.json;
none of these files, datasets or weights is committed.

This does not establish new accuracy/latency results or independently authenticate
GPU execution, source labels, dataset rights or hardware. A source author can
fabricate an internally consistent unsigned archive; checksums do not make it
trusted evidence. Failed/incomplete runs lacking successful completion are not
repaired. Unsupported future schemas/clock domains require explicit adapters and
versioning. No missing original asset is reconstructed, and no source path is
silently remapped.

RV-035 can retain these bundles alongside original logs/assets and verify their
manifests/content/source identities. The exporter is not a complete reproduction
or dependency-lock system. Future Modal/cooperative sources can add validated
schema adapters and separately identified clock domains without changing local
metric units or promoting saved/synthetic evidence to real inference.
