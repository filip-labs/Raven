# Experiment reproducibility (RV-035)

This methodology covers RV-030 configuration, RV-031 execution, RV-032 evaluation,
RV-033 benchmarking and RV-034 export. It establishes conditional, inspectable
consistency. Matching JSON/CSV hashes does not authenticate GPU execution or
establish detection accuracy. No production preprocessing, inference, evaluation
or timing algorithms were changed for this validation.

## Contract and comparison policy

| Category | Expected behavior | Verification |
| --- | --- | --- |
| Bitwise deterministic | Normalized configuration, ordered frame selection, split identity, input bytes, seeded processed XYZI, unchanged-source export content and CSV | Exact values/SHA256 under identical inputs and compatible software |
| Numerically reproducible | AP_R40 values for the same protocol/class/difficulty/IoU and available inputs | Absolute tolerance `1e-8` percentage points; relative tolerance zero |
| Environment dependent | Actual latency samples/statistics, timestamps, software/hardware descriptions, output locations and enclosing artifact hashes | Recorded and reported; equality is not required |

AP tolerance reuses RV-032's fixture policy. It is small compared with the
metric's recall discretization and permits only floating-point rounding; it
never relaxes IoU thresholds, eligibility, availability or class identities.
Null is unavailable, not zero. Matching null metrics alone cannot establish
numerical reproduction. Real wall-clock latency is never required to match.
Benchmark statistics are recomputed against each source's own samples using the
existing RV-034 `rel_tol=1e-12`, `abs_tol=1e-12 ms` policy. Counts, warmup/measured
labels, clock domains, units, percentile interpolation (`linear`), population
standard deviation (`ddof=0`) and nested-stage relationships remain exact.

Processed hashes cover ordered little-endian float32 XYZI bytes. They require
identical input, stage order, parameters and RNG state in a compatible environment.
Float64 intermediate calculations, float32 storage, NumPy/SciPy versions and
platform changes can affect rounding or neighbor behavior. A cross-environment
hash match is useful evidence, not a universal portability guarantee.

Experiment selection uses the existing explicit unsigned 64-bit seed policy;
noise stages separately require a nonnegative integer seed, including integers
larger than 64 bits. Noise uses a local NumPy `Generator(PCG64(seed))`. Experiment
seed does not replace a stage's explicit noise seed. Repeated stages and their
order are significant. Global NumPy RNG state is not reset.

`raven.reproducibility.policy.projection()` version 1 partitions fields explicitly:

- Preserve configuration except `output`, canonical split, ordered frame IDs,
  source type/status/evidence kind, declared model settings and preprocessing.
  Configuration hashes are preserved in reports. Changing the output directory
  legitimately changes the original canonical configuration hash; the derived
  scientific projection digest is separately named and never replaces it.
- Runs preserve scientific summaries, frames, geometry/predictions, processed
  hashes/counts, input hashes/sizes and stage metadata. Partition output locations,
  run timestamps/environment, recorded input paths, pipeline/stage durations and
  known pipeline software-version fields. Live model environment/version/config
  location is variable; model asset identities remain scientific. Historical
  saved-capture metadata remains intact, with its evidence type preserved.
- Benchmarks preserve selection, specification, methodology, model provenance,
  frame/input identities, observation status/phase/index/counts/hashes/predictions,
  operation counts and timing-series contracts. Partition raw durations, numeric
  statistics, audit/output locations and initialization measurements.
- Evaluations preserve protocol, thresholds, purpose (including training-smoke),
  metric identities/availability, diagnostics, warnings and failures. Rows are
  compared by metric/class/difficulty and diagnostic frame identity. AP values
  alone use numerical tolerance. Execution provenance and enclosing run/config
  hashes remain separately reported variable evidence; direct successful
  evaluations additionally validate `source_execution` against their linked run.
- Export generation metadata is not scientific content. Original source hashes,
  configuration identities and environments are retained in the report.

Unknown top-level scientific fields remain comparable. This is a deliberately
conservative comparison, not a general path-remapping or archive migration tool.
Dataset/model paths in configuration must still match. Differences reported as
`environment_differences` include output/source locations and other explicitly
variable metadata; they are not all hardware differences. Source contents are
never rewritten to make comparisons pass.

## Verification workflow

Install Raven normally; SciPy is needed only for replaying statistical filtering.
Saved-result inspection requires no PyTorch, CUDA, OpenPCDet, spconv, Modal,
Open3D or graphical desktop. Use a **new** report filename whose parent exists:

```bash
# Run/evaluation/benchmark source, or RV-034 export's final manifest.json:
raven verify-results /path/to/bundle/manifest.json
raven verify-results /path/to/bundle/manifest.json --check-sources
raven verify-results /path/to/run/baseline.manifest.json --check-inputs
raven verify-results /path/to/run/baseline.manifest.json --replay-preprocessing
raven verify-results /path/to/run-A/baseline.manifest.json \
  --compare /path/to/run-B/baseline.manifest.json --replay-preprocessing \
  --output /existing/report-directory/new-verification.json
```

The equivalent Python API is
`raven.reproducibility.verify_results(path, compare=None, check_sources=False,
check_inputs=False, replay_preprocessing=False)`.

Default export inspection verifies self-contained file sizes/hashes, deterministic
content digest, source inventory and byte-exact CSV projection. It does not read
original dataset/source paths. Direct source inspection uses existing strict
RV-034 loaders and source-reference requirements. Export `--check-sources` enables
those original checks; `--check-inputs` and replay imply that step. Input checks
verify original regular files, sizes and SHA256 without regenerating identities.
Replay runs only CPU preprocessing and compares processed hash/count/frame
metadata. It is bounded to 32 total source-frame records and 128 MB per scan,
with no implicit subset selection. Sources must have successful recorded results.

`--compare` compares the sources in their recorded order, with equal source counts
and types. Both independently loaded artifacts must pass their relevant integrity
checks. Original-source/input/replay flags apply to the primary source; the
comparison artifact is independently inspected and compared, without silently
replaying its inputs. To verify both original sets explicitly, invoke the flags
once for each side. For export archives, request `--check-sources` on each side
when original-source verification is required.

The report separates artifact integrity, original source references, original
inputs, CPU preprocessing replay, deterministic comparison, numerical comparison,
actual model inference and cross-environment reproduction. It includes original
configuration identities, canonical split, frame/input identities, tolerances,
scientific projection digests and explicit variable-field differences.

Overall statuses are `integrity_only`, `checks_passed`, `unavailable` or `failed`.
`checks_passed` covers the requested implemented checks, not every possible claim.
Model inference is always `not_performed`; cross-environment reproduction is
always `not_established`. Blocked/failed results can be inspected but cannot prove
successful scientific replay. Exit codes: 0 for integrity-only/passed checks,
1 for failed verification, 3 for unavailable checks; command/report-file errors
use argparse's exit code 2. Reports use exclusive creation, without overwriting.

## Retention and trust

An RV-034 archive needs its final manifest and all listed JSON/CSV files for
self-contained inspection. Embedded source records preserve original locations,
hashes and scientific values. Offline checks establish internal consistency,
not authenticity: an attacker can forge an entirely self-consistent archive.
No signature or independent execution attestation is added.

Original source validation additionally needs the original run manifests and
referenced configuration/frames/provenance/results/started files, benchmark or
evaluation sources, linked runs and external selection manifests. Input checks
and replay need the original files at recorded locations: scans and any recorded
labels/calibration/saved captures. AP reruns also need genuine images and all
RV-032 inputs. A future inference rerun requires licensed checkpoint/config,
pinned upstream code, compatible GPU environment and actual model execution;
this verifier never loads weights. Retain trusted external hashes and original
artifacts separately. Relocating data is not silently accepted as replay.

## Regression scope and repeated-run procedure

Existing suites already test configuration validation/split membership, duplicate
frames, ordered/repeated pipelines, independent synthetic preprocessing oracles,
RV-032 reference fixtures, RV-033 fake-clock statistics and RV-034 schema/CSV/
formula safety. RV-035 adds 49 cases for the missing cross-stage guarantees:
independent runner repeats through all six variants, separate Python processes,
saved-capture identity, seed/hash changes, numerical perturbations, metric-order
comparison, fake and real clock separation, same-source versus independent export
identity, offline retention, CLI behavior, and end-to-end corruption/provenance
failures. Corruptions affect only synthetic pytest temporary data. No real scans
or existing artifacts are modified. Standard tests need no GPU/network/downloads.

To repeat on your own licensed data:

1. Validate the configuration and retain canonical split/frame manifest and input
   checksums. Keep identifiers, selection, model declarations and stage order fixed.
2. Execute RV-031 twice with only `output.directory` changed to distinct, fresh
   locations. Keep explicit noise seeds fixed. Saved predictions are allowed only
   where RV-031 validates their input binding; do not apply them to modified clouds.
3. For preprocessing variants, run RV-033 CPU mode independently twice with the
   same benchmark specification. Compare processed hashes, never raw time equality.
4. Evaluate completed compatible inputs via RV-032 when genuinely available;
   export each source via RV-034, then run the verification commands above.
5. Re-export an unchanged source twice: deterministic content hash and CSV bytes
   must match. Independent executions may have different raw content hashes because
   they include original timestamps, paths, timings and original source hashes.
6. Retain reports, versions, source commit, normalized configs and original hashes.
   Record missing checks and failures alongside successful ones.

See [runner](experiment-runner.md), [benchmarking](latency-benchmarking.md),
[evaluation](kitti-evaluation.md) and [export](results-export.md) guides for the
existing execution commands. The verifier does not execute an experiment or AP.

## Bounded local KITTI evidence

Executed on 2026-11-26 from clean source
`b0018f1e756b5b613a00b996942f1f5b4350def1` on macOS 26.6.2 arm64,
Python 3.11.12, NumPy 2.4.6, SciPy 1.17.1. Frame `000000` has 115,384 points,
1,846,144 bytes, SHA256
`0e09c85e3f6078ecbdd1e706ee9624519f1bd29417437167a9ed7fbe6f54b4b1`.

Each CPU variant ran twice into fresh independent directories, each with one
warmup and two measured repetitions; explicit CPU replay also passed. All
observations and both executions agreed on processed hashes and counts:

| Variant | Output points | Processed XYZI SHA256 |
| --- | ---: | --- |
| Identity | 115,384 | `0e09c85e3f6078ecbdd1e706ee9624519f1bd29417437167a9ed7fbe6f54b4b1` |
| Range | 62,853 | `24ee9bd452bb3103d7016359253ac8f953e18c4926c953a0e4e35dd1b5f2f86f` |
| Voxel 0.2 m | 22,595 | `d6f7459d1a72dd3e70e558da6222e487e862fac7e9dece7c587f3c00c8f4e928` |
| Statistical | 111,261 | `60675f745ee361b114d2bfef4dc07084ea0d31b04e592bd90a50457804791bae` |
| Noise | 115,384 | `8ce06b2bd22bfcd5579c37cfc05a6fef1abe9430e826ecf58c482a50fdaf6d2c` |
| Range → voxel → statistical → noise | 12,232 | `a1f5be2fb87d17521feae88c1dafcd68367ecca326630656fd60d46e65f264e0` |

Range bounds are inclusive X `[0,70]`, Y `[-40,40]`, Z `[-3,1]` m.
Statistical parameters: K=16, std ratio=1. Noise: std=0.01 m, seed=42.
Full pipeline uses those stages and voxel=0.2 m in the listed order.
Every pair retained distinct original configuration hashes (different output
locations) and matching scientific projections. Real timings varied; each
artifact's statistics passed validation. Restricted sandbox `sysctl` queries
printed permission diagnostics; execution and checksum checks succeeded.

Two real RV-031 saved-baseline runs preserved the same historical 29 detections
(Car 4, Pedestrian 19, Cyclist 6), capture identity and scan binding. These runs
read saved predictions; they do not reproduce PointPillars GPU inference.
Original scan, capture, calibration and labels were SHA256-checked before/after
and unchanged. No capture, scan, checkpoint or generated artifact is committed.

Both real evaluation attempts remained **blocked**: genuine
`training/image_2/000000.png` is unavailable. They contain no AP or diagnostic
values. Training-frame smoke evidence is not held-out validation accuracy.
Numerical AP reproduction is covered only by the existing evaluator's synthetic
completed inputs and controlled tolerance tests in this environment.

All 12 CPU export bundles passed offline integrity checks; each independent pair
also passed scientific/source/input comparison. Their raw content hashes differed
as expected. Mixed saved-run/blocked-evaluation bundles remained inspectable.
Two exports of the identical saved source had equal content hash and byte-identical
CSV. Local reports and the execution script are retained under ignored
`outputs/rv035/`; they contain machine-specific paths and are not distributed.

## Remaining work

Future Modal experiments must retain pinned Raven/OpenPCDet revisions, image and
CUDA/runtime metadata, GPU details, checkpoint/config hashes and rights, canonical
split/ordered frames, input identities, seeds, raw prediction artifacts and
synchronized timing methodology. Run real inference independently, then use
appropriate numerical prediction comparisons; saved-capture replay cannot supply
that evidence. PR #77 is outside this task and was not changed or executed.
Cross-hardware/version experiments, genuine held-out AP reproduction, independent
GPU authenticity and statistical latency repeatability require additional
controlled evidence. Matching counts/hashes alone establishes none of these.
