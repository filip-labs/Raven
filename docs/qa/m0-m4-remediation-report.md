# M0–M4 post-audit remediation and M5 readiness

Validation dates: November 17–18, 2026 (Europe/Belgrade). Base and latest fetched
`origin/main`: `b6ae4d4ff563254c4263cddbbf54051306cd44ee`.
Related existing issues: [RV-024](https://github.com/filip-labs/Raven/issues/24),
[RV-029](https://github.com/filip-labs/Raven/issues/29),
[RV-027](https://github.com/filip-labs/Raven/issues/27),
[RV-018](https://github.com/filip-labs/Raven/issues/18).

## Decision

**Ready to begin M5 development, with external/manual verification still open.**
No reproducible Raven production-code defect was established. No inference,
preprocessing or rendering implementation was changed, and M5 evaluation was
not implemented. Local editable CLI routing is verified working on current main.
The missing historical RV-024 pair and native physical `=`/modifier-drag checks
remain explicit gaps; this report does not certify independent GPU equivalence,
KITTI accuracy, cross-platform GUI behavior, or the absence of memory leaks.

## Original findings and disposition

| Original audit finding | Action / root cause | Current classification |
| --- | --- | --- |
| Normal CLI imported an older feature checkout | Editable install follows its source checkout. Original checkout had already moved to current main before remediation; no reinstall/overwrite was required. Normal activated CLI now exposes all options and reads real KITTI. | Verified working; prior routing symptom resolved before this task |
| Capture hash differed from PR60 candidate | Original pair/report not recovered. Current capture independently validates schema, counts and pinned config hash; preserve it as a separate artifact. Exact cause of differing bytes remains unknown. | Historical evidence unresolved; documentation completed |
| Fresh GPU inference unavailable | Mac has no trusted NVIDIA runtime/approved external execution session. No GPU packages/weights downloaded. | Blocked externally |
| Fresh independent numerical replay unavailable | Authentic historical paired JSON/report absent. Existing reference and adapter procedure now pins an available reviewed revision and records required evidence. | Blocked externally |
| Native point-size increment inconclusive | Official v0.20 backend uses unshifted EQUAL; automation equal had no observed effect. ABC layout confirmed. Raven leaves native bindings intact. | Environment/input limitation; physical verification required |
| Native modifier-drag pan untested | Tool rejects modifier-only key hold and offers no modified/middle drag. Backend source supports Ctrl+left or middle drag; real ViewControl.translate was tested through audit probe. | Environment/input limitation; physical gesture remains unverified |
| Escape / native window-close lifecycle untested | Fresh raw Escape and overlay native close-button sessions both destroyed windows and exited 0; comparison Q destroyed both windows and exited 0. | Verified working |
| Longer bounded GUI stability / other platforms untested | Fresh 141-second comparison session, 22 successful transitions, screenshots and recovery observed. Other platforms not exercised. | Bounded macOS check passed; other platforms not tested |
| M5 AP/mAP evaluation absent | Deliberately outside this task. | Deferred to M5 |

## Changes and local setup

- Added [local setup and activation](../local-development.md), including editable `.pth`,
  console-script/interpreter/source checks and safe handling of multiple checkouts.
- Updated [native controls](../scene-navigation.md#verified-backend-controls-and-macos-input-limitations)
  to distinguish point size from Shift-modified line width and actual native
  checks from programmatic renderer probes.
- Updated [reference validation](../reference-validation.md#november-1718-artifact-recovery-and-revalidation-status)
  with exact artifact hashes, captured environment/config/patch identifiers, unknowns,
  external execution steps and evidence retention. Replaced deleted feature-branch
  checkout instructions with the reviewed base SHA above.
- Added two CPU regression tests: interactive single viewer and both comparison
  windows must leave GLFW equal/minus/Q/Escape unbound by Raven while Raven
  function-key callbacks are active. These verify delegation, not Open3D native
  input behavior; no redundant custom point-size/mouse handler was added.
- README links this report and the supported local setup. No dependency or CI
  configuration changed. No issue, board or milestone status changed.

Existing installation at `/Users/filip_macbook/Downloads/Raven/.venv` imports
`/Users/filip_macbook/Downloads/Raven/src/raven/__init__.py`; `direct_url.json`
declares an editable installation of that checkout, its `.pth` selects `src`, and
the `raven` shebang selects the same venv Python. The original checkout remains
on main with its pre-existing `.gitignore` edit and four untracked camera/capture
files. Work was isolated on `codex/post-audit-m5-readiness` in the existing clean
managed worktree; other worktrees were not changed. Production files in both
tested checkouts match the base revision.

Verified activation/execution (no manual `PYTHONPATH` override):

```bash
cd /Users/filip_macbook/Downloads/Raven
source .venv/bin/activate
unset PYTHONPATH
which raven
python -c 'import raven,sys; print(sys.executable); print(raven.__file__)'
raven --help
raven visualize --help
raven inspect data/kitti/training/velodyne/000000.bin
```

The real inspect command returned frame 000000 / `kitti_velodyne` / 115,384
points. Help exposes `--navigate`, `--start-frame`, `--predictions-dir`,
`--screenshot-dir` and all existing overlay/comparison/color/point-size options.

## Tests and exact results

Environment: Python 3.11.12, NumPy 2.4.6, SciPy 1.17.1, Open3D 0.20.0,
pytest 9.2.0, Ruff 0.17.4, macOS 26.6.2 arm64. No GPU frameworks installed.

- Full branch test suite: **1035 passed, 3 skipped, 0 failed in 3.12 s**
  (baseline: 1033 passed, 3 skipped; two new tests).
- All three original optional local KITTI tests ran separately against the
  dataset-bearing original checkout: **3 passed in 0.49 s**. The managed
  worktree skips those checkout-relative paths; there is no configurable
  external dataset-root setting in those tests. No full dataset was moved/copied.
- Both pytest invocations report the existing external Dash plugin old-hook
  deprecation warning. No assertions were weakened or warnings filtered.
- `python -m ruff check .`, `python -m pip check`, `git diff --check`: exit 0.
- All three script help commands and all five real KITTI inspect/labels/calib
  CLI batches: exit 0. Missing-GPU paths remain outside live execution.
- Original tracked files, `.gitignore`, camera/capture files and all 16 used
  scan/label/calibration/capture hashes match the remediation baseline.

The existing `.github/workflows/ci.yml` was inspected: Ubuntu/Python 3.11,
editable `.[dev]` installation, Ruff and pytest; no GPU or rendering setup.
[Base main CI](https://github.com/filip-labs/Raven/actions/runs/43656301200)
is success for the base SHA; that is historical evidence, not this local run.
The remediation PR has its own fresh checks, reported separately on GitHub.

The full suite executes branch tests with the existing venv; its installed source
is current main with identical production bytes (this branch changes only docs/tests).
No temporary import override or environment repointing was used.

| Frame | Input | Range | Voxel 0.2 m on raw | Full pipeline |
| --- | ---: | ---: | ---: | ---: |
| 000000 | 115384 | 62853 | 22595 | 12232 |
| 000001 | 120268 | 61545 | 37873 | 16514 |
| 000002 | 126891 | 63761 | 16513 | 6841 |
| 000010 | 115875 | 52649 | 33042 | 11590 |
| 000100 | 123183 | 61899 | 29664 | 11199 |

All five frames ran the checked-in range/voxel/full configurations with three
additional exact point-output repetitions per configuration. Source rows remain
unchanged; annotation/calibration parsing and finite transforms pass. Frame-000000
documented counts **62,853 / 22,595 / 12,232** match. No accuracy/latency improvement
is inferred from reduced counts. First-run observational CPU times are retained
in `smoke-results.json`; fixed-environment repeats do not promise cross-version
or cross-platform floating-point identity.

## Native macOS GUI and stability

| Fresh session | Seconds | Navigation events / successful transitions | Pipeline calls | Peak RSS MiB | Exit |
| --- | ---: | ---: | ---: | ---: | --- |
| gui-controls | 105.0 | 0 / 0 | 0 | 176.6 | 0 |
| gui-overlays-close | 16.1 | 0 / 0 | 0 | 199.6 | 0 |
| gui-stability | 141.2 | 23 / 22 | 23 | 270.1 | 0 |

Actual desktop Open3D windows were used, with an audit-only logging proxy; no
production viewer was patched. Native key actions rechecked overlay F1–F9,
comparison F1–F3, navigation F10/F11 including a boundary, F12 reset and repeated
Shift+F12 export. Overlays used the genuine saved 29-detection frame-000000
capture, not fabricated predictions. Comparison used three existing private QA
copies of real frames 000000–000002 and the checked-in full configuration.

Three actual ViewControl orbit/pan/zoom probes used clearly marked audit-only
J/K/L callbacks. Three subsequent camera-link checks found equal view state and
matching intrinsic/extrinsic matrices. They do not establish native modifier-drag
pan; the previous audit separately verified native orbit/wheel zoom. Fresh native
minus changed point size 3 → 2; automation equal still did not increase it.
No redundant custom controls or backend patch were added.

The longer comparison run executed exactly **23 pipelines = initial + 22 successful
selections**. Camera/matching/reset/export callbacks added none. Five paired
screenshot exports plus three single-view exports produced **13 decoded,
nonblank PNGs** with correct frame/role/counter names and distinct raw/processed
pair hashes. Repeated export retained earlier files. A private screenshot path
was temporarily replaced by a file: export reported the expected error, frame
navigation continued, and export succeeded after restoring the destination.
Original scans and prior audit artifacts were never modified for this test.

Memory during the 141-second comparison sampled roughly 100.5–268.9 MiB RSS;
peak 270.1 MiB; exit RSS 145.4 MiB. These observations include warm-up, repeated
selection, exports and allocator/driver retention; they do not prove no leaks.
All three processes exited 0, logs show one/two corresponding window destructions,
and a process-list check found no matching QA renderer process left running.
Native Escape caused the expected process exit although the UI observation timed
out after window disappearance; renderer exit/destroy logs provide the outcome.

## RV-024 evidence status

Original PR60 and its experimenter comment were read again. It reports all 29
predictions matching numerically on a T4 at Raven `59c0916...`; that remains
historical external evidence. Current local capture has full SHA256
`cbcca87d066c66843774cce94bb3129254245a4498290d5a9d25e4a45aeff5d3`,
not recorded candidate `0816479c95c557ba35206cfb6ec466b26297790375c7cae7af984f5cc467e1b0`.
Its declared creation time follows the experimenter comment. Config SHA256
`170a9ffe76cfd8509d1044cfbcf1cbd44c5d320fda81bf0089a8d5efaf1c91c8`
matches the independently retrieved pinned official PointPillars YAML.

The currently available file validates as 29 schema-v1 Velodyne detections
(Car 4, Pedestrian 19, Cyclist 6). It is preserved as a separately identified
read-only local copy plus hashes. Relevant accessible artifact filenames were
searched again; original reference, original candidate and report were not found.
Different bytes are certain; different execution or reserialization is possible,
but the exact cause cannot be established from this single file and declared
timestamps. No numerical equivalence or tampering is claimed.

The complete runnable external two-process workflow, pinned config/revision,
known dependencies/flags, checkpoint checksum/rights prerequisite, optional
explicit Argo2 workaround, comparator tolerances and evidence-export instructions
are in [reference validation](../reference-validation.md). The original full
environment lock and upstream patch bytes remain unavailable. No trusted GPU
session is available locally, no checkpoint was downloaded/deserialized, and no
fresh inference or self-reference comparison was performed in this remediation.
RV-024 was already closed; this task did not close, reopen or reclassify it.

## Evidence locations and remaining work

Private ignored bundle:
`/Users/filip_macbook/Downloads/Raven/outputs/qa/remediation-2026-11-17`
(directory name reflects start date; run continued November 18). Original audit:
`outputs/qa/raven-full-system-validation.md` and
`outputs/qa/run-2026-11-17-001`. Generated outputs/captures/screenshots are not
included in this PR.

- `environment.json`, `normal-cli-routing.stdout.txt`: actual editable config,
  shebang/import/help and real KITTI output without PYTHONPATH.
- `artifact-search.json`, `pr60-comments.json`, `pointpillar.pinned.yaml`,
  `preserved-current-adapter-capture.json`: artifact investigation and retained evidence.
- `commands.json`, command stdout/stderr and JUnit: exact fresh checks, working
  directories, return codes and durations.
- `smoke-results.json`, `input-hashes.json`, `preservation.json`: repeated real
  preprocessing outputs and preservation checks.
- `gui-*.jsonl`, stdout/stderr, `gui-summary.json`, `screenshot-validation.json`,
  `screenshots/`: actual GUI control/state/resource and framebuffer evidence.

Remaining work before stronger claims:

1. **Blocked externally:** obtain authentic historical paired artifacts or execute
   the documented fresh independent GPU reference/adapter runs and export evidence.
2. **Manual verification:** physical unshifted `=` increase and Ctrl/middle-button
   pan on the intended Mac input device. No confirmed Raven bug justifies a fix yet.
3. **Not tested:** other OS/input layouts, much longer sessions and broad GPU
   environment compatibility. Bounded stability does not remove these limits.
4. **Deferred to M5:** KITTI AP/mAP evaluation and scientifically controlled
   accuracy/latency experiments; current counts/overlays are not those metrics.

The remediation documentation and regression protections are complete within the
available environment. GPU/manual evidence items remain explicitly incomplete;
M5 development can begin without representing them as newly validated.
