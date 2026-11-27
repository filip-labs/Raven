# Raven Studio

Raven Studio is an optional, native PySide6 desktop workbench over Raven's existing
CPU data, preprocessing and saved-results APIs. The interface uses Heimdall's
charcoal navigation, light surfaces, amber actions and restrained typography.
It runs no model inference and makes no cloud call. Data and source experiments
remain read-only.

## Install and launch

From the Raven checkout containing Studio, with the intended virtual environment
activated:

```bash
python -m pip install -e ".[dev,studio,visualization,preprocessing]"
raven studio
```

`studio` installs PySide6-Essentials (Qt Widgets/Core/Gui/Test), without WebEngine or
a separate web frontend. Release 6.12.0 is excluded after the Linux/Python 3.11
Studio job reproduced a binding-level `none_dealloc` abort, including after all
30 tests passed. OpenUSD documents the same version-specific failure and uses
6.11.2 in its [upstream CI](https://github.com/PixarAnimationStudios/OpenUSD/blob/dev/.github/workflows/buildusd.yml).
Reinstall the Studio extra to resolve the compatible version in an existing environment. `visualization` supplies Open3D; `preprocessing` supplies
SciPy for statistical outlier removal. Normal Raven installation, CLI commands and
`raven studio --help` do not import Qt. Missing Qt yields an actionable installation
error. No PyTorch, CUDA, OpenPCDet, weights or Modal installation is needed for
Studio or saved-prediction visualization.

The presentation target is macOS Apple Silicon. A real macOS desktop and working
OpenGL are needed for Open3D. Headless Qt tests run offscreen on Linux; they do not
establish graphical Open3D support on Linux or Windows. A signed/notarized macOS
`.app` and embedded 3D viewport are future work.

The light theme stays readable when macOS uses dark appearance. Figtree is used
if installed; otherwise an available system sans-serif font is selected. No fonts,
Rivian branding or reference-repository artwork are bundled. Tokens and Qt styles
live in `studio/theme.py`; the reference is Heimdall's `dashboard/src/styles.css`
and `layout/Shell.tsx`, translated to Qt rather than copied as a web interface.

## Dataset

1. Select **Open KITTI Dataset** and choose the root containing `training/` or
   `testing/`, not an individual Velodyne folder.
2. Choose `train`, `val` or `test`. Train/validation lists intersect available
   six-digit `.bin` filenames with Raven's verified canonical split manifests.
   Both use `training/velodyne/`. Test lists `testing/velodyne/` and explicitly
   reports that authoritative test membership is unavailable/unlabeled.
3. Select a frame, search by ID, or use **Previous / Next**. Enumeration reads
   filenames only; a selected scan, labels and calibration load in a background
   task. Only the current cloud and optional processed cloud are retained.
4. Inspect actual point count, XYZ bounds, coordinate frame, class counts and
   calibration availability. Corrupt/missing inputs produce visible errors.
5. Select point coloring (distance, intensity, uniform), size and optional ground
   truth, then **Open 3D View**. Ground truth requires valid matching calibration
   and labels. Raven's existing transforms and scientific color conventions apply.

Frame navigation is controlled in Studio and respects the selected canonical
split. Close/stop the current viewer before opening another selected frame. This
MVP does not enable Open3D directory navigation across the shared train/val folder,
which could otherwise escape the chosen split. Existing CLI navigation is unchanged.

## Processing

The Processing screen operates on the selected Dataset frame:

- Add range filtering, voxel downsampling, statistical outlier removal or Gaussian
  XYZ noise. Defaults are editable examples, not performance recommendations.
- Select a stage and edit its parameters. Check/uncheck rows to enable/disable them;
  use Up/Down, Remove or Reset. Repeated stages are supported.
- **Load Pipeline** accepts Raven's existing ordered JSON stage array, for example
  `configs/preprocessing/voxel.json`. No parallel configuration schema is added.
  Scalar and existing per-axis voxel sizes are preserved. Parameter text is parsed
  and validated by the existing pipeline; no UI rounding of loaded float values.
- **Apply Processing** prevalidates and runs `run_preprocessing_pipeline()` on a
  worker thread. Counts, reduction, per-stage and total durations are actual results.
  Changing frame/configuration invalidates the comparison until processing runs again.
- **Open RAW vs PROCESSED** passes the already-computed processed XYZI through an
  exclusive temporary `.bin` into Raven's existing comparison viewer. The source
  hash and frame conventions are checked. Viewer interactions never recompute the
  pipeline. Empty processed results are supported; raw empty scans cannot render.

Comparison uses the existing matched/synchronized two-window camera behavior and
one color/coordinate convention. It cannot be combined with ground-truth or
prediction overlays; those belong to Dataset/Results. Downsampling and outlier
removal lose information; timings/counts establish no detection-quality or inference
latency improvement.

## Saved results

**Open Manifest** selects an existing completed experiment manifest; **Open Run
Folder** requires exactly one `*.manifest.json`. Existing completed-run and evidence
readers validate artifact hashes, prediction schema and remote request/model/source
bindings. Failed, incomplete, altered or unsupported historical-profile runs fail
clearly. Arbitrary filenames are never proof of successful GPU execution.

The Results screen displays saved experiment identity/status, split/frame, model
and source, evidence kind, class counts, recorded GPU and timing fields. Synthetic
records stay labeled synthetic. Missing measurements are **Unavailable**, not zero.
Advanced provenance identifies configuration/model/checkpoint identities and
measurement limits. This MVP inspects runs, not general evaluation/benchmark bundles.

For the genuine presentation evidence, select:

```text
outputs/rv049/runs/modal-e2e-000000-20261127/
```

The files, rather than application constants, supply 29 detections (Car 4,
Pedestrian 19, Cyclist 6), Tesla T4 and the recorded initialization/detection/client
measurement values. One training frame is neither AP_R40 nor proof of 29 correct
predictions. Client end-to-end excludes initial loading/final persistence.

**View Predictions in 3D** revalidates the source run and selected frame, checks
that the original scan still matches its saved SHA256, validates the native
prediction document and writes an exclusive new temporary viewer copy. Optional
matching ground truth uses the existing calibration path. No inference is submitted.
For safety, this MVP overlays **raw-baseline runs only**; processed-run overlays
require a future exact processed-cloud replay adapter and are rejected explicitly.

**Verify Results** runs the existing offline verifier in a background thread.
Enable **Check original inputs and replay CPU preprocessing** for stronger checks
when the original dataset is available. Default integrity-only verification is
reported as such. CPU replay is not GPU inference reproduction, independent hardware
attestation or demonstrated cross-environment reproducibility.

## Viewer controls, screenshots and shutdown

Open3D/GLFW stays in a separate managed `QProcess`, using the current interpreter,
literal argument arrays and the launching Raven module path. One viewer process
may run at a time (the comparison process owns two linked windows). No unsafe Qt
embedding or replacement renderer is used. Camera controls and scientific box
colors remain those of the existing viewer.

- Drag/scroll in the Open3D window for existing camera navigation.
- F1: overlay help; F2: ground truth; F3: predictions; F4: both overlays.
- F5/F6/F7: existing class filters; F8: all classes; F9: confidence details in logs.
- F12: reset scene view; Shift+F12: export real rendered PNG(s). macOS may need Fn.
- Use **Screenshot Folder** before launching to retain PNGs in a chosen directory
  outside the dataset/source run. Existing export uses exclusive names.
- Without a chosen persistent folder, screenshot/default Open3D file outputs stay
  in the session's temporary cache and are removed on exit. No implicit capture
  writes into a dataset, source run or the home directory.
- **Details** shows captured stdout/stderr. Launch/exit errors are visible; a running
  process is not presented as proof of a successful graphical render.
- **Stop 3D View** requests termination, then kills a nonresponsive child after
  1.5 seconds. Studio waits for child termination on close. CPU stages are not
  forcibly interrupted: the UI stays responsive while shutdown waits for the
  current finite CPU operation, then releases the session cache.

Separate Open3D windows may need focus/selection through the macOS Dock or Mission
Control. This is an existing native-window/platform limitation, not an embedded
preview. See the [visualization guide](visualization.md) and
[comparison guide](cloud-comparison.md) for renderer-specific controls/limits.

## Validation and presentation checklist

CPU tests cover lazy/canonical discovery, malformed inputs, pipeline ordering,
repeated/disabled stages, backend parameter validation, float preservation,
saved-run integrity, exclusive prediction copies, raw/processed binding refusal,
Qt navigation/async states, missing optional dependencies, command/environment
construction, process failures, duplicate prevention and shutdown cleanup.
The standard CPU CI remains Qt-free; a separate job installs only `dev,studio`
and runs offscreen integration. Its Linux runner installs the EGL/OpenGL loader
libraries required to import Qt; no CUDA or NVIDIA runtime is installed. No test
requires a GPU, credentials or KITTI data.

Initial agent-observed macOS validation during MVP implementation:

| Check | Observed result |
| --- | --- |
| Native window, three screens, folder pickers | Confirmed on Apple Silicon; genuine Studio screenshots captured |
| Select train frame `000000` | Actual 115,384 points, bounds, Pedestrian annotation and calibration shown |
| Apply existing 0.2 m voxel configuration | Actual 22,595 output points, 80.42% reduction, measured stage/total duration |
| Open raw/GT and comparison | Separate processes launched/stopped through GUI; raw viewer sampled inside Open3D's event loop |
| Load genuine RV-049 results | Actual 29 detections, class distribution, T4 and recorded timings shown |
| GUI offline input/replay verification | `checks_passed`; integrity/source/input/preprocessing checks verified |
| Saved prediction + GT launcher | Genuine run revalidated and temporary document passed to the existing CLI viewer |
| Interactive Open3D camera/overlay appearance and Shift+F12 PNG | Still require direct operator confirmation; automation could not select the separate Python renderer window reliably |

The last row records the initial automation limitation. The owner subsequently
reported the working MVP as manually validated before requesting the polish pass;
that report is distinct from the agent's observations above. RV-050 and PR #82
remain open for review. Generated GUI screenshots and private run/data artifacts
remain local and untracked.

### UI refinement validation

The centralized theme retains the Heimdall palette. A small Qt proxy style paints
20 px rounded checkbox indicators (including stage-list checks) with checked,
unchecked, disabled, hover and keyboard-focus states. Native Qt input behavior is
retained. Combo boxes use a painted chevron; scrollbars have thin rounded thumbs
and no arrow buttons. Buttons use consistent padding with compact action widths.

Results separates total detections, individual class counts, experiment status
and recorded timings into readable blocks. Values still come from the existing
validated summary API; missing measurements stay unavailable and synthetic evidence
is labeled. Long source paths elide in the middle with the full path accessible
through a tooltip and accessibility name. Provenance remains expandable.

During the refinement pass, the native macOS app was relaunched and visually
inspected with the genuine saved RV-049 results: 29 detections, Car 4, Pedestrian 19,
Cyclist 6, and unchanged recorded timing values. A genuine native screenshot was
saved locally. Offscreen Qt layout checks also covered 1120 × 800 and 880 × 640
windows, checked controls and expanded provenance. These are widget/layout checks,
not fresh Open3D rendering validation. The automation bridge timed out during a
follow-up native interaction check. Regression tests cover checkbox mouse/keyboard
behavior, combo selection, readouts, loading failures and narrow-window overflow.
Narrow/short windows still require vertical scrolling to reach all actions; native
file pickers keep their platform appearance. No backend or renderer was changed.

Deferred: embedded viewport, broad experiment editor/execution, cloud management,
remote benchmarks, dark theme and signed application packaging. These are not
inactive controls masquerading as implemented features.
