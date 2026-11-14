# KITTI scene navigation and rendered PNG export (RV-029)

The existing legacy Open3D GLFW/OpenGL viewers now support lazy frame navigation
and protected framebuffer screenshots. No new GUI framework, inference backend
or preprocessing algorithm is introduced. Install `.[visualization]`, plus
`.[preprocessing]` for comparison configs that use statistical filtering.

## Commands and supported modes

Run from the repository root with your own licensed dataset:

```bash
# Navigate a Velodyne directory; start is an exact filename stem, not an index.
raven visualize data/kitti/training/velodyne --navigate --start-frame 000000

# Ground truth follows the active scan through matching sibling label/calib files.
raven visualize data/kitti/training/velodyne --navigate --ground-truth \
  --screenshot-dir outputs/screenshots

# Saved predictions: one schema-v1 FRAME.json per frame in this directory.
raven visualize data/kitti/training/velodyne --navigate --ground-truth \
  --predictions-dir /path/to/saved-predictions --screenshot-dir outputs/screenshots

# Linked RAW/PROCESSED navigation, with the existing pipeline configuration.
raven visualize data/kitti/training/velodyne --navigate \
  --compare configs/preprocessing/full.json --screenshot-dir outputs/screenshots

# Existing single-frame commands remain valid; opt into export with a directory.
raven visualize data/kitti/training/velodyne/000000.bin \
  --screenshot-dir outputs/screenshots
raven visualize data/kitti/training/velodyne/000000.bin \
  --compare configs/preprocessing/full.json --screenshot-dir outputs/screenshots
```

`--color distance|intensity|uniform` and `--point-size 1..25` retain their meanings.
`--start-frame` and `--predictions-dir` require `--navigate`. Navigation requires
the directory containing `.bin` files, rather than a single file or dataset root.
Without a starting frame, the first filename is selected. Navigation enables
export to `outputs/screenshots` by default; single-frame export controls are
enabled by explicitly supplying `--screenshot-dir` (the old command path is
otherwise unchanged).

| Mode | Navigation | Frame-matched overlays | PNG export |
| --- | --- | --- | --- |
| Raw cloud | yes | optional GT and/or saved predictions | one view image |
| RAW/PROCESSED comparison | yes | unsupported; rejected explicitly | matching raw/processed pair |
| Existing single frame | unchanged | existing explicit/inferred files | opt in with directory |

For navigation, `--ground-truth` requires the standard
`SPLIT/velodyne/FRAME.bin`, `SPLIT/label_2/FRAME.txt`, `SPLIT/calib/FRAME.txt`
layout. Explicit single-frame `--labels`, `--calib` and `--predictions` files are
rejected with `--navigate`. Use `--predictions-dir`, where the exact filename is
`FRAME.json`; the RV-023 reader verifies the six-digit ID and supported coordinate
conventions inside the document. Existing capture filenames can be copied by the
user to this layout; Raven does not rename or modify captures.

Missing, malformed or mismatched label/calibration/prediction files fail the
candidate selection. The current scene remains visible; previous overlays are
never reused on a new frame. Only after all candidate data and preprocessing
validate does the display change. Layer visibility and class filters persist,
and new boxes are transformed using that frame's actual calibration. Frame-ID
agreement cannot authenticate data provenance: use the corresponding source files.

## Controls and state

Focus a viewer window; macOS may require Fn with function keys.

| Key | Action |
| --- | --- |
| F10 | Previous frame; first-frame boundary reports without wrapping |
| F11 | Next frame; last-frame boundary reports without wrapping |
| F12 | Reset/fill the view for the current frame or current comparison pair |
| Shift + F12 | Export rendered PNG(s), reporting absolute saved paths |
| F1 | Existing help, plus navigation/export help when enabled |
| Q / Esc / window close | Existing close behavior; comparison closes both windows |

Existing overlay F2–F9 and comparison F2/F3 controls remain intact. Comparison
F2 explicitly matches from the focused view; F3 restores canonical window sizes
and fits the current pair. Native mouse navigation and point-size controls remain
available. In a single-frame export session, F10/F11 report that `--navigate` is
required. Releases/repeated key events and unrelated modifiers do not trigger
frame transitions or exports.

The complete camera state (orientation, look-at, zoom/FOV and bounds determining
position/clipping) is restored across transitions. This can leave a very different
next scene outside the old view; F12 explicitly fits the new data. In comparison,
both views retain the same camera and use a shared per-pair color scale. Reset
uses the active pair's union bounds, not the initial frame's bounds.

The filename list is snapshotted and sorted lexicographically by filename; it is
not temporal sequence reconstruction or numeric reordering. Only lowercase
`.bin` entries are selected. KITTI object-detection identifiers are separate
samples and must not be interpreted as a continuous drive. The implementation
keeps only the active raw/processed data, with temporary old/new data during a
transition; it does not read the entire dataset into memory. Every successful
selection, including a revisit, executes the configured pipeline once. Camera
movement, matching, reset and capture do not run preprocessing. Corrupt/empty
candidates are reported without changing the active frame; restart with
`--start-frame` to select a later valid sample rather than silently skipping data.

Titles and terminal output show the active ID and 1-based position. Comparison
also reports raw/processed counts and percentage reduction. Window titles can
truncate on small/Retina windows; full status is always printed to the terminal.
Native legacy window renaming uses `create_window` on the **existing** visualizer,
which reuses its GLFW window. Transitions restore canonical dimensions/positions
(1280×800 single view, 640×720 each comparison view), then restore the camera;
no new visualizer objects are created per frame. Keep comparison dimensions equal
for matching aspect ratio. This relies on the existing supported
[Open3D Visualizer API](https://www.open3d.org/docs/release/python_api/open3d.visualization.Visualizer.html).

## Protected screenshot behavior

Shift+F12 calls `capture_screen_image(..., do_render=True)` on the actual active
Open3D framebuffer. It requires a working graphical rendering context; CPU CI
mocks capture and does not establish headless-rendering support. The PNG contains
the rendered scene/overlays, without OS title bars or terminal labels.

Names use the first available positive four-digit counter (larger numbers remain
valid) and include the frame identifier:

```text
raven_000001_view_0001.png
raven_000001_raw_0001.png
raven_000001_processed_0001.png
```

Comparison exports use the same counter for both roles and synchronize cameras
from the focused window immediately before capture. Existing filenames, including
partial pairs, are never overwritten: paths are reserved using exclusive creation
and the next free complete set is chosen. Failed capture/PNG verification removes
only this attempt's reserved files; no pair is reported successful until both
images pass. PNG signature, dimensions, chunk integrity/CRC and IEND are checked.
Filesystem/rendering errors are printed and the current scene remains usable.
Directories are created lazily when exporting. Outputs are ignored through the
existing `/outputs/` rule and a `raven_*.png` rule covering custom directories.
No source scan or processed point array is written by this export path. Use this
Raven shortcut for its filename/overwrite checks; native backend capture keys are
not managed by the new export helper.

## Reusable APIs

`FrameSequence(directory, start_frame=None, ground_truth=False,
predictions_dir=None, stages=None)` discovers filenames and validates/loads the
initial `SceneFrame`. `active` contains `cloud`, zero-based `position`, `boxes`,
`predictions` and optional `processed` cloud. `move(-1|1)` loads a candidate then
updates `active`, returning None at a boundary. Load failures raise SceneLoadError
and preserve `active`; `status()` gives the ID and human-readable position.

Pass the actual active cloud to the existing viewer APIs:

```python
from raven.perception.visualization.scenes import FrameSequence
from raven.perception.visualization.viewer import view_point_cloud

sequence = FrameSequence('/path/to/training/velodyne', ground_truth=True)
view_point_cloud(sequence.active.cloud, sequence=sequence,
                 screenshot_dir='/path/to/screenshots')
```

For comparison, construct the sequence with `stages=the_existing_stage_list` and
pass `sequence.active.cloud`, `sequence.active.processed` and `sequence=sequence`
to `view_cloud_comparison`. Comparison sequences reject overlays. Sequence overlays
must be configured in FrameSequence, rather than mixed with manual viewer boxes.
`export_screenshots(renderers, frame_id, output_dir, roles=('view',))` is reusable
with initialized renderers and returns a tuple of saved absolute Paths.

## Native macOS verification and automated coverage

On 2026-11-15, Python 3.11.12 / NumPy 2.4.6 / SciPy 1.17.1 / Open3D 0.20.0,
macOS 26.6.2 arm64, the local directory contained 7,481 frames. Native F11/F10
navigation exercised **000000 → 000001 → 000000** in both the GT viewer and the
full comparison mode. The first-frame no-wrap boundary, F12/current-frame reset,
comparison F3, Shift+F12 and clean Q exit were exercised.

| Frame | Raw count | Full pipeline count | Visible unfiltered GT objects |
| --- | --- | --- | --- |
| 000000 | 115,384 | 12,232 | 1 |
| 000001 | 120,268 | 16,514 | 3 |

GT objects were loaded from each frame's matching local labels/calibration and
visually inspected. Nine actual framebuffer PNGs were saved outside Git and decoded
using Open3D's image reader; two representative exports were additionally
inspected visually; single-view images were 1280×800 and
comparison images 640×720. Repeated exports chose new counters. Raw/processed
images were distinct. Both scan hashes were unchanged after testing. Prediction
navigation was validated with synthetic frame-matched/mismatched JSON in CPU
tests; no unavailable GPU capture is claimed as a native navigation test.

A temporary unshipped keyboard probe applied native orbit/zoom/pan methods; exact
view-state equality across transitions was asserted programmatically. This is
separate from the native navigation/export key checks and visual inspection.
Direct mouse gestures and all other display/driver combinations were not verified
in this change. The known new-GUI/Metal black-window limitation from RV-027 is
avoided by retaining GLFW/OpenGL. Open3D must run on the main thread in a graphical
desktop. Native driver crashes cannot be repaired by Python cleanup logic.

`tests/test_scenes.py` uses tiny synthetic scans, labels/calibrations/predictions
and mocked renderers. It covers discovery/start/boundaries, lazy loading,
corrupt/disappeared/empty candidates, overlay matching, camera/filter persistence,
geometry replacement/cleanup, comparison processing counts, reset, exclusive PNG
numbering, pair rollback, corrupt capture, filesystem errors, ignored key events,
nonfatal capture errors and CLI validation. Existing overlay/comparison/M2 tests
remain in the full CPU suite; no dataset download, weights, CUDA or display is
needed by automated tests. Screenshots alone establish no detection accuracy or
inference/end-to-end latency result.

## A 3–5 minute demonstration

1. Open the GT navigation command above at 000000; explain axes, frame ID/index
   and the terminal's GT/class status (about 45 seconds).
2. Orbit/zoom, press F11 and F10, then F12. Explain camera persistence versus
   explicit fitting and frame-matched calibration (about one minute).
3. Press Shift+F12 twice; inspect the printed paths and two distinct counters
   using your image viewer (about 30 seconds). Close with Q.
4. Open comparison navigation with `full.json`. Explain raw/processed counts,
   shared view/color scale, then navigate forward and back (about one minute).
5. Export the matching RAW/PROCESSED pair, explain the explicit pipeline and its
   provenance, then close. Reduced point count/visible geometry changes are not
   claims about detector quality or total latency (about 45 seconds).
