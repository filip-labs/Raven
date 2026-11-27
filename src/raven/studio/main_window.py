"""Native offline workbench. All expensive backend work runs outside the UI thread."""

import hashlib
import re
import tempfile
from copy import deepcopy
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from raven.perception.preprocessing.pipeline import run_preprocessing_pipeline
from raven.reproducibility.verify import verify_results
from raven.studio.services import (
    STAGES,
    ViewOptions,
    discover_dataset,
    export_prediction_copy,
    inspect_frame,
    load_pipeline,
    load_saved_run,
    summarize_result,
    viewer_arguments,
)
from raven.studio.state import PipelineEditor
from raven.studio.tasks import Task, ViewerProcess
from raven.studio.theme import TOKENS
from raven.studio.widgets import ClassCounts, ComboBox, PathLabel, Readout


def label(text="", name=None):
    widget = QLabel(text)
    widget.setWordWrap(True)
    widget.setTextFormat(Qt.TextFormat.PlainText)
    widget.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    if name:
        widget.setObjectName(name)
    return widget


def button(text, callback, primary=False):
    widget = QPushButton(text)
    widget.setObjectName("action_" + re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_"))
    widget.setProperty("tone", "primary" if primary else "secondary")
    widget.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
    widget.clicked.connect(callback)
    return widget


def row(*widgets):
    layout = QHBoxLayout()
    layout.setSpacing(10)
    for widget in widgets:
        layout.addWidget(widget, 1 if isinstance(widget, (QLabel, QLineEdit)) else 0)
    if all(isinstance(widget, QPushButton) for widget in widgets):
        layout.addStretch()
    return layout


def panel():
    widget = QFrame()
    widget.setObjectName("panel")
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(24, 24, 24, 24)
    layout.setSpacing(14)
    return widget, layout


def seconds(value):
    return "Unavailable" if value is None else f"{value:.6f} s"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Raven Studio — LiDAR Perception Workbench")
        self.resize(1120, 800)
        self.setMinimumSize(880, 640)
        self.index = self.frame = self.saved = self.processed = None
        self.processed_info = None
        self.editor = PipelineEditor()
        self.task = None
        self.closing = False
        self.temporary = tempfile.TemporaryDirectory(prefix="raven-studio-")
        self.viewer = ViewerProcess(self, working_directory=self.temporary.name)
        self.viewer.message.connect(self.notify)
        self.viewer.changed.connect(self._viewer_changed)
        self.viewer.ended.connect(self._maybe_close)
        self.viewer.output.connect(self._viewer_output)
        self.parameter_fields = {}
        self.parameter_index = -1
        self._job_value = self._job_error = None
        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        side = QFrame()
        side.setObjectName("sidebar")
        side.setFixedWidth(164)
        nav = QVBoxLayout(side)
        nav.setContentsMargins(16, 28, 16, 24)
        nav.setSpacing(10)
        nav.addWidget(label("RAVEN", "brand"))
        nav.addWidget(label("STUDIO  /  VISION", "sidebar_caption"))
        nav.addSpacing(35)
        self.nav_buttons = []
        for i, title in enumerate(["Dataset", "Processing", "Results"]):
            b = QPushButton(title)
            b.setObjectName("nav_" + title.lower())
            b.setCheckable(True)
            b.setAutoExclusive(True)
            b.toggled.connect(lambda checked, n=i: self.navigate(n) if checked else None)
            nav.addWidget(b)
            self.nav_buttons.append(b)
        nav.addStretch()
        nav.addWidget(label("LIDAR WORKBENCH"))
        nav.addWidget(label("Local. Read-only data.\nNo cloud execution."))
        root.addWidget(side)
        workspace = QWidget()
        workspace.setObjectName("workspace")
        content = QVBoxLayout(workspace)
        content.setContentsMargins(28, 24, 28, 18)
        content.setSpacing(12)
        self.pages = QStackedWidget()
        for widget in (self._dataset_page(), self._processing_page(), self._results_page()):
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            scroll.setWidget(widget)
            self.pages.addWidget(scroll)
        content.addWidget(self.pages, 1)
        self.screenshot_path = QLineEdit()
        self.screenshot_path.setReadOnly(True)
        self.screenshot_path.setPlaceholderText(
            "Choose a folder to keep Shift+F12 screenshots; otherwise they are temporary"
        )
        self.screenshot_path.setAccessibleName("Screenshot output directory")
        content.addLayout(
            row(self.screenshot_path, button("Screenshot Folder", self.choose_screenshots))
        )
        self.stop_button = button("Stop 3D View", self.viewer.stop)
        self.stop_button.setEnabled(False)
        self.status = label("Choose a local KITTI dataset to begin.", "muted")
        self.status.setMaximumHeight(90)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setMaximumHeight(100)
        self.details.hide()
        details_button = button(
            "Details", lambda: self.details.setVisible(not self.details.isVisible())
        )
        content.addLayout(row(self.status, details_button, self.stop_button))
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.hide()
        content.addWidget(self.progress)
        content.addWidget(self.details)
        root.addWidget(workspace, 1)
        self.setCentralWidget(central)
        self.navigate(0)
        self._invalidate_processing()
        self.raw_button.setEnabled(False)
        self.prediction_button.setEnabled(False)
        self.verify_button.setEnabled(False)

    def navigate(self, index):
        self.pages.setCurrentIndex(index)
        for i, b in enumerate(self.nav_buttons):
            b.setChecked(i == index)

    def _page(self, title, subtitle):
        widget = QWidget()
        widget.setObjectName("page_" + re.sub(r"[^a-z0-9]+", "_", title.lower()))
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 12)
        layout.setSpacing(16)
        layout.addWidget(label(title, "title"))
        layout.addWidget(label(subtitle, "subtitle"))
        return widget, layout

    def _dataset_page(self):
        widget, layout = self._page(
            "Explore your scene", "Local KITTI data. Original geometry, untouched."
        )
        self.dataset_path = QLineEdit()
        self.dataset_path.setReadOnly(True)
        self.dataset_path.setPlaceholderText("No dataset selected")
        self.dataset_path.setAccessibleName("KITTI dataset root")
        self.split = ComboBox()
        self.split.addItems(["train", "val", "test"])
        self.split.setAccessibleName("Dataset split")
        self.split.currentTextChanged.connect(self._change_split)
        layout.addLayout(row(button("Open KITTI Dataset", self.choose_dataset, True), self.split))
        layout.addWidget(self.dataset_path)
        body = QHBoxLayout()
        browser = QVBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Find frame ID…")
        self.search.setAccessibleName("Search frame IDs")
        self.search.textChanged.connect(self._filter_frames)
        self.frames = QListWidget()
        self.frames.setMinimumWidth(150)
        self.frames.setMaximumWidth(230)
        self.frames.setAccessibleName("Available frames")
        self.frames.currentRowChanged.connect(self._select_frame)
        browser.addWidget(self.search)
        browser.addWidget(self.frames, 1)
        browser.addLayout(
            row(
                button("Previous", lambda: self.move_frame(-1)),
                button("Next", lambda: self.move_frame(1)),
            )
        )
        body.addLayout(browser, 1)
        card, details = panel()
        self.frame_title = label("No frame selected", "title")
        self.point_count = label("—", "hero")
        details.addWidget(self.frame_title)
        details.addWidget(self.point_count)
        details.addWidget(label("LIDAR POINTS", "muted"))
        self.frame_details = label(
            "Open a dataset and select a frame.\nScans load one at a time.", "muted"
        )
        details.addWidget(self.frame_details)
        details.addStretch()
        self.gt = QCheckBox("Ground-truth boxes")
        details.addWidget(self.gt)
        self.color = ComboBox()
        self.color.addItems(["distance", "intensity", "uniform"])
        self.color.setAccessibleName("Point color")
        self.point_size = QDoubleSpinBox()
        self.point_size.setRange(1, 25)
        self.point_size.setValue(2)
        self.point_size.setSuffix(" px")
        self.point_size.setAccessibleName("Point size")
        details.addLayout(row(self.color, self.point_size))
        self.raw_button = button("Open 3D View", self.open_raw, True)
        details.addWidget(self.raw_button)
        details.addWidget(
            label(
                "Interactive Open3D opens in a separate window.\nF2 ground truth · F1 help · F12 reset · Shift+F12 screenshot",
                "muted",
            )
        )
        body.addWidget(card, 3)
        layout.addLayout(body, 1)
        return widget

    def _processing_page(self):
        widget, layout = self._page(
            "Shape the signal", "A deliberate pipeline. The same tested Raven processing, on CPU."
        )
        self.processing_source = label("Select a frame in Dataset first.", "muted")
        layout.addWidget(self.processing_source)
        self.stage_type = ComboBox()
        self.stage_type.addItems(list(STAGES))
        self.stage_type.setAccessibleName("New processing stage")
        layout.addLayout(
            row(
                self.stage_type,
                button("Add Stage", self.add_stage),
                button("Load Pipeline", self.choose_pipeline),
                button("Reset", self.reset_pipeline),
            )
        )
        columns = QHBoxLayout()
        left = QVBoxLayout()
        self.stages = QListWidget()
        self.stages.setAccessibleName("Ordered pipeline stages")
        self.stages.currentRowChanged.connect(self._stage_selected)
        self.stages.itemChanged.connect(self._stage_enabled)
        left.addWidget(self.stages)
        left.addLayout(
            row(
                button("↑ Up", lambda: self.move_stage(-1)),
                button("↓ Down", lambda: self.move_stage(1)),
                button("Remove", self.remove_stage),
            )
        )
        columns.addLayout(left, 1)
        card, parameters = panel()
        parameters.addWidget(label("Stage parameters"))
        self.parameter_hint = label("Add or select a stage. Checked rows are enabled.", "muted")
        parameters.addWidget(self.parameter_hint)
        self.parameter_form = QFormLayout()
        self.parameter_form.setSpacing(12)
        parameters.addLayout(self.parameter_form)
        parameters.addStretch()
        columns.addWidget(card, 1)
        layout.addLayout(columns, 1)
        self.apply_button = button("Apply Processing", self.apply_processing, True)
        self.compare_button = button("Open RAW vs PROCESSED", self.open_comparison)
        layout.addLayout(row(self.apply_button, self.compare_button))
        self.processing_summary = label(
            "No processing applied. Source data is never modified.", "muted"
        )
        layout.addWidget(self.processing_summary)
        self.stage_stats = QTableWidget(0, 4)
        self.stage_stats.setHorizontalHeaderLabels(["Stage", "Input", "Output", "Duration"])
        self.stage_stats.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.stage_stats.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.stage_stats.setMaximumHeight(170)
        layout.addWidget(self.stage_stats)
        layout.addWidget(
            label(
                "Comparison uses the computed cloud; interactions do not rerun processing.\nGT/prediction overlays belong to Dataset or Results, not comparison mode.",
                "muted",
            )
        )
        return widget

    def _results_page(self):
        widget, layout = self._page(
            "Inspect the evidence", "Saved results, validated locally. No inference is launched."
        )
        self.result_path = PathLabel("No experiment open")
        layout.addLayout(
            row(
                button("Open Manifest", self.choose_run, True),
                button("Open Run Folder", self.choose_run_folder),
            )
        )
        layout.addWidget(self.result_path)
        self.result_frames = ComboBox()
        self.result_frames.setAccessibleName("Saved experiment frame")
        self.result_frames.currentTextChanged.connect(self._show_result)
        layout.addWidget(self.result_frames)
        card, body = panel()
        self.result_title = label("Your results stay local", "section_title")
        self.result_status = label("No run open", "result_status")
        self.result_status.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        self.result_count = label("—", "hero")
        self.result_classes = ClassCounts()
        self.result_summary = label("Missing measurements are shown as unavailable.", "muted")
        body.addLayout(row(self.result_title, self.result_status))
        body.addWidget(self.result_count)
        body.addWidget(self.result_classes)
        body.addWidget(self.result_summary)
        self.result_timings = {}
        timing_row = QHBoxLayout()
        timing_row.setSpacing(10)
        for key, title in (
            ("initialization", "Model initialization"),
            ("detection", "GPU detection"),
            ("end_to_end", "Client end-to-end"),
        ):
            metric = Readout(title, compact=True)
            metric.set_value("Unavailable")
            self.result_timings[key] = metric
            timing_row.addWidget(metric, 1)
        body.addLayout(timing_row)
        body.addStretch()
        self.result_gt = QCheckBox("Include matching ground truth, when available")
        body.addWidget(self.result_gt)
        self.prediction_button = button("View Predictions in 3D", self.open_predictions, True)
        body.addWidget(self.prediction_button)
        body.addWidget(
            label(
                "Actual saved boxes. A new temporary viewer copy leaves the source run unchanged.",
                "muted",
            )
        )
        layout.addWidget(card, 1)
        self.verify_button = button("Verify Results", self.verify_run)
        self.verify_inputs = QCheckBox("Check original inputs and replay CPU preprocessing")
        layout.addLayout(row(self.verify_button, self.verify_inputs))
        self.verification_label = label(
            "Verification checks saved consistency; it does not repeat GPU inference.", "muted"
        )
        layout.addWidget(self.verification_label)
        self.provenance_group = QGroupBox("Provenance · measurement boundaries")
        self.provenance_group.setCheckable(True)
        self.provenance_group.setChecked(False)
        provenance_layout = QVBoxLayout(self.provenance_group)
        self.provenance_text = label("", "muted")
        self.provenance_text.hide()
        self.provenance_group.setMaximumHeight(48)
        self.provenance_group.toggled.connect(self.provenance_text.setVisible)
        self.provenance_group.toggled.connect(
            lambda expanded: self.provenance_group.setMaximumHeight(16777215 if expanded else 48)
        )
        provenance_layout.addWidget(self.provenance_text)
        layout.addWidget(self.provenance_group)
        return widget

    def notify(self, message):
        self.status.setText(message[:700])
        self.details.setPlainText(message + ("\n\n" + self.viewer.log if self.viewer.log else ""))

    def error(self, exc):
        self.notify(str(exc))
        self.status.setStyleSheet(f"color: {TOKENS['danger']};")

    def start_job(self, title, function, callback):
        if self.task is not None or self.closing:
            return
        self.status.setStyleSheet("")
        self.notify(title)
        self.pages.setEnabled(False)
        self.progress.show()
        self._job_value = self._job_error = None
        self._job_callback = callback
        self.task = Task(function, self)
        self.task.progress.connect(self.notify)
        self.task.succeeded.connect(self._receive_result)
        self.task.failed.connect(self._receive_error)
        self.task.finished.connect(self._finish_job)
        self.task.start()

    def _receive_result(self, value):
        self._job_value = value

    def _receive_error(self, message):
        self._job_error = message

    def _finish_job(self):
        task, self.task = self.task, None
        self.pages.setEnabled(True)
        self.progress.hide()
        task.deleteLater()
        if self.closing:
            self._maybe_close()
            return
        if self._job_error:
            if self._job_callback == self._run_loaded:
                self.result_title.setText("No experiment loaded")
                self.result_status.setText("Not loaded")
            self.error(self._job_error)
        else:
            try:
                self._job_callback(self._job_value)
            except Exception as exc:  # noqa: BLE001 -- Report backend failures at the GUI boundary.
                self.error(exc)

    def choose_dataset(self):
        path = QFileDialog.getExistingDirectory(
            self, "Choose KITTI root (contains training/ or testing/)"
        )
        if path:
            self.open_dataset(path)

    def open_dataset(self, path):
        split = self.split.currentText()
        self.frame = self.index = None
        self.raw_button.setEnabled(False)
        self._invalidate_processing()
        self.start_job(
            "Discovering available frames…",
            lambda progress: discover_dataset(path, split),
            self._dataset_loaded,
        )

    def _change_split(self):
        if self.dataset_path.text():
            self.open_dataset(self.dataset_path.text())

    def _dataset_loaded(self, index):
        self.index = index
        self.dataset_path.setText(str(index.config.root))
        self.frames.blockSignals(True)
        self.frames.clear()
        self.frames.addItems([p.stem for p in index.paths])
        self.frames.blockSignals(False)
        self._filter_frames(self.search.text())
        self.notify(f"{len(index.paths)} available frames. {index.membership}")
        first = next(
            (i for i in range(self.frames.count()) if not self.frames.item(i).isHidden()), -1
        )
        self.frames.setCurrentRow(first)

    def _filter_frames(self, text):
        for i in range(self.frames.count()):
            self.frames.item(i).setHidden(text.strip() not in self.frames.item(i).text())

    def move_frame(self, delta):
        if self.task or not self.index:
            return
        index = self.frames.currentRow() + delta
        while 0 <= index < self.frames.count():
            if not self.frames.item(index).isHidden():
                self.frames.setCurrentRow(index)
                return
            index += delta

    def _select_frame(self, index):
        if self.index is None or index < 0:
            return
        self.frame = None
        self.raw_button.setEnabled(False)
        self._invalidate_processing()
        self.frame_title.setText("Loading frame…")
        self.frame_details.setText("Reading the selected scan and matching metadata…")
        self.processing_source.setText("Loading a new frame…")
        self.point_count.setText("—")
        self.start_job(
            "Loading scan, labels and calibration…",
            lambda progress: inspect_frame(self.index.paths[index]),
            self._frame_loaded,
        )

    def _frame_loaded(self, info):
        self.frame = info
        self.frame_title.setText(f"Frame {info.cloud.frame_id}")
        self.point_count.setText(f"{info.cloud.count:,}")
        bounds = (
            "Empty cloud"
            if info.bounds is None
            else "\n".join(
                f"{axis}  {lo:.3f} → {hi:.3f} m" for axis, lo, hi in zip("XYZ", *info.bounds)
            )
        )
        gt = (
            "Unavailable"
            if info.ground_truth is None
            else ", ".join(f"{k}: {v}" for k, v in info.ground_truth.items()) or "No annotations"
        )
        self.frame_details.setText(
            f"{info.cloud.coordinate_frame}\n\n{bounds}\n\nGround truth  {gt}\nCalibration  {'Available' if info.calibration_available else 'Unavailable'}"
        )
        self.gt.setEnabled(info.ground_truth is not None and info.calibration_available)
        if not self.gt.isEnabled():
            self.gt.setChecked(False)
        self.raw_button.setEnabled(info.cloud.count > 0)
        self.apply_button.setEnabled(True)
        self.processing_source.setText(
            f"Frame {info.cloud.frame_id} · {info.cloud.count:,} points · {info.cloud.coordinate_frame}"
        )
        self.notify(f"Frame {info.cloud.frame_id} loaded. Original scan is read-only.")

    def _invalidate_processing(self):
        self.processed = self.processed_info = None
        self.compare_button.setEnabled(False)
        self.apply_button.setEnabled(self.frame is not None)
        self.processing_summary.setText(
            "Apply Processing to calculate current counts and durations."
        )
        self.stage_stats.setRowCount(0)

    def _refresh_pipeline(self, selected=-1):
        self.stages.blockSignals(True)
        self.stages.clear()
        for index, entry in enumerate(self.editor.rows):
            item = QListWidgetItem(f"{index + 1:02d}   {entry['stage']['name'].replace('_', ' ')}")
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked if entry["enabled"] else Qt.CheckState.Unchecked
            )
            self.stages.addItem(item)
        self.stages.setCurrentRow(selected)
        self.stages.blockSignals(False)
        self._stage_selected(selected)

    def _stage_selected(self, index):
        # Commit edits before changing selection; invalid values retain their stage.
        if (
            self.parameter_index >= 0
            and self.parameter_index != index
            and self.parameter_index < len(self.editor.rows)
        ):
            try:
                self._save_parameters()
            except ValueError as exc:
                self.error(exc)
                self.stages.blockSignals(True)
                self.stages.setCurrentRow(self.parameter_index)
                self.stages.blockSignals(False)
                return
        self.parameter_index = index
        self.parameter_fields = {}
        while self.parameter_form.rowCount():
            self.parameter_form.removeRow(0)
        if index < 0:
            self.parameter_hint.setText("Add or select a stage. Checked rows are enabled.")
            return
        stage = self.editor.rows[index]["stage"]
        self.parameter_hint.setText(
            "Spatial units: meters. Backend validation runs before processing."
        )
        for key, value in stage.items():
            if key == "name":
                continue
            values = value if isinstance(value, (list, tuple)) else [value]
            fields = []
            for offset, number in enumerate(values):
                field = QLineEdit(str(number))
                field.setAccessibleName(f"{key} {offset + 1}")
                field.textChanged.connect(self._invalidate_processing)
                name = key.replace("_", " ")
                if len(values) == 2:
                    name += " min" if offset == 0 else " max"
                elif len(values) == 3:
                    name += " " + "XYZ"[offset]
                self.parameter_form.addRow(name, field)
                fields.append(field)
            self.parameter_fields[key] = (fields, isinstance(value, (list, tuple)))

    def _save_parameters(self):
        if not 0 <= self.parameter_index < len(self.editor.rows):
            return
        stage = {"name": self.editor.rows[self.parameter_index]["stage"]["name"]}
        for key, (fields, vector) in self.parameter_fields.items():
            cast = int if key in ("seed", "nb_neighbors") else float
            try:
                values = [cast(field.text()) for field in fields]
            except ValueError as exc:
                raise ValueError(
                    f"{key}: enter a valid {'integer' if cast is int else 'number'}."
                ) from exc
            stage[key] = values if vector else values[0]
        self.editor.replace(self.parameter_index, stage)

    def add_stage(self):
        try:
            self._save_parameters()
            self.editor.add(deepcopy(STAGES[self.stage_type.currentText()]))
            self.parameter_index = -1
            self._refresh_pipeline(len(self.editor.rows) - 1)
            self._invalidate_processing()
        except ValueError as exc:
            self.error(exc)

    def remove_stage(self):
        index = self.stages.currentRow()
        if index >= 0:
            del self.editor.rows[index]
            self.parameter_index = -1
            self._refresh_pipeline(min(index, len(self.editor.rows) - 1))
            self._invalidate_processing()

    def move_stage(self, delta):
        try:
            self._save_parameters()
            index = self.stages.currentRow()
            if index >= 0:
                target = self.editor.move(index, delta)
                self.parameter_index = -1
                self._refresh_pipeline(target)
                self._invalidate_processing()
        except ValueError as exc:
            self.error(exc)

    def _stage_enabled(self, item):
        self.editor.rows[self.stages.row(item)]["enabled"] = (
            item.checkState() == Qt.CheckState.Checked
        )
        self._invalidate_processing()

    def reset_pipeline(self):
        self.editor.load([])
        self.parameter_index = -1
        self._refresh_pipeline()
        self._invalidate_processing()

    def choose_pipeline(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open existing stage-array JSON", filter="JSON (*.json)"
        )
        if path:

            def loaded(stages):
                self.editor.load(stages)
                self.parameter_index = -1
                self._refresh_pipeline(0 if stages else -1)
                self._invalidate_processing()
                self.notify("Pipeline loaded. Apply Processing to run it.")

            self.start_job("Validating pipeline…", lambda progress: load_pipeline(path), loaded)

    def apply_processing(self):
        try:
            self._save_parameters()
            stages = self.editor.active()
            info = self.frame
            if info is None:
                raise ValueError("Select a frame first.")
        except (ValueError, TypeError) as exc:
            self.error(exc)
            return

        def execute(progress):
            progress("Running CPU preprocessing…")
            return run_preprocessing_pipeline(
                info.cloud,
                stages,
                stage_observer=lambda event, position, name, count: progress(
                    f"Stage {position} · {name} · {event} · {count:,} points"
                ),
            )

        def completed(result):
            self.processed, self.processed_info = result, info
            reduction = (
                0.0 if not info.cloud.count else 100.0 * (1 - result.cloud.count / info.cloud.count)
            )
            self.processing_summary.setText(
                f"{info.cloud.count:,} → {result.cloud.count:,} points  ·  {reduction:.2f}% removed  ·  {result.duration_seconds:.6f} s total"
            )
            self.stage_stats.setRowCount(len(result.stages))
            for i, stage in enumerate(result.stages):
                for j, value in enumerate(
                    [
                        stage.name,
                        f"{stage.input_count:,}",
                        f"{stage.output_count:,}",
                        seconds(stage.duration_seconds),
                    ]
                ):
                    self.stage_stats.setItem(i, j, QTableWidgetItem(value))
            self.compare_button.setEnabled(info.cloud.count > 0)
            self.notify("Processing complete. Original data is unchanged.")

        self.start_job("Applying pipeline…", execute, completed)

    def choose_screenshots(self):
        path = QFileDialog.getExistingDirectory(
            self, "Choose screenshot output folder (new PNG files only)"
        )
        if path:
            try:
                self._validate_screenshot_path(Path(path))
                self.screenshot_path.setText(path)
            except ValueError as exc:
                self.error(exc)

    def _validate_screenshot_path(self, path):
        roots = []
        if self.index:
            roots.append(self.index.config.root)
        if self.saved:
            roots += [self.saved.run.config.dataset.root, self.saved.run.manifest_path.parent]
        if any(path.resolve().is_relative_to(root) for root in roots):
            raise ValueError("Choose a screenshot folder outside datasets and source runs.")

    def _view_options(self):
        path = (
            Path(self.screenshot_path.text()).resolve()
            if self.screenshot_path.text()
            else Path(self.temporary.name) / "screenshots"
        )
        if path:
            self._validate_screenshot_path(path)
        return ViewOptions(self.color.currentText(), self.point_size.value(), path)

    def _launch_viewer(self, arguments):
        try:
            self.viewer.start(arguments)
        except (OSError, ValueError, RuntimeError) as exc:
            self.error(exc)

    def open_raw(self):
        if self.frame is None:
            return
        try:
            self._launch_viewer(
                viewer_arguments(
                    self.frame.path, self._view_options(), ground_truth=self.gt.isChecked()
                )
            )
        except (ValueError, OSError) as exc:
            self.error(exc)

    def open_comparison(self):
        if self.processed is None or self.viewer.running:
            if self.viewer.running:
                self.error("Close the current 3D view first.")
            return
        info, result = self.processed_info, self.processed
        try:
            options = self._view_options()
        except ValueError as exc:
            self.error(exc)
            return

        def prepare(progress):
            directory = Path(tempfile.mkdtemp(dir=self.temporary.name))
            target = directory / f"{info.cloud.frame_id}.bin"
            with target.open("xb") as stream:
                stream.write(result.cloud.points.astype("<f4").tobytes())
            raw_hash = hashlib.sha256(info.cloud.points.astype("<f4").tobytes()).hexdigest()
            return [
                "-m",
                "raven.studio.viewer_process",
                "--raw",
                str(info.path),
                "--processed",
                str(target),
                "--raw-sha256",
                raw_hash,
                *options.arguments(),
            ]

        self.start_job(
            "Preparing the existing processed cloud for comparison…", prepare, self._launch_viewer
        )

    def choose_run(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Raven experiment manifest",
            filter="Run manifests (*.manifest.json);;JSON (*.json)",
        )
        if path:
            self.open_run(path)

    def choose_run_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Open completed Raven run folder")
        if path:
            self.open_run(path)

    def open_run(self, path):
        self.saved = None
        self.result_title.setText("Opening saved evidence…")
        self.result_count.setText("—")
        self.result_classes.clear()
        self.result_status.setText("Opening…")
        self.result_status.setProperty("state", "neutral")
        self.result_status.style().polish(self.result_status)
        for metric in self.result_timings.values():
            metric.set_value("Unavailable")
        self.result_summary.setText("")
        self.provenance_text.setText("")
        self.prediction_button.setEnabled(False)
        self.verify_button.setEnabled(False)
        self.verification_label.setText("Validating artifact identities…")
        self.start_job(
            "Reading and validating saved results…",
            lambda progress: load_saved_run(path),
            self._run_loaded,
        )

    def _run_loaded(self, saved):
        self.saved = saved
        self.result_path.setText(str(saved.run.manifest_path))
        self.result_frames.blockSignals(True)
        self.result_frames.clear()
        self.result_frames.addItems([r["frame_id"] for r in saved.frames])
        self.result_frames.blockSignals(False)
        self._show_result(self.result_frames.currentText())
        self.verify_button.setEnabled(True)
        self.verification_label.setText(
            "Artifact identities and prediction binding validated. Optional input/replay checks have not run."
        )
        self.notify("Saved experiment loaded. No model or remote service was invoked.")

    def _show_result(self, frame_id):
        if self.saved is None or not frame_id:
            return
        summary = summarize_result(self.saved, frame_id)
        self.result_title.setText(summary["experiment"])
        self.result_count.setText(f"{summary['detections']:,} detections")
        self.result_classes.set_counts(summary["classes"])
        synthetic = summary["evidence"] == "synthetic_test"
        self.result_status.setText(
            "Synthetic evidence" if synthetic else summary["status"].capitalize()
        )
        self.result_status.setProperty("state", "synthetic" if synthetic else "success")
        self.result_status.style().polish(self.result_status)
        for key, metric in self.result_timings.items():
            metric.set_value(seconds(summary[key]))
        self.result_summary.setText(
            f"{summary['model']} / {summary['backend']}  ·  {summary['gpu'] or 'GPU unavailable'}  ·  Frame {frame_id} / {summary['split']}\n"
            f"Evidence: {summary['evidence']}\n"
            f"Preprocessing: {', '.join(summary['preprocessing']) or 'None (raw baseline)'}"
        )
        self.provenance_text.setText(
            f"Model: {summary['model']}\nOpenPCDet: {summary['model_identity'].get('openpcdet_revision', 'Unavailable')}\nCheckpoint SHA256: {summary['model_identity'].get('checkpoint_sha256', 'Unavailable')}\nEvidence kind: {summary['evidence']}\nDataset: {summary['dataset']}\nConfiguration SHA256: {summary['config_sha256']}\n\nSingle-frame timings are observations, not statistical benchmarks or FPS. Client end-to-end excludes loading/persistence. Detection time includes adapter preparation/transfers/postprocessing. Counts do not establish accuracy. Stored hashes establish consistency, not hardware attestation."
        )
        self.prediction_button.setEnabled(True)

    def open_predictions(self):
        if self.saved is None:
            return
        if self.viewer.running:
            self.error("Close the current 3D view first.")
            return
        saved, frame_id, gt = (
            self.saved,
            self.result_frames.currentText(),
            self.result_gt.isChecked(),
        )
        try:
            options = self._view_options()
        except ValueError as exc:
            self.error(exc)
            return

        def prepare(progress):
            directory = Path(tempfile.mkdtemp(dir=self.temporary.name))
            scan, predictions = export_prediction_copy(
                saved, frame_id, directory / f"{frame_id}.json"
            )
            return viewer_arguments(scan, options, predictions=predictions, ground_truth=gt)

        self.start_job(
            "Checking the original scan and preparing a new prediction copy…",
            prepare,
            self._launch_viewer,
        )

    def verify_run(self):
        if self.saved is None:
            return
        path, replay = self.saved.run.manifest_path, self.verify_inputs.isChecked()

        def done(report):
            checks = ", ".join(
                f"{key}: {value.get('status')}" for key, value in report.checks.items()
            )
            self.verification_label.setText(f"{report.status}\n{checks}")
            self.notify(
                f"Offline verification: {report.status}. " + "; ".join(map(str, report.failures))
            )

        self.start_job(
            "Verifying saved evidence offline…",
            lambda progress: verify_results(path, check_inputs=replay, replay_preprocessing=replay),
            done,
        )

    def _viewer_output(self, log):
        self.details.setPlainText(self.status.text() + "\n\n" + log)

    def _viewer_changed(self, active):
        self.stop_button.setEnabled(active)
        self.raw_button.setEnabled(
            not active and self.frame is not None and self.frame.cloud.count > 0
        )
        self.prediction_button.setEnabled(not active and self.saved is not None)
        self.compare_button.setEnabled(
            not active and self.processed is not None and self.processed_info.cloud.count > 0
        )

    def closeEvent(self, event):
        self.closing = True
        if self.task is not None or self.viewer.running:
            self.viewer.stop()
            self.notify("Closing the viewer and waiting for the current CPU operation to finish…")
            event.ignore()
            return
        self.temporary.cleanup()
        event.accept()

    def _maybe_close(self):
        if self.closing and self.task is None and not self.viewer.running:
            QTimer.singleShot(0, self.close)
