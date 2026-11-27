"""Headless Qt integration; no Open3D renderer, datasets, credentials or network."""

import os
import threading
import time
from dataclasses import replace

import pytest
from test_modal_m5_integration import execute
from test_modal_m5_integration import remote as remote  # noqa: PLC0414
from test_studio_services import dataset as dataset  # noqa: PLC0414

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from test_remote_inference import config as config  # noqa: PLC0414 -- Imported fixture.

from raven.studio.main_window import MainWindow
from raven.studio.tasks import ViewerProcess, child_environment
from raven.studio.theme import apply_theme


@pytest.fixture(scope="module")
def app():
    instance = QApplication.instance() or QApplication([])
    apply_theme(instance)
    return instance


def wait(app, predicate, timeout=10):
    started = time.monotonic()
    while not predicate():
        assert time.monotonic() - started < timeout, "Qt operation did not finish"
        app.processEvents()
        QTest.qWait(10)
    app.processEvents()


@pytest.fixture
def window(app):
    w = MainWindow()
    w.show()
    app.processEvents()
    yield w
    w.close()
    wait(app, lambda: w.task is None and not w.viewer.running)
    app.processEvents()


def open_frame(window, app, dataset):
    window.open_dataset(dataset)
    wait(app, lambda: window.task is None and window.frame is not None)


def test_navigation_dataset_lazy_load_and_invalid_split_feedback(window, app, dataset):
    window.nav_buttons[1].setChecked(True)
    assert window.pages.currentIndex() == 1
    window.nav_buttons[0].click()
    open_frame(window, app, dataset)
    assert window.point_count.text() == "3" and window.frame.cloud.frame_id == "000000"
    assert not window.gt.isEnabled() and window.raw_button.isEnabled()
    window.split.setCurrentText("test")
    wait(app, lambda: window.task is None and window.index.config.split == "test")
    window.move_frame(1)
    wait(app, lambda: window.task is None and window.frame.cloud.frame_id == "000001")
    window.search.setText("000000")
    assert window.frames.item(1).isHidden()


def test_visual_pipeline_edit_apply_order_toggle_and_empty_comparison(
    window, app, dataset, monkeypatch
):
    open_frame(window, app, dataset)
    window.stage_type.setCurrentText("Voxel downsampling")
    window.add_stage()
    window.parameter_fields["voxel_size"][0][0].setText("0.2")
    window.apply_processing()
    wait(app, lambda: window.task is None)
    assert window.processed.cloud.count == 2
    assert "33.33%" in window.processing_summary.text()
    assert window.stage_stats.rowCount() == 1
    recorded = []
    monkeypatch.setattr(window.viewer, "start", recorded.append)
    window.open_comparison()
    wait(app, lambda: window.task is None)
    assert recorded and "raven.studio.viewer_process" in recorded[0]
    assert "--compare" not in recorded[0]  # No second pipeline execution.
    window.parameter_fields["voxel_size"][0][0].setText("-1")
    window.apply_processing()
    assert "positive" in window.status.text()
    window.parameter_fields["voxel_size"][0][0].setText(".2")
    window.stage_type.setCurrentText("Gaussian XYZ noise")
    window.add_stage()
    window.move_stage(-1)
    assert window.editor.active()[0]["name"] == "noise"
    window.stages.item(0).setCheckState(
        __import__("PySide6.QtCore", fromlist=["Qt"]).Qt.CheckState.Unchecked
    )
    assert window.editor.active()[0]["name"] == "voxel_downsample"
    window.reset_pipeline()
    assert not window.editor.active()


def test_saved_result_screen_verification_and_actual_prediction_copy(
    window, app, remote, monkeypatch
):
    config, _ = remote
    config = replace(config, preprocessing=())
    manifest = execute(config)
    window.open_run(manifest)
    wait(app, lambda: window.task is None)
    assert window.saved and window.result_count.text() == "1 detections"
    assert "Tesla T4" in window.result_summary.text()
    assert "synthetic_test" in window.provenance_text.text()
    recorded = []
    monkeypatch.setattr(window.viewer, "start", recorded.append)
    window.open_predictions()
    wait(app, lambda: window.task is None)
    assert "--predictions" in recorded[0]
    assert str(manifest.parent) not in recorded[0][-1]
    window.verify_inputs.setChecked(True)
    window.verify_run()
    wait(app, lambda: window.task is None)
    assert "checks_passed" in window.verification_label.text()


def test_viewer_qprocess_failure_duplicate_and_safe_environment(app):
    viewer = ViewerProcess()
    messages = []
    viewer.message.connect(messages.append)
    viewer.start(
        ["-c", "import time; time.sleep(.1); raise RuntimeError('fixture viewer failure')"]
    )
    with pytest.raises(RuntimeError, match="already open"):
        viewer.start(["-c", "pass"])
    wait(app, lambda: not viewer.running)
    assert any(
        "failed (exit" in message and "fixture viewer failure" in message for message in messages
    )
    env = child_environment()
    assert env.value("RAVEN_ALLOW_MODAL_GPU") == "0"
    assert "src" in env.value("PYTHONPATH") and not env.contains("RAVEN_MODAL_SMOKE_CONFIG")
    viewer.process.setProgram("/no/such/python")
    viewer.process.start()
    wait(app, lambda: not viewer.running)
    assert any("process error" in m for m in messages)


def test_close_waits_for_cpu_thread_and_stops_child_without_orphan(window, app):
    ids = []
    main = threading.get_ident()

    def job(progress):
        ids.append(threading.get_ident())
        time.sleep(0.15)
        return 1

    window.start_job("fixture", job, lambda value: None)
    window.viewer.start(["-c", "import time; time.sleep(30)"])
    window.close()
    assert window.task is not None and window.closing
    wait(app, lambda: window.task is None and not window.viewer.running)
    assert ids and ids[0] != main
    wait(app, lambda: not window.isVisible())


def test_empty_processed_result_and_screenshot_source_protection(window, app, dataset, monkeypatch):
    open_frame(window, app, dataset)
    window.stage_type.setCurrentText("Range filter")
    window.add_stage()
    window.parameter_fields["x_range"][0][0].setText("100")
    window.parameter_fields["x_range"][0][1].setText("101")
    window.apply_processing()
    wait(app, lambda: window.task is None)
    assert window.processed.cloud.count == 0 and window.compare_button.isEnabled()
    assert "100.00%" in window.processing_summary.text()
    with pytest.raises(ValueError, match="outside"):
        window._validate_screenshot_path(dataset)
    window.screenshot_path.setText(str(dataset))
    recorded = []
    monkeypatch.setattr(window.viewer, "start", recorded.append)
    window.open_raw()
    assert not recorded and "outside" in window.status.text()


def test_failed_loading_does_not_leave_previous_result_controls_enabled(
    window, app, remote, tmp_path
):
    config, _ = remote
    manifest = execute(config)
    window.open_run(manifest)
    wait(app, lambda: window.task is None)
    assert window.saved
    window.open_run(tmp_path / "missing.json")
    wait(app, lambda: window.task is None)
    assert window.saved is None and not window.prediction_button.isEnabled()
    assert window.result_count.text() == "—"


def test_action_accessibility_identifiers_are_unique_across_pages(window):
    from PySide6.QtWidgets import QPushButton

    buttons = window.findChildren(QPushButton)
    names = [b.objectName() for b in buttons]
    assert all(names) and len(names) == len(set(names))
    window.nav_buttons[2].setChecked(True)
    assert window.pages.currentIndex() == 2
    assert not window.nav_buttons[0].isChecked()


def test_implicit_viewer_files_stay_in_temporary_session_cache(window):
    from pathlib import Path

    root = Path(window.temporary.name)
    assert window.viewer.working_directory == root
    assert window._view_options().screenshot_dir == root / "screenshots"


def test_real_studio_cli_entry_opens_and_closes_a_qt_event_loop():
    import subprocess
    import sys

    script = """
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer
app=QApplication([])
QTimer.singleShot(300,app.closeAllWindows)
from raven.cli import main
main(['studio'])
"""
    subprocess.run(
        [sys.executable, "-c", script], check=True, capture_output=True, text=True, timeout=15
    )


def test_accessible_parameter_change_invalidates_previous_comparison(window, app, dataset):
    open_frame(window, app, dataset)
    window.stage_type.setCurrentText("Voxel downsampling")
    window.add_stage()
    window.apply_processing()
    wait(app, lambda: window.task is None)
    assert window.processed is not None
    window.parameter_fields["voxel_size"][0][0].setText(".4")
    assert window.processed is None and not window.compare_button.isEnabled()
