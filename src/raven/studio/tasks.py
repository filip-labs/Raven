"""A single CPU task and one managed external viewer; never use a shell."""

import os
import sys
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QThread, QTimer, Signal


class Task(QThread):
    succeeded = Signal(object)
    failed = Signal(str)
    progress = Signal(str)

    def __init__(self, function, parent=None):
        super().__init__(parent)
        self.function = function

    def run(self):
        try:
            self.succeeded.emit(self.function(self.progress.emit))
        except Exception as exc:  # noqa: BLE001 -- Report backend failures at the GUI boundary.
            self.failed.emit(f"{type(exc).__name__}: {exc}")


def child_environment():
    env = QProcessEnvironment.systemEnvironment()
    # Route to the source tree that actually launched Studio, even with an editable install elsewhere.
    source = str(Path(__file__).resolve().parents[2])
    previous = os.environ.get("PYTHONPATH", "")
    env.insert("PYTHONPATH", os.pathsep.join(p for p in (source, previous) if p))
    for gate in (
        "RAVEN_ALLOW_MODAL_GPU",
        "RAVEN_MODAL_ASSET_RIGHTS_CONFIRMED",
        "RAVEN_MODAL_TRUSTED_LOADING_SECRET",
    ):
        env.insert(gate, "0")
    env.insert("PYTHONUNBUFFERED", "1")
    env.remove("RAVEN_MODAL_SMOKE_CONFIG")
    return env


class ViewerProcess(QObject):
    message = Signal(str)
    changed = Signal(bool)
    ended = Signal()
    output = Signal(str)

    def __init__(self, parent=None, *, working_directory=None):
        super().__init__(parent)
        self.working_directory = Path(working_directory) if working_directory else Path.home()
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read)
        self.process.started.connect(
            lambda: self.message.emit(
                "3D viewer process running. Use its separate Open3D window; Details shows startup output."
            )
        )
        self.process.finished.connect(self._finished)
        self.process.errorOccurred.connect(self._error)
        self.log = ""
        self.stopping = False
        self.kill_timer = QTimer(self)
        self.kill_timer.setSingleShot(True)
        self.kill_timer.timeout.connect(self.process.kill)

    @property
    def running(self):
        return self.process.state() != QProcess.ProcessState.NotRunning

    def start(self, arguments):
        if self.running:
            raise RuntimeError(
                "A 3D viewer is already open. Close it or select Stop 3D View first."
            )
        self.log = ""
        self.stopping = False
        self.process.setProgram(sys.executable)
        self.process.setArguments(arguments)
        self.process.setProcessEnvironment(child_environment())
        self.process.setWorkingDirectory(str(self.working_directory))
        self.process.start()
        self.changed.emit(True)
        self.message.emit("Opening interactive Open3D window…")

    def _read(self):
        self.log = (
            self.log + bytes(self.process.readAllStandardOutput()).decode(errors="replace")
        )[-16000:]
        self.output.emit(self.log)

    def _error(self, error):
        self.message.emit(f"Viewer process error: {self.process.errorString()}")
        if error == QProcess.ProcessError.FailedToStart:
            self.changed.emit(False)
            self.ended.emit()

    def _finished(self, code, status):
        self.kill_timer.stop()
        self._read()
        if self.stopping:
            message = "3D view stopped."
        elif code != 0 or status != QProcess.ExitStatus.NormalExit:
            message = f"3D viewer failed (exit {code}). {self.log[-2000:]}"
        else:
            message = "3D view closed."
        self.message.emit(message)
        self.changed.emit(False)
        self.ended.emit()

    def stop(self):
        if self.running:
            self.stopping = True
            self.process.terminate()
            self.kill_timer.start(1500)
