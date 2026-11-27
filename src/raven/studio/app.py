"""Lazy optional desktop entry point."""

import sys


def launch():
    try:
        from PySide6.QtWidgets import QApplication

        from raven.studio.main_window import MainWindow
        from raven.studio.theme import apply_theme
    except ImportError as exc:
        raise RuntimeError(
            'Raven Studio requires the optional desktop dependency. Install: python -m pip install -e ".[studio,visualization,preprocessing]"'
        ) from exc
    app = QApplication.instance() or QApplication([sys.argv[0]])
    app.setApplicationName("Raven Studio")
    app.setOrganizationName("Raven")
    apply_theme(app)
    window = MainWindow()
    window.show()
    return app.exec()
