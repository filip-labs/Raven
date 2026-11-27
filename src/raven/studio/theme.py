"""Heimdall palette translated to native Qt widgets; no copied artwork or fonts."""

TOKENS = {
    "background": "#F2F2F2",
    "panel": "#FFFFFF",
    "text": "#151515",
    "muted": "#616161",
    "border": "#E3E5E7",
    "accent": "#FFAC00",
    "success": "#629B5C",
    "danger": "#C6391B",
}


def apply_theme(app):
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QColor, QFont, QFontDatabase, QPalette

    app.setStyle("Fusion")  # Consistent light controls regardless of OS dark preference.
    font = QFont()
    font.setFamily(
        next(
            (
                family
                for family in ("Figtree", "SF Pro Text", "Helvetica Neue", "Arial")
                if family in QFontDatabase.families()
            ),
            app.font().family(),
        )
    )
    font.setPointSize(12)
    app.setFont(font)
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(TOKENS["background"]))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(TOKENS["text"]))
    palette.setColor(QPalette.ColorRole.Base, QColor(TOKENS["panel"]))
    palette.setColor(QPalette.ColorRole.Text, QColor(TOKENS["text"]))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(TOKENS["accent"]))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(TOKENS["text"]))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(TOKENS["muted"]))
    app.setPalette(palette)
    stylesheet = """
    QWidget { color: @text; }
    QMainWindow, QWidget#workspace { background: @background; }
    QFrame#sidebar { background: @text; border: none; }
    #sidebar QLabel { color: #A6A6A6; background: transparent; }
    #sidebar QLabel#brand { color: white; font-size: 19px; font-weight: 600; }
    #sidebar QPushButton { text-align: left; background: transparent; color: #C8C8C8; border: none; border-radius: 12px; padding: 13px 15px; }
    #sidebar QPushButton:checked { background: @accent; color: @text; }
    #sidebar QPushButton:hover { background: #353535; color: white; }
    #sidebar QPushButton:checked:hover { background: #F2A400; color: @text; }
    QLabel#title { font-size: 30px; font-weight: 500; }
    QLabel#subtitle, QLabel#muted { color: @muted; }
    QLabel#hero { font-size: 42px; font-weight: 500; }
    QFrame#panel { background: white; border: 1px solid @border; border-radius: 20px; }
    QPushButton { background: @text; color: white; border: 1px solid @text; border-radius: 18px; padding: 10px 18px; min-height: 18px; }
    QPushButton:hover { background: #353535; }
    QPushButton[tone="primary"] { background: @accent; color: @text; border-color: @accent; }
    QPushButton[tone="secondary"] { background: @panel; color: @text; border-color: @border; }
    QPushButton[tone="primary"]:disabled, QPushButton[tone="secondary"]:disabled, QPushButton:disabled { background: @border; color: #777777; border-color: @border; }
    QPushButton:focus, QLineEdit:focus, QComboBox:focus, QAbstractSpinBox:focus, QListWidget:focus { border: 2px solid #B87800; }
    QLineEdit, QComboBox, QAbstractSpinBox { background: white; border: 1px solid #D3D5D7; border-radius: 8px; padding: 7px; min-height: 20px; selection-background-color: @accent; }
    QComboBox QAbstractItemView { background: white; color: @text; selection-background-color: @accent; }
    QListWidget { background: white; border: 1px solid @border; border-radius: 12px; padding: 6px; outline: none; }
    QListWidget::item { padding: 10px; border-radius: 7px; }
    QListWidget::item:selected { background: #FFF1CF; color: @text; }
    QHeaderView::section { background: @background; border: none; padding: 8px; color: @muted; }
    QTableWidget { background: white; gridline-color: @border; border: none; }
    QGroupBox { border: 1px solid @border; border-radius: 10px; margin-top: 12px; padding: 12px; }
    QGroupBox::title { subcontrol-origin: margin; left: 12px; }
    QProgressBar { border: none; background: @border; border-radius: 3px; max-height: 5px; }
    QProgressBar::chunk { background: @accent; }
    QToolTip { background: @text; color: white; padding: 8px; }
    """
    for token, value in TOKENS.items():
        stylesheet = stylesheet.replace("@" + token, value)
    app.setStyleSheet(stylesheet)
    app.setAttribute(Qt.ApplicationAttribute.AA_DontShowIconsInMenus, True)
