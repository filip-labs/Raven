"""Heimdall palette translated to native Qt widgets; no copied artwork or fonts."""

import re

TOKENS = {
    "background": "#F2F2F2",
    "panel": "#FFFFFF",
    "text": "#151515",
    "muted": "#616161",
    "border": "#E3E5E7",
    "accent": "#FFAC00",
    "success": "#629B5C",
    "danger": "#C6391B",
    "focus": "#B87800",
    "accent_soft": "#FFF3D6",
    "accent_hover": "#F5A500",
    "field_border": "#D8DCE0",
    "indicator_border": "#858D95",
    "readout_surface": "#F7F8F9",
    "disabled_surface": "#ECEEF0",
    "disabled_text": "#8B9197",
    "scroll_thumb": "#BDC3C8",
    "scroll_hover": "#959DA5",
    "success_surface": "#EDF5EB",
    "success_text": "#3E6B39",
}


def apply_theme(app):
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QColor, QFont, QFontDatabase, QPalette

    from raven.studio.widgets import StudioStyle

    app.setStyle(StudioStyle())  # Native behavior, consistent light controls.
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
    #sidebar QLabel { color: #A6A6A6; background: transparent; font-size: 12px; }
    #sidebar QLabel#sidebar_caption { font-size: 10px; font-weight: 500; }
    #sidebar QLabel#brand { color: @panel; font-size: 19px; font-weight: 600; }
    #sidebar QPushButton { text-align: left; background: transparent; color: #C8C8C8; border: 1px solid transparent; border-radius: 12px; padding: 11px 14px; }
    #sidebar QPushButton:checked { background: @accent; color: @text; }
    #sidebar QPushButton:hover { background: #303030; color: @panel; }
    #sidebar QPushButton:checked:hover { background: @accent_hover; color: @text; }
    #sidebar QPushButton:focus { border-color: @focus; }
    QLabel#title { font-size: 28px; font-weight: 500; }
    QLabel#section_title { font-size: 21px; font-weight: 500; }
    QLabel#subtitle, QLabel#muted { color: @muted; font-size: 13px; }
    QLabel#hero { font-size: 38px; font-weight: 500; }
    QFrame#panel { background: @panel; border: 1px solid @border; border-radius: 20px; }
    QPushButton { background: @text; color: @panel; border: 1px solid @text; border-radius: 18px; padding: 8px 16px; min-height: 20px; font-size: 14px; font-weight: 500; }
    QPushButton:hover { background: #303030; }
    QPushButton:pressed { background: #050505; }
    QPushButton[tone="primary"] { background: @accent; color: @text; border-color: @accent; }
    QPushButton[tone="primary"]:hover { background: @accent_hover; border-color: @accent_hover; }
    QPushButton[tone="primary"]:pressed { background: #E89C00; }
    QPushButton[tone="secondary"] { background: @panel; color: @text; border-color: @field_border; }
    QPushButton[tone="secondary"]:hover { background: @readout_surface; border-color: @indicator_border; }
    QPushButton[tone="secondary"]:pressed { background: @disabled_surface; }
    QPushButton:focus, QPushButton[tone="primary"]:focus, QPushButton[tone="secondary"]:focus { border-color: @focus; }
    QPushButton[tone="primary"]:disabled, QPushButton[tone="secondary"]:disabled, QPushButton:disabled { background: @disabled_surface; color: @disabled_text; border-color: @border; }
    QLineEdit, QComboBox, QAbstractSpinBox { background: @panel; border: 1px solid @field_border; border-radius: 10px; padding: 7px 10px; min-height: 22px; selection-background-color: @accent; font-size: 14px; }
    QLineEdit:hover, QComboBox:hover, QAbstractSpinBox:hover { border-color: @indicator_border; }
    QLineEdit:focus, QComboBox:focus, QAbstractSpinBox:focus { border-color: @focus; }
    QLineEdit:disabled, QComboBox:disabled, QAbstractSpinBox:disabled { background: @disabled_surface; color: @disabled_text; border-color: @border; }
    QComboBox { padding-right: 32px; }
    QComboBox::drop-down { subcontrol-origin: padding; subcontrol-position: top right; width: 30px; border: none; background: transparent; }
    QComboBox::down-arrow { image: none; width: 0px; height: 0px; }
    QComboBox QAbstractItemView { background: @panel; color: @text; border: 1px solid @border; padding: 6px; selection-background-color: @accent_soft; selection-color: @text; outline: none; }
    QCheckBox { spacing: 10px; padding: 6px 2px; min-height: 24px; font-size: 14px; }
    QCheckBox::indicator, QGroupBox::indicator, QListWidget::indicator { width: 20px; height: 20px; }
    QCheckBox:focus { color: @focus; }
    QCheckBox:disabled { color: @disabled_text; }
    QListWidget { background: @panel; border: 1px solid @border; border-radius: 12px; padding: 5px; outline: none; }
    QListWidget:focus { border-color: @focus; }
    QListWidget::item { padding: 9px; border-radius: 7px; }
    QListWidget::item:selected { background: @accent_soft; color: @text; }
    QHeaderView::section { background: @background; border: none; padding: 8px; color: @muted; font-size: 12px; }
    QTableWidget { background: @panel; gridline-color: @border; border: none; }
    QGroupBox { border: 1px solid @border; border-radius: 12px; margin-top: 12px; padding: 14px; }
    QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; }
    QProgressBar { border: none; background: @border; border-radius: 3px; max-height: 4px; }
    QProgressBar::chunk { background: @accent; }
    QScrollBar:vertical { background: transparent; width: 10px; margin: 3px 1px; border: none; }
    QScrollBar::handle:vertical { background: @scroll_thumb; border-radius: 4px; min-height: 32px; }
    QScrollBar::handle:vertical:hover { background: @scroll_hover; }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; border: none; background: none; }
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }
    QScrollBar:horizontal { background: transparent; height: 10px; margin: 1px 3px; border: none; }
    QScrollBar::handle:horizontal { background: @scroll_thumb; border-radius: 4px; min-width: 32px; }
    QScrollBar::handle:horizontal:hover { background: @scroll_hover; }
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0px; border: none; background: none; }
    QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal, QAbstractScrollArea::corner { background: none; }
    QFrame#readout { background: @readout_surface; border: none; border-radius: 12px; }
    QLabel#readout_caption { color: @muted; font-size: 13px; }
    QLabel#readout_value { color: @text; font-size: 28px; font-weight: 600; }
    QLabel#readout_value[compact="true"] { font-size: 18px; font-weight: 500; }
    QLabel#result_status { border-radius: 9px; padding: 5px 10px; background: @readout_surface; color: @muted; font-size: 12px; font-weight: 500; }
    QLabel#result_status[state="success"] { background: @success_surface; color: @success_text; }
    QLabel#result_status[state="synthetic"] { background: @accent_soft; color: @text; }
    QToolTip { background: @text; color: @panel; padding: 8px; border: none; }
    """
    stylesheet = re.sub(r"@([a-z_]+)", lambda match: TOKENS[match[1]], stylesheet)
    app.setStyleSheet(stylesheet)
    app.setAttribute(Qt.ApplicationAttribute.AA_DontShowIconsInMenus, True)
