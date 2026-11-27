"""Small presentation-only widgets; native Qt interaction and accessibility are retained."""

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QLabel,
    QProxyStyle,
    QSizePolicy,
    QStyle,
    QStyleOptionComboBox,
    QVBoxLayout,
    QWidget,
)

from raven.studio.theme import TOKENS


def _chevron(painter, rect, color, *, up=False):
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(
        QPen(
            QColor(color),
            1.7,
            Qt.PenStyle.SolidLine,
            Qt.PenCapStyle.RoundCap,
            Qt.PenJoinStyle.RoundJoin,
        )
    )
    center = rect.center()
    direction = -1 if up else 1
    path = QPainterPath(QPointF(center.x() - 4, center.y() - direction * 2))
    path.lineTo(center.x(), center.y() + direction * 2)
    path.lineTo(center.x() + 4, center.y() - direction * 2)
    painter.drawPath(path)
    painter.restore()


class StudioStyle(QProxyStyle):
    """Paint crisp vector indicators at every display scale, including list/group checks."""

    def __init__(self):
        super().__init__("Fusion")

    def pixelMetric(self, metric, option=None, widget=None):
        if metric in (QStyle.PixelMetric.PM_IndicatorWidth, QStyle.PixelMetric.PM_IndicatorHeight):
            return 20
        if metric == QStyle.PixelMetric.PM_CheckBoxLabelSpacing:
            return 10
        return super().pixelMetric(metric, option, widget)

    def drawPrimitive(self, element, option, painter, widget=None):
        if element in (
            QStyle.PrimitiveElement.PE_IndicatorArrowDown,
            QStyle.PrimitiveElement.PE_IndicatorArrowUp,
        ):
            _chevron(
                painter,
                QRectF(option.rect),
                TOKENS["muted"],
                up=element == QStyle.PrimitiveElement.PE_IndicatorArrowUp,
            )
            return
        if element not in (
            QStyle.PrimitiveElement.PE_IndicatorCheckBox,
            QStyle.PrimitiveElement.PE_IndicatorItemViewItemCheck,
        ):
            return super().drawPrimitive(element, option, painter, widget)
        enabled = bool(option.state & QStyle.StateFlag.State_Enabled)
        checked = bool(option.state & (QStyle.StateFlag.State_On | QStyle.StateFlag.State_NoChange))
        hover = bool(option.state & QStyle.StateFlag.State_MouseOver)
        focus = bool(option.state & QStyle.StateFlag.State_HasFocus)
        fill = TOKENS["accent"] if checked else TOKENS["panel"]
        border = TOKENS["text"] if checked else TOKENS["indicator_border"]
        if hover or focus:
            border = TOKENS["focus"]
            if not checked:
                fill = TOKENS["accent_soft"]
        if not enabled:
            fill, border = TOKENS["disabled_surface"], TOKENS["border"]
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QColor(fill))
        painter.setPen(QPen(QColor(border), 1.8 if focus else 1.3))
        rect = QRectF(option.rect).adjusted(1.5, 1.5, -1.5, -1.5)
        painter.drawRoundedRect(rect, 4.5, 4.5)
        if checked:
            painter.setPen(
                QPen(
                    QColor(TOKENS["text"] if enabled else TOKENS["muted"]),
                    2.0,
                    Qt.PenStyle.SolidLine,
                    Qt.PenCapStyle.RoundCap,
                    Qt.PenJoinStyle.RoundJoin,
                )
            )
            if option.state & QStyle.StateFlag.State_NoChange:
                painter.drawLine(
                    QPointF(rect.left() + 4, rect.center().y()),
                    QPointF(rect.right() - 4, rect.center().y()),
                )
            else:
                path = QPainterPath(QPointF(rect.left() + 3.5, rect.center().y()))
                path.lineTo(rect.left() + 7, rect.bottom() - 4)
                path.lineTo(rect.right() - 3.5, rect.top() + 4)
                painter.drawPath(path)
        painter.restore()


class ComboBox(QComboBox):
    """Native combo behavior with one restrained, antialiased chevron."""

    def paintEvent(self, event):
        super().paintEvent(event)
        option = QStyleOptionComboBox()
        self.initStyleOption(option)
        rect = self.style().subControlRect(
            QStyle.ComplexControl.CC_ComboBox, option, QStyle.SubControl.SC_ComboBoxArrow, self
        )
        painter = QPainter(self)
        _chevron(
            painter, QRectF(rect), TOKENS["muted"] if self.isEnabled() else TOKENS["disabled_text"]
        )
        painter.end()


class PathLabel(QLabel):
    """Keep long source paths available without forcing horizontal page overflow."""

    def __init__(self, text):
        super().__init__()
        self._full_text = ""
        self.setObjectName("muted")
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setTextFormat(Qt.TextFormat.PlainText)
        self.setText(text)

    def setText(self, text):
        self._full_text = text
        self.setToolTip(text)
        self.setAccessibleName(text)
        self._elide()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._elide()

    def _elide(self):
        super().setText(
            self.fontMetrics().elidedText(
                self._full_text, Qt.TextElideMode.ElideMiddle, max(0, self.width())
            )
        )


class Readout(QFrame):
    """A label/value pair, not a new source of scientific values."""

    def __init__(self, title, *, compact=False):
        super().__init__()
        self.setObjectName("readout")
        self.setProperty("compact", compact)
        self.title = QLabel(title)
        self.title.setObjectName("readout_caption")
        self.title.setWordWrap(True)
        self.title.setTextFormat(Qt.TextFormat.PlainText)
        self.value = QLabel("—")
        self.value.setObjectName("readout_value")
        self.value.setProperty("compact", compact)
        self.value.setTextFormat(Qt.TextFormat.PlainText)
        self.value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(5)
        layout.addWidget(self.title)
        layout.addWidget(self.value)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.setMinimumWidth(0)

    def set_value(self, value):
        self.value.setText(value)
        self.value.setAccessibleName(f"{self.title.text()}: {value}")


class ClassCounts(QWidget):
    def __init__(self):
        super().__init__()
        self.grid = QGridLayout(self)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(10)
        self.counts = {}
        self.readouts = {}
        self.clear()

    def clear(self):
        self.set_counts(None)

    def set_counts(self, counts):
        while self.grid.count():
            widget = self.grid.takeAt(0).widget()
            widget.hide()
            widget.deleteLater()
        self.counts = {} if counts is None else dict(counts)
        self.readouts = {}
        if not counts:
            empty = QLabel(
                "Class counts appear here after opening a run."
                if counts is None
                else "No saved detections in this frame."
            )
            empty.setObjectName("muted")
            empty.setWordWrap(True)
            self.grid.addWidget(empty, 0, 0, 1, 3)
            return
        order = [name for name in ("Car", "Pedestrian", "Cyclist") if name in counts]
        order += sorted(name for name in counts if name not in order)
        for index, name in enumerate(order):
            metric = Readout(name)
            metric.set_value(f"{counts[name]:,}")
            self.readouts[name] = metric
            self.grid.addWidget(metric, index // 3, index % 3)
        for column in range(3):
            self.grid.setColumnStretch(column, 1)
