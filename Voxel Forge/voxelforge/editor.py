import re

from PyQt5 import QtWidgets, QtGui
from pyqtgraph.Qt import QtCore


class VFLineNumberArea(QtWidgets.QWidget):
    def __init__(self, editor):
        super().__init__(editor)
        self.editor = editor

    def sizeHint(self):
        return QtCore.QSize(self.editor.line_number_area_width(), 0)

    def paintEvent(self, event):
        self.editor.line_number_area_paint_event(event)


class VFCodeEditor(QtWidgets.QPlainTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._line_number_area = VFLineNumberArea(self)
        self.blockCountChanged.connect(self._update_line_number_area_width)
        self.updateRequest.connect(self._update_line_number_area)
        self.cursorPositionChanged.connect(self._highlight_current_line)
        self._update_line_number_area_width(0)
        self._highlight_current_line()
        self.setTabStopDistance(4 * self.fontMetrics().horizontalAdvance(" "))

    def line_number_area_width(self):
        digits = len(str(max(1, self.blockCount())))
        return 12 + self.fontMetrics().horizontalAdvance("9") * digits

    def _update_line_number_area_width(self, _):
        self.setViewportMargins(self.line_number_area_width(), 0, 0, 0)

    def _update_line_number_area(self, rect, dy):
        if dy:
            self._line_number_area.scroll(0, dy)
        else:
            self._line_number_area.update(0, rect.y(), self._line_number_area.width(), rect.height())
        if rect.contains(self.viewport().rect()):
            self._update_line_number_area_width(0)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        cr = self.contentsRect()
        self._line_number_area.setGeometry(QtCore.QRect(cr.left(), cr.top(), self.line_number_area_width(), cr.height()))

    def line_number_area_paint_event(self, event):
        painter = QtGui.QPainter(self._line_number_area)
        painter.fillRect(event.rect(), QtGui.QColor("#1E1E1E"))
        block = self.firstVisibleBlock()
        block_number = block.blockNumber()
        top = int(self.blockBoundingGeometry(block).translated(self.contentOffset()).top())
        bottom = top + int(self.blockBoundingRect(block).height())
        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                number = str(block_number + 1)
                painter.setPen(QtGui.QColor("#858585"))
                painter.drawText(
                    0,
                    top,
                    self._line_number_area.width() - 6,
                    self.fontMetrics().height(),
                    QtCore.Qt.AlignRight,
                    number,
                )
            block = block.next()
            top = bottom
            bottom = top + int(self.blockBoundingRect(block).height())
            block_number += 1

    def _highlight_current_line(self):
        selections = []
        if not self.isReadOnly():
            selection = QtWidgets.QTextEdit.ExtraSelection()
            selection.format.setBackground(QtGui.QColor("#2A2A2A"))
            selection.format.setProperty(QtGui.QTextFormat.FullWidthSelection, True)
            selection.cursor = self.textCursor()
            selection.cursor.clearSelection()
            selections.append(selection)
        self.setExtraSelections(selections)


class VFCodeHighlighter(QtGui.QSyntaxHighlighter):
    def __init__(self, document):
        super().__init__(document)
        self.rules = []
        cmd_fmt = QtGui.QTextCharFormat()
        cmd_fmt.setForeground(QtGui.QColor("#4FC1FF"))
        cmd_fmt.setFontWeight(QtGui.QFont.Bold)
        cmds = (
            "GROUND_SIZE", "GROUND_COLOR", "SKY_COLOR", "CLEAR", "IMAGE", "MODEL", "PICTURE",
            "AUDIO", "VIDEO", "URLCHECK", "FACEMODULE", "WAIT", "AT", "ROTATE", "SETCOLOR",
            "BOX", "SPHERE", "CYLINDER", "CLASS", "NEW",
            "SET", "APPLY", "DELETE", "SAVEVF", "LOADVF", "HELP",
            "SCENE", "VIEWPORT", "CAMERA", "LAYER", "SPRITE", "MOVE", "SOLID", "TAG", "HIDE", "SHOW",
        )
        self.rules.append((re.compile(r"^\s*(" + "|".join(cmds) + r")\b"), cmd_fmt))

        key_fmt = QtGui.QTextCharFormat()
        key_fmt.setForeground(QtGui.QColor("#C586C0"))
        self.rules.append((re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)="), key_fmt))

        str_fmt = QtGui.QTextCharFormat()
        str_fmt.setForeground(QtGui.QColor("#CE9178"))
        self.rules.append((re.compile(r"\"[^\"\\]*(?:\\.[^\"\\]*)*\""), str_fmt))
        self.rules.append((re.compile(r"'[^'\\]*(?:\\.[^'\\]*)*'"), str_fmt))

        num_fmt = QtGui.QTextCharFormat()
        num_fmt.setForeground(QtGui.QColor("#B5CEA8"))
        self.rules.append((re.compile(r"\b-?\d+(?:\.\d+)?\b"), num_fmt))

        hash_fmt = QtGui.QTextCharFormat()
        hash_fmt.setForeground(QtGui.QColor("#D7BA7D"))
        self.rules.append((re.compile(r"#(?:[0-9a-fA-F]{6}|[0-9a-fA-F]{3})\b"), hash_fmt))

        url_fmt = QtGui.QTextCharFormat()
        url_fmt.setForeground(QtGui.QColor("#9CDCFE"))
        self.rules.append((re.compile(r"https?://\S+"), url_fmt))

        self.comment_fmt = QtGui.QTextCharFormat()
        self.comment_fmt.setForeground(QtGui.QColor("#6A9955"))

    def highlightBlock(self, text):
        stripped = text.lstrip()
        if stripped.startswith("#"):
            comment_idx = len(text) - len(stripped)
            self.setFormat(comment_idx, len(text) - comment_idx, self.comment_fmt)
            return
        for pattern, fmt in self.rules:
            for m in pattern.finditer(text):
                self.setFormat(m.start(), m.end() - m.start(), fmt)
