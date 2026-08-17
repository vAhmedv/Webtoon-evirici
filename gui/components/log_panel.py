"""Log panel and Qt log emitter for thread-safe UI log handling."""

from __future__ import annotations

import threading
from typing import Optional

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import QPushButton, QTextEdit, QVBoxLayout, QWidget
from loguru import logger


class QtLogEmitter(QObject):
    """Qt signal emitter to forward loguru messages safely to the GUI thread."""

    message = Signal(str)


class LogPanel(QWidget):
    """Scrollable thread-safe log panel widget."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._text_edit = QTextEdit(self)
        self._text_edit.setReadOnly(True)
        self._clear_btn = QPushButton("Clear Log", self)
        self._clear_btn.clicked.connect(self._text_edit.clear)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._text_edit)
        layout.addWidget(self._clear_btn, 0, Qt.AlignRight)

        self._emitter = QtLogEmitter(self)
        self._handler_id = logger.add(
            self._emit_log,
            format="{time:HH:mm:ss} | {message}",
            level="DEBUG",
        )
        self._emitter.message.connect(self._append_log)

    def _emit_log(self, message: str) -> None:
        """loguru sink callback — thread-safe signal emit."""
        if getattr(self, "_emitting", False):
            return
        self._emitting = True
        try:
            self._emitter.message.emit(message.rstrip())
        finally:
            self._emitting = False

    def _append_log(self, message: str) -> None:
        """Appends log text in GUI thread."""
        if getattr(self, "_appending", False):
            return
        self._appending = True
        try:
            self._text_edit.append(message)
            cursor = self._text_edit.textCursor()
            cursor.movePosition(QTextCursor.End)
            self._text_edit.setTextCursor(cursor)
        finally:
            self._appending = False

    def cleanup(self) -> None:
        """Disconnects signals and removes loguru handler."""
        try:
            self._emitter.message.disconnect(self._append_log)
        except (RuntimeError, TypeError):
            pass
        try:
            logger.remove(self._handler_id)
        except ValueError:
            pass

