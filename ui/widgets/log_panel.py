"""Log paneli widget'ı."""

from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import QObject, Signal, Qt
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import QTextEdit, QVBoxLayout, QPushButton, QWidget

from loguru import logger


class LogPanel(QWidget):
    """Scrollable log paneli."""

    def __init__(self, parent: Optional[object] = None) -> None:
        super().__init__(parent)
        self._text_edit = QTextEdit(self)
        self._text_edit.setReadOnly(True)
        self._clear_btn = QPushButton("Clear Log", self)
        self._clear_btn.clicked.connect(self._text_edit.clear)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._text_edit)
        layout.addWidget(self._clear_btn, 0, Qt.AlignRight)

        self._log_handler = _QtLogHandler(self._append_log)
        logger.add(self._log_handler, format="{time:HH:mm:ss} | {message}", level="DEBUG")

    def _append_log(self, message: str) -> None:
        self._text_edit.append(message)
        cursor = self._text_edit.textCursor()
        cursor.movePosition(QTextCursor.End)
        self._text_edit.setTextCursor(cursor)


class _QtLogHandler:
    """loguru handler that emits Qt-safe signals."""

    def __init__(self, callback) -> None:
        self._callback = callback

    def write(self, message: str) -> None:
        self._callback(message.rstrip())
