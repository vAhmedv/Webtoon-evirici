"""Application entry point for the Linear-Style Webtoon Translator GUI."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
from gui.main_window import MainWindow


def main() -> None:
    # High DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("Webtoon Translator")
    app.setOrganizationName("Antigravity")

    window = MainWindow()
    window.show()

    # If chapter passed as argument, open it directly
    if len(sys.argv) > 1:
        chapter_dir = sys.argv[1]
        if Path(chapter_dir).exists():
            window.open_chapter(chapter_dir)
    else:
        # Default load Chapter 1 if available
        default_ch = Path(
            r"C:\Users\Ahmed\AppData\Local\Tachidesk\downloads\mangas\Asmodeus Scans (EN)\Reincarnated as a God-Tier Crafter\Chapter 1"
        )
        if default_ch.exists():
            window.open_chapter(default_ch)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
