"""Webtoon Çevirici — masaüstü uygulama giriş noktası."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from ui.main_window import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("Webtoon Çevirici")
    app.setOrganizationName("WebtoonTranslator")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
