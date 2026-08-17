"""LeftSidebar component for page navigation and thumbnail list."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QPixmap, QIcon, QPainter, QColor, QPen
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)
from core.models import Page


class PageCardWidget(QWidget):
    """Custom widget representing a single page in the sidebar list."""

    def __init__(
        self,
        page: Page,
        region_count: int = 0,
        review_count: int = 0,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.page = page

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(10)

        # Thumbnail Label
        self.thumb_label = QLabel()
        self.thumb_label.setFixedSize(48, 64)
        self.thumb_label.setStyleSheet("background-color: #18181B; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 4px;")
        self.thumb_label.setAlignment(Qt.AlignCenter)
        self._load_thumbnail()
        layout.addWidget(self.thumb_label)

        # Details Column
        details_box = QVBoxLayout()
        details_box.setSpacing(3)
        details_box.setContentsMargins(0, 0, 0, 0)

        title_row = QHBoxLayout()
        title_row.setSpacing(6)
        page_num_label = QLabel(f"PAGE #{page.index + 1:02d}")
        page_num_label.setStyleSheet("font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 600; color: #FAFAFA;")
        
        self.status_dot = QLabel("●")
        if review_count > 0:
            self.status_dot.setStyleSheet("color: #F59E0B; font-size: 9px;")
            self.status_dot.setToolTip(f"{review_count} region(s) require human review")
        else:
            self.status_dot.setStyleSheet("color: #10B981; font-size: 9px;")
            self.status_dot.setToolTip("All regions confirmed or auto")

        title_row.addWidget(page_num_label)
        title_row.addWidget(self.status_dot)
        title_row.addStretch(1)
        details_box.addLayout(title_row)

        dim_label = QLabel(f"{page.width} × {page.height} px")
        dim_label.setStyleSheet("color: #71717A; font-size: 10px; font-family: 'JetBrains Mono', monospace;")
        details_box.addWidget(dim_label)

        region_label = QLabel(f"{region_count} regions" if region_count > 0 else "Ready")
        region_label.setStyleSheet("color: #A1A1AA; font-size: 10px;")
        details_box.addWidget(region_label)

        layout.addLayout(details_box)

    def _load_thumbnail(self) -> None:
        if self.page.path and Path(self.page.path).exists():
            pix = QPixmap(str(self.page.path))
            if not pix.isNull():
                scaled = pix.scaled(48, 64, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
                self.thumb_label.setPixmap(scaled)
                return
        self.thumb_label.setText(f"#{self.page.index + 1}")
        self.thumb_label.setStyleSheet("color: #52525B; font-size: 10px;")


class LeftSidebar(QFrame):
    """Left sidebar containing the list of pages and thumbnails."""

    page_selected = Signal(int)  # page_index

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("leftSidebar")
        self.setFixedWidth(240)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 12, 8, 12)
        layout.setSpacing(8)

        # Header
        header_row = QHBoxLayout()
        header_row.setContentsMargins(4, 0, 4, 0)
        self.title_label = QLabel("PAGES")
        self.title_label.setStyleSheet("font-size: 11px; font-weight: 600; color: #71717A; letter-spacing: 0.5px;")
        
        self.count_badge = QLabel("0")
        self.count_badge.setStyleSheet(
            "background-color: #18181B; color: #FAFAFA; border: 1px solid rgba(255, 255, 255, 0.08); "
            "border-radius: 10px; padding: 1px 6px; font-size: 10px; font-family: 'JetBrains Mono', monospace;"
        )

        header_row.addWidget(self.title_label)
        header_row.addStretch(1)
        header_row.addWidget(self.count_badge)
        layout.addLayout(header_row)

        # Page List
        self.page_list = QListWidget(self)
        self.page_list.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.page_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.page_list.currentRowChanged.connect(self._on_row_changed)
        layout.addWidget(self.page_list)

    def load_pages(
        self,
        pages: Sequence[Page],
        regions_per_page: Optional[dict[int, int]] = None,
        reviews_per_page: Optional[dict[int, int]] = None,
    ) -> None:
        self.page_list.clear()
        self.count_badge.setText(str(len(pages)))

        regions_per_page = regions_per_page or {}
        reviews_per_page = reviews_per_page or {}

        for page in pages:
            item = QListWidgetItem(self.page_list)
            item.setSizeHint(QSize(220, 78))
            card = PageCardWidget(
                page=page,
                region_count=regions_per_page.get(page.index, 0),
                review_count=reviews_per_page.get(page.index, 0),
                parent=self.page_list,
            )
            self.page_list.addItem(item)
            self.page_list.setItemWidget(item, card)

    def select_page(self, page_index: int) -> None:
        if 0 <= page_index < self.page_list.count():
            self.page_list.setCurrentRow(page_index)

    def _on_row_changed(self, row: int) -> None:
        if row >= 0:
            self.page_selected.emit(row)
