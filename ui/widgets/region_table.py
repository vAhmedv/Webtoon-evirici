"""Region listesi widget'ı."""

from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView

from core.detection import Region


class RegionTable(QTableWidget):
    """Region sonuçlarını tablo halinde gösteren widget."""

    region_selected = Signal(object)  # Region

    def __init__(self, parent: Optional[object] = None) -> None:
        super().__init__(parent)
        self._regions: List[Region] = []
        self.setColumnCount(8)
        self.setHorizontalHeaderLabels(["ID", "Type", "Conf", "OCR Conf", "Status", "BBox", "Windows", "Text"])
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeToContents)
        self.horizontalHeader().setSectionResizeMode(7, QHeaderView.ResizeToContents)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setAlternatingRowColors(True)
        self.itemSelectionChanged.connect(self._on_selection_changed)

    def set_regions(self, regions: List[Region]) -> None:
        """Region listesini yükler."""
        self._regions = regions
        self.setRowCount(len(regions))
        for row, reg in enumerate(regions):
            self.setItem(row, 0, QTableWidgetItem(f"R{reg.id:03d}"))
            self.setItem(row, 1, QTableWidgetItem(reg.type.value.upper()))
            self.setItem(row, 2, QTableWidgetItem(f"{reg.detection_confidence:.2f}"))
            ocr_conf = f"{reg.ocr_confidence:.2f}" if reg.ocr_confidence is not None else "-"
            self.setItem(row, 3, QTableWidgetItem(ocr_conf))
            self.setItem(row, 4, QTableWidgetItem(reg.status.value.upper()))
            bbox = reg.global_bbox
            self.setItem(row, 5, QTableWidgetItem(f"{bbox.x1},{bbox.y1},{bbox.x2},{bbox.y2}"))
            self.setItem(row, 6, QTableWidgetItem(",".join(str(w) for w in reg.source_window_ids)))
            text_preview = (reg.text or "")[:50]
            self.setItem(row, 7, QTableWidgetItem(text_preview))

        self.resizeColumnsToContents()

    def _on_selection_changed(self) -> None:
        selected = self.selectedItems()
        if not selected:
            return
        row = selected[0].row()
        if 0 <= row < len(self._regions):
            self.region_selected.emit(self._regions[row])
