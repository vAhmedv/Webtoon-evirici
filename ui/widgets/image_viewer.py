"""Görüntü önizleme widget'ı — QGraphicsView tabanlı."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem


class ImageViewer(QGraphicsView):
    """Window görselini gösteren önizleyici.

    Özellikler:
    - Otomatik sığdır (fit)
    - Mouse wheel zoom
    - Pencereler arası gezinme (set_window_image ile)
    """

    def __init__(self, parent: Optional[object] = None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item: Optional[QGraphicsPixmapItem] = None
        self._zoom = 1.0

        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setDragMode(QGraphicsView.ScrollHandDrag)

    def set_window_image(self, image_path: Path) -> None:
        """Yeni pencere görselini yükler ve sığdırır."""
        self._scene.clear()
        self._pixmap_item = None
        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            return
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._zoom = 1.0
        self.fit_in_view()

    def set_pixmap(self, pixmap: QPixmap) -> None:
        """QPixmap ile doğrudan görseli ayarlar."""
        self._scene.clear()
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._zoom = 1.0
        self.fit_in_view()

    def fit_in_view(self) -> None:
        if self._pixmap_item is None:
            return
        self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)
        self._zoom = 1.0

    def wheelEvent(self, event) -> None:  # type: ignore[override]
        """Mouse wheel ile zoom."""
        zoom_in = 1.15
        zoom_out = 1.0 / zoom_in
        if event.angleDelta().y() > 0:
            self.scale(zoom_in, zoom_in)
            self._zoom *= zoom_in
        else:
            self.scale(zoom_out, zoom_out)
            self._zoom *= zoom_out
        event.accept()
