"""WebtoonCanvas component: high-performance vertical scrollable QGraphicsView with page dividers, auto-scroll, Before/After split slider, and hold-to-peek."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Any
from PySide6.QtCore import Qt, Signal, QRectF, QPointF
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QPainter,
    QPen,
    QPixmap,
    QWheelEvent,
    QMouseEvent,
    QKeyEvent,
    QCursor,
)
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGraphicsDropShadowEffect,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.detection.detection import Region, RegionStatus, RegionType
from core.models import Page


class RegionBoxGraphicsItem(QGraphicsRectItem):
    """Monochrome Bounding Box with Linear/Raycast subtle glow on selection."""

    def __init__(
        self,
        region: Region,
        on_clicked_callback: Any,
        parent: Optional[QGraphicsItem] = None,
    ) -> None:
        gb = region.global_bbox
        rect = QRectF(gb.x1, gb.y1, gb.width, gb.height)
        super().__init__(rect, parent)

        self.region = region
        self.on_clicked_callback = on_clicked_callback
        self._is_selected = False

        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.PointingHandCursor)

        # ID Chip Text Item
        self.label_item = QGraphicsSimpleTextItem(f"#{region.id}", self)
        font = QFont("JetBrains Mono", 8, QFont.Bold)
        self.label_item.setFont(font)
        self.label_item.setPos(gb.x1 + 3, gb.y1 + 2)

        self._update_appearance()

    def set_region_selected(self, selected: bool) -> None:
        self._is_selected = selected
        self._update_appearance()

    def _update_appearance(self, is_hover: bool = False) -> None:
        status = self.region.status

        if self._is_selected:
            # Active selected state: crisp white border + subtle white glow
            pen = QPen(QColor(255, 255, 255, 255), 2.0, Qt.SolidLine)
            brush = QBrush(QColor(255, 255, 255, 20))
            self.label_item.setBrush(QBrush(QColor(255, 255, 255)))

            glow = QGraphicsDropShadowEffect()
            glow.setColor(QColor(255, 255, 255, 90))
            glow.setBlurRadius(14)
            glow.setOffset(0, 0)
            self.setGraphicsEffect(glow)
            self.setZValue(100)
        else:
            self.setGraphicsEffect(None)
            self.setZValue(10)

            if status == RegionStatus.REVIEW:
                alpha = 230 if is_hover else 180
                pen = QPen(QColor(245, 158, 11, alpha), 1.5, Qt.SolidLine)
                brush = QBrush(QColor(245, 158, 11, 15 if not is_hover else 30))
                self.label_item.setBrush(QBrush(QColor(245, 158, 11)))
            elif status == RegionStatus.SKIP:
                alpha = 80 if is_hover else 35
                pen = QPen(QColor(255, 255, 255, alpha), 1.0, Qt.DotLine)
                brush = QBrush(QColor(0, 0, 0, 0))
                self.label_item.setBrush(QBrush(QColor(255, 255, 255, alpha)))
            else:
                alpha = 180 if is_hover else 100
                pen = QPen(QColor(255, 255, 255, alpha), 1.2, Qt.DashLine)
                brush = QBrush(QColor(255, 255, 255, 8 if not is_hover else 20))
                self.label_item.setBrush(QBrush(QColor(255, 255, 255, alpha)))

        self.setPen(pen)
        self.setBrush(brush)

    def hoverEnterEvent(self, event) -> None:
        if not self._is_selected:
            self._update_appearance(is_hover=True)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        if not self._is_selected:
            self._update_appearance(is_hover=False)
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self.on_clicked_callback(self.region.id)
            event.accept()
        else:
            super().mousePressEvent(event)


class SplitPageGraphicsItem(QGraphicsItem):
    """High-performance item rendering a single page with Before/After Split clip masks."""

    def __init__(
        self,
        original_pixmap: QPixmap,
        rendered_pixmap: QPixmap,
        page_rect: QRectF,
        parent: Optional[QGraphicsItem] = None,
    ) -> None:
        super().__init__(parent)
        self.original_pixmap = original_pixmap
        self.rendered_pixmap = rendered_pixmap
        self.page_rect = page_rect
        self.split_x: float = page_rect.width() * 0.5
        self.view_mode: str = "original"  # "original", "translated", "split", "peek_original"
        self.setZValue(1)

    def boundingRect(self) -> QRectF:
        return self.page_rect

    def set_split_x(self, x: float) -> None:
        self.split_x = max(0.0, min(float(x), self.page_rect.width()))
        self.update()

    def set_view_mode(self, mode: str) -> None:
        self.view_mode = mode
        self.update()

    def paint(self, painter: QPainter, option: Any, widget: Optional[QWidget] = None) -> None:
        w = self.page_rect.width()
        h = self.page_rect.height()
        has_orig = not self.original_pixmap.isNull()
        has_rendered = not self.rendered_pixmap.isNull()

        # If no pixmap is loaded yet, draw modern dark skeleton placeholder
        if not has_orig and not has_rendered:
            painter.save()
            painter.setBrush(QBrush(QColor(18, 18, 22)))
            painter.setPen(QPen(QColor(255, 255, 255, 20), 1.0, Qt.DashLine))
            painter.drawRect(self.page_rect)

            font = QFont("JetBrains Mono", 10, QFont.Bold)
            painter.setFont(font)
            painter.setPen(QColor(113, 113, 122))
            painter.drawText(self.page_rect, Qt.AlignCenter, f"PAGE • {int(w)} × {int(h)} px")
            painter.restore()
            return

        if self.view_mode in ("original", "peek_original") or not has_rendered:
            if has_orig:
                painter.drawPixmap(0, 0, self.original_pixmap)
            return

        if self.view_mode == "translated":
            if has_rendered:
                painter.drawPixmap(0, 0, self.rendered_pixmap)
            elif has_orig:
                painter.drawPixmap(0, 0, self.original_pixmap)
            return

        if self.view_mode == "split":
            split_pos = self.split_x

            # 1. Left partition: Original Pixmap clipped to [0, 0, split_pos, h]
            if split_pos > 0 and has_orig:
                painter.save()
                painter.setClipRect(QRectF(0, 0, split_pos, h))
                painter.drawPixmap(0, 0, self.original_pixmap)
                painter.restore()

            # 2. Right partition: Rendered Pixmap clipped to [split_pos, 0, w - split_pos, h]
            if split_pos < w:
                target_pix = self.rendered_pixmap if has_rendered else self.original_pixmap
                if not target_pix.isNull():
                    painter.save()
                    painter.setClipRect(QRectF(split_pos, 0, w - split_pos, h))
                    painter.drawPixmap(0, 0, target_pix)
                    painter.restore()

            # 3. Crisp Split Divider Line & Floating Handle
            painter.save()
            shadow_pen = QPen(QColor(0, 0, 0, 160), 3.0, Qt.SolidLine)
            painter.setPen(shadow_pen)
            painter.drawLine(QPointF(split_pos, 0), QPointF(split_pos, h))

            split_pen = QPen(QColor(56, 189, 248, 240), 2.0, Qt.SolidLine)  # Cyan blue accent
            painter.setPen(split_pen)
            painter.drawLine(QPointF(split_pos, 0), QPointF(split_pos, h))

            # Floating Handle circle at mid height
            handle_y = h * 0.5
            painter.setBrush(QBrush(QColor(15, 23, 42)))
            painter.setPen(QPen(QColor(56, 189, 248), 2.0))
            painter.drawEllipse(QPointF(split_pos, handle_y), 13, 13)

            painter.setPen(QColor(255, 255, 255))
            handle_font = QFont("Arial", 8, QFont.Bold)
            painter.setFont(handle_font)
            painter.drawText(QRectF(split_pos - 12, handle_y - 12, 24, 24), Qt.AlignCenter, "⬌")
            painter.restore()


class WebtoonCanvas(QWidget):
    """Main viewport showing vertically stacked webtoon pages with Before/After Split slider and Quick-Peek."""

    region_selected = Signal(int)       # region_id
    zoom_changed = Signal(float)        # zoom factor
    view_mode_changed = Signal(str)     # "original", "translated", or "split"

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._zoom_factor = 1.0
        self._view_mode = "original"  # "original", "translated", "split"
        self._prev_view_mode = "original"
        self._is_peeking = False
        self._split_x = 400.0  # Default 50% split on 800px width
        self._is_dragging_split = False

        self._pages: list[Page] = []
        self._original_pixmaps: list[QPixmap] = []
        self._rendered_pixmaps: list[QPixmap] = []
        self._page_items: list[SplitPageGraphicsItem] = []
        self._region_items: dict[int, RegionBoxGraphicsItem] = {}
        self._page_y_offsets: list[int] = []
        self._selected_region_id: Optional[int] = None

        self._build_ui()
        self.setFocusPolicy(Qt.StrongFocus)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Scene and View
        self.scene = QGraphicsScene(self)
        self.scene.setBackgroundBrush(QBrush(QColor(9, 9, 11)))

        self.view = QGraphicsView(self.scene, self)
        self.view.setRenderHint(QPainter.Antialiasing, True)
        self.view.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.view.setDragMode(QGraphicsView.ScrollHandDrag)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.view.setStyleSheet("background-color: #09090B; border: none;")

        # Enable mouse tracking for interactive split handle hovering
        self.view.setMouseTracking(True)
        self.view.viewport().installEventFilter(self)

        layout.addWidget(self.view)

        # Floating Bottom Controls (View Mode & Zoom)
        self.floating_bar = QFrame(self)
        self.floating_bar.setObjectName("badgeFrame")
        self.floating_bar.setStyleSheet(
            "background-color: #121215; border: 1px solid rgba(255, 255, 255, 0.12); "
            "border-radius: 6px; padding: 2px 6px;"
        )
        bar_layout = QHBoxLayout(self.floating_bar)
        bar_layout.setContentsMargins(6, 3, 6, 3)
        bar_layout.setSpacing(6)

        # 1. Mode Buttons
        self.btn_split_mode = QPushButton("↔ Split")
        self.btn_split_mode.setObjectName("ghostButton")
        self.btn_split_mode.setToolTip("Toggle Before/After Split Slider View")
        self.btn_split_mode.clicked.connect(self.toggle_split_view)
        bar_layout.addWidget(self.btn_split_mode)

        self.btn_peek = QPushButton("👁 Peek (Space)")
        self.btn_peek.setObjectName("ghostButton")
        self.btn_peek.setToolTip("Hold Space or Click to temporarily peek at original")
        self.btn_peek.pressed.connect(self.start_peek_original)
        self.btn_peek.released.connect(self.end_peek_original)
        bar_layout.addWidget(self.btn_peek)

        self.btn_mode_toggle = QPushButton("ORIGINAL [Tab]")
        self.btn_mode_toggle.setObjectName("ghostButton")
        self.btn_mode_toggle.setStyleSheet("font-family: 'JetBrains Mono', monospace; font-size: 10px; font-weight: 600;")
        self.btn_mode_toggle.setToolTip("Toggle Original vs Translated (Tab)")
        self.btn_mode_toggle.clicked.connect(lambda: self.toggle_view_mode())
        bar_layout.addWidget(self.btn_mode_toggle)

        div = QLabel("|")
        div.setStyleSheet("color: #3F3F46;")
        bar_layout.addWidget(div)

        # 2. Zoom Controls
        self.btn_zoom_out = QPushButton("-")
        self.btn_zoom_out.setFixedSize(22, 20)
        self.btn_zoom_out.setObjectName("ghostButton")
        self.btn_zoom_out.clicked.connect(self.zoom_out)

        self.zoom_label = QLabel("100%")
        self.zoom_label.setObjectName("monoLabel")
        self.zoom_label.setStyleSheet("font-size: 11px; color: #A1A1AA;")

        self.btn_zoom_in = QPushButton("+")
        self.btn_zoom_in.setFixedSize(22, 20)
        self.btn_zoom_in.setObjectName("ghostButton")
        self.btn_zoom_in.clicked.connect(self.zoom_in)

        self.btn_fit_width = QPushButton("Fit Width")
        self.btn_fit_width.setObjectName("ghostButton")
        self.btn_fit_width.clicked.connect(self.fit_width)

        bar_layout.addWidget(self.btn_zoom_out)
        bar_layout.addWidget(self.zoom_label)
        bar_layout.addWidget(self.btn_zoom_in)
        bar_layout.addWidget(self.btn_fit_width)

        self.floating_bar.raise_()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.floating_bar.move(
            self.width() - self.floating_bar.width() - 24,
            self.height() - self.floating_bar.height() - 24,
        )

    def load_chapter_pages(
        self,
        pages: Sequence[Page],
        regions: Optional[Sequence[Region]] = None,
        rendered_pages: Optional[Sequence[Path | str]] = None,
    ) -> None:
        self.scene.clear()
        self._region_items.clear()
        self._page_y_offsets.clear()
        self._page_items.clear()
        self._original_pixmaps.clear()
        self._rendered_pixmaps.clear()
        self._pages = list(pages)

        if not pages:
            return

        current_y = 0.0
        gap = 16.0  # Clean gap between vertical pages

        for idx, page in enumerate(pages):
            self._page_y_offsets.append(int(current_y))

            # Page Divider & Badge (for all pages)
            if idx > 0:
                divider_line = QGraphicsLineItem(0, current_y - (gap / 2), 800, current_y - (gap / 2))
                divider_pen = QPen(QColor(255, 255, 255, 30), 1.0, Qt.DashLine)
                divider_line.setPen(divider_pen)
                self.scene.addItem(divider_line)

            # Monospaced Page Badge at top-left
            badge_bg = QGraphicsRectItem(8, current_y + 8, 140, 20)
            badge_bg.setBrush(QBrush(QColor(18, 18, 21, 200)))
            badge_bg.setPen(QPen(QColor(255, 255, 255, 35), 1.0))
            badge_bg.setZValue(5)
            self.scene.addItem(badge_bg)

            badge_text = QGraphicsSimpleTextItem(f"PAGE #{page.index + 1:02d} • {page.width}x{page.height}", badge_bg)
            badge_text.setFont(QFont("JetBrains Mono", 8, QFont.Bold))
            badge_text.setBrush(QBrush(QColor(161, 161, 170)))
            badge_text.setPos(14, current_y + 11)

            # Skeleton Pixmaps (populated asynchronously via update_page_image)
            orig_pix = QPixmap()
            self._original_pixmaps.append(orig_pix)

            rend_pix = QPixmap()
            if rendered_pages and idx < len(rendered_pages):
                r_item = rendered_pages[idx]
                r_path = Path(r_item.path) if hasattr(r_item, "path") else Path(r_item)
                if r_path.exists():
                    rend_pix.load(str(r_path))
            self._rendered_pixmaps.append(rend_pix)

            # Add Split Page Item with skeleton dimensions to Scene
            page_rect = QRectF(0, 0, page.width, page.height)
            page_item = SplitPageGraphicsItem(orig_pix, rend_pix, page_rect)
            page_item.setPos(0, current_y)
            page_item.set_split_x(self._split_x)
            page_item.set_view_mode(self._view_mode)
            self.scene.addItem(page_item)
            self._page_items.append(page_item)

            current_y += page.height + gap

        # Add region bounding boxes
        if regions:
            for region in regions:
                item = RegionBoxGraphicsItem(region, on_clicked_callback=self._on_region_item_clicked)
                self.scene.addItem(item)
                self._region_items[region.id] = item

        self.scene.setSceneRect(0, 0, 800, current_y)
        self.fit_width()

    def update_page_image(self, page_index: int, image_or_pixmap: Any) -> None:
        """Asenkron olarak çözülen sayfa görüntüsünü kanvasa yükler."""
        if 0 <= page_index < len(self._page_items):
            pix = image_or_pixmap if isinstance(image_or_pixmap, QPixmap) else QPixmap.fromImage(image_or_pixmap)
            self._original_pixmaps[page_index] = pix
            self._page_items[page_index].original_pixmap = pix
            if self._rendered_pixmaps[page_index].isNull():
                self._rendered_pixmaps[page_index] = pix
                self._page_items[page_index].rendered_pixmap = pix
            self._page_items[page_index].update()

    def set_rendered_pages(self, rendered_paths: Sequence[Any]) -> None:
        self._rendered_pixmaps.clear()
        for idx, r_item in enumerate(rendered_paths):
            p = Path(r_item.path) if hasattr(r_item, "path") else Path(r_item)
            pix = QPixmap(str(p)) if p.exists() else (self._original_pixmaps[idx] if idx < len(self._original_pixmaps) else QPixmap())
            self._rendered_pixmaps.append(pix)
            if idx < len(self._page_items):
                self._page_items[idx].rendered_pixmap = pix
                self._page_items[idx].update()

    def toggle_split_view(self) -> None:
        if self._view_mode == "split":
            self.toggle_view_mode(self._prev_view_mode if self._prev_view_mode != "split" else "translated")
        else:
            self._prev_view_mode = self._view_mode
            self._set_mode("split")

    def toggle_view_mode(self, mode: Optional[str] = None) -> None:
        if mode in ("original", "translated", "split"):
            new_mode = mode
        else:
            new_mode = "translated" if self._view_mode == "original" else "original"
        self._set_mode(new_mode)

    def _set_mode(self, mode: str) -> None:
        self._view_mode = mode
        for item in self._page_items:
            item.set_view_mode(mode)

        if mode == "split":
            self.btn_split_mode.setStyleSheet("color: #38BDF8; font-weight: 700; border: 1px solid #38BDF8;")
            self.btn_mode_toggle.setText("SPLIT [↔]")
            self.btn_mode_toggle.setStyleSheet("font-family: 'JetBrains Mono', monospace; font-size: 10px; font-weight: 700; color: #38BDF8;")
        else:
            self.btn_split_mode.setStyleSheet("")
            self.btn_mode_toggle.setText(f"{mode.upper()} [Tab]")
            if mode == "translated":
                self.btn_mode_toggle.setStyleSheet("font-family: 'JetBrains Mono', monospace; font-size: 10px; font-weight: 700; color: #10B981;")
            else:
                self.btn_mode_toggle.setStyleSheet("font-family: 'JetBrains Mono', monospace; font-size: 10px; font-weight: 600; color: #A1A1AA;")

        self.view_mode_changed.emit(self._view_mode)
        self.scene.update()

    def start_peek_original(self) -> None:
        """Hold-to-Peek: temporarily display original image."""
        if not self._is_peeking:
            self._is_peeking = True
            for item in self._page_items:
                item.set_view_mode("peek_original")
            self.btn_peek.setStyleSheet("color: #F59E0B; font-weight: 700; border: 1px solid #F59E0B;")
            self.scene.update()

    def end_peek_original(self) -> None:
        """Release Hold-to-Peek: return to active mode."""
        if self._is_peeking:
            self._is_peeking = False
            for item in self._page_items:
                item.set_view_mode(self._view_mode)
            self.btn_peek.setStyleSheet("")
            self.scene.update()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        focus_w = QApplication.focusWidget()
        if isinstance(focus_w, (QLineEdit, QTextEdit, QPlainTextEdit)):
            super().keyPressEvent(event)
            return

        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self.start_peek_original()
            event.accept()
            return
        elif event.key() == Qt.Key_Tab:
            self.toggle_view_mode()
            event.accept()
            return

        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self.end_peek_original()
            event.accept()
            return

        super().keyReleaseEvent(event)

    def eventFilter(self, watched, event) -> bool:
        if watched == self.view.viewport() and self._view_mode == "split":
            if event.type() == QMouseEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                scene_pos = self.view.mapToScene(event.pos())
                if abs(scene_pos.x() - self._split_x) < 30:
                    self._is_dragging_split = True
                    self.view.setDragMode(QGraphicsView.NoDrag)
                    return True
            elif event.type() == QMouseEvent.MouseMove:
                scene_pos = self.view.mapToScene(event.pos())
                if self._is_dragging_split:
                    max_w = max((p.width for p in self._pages), default=800)
                    self._split_x = max(10.0, min(float(scene_pos.x()), max_w - 10.0))
                    for item in self._page_items:
                        item.set_split_x(self._split_x)
                    return True
                elif abs(scene_pos.x() - self._split_x) < 24:
                    self.view.viewport().setCursor(Qt.SizeHorCursor)
                else:
                    if self.view.dragMode() == QGraphicsView.ScrollHandDrag:
                        self.view.viewport().setCursor(Qt.OpenHandCursor)
            elif event.type() == QMouseEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
                if self._is_dragging_split:
                    self._is_dragging_split = False
                    self.view.setDragMode(QGraphicsView.ScrollHandDrag)
                    return True

        return super().eventFilter(watched, event)

    def auto_scroll_to_region(self, region_id: int) -> None:
        """Smoothly centers the selected bounding box in the viewport."""
        if region_id in self._region_items:
            item = self._region_items[region_id]
            center = item.rect().center()
            self.view.centerOn(center)

    def select_region(self, region_id: int, auto_scroll: bool = True) -> None:
        if self._selected_region_id in self._region_items:
            self._region_items[self._selected_region_id].set_region_selected(False)

        self._selected_region_id = region_id

        if region_id in self._region_items:
            item = self._region_items[region_id]
            item.set_region_selected(True)
            if auto_scroll:
                self.auto_scroll_to_region(region_id)

    def scroll_to_page(self, page_index: int) -> None:
        if 0 <= page_index < len(self._page_y_offsets):
            y_pos = self._page_y_offsets[page_index]
            self.view.centerOn(400, y_pos + 300)

    def zoom_in(self) -> None:
        self._set_zoom(self._zoom_factor * 1.2)

    def zoom_out(self) -> None:
        self._set_zoom(self._zoom_factor / 1.2)

    def fit_width(self) -> None:
        viewport_width = self.view.viewport().width()
        if viewport_width > 50:
            factor = (viewport_width - 32) / 800.0
            self._set_zoom(factor)

    def _set_zoom(self, factor: float) -> None:
        factor = max(0.2, min(factor, 4.0))
        scale_change = factor / self._zoom_factor
        self._zoom_factor = factor
        self.view.scale(scale_change, scale_change)
        self.zoom_label.setText(f"{int(self._zoom_factor * 100)}%")
        self.zoom_changed.emit(self._zoom_factor)

    def wheelEvent(self, event: QWheelEvent) -> None:
        if event.modifiers() == Qt.ControlModifier:
            angle = event.angleDelta().y()
            if angle > 0:
                self.zoom_in()
            else:
                self.zoom_out()
            event.accept()
        else:
            super().wheelEvent(event)

    def _on_region_item_clicked(self, region_id: int) -> None:
        self.select_region(region_id, auto_scroll=False)
        self.region_selected.emit(region_id)
