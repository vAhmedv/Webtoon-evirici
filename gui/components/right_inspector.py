"""RightInspector component: Linear-style region details, OCR text, Turkish translation editor and quick review actions."""

from __future__ import annotations

from typing import Optional, Any
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFrame,
    QGraphicsDropShadowEffect,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QApplication,
)

from core.detection.detection import Region, RegionStatus, RegionType


class RightInspector(QFrame):
    """Right sidebar (380px) for inspecting and editing translation/status of selected region."""

    translation_updated = Signal(int, str)      # (region_id, new_text)
    status_changed = Signal(int, RegionStatus)  # (region_id, new_status)
    navigate_requested = Signal(int)           # -1 or +1
    confirm_requested = Signal(int)            # region_id
    skip_requested = Signal(int)               # region_id
    view_mode_toggled = Signal(str)            # "original" or "translated"

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("rightInspector")
        self.setFixedWidth(380)

        self._current_region: Optional[Region] = None
        self._current_view_mode: str = "original"
        self._build_ui()
        self._setup_shortcuts()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # 1. Header & Navigation Row
        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        
        self.title_label = QLabel("INSPECTOR")
        self.title_label.setStyleSheet("font-size: 11px; font-weight: 600; color: #71717A; letter-spacing: 0.5px;")

        self.nav_counter = QLabel("0 / 0")
        self.nav_counter.setObjectName("monoLabel")
        self.nav_counter.setStyleSheet("color: #71717A; font-size: 11px;")

        self.btn_prev = QPushButton("←")
        self.btn_prev.setFixedSize(26, 22)
        self.btn_prev.setObjectName("ghostButton")
        self.btn_prev.setToolTip("Previous Region (Left Arrow)")
        self.btn_prev.clicked.connect(lambda: self.navigate_requested.emit(-1))

        self.btn_next = QPushButton("→")
        self.btn_next.setFixedSize(26, 22)
        self.btn_next.setObjectName("ghostButton")
        self.btn_next.setToolTip("Next Region (Right Arrow)")
        self.btn_next.clicked.connect(lambda: self.navigate_requested.emit(1))

        header_row.addWidget(self.title_label)
        header_row.addStretch(1)
        header_row.addWidget(self.nav_counter)
        header_row.addWidget(self.btn_prev)
        header_row.addWidget(self.btn_next)
        layout.addLayout(header_row)

        # 2. View Mode Toggle Switch (Original / Translated)
        toggle_frame = QFrame(self)
        toggle_frame.setObjectName("toggleGroup")
        t_layout = QHBoxLayout(toggle_frame)
        t_layout.setContentsMargins(2, 2, 2, 2)
        t_layout.setSpacing(2)

        self.btn_view_orig = QPushButton("Original")
        self.btn_view_orig.setObjectName("toggleSegment")
        self.btn_view_orig.setCheckable(True)
        self.btn_view_orig.setChecked(True)
        self.btn_view_orig.clicked.connect(lambda: self._set_view_mode("original"))

        self.btn_view_trans = QPushButton("Translated (Tab)")
        self.btn_view_trans.setObjectName("toggleSegment")
        self.btn_view_trans.setCheckable(True)
        self.btn_view_trans.clicked.connect(lambda: self._set_view_mode("translated"))

        self.view_btn_group = QButtonGroup(self)
        self.view_btn_group.addButton(self.btn_view_orig)
        self.view_btn_group.addButton(self.btn_view_trans)
        self.view_btn_group.setExclusive(True)

        t_layout.addWidget(self.btn_view_orig)
        t_layout.addWidget(self.btn_view_trans)
        layout.addWidget(toggle_frame)

        # 3. Metadata Card
        self.meta_card = QFrame(self)
        self.meta_card.setObjectName("cardFrame")
        meta_layout = QGridLayout(self.meta_card)
        meta_layout.setContentsMargins(10, 10, 10, 10)
        meta_layout.setSpacing(8)

        # Row 0: ID & Status Dropdown
        self.lbl_id = QLabel("#---")
        self.lbl_id.setObjectName("monoLabel")
        self.lbl_id.setStyleSheet("font-size: 13px; font-weight: 700; color: #FAFAFA;")

        self.combo_status = QComboBox()
        self.combo_status.addItems(["AUTO", "REVIEW", "SKIP"])
        self.combo_status.currentIndexChanged.connect(self._on_status_dropdown_changed)

        meta_layout.addWidget(QLabel("Region:"), 0, 0)
        meta_layout.addWidget(self.lbl_id, 0, 1)
        meta_layout.addWidget(QLabel("Status:"), 0, 2)
        meta_layout.addWidget(self.combo_status, 0, 3)

        # Row 1: Type & Confidence
        self.lbl_type = QLabel("---")
        self.lbl_type.setStyleSheet(
            "background-color: #18181B; color: #A1A1AA; border: 1px solid rgba(255, 255, 255, 0.08); "
            "border-radius: 4px; padding: 2px 6px; font-size: 10px; font-family: 'JetBrains Mono', monospace;"
        )

        self.conf_bar = QProgressBar()
        self.conf_bar.setRange(0, 100)
        self.conf_bar.setValue(0)
        self.conf_bar.setFixedHeight(8)

        meta_layout.addWidget(QLabel("Type:"), 1, 0)
        meta_layout.addWidget(self.lbl_type, 1, 1)
        meta_layout.addWidget(QLabel("Confidence:"), 1, 2)
        meta_layout.addWidget(self.conf_bar, 1, 3)

        # Row 2: BBox Coordinates
        self.lbl_bbox = QLabel("[-, -, -, -]")
        self.lbl_bbox.setObjectName("monoLabel")
        self.lbl_bbox.setStyleSheet("color: #71717A; font-size: 10px;")
        meta_layout.addWidget(QLabel("BBox:"), 2, 0)
        meta_layout.addWidget(self.lbl_bbox, 2, 1, 1, 3)

        layout.addWidget(self.meta_card)

        # Review Reason Banner (Conditional)
        self.reason_banner = QFrame(self)
        self.reason_banner.setStyleSheet(
            "background-color: #1C1710; border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 6px; padding: 6px 10px;"
        )
        r_layout = QHBoxLayout(self.reason_banner)
        r_layout.setContentsMargins(0, 0, 0, 0)
        r_layout.setSpacing(6)
        
        self.reason_icon = QLabel("⚠️")
        self.reason_icon.setStyleSheet("font-size: 11px;")
        self.reason_text = QLabel("Review required")
        self.reason_text.setStyleSheet("color: #F59E0B; font-size: 11px; font-weight: 500;")
        
        r_layout.addWidget(self.reason_icon)
        r_layout.addWidget(self.reason_text)
        r_layout.addStretch(1)
        self.reason_banner.hide()
        layout.addWidget(self.reason_banner)

        # 4. Source OCR Text Card
        self.src_card = QFrame(self)
        self.src_card.setObjectName("cardFrame")
        src_layout = QVBoxLayout(self.src_card)
        src_layout.setContentsMargins(10, 10, 10, 10)
        src_layout.setSpacing(6)

        src_header = QHBoxLayout()
        src_header.addWidget(QLabel("ORIGINAL OCR TEXT"))
        self.btn_copy_src = QPushButton("Copy")
        self.btn_copy_src.setObjectName("ghostButton")
        self.btn_copy_src.setFixedSize(45, 20)
        self.btn_copy_src.setStyleSheet("font-size: 10px;")
        self.btn_copy_src.clicked.connect(self._copy_source_text)
        src_header.addStretch(1)
        src_header.addWidget(self.btn_copy_src)
        src_layout.addLayout(src_header)

        self.src_text = QTextEdit()
        self.src_text.setReadOnly(True)
        self.src_text.setFixedHeight(80)
        self.src_text.setStyleSheet("font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #E4E4E7;")
        src_layout.addWidget(self.src_text)

        layout.addWidget(self.src_card)

        # 5. Turkish Translation Editor Card
        self.tr_card = QFrame(self)
        self.tr_card.setObjectName("cardFrame")
        tr_layout = QVBoxLayout(self.tr_card)
        tr_layout.setContentsMargins(10, 10, 10, 10)
        tr_layout.setSpacing(6)

        tr_header = QHBoxLayout()
        tr_title = QLabel("TURKISH TRANSLATION")
        tr_title.setStyleSheet("font-size: 11px; font-weight: 600; color: #FAFAFA;")
        tr_header.addWidget(tr_title)
        tr_header.addStretch(1)
        tr_layout.addLayout(tr_header)

        self.tr_text = QTextEdit()
        self.tr_text.setPlaceholderText("Enter Turkish translation...")
        self.tr_text.setStyleSheet("font-size: 12px; line-height: 1.4;")
        self.tr_text.textChanged.connect(self._on_translation_edited)
        tr_layout.addWidget(self.tr_text)

        layout.addWidget(self.tr_card, 1)

        # 6. Quick Actions Bar (Bottom)
        actions_bar = QHBoxLayout()
        actions_bar.setSpacing(8)

        self.btn_skip = QPushButton("Skip (Esc)")
        self.btn_skip.setObjectName("dangerButton")
        self.btn_skip.setCursor(Qt.PointingHandCursor)
        self.btn_skip.clicked.connect(self._on_skip_clicked)

        self.btn_confirm = QPushButton("Confirm (Ctrl+↵)")
        self.btn_confirm.setObjectName("primaryButton")
        self.btn_confirm.setCursor(Qt.PointingHandCursor)
        self.btn_confirm.clicked.connect(self._on_confirm_clicked)

        glow = QGraphicsDropShadowEffect(self)
        glow.setColor(QColor(255, 255, 255, 40))
        glow.setBlurRadius(10)
        glow.setOffset(0, 0)
        self.btn_confirm.setGraphicsEffect(glow)

        actions_bar.addWidget(self.btn_skip, 1)
        actions_bar.addWidget(self.btn_confirm, 2)
        layout.addLayout(actions_bar)

    def _setup_shortcuts(self) -> None:
        shortcut_confirm = QShortcut(QKeySequence("Ctrl+Return"), self)
        shortcut_confirm.activated.connect(self._on_confirm_clicked)

        shortcut_confirm2 = QShortcut(QKeySequence("Ctrl+Enter"), self)
        shortcut_confirm2.activated.connect(self._on_confirm_clicked)

        shortcut_skip = QShortcut(QKeySequence("Esc"), self)
        shortcut_skip.activated.connect(self._on_skip_clicked)

        shortcut_prev = QShortcut(QKeySequence("Left"), self)
        shortcut_prev.activated.connect(lambda: self.navigate_requested.emit(-1))

        shortcut_next = QShortcut(QKeySequence("Right"), self)
        shortcut_next.activated.connect(lambda: self.navigate_requested.emit(1))

        shortcut_tab = QShortcut(QKeySequence("Tab"), self)
        shortcut_tab.activated.connect(self.toggle_view_mode)

    def toggle_view_mode(self) -> None:
        new_mode = "translated" if self._current_view_mode == "original" else "original"
        self._set_view_mode(new_mode)

    def _set_view_mode(self, mode: str) -> None:
        self._current_view_mode = mode
        if mode == "translated":
            self.btn_view_trans.setChecked(True)
        else:
            self.btn_view_orig.setChecked(True)
        self.view_mode_toggled.emit(mode)

    def sync_view_mode(self, mode: str) -> None:
        self._current_view_mode = mode
        if mode == "translated":
            self.btn_view_trans.setChecked(True)
        else:
            self.btn_view_orig.setChecked(True)

    def display_region(
        self,
        region: Optional[Region],
        current_index: int = 0,
        total_count: int = 0,
    ) -> None:
        self._current_region = region
        self.nav_counter.setText(f"{current_index + 1 if total_count > 0 else 0} / {total_count}")

        if not region:
            self.lbl_id.setText("#---")
            self.lbl_type.setText("---")
            self.lbl_bbox.setText("[-, -, -, -]")
            self.conf_bar.setValue(0)
            self.src_text.setPlainText("")
            self.tr_text.setPlainText("")
            self.reason_banner.hide()
            return

        self.lbl_id.setText(f"#{region.id}")
        self.lbl_type.setText(region.type.value.upper())
        
        gb = region.global_bbox
        self.lbl_bbox.setText(f"[{gb.x1}, {gb.y1}, {gb.x2}, {gb.y2}] • {gb.width}x{gb.height} px")

        conf_pct = int((region.ocr_confidence or 0.0) * 100)
        self.conf_bar.setValue(conf_pct)

        # Status dropdown index
        status_map = {RegionStatus.AUTO: 0, RegionStatus.REVIEW: 1, RegionStatus.SKIP: 2}
        self.combo_status.blockSignals(True)
        self.combo_status.setCurrentIndex(status_map.get(region.status, 0))
        self.combo_status.blockSignals(False)

        # Review Reason
        if region.status == RegionStatus.REVIEW and region.review_reason:
            self.reason_text.setText(f"Review: {region.review_reason}")
            self.reason_banner.show()
        else:
            self.reason_banner.hide()

        self.src_text.setPlainText(region.text or "")

        self.tr_text.blockSignals(True)
        self.tr_text.setPlainText(region.translation or "")
        self.tr_text.blockSignals(False)

    def _on_status_dropdown_changed(self, index: int) -> None:
        if not self._current_region:
            return
        status_list = [RegionStatus.AUTO, RegionStatus.REVIEW, RegionStatus.SKIP]
        if 0 <= index < len(status_list):
            new_status = status_list[index]
            self.status_changed.emit(self._current_region.id, new_status)

    def _on_translation_edited(self) -> None:
        if self._current_region:
            text = self.tr_text.toPlainText()
            self.translation_updated.emit(self._current_region.id, text)

    def _copy_source_text(self) -> None:
        if self._current_region and self._current_region.text:
            clipboard = QApplication.clipboard()
            if clipboard:
                clipboard.setText(self._current_region.text)

    def _on_confirm_clicked(self) -> None:
        if self._current_region:
            self.confirm_requested.emit(self._current_region.id)

    def _on_skip_clicked(self) -> None:
        if self._current_region:
            self.skip_requested.emit(self._current_region.id)
