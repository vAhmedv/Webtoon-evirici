"""TopBar component featuring a 2-line granular Pipeline Stepper and slim progress bar."""

from __future__ import annotations

import os
from typing import Optional
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QGraphicsDropShadowEffect,
)
from PySide6.QtGui import QColor

try:
    import psutil
except ImportError:
    psutil = None

try:
    import torch
except ImportError:
    torch = None


class PipelineStepperStep(QFrame):
    """Linear/Raycast inspired 2-Line Stepper node with real-time granular progress."""

    def __init__(
        self,
        step_number: int,
        step_name: str,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.step_number = step_number
        self.step_name = step_name
        self._state = "idle"  # "idle", "running", "completed", "failed"
        self._current = 0
        self._total = 0

        self.setObjectName("stepperBadge")
        self.setMinimumWidth(110)
        self._build_ui()
        self._apply_state_style()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(2)

        # Row 1: Step Number & Title
        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.setSpacing(4)

        self.icon_label = QLabel(f"{self.step_number}.")
        self.icon_label.setStyleSheet("font-size: 10px; font-family: 'JetBrains Mono', monospace; font-weight: 700;")

        self.title_label = QLabel(self.step_name.upper())
        self.title_label.setStyleSheet("font-size: 10px; font-weight: 700; letter-spacing: 0.5px;")

        title_row.addWidget(self.icon_label)
        title_row.addWidget(self.title_label)
        title_row.addStretch(1)
        layout.addLayout(title_row)

        # Row 2: Granular Progress Badge
        self.progress_label = QLabel("Bekliyor")
        self.progress_label.setStyleSheet(
            "font-size: 9px; font-family: 'JetBrains Mono', monospace; color: #52525B;"
        )
        layout.addWidget(self.progress_label)

    def set_state(
        self,
        state: str,
        current: int = 0,
        total: int = 0,
        custom_text: str = "",
    ) -> None:
        self._state = state
        self._current = current
        self._total = total
        self._apply_state_style(custom_text)

    def _apply_state_style(self, custom_text: str = "") -> None:
        if self._state == "running":
            self.setProperty("state", "running")
            self.icon_label.setText(f"{self.step_number}.")
            self.icon_label.setStyleSheet(
                "font-size: 10px; font-family: 'JetBrains Mono', monospace; font-weight: 700; color: #FAFAFA;"
            )
            self.title_label.setStyleSheet(
                "font-size: 10px; font-weight: 700; color: #FAFAFA; letter-spacing: 0.5px;"
            )

            # Build detailed progress text
            if custom_text:
                txt = custom_text
            elif self._total > 0:
                pct = int((self._current / self._total) * 100)
                remaining = max(0, self._total - self._current)
                txt = f"{self._current} / {self._total} (%{pct}) • Kalan: {remaining}"
            elif self._current > 0:
                txt = f"İşleniyor: {self._current}"
            else:
                txt = "İşleniyor..."

            self.progress_label.setText(txt)
            self.progress_label.setStyleSheet(
                "font-size: 9px; font-family: 'JetBrains Mono', monospace; color: #E4E4E7; font-weight: 600;"
            )
            self.setStyleSheet(
                "background-color: #09090B; border: 1px solid #FAFAFA; border-radius: 4px;"
            )

            # Subtle white glow
            glow = QGraphicsDropShadowEffect(self)
            glow.setColor(QColor(255, 255, 255, 60))
            glow.setBlurRadius(10)
            glow.setOffset(0, 0)
            self.setGraphicsEffect(glow)

        elif self._state == "completed":
            self.setProperty("state", "completed")
            self.icon_label.setText("✓")
            self.icon_label.setStyleSheet("font-size: 10px; font-weight: 700; color: #10B981;")
            self.title_label.setStyleSheet(
                "font-size: 10px; font-weight: 600; color: #A1A1AA; letter-spacing: 0.5px;"
            )

            if self._total > 0:
                txt = f"✓ {self._total} / {self._total}"
            else:
                txt = "✓ Tamamlandı"

            self.progress_label.setText(txt)
            self.progress_label.setStyleSheet(
                "font-size: 9px; font-family: 'JetBrains Mono', monospace; color: #71717A;"
            )
            self.setStyleSheet(
                "background-color: #18181B; border: 1px solid rgba(255, 255, 255, 0.12); border-radius: 4px;"
            )
            self.setGraphicsEffect(None)

        elif self._state == "failed":
            self.setProperty("state", "failed")
            self.icon_label.setText("✕")
            self.icon_label.setStyleSheet("font-size: 10px; font-weight: 700; color: #EF4444;")
            self.title_label.setStyleSheet(
                "font-size: 10px; font-weight: 600; color: #EF4444; letter-spacing: 0.5px;"
            )
            self.progress_label.setText(custom_text or "✕ Hata")
            self.progress_label.setStyleSheet(
                "font-size: 9px; font-family: 'JetBrains Mono', monospace; color: #EF4444;"
            )
            self.setStyleSheet(
                "background-color: #1C1215; border: 1px solid rgba(239, 68, 68, 0.4); border-radius: 4px;"
            )
            self.setGraphicsEffect(None)

        else:  # idle
            self.setProperty("state", "idle")
            self.icon_label.setText(f"{self.step_number}.")
            self.icon_label.setStyleSheet(
                "font-size: 10px; font-family: 'JetBrains Mono', monospace; font-weight: 600; color: #52525B;"
            )
            self.title_label.setStyleSheet(
                "font-size: 10px; font-weight: 600; color: #52525B; letter-spacing: 0.5px;"
            )
            self.progress_label.setText("Bekliyor")
            self.progress_label.setStyleSheet(
                "font-size: 9px; font-family: 'JetBrains Mono', monospace; color: #52525B;"
            )
            self.setStyleSheet(
                "background-color: #121215; border: 1px solid #27272A; border-radius: 4px;"
            )
            self.setGraphicsEffect(None)


class TopBar(QFrame):
    """Linear/Raycast style TopBar (56px) with 2-line Stepper and slim progress bar."""

    open_chapter_clicked = Signal()
    run_pipeline_clicked = Signal()
    cancel_pipeline_clicked = Signal()

    STAGES = [
        (1, "DETECT"),
        (2, "OCR"),
        (3, "REPAIR"),
        (4, "TRANSLATE"),
        (5, "INPAINT"),
        (6, "RENDER"),
    ]

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("topBar")
        self._build_ui()
        self._init_memory_timer()

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Top Content Row
        content_row = QHBoxLayout()
        content_row.setContentsMargins(12, 4, 12, 4)
        content_row.setSpacing(10)

        # 1. Left: Title & Chapter badge
        left_box = QHBoxLayout()
        left_box.setSpacing(8)

        self.title_label = QLabel("WEBTOON TRANSLATOR")
        self.title_label.setObjectName("titleLabel")

        self.chapter_badge = QLabel("NO CHAPTER")
        self.chapter_badge.setObjectName("monoLabel")
        self.chapter_badge.setStyleSheet(
            "background-color: #18181B; color: #A1A1AA; border: 1px solid rgba(255, 255, 255, 0.08); "
            "border-radius: 4px; padding: 2px 8px; font-size: 11px;"
        )

        self.open_btn = QPushButton("Open Chapter")
        self.open_btn.setObjectName("ghostButton")
        self.open_btn.setCursor(Qt.PointingHandCursor)
        self.open_btn.clicked.connect(self.open_chapter_clicked.emit)

        left_box.addWidget(self.title_label)
        left_box.addWidget(self.chapter_badge)
        left_box.addWidget(self.open_btn)
        content_row.addLayout(left_box)

        content_row.addStretch(1)

        # 2. Center: 6-step Connected Stepper (2-line cards)
        stepper_box = QHBoxLayout()
        stepper_box.setSpacing(4)
        self.step_widgets: dict[str, PipelineStepperStep] = {}

        for num, name in self.STAGES:
            step = PipelineStepperStep(num, name, self)
            self.step_widgets[name] = step
            stepper_box.addWidget(step)
            if num < 6:
                arrow = QLabel("→")
                arrow.setStyleSheet("color: #3F3F46; font-size: 11px; font-weight: bold;")
                stepper_box.addWidget(arrow)

        content_row.addLayout(stepper_box)

        content_row.addStretch(1)

        # 3. Right: Memory Monitor & Actions
        right_box = QHBoxLayout()
        right_box.setSpacing(10)

        self.memory_label = QLabel("RAM: -- | VRAM: --")
        self.memory_label.setObjectName("monoLabel")
        self.memory_label.setStyleSheet("color: #71717A; font-size: 11px;")

        self.cancel_btn = QPushButton("Stop")
        self.cancel_btn.setObjectName("dangerButton")
        self.cancel_btn.setCursor(Qt.PointingHandCursor)
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_pipeline_clicked.emit)

        self.run_btn = QPushButton("Run Pipeline")
        self.run_btn.setObjectName("primaryButton")
        self.run_btn.setCursor(Qt.PointingHandCursor)
        self.run_btn.clicked.connect(self.run_pipeline_clicked.emit)

        glow = QGraphicsDropShadowEffect(self)
        glow.setColor(QColor(255, 255, 255, 40))
        glow.setBlurRadius(12)
        glow.setOffset(0, 0)
        self.run_btn.setGraphicsEffect(glow)

        right_box.addWidget(self.memory_label)
        right_box.addWidget(self.cancel_btn)
        right_box.addWidget(self.run_btn)
        content_row.addLayout(right_box)

        main_layout.addLayout(content_row)

        # 4. Bottom 2px Slim Total Progress Bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setObjectName("pipelineProgressBar")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(2)
        main_layout.addWidget(self.progress_bar)

    def set_chapter_info(self, chapter_name: str, page_count: int) -> None:
        self.chapter_badge.setText(f"{chapter_name.upper()} • {page_count} PAGES")
        self.chapter_badge.setStyleSheet(
            "background-color: #1C1C20; color: #FAFAFA; border: 1px solid rgba(255, 255, 255, 0.15); "
            "border-radius: 4px; padding: 2px 8px; font-size: 11px;"
        )

    def update_step_progress(
        self,
        stage_name: str,
        current: int = 0,
        total: int = 0,
        status: str = "running",
        custom_text: str = "",
    ) -> None:
        """Updates granular progress for a step and sets previous steps to completed."""
        key = stage_name.upper()
        stages_order = ["DETECT", "OCR", "REPAIR", "TRANSLATE", "INPAINT", "RENDER"]

        if key in stages_order:
            active_idx = stages_order.index(key)
            for idx, s in enumerate(stages_order):
                if idx < active_idx:
                    if self.step_widgets[s]._state != "completed":
                        self.step_widgets[s].set_state("completed")
                elif idx == active_idx:
                    self.step_widgets[s].set_state(status, current=current, total=total, custom_text=custom_text)
                else:
                    self.step_widgets[s].set_state("idle")

            # Calculate global progress percentage
            step_weight = 100.0 / len(stages_order)
            in_step_pct = (current / total) if total > 0 else 0.0
            global_pct = int((active_idx * step_weight) + (in_step_pct * step_weight))
            self.set_total_progress(global_pct)

    def set_stage_state(
        self,
        stage_name: str,
        state: str,
        current: int = 0,
        total: int = 0,
        progress_text: str = "",
    ) -> None:
        key = stage_name.upper()
        if key in self.step_widgets:
            self.step_widgets[key].set_state(state, current=current, total=total, custom_text=progress_text)

    def set_total_progress(self, percentage: int) -> None:
        self.progress_bar.setValue(max(0, min(percentage, 100)))

    def reset_stages(self) -> None:
        for step in self.step_widgets.values():
            step.set_state("idle")
        self.progress_bar.setValue(0)

    def set_pipeline_running(self, running: bool) -> None:
        self.run_btn.setEnabled(not running)
        self.cancel_btn.setEnabled(running)
        self.open_btn.setEnabled(not running)

    def _init_memory_timer(self) -> None:
        self._mem_timer = QTimer(self)
        self._mem_timer.timeout.connect(self._update_memory_info)
        self._mem_timer.start(3000)
        self._update_memory_info()

    def _update_memory_info(self) -> None:
        ram_str = "--"
        vram_str = "--"
        if psutil is not None:
            mem = psutil.virtual_memory()
            ram_used_gb = (mem.total - mem.available) / (1024 ** 3)
            ram_str = f"RAM: {ram_used_gb:.1f}GB"

        if torch is not None and torch.cuda.is_available():
            try:
                vram_used = torch.cuda.memory_allocated(0) / (1024 ** 3)
                vram_str = f"VRAM: {vram_used:.1f}GB"
            except Exception:
                pass

        self.memory_label.setText(f"{ram_str} | {vram_str}")
