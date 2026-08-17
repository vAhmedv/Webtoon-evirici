"""TelemetryStatusBar component: modern bottom status bar displaying live GPU VRAM, system RAM, and active AI engine badges."""

from __future__ import annotations

import os
from typing import Optional, Any
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QWidget,
)


class SystemTelemetry:
    """Lightweight non-blocking telemetry collector for GPU VRAM and System RAM."""

    @staticmethod
    def get_gpu_telemetry() -> dict[str, Any]:
        """Collects CUDA VRAM statistics if torch and CUDA are available."""
        try:
            import torch
            if not torch.cuda.is_available():
                return {"has_gpu": False}

            device_idx = 0
            device_name = torch.cuda.get_device_name(device_idx)
            total_vram = torch.cuda.get_device_properties(device_idx).total_memory
            reserved_vram = torch.cuda.memory_reserved(device_idx)
            allocated_vram = torch.cuda.memory_allocated(device_idx)

            return {
                "has_gpu": True,
                "device_name": device_name,
                "total_gb": total_vram / (1024 ** 3),
                "reserved_gb": reserved_vram / (1024 ** 3),
                "allocated_gb": allocated_vram / (1024 ** 3),
                "usage_percent": (reserved_vram / total_vram) * 100 if total_vram > 0 else 0,
            }
        except Exception:
            return {"has_gpu": False}

    @staticmethod
    def get_ram_telemetry() -> dict[str, Any]:
        """Collects System RAM statistics via psutil or fallback."""
        try:
            import psutil
            vm = psutil.virtual_memory()
            return {
                "total_gb": vm.total / (1024 ** 3),
                "used_gb": vm.used / (1024 ** 3),
                "percent": vm.percent,
            }
        except Exception:
            return {
                "total_gb": 0.0,
                "used_gb": 0.0,
                "percent": 0.0,
            }


class EngineBadge(QFrame):
    """Monochrome badge for an individual AI engine."""

    def __init__(self, name: str, tooltip: str = "", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.name = name
        self.setToolTip(tooltip)
        self._is_active = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 2, 6, 2)
        layout.setSpacing(4)

        self.dot = QLabel("●")
        self.dot.setStyleSheet("font-size: 8px; color: #52525B;")
        layout.addWidget(self.dot)

        self.label = QLabel(name)
        self.label.setFont(QFont("JetBrains Mono", 9, QFont.Bold))
        self.label.setStyleSheet("color: #71717A;")
        layout.addWidget(self.label)

        self._update_style()

    def set_active(self, active: bool) -> None:
        self._is_active = active
        self._update_style()

    def _update_style(self) -> None:
        if self._is_active:
            self.setStyleSheet(
                "background-color: #18181B; border: 1px solid #FAFAFA; border-radius: 4px;"
            )
            self.dot.setStyleSheet("font-size: 8px; color: #10B981;")
            self.label.setStyleSheet("color: #FAFAFA;")
            glow = QGraphicsDropShadowEffect()
            glow.setColor(QColor(255, 255, 255, 60))
            glow.setBlurRadius(8)
            glow.setOffset(0, 0)
            self.setGraphicsEffect(glow)
        else:
            self.setStyleSheet(
                "background-color: #121215; border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 4px;"
            )
            self.dot.setStyleSheet("font-size: 8px; color: #3F3F46;")
            self.label.setStyleSheet("color: #52525B;")
            self.setGraphicsEffect(None)


class TelemetryStatusBar(QFrame):
    """Modern dark status bar showing live system telemetries and active model states."""

    batch_settings_requested = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(30)
        self.setStyleSheet(
            "background-color: #09090B; border-top: 1px solid rgba(255, 255, 255, 0.08);"
        )

        self._build_ui()

        # Telemetry update timer (1s polling)
        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.update_telemetry)
        self.timer.start()
        self.update_telemetry()
        self.update_batch_badge()

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 0, 12, 0)
        layout.setSpacing(10)

        # 1. Left: Status Indicator & Message
        self.status_dot = QLabel("●")
        self.status_dot.setStyleSheet("font-size: 10px; color: #10B981;")
        layout.addWidget(self.status_dot)

        self.status_msg = QLabel("Ready")
        self.status_msg.setFont(QFont("Inter", 9))
        self.status_msg.setStyleSheet("color: #A1A1AA;")
        layout.addWidget(self.status_msg)

        layout.addStretch(1)

        # 2. Center: Engine Badges
        self.badges_layout = QHBoxLayout()
        self.badges_layout.setSpacing(6)

        self.badge_ctd = EngineBadge("CTD", tooltip="ComicTextDetector (CUDA / ONNX)")
        self.badge_ocr = EngineBadge("OCR", tooltip="Dual OCR (PP-OCRv6 + PaddleOCR-VL)")
        self.badge_repair = EngineBadge("QWEN", tooltip="Qwen Visual OCR Repair")
        self.badge_trans = EngineBadge("HY-MT2", tooltip="Hy-MT2 GGUF Translation Engine")
        self.badge_lama = EngineBadge("LAMA", tooltip="LaMa GPU Neural Inpainter")

        self.engine_badges = {
            "DETECT": self.badge_ctd,
            "OCR": self.badge_ocr,
            "REPAIR": self.badge_repair,
            "TRANSLATE": self.badge_trans,
            "INPAINT": self.badge_lama,
            "RENDER": self.badge_lama,
        }

        self.badges_layout.addWidget(self.badge_ctd)
        self.badges_layout.addWidget(self.badge_ocr)
        self.badges_layout.addWidget(self.badge_repair)
        self.badges_layout.addWidget(self.badge_trans)
        self.badges_layout.addWidget(self.badge_lama)

        layout.addLayout(self.badges_layout)
        layout.addStretch(1)

        # 3. Right: Batch Mode Badge + VRAM & RAM Telemetry
        self.btn_batch_settings = QPushButton("⚡ BATCH: OTO %95")
        self.btn_batch_settings.setFont(QFont("JetBrains Mono", 9, QFont.Bold))
        self.btn_batch_settings.setCursor(Qt.PointingHandCursor)
        self.btn_batch_settings.setStyleSheet(
            "QPushButton { background-color: #121215; color: #10B981; border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 4px; padding: 2px 8px; } "
            "QPushButton:hover { background-color: #18181B; border: 1px solid #10B981; color: #FAFAFA; }"
        )
        self.btn_batch_settings.setToolTip("Donanım & Elastik Batch Ayarlarını Aç (Ctrl+Shift+B)")
        self.btn_batch_settings.clicked.connect(self._on_batch_settings_clicked)
        layout.addWidget(self.btn_batch_settings)

        self.vram_badge = QLabel("VRAM: --")
        self.vram_badge.setFont(QFont("JetBrains Mono", 9))
        self.vram_badge.setStyleSheet("color: #71717A; padding: 2px 6px; background-color: #121215; border-radius: 4px; border: 1px solid rgba(255, 255, 255, 0.06);")
        layout.addWidget(self.vram_badge)

        self.ram_badge = QLabel("RAM: --")
        self.ram_badge.setFont(QFont("JetBrains Mono", 9))
        self.ram_badge.setStyleSheet("color: #71717A; padding: 2px 6px; background-color: #121215; border-radius: 4px; border: 1px solid rgba(255, 255, 255, 0.06);")
        layout.addWidget(self.ram_badge)

    def _on_batch_settings_clicked(self) -> None:
        self.batch_settings_requested.emit()
        from gui.dialogs.batch_settings_dialog import BatchSettingsDialog
        dlg = BatchSettingsDialog(self.window())
        dlg.config_applied.connect(self.update_batch_badge)
        dlg.exec()

    def update_batch_badge(self, config: Optional[Any] = None) -> None:
        from core.system.adaptive_batcher import get_batch_config
        cfg = config or get_batch_config()
        if cfg.mode == "auto":
            pct = int(cfg.vram_ceiling * 100)
            self.btn_batch_settings.setText(f"⚡ BATCH: OTO %{pct}")
            self.btn_batch_settings.setStyleSheet(
                "QPushButton { background-color: #121215; color: #10B981; border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 4px; padding: 2px 8px; } "
                "QPushButton:hover { background-color: #18181B; border: 1px solid #10B981; color: #FAFAFA; }"
            )
        else:
            self.btn_batch_settings.setText(f"⚡ BATCH: MANUEL ({cfg.lama_batch})")
            self.btn_batch_settings.setStyleSheet(
                "QPushButton { background-color: #121215; color: #38BDF8; border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 4px; padding: 2px 8px; } "
                "QPushButton:hover { background-color: #18181B; border: 1px solid #38BDF8; color: #FAFAFA; }"
            )

    def set_status(self, message: str, is_busy: bool = False) -> None:
        self.status_msg.setText(message)
        if is_busy:
            self.status_dot.setStyleSheet("font-size: 10px; color: #38BDF8;")
        else:
            self.status_dot.setStyleSheet("font-size: 10px; color: #10B981;")

    def set_active_stage(self, stage_name: Optional[str] = None) -> None:
        """Highlights the active AI model badge."""
        stage_upper = stage_name.upper() if stage_name else None
        for name, badge in self.engine_badges.items():
            badge.set_active(name == stage_upper)

    def reset_badges(self) -> None:
        for badge in self.engine_badges.values():
            badge.set_active(False)

    def update_telemetry(self) -> None:
        """Polls GPU and RAM telemetry and updates labels."""
        gpu_info = SystemTelemetry.get_gpu_telemetry()
        if gpu_info.get("has_gpu"):
            res_gb = gpu_info["reserved_gb"]
            tot_gb = gpu_info["total_gb"]
            pct = gpu_info["usage_percent"]
            self.vram_badge.setText(f"VRAM: {res_gb:.1f} / {tot_gb:.1f} GB ({int(pct)}%)")
            if pct > 85:
                self.vram_badge.setStyleSheet("color: #EF4444; padding: 2px 6px; background-color: #121215; border-radius: 4px; border: 1px solid #EF4444;")
            else:
                self.vram_badge.setStyleSheet("color: #FAFAFA; padding: 2px 6px; background-color: #121215; border-radius: 4px; border: 1px solid rgba(255, 255, 255, 0.06);")
        else:
            self.vram_badge.setText("VRAM: CPU Mode")
            self.vram_badge.setStyleSheet("color: #71717A; padding: 2px 6px; background-color: #121215; border-radius: 4px; border: 1px solid rgba(255, 255, 255, 0.06);")

        ram_info = SystemTelemetry.get_ram_telemetry()
        if ram_info.get("total_gb", 0) > 0:
            used_gb = ram_info["used_gb"]
            tot_gb = ram_info["total_gb"]
            pct = ram_info["percent"]
            self.ram_badge.setText(f"RAM: {used_gb:.1f} / {tot_gb:.1f} GB ({int(pct)}%)")
        else:
            self.ram_badge.setText("RAM: OK")
