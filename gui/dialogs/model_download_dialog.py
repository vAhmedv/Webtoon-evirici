"""ModelDownloadDialog & Worker: Modern modal dialog for downloading missing AI models with speed and ETA tracking."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core.models.manager import ModelManager, ModelSpec


class ModelDownloadWorker(QThread):
    """Background worker that sequentially downloads required missing models."""

    progress_updated = Signal(int, int, float, float)  # (downloaded_bytes, total_bytes, speed_bytes_per_sec, eta_sec)
    model_started = Signal(str, str, int, int)         # (model_name, description, current_idx, total_count)
    download_completed = Signal()
    download_failed = Signal(str)

    def __init__(self, manager: ModelManager, missing_specs: Sequence[ModelSpec], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.manager = manager
        self.missing_specs = list(missing_specs)
        self._is_cancelled = False

    def request_cancel(self) -> None:
        self._is_cancelled = True

    def run(self) -> None:
        total_count = len(self.missing_specs)
        for idx, spec in enumerate(self.missing_specs, start=1):
            if self._is_cancelled:
                return

            self.model_started.emit(spec.name, spec.description, idx, total_count)

            def _on_progress(downloaded: int, total: int, speed: float, eta: float) -> None:
                self.progress_updated.emit(downloaded, total, speed, eta)

            try:
                self.manager.download_model(
                    spec=spec,
                    progress_callback=_on_progress,
                    cancel_check=lambda: self._is_cancelled,
                )
            except InterruptedError:
                return
            except Exception as e:
                self.download_failed.emit(f"'{spec.name}' indirilirken hata oluştu:\n{e}")
                return

        self.download_completed.emit()


class ModelDownloadDialog(QDialog):
    """Modern dark themed dialog for downloading missing models."""

    def __init__(self, manager: ModelManager, missing_specs: Sequence[ModelSpec], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.manager = manager
        self.missing_specs = list(missing_specs)
        self._worker: Optional[ModelDownloadWorker] = None
        self._was_successful = False

        self.setWindowTitle("Gerekli Yapay Zeka Modelleri İndiriliyor")
        self.setFixedSize(540, 320)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setStyleSheet(
            "QDialog { background-color: #09090B; border: 1px solid #27272A; } "
            "QLabel { color: #FAFAFA; font-family: 'Inter', sans-serif; } "
            "QPushButton { font-family: 'Inter', sans-serif; font-size: 11px; font-weight: 600; border-radius: 6px; padding: 6px 12px; } "
        )

        self._build_ui()
        self._start_download()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(14)

        # 1. Header Title
        self.lbl_title = QLabel(f"Gerekli Yapay Zeka Modelleri İndiriliyor (1 / {len(self.missing_specs)})")
        self.lbl_title.setFont(QFont("Inter", 12, QFont.Bold))
        self.lbl_title.setStyleSheet("color: #FAFAFA;")
        layout.addWidget(self.lbl_title)

        # 2. Active Model Info Card
        card = QFrame(self)
        card.setStyleSheet("background-color: #121215; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px;")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(14, 12, 14, 12)
        card_layout.setSpacing(4)

        self.lbl_model_name = QLabel("Model Adı")
        self.lbl_model_name.setFont(QFont("Inter", 10, QFont.Bold))
        self.lbl_model_name.setStyleSheet("color: #38BDF8;")
        card_layout.addWidget(self.lbl_model_name)

        self.lbl_model_desc = QLabel("Model açıklaması...")
        self.lbl_model_desc.setFont(QFont("Inter", 9))
        self.lbl_model_desc.setStyleSheet("color: #A1A1AA;")
        self.lbl_model_desc.setWordWrap(True)
        card_layout.addWidget(self.lbl_model_desc)

        layout.addWidget(card)

        # 3. Progress Bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setFixedHeight(8)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet(
            "QProgressBar { background-color: #18181B; border: none; border-radius: 4px; } "
            "QProgressBar::chunk { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #10B981, stop:1 #38BDF8); border-radius: 4px; }"
        )
        layout.addWidget(self.progress_bar)

        # 4. Status and Telemetry Row
        telemetry_layout = QHBoxLayout()
        self.lbl_size = QLabel("0 MB / 0 MB (%0)")
        self.lbl_size.setFont(QFont("JetBrains Mono", 9))
        self.lbl_size.setStyleSheet("color: #71717A;")
        telemetry_layout.addWidget(self.lbl_size)

        telemetry_layout.addStretch(1)

        self.lbl_speed = QLabel("Hız: -- MB/s")
        self.lbl_speed.setFont(QFont("JetBrains Mono", 9))
        self.lbl_speed.setStyleSheet("color: #71717A;")
        telemetry_layout.addWidget(self.lbl_speed)

        self.lbl_eta = QLabel("Kalan: --")
        self.lbl_eta.setFont(QFont("JetBrains Mono", 9))
        self.lbl_eta.setStyleSheet("color: #71717A; margin-left: 8px;")
        telemetry_layout.addWidget(self.lbl_eta)

        layout.addLayout(telemetry_layout)
        layout.addStretch(1)

        # 5. Footer (Model Dir & Cancel Button)
        footer_layout = QHBoxLayout()
        self.lbl_dir = QLabel(f"Dizin: {self.manager.get_model_dir().name}")
        self.lbl_dir.setFont(QFont("Inter", 8))
        self.lbl_dir.setStyleSheet("color: #52525B;")
        footer_layout.addWidget(self.lbl_dir)

        btn_change_dir = QPushButton("Değiştir")
        btn_change_dir.setStyleSheet("background-color: #18181B; color: #A1A1AA; border: 1px solid rgba(255, 255, 255, 0.1);")
        btn_change_dir.clicked.connect(self._on_change_dir_clicked)
        footer_layout.addWidget(btn_change_dir)

        footer_layout.addStretch(1)

        self.btn_cancel = QPushButton("İptal Et")
        self.btn_cancel.setStyleSheet("background-color: #27272A; color: #FAFAFA; border: none;")
        self.btn_cancel.clicked.connect(self._on_cancel_clicked)
        footer_layout.addWidget(self.btn_cancel)

        layout.addLayout(footer_layout)

    def _start_download(self) -> None:
        self._worker = ModelDownloadWorker(self.manager, self.missing_specs, self)
        self._worker.model_started.connect(self._on_model_started)
        self._worker.progress_updated.connect(self._on_progress_updated)
        self._worker.download_completed.connect(self._on_download_completed)
        self._worker.download_failed.connect(self._on_download_failed)
        self._worker.start()

    def _on_model_started(self, name: str, description: str, current_idx: int, total_count: int) -> None:
        self.lbl_title.setText(f"Gerekli Yapay Zeka Modelleri İndiriliyor ({current_idx} / {total_count})")
        self.lbl_model_name.setText(name)
        self.lbl_model_desc.setText(description)
        self.progress_bar.setValue(0)

    def _on_progress_updated(self, downloaded: int, total: int, speed: float, eta: float) -> None:
        if total > 0:
            pct = int((downloaded / total) * 100)
            self.progress_bar.setValue(pct)
            dl_mb = downloaded / (1024 * 1024)
            tot_mb = total / (1024 * 1024)
            if tot_mb > 1024:
                self.lbl_size.setText(f"{dl_mb / 1024:.2f} GB / {tot_mb / 1024:.2f} GB (%{pct})")
            else:
                self.lbl_size.setText(f"{dl_mb:.1f} MB / {tot_mb:.1f} MB (%{pct})")

        speed_mb = speed / (1024 * 1024)
        self.lbl_speed.setText(f"Hız: {speed_mb:.1f} MB/s")

        if eta > 60:
            mins = int(eta // 60)
            secs = int(eta % 60)
            self.lbl_eta.setText(f"Kalan: ~{mins} dk {secs} sn")
        elif eta > 0:
            self.lbl_eta.setText(f"Kalan: ~{int(eta)} sn")
        else:
            self.lbl_eta.setText("Kalan: --")

    def _on_download_completed(self) -> None:
        self._was_successful = True
        self.accept()

    def _on_download_failed(self, error_msg: str) -> None:
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.critical(self, "İndirme Hatası", error_msg)
        self.reject()

    def _on_change_dir_clicked(self) -> None:
        selected_dir = QFileDialog.getExistingDirectory(self, "Model Depolama Klasörünü Seçin", str(self.manager.get_model_dir()))
        if selected_dir:
            self.manager.set_model_dir(Path(selected_dir))
            self.lbl_dir.setText(f"Dizin: {self.manager.get_model_dir().name}")

    def _on_cancel_clicked(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.request_cancel()
            self._worker.wait(1500)
        self.reject()

    def closeEvent(self, event) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.request_cancel()
            self._worker.wait(1500)
        super().closeEvent(event)

    @property
    def was_successful(self) -> bool:
        return self._was_successful
