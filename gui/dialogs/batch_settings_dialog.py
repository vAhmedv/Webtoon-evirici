from typing import Optional, Any
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from core.system.adaptive_batcher import BatchConfig, get_batch_config, set_batch_config
from gui.components.telemetry_bar import SystemTelemetry


class BatchSettingsDialog(QDialog):
    """Modern dark themed dialog for GPU VRAM ceiling and batch parameters."""

    config_applied = Signal(BatchConfig)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("⚡ Donanım & Elastik Batch Ayarları")
        self.setFixedSize(580, 580)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setStyleSheet(
            "QDialog { background-color: #09090B; border: 1px solid #27272A; } "
            "QLabel { color: #FAFAFA; font-family: 'Inter', sans-serif; } "
            "QRadioButton { color: #E4E4E7; font-family: 'Inter', sans-serif; font-size: 11px; } "
            "QRadioButton::indicator { width: 14px; height: 14px; } "
            "QSlider::groove:horizontal { height: 6px; background: #18181B; border-radius: 3px; } "
            "QSlider::sub-page:horizontal { background: #10B981; border-radius: 3px; } "
            "QSlider::handle:horizontal { background: #FAFAFA; width: 14px; margin-top: -4px; margin-bottom: -4px; border-radius: 7px; } "
            "QPushButton { font-family: 'Inter', sans-serif; font-size: 11px; font-weight: 600; border-radius: 6px; padding: 6px 14px; } "
        )

        self._config = get_batch_config()
        self._bench_worker: Optional[Any] = None
        self._build_ui()
        self._load_values()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)

        # 1. Header Title
        title_label = QLabel("⚡ Donanım & Elastik Batch Ayarları")
        title_label.setFont(QFont("Inter", 12, QFont.Bold))
        title_label.setStyleSheet("color: #FAFAFA;")
        layout.addWidget(title_label)

        # 2. GPU / System Info Card
        gpu_info = SystemTelemetry.get_gpu_telemetry()
        card = QFrame(self)
        card.setStyleSheet("background-color: #121215; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px;")
        card_layout = QHBoxLayout(card)
        card_layout.setContentsMargins(14, 10, 14, 10)

        gpu_dot = QLabel("●")
        gpu_dot.setStyleSheet("font-size: 10px; color: #10B981;" if gpu_info.get("has_gpu") else "font-size: 10px; color: #71717A;")
        card_layout.addWidget(gpu_dot)

        if gpu_info.get("has_gpu"):
            dev_name = gpu_info.get("device_name", "NVIDIA GPU")
            res_gb = gpu_info.get("reserved_gb", 0)
            tot_gb = gpu_info.get("total_gb", 0)
            pct = gpu_info.get("usage_percent", 0)
            card_text = f"{dev_name}  |  VRAM: {res_gb:.1f} / {tot_gb:.1f} GB (%{int(pct)})"
        else:
            card_text = "CPU Modu  |  GPU Hızlandırıcı Bulunamadı"

        lbl_gpu = QLabel(card_text)
        lbl_gpu.setFont(QFont("JetBrains Mono", 9, QFont.Bold))
        lbl_gpu.setStyleSheet("color: #E4E4E7;")
        card_layout.addWidget(lbl_gpu)
        card_layout.addStretch(1)

        layout.addWidget(card)

        # 3. Mode Selection (Auto vs Manual)
        self.radio_auto = QRadioButton("● Otomatik Elastik (%95 VRAM Tavanı)")
        self.radio_auto.setFont(QFont("Inter", 10, QFont.Bold))
        self.radio_auto.setToolTip("Önerilen: OOM anında 28->27->26 adım adım küçülerek GPU'yu maksimum doygunlukta tutar.")

        self.radio_manual = QRadioButton("○ Manuel Kontrol")
        self.radio_manual.setFont(QFont("Inter", 10, QFont.Bold))
        self.radio_manual.setToolTip("Modül bazında sabit batch boyutlarını elle ayarlamanıza olanak tanır.")

        self.btn_group = QButtonGroup(self)
        self.btn_group.addButton(self.radio_auto)
        self.btn_group.addButton(self.radio_manual)
        self.radio_auto.toggled.connect(self._on_mode_changed)

        mode_layout = QHBoxLayout()
        mode_layout.addWidget(self.radio_auto)
        mode_layout.addWidget(self.radio_manual)
        mode_layout.addStretch(1)
        layout.addLayout(mode_layout)

        # 4. Sliders Section
        sliders_frame = QFrame(self)
        sliders_frame.setStyleSheet("background-color: #121215; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px;")
        grid = QGridLayout(sliders_frame)
        grid.setContentsMargins(14, 12, 14, 12)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(10)

        # VRAM Ceiling Slider
        lbl_vram_title = QLabel("VRAM Tavan Sınırı:")
        lbl_vram_title.setFont(QFont("Inter", 9))
        self.lbl_vram_val = QLabel("95%")
        self.lbl_vram_val.setFont(QFont("JetBrains Mono", 9, QFont.Bold))
        self.lbl_vram_val.setStyleSheet("color: #10B981;")

        self.slider_vram = QSlider(Qt.Horizontal)
        self.slider_vram.setRange(70, 98)
        self.slider_vram.setValue(95)
        self.slider_vram.valueChanged.connect(lambda v: self.lbl_vram_val.setText(f"{v}%"))

        grid.addWidget(lbl_vram_title, 0, 0)
        grid.addWidget(self.slider_vram, 0, 1)
        grid.addWidget(self.lbl_vram_val, 0, 2)

        # LaMa Inpaint Slider
        lbl_lama_title = QLabel("LaMa Inpainting Batch:")
        lbl_lama_title.setFont(QFont("Inter", 9))
        self.lbl_lama_val = QLabel("24")
        self.lbl_lama_val.setFont(QFont("JetBrains Mono", 9, QFont.Bold))
        self.lbl_lama_val.setStyleSheet("color: #38BDF8;")

        self.slider_lama = QSlider(Qt.Horizontal)
        self.slider_lama.setRange(1, 256)
        self.slider_lama.setValue(24)
        self.slider_lama.valueChanged.connect(lambda v: self.lbl_lama_val.setText(str(v)))

        grid.addWidget(lbl_lama_title, 1, 0)
        grid.addWidget(self.slider_lama, 1, 1)
        grid.addWidget(self.lbl_lama_val, 1, 2)

        # PaddleOCR-VL Slider
        lbl_ocr_title = QLabel("PaddleOCR-VL GPU Batch:")
        lbl_ocr_title.setFont(QFont("Inter", 9))
        self.lbl_ocr_val = QLabel("32")
        self.lbl_ocr_val.setFont(QFont("JetBrains Mono", 9, QFont.Bold))
        self.lbl_ocr_val.setStyleSheet("color: #38BDF8;")

        self.slider_ocr = QSlider(Qt.Horizontal)
        self.slider_ocr.setRange(1, 256)
        self.slider_ocr.setValue(32)
        self.slider_ocr.valueChanged.connect(lambda v: self.lbl_ocr_val.setText(str(v)))

        grid.addWidget(lbl_ocr_title, 2, 0)
        grid.addWidget(self.slider_ocr, 2, 1)
        grid.addWidget(self.lbl_ocr_val, 2, 2)

        # Hy-MT2 LLM Chunk Slider
        lbl_llm_title = QLabel("Hy-MT2 LLM Chunk:")
        lbl_llm_title.setFont(QFont("Inter", 9))
        self.lbl_llm_val = QLabel("16")
        self.lbl_llm_val.setFont(QFont("JetBrains Mono", 9, QFont.Bold))
        self.lbl_llm_val.setStyleSheet("color: #38BDF8;")

        self.slider_llm = QSlider(Qt.Horizontal)
        self.slider_llm.setRange(1, 64)
        self.slider_llm.setValue(16)
        self.slider_llm.valueChanged.connect(lambda v: self.lbl_llm_val.setText(str(v)))

        grid.addWidget(lbl_llm_title, 3, 0)
        grid.addWidget(self.slider_llm, 3, 1)
        grid.addWidget(self.lbl_llm_val, 3, 2)

        # CPU OCR Workers Slider
        lbl_cpu_title = QLabel("CPU OCR Workers:")
        lbl_cpu_title.setFont(QFont("Inter", 9))
        self.lbl_cpu_val = QLabel("10")
        self.lbl_cpu_val.setFont(QFont("JetBrains Mono", 9, QFont.Bold))
        self.lbl_cpu_val.setStyleSheet("color: #38BDF8;")

        self.slider_cpu = QSlider(Qt.Horizontal)
        self.slider_cpu.setRange(1, 16)
        self.slider_cpu.setValue(10)
        self.slider_cpu.valueChanged.connect(lambda v: self.lbl_cpu_val.setText(str(v)))

        grid.addWidget(lbl_cpu_title, 4, 0)
        grid.addWidget(self.slider_cpu, 4, 1)
        grid.addWidget(self.lbl_cpu_val, 4, 2)

        layout.addWidget(sliders_frame)

        # 5. Live Benchmark & Auto-Tuning Card
        bench_frame = QFrame(self)
        bench_frame.setStyleSheet("background-color: #121215; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px;")
        bench_layout = QVBoxLayout(bench_frame)
        bench_layout.setContentsMargins(14, 10, 14, 10)
        bench_layout.setSpacing(8)

        bench_header_layout = QHBoxLayout()
        self.btn_benchmark = QPushButton("🚀 Donanımı Canlı Test Et & Kalibre Et")
        self.btn_benchmark.setStyleSheet(
            "QPushButton { background-color: #18181B; color: #38BDF8; border: 1px solid rgba(56, 189, 248, 0.3); } "
            "QPushButton:hover { background-color: #27272A; color: #FAFAFA; border: 1px solid #38BDF8; }"
        )
        self.btn_benchmark.clicked.connect(self._on_start_benchmark)
        bench_header_layout.addWidget(self.btn_benchmark)

        self.lbl_bench_status = QLabel("Hazır: GPU stres testi ile tavan batch sınırını bulun.")
        self.lbl_bench_status.setFont(QFont("Inter", 8))
        self.lbl_bench_status.setStyleSheet("color: #71717A;")
        bench_header_layout.addWidget(self.lbl_bench_status, 1)

        bench_layout.addLayout(bench_header_layout)

        self.bench_progress = QProgressBar(self)
        self.bench_progress.setFixedHeight(6)
        self.bench_progress.setTextVisible(False)
        self.bench_progress.setRange(0, 100)
        self.bench_progress.setValue(0)
        self.bench_progress.setStyleSheet(
            "QProgressBar { background-color: #18181B; border: none; border-radius: 3px; } "
            "QProgressBar::chunk { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #10B981, stop:1 #38BDF8); border-radius: 3px; }"
        )
        bench_layout.addWidget(self.bench_progress)

        layout.addWidget(bench_frame)
        layout.addStretch(1)

        # 6. Buttons Footer
        btn_layout = QHBoxLayout()

        btn_reset = QPushButton("Varsayılana Sıfırla")
        btn_reset.setStyleSheet("background-color: #18181B; color: #A1A1AA; border: 1px solid rgba(255, 255, 255, 0.1);")
        btn_reset.clicked.connect(self._on_reset_clicked)
        btn_layout.addWidget(btn_reset)

        btn_layout.addStretch(1)

        btn_cancel = QPushButton("İptal")
        btn_cancel.setStyleSheet("background-color: #18181B; color: #A1A1AA; border: 1px solid rgba(255, 255, 255, 0.1);")
        btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(btn_cancel)

        btn_save = QPushButton("Kaydet & Uygula")
        btn_save.setStyleSheet("background-color: #10B981; color: #09090B; border: none;")
        btn_save.clicked.connect(self._on_save_clicked)
        btn_layout.addWidget(btn_save)

        layout.addLayout(btn_layout)

    def _load_values(self) -> None:
        cfg = self._config
        if cfg.mode == "auto":
            self.radio_auto.setChecked(True)
        else:
            self.radio_manual.setChecked(True)

        self.slider_vram.setValue(int(cfg.vram_ceiling * 100))
        self.slider_lama.setValue(cfg.lama_batch)
        self.slider_ocr.setValue(cfg.ocr_vl_batch)
        self.slider_llm.setValue(cfg.llm_chunk)
        self.slider_cpu.setValue(cfg.cpu_ocr_workers)
        self._on_mode_changed(self.radio_auto.isChecked())

    def _on_mode_changed(self, is_auto: bool) -> None:
        # In auto mode, module batch sliders are dimmed since adaptive batcher dynamically scales them
        self.slider_lama.setEnabled(not is_auto)
        self.slider_ocr.setEnabled(not is_auto)
        self.slider_llm.setEnabled(not is_auto)
        self.slider_cpu.setEnabled(not is_auto)

        opacity = "0.4" if is_auto else "1.0"
        dim_style = f"opacity: {opacity};"
        self.lbl_lama_val.setStyleSheet(f"color: #38BDF8; {dim_style}")
        self.lbl_ocr_val.setStyleSheet(f"color: #38BDF8; {dim_style}")
        self.lbl_llm_val.setStyleSheet(f"color: #38BDF8; {dim_style}")
        self.lbl_cpu_val.setStyleSheet(f"color: #38BDF8; {dim_style}")

    def _on_reset_clicked(self) -> None:
        self.radio_auto.setChecked(True)
        self.slider_vram.setValue(95)
        self.slider_lama.setValue(24)
        self.slider_ocr.setValue(32)
        self.slider_llm.setValue(16)
        self.slider_cpu.setValue(10)

    def _on_start_benchmark(self) -> None:
        self.btn_benchmark.setEnabled(False)
        self.lbl_bench_status.setText("Donanım stres testi başlatılıyor...")
        self.lbl_bench_status.setStyleSheet("color: #38BDF8;")
        self.bench_progress.setValue(10)

        from core.system.hardware_benchmark import HardwareBenchmarkWorker
        vram_limit = self.slider_vram.value() / 100.0
        self._bench_worker = HardwareBenchmarkWorker(vram_ceiling=vram_limit, parent=self)
        self._bench_worker.step_updated.connect(self._on_bench_step)
        self._bench_worker.benchmark_completed.connect(self._on_bench_completed)
        self._bench_worker.benchmark_failed.connect(self._on_bench_failed)
        self._bench_worker.start()

    def _on_bench_step(self, batch: int, used_gb: float, tot_gb: float, pct: float) -> None:
        self.bench_progress.setValue(min(95, int((batch / 256) * 100)))
        self.lbl_bench_status.setText(f"Batch {batch} deneniyor... VRAM: {used_gb:.1f}/{tot_gb:.1f} GB (%{int(pct)})")
        self.lbl_bench_status.setStyleSheet("color: #F59E0B;")

    def _on_bench_completed(self, optimal_lama: int, optimal_ocr: int, optimal_llm: int, max_vram_pct: float) -> None:
        self.bench_progress.setValue(100)
        self.lbl_bench_status.setText(f"✔ Test Başarılı! Maksimum Kararlı Batch: {optimal_lama} (VRAM %{int(max_vram_pct)})")
        self.lbl_bench_status.setStyleSheet("color: #10B981; font-weight: 600;")
        self.btn_benchmark.setEnabled(True)

        # Auto-populate sliders with optimal hardware values
        self.slider_lama.setValue(optimal_lama)
        self.slider_ocr.setValue(optimal_ocr)
        self.slider_llm.setValue(optimal_llm)

    def _on_bench_failed(self, err: str) -> None:
        self.bench_progress.setValue(0)
        self.lbl_bench_status.setText(f"Hata: {err}")
        self.lbl_bench_status.setStyleSheet("color: #EF4444;")
        self.btn_benchmark.setEnabled(True)

    def _on_save_clicked(self) -> None:
        new_config = BatchConfig(
            mode="auto" if self.radio_auto.isChecked() else "manual",
            vram_ceiling=self.slider_vram.value() / 100.0,
            lama_batch=self.slider_lama.value(),
            ocr_vl_batch=self.slider_ocr.value(),
            llm_chunk=self.slider_llm.value(),
            cpu_ocr_workers=self.slider_cpu.value(),
        )
        set_batch_config(new_config)
        self.config_applied.emit(new_config)
        self.accept()

    def closeEvent(self, event) -> None:
        if self._bench_worker and self._bench_worker.isRunning():
            self._bench_worker.request_cancel()
            self._bench_worker.wait(1000)
        super().closeEvent(event)
