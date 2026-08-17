"""Modern dark themed dialog for GPU VRAM ceiling, batch parameters, and Gemini API settings."""

from __future__ import annotations

import os
from typing import Optional, Any
from PySide6.QtCore import Qt, Signal, QUrl
from PySide6.QtGui import QFont, QDesktopServices
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from core.config import load_config, update_gemini_api_key
from core.system.adaptive_batcher import (
    BatchConfig,
    get_batch_config,
    save_batch_config,
    set_batch_config,
)
from gui.components.telemetry_bar import SystemTelemetry


class BatchSettingsDialog(QDialog):
    """Modern dark studio modal for hardware parameters, benchmark tuning, and Google AI API configuration."""

    config_applied = Signal(BatchConfig)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("⚙️ Donanım, Batch & Çeviri API Ayarları")
        self.setFixedSize(620, 720)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setStyleSheet(
            "QDialog { background-color: #09090B; border: 1px solid #27272A; } "
            "QLabel { color: #FAFAFA; font-family: 'Inter', sans-serif; } "
            "QLineEdit { background-color: #18181B; color: #FAFAFA; border: 1px solid #27272A; border-radius: 6px; padding: 6px 10px; font-family: 'JetBrains Mono', monospace; font-size: 11px; } "
            "QLineEdit:focus { border: 1px solid #38BDF8; } "
            "QComboBox { background-color: #18181B; color: #FAFAFA; border: 1px solid #27272A; border-radius: 6px; padding: 4px 8px; font-size: 11px; } "
            "QRadioButton { color: #E4E4E7; font-family: 'Inter', sans-serif; font-size: 11px; } "
            "QRadioButton::indicator { width: 14px; height: 14px; } "
            "QSlider::groove:horizontal { height: 6px; background: #18181B; border-radius: 3px; } "
            "QSlider::sub-page:horizontal { background: #10B981; border-radius: 3px; } "
            "QSlider::handle:horizontal { background: #FAFAFA; width: 14px; margin-top: -4px; margin-bottom: -4px; border-radius: 7px; } "
            "QPushButton { font-family: 'Inter', sans-serif; font-size: 11px; font-weight: 600; border-radius: 6px; padding: 6px 14px; } "
        )

        self._config = get_batch_config()
        self._bench_worker: Optional[Any] = None
        self._show_key = False
        self._build_ui()
        self._load_values()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 18, 24, 18)
        layout.setSpacing(10)

        # 1. Header Title
        header_layout = QHBoxLayout()
        title_label = QLabel("⚡ Donanım, Batch & Çeviri API Ayarları")
        title_label.setFont(QFont("Inter", 12, QFont.Bold))
        title_label.setStyleSheet("color: #FAFAFA;")
        header_layout.addWidget(title_label)
        header_layout.addStretch(1)
        layout.addLayout(header_layout)

        # 2. Google AI (Gemini Flash API) Section Card
        gemini_card = QFrame(self)
        gemini_card.setStyleSheet("background-color: #121215; border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 8px;")
        gemini_layout = QVBoxLayout(gemini_card)
        gemini_layout.setContentsMargins(14, 10, 14, 10)
        gemini_layout.setSpacing(8)

        gemini_header = QHBoxLayout()
        gemini_title = QLabel("🌐 Google AI Çeviri Motoru (Gemini Free API)")
        gemini_title.setFont(QFont("Inter", 10, QFont.Bold))
        gemini_title.setStyleSheet("color: #38BDF8;")
        gemini_header.addWidget(gemini_title)

        gemini_header.addStretch(1)

        btn_get_key = QPushButton("🔑 Ücretsiz Key Al")
        btn_get_key.setStyleSheet(
            "QPushButton { background-color: #18181B; color: #38BDF8; border: 1px solid rgba(56, 189, 248, 0.4); padding: 3px 8px; font-size: 10px; } "
            "QPushButton:hover { background-color: #27272A; border: 1px solid #38BDF8; }"
        )
        btn_get_key.setCursor(Qt.PointingHandCursor)
        btn_get_key.clicked.connect(lambda: QDesktopServices.openUrl(QUrl("https://aistudio.google.com/app/apikey")))
        gemini_header.addWidget(btn_get_key)

        gemini_layout.addLayout(gemini_header)

        # API Key Input & Action Row
        key_row = QHBoxLayout()
        key_row.setSpacing(6)

        self.edit_api_key = QLineEdit(self)
        self.edit_api_key.setPlaceholderText("Google AI Studio API Anahtarınızı yapıştırın (AIzaSy...)")
        self.edit_api_key.setEchoMode(QLineEdit.Password)
        key_row.addWidget(self.edit_api_key, 1)

        self.btn_toggle_eye = QPushButton("👁️")
        self.btn_toggle_eye.setFixedWidth(32)
        self.btn_toggle_eye.setStyleSheet("background-color: #18181B; color: #A1A1AA; border: 1px solid #27272A;")
        self.btn_toggle_eye.setCursor(Qt.PointingHandCursor)
        self.btn_toggle_eye.clicked.connect(self._on_toggle_eye)
        key_row.addWidget(self.btn_toggle_eye)

        self.btn_test_gemini = QPushButton("🧪 Test Et")
        self.btn_test_gemini.setStyleSheet(
            "QPushButton { background-color: #18181B; color: #10B981; border: 1px solid rgba(16, 185, 129, 0.4); } "
            "QPushButton:hover { background-color: #27272A; border: 1px solid #10B981; }"
        )
        self.btn_test_gemini.setCursor(Qt.PointingHandCursor)
        self.btn_test_gemini.clicked.connect(self._on_test_gemini_api)
        key_row.addWidget(self.btn_test_gemini)

        gemini_layout.addLayout(key_row)

        # Status & Model Selection Row
        status_row = QHBoxLayout()
        self.lbl_gemini_status = QLabel("Durum: Kontrol ediliyor...")
        self.lbl_gemini_status.setFont(QFont("Inter", 9))
        self.lbl_gemini_status.setStyleSheet("color: #71717A;")
        status_row.addWidget(self.lbl_gemini_status, 1)

        self.combo_model = QComboBox(self)
        self.combo_model.addItems(["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"])
        status_row.addWidget(self.combo_model)

        gemini_layout.addLayout(status_row)
        layout.addWidget(gemini_card)

        # 3. GPU / System Info Card
        gpu_info = SystemTelemetry.get_gpu_telemetry()
        card = QFrame(self)
        card.setStyleSheet("background-color: #121215; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px;")
        card_layout = QHBoxLayout(card)
        card_layout.setContentsMargins(14, 8, 14, 8)

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

        # 4. Mode Selection (Auto vs Manual)
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

        # 5. Sliders Section
        sliders_frame = QFrame(self)
        sliders_frame.setStyleSheet("background-color: #121215; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px;")
        grid = QGridLayout(sliders_frame)
        grid.setContentsMargins(14, 10, 14, 10)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(8)

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
        self.lbl_ocr_val = QLabel("64")
        self.lbl_ocr_val.setFont(QFont("JetBrains Mono", 9, QFont.Bold))
        self.lbl_ocr_val.setStyleSheet("color: #38BDF8;")

        self.slider_ocr = QSlider(Qt.Horizontal)
        self.slider_ocr.setRange(1, 256)
        self.slider_ocr.setValue(64)
        self.slider_ocr.valueChanged.connect(lambda v: self.lbl_ocr_val.setText(str(v)))

        grid.addWidget(lbl_ocr_title, 2, 0)
        grid.addWidget(self.slider_ocr, 2, 1)
        grid.addWidget(self.lbl_ocr_val, 2, 2)

        # CTD Detector Tile Batch Slider
        lbl_det_title = QLabel("CTD Detector Tile Batch:")
        lbl_det_title.setFont(QFont("Inter", 9))
        self.lbl_det_val = QLabel("16")
        self.lbl_det_val.setFont(QFont("JetBrains Mono", 9, QFont.Bold))
        self.lbl_det_val.setStyleSheet("color: #38BDF8;")

        self.slider_det = QSlider(Qt.Horizontal)
        self.slider_det.setRange(1, 32)
        self.slider_det.setValue(16)
        self.slider_det.valueChanged.connect(lambda v: self.lbl_det_val.setText(str(v)))

        grid.addWidget(lbl_det_title, 3, 0)
        grid.addWidget(self.slider_det, 3, 1)
        grid.addWidget(self.lbl_det_val, 3, 2)

        # Hy-MT2 LLM Chunk Slider
        lbl_llm_title = QLabel("Hy-MT2 LLM Chunk:")
        lbl_llm_title.setFont(QFont("Inter", 9))
        self.lbl_llm_val = QLabel("32")
        self.lbl_llm_val.setFont(QFont("JetBrains Mono", 9, QFont.Bold))
        self.lbl_llm_val.setStyleSheet("color: #38BDF8;")

        self.slider_llm = QSlider(Qt.Horizontal)
        self.slider_llm.setRange(1, 64)
        self.slider_llm.setValue(32)
        self.slider_llm.valueChanged.connect(lambda v: self.lbl_llm_val.setText(str(v)))

        grid.addWidget(lbl_llm_title, 4, 0)
        grid.addWidget(self.slider_llm, 4, 1)
        grid.addWidget(self.lbl_llm_val, 4, 2)

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

        grid.addWidget(lbl_cpu_title, 5, 0)
        grid.addWidget(self.slider_cpu, 5, 1)
        grid.addWidget(self.lbl_cpu_val, 5, 2)

        layout.addWidget(sliders_frame)

        # 6. Live Benchmark & Auto-Tuning Card
        bench_frame = QFrame(self)
        bench_frame.setStyleSheet("background-color: #121215; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px;")
        bench_layout = QVBoxLayout(bench_frame)
        bench_layout.setContentsMargins(14, 8, 14, 8)
        bench_layout.setSpacing(6)

        bench_header_layout = QHBoxLayout()
        self.btn_benchmark = QPushButton("🚀 Donanımı Canlı Test Et")
        self.btn_benchmark.setStyleSheet(
            "QPushButton { background-color: #18181B; color: #38BDF8; border: 1px solid rgba(56, 189, 248, 0.3); } "
            "QPushButton:hover { background-color: #27272A; color: #FAFAFA; border: 1px solid #38BDF8; }"
        )
        self.btn_benchmark.clicked.connect(self._on_start_benchmark)
        bench_header_layout.addWidget(self.btn_benchmark)

        self.lbl_bench_status = QLabel("GPU stres testi ile maksimum kararlı batch sınırını bulun.")
        self.lbl_bench_status.setFont(QFont("Inter", 8))
        self.lbl_bench_status.setStyleSheet("color: #71717A;")
        bench_header_layout.addWidget(self.lbl_bench_status, 1)

        bench_layout.addLayout(bench_header_layout)

        self.bench_progress = QProgressBar(self)
        self.bench_progress.setFixedHeight(4)
        self.bench_progress.setTextVisible(False)
        self.bench_progress.setRange(0, 100)
        self.bench_progress.setValue(0)
        self.bench_progress.setStyleSheet(
            "QProgressBar { background-color: #18181B; border: none; border-radius: 2px; } "
            "QProgressBar::chunk { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #10B981, stop:1 #38BDF8); border-radius: 2px; }"
        )
        bench_layout.addWidget(self.bench_progress)

        layout.addWidget(bench_frame)

        # 7. Buttons Footer
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
        btn_save.setStyleSheet("background-color: #10B981; color: #09090B; border: none; font-weight: 700;")
        btn_save.clicked.connect(self._on_save_clicked)
        btn_layout.addWidget(btn_save)

        layout.addLayout(btn_layout)

    def _on_toggle_eye(self) -> None:
        self._show_key = not self._show_key
        if self._show_key:
            self.edit_api_key.setEchoMode(QLineEdit.Normal)
            self.btn_toggle_eye.setStyleSheet("background-color: #27272A; color: #38BDF8; border: 1px solid #38BDF8;")
        else:
            self.edit_api_key.setEchoMode(QLineEdit.Password)
            self.btn_toggle_eye.setStyleSheet("background-color: #18181B; color: #A1A1AA; border: 1px solid #27272A;")

    def _on_test_gemini_api(self) -> None:
        key = self.edit_api_key.text().strip()
        if not key:
            self.lbl_gemini_status.setText("❌ Lütfen önce bir API anahtarı girin.")
            self.lbl_gemini_status.setStyleSheet("color: #EF4444;")
            return

        model = self.combo_model.currentText()
        self.lbl_gemini_status.setText("Bağlantı test ediliyor...")
        self.lbl_gemini_status.setStyleSheet("color: #F59E0B;")

        from providers.translation.gemini_translation import GeminiTranslationProvider
        try:
            ok, msg = GeminiTranslationProvider.verify_connection(api_key=key, model_name=model, timeout_sec=10.0)
            if ok:
                self.lbl_gemini_status.setText(f"✅ {msg}")
                self.lbl_gemini_status.setStyleSheet("color: #10B981; font-weight: bold;")
            else:
                self.lbl_gemini_status.setText(f"❌ {msg}")
                self.lbl_gemini_status.setStyleSheet("color: #EF4444;")
        except Exception as e:
            self.lbl_gemini_status.setText(f"❌ Hata: {str(e)[:50]}")
            self.lbl_gemini_status.setStyleSheet("color: #EF4444;")

    def _load_values(self) -> None:
        # Load API key and model from config or environment
        try:
            app_cfg = load_config()
            saved_key = app_cfg.translator.gemini_api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
            self.edit_api_key.setText(saved_key)
            idx = self.combo_model.findText(app_cfg.translator.gemini_model)
            if idx >= 0:
                self.combo_model.setCurrentIndex(idx)
            if saved_key:
                self.lbl_gemini_status.setText("✅ API Anahtarı Kayıtlı.")
                self.lbl_gemini_status.setStyleSheet("color: #10B981;")
            else:
                self.lbl_gemini_status.setText("⚠️ Anahtar yok (Yerel Hy-MT2 GGUF kullanılır).")
                self.lbl_gemini_status.setStyleSheet("color: #71717A;")
        except Exception:
            pass

        cfg = self._config
        if cfg.mode == "auto":
            self.radio_auto.setChecked(True)
        else:
            self.radio_manual.setChecked(True)

        self.slider_vram.setValue(int(cfg.vram_ceiling * 100))
        self.slider_lama.setValue(cfg.lama_batch)
        self.slider_ocr.setValue(cfg.ocr_vl_batch)
        self.slider_det.setValue(cfg.detector_tile_batch)
        self.slider_llm.setValue(cfg.llm_chunk)
        self.slider_cpu.setValue(cfg.cpu_ocr_workers)
        self._on_mode_changed(self.radio_auto.isChecked())

    def _on_mode_changed(self, is_auto: bool) -> None:
        self.slider_lama.setEnabled(not is_auto)
        self.slider_ocr.setEnabled(not is_auto)
        self.slider_det.setEnabled(not is_auto)
        self.slider_llm.setEnabled(not is_auto)
        self.slider_cpu.setEnabled(not is_auto)

        opacity = "0.4" if is_auto else "1.0"
        dim_style = f"opacity: {opacity};"
        self.lbl_lama_val.setStyleSheet(f"color: #38BDF8; {dim_style}")
        self.lbl_ocr_val.setStyleSheet(f"color: #38BDF8; {dim_style}")
        self.lbl_det_val.setStyleSheet(f"color: #38BDF8; {dim_style}")
        self.lbl_llm_val.setStyleSheet(f"color: #38BDF8; {dim_style}")
        self.lbl_cpu_val.setStyleSheet(f"color: #38BDF8; {dim_style}")

    def _on_reset_clicked(self) -> None:
        self.radio_auto.setChecked(True)
        self.slider_vram.setValue(95)
        self.slider_lama.setValue(24)
        self.slider_ocr.setValue(64)
        self.slider_det.setValue(16)
        self.slider_llm.setValue(32)
        self.slider_cpu.setValue(10)

    def _on_start_benchmark(self) -> None:
        self.btn_benchmark.setEnabled(False)
        self.lbl_bench_status.setText("Donanım test ediliyor...")
        self.lbl_bench_status.setStyleSheet("color: #38BDF8;")
        self.bench_progress.setValue(10)

        from core.system.hardware_benchmark import HardwareBenchmarkWorker
        vram_limit = self.slider_vram.value() / 100.0
        self._bench_worker = HardwareBenchmarkWorker(vram_ceiling=vram_limit, parent=self)
        self._bench_worker.step_updated.connect(self._on_bench_step)
        self._bench_worker.benchmark_completed.connect(self._on_bench_completed)
        self._bench_worker.benchmark_failed.connect(self._on_bench_failed)
        self._bench_worker.start()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._config = get_batch_config()
        self._load_values()

    def _on_bench_step(self, batch: int = 4, used_gb: float = 0.0, tot_gb: float = 0.0, pct: float = 0.0, **kwargs: Any) -> None:
        self.bench_progress.setValue(min(98, max(5, int((batch / 256) * 100))))
        if tot_gb > 0:
            self.lbl_bench_status.setText(f"Batch {batch} deneniyor... VRAM: {used_gb:.1f}/{tot_gb:.1f} GB (%{int(pct)})")
        else:
            self.lbl_bench_status.setText(f"Batch {batch} test ediliyor...")
        self.lbl_bench_status.setStyleSheet("color: #F59E0B;")

    def _on_bench_completed(
        self,
        optimal_lama: int = 12,
        optimal_ocr: int = 12,
        optimal_det: int = 4,
        optimal_llm: int = 16,
        max_batch_tested: int = 12,
        max_vram_pct: float = 0.85,
        **kwargs: Any,
    ) -> None:
        self.bench_progress.setValue(100)
        pct_val = max_vram_pct if max_vram_pct > 1.0 else max_vram_pct * 100.0
        self.lbl_bench_status.setText(
            f"✅ Test Tamamlandı! Maksimum Kararlı Batch: {optimal_lama} (OCR: {optimal_ocr}, VRAM: %{pct_val:.1f})"
        )
        self.lbl_bench_status.setStyleSheet("color: #10B981; font-weight: bold;")
        self.btn_benchmark.setEnabled(True)

        self.slider_lama.setValue(optimal_lama)
        self.slider_ocr.setValue(optimal_ocr)
        self.slider_det.setValue(optimal_det)
        if optimal_llm:
            self.slider_llm.setValue(optimal_llm)

        sticky = dict(self._config.sticky_optimal_batch)
        sticky.update({
            "lama": optimal_lama,
            "lama_batch": optimal_lama,
            "ocr_vl": optimal_ocr,
            "ocr_vl_batch": optimal_ocr,
            "detector": optimal_det,
            "detector_tile_batch": optimal_det,
            "llm": optimal_llm,
            "llm_chunk": optimal_llm,
            "benchmark_vram_pct": max_vram_pct,
            "vram_usage_pct": max_vram_pct,
        })
        self._config.sticky_optimal_batch = sticky
        save_batch_config(self._config)

    def _on_bench_failed(self, err: str) -> None:
        self.bench_progress.setValue(0)
        self.lbl_bench_status.setText(f"Hata: {err}")
        self.lbl_bench_status.setStyleSheet("color: #EF4444;")
        self.btn_benchmark.setEnabled(True)

    def _on_save_clicked(self) -> None:
        # 1. Save Gemini API key & model
        api_key = self.edit_api_key.text().strip()
        model_name = self.combo_model.currentText()
        try:
            update_gemini_api_key(api_key, model_name=model_name)
        except Exception:
            pass

        # 2. Save hardware batch configuration
        new_config = BatchConfig(
            mode="auto" if self.radio_auto.isChecked() else "manual",
            vram_ceiling=self.slider_vram.value() / 100.0,
            lama_batch=self.slider_lama.value(),
            ocr_vl_batch=self.slider_ocr.value(),
            llm_chunk=self.slider_llm.value(),
            cpu_ocr_workers=self.slider_cpu.value(),
            detector_tile_batch=self.slider_det.value(),
            sticky_optimal_batch=dict(self._config.sticky_optimal_batch),
        )
        set_batch_config(new_config)
        save_batch_config(new_config)
        self.config_applied.emit(new_config)
        self.accept()

    def closeEvent(self, event) -> None:
        if self._bench_worker and self._bench_worker.isRunning():
            self._bench_worker.request_cancel()
            self._bench_worker.wait(1000)
        super().closeEvent(event)
