"""Analysis worker — pipeline'ı background thread'de çalıştırır."""

from __future__ import annotations

import threading
import traceback
from pathlib import Path
from typing import Any

from PySide6.QtCore import QThread, Signal

from application.cancellation import CancellationToken, CancelledError
from application.chapter_analyzer import AnalysisResult, ChapterAnalyzer
from application.progress import ProgressEvent
from core.config import Config
from loguru import logger
from providers.detector.registry import get_registry
from providers.ocr.base import OCRProvider
from providers.ocr.registry import get_ocr_registry


class AnalysisWorker(QThread):
    """Pipeline'ı arka planda çalıştıran worker.

    Signals:
        progress: İlerleme güncellendiğinde.
        result: Analiz tamamlandığında.
        error: Hata oluştuğunda.
        cancelled: İptal edildiğinde.
    """

    progress = Signal(object)  # ProgressEvent
    result = Signal(object)    # AnalysisResult
    error = Signal(str)        # error message
    cancelled = Signal()

    def __init__(
        self,
        chapter_path: str | Path,
        output_path: str | Path,
        detector_name: str,
        config: Config,
        ocr_name: str | None = None,
        parent: Any = None,
    ) -> None:
        super().__init__(parent)
        self.chapter_path = chapter_path
        self.output_path = output_path
        self._detector_name = detector_name
        self.config = config
        self._ocr_name = ocr_name
        self._cancellation_token = CancellationToken()

    def request_cancel(self) -> None:
        """İptal isteği gönderir."""
        self._cancellation_token.cancel()

    @property
    def is_cancelled(self) -> bool:
        return self._cancellation_token.is_cancelled

    def run(self) -> None:
        """Thread giriş noktası."""
        worker_thread_id = threading.get_ident()
        logger.debug(f"[THREAD] AnalysisWorker.run thread id: {worker_thread_id}")

        analyzer = ChapterAnalyzer(self.config)

        def on_progress(event: ProgressEvent) -> None:
            self.progress.emit(event)

        ocr_provider: OCRProvider | None = None
        try:
            logger.debug(f"[THREAD] Creating detector '{self._detector_name}' in worker thread")
            provider = get_registry().create(self._detector_name)
            logger.debug(f"[THREAD] Provider created: {type(provider).__name__}")

            if hasattr(provider, "confidence_threshold"):
                provider.confidence_threshold = self.config.min_confidence
                logger.debug(
                    f"[THREAD] Provider confidence set to {self.config.min_confidence}"
                )

            logger.debug(f"[THREAD] Loading detector...")
            provider.load()
            logger.debug(f"[THREAD] Detector loaded")

            if self._ocr_name:
                try:
                    ocr_provider = get_ocr_registry().create(self._ocr_name)
                    logger.debug(f"[THREAD] OCR provider created: {type(ocr_provider).__name__}")
                except Exception as e:
                    logger.warning(f"[THREAD] OCR provider '{self._ocr_name}' could not be created: {e}")

            try:
                result = analyzer.analyze(
                    chapter_path=self.chapter_path,
                    output_path=self.output_path,
                    detector=provider,
                    progress_callback=on_progress,
                    cancellation_token=self._cancellation_token,
                    ocr_provider=ocr_provider,
                )
            finally:
                logger.debug(f"[THREAD] Unloading detector...")
                provider.unload()
                logger.debug(f"[THREAD] Detector unloaded")

            if ocr_provider is not None:
                try:
                    ocr_provider.unload()
                except Exception:
                    pass

            if self._cancellation_token.is_cancelled:
                self.cancelled.emit()
            else:
                self.result.emit(result)
        except CancelledError:
            self.cancelled.emit()
        except Exception as exc:
            tb = traceback.format_exc()
            self.error.emit(f"{type(exc).__name__}: {repr(exc)}\n{tb}")
