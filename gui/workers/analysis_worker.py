"""Production pipeline worker running all heavy providers off the GUI thread."""

from __future__ import annotations

import threading
import traceback
from pathlib import Path
from typing import Any

from PySide6.QtCore import QThread, Signal

from application.cancellation import CancellationToken, CancelledError
from application.chapter_analyzer import ChapterAnalyzer, ProductionPipelineResult
from application.progress import ProgressEvent
from core.config import Config
from loguru import logger
from providers.detector.registry import get_registry
from providers.ocr.paddleocr import PaddleOCRProvider
from providers.ocr.paddleocr_vl import PaddleOCRVLOcrProvider
from providers.ocr.qwen_repair import QwenRepairConfig, QwenRepairProvider
from providers.translation.hy_mt2_gguf_translation import HyMT2GGUFTranslationProvider
from providers.translation.hy_mt2_gguf_translation import (
    DEFAULT_HY_MT2_MODEL_PATH,
    DEFAULT_LLAMA_SERVER_PATH as DEFAULT_HY_LLAMA_SERVER_PATH,
    DEFAULT_HY_MT2_SERVER_URL,
)


class AnalysisWorker(QThread):
    """Pipeline'ı arka planda çalıştıran worker.

    Signals:
        progress: İlerleme güncellendiğinde.
        result: Analiz tamamlandığında.
        error: Hata oluştuğunda.
        cancelled: İptal edildiğinde.
    """

    progress = Signal(object)  # ProgressEvent
    result = Signal(object)    # ProductionPipelineResult
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

        providers: list[Any] = []
        try:
            logger.debug(f"[THREAD] Creating detector '{self._detector_name}' in worker thread")
            provider = get_registry().create(self._detector_name)
            logger.debug(f"[THREAD] Provider created: {type(provider).__name__}")

            if hasattr(provider, "confidence_threshold"):
                provider.confidence_threshold = self.config.min_confidence
                logger.debug(
                    f"[THREAD] Provider confidence set to {self.config.min_confidence}"
                )

            primary = PaddleOCRProvider(self.config.ocr.primary_model)
            verifier = PaddleOCRVLOcrProvider()
            qwen_defaults = QwenRepairConfig()
            repair = QwenRepairProvider(QwenRepairConfig(
                model_path=self.config.ocr.qwen_model_path or qwen_defaults.model_path,
                mmproj_path=self.config.ocr.qwen_mmproj_path or qwen_defaults.mmproj_path,
                server_path=self.config.ocr.qwen_server_path or qwen_defaults.server_path,
                server_url=self.config.ocr.qwen_server_url,
                server_port=self.config.ocr.qwen_server_port,
            ))
            translator = HyMT2GGUFTranslationProvider(
                model_path=self.config.translator.model_path
                or DEFAULT_HY_MT2_MODEL_PATH,
                executable_path=self.config.translator.llama_executable
                or DEFAULT_HY_LLAMA_SERVER_PATH,
                server_url=self.config.translator.server_url
                or DEFAULT_HY_MT2_SERVER_URL,
            )
            providers = [provider, primary, verifier, repair, translator]

            result: ProductionPipelineResult = analyzer.process_chapter(
                chapter_path=self.chapter_path,
                output_path=self.output_path,
                detector=provider,
                primary_ocr=primary,
                verifier_ocr=verifier,
                qwen_repair=repair,
                translator=translator,
                progress_callback=on_progress,
                cancellation_token=self._cancellation_token,
            )

            if self._cancellation_token.is_cancelled:
                self.cancelled.emit()
            else:
                self.result.emit(result)
        except CancelledError:
            self.cancelled.emit()
        except Exception as exc:
            tb = traceback.format_exc()
            self.error.emit(f"{type(exc).__name__}: {repr(exc)}\n{tb}")
        finally:
            for active_provider in reversed(providers):
                try:
                    active_provider.unload()
                except Exception as cleanup_error:
                    logger.warning(
                        "[THREAD] Provider cleanup failed for %s: %s",
                        type(active_provider).__name__,
                        cleanup_error,
                    )
