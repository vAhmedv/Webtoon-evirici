"""Production pipeline worker running all heavy providers off the GUI thread."""

from __future__ import annotations

import os
import threading
import traceback
from pathlib import Path
from typing import Any

from PySide6.QtCore import QThread, Signal

from application.cancellation import CancellationToken, CancelledError
from application.chapter_analyzer import ChapterAnalyzer, ProductionPipelineResult
from application.progress import ProgressEvent
from core.config import Config, validate_config
from loguru import logger
from providers.detector.registry import get_registry
from providers.ocr.paddleocr import PaddleOCRProvider
from providers.ocr.paddleocr_vl import PaddleOCRVLOcrProvider
from providers.ocr.registry import get_ocr_registry, resolve_ocr_provider_name
from providers.translation.base import TranslationProvider
from providers.translation.hy_mt2_gguf_translation import HyMT2GGUFTranslationProvider
from providers.translation.hy_mt2_gguf_translation import (
    DEFAULT_HY_MT2_MODEL_PATH,
    DEFAULT_LLAMA_SERVER_PATH as DEFAULT_HY_LLAMA_SERVER_PATH,
    DEFAULT_HY_MT2_SERVER_URL,
)


def _create_primary_ocr(config: Config, ocr_name_override: str | None = None):
    """Config + override'e göre primary OCR kur; registry'yi bypass etme.

    Öncelik: explicit ocr_name > config.ocr.provider > primary_model default.
    PaddleOCR ailesi için config.ocr.primary_model model adı olarak geçirilir.
    """
    from providers.ocr.base import OCRProvider

    requested = ocr_name_override or config.ocr.provider
    canonical = resolve_ocr_provider_name(requested) if requested else None

    if canonical in (None, "PaddleOCR-PP-OCRv6"):
        # Varsayılan yol: model adıyla direkt kur (registry factory argsız).
        return PaddleOCRProvider(config.ocr.primary_model or "PP-OCRv6_medium_rec")
    try:
        provider: OCRProvider = get_ocr_registry().create(canonical)
        # Registry PaddleOCR'u default modelle kurar; config farklı model
        # istiyorsa model adını üzerine yaz.
        if isinstance(provider, PaddleOCRProvider) and config.ocr.primary_model:
            provider._model_name = config.ocr.primary_model
        logger.info(f"Using primary OCR from registry: {canonical}")
        return provider
    except KeyError:
        logger.warning(
            f"Unknown primary OCR {requested!r}, falling back to "
            f"PaddleOCRProvider({config.ocr.primary_model})"
        )
        return PaddleOCRProvider(config.ocr.primary_model or "PP-OCRv6_medium_rec")


def _create_verifier_ocr(config: Config):
    """Verifier OCR kur; bilinmeyen isimde safe fallback."""
    requested = config.ocr.verifier_provider or "PaddleOCR-VL-1.6"
    canonical = resolve_ocr_provider_name(requested) or "PaddleOCR-VL-1.6"
    try:
        verifier = get_ocr_registry().create(canonical)
        logger.info(f"Using verifier OCR from registry: {canonical}")
        return verifier
    except KeyError:
        logger.warning(f"Unknown verifier OCR {requested!r}, using PaddleOCRVLOcrProvider")
        return PaddleOCRVLOcrProvider()


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
            problems = validate_config(self.config)
            if problems:
                logger.warning(f"[THREAD] Config issues: {problems}")

            detector_name = self._detector_name or self.config.detector.provider
            logger.debug(f"[THREAD] Creating detector '{detector_name}' in worker thread")
            try:
                provider = get_registry().create(detector_name)
            except KeyError as e:
                available = ", ".join(sorted(get_registry().list_providers())) or "<none>"
                raise RuntimeError(f"{e}. Available detectors: {available}") from e
            logger.debug(f"[THREAD] Provider created: {type(provider).__name__}")

            if self.config.detector.threshold is not None and hasattr(
                provider, "confidence_threshold"
            ):
                provider.confidence_threshold = self.config.detector.threshold
                logger.debug(
                    f"[THREAD] Provider confidence set to {self.config.detector.threshold}"
                )

            primary = _create_primary_ocr(self.config, self._ocr_name)
            verifier = _create_verifier_ocr(self.config)
            gemini_key = (
                self.config.translator.gemini_api_key
                or os.environ.get("GEMINI_API_KEY")
                or os.environ.get("GOOGLE_API_KEY")
            )
            translator: TranslationProvider
            # GUI yedeği kapatıldı (kör-oy hükmü: Gemini birincilik ve
            # hakemlikten çıktı) — anahtar olsa bile yerel Hy-MT2 koşar.
            if gemini_key:
                logger.warning("GUI Gemini yedeği kapalı; yerel Hy-MT2 kullanılıyor.")
            logger.info("Using Local Hy-MT2 GGUF Translation Provider")
            translator = HyMT2GGUFTranslationProvider(
                    model_path=self.config.translator.model_path
                    or DEFAULT_HY_MT2_MODEL_PATH,
                    executable_path=self.config.translator.llama_executable
                    or DEFAULT_HY_LLAMA_SERVER_PATH,
                    server_url=self.config.translator.server_url
                    or DEFAULT_HY_MT2_SERVER_URL,
                )
            providers = [provider, primary, verifier, translator]

            result: ProductionPipelineResult = analyzer.process_chapter(
                chapter_path=self.chapter_path,
                output_path=self.output_path,
                detector=provider,
                primary_ocr=primary,
                verifier_ocr=verifier,
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
            logger.error(f"[THREAD] AnalysisWorker failed: {tb}")
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
