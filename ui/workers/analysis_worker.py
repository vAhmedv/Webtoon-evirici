"""Analysis worker — pipeline'ı background thread'de çalıştırır."""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

from PySide6.QtCore import QThread, Signal

from application.cancellation import CancellationToken, CancelledError
from application.chapter_analyzer import AnalysisResult, ChapterAnalyzer
from application.progress import ProgressEvent
from core.config import Config
from providers.detector.base import DetectorProvider


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
        detector: DetectorProvider,
        config: Config,
        parent: Any = None,
    ) -> None:
        super().__init__(parent)
        self.chapter_path = chapter_path
        self.output_path = output_path
        self.detector = detector
        self.config = config
        self._cancellation_token = CancellationToken()

    def request_cancel(self) -> None:
        """İptal isteği gönderir."""
        self._cancellation_token.cancel()

    @property
    def is_cancelled(self) -> bool:
        return self._cancellation_token.is_cancelled

    def run(self) -> None:
        """Thread giriş noktası."""
        analyzer = ChapterAnalyzer(self.config)

        def on_progress(event: ProgressEvent) -> None:
            self.progress.emit(event)

        try:
            result = analyzer.analyze(
                chapter_path=self.chapter_path,
                output_path=self.output_path,
                detector=self.detector,
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
            self.error.emit(f"{exc}\n{tb}")
