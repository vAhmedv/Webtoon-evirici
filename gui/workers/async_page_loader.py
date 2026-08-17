"""Asenkron sayfa yükleyici worker modülü (Async Page Loader).

Büyük boyutlu (14.000+ px) webtoon sayfalarını UI thread'ini kilitlemeden
arka planda QThread üzerinde açar, iskelet boyutlarını anında bildirir,
çözülen pikselleri ve küçük resimleri (thumbnail) GUI'ye sinyallerle aktarır.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import cv2
import numpy as np
from loguru import logger
from PIL import Image
from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtGui import QImage, QPixmap

from application.cancellation import CancellationToken
from core.imaging.fast_loader import get_image_dimensions
from core.models import Page


class AsyncPageLoaderWorker(QThread):
    """Bölüm sayfalarını arka planda asenkron yükleyen worker sınıfı.

    Signals:
        page_header_ready (int, int, int): (page_index, width, height) - Kanvas iskeleti için anında tetiklenir.
        page_loaded (int, QImage, object): (page_index, qimage, np_bgr_array) - Çözülen pikselleri GUI'ye aktarır.
        thumbnail_ready (int, QPixmap): (page_index, pixmap) - Kenar çubuğu küçük resimlerini günceller.
        loading_failed (int, str): (page_index, error_msg) - Yükleme hatası oluştuğunda tetiklenir.
    """

    page_header_ready = Signal(int, int, int)
    page_loaded = Signal(int, QImage, object)
    thumbnail_ready = Signal(int, QPixmap)
    loading_failed = Signal(int, str)

    def __init__(
        self,
        pages: Sequence[Page] | Sequence[Path | str],
        target_thumbnail_size: tuple[int, int] = (48, 64),
        cancellation_token: Optional[CancellationToken] = None,
        parent: Optional[Any] = None,
    ) -> None:
        super().__init__(parent)
        self.pages = list(pages)
        self.target_thumbnail_size = target_thumbnail_size
        self._cancellation_token = cancellation_token or CancellationToken()

    def request_cancel(self) -> None:
        """Yükleme işlemini iptal eder."""
        self._cancellation_token.cancel()
        self.requestInterruption()

    def is_cancelled(self) -> bool:
        """İşlemin iptal edilip edilmediğini kontrol eder."""
        return self._cancellation_token.is_cancelled or self.isInterruptionRequested()

    def run(self) -> None:
        """Arka plan iş parçacığında sayfaları sırayla işler."""
        logger.debug(f"AsyncPageLoaderWorker başlatıldı: {len(self.pages)} sayfa")

        for idx, page_item in enumerate(self.pages):
            if self.is_cancelled():
                logger.debug("AsyncPageLoaderWorker iptal edildi.")
                break

            # 1. Dosya yolu ve sayfa indeksini belirle
            if isinstance(page_item, Page):
                page_index = page_item.index
                page_path = Path(page_item.path)
            else:
                page_index = idx
                page_path = Path(page_item)

            if not page_path.exists() or not page_path.is_file():
                err_msg = f"Sayfa dosyası bulunamadı: {page_path}"
                logger.warning(err_msg)
                self.loading_failed.emit(page_index, err_msg)
                continue

            try:
                # 2. Hızlı başlık ayrıştırma (<1 ms)
                width, height = get_image_dimensions(page_path)
                self.page_header_ready.emit(page_index, width, height)

                if self.is_cancelled():
                    break

                # 3. Pikselleri güvenli bir şekilde çöz (Windows Unicode yolları destekli)
                np_bgr: Optional[np.ndarray] = None
                try:
                    file_bytes = np.fromfile(str(page_path), dtype=np.uint8)
                    np_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                except Exception as cv_err:
                    logger.debug(f"OpenCV decode hatası ({page_path.name}): {cv_err}")

                if np_bgr is None:
                    # PIL Fallback
                    with Image.open(page_path) as pil_img:
                        np_rgb_raw = np.array(pil_img.convert("RGB"))
                        np_bgr = cv2.cvtColor(np_rgb_raw, cv2.COLOR_RGB2BGR)

                if self.is_cancelled():
                    break

                # 4. QImage oluştur
                h, w, ch = np_bgr.shape
                np_rgb = cv2.cvtColor(np_bgr, cv2.COLOR_BGR2RGB)
                bytes_per_line = ch * w
                qimage = QImage(np_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()

                # 5. Thumbnail üret ve emit et
                thumb_w, thumb_h = self.target_thumbnail_size
                qimage_thumb = qimage.scaled(
                    thumb_w,
                    thumb_h,
                    Qt.KeepAspectRatioByExpanding,
                    Qt.SmoothTransformation,
                )
                pixmap_thumb = QPixmap.fromImage(qimage_thumb)
                self.thumbnail_ready.emit(page_index, pixmap_thumb)

                # 6. Çözülen sayfayı emit et
                self.page_loaded.emit(page_index, qimage, np_bgr)

            except Exception as e:
                err_str = f"Sayfa yükleme hatası (#{page_index} - {page_path.name}): {e}"
                logger.error(err_str)
                self.loading_failed.emit(page_index, err_str)

        logger.debug("AsyncPageLoaderWorker tamamlandı.")
