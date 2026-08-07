"""İşlem iptal belirteci.

Pipeline'ın çalışma sırasında iptal isteğini kontrol edebilmesi
için kullanılan thread-safe belirteç.
"""

from __future__ import annotations

import threading


class CancellationToken:
    """İptal isteği taşıyıcısı.

    Kullanım:
        token = CancellationToken()
        # ... başka thread'den:
        token.cancel()
        # ... pipeline thread'den:
        if token.is_cancelled:
            raise CancelledError()
    """

    def __init__(self) -> None:
        self._cancelled = False
        self._lock = threading.Lock()

    @property
    def is_cancelled(self) -> bool:
        """İptal istenip istenmediği."""
        with self._lock:
            return self._cancelled

    def cancel(self) -> None:
        """İptal isteği gönderir."""
        with self._lock:
            self._cancelled = True


class CancelledError(Exception):
    """Pipeline kullanıcı tarafından iptal edildi."""
