"""İlerleme olayı modeli.

Pipeline ilerlemesini UI'a bildirmek için kullanılan basit veri sınıfı.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ProgressEvent:
    """Tek bir ilerleme olayı.

    Attributes:
        stage: Pipeline aşaması (ör. "Detecting", "Merging regions").
        current: Aşama içinde ilerleme sayacı (1 tabanlı).
        total: Aşama içindeki toplam adım.
        message: Ek açıklama.
        percent: 0.0-1.0 arası genel ilerleme yüzdesi (tahmini).
    """

    stage: str
    current: int = 0
    total: int = 0
    message: str = ""
    percent: float = 0.0
