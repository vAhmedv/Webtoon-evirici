"""OCR repair interface ve data model.

Repair katmanı TRANSLATION DEĞİL — yalnız İngilizce source transcription repair.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class OCRRepairInput:
    """OCR repair girdisi."""

    primary_raw: str
    primary_normalized: str
    verifier_raw: str
    verifier_normalized: str
    reason: str
    known_names: list[str] = field(default_factory=list)
    nearby_context: str | None = None


@dataclass(frozen=True)
class OCRRepairResult:
    """OCR repair çıktısı."""

    repaired_text: str | None
    changed: bool = False
    unresolved: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class OCRRepairProvider:
    """OCR repair provider arayüzü.

    Kontrat (Qwen repair modeli için):
    - Görüntüde olmayan kelime ekleme.
    - İngilizceyi Türkçeye çevirme.
    - Stil değiştirme.
    - Cümleyi yeniden yazma.
    - Proper name tahmin etme.
    - İki OCR aynıysa gereksiz düzeltme yapma.
    - Belirsizse `unresolved` döndür.
    """

    def repair(self, repair_input: OCRRepairInput) -> OCRRepairResult:
        raise NotImplementedError(
            "OCRRepairProvider.repair() implement edilmedi. "
            "Bu görevde yalnız kontrat tanımlanır."
        )