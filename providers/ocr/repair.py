"""OCR repair interface ve data model.

Repair katmanı TRANSLATION DEĞİL — yalnız İngilizce source transcription repair.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class OCRRepairInput:
    """OCR repair girdisi."""

    primary_raw: str = ""
    primary_normalized: str = ""
    verifier_raw: str = ""
    verifier_normalized: str = ""
    reason: str = ""
    known_names: list[str] = field(default_factory=list)
    nearby_context: str | None = None
    image: Any = None
    raw_text: str = ""
    confidence: float = 0.0
    context_hint: str = ""
    region_id: int | None = None
    block_id: int | None = None
    primary_text: str = ""
    primary_confidence: float = 0.0
    verifier_text: str | None = None
    verifier_confidence: float | None = None
    agreement_verdict: str = ""

    def __post_init__(self) -> None:
        # Sync primary text aliases
        if self.primary_text and not self.primary_raw:
            self.primary_raw = self.primary_text
        if self.primary_raw and not self.primary_text:
            self.primary_text = self.primary_raw
        if self.primary_raw and not self.primary_normalized:
            self.primary_normalized = self.primary_raw

        # Sync verifier text aliases
        if self.verifier_text and not self.verifier_raw:
            self.verifier_raw = self.verifier_text
        if self.verifier_raw and not self.verifier_text:
            self.verifier_text = self.verifier_raw
        if self.verifier_raw and not self.verifier_normalized:
            self.verifier_normalized = self.verifier_raw

        # Sync reason / agreement_verdict
        if self.agreement_verdict and not self.reason:
            self.reason = self.agreement_verdict
        if self.reason and not self.agreement_verdict:
            self.agreement_verdict = self.reason

        # Sync raw_text / confidence
        if not self.raw_text and self.primary_raw:
            self.raw_text = self.primary_raw
        if self.confidence == 0.0 and self.primary_confidence > 0.0:
            self.confidence = self.primary_confidence


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