"""Dummy detector — mimariyi test etmek için sabit/sentetik çıktı üretir."""

from __future__ import annotations

import random
from typing import Sequence

from core.detection import BBox, Detection, RegionType
from .base import DetectorProvider


class DummyDetector(DetectorProvider):
    """Sabit bbox'lar üreten dummy detector.

    Deterministik çıktı için seed kullanılabilir.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._seed = seed
        self._loaded = False

    def load(self) -> None:
        """Yükleme simülasyonu."""
        self._loaded = True

    def unload(self) -> None:
        """Serbest bırakma simülasyonu."""
        self._loaded = False

    @property
    def name(self) -> str:
        return "dummy"

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def detect(self, image, window_id: int) -> Sequence[Detection]:
        """Dummy tespitler üretir.

        Sabit dört kutu üretir:
        - DIAGLOGUE
        - NARRATION
        - SFX
        - WATERMARK

        Eğer seed verilmişse deterministik; aksi halde rastgele küçük
        ofsetler eklenir.

        Args:
            image: Giriş görüntüsü (PIL).
            window_id: Window kimliği.

        Returns:
            Detection listesi.
        """
        if not self._loaded:
            raise RuntimeError("DummyDetector yüklenmedi; load() çağrılmalı")

        w, h = image.size
        rng = random.Random(self._seed)

        base_boxes = [
            (int(w * 0.05), int(h * 0.05), int(w * 0.5), int(h * 0.15)),
            (int(w * 0.05), int(h * 0.20), int(w * 0.95), int(h * 0.35)),
            (int(w * 0.7), int(h * 0.40), int(w * 0.95), int(h * 0.55)),
            (int(w * 0.5), int(h * 0.70), int(w * 0.95), int(h * 0.85)),
        ]

        types = [
            RegionType.DIALOGUE,
            RegionType.NARRATION,
            RegionType.SFX,
            RegionType.WATERMARK,
        ]

        results: list[Detection] = []
        for (x1, y1, x2, y2), rtype in zip(base_boxes, types):
            # Küçük rastgele ofset
            dx = rng.randint(-5, 5)
            dy = rng.randint(-5, 5)
            x1 = max(0, x1 + dx)
            y1 = max(0, y1 + dy)
            x2 = min(w, x2 + dx)
            y2 = min(h, y2 + dy)
            if x2 <= x1 or y2 <= y1:
                continue

            if rtype == RegionType.WATERMARK:
                confidence = 0.9
            elif rtype == RegionType.SFX:
                confidence = 0.8
            elif rtype == RegionType.NARRATION:
                confidence = 0.7
            else:
                confidence = 0.6

            results.append(
                Detection(
                    bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    confidence=float(confidence),
                    type=rtype,
                    source_window_id=window_id,
                )
            )

        return results