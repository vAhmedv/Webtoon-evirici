"""Hizli mini-bolum dogrulama (tam audit'in kucuk kardesi).

Kullanim:
    .\\.venv\\Scripts\\python.exe scripts\\fast_verify_mini.py <kaynak_dir> <cikti_dir>

Tam 25 sayfalik audit (~4.5 dk) yerine secilmis birkac sayfada ayni
uretim hattini kosturur (~1 dk cogunlukla model yukleme). Davranis
uretimi audit ile birebirdir; yalnizca metrik ozeti yazdirir.
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from application.chapter_analyzer import ChapterAnalyzer
from providers.detector.ctd import ComicTextDetector
from providers.ocr.paddleocr import PaddleOCRProvider
from providers.ocr.paddleocr_vl import PaddleOCRVLOcrProvider
from providers.ocr.qwen_repair import QwenRepairProvider
from providers.translation.hy_mt2_gguf_translation import HyMT2GGUFTranslationProvider


def main() -> None:
    source = Path(sys.argv[1])
    output = Path(sys.argv[2])
    assert source.is_dir(), f"Kaynak yok: {source}"
    assert source.resolve() != output.resolve(), "Cikti kaynakla ayni olamaz!"
    output.mkdir(parents=True, exist_ok=True)

    analyzer = ChapterAnalyzer()
    analyzer._cache.enabled = False
    t0 = time.time()
    result = analyzer.process_chapter(
        source,
        output,
        detector=ComicTextDetector(),
        primary_ocr=PaddleOCRProvider("PP-OCRv6_medium_rec"),
        verifier_ocr=PaddleOCRVLOcrProvider(),
        qwen_repair=QwenRepairProvider(),
        translator=HyMT2GGUFTranslationProvider(),
    )
    print(f"[MINI-OK] {len(result.pages)} sayfa, {time.time() - t0:.1f} sn -> {output}")


if __name__ == "__main__":
    main()
