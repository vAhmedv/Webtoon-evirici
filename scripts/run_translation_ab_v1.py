"""IS 4.2 — Dondurulmuş kör seti tek modelle çevir, sonucu tarihli kaydet.

KULLANIM:
    .venv\Scripts\python.exe scripts/run_translation_ab_v1.py hy_mt2
    .venv\Scripts\python.exe scripts/run_translation_ab_v1.py translategemma
    .venv\Scripts\python.exe scripts/run_translation_ab_v1.py gemini

GİRDİ: benchmark/translation_ab_v1.json (değişmez kör set).
ÇIKTI: benchmark/results/translation_ab_v1/<model>_YYYYMMDD-HHMM.json
  [{block_id, source, translation}] + meta {model, glossary: []}.

Adillik: iki modele de glossary'siz HAM kaynak verilir (üretim kilidi kapalı).
Provider kendi llama-server'ını yönetir (managed); tek model koşar (12GB tavan).
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from providers.translation.base import TranslationInput, TranslationItem

FROZEN = ROOT / "benchmark" / "translation_ab_v1.json"
OUT_DIR = ROOT / "benchmark" / "results" / "translation_ab_v1"


def _provider(name: str):
    if name == "hy_mt2":
        from providers.translation.hy_mt2_gguf_translation import (
            HyMT2GGUFTranslationProvider,
        )

        return HyMT2GGUFTranslationProvider(), "hy_mt2"
    if name == "translategemma":
        from providers.translation.translategemma_gguf_translation import (
            TranslateGemmaGGUFTranslationProvider,
        )

        return TranslateGemmaGGUFTranslationProvider(), "translategemma"
    if name == "gemini":
        import os

        from providers.translation.gemini_translation import (
            GeminiTranslationProvider,
        )

        # Prob kanıtı (2026-09-13): 3.5-flash bu key'de boş dönüyor,
        # 2.5-flash 404 (kapanmış). Çalışan: gemini-3.1-flash-lite, temp 0.0.
        return (
            GeminiTranslationProvider(
                api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "",
                model_name="gemini-3.1-flash-lite",
                temperature=0.0,
            ),
            "gemini",
        )
    raise SystemExit(f"Bilinmeyen model: {name} (hy_mt2 | translategemma | gemini)")


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Kullanım: run_translation_ab_v1.py <hy_mt2|translategemma>")
    provider, model = _provider(sys.argv[1])
    assert FROZEN.is_file(), f"Kör set yok: {FROZEN} (önce freeze_translation_ab_v1.py)"
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    items_in = frozen["items"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    provider.load()
    try:
        t0 = time.time()
        out = provider.translate(
            TranslationInput(
                items=[
                    TranslationItem(region_id=e["block_id"], source=e["source"])
                    for e in items_in
                ],
                glossary=[],
            )
        )
        elapsed = time.time() - t0
    finally:
        provider.unload()

    by_id = {r.region_id: (r.translation or "") for r in out.results}
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    out_path = OUT_DIR / f"{model}_{stamp}.json"
    out_path.write_text(
        json.dumps(
            {
                "model": model,
                "frozen_set": "translation_ab_v1",
                "created": stamp,
                "elapsed_seconds": round(elapsed, 1),
                "glossary": [],
                "results": [
                    {
                        "block_id": e["block_id"],
                        "source": e["source"],
                        "translation": by_id.get(e["block_id"], ""),
                    }
                    for e in items_in
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    empty = sum(1 for e in items_in if not by_id.get(e["block_id"]))
    print(f"[AB-RUN] {model}: {len(items_in)} balon, {elapsed:.1f} sn, boş={empty} -> {out_path}")


if __name__ == "__main__":
    main()
