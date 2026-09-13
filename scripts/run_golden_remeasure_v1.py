"""F2+ hızlı altın-set yeniden-ölçümü (kalıcı altyapı).

Yavaş runner'in (run_golden_baseline_v1) sorunu: her bölüm için ayrı
process + tüm modelleri baştan yükleme (~90-120 sn/bölüm çöp).
Bu runner provider'ları BİR kez yükler, 4 bölümü aynı process'te
sırayla ölçer. Davranış audit ile birebirdir (aynı process_chapter
+ aynı metrik fonksiyonları); yalnızca yükleme maliyeti ödenmez.

Girdi: 4 Asura Ch1 (terazi). Çıktı: benchmark/golden_f2_v1.json
(baseline dosyasına DOKUNULMAZ) + her bölümde e2e_audit_metrics.json.

Geçerlilik bekçisi: ilk bölüm (dungeon) sonucu dondurulmuş F1
bazıyla karşılaştırılır; tutmazsa koşu DURUR (hız-doğruluk takası yok).

KULLANIM:
    .venv\\Scripts\\python.exe scripts\\run_golden_remeasure_v1.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from application.chapter_analyzer import ChapterAnalyzer
from audit_e2e_real_chapter1 import (  # aynı metrik fonksiyonları
    count_short_dialogue_untranslated,
    summarize_final_region_states,
)
from providers.detector.ctd import ComicTextDetector
from providers.ocr.paddleocr import PaddleOCRProvider
from providers.ocr.paddleocr_vl import PaddleOCRVLOcrProvider
from providers.ocr.qwen_repair import QwenRepairProvider
from providers.translation.hy_mt2_gguf_translation import HyMT2GGUFTranslationProvider

BASE = Path(r"C:\Users\Ahmed\AppData\Local\Tachidesk\downloads\mangas\Asura Scans (EN)")
CHAPTERS = [
    ("dungeon_odyssey_ch1", BASE / "Dungeon Odyssey" / "Chapter 1"),
    ("omniscient_reader_prologue", BASE / "Omniscient Reader\u2019s Viewpoint" / "Chapter 0 - Prologue"),
    ("swordmaster_ch1", BASE / "Swordmaster\u2019s Youngest Son" / "Chapter 1"),
    ("estate_developer_ch1", BASE / "The Greatest Estate Developer" / "Chapter 1"),
]

SUMMARY_KEYS = [
    "source_page_count",
    "output_page_count",
    "text_block_count",
    "translation_eligible_blocks_count",
    "translated_blocks_count",
    "successfully_inpainted_blocks_count",
    "review_inpaint_blocks_count",
    "actually_rendered_blocks_count",
    "overflow_blocks_count",
    "short_dialogue_untranslated_count",
    "final_auto_regions",
    "final_review_regions",
    "final_skip_regions",
    "ocr_elapsed_seconds",
    "translation_elapsed_seconds",
    "inpainting_rendering_elapsed_seconds",
    "elapsed_seconds",
]


def _run_pipeline(tag: str, src: Path, shared: dict) -> float:
    out = ROOT / "audit_output" / "golden" / tag
    out.mkdir(parents=True, exist_ok=True)
    analyzer = ChapterAnalyzer()
    analyzer._cache.enabled = False
    t0 = time.time()
    analyzer.process_chapter(
        src,
        out,
        detector=shared["detector"],
        primary_ocr=shared["primary_ocr"],
        verifier_ocr=shared["verifier_ocr"],
        qwen_repair=shared["qwen_repair"],
        translator=shared["translator"],
    )
    return round(time.time() - t0, 2)


_IMG_EXTS = {".webp", ".png", ".jpg", ".jpeg"}


def _extract_metrics(tag: str, src: Path, elapsed: float | None) -> dict:
    out = ROOT / "audit_output" / "golden" / tag
    regions_data = json.loads((out / "analysis" / "regions.json").read_text(encoding="utf-8"))
    summary_data = json.loads((out / "analysis" / "summary.json").read_text(encoding="utf-8"))
    raw_regions = regions_data["regions"]
    stage = summary_data.get("stage_timings", {}) or {}
    n_src = sum(1 for f in src.iterdir() if f.is_file() and f.suffix.lower() in _IMG_EXTS)
    pages_dir = out / "pages"
    page_root = pages_dir if pages_dir.is_dir() else out
    n_out = sum(1 for f in page_root.iterdir() if f.is_file() and f.suffix.lower() in _IMG_EXTS)
    metrics = {
        "source_page_count": n_src,
        "output_page_count": n_out,
        **summarize_final_region_states(raw_regions),
        "final_auto_regions": sum(1 for r in raw_regions if r.get("status") == "auto"),
        "final_review_regions": sum(1 for r in raw_regions if r.get("status") == "review"),
        "final_skip_regions": sum(1 for r in raw_regions if r.get("status") == "skip"),
        "text_block_count": regions_data["text_blocks_count"],
        "translation_eligible_blocks_count": regions_data["translation_eligible_blocks_count"],
        "translated_blocks_count": summary_data["translated_blocks_count"],
        "successfully_inpainted_blocks_count": summary_data["inpainted_blocks_count"],
        "review_inpaint_blocks_count": summary_data["review_inpaint_blocks_count"],
        "actually_rendered_blocks_count": summary_data["rendered_blocks_count"],
        "overflow_blocks_count": summary_data["overflow_blocks_count"],
        "short_dialogue_untranslated_count": count_short_dialogue_untranslated(raw_regions),
        "ocr_elapsed_seconds": stage.get("ocr"),
        "translation_elapsed_seconds": stage.get("translation"),
        "inpainting_rendering_elapsed_seconds": stage.get("inpainting"),
        "elapsed_seconds": elapsed,
    }
    assert metrics["output_page_count"] == metrics["source_page_count"], "sayfa uyumsuzlugu!"
    metrics["gates"] = {
        "short_dialogue_untranslated_count <= 6": (
            "pass" if metrics["short_dialogue_untranslated_count"] <= 6 else "FAIL"
        ),
        "overflow_blocks_count == 0": (
            "pass" if metrics["overflow_blocks_count"] == 0 else "FAIL"
        ),
    }
    (out / "e2e_audit_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics


def main() -> None:
    for _, src in CHAPTERS:
        assert src.is_dir(), f"Kaynak yok: {src}"

    print("[FAST] providerlar yukleniyor (bir kez)...", flush=True)
    t_load = time.time()
    shared = {
        "detector": ComicTextDetector(),
        "primary_ocr": PaddleOCRProvider("PP-OCRv6_medium_rec"),
        "verifier_ocr": PaddleOCRVLOcrProvider(),
        "qwen_repair": QwenRepairProvider(),
        "translator": HyMT2GGUFTranslationProvider(),
    }
    shared["translator"].load()
    print(f"[FAST] yukleme {round(time.time() - t_load, 1)} sn", flush=True)

    baseline = json.loads((ROOT / "benchmark" / "golden_baseline_v1.json").read_text(encoding="utf-8"))
    frozen: dict = {
        "name": "golden_f2_v1",
        "created": time.strftime("%Y-%m-%d"),
        "code": "F2 sonrasi olcum (hizli runner, tek yukleme)",
        "chapters": {},
    }
    try:
        for tag, src in CHAPTERS:
            out = ROOT / "audit_output" / "golden" / tag
            metrics_path = out / "e2e_audit_metrics.json"
            if metrics_path.is_file():
                # Ölçüm hazır (önceki koşu) — hattı tekrar koşturma.
                print(f"[FAST] atlaniyor (olcum var): {tag}", flush=True)
                m = json.loads(metrics_path.read_text(encoding="utf-8"))
            elif (out / "analysis" / "regions.json").is_file():
                # Hattı bitmiş ama metriği yazılamamış koşu (çökme artığı):
                # hattı tekrar koşturmadan metriği çıkar.
                print(f"[FAST] devam (hat hazır): {tag}", flush=True)
                m = _extract_metrics(tag, src, None)
            else:
                print(f"[FAST] basliyor: {tag}", flush=True)
                elapsed = _run_pipeline(tag, src, shared)
                m = _extract_metrics(tag, src, elapsed)
            frozen["chapters"][tag] = {
                "source": str(src),
                "output": str(ROOT / "audit_output" / "golden" / tag),
                **{k: m.get(k) for k in SUMMARY_KEYS},
                "gates": m.get("gates"),
            }
            print(
                f"[FAST] bitti: {tag} blok={m['text_block_count']} cevrilmis={m['translated_blocks_count']} "
                f"basildi={m['actually_rendered_blocks_count']} overflow={m['overflow_blocks_count']} "
                f"kisacevrilmemis={m['short_dialogue_untranslated_count']} review={m['final_review_regions']} "
                f"sure={m['elapsed_seconds'] if m['elapsed_seconds'] is not None else '?'}sn",
                flush=True,
            )
            if tag == "dungeon_odyssey_ch1":
                base = baseline["chapters"][tag]
                check = ["text_block_count", "translated_blocks_count", "actually_rendered_blocks_count",
                         "overflow_blocks_count", "final_review_regions"]
                diff = {k: (base.get(k), m.get(k)) for k in check if base.get(k) != m.get(k)}
                # F2 guard'ları bilinçli olarak daha çok REVIEW üretir
                # (çevrilen/basılan/review kaymaları ÖLÇÜMDÜR, hata değil).
                # Altyapı hatası yalnız blok-sayımı değişirse vardır.
                hard = {k: v for k, v in diff.items() if k in ("text_block_count",)}
                print(f"[FAST] gecerlilik: fark={diff or 'yok'}", flush=True)
                if hard:
                    raise SystemExit(f"GECERLILIK HATASI (altyapi): {hard}")
    finally:
        try:
            shared["translator"].unload()
        except Exception:
            pass

    dest = ROOT / "benchmark" / "golden_f2_v1.json"
    dest.write_text(json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[FAST-OK] -> {dest}", flush=True)


if __name__ == "__main__":
    main()
