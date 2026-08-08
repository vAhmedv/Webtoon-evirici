#!/usr/bin/env python3
"""
DeepSeek-OCR-2 (native Transformers) vs PaddleOCR English v5
24-region benchmark on the same chapter.

Pipeline:
  1. Load chapter via project loader
  2. GlobalCoordinateSystem + sliding windows (config defaults)
  3. YOLOv8 comic text segmenter detection (existing provider, untouched)
  4. window→global conversion (existing core)
  5. merge_duplicates (existing core; untouched)
  6. RegionCropper crops (existing)
  7. OCR each crop twice:
       A) PaddleOCR en_PP-OCRv5_mobile_rec (onyxruntime)
       B) deepseek-community/DeepSeek-OCR-2 (native, bf16, prompt 2 winner)

Outputs:
  benchmark/deepseek_native_vs_paddle_v5.json
  benchmark/deepseek_native_vs_paddle_v5.txt
"""
import json
import os
import re
import sys
import time
import traceback
from dataclasses import replace
from difflib import SequenceMatcher
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image

from core.coordinate.global_coords import GlobalCoordinateSystem
from core.coordinate.sliding_window import generate_windows_for_pages
from core.detection import (
    window_bbox_to_global,
    window_polygon_to_global,
)
from core.detection.merge import merge_duplicates
from core.imaging.region_cropper import RegionCropper
from core.imaging.window_extractor import extract_window_image
from core.io.input_loader import load_chapter

CHAPTER = r"C:\Users\Ahmed\Desktop\Yeni klasör\koharu test"
MODEL_ID = "deepseek-community/DeepSeek-OCR-2"
PADDLE_REC = "en_PP-OCRv5_mobile_rec"
DEEPSEEK_PROMPT = "<image>\nTranscribe all visible English text exactly as written."

MIN_CONF = 0.25   # detector confidence threshold (default)
WINDOW_HEIGHT = 5000
WINDOW_OVERLAP = 1000
CROP_PADDING = 20

OUT_JSON = r"benchmark\deepseek_native_vs_paddle_v5.json"
OUT_TXT = r"benchmark\deepseek_native_vs_paddle_v5.txt"

# Hard-case reference texts (for reporting only; normalized matching used)
HARD_CASES = [
    "JUDGING BY LUO TIAN'S PERFORMANCE JUST NOW",
    "YOUNG MASTER YU, CAPTAIN GAO,\nWE NEED TO BE CAREFUL FROM HERE ON.",
    "THESE GRAY WOLF BEASTS ARE SUPPOSED TO BE ACTIVE\nIN BLACKWIND RAVINE AHEAD OF US.",
    "HU SAN, YOU'RE THE FASTEST.\nGO SCOUT THE PATH AHEAD.",
    "THE FACT THAT THEY'VE APPEARED HERE\nIS PROBABLY NOT A GOOD SIGN.",
    "CAPTAIN GAO YUAN IS A PEAK LEVEL 1 ABILITY USER...\nNO PUSHOVERS EITHER.",
    "RELAX, KID. YOU SAW IT YOURSELF JUST NOW.",
    "I'M USED TO IT.",
    "COUNTLESS SPATIAL SECRET REALMS HAVE FORMED\nALL AROUND THE WORLD.",
    "MY NAME IS LUO TIAN.\nI'M NOT AN ABILITY USER—\nI'M A SECRET REALM GUIDE.",
]

CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").upper()).strip()


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize(a), normalize(b)).ratio()


def contains_cjk(text: str) -> bool:
    return bool(CJK_RE.search(text or ""))


def obvious_repetition(text: str, max_reps: int = 3) -> bool:
    """Detect obvious repetition of a word/token >= max_reps consecutive times."""
    if not text:
        return False
    norm = normalize(text)
    tokens = norm.split()
    for tok in set(tokens):
        if len(tok) < 2:
            continue
        if tokens.count(tok) >= max_reps:
            return True
    return False


def load_deepseek():
    """Load DeepSeek-OCR-2 natively. Returns (processor, model, load_vram_mb)."""
    from transformers import AutoProcessor, AutoModelForImageTextToText

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
    model = model.to("cuda:0")
    model.eval()

    load_vram = torch.cuda.memory_allocated() / 1024**2
    torch.cuda.reset_peak_memory_stats()
    return processor, model, load_vram


def deepseek_ocr(processor, model, crop: Image.Image):
    """Run DeepSeek OCR on a crop. Returns (text, seconds, peak_vram_mb)."""
    start = time.perf_counter()
    inputs = processor(
        images=crop,
        text=DEEPSEEK_PROMPT,
        return_tensors="pt",
    )
    inputs = {
        k: v.to(torch.bfloat16).to("cuda:0") if v.dtype == torch.float32 else v.to("cuda:0")
        for k, v in inputs.items()
    }
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=256,
        )
    new_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
    text = processor.decode(new_ids[0], skip_special_tokens=True)
    seconds = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated() / 1024**2
    torch.cuda.reset_peak_memory_stats()
    return text, seconds, peak


def load_paddle():
    """Load PaddleOCR en v5. Returns provider or None."""
    from providers.ocr.paddleocr import PaddleOCRProvider

    provider = PaddleOCRProvider(model_name=PADDLE_REC)
    provider.load()
    return provider


def main():
    # ── 1. Chapter + coords + windows ────────────────────────────────────────
    print(f"Loading chapter: {CHAPTER}")
    pages = load_chapter(CHAPTER)
    coords = GlobalCoordinateSystem(tuple(pages))
    windows = generate_windows_for_pages(
        pages, window_height=WINDOW_HEIGHT, overlap=WINDOW_OVERLAP
    )
    print(f"Pages: {len(pages)}, windows: {len(windows)}")

    # ── 2. YOLO detection (existing provider, untouched) ─────────────────────
    from providers.detector.yolo8_comic import Yolo8ComicTextDetector

    detector = Yolo8ComicTextDetector(confidence_threshold=MIN_CONF)
    detector.load()
    all_detections = []
    try:
        for w in windows:
            wi = extract_window_image(tuple(pages), w, coords)
            dets = detector.detect(wi.image, w.id)
            for d in dets:
                g = window_bbox_to_global(d.bbox, w.y_start)
                meta = dict(d.metadata)
                polygon = meta.get("polygon")
                if isinstance(polygon, list) and len(polygon) > 0:
                    meta["polygon"] = window_polygon_to_global(polygon, w.y_start)
                all_detections.append(
                    replace(d, bbox=g, metadata=meta)
                )
    finally:
        detector.unload()

    # Use the canonical merge pipeline (existing, untouched)
    regions = merge_duplicates(all_detections, min_confidence=MIN_CONF)
    print(f"Detections: {len(all_detections)}, merged regions: {len(regions)}")

    if len(regions) == 0:
        print("BENCHMARK FAIL: No regions detected")
        sys.exit(5)

    # ── 3. Crop each region ──────────────────────────────────────────────────
    cropper = RegionCropper(pages, coords, padding=CROP_PADDING)
    crops = []
    for r in regions:
        try:
            c = cropper.crop_region(r)
            crops.append(c)
        except Exception as e:
            print(f"  Crop fail region {r.id}: {e}")
            crops.append(None)

    # ── 4. PaddleOCR en v5 ───────────────────────────────────────────────────
    print(f"\nLoading PaddleOCR {PADDLE_REC} …")
    paddle = None
    try:
        paddle = load_paddle()
        print("PaddleOCR loaded OK")
    except Exception as e:
        print(f"PADDLE_LOAD_FAIL: {e}")
        traceback.print_exc()

    paddle_results = {}
    if paddle is not None:
        for i, (r, c) in enumerate(zip(regions, crops), start=1):
            if c is None:
                paddle_results[r.id] = {"raw_text": "", "confidence": None, "warnings": ["crop_failed"], "empty": True}
                continue
            print(f"  Paddle region {i}/{len(regions)} …", end=" ")
            try:
                res = paddle.recognize(c.image, region_bbox=r.global_bbox)
                paddle_results[r.id] = {
                    "raw_text": res.raw_text,
                    "confidence": res.confidence if res.lines else None,
                    "warnings": res.warnings,
                    "empty": not res.text.strip(),
                }
                print("OK" if res.text.strip() else "EMPTY")
            except Exception as e:
                print(f"FAIL: {e}")
                paddle_results[r.id] = {"raw_text": "", "confidence": None, "warnings": [f"error: {e}"], "empty": True}
        try:
            paddle.unload()
        except Exception:
            pass

    # ── 5. DeepSeek-OCR-2 (native, winning prompt) ───────────────────────────
    print(f"\nLoading {MODEL_ID} …")
    try:
        processor, model, load_vram = load_deepseek()
        print(f"DeepSeek loaded, model-load VRAM: {load_vram:.1f} MB")
    except Exception as e:
        print(f"DEEPSEEK_LOAD_FAIL: {e}")
        traceback.print_exc()
        sys.exit(6)

    deepseek_results = {}
    max_deepseek_vram = 0.0
    for i, (r, c) in enumerate(zip(regions, crops), start=1):
        if c is None:
            deepseek_results[r.id] = {
                "raw_text": "",
                "inference_seconds": None,
                "peak_vram_mb": None,
                "empty": True,
                "contains_cjk": False,
                "obvious_repetition": False,
            }
            continue
        print(f"  DeepSeek region {i}/{len(regions)} …", end=" ")
        try:
            text, secs, peak = deepseek_ocr(processor, model, c.image)
            max_deepseek_vram = max(max_deepseek_vram, peak)
            deepseek_results[r.id] = {
                "raw_text": text,
                "inference_seconds": round(secs, 3),
                "peak_vram_mb": round(peak, 1),
                "empty": not (text or "").strip(),
                "contains_cjk": contains_cjk(text),
                "obvious_repetition": obvious_repetition(text),
            }
            print("OK" if (text or "").strip() else "EMPTY")
        except Exception as e:
            print(f"FAIL: {e}")
            deepseek_results[r.id] = {
                "raw_text": "",
                "inference_seconds": None,
                "peak_vram_mb": None,
                "empty": True,
                "contains_cjk": False,
                "obvious_repetition": False,
                "error": str(e),
            }

    del model
    torch.cuda.empty_cache()

    # ── 6. Build result records ──────────────────────────────────────────────
    records = []
    for r in regions:
        bbox = r.global_bbox
        crop = crops[r.id]
        cw = crop.image.width if crop else None
        ch = crop.image.height if crop else None
        p = paddle_results.get(r.id, {})
        d = deepseek_results.get(r.id, {})
        records.append({
            "region_id": r.id,
            "bbox": {"x1": bbox.x1, "y1": bbox.y1, "x2": bbox.x2, "y2": bbox.y2},
            "crop_width": cw,
            "crop_height": ch,
            "detection_confidence": r.detection_confidence,
            "paddle": {
                "raw_text": p.get("raw_text", ""),
                "confidence": p.get("confidence"),
                "warnings": p.get("warnings", []),
                "empty": p.get("empty", True),
            },
            "deepseek": {
                "raw_text": d.get("raw_text", ""),
                "inference_seconds": d.get("inference_seconds"),
                "peak_vram_mb": d.get("peak_vram_mb"),
                "empty": d.get("empty", True),
                "contains_cjk": d.get("contains_cjk", False),
                "obvious_repetition": d.get("obvious_repetition", False),
            },
        })

    # ── 7. Hard cases ────────────────────────────────────────────────────────
    hard_case_results = []
    for ref in HARD_CASES:
        ref_norm = normalize(ref)
        best = None
        best_sim = -1.0
        for rec in records:
            for key in ("paddle", "deepseek"):
                txt = rec[key].get("raw_text", "")
                sim = similarity(txt, ref)
                if sim > best_sim:
                    best_sim = sim
                    best = (rec["region_id"], key, txt)
        hard_case_results.append({
            "reference": ref,
            "matched_region": best[0] if best else None,
            "best_similarity": round(best_sim, 4),
            "paddle_raw": next(
                (r["paddle"]["raw_text"] for r in records if r["region_id"] == (best[0] if best else -1)),
                ""
            ),
            "deepseek_raw": next(
                (r["deepseek"]["raw_text"] for r in records if r["region_id"] == (best[0] if best else -1)),
                ""
            ),
        })

    # ── 8. Problematic counts ────────────────────────────────────────────────
    paddle_problematic = sum(
        1 for r in records
        if r["paddle"]["empty"]
        or r["paddle"]["raw_text"].strip() == ""
    )
    deepseek_problematic = sum(
        1 for r in records
        if r["deepseek"]["empty"]
        or r["deepseek"]["contains_cjk"]
        or r["deepseek"]["obvious_repetition"]
        or r["deepseek"]["raw_text"].strip() == ""
    )

    # ── 9. Write JSON ──────────────────────────────────────────────────────
    benchmark = {
        "model": {
            "deepseek": MODEL_ID,
            "paddle": PADDLE_REC,
            "deepseek_prompt": DEEPSEEK_PROMPT,
        },
        "chapter": CHAPTER,
        "crop_padding": CROP_PADDING,
        "detector": {
            "name": "YOLOv8 Comic Text Segmenter",
            "min_confidence": MIN_CONF,
            "window_height": WINDOW_HEIGHT,
            "window_overlap": WINDOW_OVERLAP,
        },
        "deepseek_model_load_vram_mb": round(load_vram, 1),
        "deepseek_max_peak_vram_mb": round(max_deepseek_vram, 1),
        "region_count": len(records),
        "paddle_problematic": paddle_problematic,
        "deepseek_problematic": deepseek_problematic,
        "regions": records,
        "hard_cases": hard_case_results,
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(benchmark, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {OUT_JSON}")

    # ── 10. Write TXT ──────────────────────────────────────────────────────
    lines = []
    lines.append("=" * 72)
    lines.append("DeepSeek-OCR-2 (native) vs PaddleOCR English v5 — 24-region benchmark")
    lines.append("=" * 72)
    lines.append(f"DeepSeek: {MODEL_ID}")
    lines.append(f"Paddle:   {PADDLE_REC}")
    lines.append(f"Chapter:  {CHAPTER}")
    lines.append(f"Regions:  {len(records)}")
    lines.append(f"Paddle problematic: {paddle_problematic}/{len(records)}")
    lines.append(f"DeepSeek problematic: {deepseek_problematic}/{len(records)}")
    lines.append(f"DeepSeek model-load VRAM: {load_vram:.1f} MB")
    lines.append(f"DeepSeek max peak VRAM: {max_deepseek_vram:.1f} MB")
    lines.append("")
    lines.append("Per-region comparison:")
    lines.append("-" * 72)
    for rec in records:
        lines.append(f"Region {rec['region_id']} bbox={rec['bbox']['x1']},{rec['bbox']['y1']},{rec['bbox']['x2']},{rec['bbox']['y2']} "
                     f"crop={rec['crop_width']}x{rec['crop_height']}")
        lines.append(f"  Paddle:   {rec['paddle']['raw_text']!r} conf={rec['paddle']['confidence']}")
        lines.append(f"  DeepSeek: {rec['deepseek']['raw_text']!r} ({rec['deepseek']['inference_seconds']}s, "
                     f"{rec['deepseek']['peak_vram_mb']}MB)"
                     f"{' CJK' if rec['deepseek']['contains_cjk'] else ''}"
                     f"{' REPETITION' if rec['deepseek']['obvious_repetition'] else ''}"
                     f"{' EMPTY' if rec['deepseek']['empty'] else ''}")
    lines.append("")
    lines.append("Hard cases:")
    lines.append("-" * 72)
    for hc in hard_case_results:
        lines.append(f"{hc['reference']!r}")
        lines.append(f"  -> matched region {hc['matched_region']} (sim {hc['best_similarity']:.4f})")
        lines.append(f"     Paddle:   {hc['paddle_raw']!r}")
        lines.append(f"     DeepSeek: {hc['deepseek_raw']!r}")
    lines.append("")
    lines.append("Overall OCR quality: MIXED (see per-region raw outputs)")

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {OUT_TXT}")

    # ── 11. Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Regions: {len(records)}")
    print(f"Paddle problematic: {paddle_problematic}/{len(records)}")
    print(f"DeepSeek problematic: {deepseek_problematic}/{len(records)}")
    print(f"DeepSeek model-load VRAM: {load_vram:.1f} MB")
    print(f"DeepSeek max peak VRAM: {max_deepseek_vram:.1f} MB")
    print("Benchmark files:")
    print(f"  {OUT_JSON}")
    print(f"  {OUT_TXT}")

    # Hard case side-by-side print
    print("\nHard cases:")
    for hc in hard_case_results:
        print(f"\n{hc['reference']!r}")
        print(f"  -> region {hc['matched_region']}")
        print(f"  Paddle:   {hc['paddle_raw']!r}")
        print(f"  DeepSeek: {hc['deepseek_raw']!r}")

    return 0


if __name__ == "__main__":
    sys.exit(main())