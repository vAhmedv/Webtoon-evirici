#!/usr/bin/env python3
"""Qwen OCR repair smoke test.

3 disagreement crop + 1 safe-agreement skip test.

Pipeline:
  Crop image -> VL-1.6 raw/normalized + Paddle v5 raw/normalized
  -> decide_ocr_agreement -> OCRVerdict
  -> (needs_repair=True) QwenRepairProvider -> repaired/unresolved

Usage:
  .venv\Scripts\python.exe scripts/qwen_repair_smoke_test.py
"""
import sys
import time
from pathlib import Path

# Ensure project root on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image

from core.coordinate.global_coords import GlobalCoordinateSystem
from core.detection import BBox, Region, RegionStatus, RegionType
from core.imaging.region_cropper import RegionCropper
from core.io.input_loader import load_chapter
from core.ocr_normalizer import normalize_ocr_text
from providers.ocr.agreement import decide_ocr_agreement
from providers.ocr.base import OCRResult
from providers.ocr.qwen_repair import QwenRepairProvider, adjudicate_ocr

SRC = r"C:\Users\Ahmed\Desktop\Yeni klasör\koharu test"
PAD = 20

# Test regions: 3 disagreements + 1 safe agreement
# Bboxes are global coordinates matching paddleocr_vl_smoke_test.py
TEST_CASES = [
    {
        "name": "LHO/LUO TIAN",
        "bbox": (235, 849, 747, 1122),
        "vl_raw": "LHO TIAN",
        "paddle_raw": "LUO TIAN",
        "category": "disagreement",
    },
    {
        "name": "HLI/HU SAN",
        "bbox": (112, 3336, 523, 3493),
        "vl_raw": "HLI SAN",
        "paddle_raw": "HU SAN",
        "category": "disagreement",
    },
    {
        "name": "PUSHOVERS/PLSHOVERS",
        "bbox": (607, 10168, 1086, 10413),
        "vl_raw": "PUSHOVERS",
        "paddle_raw": "PLSHOVERS",
        "category": "disagreement",
    },
    {
        "name": "RELAX KID",
        "bbox": (155, 8937, 591, 9119),
        "vl_raw": "RELAX, KID. YOU SAW IT YOURSELF\nJUST NOW.",
        "paddle_raw": "RELAX, KID. YOU\nSAW IT YOURSELF\nJUST NOW.",
        "category": "agreement",
    },
]


def produce_crop(pages, coords, cropper, x1, y1, x2, y2):
    region = Region(
        id=0,
        global_bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
        type=RegionType.DIALOGUE,
        detection_confidence=1.0,
        source_window_ids=(0,),
        status=RegionStatus.AUTO,
    )
    return cropper.crop_region(region)


def make_ocr_result(raw: str) -> OCRResult:
    return OCRResult(
        text=normalize_ocr_text(raw),
        confidence=0.9,
        raw_text=raw,
        lines=[],
        warnings=[],
    )


def main():
    print(f"Python: {sys.version.split()[0]}")
    print(f"Torch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")

    # 1. Load chapter + produce crops
    print(f"\nLoading chapter: {SRC}")
    pages = load_chapter(SRC)
    coords = GlobalCoordinateSystem(tuple(pages))
    cropper = RegionCropper(pages, coords, padding=PAD)
    print(f"  {len(pages)} pages, total height {coords.total_height}px")

    crops = {}
    for tc in TEST_CASES:
        x1, y1, x2, y2 = tc["bbox"]
        crop = produce_crop(pages, coords, cropper, x1, y1, x2, y2)
        crops[tc["name"]] = crop.image
        print(f"  {tc['name']}: crop {crop.image.size}")

    # 2. Construct OCRResults + verdicts
    print("\n=== Agreement / Repair Pipeline ===")
    verdicts = {}
    for tc in TEST_CASES:
        vl_result = make_ocr_result(tc["vl_raw"])
        paddle_result = make_ocr_result(tc["paddle_raw"])
        verdict = decide_ocr_agreement(vl_result, paddle_result)
        verdicts[tc["name"]] = verdict
        print(f"\n--- {tc['name']} ({tc['category']}) ---")
        print(f"  VL raw:      {tc['vl_raw']!r}")
        print(f"  Paddle raw:  {tc['paddle_raw']!r}")
        print(f"  needs_repair: {verdict.needs_repair}")
        if verdict.reason:
            print(f"  reason:       {verdict.reason}")
        if verdict.accepted_text:
            print(f"  accepted:     {verdict.accepted_text!r}")

    # 3. Agreement skip test: Qwen should be SKIPPED before model is loaded
    print("\n=== Agreement Skip Test (Qwen NOT loaded) ===")
    agree_name = "RELAX KID"
    agree_verdict = verdicts[agree_name]
    assert not agree_verdict.needs_repair, "RELAX KID should be safe agreement"
    assert agree_verdict.accepted_text is not None, "Should have accepted_text"

    # Create provider but DON'T load — skip test works even without loaded model
    unloaded_provider = QwenRepairProvider()
    assert not unloaded_provider.is_loaded, "Provider should not be loaded yet"

    outcome = adjudicate_ocr(agree_verdict, crops[agree_name], unloaded_provider)
    assert outcome.repair_result is None, "Qwen repair should NOT be called for agreement"
    assert outcome.clean_source_text == agree_verdict.accepted_text, (
        "clean_source_text should be accepted_text for safe agreement"
    )
    assert not outcome.requires_review, "Should not require review for safe agreement"
    print(f"  needs_repair={agree_verdict.needs_repair} -> Qwen SKIP (not loaded)")
    print(f"  repair_result=None: Qwen not called OK")
    print(f"  clean_source_text={outcome.clean_source_text!r} OK")

    # 4. Load Qwen repair provider (8-bit, then 4-bit fallback)
    print("\n=== Loading Qwen Repair Provider ===")
    repair_provider = QwenRepairProvider()
    try:
        t0 = time.perf_counter()
        repair_provider.load()
        load_time = time.perf_counter() - t0
        print(f"  Model loaded in {load_time:.1f}s")
        print(f"  Repair model: {repair_provider.metrics.repair_model}")
        print(f"  Model-load VRAM: {repair_provider.metrics.model_load_vram_gb:.2f} GB")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  LOAD FAIL: {e}")
        sys.exit(3)

    # Peak VRAM after load
    if torch.cuda.is_available():
        peak_after_load = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  Peak VRAM after load: {peak_after_load:.2f} GB")
        torch.cuda.reset_peak_memory_stats()

    # 5. Run Qwen repair on disagreement crops only
    # Also verify agreement case still skips even when Qwen IS loaded
    print("\n=== Qwen Repair Results ===")
    results = {}

    for tc in TEST_CASES:
        verdict = verdicts[tc["name"]]
        crop_img = crops[tc["name"]]

        if tc["category"] == "agreement":
            print(f"\n--- {tc['name']}: SKIP (safe agreement, Qwen loaded) ---")
            outcome = adjudicate_ocr(verdict, crop_img, repair_provider)
            assert outcome.repair_result is None, (
                "Qwen should NOT be called when needs_repair=False even if loaded"
            )
            print(f"  Qwen called: no (needs_repair=False) OK")
            print(f"  clean_source_text: {outcome.clean_source_text!r}")
            results[tc["name"]] = outcome
            continue

        print(f"\n--- {tc['name']}: Qwen repair ---")
        outcome = adjudicate_ocr(verdict, crop_img, repair_provider)
        results[tc["name"]] = outcome

        if outcome.repair_result:
            rm = outcome.repair_result
            print(f"  repair_model:  {rm.metadata.get('repair_model', 'N/A')}")
            print(f"  repaired_text: {rm.repaired_text!r}")
            print(f"  unresolved:    {rm.unresolved}")
            print(f"  repair_reason: {rm.metadata.get('repair_reason', 'N/A')}")
            raw = rm.metadata.get('raw_output', 'N/A')
            print(f"  raw_output:    {raw[:300]}")
        print(f"  clean_source_text: {outcome.clean_source_text!r}")
        print(f"  requires_review:   {outcome.requires_review}")

    # 6. Unload + VRAM
    repair_provider.unload()
    peak_vram = 0.0
    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / 1024**3
        print(f"\n  Peak VRAM (full run): {peak_vram:.2f} GB")
        print(f"  12 GB limit: {'OK' if peak_vram < 12 else 'EXCEEDED'}")

    # 7. Summary
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"Repair model: {repair_provider.metrics.repair_model}")
    print(f"Model-load VRAM: {repair_provider.metrics.model_load_vram_gb:.2f} GB")
    if torch.cuda.is_available():
        print(f"Peak VRAM: {peak_vram:.2f} GB")

    for tc in TEST_CASES:
        name = tc["name"]
        outcome = results.get(name)
        if outcome:
            print(f"\n  {name}:")
            print(f"    clean_source_text: {outcome.clean_source_text!r}")
            print(f"    requires_review:   {outcome.requires_review}")
            if outcome.repair_result:
                print(f"    repaired:          {outcome.repair_result.repaired_text!r}")
                print(f"    unresolved:        {outcome.repair_result.unresolved}")

    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()
