#!/usr/bin/env python3
"""PaddleOCR-VL-1.6 primary + Paddle v5 verifier smoke test on real crops."""
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from core.coordinate.global_coords import GlobalCoordinateSystem
from core.detection import BBox, Region, RegionStatus, RegionType
from core.imaging.region_cropper import RegionCropper
from core.io.input_loader import load_chapter
from providers.ocr.agreement import decide_ocr_agreement
from providers.ocr.paddleocr import PaddleOCRProvider
from providers.ocr.paddleocr_vl import PaddleOCRVLOcrProvider

CHAPTER = r"C:\Users\Ahmed\Desktop\Yeni klasör\koharu test"
PADDLE_REC = "en_PP-OCRv5_mobile_rec"

TEST_REGIONS = [
    {"name": "LUO TIAN", "bbox": (235, 849, 747, 1122)},
    {"name": "HU SAN", "bbox": (112, 3336, 523, 3493)},
    {"name": "PUSHOVERS", "bbox": (607, 10168, 1086, 10413)},
    {"name": "RELAX KID", "bbox": (155, 8937, 591, 9119)},
]


def main():
    print(f"Python: {sys.version.split()[0]}")
    print(f"Torch: {torch.__version__}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")

    print(f"\nLoading chapter: {CHAPTER}")
    pages = load_chapter(CHAPTER)
    coords = GlobalCoordinateSystem(tuple(pages))
    cropper = RegionCropper(pages, coords, padding=20)

    crops = []
    for t in TEST_REGIONS:
        x1, y1, x2, y2 = t["bbox"]
        region = Region(
            id=0,
            global_bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
            type=RegionType.DIALOGUE,
            detection_confidence=1.0,
            source_window_ids=(0,),
            status=RegionStatus.AUTO,
        )
        crop = cropper.crop_region(region)
        crops.append((t["name"], crop.image))
        print(f"  {t['name']}: crop {crop.image.size}")

    print(f"\nLoading PaddleOCR-VL-1.6 ...")
    vl = PaddleOCRVLOcrProvider()
    try:
        vl.load()
        load_vram = torch.cuda.memory_allocated() / 1024**2
        print(f"VL-1.6 loaded OK, VRAM after load: {load_vram:.1f} MB")
        torch.cuda.reset_peak_memory_stats()
    except Exception as e:
        print(f"VL_LOAD_FAIL: {e}")
        traceback.print_exc()
        sys.exit(3)

    print(f"\nLoading PaddleOCR {PADDLE_REC} verifier ...")
    v5 = PaddleOCRProvider(model_name=PADDLE_REC)
    try:
        v5.load()
        print("Paddle v5 loaded OK")
    except Exception as e:
        print(f"PADDLE_V5_LOAD_FAIL: {e}")
        traceback.print_exc()
        v5 = None

    print("\n" + "=" * 72)
    print("VL-1.6 vs Paddle v5 agreement results")
    print("=" * 72)

    max_vram = 0.0
    for name, img in crops:
        print(f"\n--- {name} ---")
        vl_result = None
        v5_result = None
        try:
            vl_result = vl.recognize(img)
            peak = torch.cuda.max_memory_allocated() / 1024**2
            max_vram = max(max_vram, peak)
            torch.cuda.reset_peak_memory_stats()
            print(f"  VL-1.6 raw:       {vl_result.raw_text!r}")
            print(f"  VL-1.6 canonical: {vl_result.text!r}")
            print(f"  VL-1.6 time:      {vl_result.metadata.get('inference_seconds')}s")
        except Exception as e:
            print(f"  VL-1.6 FAIL: {e}")
        if v5 is not None:
            try:
                v5_result = v5.recognize(img)
                print(f"  Paddle v5 raw:    {v5_result.raw_text!r}")
                print(f"  Paddle v5 conf:   {v5_result.confidence}")
            except Exception as e:
                print(f"  Paddle v5 FAIL: {e}")
        if vl_result is not None and v5_result is not None:
            verdict = decide_ocr_agreement(vl_result, v5_result)
            print(f"  -> Accepted: {verdict.accepted_text!r}")
            print(f"  -> Source: {verdict.source}")
            print(f"  -> Requires review: {verdict.requires_review}")
            if verdict.reason:
                print(f"  -> Reason: {verdict.reason}")

    vl.unload()
    if v5 is not None:
        v5.unload()

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"VL-1.6 load: OK")
    print(f"VL-1.6 model-load VRAM: {load_vram:.1f} MB")
    print(f"VL-1.6 max peak VRAM: {max_vram:.1f} MB")
    print(f"12 GB limit: {'OK' if max_vram < 12000 else 'EXCEEDED'}")
    print("=" * 72)


if __name__ == "__main__":
    main()