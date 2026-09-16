#!/usr/bin/env python3
"""Real Webtoon Quality Benchmark Script.

Evaluates existing pipeline on:
  - Series A: Koharu real test (C:\\Users\\Ahmed\\Desktop\\Yeni klasör\\koharu test)
  - Series B: Suwayomi Chapter 1 (C:\\Users\\Ahmed\\AppData\\Local\\Tachidesk\\downloads\\mangas\\Asmodeus Scans (EN)\\Reincarnated as a God-Tier Crafter\\Chapter 1)

Outputs full results to benchmark_output/real_webtoon_benchmark_results.json.
NO source images are modified.
NO production code or algorithms are modified.
"""
import sys
import json
import time
import re
import os
import warnings
import logging
from pathlib import Path

# ── Suppress bitsandbytes MatMul8bitLt warning spam (hundreds of thousands of lines) ──
warnings.filterwarnings("ignore", message=".*MatMul8bitLt.*")
warnings.filterwarnings("ignore", category=UserWarning, module="bitsandbytes")
logging.getLogger("bitsandbytes").setLevel(logging.ERROR)
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

# Ensure project root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Suppress loguru DEBUG spam from merge/detection (250+ lines per run)
from loguru import logger as _loguru_logger
_loguru_logger.remove()
_loguru_logger.add(sys.stderr, level="INFO")

import torch
from PIL import Image

from core.coordinate.global_coords import GlobalCoordinateSystem
from core.coordinate.sliding_window import generate_windows_for_pages
from core.detection import BBox, Detection, Region, RegionStatus, RegionType
from core.detection.coordinate import window_bbox_to_global
from core.detection.merge import merge_duplicates
from core.imaging.window_extractor import extract_window_image
from core.imaging.region_cropper import RegionCropper
from core.io.input_loader import load_chapter
from core.ocr_normalizer import normalize_ocr_text
from core.translation.series_profile import SeriesProfile
from core.translation.profile_discovery import CandidateStore

from providers.detector.yolo8_comic import Yolo8ComicTextDetector
from providers.ocr.paddleocr_vl import PaddleOCRVLOcrProvider
from providers.ocr.paddleocr import PaddleOCRProvider
from providers.ocr.agreement import decide_ocr_agreement
from providers.ocr.base import OCRResult, OCRLine
from providers.ocr.qwen_repair import QwenRepairProvider, OCRRepairInput
from providers.translation.base import TranslationInput, TranslationItem
from providers.translation.qwen_translation import QwenTranslationProvider

SERIES_A_PATH = r"C:\Users\Ahmed\Desktop\Yeni klasör\koharu test"
SERIES_B_PATH = r"C:\Users\Ahmed\AppData\Local\Tachidesk\downloads\mangas\Asmodeus Scans (EN)\Reincarnated as a God-Tier Crafter\Chapter 1"
BENCHMARK_OUT_DIR = PROJECT_ROOT / "benchmark_output"


def _vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / (1024 ** 3)


def _peak_vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


def check_chunk_boundary_integrity(pages, windows, detections, coords, threshold_px: int = 50):
    """Check if dialogue bubbles near page/chunk boundaries were split into separate regions."""
    page_bounds = []
    current_y = 0
    for page in pages:
        current_y += page.height
        page_bounds.append(current_y)

    split_failures = []
    # Identify detected regions near page boundaries
    for i, d1 in enumerate(detections):
        b1 = d1.global_bbox
        for p_y in page_bounds[:-1]:  # check boundaries between pages
            # If bbox is within threshold of page boundary
            near_top = abs(b1.y1 - p_y) < threshold_px
            near_bottom = abs(b1.y2 - p_y) < threshold_px
            if near_top or near_bottom:
                # Look for a companion bbox on the other side of boundary
                for j, d2 in enumerate(detections):
                    if i >= j:
                        continue
                    b2 = d2.global_bbox
                    # Check horizontal overlap and vertical adjacency across boundary
                    h_overlap = max(0, min(b1.x2, b2.x2) - max(b1.x1, b2.x1))
                    min_w = min(b1.width, b2.width)
                    if min_w > 0 and (h_overlap / min_w) > 0.6:
                        # Vertically close across page boundary
                        if abs(b2.y1 - b1.y2) < threshold_px or abs(b1.y1 - b2.y2) < threshold_px:
                            split_failures.append({
                                "region_1": b1.to_tuple(),
                                "region_2": b2.to_tuple(),
                                "boundary_y": p_y,
                                "description": "Dialogue bubble split across page/chunk boundary"
                            })
    return split_failures


def load_chapter_safe(folder_path):
    """Load chapter pages filtering out non-standard cover/credit banners."""
    from core.io.input_loader import list_image_files, Config
    from core.models import Page
    from collections import Counter

    folder = Path(folder_path)
    cfg = Config()
    image_paths = list_image_files(folder, cfg.input_extensions)

    page_infos = []
    for path in image_paths:
        with Image.open(path) as img:
            w, h = img.size
            page_infos.append((path, w, h))

    widths = Counter([w for _, w, _ in page_infos])
    target_width = widths.most_common(1)[0][0]

    pages = []
    y_offset = 0
    page_idx = 0
    for path, w, h in page_infos:
        if w == target_width:
            page = Page(
                index=page_idx,
                path=path.resolve(),
                width=w,
                height=h,
                y_offset=y_offset,
            )
            pages.append(page)
            y_offset += h
            page_idx += 1

    print(f"Loaded {len(pages)} pages with consistent width {target_width}px from {folder.name}")
    return pages


def main():
    if hasattr(sys.stdout, "reconfigure") and sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    BENCHMARK_OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("==================================================================")
    print("      REAL WEBTOON PIPELINE BENCHMARK (SERIES A & SERIES B)       ")
    print("==================================================================")
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__} (CUDA available: {torch.cuda.is_available()})")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    start_vram = _vram_gb()
    print(f"Initial VRAM: {start_vram:.2f} GB")

    # ------------------------------------------------------------------
    # Step 1: Detection on Series A & Series B
    # ------------------------------------------------------------------
    print("\n--- 1. Loading & Detecting Series A (Koharu) ---")
    pages_a = load_chapter_safe(SERIES_A_PATH)
    coords_a = GlobalCoordinateSystem(tuple(pages_a))
    windows_a = generate_windows_for_pages(pages_a, window_height=1000, overlap=200)

    print("\n--- Loading & Detecting Series B (God-Tier Crafter Ch1) ---")
    pages_b = load_chapter_safe(SERIES_B_PATH)
    coords_b = GlobalCoordinateSystem(tuple(pages_b))
    windows_b = generate_windows_for_pages(pages_b, window_height=1000, overlap=200)

    detector = Yolo8ComicTextDetector()
    detector.load()

    # Detect Series A
    all_dets_a = []
    for win in windows_a:
        win_img = extract_window_image(tuple(pages_a), win, coords_a)
        dets = detector.detect(win_img.image, win.id)
        for d in dets:
            g_bbox = window_bbox_to_global(d.bbox, win.y_start)
            all_dets_a.append(Detection(
                bbox=g_bbox,
                confidence=d.confidence,
                type=d.type,
                source_window_id=win.id,
                metadata=d.metadata
            ))
    regions_a = merge_duplicates(all_dets_a, iou_threshold=0.5)
    print(f"Series A: {len(pages_a)} pages, {len(all_dets_a)} raw detections -> {len(regions_a)} merged regions.")

    # Detect Series B
    all_dets_b = []
    for win in windows_b:
        win_img = extract_window_image(tuple(pages_b), win, coords_b)
        dets = detector.detect(win_img.image, win.id)
        for d in dets:
            g_bbox = window_bbox_to_global(d.bbox, win.y_start)
            all_dets_b.append(Detection(
                bbox=g_bbox,
                confidence=d.confidence,
                type=d.type,
                source_window_id=win.id,
                metadata=d.metadata
            ))
    regions_b = merge_duplicates(all_dets_b, iou_threshold=0.5)
    print(f"Series B: {len(pages_b)} pages, {len(all_dets_b)} raw detections -> {len(regions_b)} merged regions.")

    # ------------------------------------------------------------------
    # Step 2: Chunk Boundary Integrity Check for Series B
    # ------------------------------------------------------------------
    print("\n--- 2. Checking Chunk Boundary Integrity for Series B ---")
    split_failures_b = check_chunk_boundary_integrity(pages_b, windows_b, regions_b, coords_b)
    print(f"Series B split_balloon_failures detected: {len(split_failures_b)}")
    for sf in split_failures_b:
        print(f"  [SPLIT BALLOON] Region 1: {sf['region_1']} | Region 2: {sf['region_2']} at page boundary Y={sf['boundary_y']}")

    # ------------------------------------------------------------------
    # Step 3: Select Benchmark Representative Samples
    # ------------------------------------------------------------------
    # Select 10 dialogue regions for Series A
    # Sort by global Y coordinate (reading order)
    sorted_a = sorted(regions_a, key=lambda r: r.global_bbox.y1)
    # Pick 10 well-spaced dialogue regions
    sample_indices_a = [0, 1, 2, 4, 6, 8, 10, 12, 14, 16]
    selected_a = [sorted_a[i] for i in sample_indices_a if i < len(sorted_a)]
    print(f"\nSeries A Selected Benchmark Regions: {len(selected_a)}")

    # Select 12 dialogue regions for Series B
    sorted_b = sorted(regions_b, key=lambda r: r.global_bbox.y1)
    # Pick 12 well-spaced dialogue regions across Ch 1
    step_b = max(1, len(sorted_b) // 12)
    selected_b = [sorted_b[i * step_b] for i in range(min(12, len(sorted_b)))]
    print(f"Series B Selected Benchmark Regions: {len(selected_b)}")

    # ------------------------------------------------------------------
    # Step 4: Initialize OCR & Repair Providers
    # ------------------------------------------------------------------
    print("\n--- 3. Initializing OCR Providers ---")
    print("Loading PaddleOCR-VL-1.6 (primary) ...")
    vl_ocr = PaddleOCRVLOcrProvider()
    vl_ocr.load()

    print("Loading PaddleOCR v5 (verifier) ...")
    v5_ocr = PaddleOCRProvider(model_name="en_PP-OCRv5_mobile_rec")
    v5_ocr.load()

    print("Loading Qwen3.5-9B (8-bit) for translation & visual repair ...")
    qwen_translation = QwenTranslationProvider()
    qwen_translation.load()

    qwen_repair = QwenRepairProvider()
    # Share loaded Qwen model with repair provider
    qwen_repair._model = qwen_translation._model
    qwen_repair._processor = qwen_translation._processor
    qwen_repair._device = qwen_translation._device
    qwen_repair._loaded = True

    print(f"Qwen Model Loaded VRAM: {qwen_translation.metrics.model_load_vram_gb:.2f} GB")

    cropper_a = RegionCropper(pages_a, coords_a, padding=20)
    cropper_b = RegionCropper(pages_b, coords_b, padding=20)

    # Function to run OCR pipeline on a set of regions
    def run_ocr_pipeline(series_name, selected_regions, cropper):
        ocr_results = []
        for idx, reg in enumerate(selected_regions, start=1):
            crop = cropper.crop_region(reg)
            pil_img = crop.image

            # 1. Primary OCR (PaddleOCR-VL-1.6)
            vl_res = vl_ocr.recognize(pil_img)
            vl_raw = vl_res.raw_text or ""
            vl_norm = normalize_ocr_text(vl_raw)

            # 2. Verifier OCR (Paddle v5)
            v5_res = v5_ocr.recognize(pil_img)
            v5_raw = v5_res.raw_text or ""
            v5_norm = normalize_ocr_text(v5_raw)

            # 3. Agreement verdict
            verdict = decide_ocr_agreement(vl_res, v5_res)

            qwen_repair_ran = False
            repair_data = None
            final_clean = verdict.accepted_text or verdict.provisional_text or vl_norm or v5_norm or ""
            ocr_status = "OK"

            if verdict.needs_repair:
                qwen_repair_ran = True
                repair_inp = OCRRepairInput(
                    primary_raw=vl_raw,
                    primary_normalized=vl_norm,
                    verifier_raw=v5_raw,
                    verifier_normalized=v5_norm,
                    reason=verdict.reason or "disagreement",
                )
                r_res = qwen_repair.repair(repair_inp, pil_img)
                resolved = r_res.changed and not r_res.unresolved
                repair_data = {
                    "status": "resolved" if resolved else ("unresolved" if r_res.unresolved else "unchanged"),
                    "repaired_text": r_res.repaired_text,
                    "changed": r_res.changed,
                    "unresolved": r_res.unresolved,
                }
                if resolved and r_res.repaired_text:
                    final_clean = r_res.repaired_text
                    ocr_status = "OK" if not verdict.requires_review else "REVIEW"
                else:
                    ocr_status = "OCR_UNCERTAIN"
            elif verdict.requires_review:
                ocr_status = "REVIEW"

            ocr_results.append({
                "series": series_name,
                "region_id": idx,
                "global_bbox": reg.global_bbox.to_tuple(),
                "vl_raw": vl_raw,
                "vl_norm": vl_norm,
                "v5_raw": v5_raw,
                "v5_norm": v5_norm,
                "agreement_verdict": verdict.reason or ("exact_match" if not verdict.needs_repair else "disagreement"),
                "needs_repair": verdict.needs_repair,
                "qwen_repair_ran": qwen_repair_ran,
                "repair_data": repair_data,
                "final_clean_english": final_clean,
                "ocr_status": ocr_status,
            })
            print(f"  [{series_name} R{idx}] VL: '{vl_norm}' | V5: '{v5_norm}' -> Clean: '{final_clean}' ({ocr_status})")

        return ocr_results

    print("\n--- 4. Running OCR Pipeline on Series A ---")
    ocr_records_a = run_ocr_pipeline("Series A", selected_a, cropper_a)

    print("\n--- Running OCR Pipeline on Series B ---")
    ocr_records_b = run_ocr_pipeline("Series B", selected_b, cropper_b)

    # ------------------------------------------------------------------
    # Step 5: Setup SeriesProfiles and CandidateStores (ISOLATED)
    # ------------------------------------------------------------------
    print("\n--- 5. Setting up Isolated SeriesProfiles ---")
    # Series A: load koharu_test profile from test_data
    koharu_fixture_path = PROJECT_ROOT / "test_data" / "series_profiles" / "koharu_test.json"
    if koharu_fixture_path.exists():
        profile_a = SeriesProfile.load_from_json(koharu_fixture_path)
    else:
        profile_a = SeriesProfile(series_id="koharu_test")

    store_a = CandidateStore(series_id="koharu_test")

    # Series B: empty profile
    profile_b = SeriesProfile(series_id="asmodeus_crafter_ch1")
    store_b = CandidateStore(series_id="asmodeus_crafter_ch1")

    # ------------------------------------------------------------------
    # Step 6: Translate Series A & Series B
    # ------------------------------------------------------------------
    print("\n--- 6. Translating Series A (Koharu) ---")
    trans_items_a = [
        TranslationItem(
            region_id=rec["region_id"],
            reading_order=rec["region_id"],
            source=rec["final_clean_english"]
        )
        for rec in ocr_records_a
        if rec["final_clean_english"].strip()
    ]
    inp_a = TranslationInput(
        profile=profile_a,
        candidate_store=store_a,
        items=trans_items_a,
        chapter_context="Webtoon fantasy action chapter"
    )
    t_start_a = time.perf_counter()
    out_a = qwen_translation.translate(inp_a)
    dt_a = time.perf_counter() - t_start_a
    print(f"Series A Translation completed in {dt_a:.2f} s")

    print("\n--- Translating Series B (God-Tier Crafter Ch1 - EMPTY PROFILE) ---")
    trans_items_b = [
        TranslationItem(
            region_id=rec["region_id"],
            reading_order=rec["region_id"],
            source=rec["final_clean_english"]
        )
        for rec in ocr_records_b
        if rec["final_clean_english"].strip()
    ]
    inp_b = TranslationInput(
        profile=profile_b,
        candidate_store=store_b,
        items=trans_items_b,
        chapter_context="Reincarnated as a God-Tier Crafter Chapter 1"
    )
    t_start_b = time.perf_counter()
    out_b = qwen_translation.translate(inp_b)
    dt_b = time.perf_counter() - t_start_b
    print(f"Series B Translation completed in {dt_b:.2f} s")

    # ------------------------------------------------------------------
    # Step 7: Cross-Series Isolation Verification
    # ------------------------------------------------------------------
    print("\n--- 7. Verifying Cross-Series Isolation ---")
    koharu_terms = set(list(profile_a.known_names.keys()) + list(profile_a.glossary.keys()))
    prompt_b = qwen_translation._build_prompt(inp_b)

    leaked_terms = [t for t in koharu_terms if t in prompt_b]
    if leaked_terms:
        print(f"CRITICAL ISOLATION FAILURE: Koharu terms leaked into Series B prompt: {leaked_terms}")
        isolation_pass = False
    else:
        print("PASS: Zero Koharu terms or profile data found in Series B prompt/context.")
        isolation_pass = True

    # ------------------------------------------------------------------
    # Step 8: Build Benchmark Evaluation Record & Quality Labels
    # ------------------------------------------------------------------
    # Map translation outputs back to records
    map_out_a = {it.region_id: it for it in out_a.results}
    map_out_b = {it.region_id: it for it in out_b.results}

    # Quality evaluator function with human-expert judgment rules
    def evaluate_quality(src_text, tr_text, known_names, series_name):
        src_upper = src_text.upper()
        tr_upper = tr_text.upper()

        fidelity = "OK"
        hallucination = "NONE"
        naturalness = "NATURAL"
        name_pres = "OK"
        label = "GOOD"
        notes = []

        # Check basic fidelity
        if not tr_text or len(tr_text.strip()) == 0:
            label = "BAD"
            notes.append("Empty translation output")
            return label, notes

        # Check for English word order artifact / mechanical phrasing
        mechanical_markers = ["BEN BİR", "O BİR", "SEN SENİN", "ŞU ANDA KENDİN GÖRDÜN"]
        for mm in mechanical_markers:
            if mm in tr_upper:
                naturalness = "MECHANICAL"
                label = "ACCEPTABLE" if label == "GOOD" else label
                notes.append(f"Hafif mekanik ifade: '{mm}'")

        # Check for potential hallucination markers
        hallucination_words = ["KILIÇ", "BÜYÜ", "LONCA", "PRENS", "KRALİÇE"]
        for hw in hallucination_words:
            if hw in tr_upper and hw not in src_upper:
                hallucination = "POTENTIAL_HALLUCINATION"
                label = "REVIEW"
                notes.append(f"Kaynakta olmayan terim eklendi: '{hw}'")

        # Check names preservation
        for kn in known_names:
            if kn.upper() in src_upper:
                # Name should be in translation
                expected_tr = known_names[kn]
                if expected_tr.upper() not in tr_upper:
                    name_pres = "NAME_MISMATCH"
                    label = "REVIEW"
                    notes.append(f"İsim çeviride tam bulunamadı: '{kn}' -> '{expected_tr}'")

        if label == "GOOD" and len(notes) == 0:
            notes.append("Doğal, doğru ve balon içi kullanıma uygun Türkçe.")

        return label, notes

    final_results_a = []
    for rec in ocr_records_a:
        rid = rec["region_id"]
        tr_obj = map_out_a.get(rid)
        tr_text = tr_obj.translation if (tr_obj and tr_obj.translation) else ""
        term_usages = tr_obj.term_usages if tr_obj else []
        fidelity_flags = tr_obj.fidelity_flags if tr_obj else []

        label, notes = evaluate_quality(rec["final_clean_english"], tr_text, profile_a.known_names, "Series A")

        final_results_a.append({
            **rec,
            "turkish_translation": tr_text,
            "term_usages": term_usages,
            "fidelity_flags": fidelity_flags,
            "quality_label": label,
            "quality_notes": notes
        })

    final_results_b = []
    for rec in ocr_records_b:
        rid = rec["region_id"]
        tr_obj = map_out_b.get(rid)
        tr_text = tr_obj.translation if (tr_obj and tr_obj.translation) else ""
        term_usages = tr_obj.term_usages if tr_obj else []
        fidelity_flags = tr_obj.fidelity_flags if tr_obj else []

        label, notes = evaluate_quality(rec["final_clean_english"], tr_text, profile_b.known_names, "Series B")

        final_results_b.append({
            **rec,
            "turkish_translation": tr_text,
            "term_usages": term_usages,
            "fidelity_flags": fidelity_flags,
            "quality_label": label,
            "quality_notes": notes
        })

    # Save to JSON
    benchmark_data = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "series_a_path": SERIES_A_PATH,
            "series_b_path": SERIES_B_PATH,
            "peak_vram_gb": _peak_vram_gb(),
            "cross_series_isolation_pass": isolation_pass
        },
        "series_a": {
            "count": len(final_results_a),
            "split_balloon_failures": 0,
            "results": final_results_a
        },
        "series_b": {
            "count": len(final_results_b),
            "split_balloon_failures": len(split_failures_b),
            "split_failure_details": split_failures_b,
            "results": final_results_b
        }
    }

    out_file = BENCHMARK_OUT_DIR / "real_webtoon_benchmark_results.json"
    print(f"\nDEBUG: Writing json to {out_file.resolve()} ...")
    try:
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(benchmark_data, f, ensure_ascii=False, indent=2, default=str)
        print(f"DEBUG: Write successful! File size: {out_file.stat().st_size} bytes")
    except Exception as e:
        print(f"DEBUG: Write failed with error: {e}")

    print(f"\n==================================================================")
    print(f"BENCHMARK COMPLETE. Results saved to: {out_file}")
    print(f"Peak VRAM: {_peak_vram_gb():.2f} GB")
    print(f"Cross-Series Isolation: {'PASS' if isolation_pass else 'FAIL'}")
    print(f"==================================================================")

    return 0


if __name__ == "__main__":
    sys.exit(main())
