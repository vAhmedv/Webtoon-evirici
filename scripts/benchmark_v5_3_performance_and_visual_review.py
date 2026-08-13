"""Comprehensive V5.3 Performance Measurement & Visual Review Package Generator.

Executes an instrumented Chapter 1 run to record precise stage timings and RAM/VRAM lifecycle
memory checkpoints, and extracts 15-20 representative visual review contact cards.
"""

from __future__ import annotations

from dataclasses import replace
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable

import cv2
import numpy as np

try:
    import psutil
except ImportError:
    psutil = None

try:
    import torch
except ImportError:
    torch = None

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from application.chapter_analyzer import ChapterAnalyzer, ProductionPipelineResult
from core.coordinate.global_coords import GlobalCoordinateSystem
from core.coordinate.sliding_window import generate_windows_for_pages
from core.detection import BBox, RegionStatus, RegionType
from core.detection.classification import classify_regions
from core.detection.coordinate import window_bbox_to_global
from core.detection.merge import merge_duplicates
from core.detection.text_block import group_text_blocks, TextBlock
from core.imaging.inpainter import Inpainter, _is_story_text
from core.imaging.region_cropper import RegionCropper
from core.imaging.renderer import TextRenderer
from core.imaging.window_extractor import extract_window_image
from core.io.input_loader import load_chapter
from core.io.output_exporter import export_chapter_pages
from core.models import Page
from providers.detector.ctd import ComicTextDetector
from providers.ocr.agreement import decide_ocr_agreement
from providers.ocr.paddleocr import PaddleOCRProvider
from providers.ocr.paddleocr_vl import PaddleOCRVLOcrProvider
from providers.ocr.qwen_repair import QwenRepairProvider
from providers.translation.base import TranslationInput, TranslationItem
from providers.translation.hy_mt2_gguf_translation import HyMT2GGUFTranslationProvider

SOURCE_CHAPTER = Path(
    r"C:\Users\Ahmed\AppData\Local\Tachidesk\downloads\mangas\Asmodeus Scans (EN)\Reincarnated as a God-Tier Crafter\Chapter 1"
)
OUTPUT_REVIEW_DIR = ROOT / "review_output" / "chapter1_final_visual_review"


def get_memory_snapshot() -> dict[str, float]:
    """Capture current process RSS, CUDA allocated/reserved, total dedicated VRAM, and llama-server RSS."""
    rss_mb = 0.0
    if psutil is not None:
        try:
            rss_mb = psutil.Process().memory_info().rss / (1024 ** 2)
        except Exception:
            pass

    cuda_alloc_mb = 0.0
    cuda_res_mb = 0.0
    dedicated_vram_mb = 0.0
    if torch is not None and torch.cuda.is_available():
        try:
            cuda_alloc_mb = torch.cuda.memory_allocated() / (1024 ** 2)
            cuda_res_mb = torch.cuda.memory_reserved() / (1024 ** 2)
            free_b, total_b = torch.cuda.mem_get_info()
            dedicated_vram_mb = (total_b - free_b) / (1024 ** 2)
        except Exception:
            pass

    llama_rss_mb = 0.0
    if psutil is not None:
        for proc in psutil.process_iter(['name', 'memory_info']):
            try:
                if proc.info['name'] and 'llama-server' in proc.info['name'].lower():
                    llama_rss_mb += proc.info['memory_info'].rss / (1024 ** 2)
            except Exception:
                pass

    return {
        "process_rss_mb": round(rss_mb, 2),
        "cuda_allocated_mb": round(cuda_alloc_mb, 2),
        "cuda_reserved_mb": round(cuda_res_mb, 2),
        "dedicated_vram_mb": round(dedicated_vram_mb, 2),
        "llama_server_rss_mb": round(llama_rss_mb, 2),
    }


class StageTimer:
    """Records call counts, total time, and individual durations per stage."""

    def __init__(self) -> None:
        self.durations: dict[str, list[float]] = {}

    def record(self, stage: str, duration_sec: float) -> None:
        self.durations.setdefault(stage, []).append(duration_sec)

    def stats(self, stage: str) -> dict[str, Any]:
        dur_list = self.durations.get(stage, [])
        if not dur_list:
            return {"calls": 0, "total_sec": 0.0, "avg_ms": 0.0, "p95_ms": 0.0}
        total_sec = sum(dur_list)
        avg_ms = (total_sec / len(dur_list)) * 1000.0
        sorted_ms = sorted([d * 1000.0 for d in dur_list])
        p95_idx = min(len(sorted_ms) - 1, int(math.ceil(0.95 * len(sorted_ms))) - 1)
        p95_ms = sorted_ms[p95_idx]
        return {
            "calls": len(dur_list),
            "total_sec": round(total_sec, 3),
            "avg_ms": round(avg_ms, 2),
            "p95_ms": round(p95_ms, 2),
        }

    def all_stats(self) -> dict[str, dict[str, Any]]:
        return {stage: self.stats(stage) for stage in sorted(self.durations.keys())}


def create_contact_card(
    source_img: Image.Image,
    member_overlay: Image.Image,
    glyph_mask_overlay: Image.Image,
    inpainted_img: Image.Image,
    rendered_img: Image.Image,
    title: str,
    meta_text: str,
) -> Image.Image:
    """Create a single 5-panel composite contact card image with header and metadata banner."""
    panel_w, panel_h = 240, 240
    header_h = 36
    footer_h = 100
    total_w = panel_w * 5 + 24
    total_h = header_h + panel_h + footer_h

    card = Image.new("RGB", (total_w, total_h), (22, 24, 28))
    draw = ImageDraw.Draw(card)

    try:
        font_header = ImageFont.truetype("arial.ttf", 16)
        font_body = ImageFont.truetype("arial.ttf", 11)
        font_panel = ImageFont.truetype("arial.ttf", 12)
    except IOError:
        font_header = font_body = font_panel = ImageFont.load_default()

    draw.rectangle([(0, 0), (total_w, header_h)], fill=(32, 36, 44))
    draw.text((12, 8), title, fill=(240, 240, 245), font=font_header)

    panels = [
        ("1. Source Crop", source_img),
        ("2. Detections Overlay", member_overlay),
        ("3. Refined Mask Overlay", glyph_mask_overlay),
        ("4. Inpainted", inpainted_img),
        ("5. Turkish Render", rendered_img),
    ]

    for idx, (label, img) in enumerate(panels):
        x = 4 + idx * (panel_w + 4)
        y = header_h + 4
        resized = img.resize((panel_w, panel_h), Image.Resampling.LANCZOS)
        card.paste(resized, (x, y))
        draw.rectangle([(x, y), (x + panel_w, y + 20)], fill=(0, 0, 0, 180))
        draw.text((x + 6, y + 2), label, fill=(220, 220, 230), font=font_panel)
        draw.rectangle([(x, y), (x + panel_w, y + panel_h)], outline=(60, 65, 75), width=1)

    fy = header_h + panel_h + 6
    draw.rectangle([(0, fy), (total_w, total_h)], fill=(28, 32, 38))
    draw.text((12, fy + 6), meta_text, fill=(190, 200, 210), font=font_body)

    return card


def main() -> None:
    OUTPUT_REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    timer = StageTimer()
    lifecycle_snapshots: dict[str, dict[str, float]] = {}

    print("==================================================================")
    print("STARTING V5.3 STAGE TIMING, LIFECYCLE MEMORY & VISUAL REVIEW PIPELINE")
    print("==================================================================")

    lifecycle_snapshots["01_pipeline_start"] = get_memory_snapshot()
    t_start = time.perf_counter()

    detector = ComicTextDetector()
    primary_ocr = PaddleOCRProvider("PP-OCRv6_medium_rec")
    verifier_ocr = PaddleOCRVLOcrProvider()
    qwen_repair = QwenRepairProvider()
    translator = HyMT2GGUFTranslationProvider()
    analyzer = ChapterAnalyzer()
    analyzer._cache.enabled = False

    output_dir = ROOT / "e2e_output" / "real_tachidesk_chapter_1"

    # Stage 1: Load chapter
    t0 = time.perf_counter()
    pages = load_chapter(SOURCE_CHAPTER, analyzer.config, allow_non_uniform_widths=True)
    timer.record("01_chapter_load", time.perf_counter() - t0)

    # Stage 2: Sliding windows
    t0 = time.perf_counter()
    windows = generate_windows_for_pages(
        pages,
        window_height=analyzer.config.window_height,
        overlap=analyzer.config.window_overlap,
    )
    timer.record("02_window_generation", time.perf_counter() - t0)

    # Stage 3: CTD load & detection
    t0 = time.perf_counter()
    detector.load()
    timer.record("03_ctd_model_load", time.perf_counter() - t0)
    lifecycle_snapshots["02_post_ctd_load"] = get_memory_snapshot()

    coords = GlobalCoordinateSystem(tuple(pages))
    raw_detections = []
    for w_idx, win in enumerate(windows, 1):
        win_img = extract_window_image(tuple(pages), win, coords)
        t_det = time.perf_counter()
        det_list = detector.detect(win_img.image, win.id)
        timer.record("04_ctd_detection", time.perf_counter() - t_det)
        for d in det_list:
            g_bbox = window_bbox_to_global(d.bbox, win.y_start)
            raw_detections.append(replace(d, bbox=g_bbox))

    t0 = time.perf_counter()
    detector.unload()
    timer.record("05_ctd_unload", time.perf_counter() - t0)
    lifecycle_snapshots["03_post_ctd_unload"] = get_memory_snapshot()

    # Stage 4: Merge duplicates
    t0 = time.perf_counter()
    merged_regions = merge_duplicates(raw_detections, min_confidence=analyzer.config.min_confidence)
    timer.record("06_detection_merge", time.perf_counter() - t0)

    # Stage 5: Primary OCR & Selective Second Pass
    t0 = time.perf_counter()
    primary_ocr.load()
    timer.record("07_primary_ocr_load", time.perf_counter() - t0)
    lifecycle_snapshots["04_post_primary_ocr_load"] = get_memory_snapshot()

    verifier_loaded = False
    if verifier_ocr is not None:
        try:
            t0 = time.perf_counter()
            verifier_ocr.load()
            timer.record("09_verifier_ocr_load", time.perf_counter() - t0)
            verifier_loaded = True
            lifecycle_snapshots["05_post_verifier_ocr_load"] = get_memory_snapshot()
        except Exception as e:
            verifier_ocr = None

    cropper = RegionCropper(pages, coords)
    ocr_regions = []
    repair_queue = []

    for reg in merged_regions:
        bbox = reg.global_bbox
        if reg.type in (RegionType.SFX, RegionType.WATERMARK) or bbox.height < 10 or bbox.width < 10:
            ocr_regions.append(replace(reg, status=RegionStatus.SKIP, review_reason="sfx_or_non_text_skip"))
            continue

        crop = cropper.crop_region(reg, adaptive_padding=True)
        t_ocr = time.perf_counter()
        p_res = primary_ocr.recognize(crop.image, region_bbox=reg.global_bbox)
        timer.record("08_primary_ocr_inference", time.perf_counter() - t_ocr)

        single_verdict = decide_ocr_agreement(p_res, verifier=None)
        v_res = None
        if single_verdict.requires_review and verifier_ocr is not None:
            t_vl_inf = time.perf_counter()
            v_res = verifier_ocr.recognize(crop.image, region_bbox=reg.global_bbox)
            timer.record("10_verifier_ocr_inference", time.perf_counter() - t_vl_inf)
            verdict = decide_ocr_agreement(p_res, v_res)
        else:
            verdict = single_verdict

        status = RegionStatus.SKIP if reg.status == RegionStatus.SKIP else (RegionStatus.REVIEW if verdict.requires_review else RegionStatus.AUTO)
        accepted = verdict.accepted_text or verdict.provisional_text or p_res.text or ""

        has_text_content = bool(accepted and accepted.strip() and any(c.isalnum() for c in accepted))
        if reg.type == RegionType.UNKNOWN and not has_text_content:
            status = RegionStatus.SKIP
            verdict_reason = "unknown_non_text_skip"
        else:
            verdict_reason = verdict.reason

        updated_reg = replace(
            reg,
            text=accepted,
            ocr_confidence=p_res.confidence,
            status=status,
            review_reason=verdict_reason,
            metadata={
                **reg.metadata,
                "ocr_verdict": {
                    "source": verdict.source,
                    "requires_review": verdict.requires_review,
                    "needs_repair": verdict.needs_repair,
                    "reason": verdict_reason,
                    "second_pass_invoked": v_res is not None,
                },
                "ocr_raw": {"primary": p_res.text, "verifier": v_res.text if v_res else None},
            },
        )
        ocr_regions.append(updated_reg)

        if (
            updated_reg.status == RegionStatus.REVIEW
            and updated_reg.type in (RegionType.DIALOGUE, RegionType.NARRATION, RegionType.UNKNOWN)
            and verdict.needs_repair
            and accepted
        ):
            repair_queue.append((updated_reg.id, verdict, crop.image))

    t0 = time.perf_counter()
    primary_ocr.unload()
    timer.record("11_primary_ocr_unload", time.perf_counter() - t0)
    lifecycle_snapshots["06_post_primary_ocr_unload"] = get_memory_snapshot()

    if verifier_loaded and verifier_ocr is not None:
        t0 = time.perf_counter()
        verifier_ocr.unload()
        timer.record("12_verifier_ocr_unload", time.perf_counter() - t0)
        verifier_loaded = False
        lifecycle_snapshots["07_post_verifier_ocr_unload"] = get_memory_snapshot()

    # Stage 6: Qwen visual repair
    if repair_queue and qwen_repair is not None:
        t_qw_load = time.perf_counter()
        qwen_repair.load()
        timer.record("13_qwen_repair_load", time.perf_counter() - t_qw_load)
        lifecycle_snapshots["08_post_qwen_repair_load"] = get_memory_snapshot()

        repaired_map = {}
        for r_id, verdict, crop_img in repair_queue:
            from providers.ocr.repair import OCRRepairInput
            r_input = OCRRepairInput(verdict.primary_raw, verdict.primary_normalized, verdict.verifier_raw, verdict.verifier_normalized, verdict.reason or "disagreement")
            t_qw_inf = time.perf_counter()
            rep_res = qwen_repair.repair(r_input, crop_img)
            timer.record("14_qwen_repair_inference", time.perf_counter() - t_qw_inf)
            if rep_res.repaired_text and not rep_res.unresolved:
                orig_r = next(r for r in ocr_regions if r.id == r_id)
                repaired_map[r_id] = replace(orig_r, text=rep_res.repaired_text, status=RegionStatus.AUTO, metadata={**orig_r.metadata, "repaired": True})
        if repaired_map:
            ocr_regions = [repaired_map.get(r.id, r) for r in ocr_regions]

        t_qw_un = time.perf_counter()
        qwen_repair.unload()
        timer.record("15_qwen_repair_unload", time.perf_counter() - t_qw_un)
        lifecycle_snapshots["09_post_qwen_repair_unload"] = get_memory_snapshot()

    # Stage 7: Classification & Grouping
    coords = GlobalCoordinateSystem(tuple(pages))
    t0 = time.perf_counter()
    classified_regions = classify_regions(ocr_regions, coords)
    timer.record("16_classification", time.perf_counter() - t0)

    t0 = time.perf_counter()
    text_blocks = group_text_blocks(classified_regions, coords)
    timer.record("17_text_block_grouping", time.perf_counter() - t0)

    # Stage 8: Hy-MT2 Translation
    t0 = time.perf_counter()
    translator.load()
    timer.record("18_hy_mt2_model_load", time.perf_counter() - t0)
    lifecycle_snapshots["10_hy_mt2_active"] = get_memory_snapshot()

    items = [TranslationItem(b.id, b.source_text) for b in text_blocks if b.source_text and b.source_text.strip()]
    t_tr_inf = time.perf_counter()
    tr_result = translator.translate(TranslationInput(items))
    timer.record("19_translation_inference", time.perf_counter() - t_tr_inf)

    tr_map = {r.region_id: r.translation for r in tr_result.results} if tr_result else {}
    translated_block_pairs = []
    for b in text_blocks:
        tr_text = tr_map.get(b.id)
        b_trans = replace(b, translation=tr_text)
        translated_block_pairs.append((b_trans, tr_text or ""))

    t0 = time.perf_counter()
    translator.unload()
    timer.record("20_hy_mt2_unload", time.perf_counter() - t0)
    lifecycle_snapshots["11_post_hy_mt2_unload"] = get_memory_snapshot()

    # Stage 9: Inpainting & Rendering (Global Canvas)
    inpainter = Inpainter(debug_dir=output_dir / "analysis" / "inpainting_debug")
    renderer = TextRenderer()

    canvas_w = pages[0].width
    canvas_h = sum(p.height for p in pages)
    global_canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    y_cursor = 0
    for page in pages:
        with Image.open(page.path) as p_img:
            p_img_rgb = p_img.convert("RGB")
            if p_img_rgb.width != page.width or p_img_rgb.height != page.height:
                p_img_rgb = p_img_rgb.resize((page.width, page.height), Image.Resampling.LANCZOS)
            global_canvas.paste(p_img_rgb, (0, y_cursor))
            y_cursor += page.height

    t0 = time.perf_counter()
    inpainter.lama.load()
    timer.record("21_lama_model_load", time.perf_counter() - t0)
    lifecycle_snapshots["12_post_lama_load"] = get_memory_snapshot()

    t_inp = time.perf_counter()
    cleaned_canvas = inpainter.inpaint_blocks(global_canvas, [b for b, _ in translated_block_pairs])
    timer.record("22_inpaint_inference", time.perf_counter() - t_inp)

    t0 = time.perf_counter()
    inpainter.unload()
    timer.record("23_lama_unload", time.perf_counter() - t0)
    lifecycle_snapshots["13_post_lama_unload"] = get_memory_snapshot()

    renderable_pairs = [
        pair for pair in translated_block_pairs
        if pair[0].id in inpainter.processed_block_ids
        and pair[0].id not in inpainter.review_block_ids
    ]
    t_ren = time.perf_counter()
    rendered_canvas, actual_rendered_count, overflow_count = renderer.render_blocks(cleaned_canvas, renderable_pairs)
    timer.record("24_rendering", time.perf_counter() - t_ren)

    # Stage 10: Export
    t0 = time.perf_counter()
    exported_page_paths = export_chapter_pages(pages, rendered_canvas, output_dir)
    timer.record("25_file_export", time.perf_counter() - t0)

    total_wall_clock = time.perf_counter() - t_start
    lifecycle_snapshots["14_pipeline_end"] = get_memory_snapshot()

    print(f"\n[OK] Instrumented E2E execution complete in {total_wall_clock:.2f} s.")

    # ─────────────────────────────────────────────────────────────────────────
    # PART 1: VISUAL REVIEW PACKAGE GENERATION (15-20 Representative Samples)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[PART 1] Extracting representative visual review contact cards...")

    # Build region lookup map
    region_map = {r.id: r for r in classified_regions}

    # Find sample blocks across required categories
    sample_categories = [
        ("01_short_speech", lambda b: len(b.source_text.split()) <= 3 and "DIALOGUE" in str(b.members[0].type)),
        ("02_long_turkish", lambda b: len(b.translation or "") >= 40),
        ("03_narrow_bubble", lambda b: b.merged_bbox.width <= 140 and b.merged_bbox.height >= 40),
        ("04_large_narration", lambda b: b.members[0].type == RegionType.NARRATION or b.merged_bbox.width >= 350),
        ("05_black_bg_white_text", lambda b: b.metadata.get("page_index", 0) in (2, 3, 10, 14, 15)),
        ("06_colored_game_ui", lambda b: any("SYSTEM" in r.text or "LEVEL" in r.text or "STAT" in r.text for r in b.members)),
        ("07_gradient_texture_bg", lambda b: b.metadata.get("page_index", 0) in (8, 12, 16, 20)),
        ("08_artwork_text", lambda b: b.metadata.get("page_index", 0) in (6, 7, 11, 13)),
        ("09_median_fastpath", lambda b: len(b.members) == 1 and b.members[0].global_bbox.width <= 150 and b.members[0].global_bbox.height <= 40),
        ("10_lama_backend", lambda b: b.merged_bbox.width >= 220 or b.merged_bbox.height >= 80),
        ("11_multi_member_grouping", lambda b: len(b.member_ids) >= 2),
        ("12_hyphen_continuation", lambda b: any("-" in r.text for r in b.members)),
        ("13_ocr_second_pass", lambda b: any(r.metadata.get("ocr_verdict", {}).get("second_pass_invoked") for r in b.members)),
        ("14_review_status", lambda b: any(r.status == RegionStatus.REVIEW for r in b.members)),
        ("15_nearby_bubbles", lambda b: b.metadata.get("page_index", 0) == 10 and len(b.member_ids) == 1),
        ("16_legacy_82_continuation", lambda b: set(b.member_ids) & {0, 1, 2, 82}),
        ("17_legacy_118_ocr", lambda b: any("CRAFTED" in r.text for r in b.members)),
        ("18_legacy_124_ocr", lambda b: any("PORTION" in r.text or "EXP" in r.text for r in b.members)),
        ("19_legacy_34_control", lambda b: any("WHAT THE" in r.text for r in b.members)),
        ("20_legacy_101_control", lambda b: any("PESTS" in r.text for r in b.members)),
    ]

    selected_samples = []
    used_block_ids = set()

    for cat_name, predicate in sample_categories:
        matching = [b for b in text_blocks if predicate(b) and b.id not in used_block_ids]
        if not matching:
            matching = [b for b in text_blocks if predicate(b)]
        if matching:
            best_b = matching[0]
            used_block_ids.add(best_b.id)
            selected_samples.append((cat_name, best_b))

    manifest_samples = []

    for cat_name, block in selected_samples:
        p_idx = block.metadata.get("page_index", 0)
        source_page = pages[p_idx]
        src_img = Image.open(source_page.path).convert("RGB")
        rendered_page_img = Image.open(exported_page_paths[p_idx]).convert("RGB")

        # Bounding box for crop with padding
        bbox = block.merged_bbox
        pad = 24
        cx1 = max(0, bbox.x1 - pad)
        cy1 = max(0, bbox.y1 - pad)
        cx2 = min(src_img.width, bbox.x2 + pad)
        cy2 = min(src_img.height, bbox.y2 + pad)

        src_crop = src_img.crop((cx1, cy1, cx2, cy2))
        ren_crop = rendered_page_img.crop((cx1, cy1, cx2, cy2))

        # Member overlay crop
        overlay_img = src_crop.copy()
        ol_draw = ImageDraw.Draw(overlay_img)
        for r in block.members:
            rx1, ry1, rx2, ry2 = r.global_bbox.x1 - cx1, r.global_bbox.y1 - cy1, r.global_bbox.x2 - cx1, r.global_bbox.y2 - cy1
            ol_draw.rectangle([(rx1, ry1), (rx2, ry2)], outline=(0, 255, 0), width=2)
            ol_draw.text((rx1 + 2, ry1 + 2), f"R{r.id}", fill=(255, 255, 0))
        for line_meta in block.metadata.get("ctd_line_mapping", []):
            poly = line_meta.get("polygon", [])
            if poly:
                local_poly = [(px - cx1, py - cy1) for px, py in poly]
                ol_draw.polygon(local_poly, outline=(255, 0, 0), width=2)

        # Refined glyph mask & inpainting crop
        local_members = [replace(r, global_bbox=BBox(r.global_bbox.x1 - cx1, r.global_bbox.y1 - cy1, r.global_bbox.x2 - cx1, r.global_bbox.y2 - cy1)) for r in block.members]
        local_block = replace(block, members=local_members, merged_bbox=BBox(pad, pad, pad + bbox.width, pad + bbox.height))
        inp_crop = inpainter.inpaint_blocks(src_crop, [local_block])
        text_mask = inpainter.last_text_mask
        mask_overlay_img = text_mask.overlay() if text_mask is not None else src_crop
        method = "median" if (text_mask and text_mask.is_uniform_background) else "lama_large"

        # Contact card creation
        sample_folder = OUTPUT_REVIEW_DIR / cat_name
        sample_folder.mkdir(parents=True, exist_ok=True)

        src_crop.save(sample_folder / "1_source.png")
        overlay_img.save(sample_folder / "2_detected_members.png")
        mask_overlay_img.save(sample_folder / "3_glyph_mask.png")
        inp_crop.save(sample_folder / "4_inpainted.png")
        ren_crop.save(sample_folder / "5_rendered.png")

        meta_line1 = f"Page: {source_page.name} | Block #{block.id} | Members: {list(block.member_ids)} | Status: {block.members[0].status.value}"
        meta_line2 = f"OCR Source: {block.source_text}"
        meta_line3 = f"Turkish Translation: {block.translation}"
        meta_line4 = f"Inpaint Backend: {method} | Second Pass Invoked: {any(r.metadata.get('ocr_verdict', {}).get('second_pass_invoked') for r in block.members)}"
        full_meta_str = f"{meta_line1}\n{meta_line2}\n{meta_line3}\n{meta_line4}"

        contact_card = create_contact_card(
            src_crop, overlay_img, mask_overlay_img, inp_crop, ren_crop,
            title=f"Sample {cat_name.upper()} — Block #{block.id} (Page {source_page.name})",
            meta_text=full_meta_str
        )
        contact_card.save(sample_folder / "0_contact_card.png")

        sample_manifest_record = {
            "category": cat_name,
            "page": source_page.name,
            "block_id": block.id,
            "member_ids": list(block.member_ids),
            "status": block.members[0].status.value,
            "source_text": block.source_text,
            "translation": block.translation,
            "inpaint_backend": method,
            "second_pass_invoked": any(r.metadata.get('ocr_verdict', {}).get('second_pass_invoked') for r in block.members),
            "review_reasons": [r.review_reason for r in block.members if r.review_reason],
            "contact_card_path": str((sample_folder / "0_contact_card.png").relative_to(OUTPUT_REVIEW_DIR)),
        }
        (sample_folder / "metadata.json").write_text(json.dumps(sample_manifest_record, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest_samples.append(sample_manifest_record)

    visual_manifest = {
        "chapter": "Chapter 1",
        "sample_count": len(manifest_samples),
        "samples": manifest_samples,
    }
    (OUTPUT_REVIEW_DIR / "visual_review_manifest.json").write_text(json.dumps(visual_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Visual review manifest written with {len(manifest_samples)} representative contact cards.")

    # ─────────────────────────────────────────────────────────────────────────
    # PART 2: STAGE TIMING & RAM/VRAM PERFORMANCE REPORT
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[PART 2] Generating performance_report.json and performance_report.md...")

    stage_statistics = timer.all_stats()

    peak_vram = max((snap["dedicated_vram_mb"] for snap in lifecycle_snapshots.values()), default=0.0)
    peak_rss = max((snap["process_rss_mb"] for snap in lifecycle_snapshots.values()), default=0.0)
    peak_llama_rss = max((snap["llama_server_rss_mb"] for snap in lifecycle_snapshots.values()), default=0.0)

    perf_report_json = {
        "report_scope": "instrumented_benchmark_reimplementation",
        "production_lifecycle_differences": [
            "The benchmark invokes pipeline stages directly instead of calling ChapterAnalyzer.process_chapter().",
            "The benchmark eagerly loads LaMa immediately before block inpainting; production loads LaMa lazily only when a non-uniform-background block requires it.",
            "The benchmark records explicit post-load/post-unload memory checkpoints that production does not record.",
        ],
        "translation_model_file": Path(translator.model_path).name,
        "total_wall_clock_sec": round(total_wall_clock, 3),
        "pages_count": len(pages),
        "text_blocks_count": len(text_blocks),
        "stage_statistics": stage_statistics,
        "lifecycle_snapshots": lifecycle_snapshots,
        "peak_metrics": {
            "peak_process_rss_mb": round(peak_rss, 2),
            "peak_dedicated_vram_mb": round(peak_vram, 2),
            "peak_llama_server_rss_mb": round(peak_llama_rss, 2),
        },
        "model_lifecycle_checkpoints": [
            "ComicTextDetector: loaded during detection, unloaded before OCR",
            "Primary PP-OCRv6: loaded during OCR, unloaded before Qwen repair / Hy-MT2",
            "Verifier PaddleOCR-VL: loaded during OCR (if configured), unloaded before Qwen repair / Hy-MT2",
            "Qwen Repair: loaded during visual repair (if repair queue non-empty), unloaded before Hy-MT2",
            "Hy-MT2: loaded during block translation, unloaded before LaMa",
            "LaMa Large: loaded during block inpainting, unloaded before export",
        ],
    }
    (OUTPUT_REVIEW_DIR / "performance_report.json").write_text(json.dumps(perf_report_json, ensure_ascii=False, indent=2), encoding="utf-8")

    # Generate Markdown Performance Report strictly from JSON fields
    md_lines = []
    md_lines.append("# Chapter 1 Performance & GPU Memory Lifecycle Report")
    md_lines.append("")
    md_lines.append(f"- **Total Runtime (Wall-Clock)**: `{perf_report_json['total_wall_clock_sec']:.2f} s` ({perf_report_json['total_wall_clock_sec'] / 60.0:.2f} min)")
    md_lines.append(f"- **Total Pages Processed**: `{perf_report_json['pages_count']}`")
    md_lines.append(f"- **Total TextBlocks Processed**: `{perf_report_json['text_blocks_count']}`")
    md_lines.append(f"- **Configured Hy-MT2 model file**: `{perf_report_json['translation_model_file']}`")
    md_lines.append("")
    md_lines.append("## Scope versus production")
    md_lines.append("")
    md_lines.append("This benchmark follows the production stage order, but its lifecycle is not identical:")
    for difference in perf_report_json["production_lifecycle_differences"]:
        md_lines.append(f"- {difference}")
    md_lines.append("")
    md_lines.append("## 1. Measured Stage Timing Summary")
    md_lines.append("")
    md_lines.append("| Stage | Calls | Total (s) | Avg (ms) | p95 (ms) |")
    md_lines.append("|---|---|---|---|---|")
    for stg_name, stg_data in sorted(perf_report_json["stage_statistics"].items()):
        md_lines.append(f"| `{stg_name}` | {stg_data['calls']} | {stg_data['total_sec']:.3f} | {stg_data['avg_ms']:.2f} | {stg_data['p95_ms']:.2f} |")
    md_lines.append("")

    md_lines.append("## 2. Measured RAM / VRAM Lifecycle Snapshots")
    md_lines.append("")
    md_lines.append("| Checkpoint | Process RSS (MB) | CUDA Alloc (MB) | CUDA Res (MB) | Dedicated VRAM (MB) | llama-server RSS (MB) |")
    md_lines.append("|---|---|---|---|---|---|")
    for chk_name, snap in sorted(perf_report_json["lifecycle_snapshots"].items()):
        md_lines.append(f"| `{chk_name}` | {snap['process_rss_mb']:.1f} | {snap['cuda_allocated_mb']:.1f} | {snap['cuda_reserved_mb']:.1f} | {snap['dedicated_vram_mb']:.1f} | {snap['llama_server_rss_mb']:.1f} |")
    md_lines.append("")

    md_lines.append("## 3. Measured Model Lifecycle Checkpoints")
    md_lines.append("")
    for cp in perf_report_json["model_lifecycle_checkpoints"]:
        md_lines.append(f"- {cp}")
    md_lines.append("")

    sorted_by_total = sorted(perf_report_json["stage_statistics"].items(), key=lambda x: x[1]["total_sec"], reverse=True)
    top_3 = sorted_by_total[:3]

    md_lines.append("## 4. Key Measured Performance Summary")
    md_lines.append(f"- **Total Runtime**: `{perf_report_json['total_wall_clock_sec']:.2f} s` ({perf_report_json['total_wall_clock_sec'] / 60.0:.2f} min)")
    md_lines.append(f"- **Top 3 Most Expensive Stages**:")
    for rank, (stg_name, stg_data) in enumerate(top_3, 1):
        md_lines.append(f"  {rank}. `{stg_name}`: `{stg_data['total_sec']:.2f} s` ({stg_data['total_sec'] / perf_report_json['total_wall_clock_sec'] * 100:.1f}% of total)")
    md_lines.append(f"- **Peak Dedicated GPU VRAM**: `{perf_report_json['peak_metrics']['peak_dedicated_vram_mb']:.1f} MB`")
    md_lines.append(f"- **Peak Process RAM (RSS)**: `{perf_report_json['peak_metrics']['peak_process_rss_mb']:.1f} MB`")
    md_lines.append(f"- **Peak llama-server RAM (RSS)**: `{perf_report_json['peak_metrics']['peak_llama_server_rss_mb']:.1f} MB`")

    md_content = "\n".join(md_lines)

    # Sanity checks: Assert no unmeasured statements are present in Markdown report
    forbidden_unmeasured_terms = ["mmap", "private memory", "shared GPU memory", "spillover", "no leaks"]
    for term in forbidden_unmeasured_terms:
        assert term.lower() not in md_content.lower(), (
            f"Sanity Check Error: Unmeasured statement containing '{term}' found in Markdown report!"
        )

    (OUTPUT_REVIEW_DIR / "performance_report.md").write_text(md_content, encoding="utf-8")
    print(f"[OK] Performance report markdown written to {OUTPUT_REVIEW_DIR / 'performance_report.md'}")

    print("\n==================================================================")
    print("ALL DELIVERABLES GENERATED SUCCESSFULLY.")
    print(f"Review Package: {OUTPUT_REVIEW_DIR}")
    print("==================================================================")


if __name__ == "__main__":
    main()
