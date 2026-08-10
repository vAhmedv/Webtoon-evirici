"""Real Chapter Translation Gate V1 (10-Chapter Real Webtoon Benchmark).

Translates 10 real chapters (5 Axe God + 5 God-Tier Crafter) comparing:
  Model A: TranslateGemma-12B-IT Q5_K_M (Variant C)
  Model B: Qwen3.5-9B Q5_K_M GGUF (Native Chat Template)

All source chapter image directories are READ-ONLY.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from loguru import logger
from PIL import Image

# Ensure project root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.coordinate.global_coords import GlobalCoordinateSystem
from core.coordinate.sliding_window import generate_windows_for_pages
from core.detection import BBox, Detection, RegionStatus, RegionType
from core.detection.coordinate import window_bbox_to_global
from core.detection.merge import merge_duplicates
from core.imaging.region_cropper import RegionCropper
from core.imaging.window_extractor import extract_window_image
from core.io.input_loader import Config, list_image_files
from core.models import Page
from core.ocr_normalizer import normalize_ocr_text
from core.translation.profile_discovery import CandidateStore
from core.translation.semantic_context import (
    SemanticContextRequest,
    resolve_controlled_bridge_with_fallback,
)
from core.translation.series_profile import SeriesProfile
from providers.detector.yolo8_comic import Yolo8ComicTextDetector
from providers.ocr.agreement import decide_ocr_agreement
from providers.ocr.base import OCRLine, OCRResult
from providers.ocr.paddleocr import PaddleOCRProvider
from providers.ocr.paddleocr_vl import PaddleOCRVLOcrProvider
from providers.translation.base import TranslationInput, TranslationItem
from providers.translation.qwen_gguf_translation_v2 import (
    DEFAULT_LLAMA_EXE_PATH,
    DEFAULT_QWEN_MODEL_PATH,
    QwenGGUFTranslationProviderV2,
)
from providers.translation.qwen_semantic_resolver import QwenSemanticResolverProvider
from providers.translation.translategemma_gguf_translation import (
    DEFAULT_GEMMA_MODEL_PATH,
    TranslateGemmaGGUFTranslationProvider,
)


# ── SOURCE DIRECTORIES (READ-ONLY) ──
SOURCE_DIR_A = Path(
    r"C:\Users\Ahmed\AppData\Local\Tachidesk\downloads\mangas\Arya Scans (EN)\Axe God_ The Road to Invincibility"
)
SOURCE_DIR_B = Path(
    r"C:\Users\Ahmed\AppData\Local\Tachidesk\downloads\mangas\Asmodeus Scans (EN)\Reincarnated as a God-Tier Crafter"
)

BENCHMARK_OUT_DIR = PROJECT_ROOT / "benchmark_results" / "real_chapter_translation_gate_v1"

SELECTED_CHAPTERS_A = ["Chapter 1", "Chapter 2", "Chapter 3", "Chapter 4", "Chapter 5"]
SELECTED_CHAPTERS_B = ["Chapter 1", "Chapter 2", "Chapter 3", "Chapter 4", "Chapter 5"]


def get_series_profile(series_key: str) -> SeriesProfile:
    if series_key == "axe_god":
        return SeriesProfile(
            series_id="axe_god",
            known_names={
                "LUO TIAN": "Luo Tian",
                "HU SAN": "Hu San",
                "GAO YUAN": "Gao Yuan",
                "YU": "Yu",
            },
            glossary={
                "ABILITY USER": "yetenek kullanıcısı",
                "SECRET REALM": "gizli âlem",
                "SECRET REALM GUIDE": "gizli âlem rehberi",
                "LEVEL 1": "1. seviye",
                "BLACKWIND RAVINE": "Blackwind Ravine",
                "FROST CHAIN": "Frost Chain",
                "AXE GOD": "Balta Tanrısı",
            },
        )
    else:
        return SeriesProfile(
            series_id="god_tier_crafter",
            known_names={
                "ETHAN": "Ethan",
                "LUCAS": "Lucas",
            },
            glossary={
                "CRAFTER": "Zanaatkar",
                "GOD-TIER": "Tanrı Seviyesi",
                "REINCARNATED": "Enkarne Olmuş",
                "SYSTEM": "Sistem",
                "SKILL": "Yetenek",
            },
        )


def calculate_sha256(data: Any) -> str:
    raw = json.dumps(data, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def load_chapter_safe(folder: Path) -> list[Page]:
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
            pages.append(
                Page(
                    index=page_idx,
                    path=path.resolve(),
                    width=w,
                    height=h,
                    y_offset=y_offset,
                )
            )
            y_offset += h
            page_idx += 1

    return pages


def is_noise_or_sfx(text: str) -> tuple[bool, str]:
    cleaned = text.strip()
    if len(cleaned) < 2:
        return True, "too_short"
    if not re.search(r"[a-zA-Z]", cleaned):
        return True, "no_letters"
    # Common SFX / credit noise check
    upper = cleaned.upper()
    if any(sfx in upper for sfx in ["BOOM", "CLANG", "SWOOSH", "THUD", "SCANLATION", "DISCORD.GG", "CHAPTER"]):
        if len(cleaned.split()) <= 2:
            return True, "sfx_or_watermark"
    return False, ""


def extract_real_chapter_dataset() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    print("\n--- PHASE 1: Real Chapter Image Loading, Detection & OCR Extraction ---")

    detector = Yolo8ComicTextDetector()
    detector.load()

    ocr_primary = PaddleOCRVLOcrProvider()
    ocr_primary.load()

    ocr_verifier = PaddleOCRProvider()
    ocr_verifier.load()

    all_accepted_dataset: list[dict[str, Any]] = []
    all_excluded_regions: list[dict[str, Any]] = []
    chapter_manifest: list[dict[str, Any]] = []

    all_series_targets = [
        ("axe_god", "Axe God: The Road to Invincibility", SOURCE_DIR_A, SELECTED_CHAPTERS_A),
        ("god_tier_crafter", "Reincarnated as a God-Tier Crafter", SOURCE_DIR_B, SELECTED_CHAPTERS_B),
    ]

    global_item_counter = 1

    for series_key, series_name, series_dir, chapters in all_series_targets:
        for ch_name in chapters:
            ch_folder = series_dir / ch_name
            if not ch_folder.is_dir():
                print(f"WARNING: Chapter directory not found: {ch_folder}")
                continue

            pages = load_chapter_safe(ch_folder)
            coords = GlobalCoordinateSystem(tuple(pages))
            windows = generate_windows_for_pages(pages, window_height=1000, overlap=200)

            # Detect text bboxes
            all_dets = []
            for win in windows:
                win_img = extract_window_image(tuple(pages), win, coords)
                found = detector.detect(win_img.image, win.id)
                for d in found:
                    g_bbox = window_bbox_to_global(d.bbox, win.y_start)
                    all_dets.append(
                        Detection(
                            bbox=g_bbox,
                            confidence=d.confidence,
                            type=d.type,
                            source_window_id=win.id,
                            metadata=d.metadata,
                        )
                    )

            merged_regions = merge_duplicates(all_dets, iou_threshold=0.5)
            sorted_regions = sorted(merged_regions, key=lambda r: r.global_bbox.y1)

            print(
                f"[{series_key.upper()} / {ch_name}] Loaded {len(pages)} pages, {len(merged_regions)} text regions detected."
            )

            cropper = RegionCropper(pages, coords)
            ch_accepted_count = 0
            ch_excluded_count = 0

            # Crop & OCR each region
            ch_items = []
            for idx, reg in enumerate(sorted_regions):
                crop_obj = cropper.crop_region(reg)
                cropped_img = crop_obj.image

                res_p = ocr_primary.recognize(cropped_img)
                res_v = ocr_verifier.recognize(cropped_img)

                verdict = decide_ocr_agreement(res_p, res_v)
                decided_text = verdict.accepted_text or verdict.provisional_text or res_p.text
                agreement_status = verdict.reason
                normalized_text = normalize_ocr_text(decided_text)

                is_excluded, ex_reason = is_noise_or_sfx(normalized_text)

                try:
                    p_idx, _ = coords.global_to_page(reg.global_bbox.y1)
                    page_obj = pages[p_idx]
                    img_rel_path = str(page_obj.path.relative_to(ch_folder))
                except Exception:
                    img_rel_path = ""

                item_id_str = f"{series_key}_{ch_name.lower().replace(' ', '')}_{idx + 1:03d}"

                rec = {
                    "id": item_id_str,
                    "series_id": series_key,
                    "series_name": series_name,
                    "chapter_name": ch_name,
                    "source_directory": str(ch_folder),
                    "source_image": img_rel_path,
                    "region_id": idx + 1,
                    "global_read_order": idx + 1,
                    "bbox": reg.global_bbox.to_tuple(),
                    "original_accepted_english": normalized_text,
                    "ocr_primary": res_p.text,
                    "ocr_verifier": res_v.text,
                    "ocr_agreement": agreement_status,
                    "excluded_from_translation_quality": is_excluded,
                    "exclusion_reason": ex_reason if is_excluded else None,
                }

                if is_excluded:
                    all_excluded_regions.append(rec)
                    ch_excluded_count += 1
                else:
                    ch_items.append(rec)

            # Pick contiguous representative window of 25-35 story items per chapter
            max_chapter_items = 30
            if len(ch_items) > max_chapter_items:
                ch_items = ch_items[:max_chapter_items]

            for item in ch_items:
                item["benchmark_global_id"] = global_item_counter
                global_item_counter += 1
                all_accepted_dataset.append(item)
                ch_accepted_count += 1

            chapter_manifest.append(
                {
                    "series_id": series_key,
                    "series_name": series_name,
                    "chapter_name": ch_name,
                    "page_count": len(pages),
                    "detected_regions": len(merged_regions),
                    "accepted_regions": ch_accepted_count,
                    "excluded_regions": ch_excluded_count,
                }
            )

    detector.unload()
    ocr_primary.unload()
    ocr_verifier.unload()

    print(
        f"\nExtraction Complete: {len(all_accepted_dataset)} accepted story regions, "
        f"{len(all_excluded_regions)} excluded regions across 10 chapters."
    )
    return all_accepted_dataset, all_excluded_regions, chapter_manifest


from core.translation.semantic_context import (
    SemanticContextRequest,
    resolve_controlled_bridge_with_fallback,
)


def run_phase_v3_resolver(
    accepted_dataset: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    print("\n--- PHASE 2: Qwen Semantic Context V3 Controlled-English Resolver ---")

    resolver = QwenSemanticResolverProvider(server_url="http://127.0.0.1:8082")
    resolver.load()

    v3_selected_targets: list[dict[str, Any]] = []

    # Group items by chapter to build clean intra-chapter context windows
    items_by_chapter: dict[str, list[dict[str, Any]]] = {}
    for item in accepted_dataset:
        key = f"{item['series_id']}_{item['chapter_name']}"
        items_by_chapter.setdefault(key, []).append(item)

    total_count = len(accepted_dataset)
    done_count = 0

    for key, items in items_by_chapter.items():
        for idx, item in enumerate(items):
            done_count += 1
            if done_count % 25 == 0 or done_count == 1 or done_count == total_count:
                print(f"[V3 Resolver Progress] {done_count}/{total_count} items resolved...")
                sys.stdout.flush()

            # Prev context (up to 3 within same chapter)
            prev_context = [
                items[i]["original_accepted_english"]
                for i in range(max(0, idx - 3), idx)
            ]
            # Next context (up to 1 within same chapter)
            next_context = [
                items[i]["original_accepted_english"]
                for i in range(idx + 1, min(len(items), idx + 2))
            ]

            req = SemanticContextRequest(
                target_source=item["original_accepted_english"],
                previous_context=tuple(prev_context),
                next_context=tuple(next_context),
            )

            outcome = resolve_controlled_bridge_with_fallback(req, resolver.resolve)

            entry = {
                "id": item["id"],
                "series_id": item["series_id"],
                "series_name": item["series_name"],
                "chapter_name": item["chapter_name"],
                "source_image": item["source_image"],
                "region_id": item["region_id"],
                "previous_context": prev_context,
                "original_accepted_english": item["original_accepted_english"],
                "next_context": next_context,
                "semantic_v3": {
                    "rewrite_used": outcome.decision.rewrite_used,
                    "selected_english": outcome.decision.selected_target,
                    "reasoning": outcome.decision.rejection_reason or "",
                    "raw_output": outcome.raw_response,
                },
            }
            v3_selected_targets.append(entry)

    resolver.unload()
    print(f"V3 Resolver completed on {len(v3_selected_targets)} items.")
    return v3_selected_targets


def run_phase_translategemma(
    v3_targets: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    print("\n--- PHASE 3: TranslateGemma-12B Translation (Port 8081) ---")

    gemma_provider = TranslateGemmaGGUFTranslationProvider(
        model_path=DEFAULT_GEMMA_MODEL_PATH,
        executable_path=DEFAULT_LLAMA_EXE_PATH,
        server_url="http://127.0.0.1:8081",
        micro_batch_enabled=False,
    )
    gemma_provider.load()

    tg_results: list[dict[str, Any]] = []

    # Process per-series to maintain terminology separation
    items_by_series: dict[str, list[dict[str, Any]]] = {}
    for t in v3_targets:
        items_by_series.setdefault(t["series_id"], []).append(t)

    for series_id, s_items in items_by_series.items():
        print(f"[TranslateGemma Phase] Translating {len(s_items)} items for series {series_id}...")
        sys.stdout.flush()
        profile = get_series_profile(series_id)
        trans_inp = TranslationInput(
            items=[
                TranslationItem(
                    region_id=item["id"],
                    source=item["semantic_v3"]["selected_english"],
                    reading_order=idx + 1,
                    known_names=profile.get_known_names_list(),
                )
                for idx, item in enumerate(s_items)
            ],
            profile=profile,
            context_items=[],
            candidate_store=CandidateStore(series_id=series_id),
            chapter_id=f"gate_v1_{series_id}",
        )

        out = gemma_provider.translate(trans_inp)
        for res in out.results:
            tg_results.append(
                {
                    "id": res.region_id,
                    "source": res.source,
                    "translation": res.translation,
                    "raw_model_response": res.raw_model_response,
                    "warnings": list(res.validation_warnings),
                    "requires_review": res.requires_review,
                }
            )

    gemma_provider.unload()
    print(f"TranslateGemma complete on {len(tg_results)} items.")
    return tg_results


def run_phase_qwen_translator(
    v3_targets: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    print("\n--- PHASE 4: Qwen3.5-9B Direct Translation (Port 8083) ---")

    qwen_provider = QwenGGUFTranslationProviderV2(
        model_path=DEFAULT_QWEN_MODEL_PATH,
        executable_path=DEFAULT_LLAMA_EXE_PATH,
        server_url="http://127.0.0.1:8083",
    )
    qwen_provider.load()

    qw_results: list[dict[str, Any]] = []

    items_by_series: dict[str, list[dict[str, Any]]] = {}
    for t in v3_targets:
        items_by_series.setdefault(t["series_id"], []).append(t)

    for series_id, s_items in items_by_series.items():
        print(f"[Qwen Translator Phase] Translating {len(s_items)} items for series {series_id}...")
        sys.stdout.flush()
        profile = get_series_profile(series_id)
        trans_inp = TranslationInput(
            items=[
                TranslationItem(
                    region_id=item["id"],
                    source=item["semantic_v3"]["selected_english"],
                    reading_order=idx + 1,
                    known_names=profile.get_known_names_list(),
                )
                for idx, item in enumerate(s_items)
            ],
            profile=profile,
            context_items=[],
            candidate_store=CandidateStore(series_id=series_id),
            chapter_id=f"gate_v1_{series_id}",
        )

        out = qwen_provider.translate(trans_inp)
        for res in out.results:
            qw_results.append(
                {
                    "id": res.region_id,
                    "source": res.source,
                    "translation": res.translation,
                    "raw_model_response": res.raw_model_response,
                    "warnings": list(res.validation_warnings),
                    "requires_review": res.requires_review,
                }
            )

    qwen_provider.unload()
    print(f"Qwen Translator complete on {len(qw_results)} items.")
    return qw_results


def generate_gate_artifacts(
    accepted_dataset: list[dict[str, Any]],
    excluded_regions: list[dict[str, Any]],
    chapter_manifest: list[dict[str, Any]],
    v3_targets: list[dict[str, Any]],
    tg_results: list[dict[str, Any]],
    qw_results: list[dict[str, Any]],
):
    print("\n--- PHASE 5: Generating Comparison & Review Artifacts ---")

    tg_by_id = {r["id"]: r for r in tg_results}
    qw_by_id = {r["id"]: r for r in qw_results}
    v3_by_id = {t["id"]: t for t in v3_targets}

    comparison_items = []
    txt_blocks = []
    compact_review_lines = []
    critical_disagreements = []

    current_series = ""
    current_chapter = ""

    chapter_stats = {}

    for item in accepted_dataset:
        item_id = item["id"]
        v3_info = v3_by_id[item_id]
        tg_info = tg_by_id[item_id]
        qw_info = qw_by_id[item_id]

        s_id = item["series_id"]
        c_name = item["chapter_name"]

        ch_key = f"{s_id} / {c_name}"
        if ch_key not in chapter_stats:
            chapter_stats[ch_key] = {
                "total_items": 0,
                "tg_review_count": 0,
                "qw_review_count": 0,
                "v3_rewrite_count": 0,
            }

        chapter_stats[ch_key]["total_items"] += 1
        if tg_info["requires_review"]:
            chapter_stats[ch_key]["tg_review_count"] += 1
        if qw_info["requires_review"]:
            chapter_stats[ch_key]["qw_review_count"] += 1
        if v3_info["semantic_v3"]["rewrite_used"]:
            chapter_stats[ch_key]["v3_rewrite_count"] += 1

        comp_rec = {
            "id": item_id,
            "series": item["series_name"],
            "chapter": c_name,
            "source_image": item["source_image"],
            "region_id": item["region_id"],
            "previous_context": v3_info["previous_context"],
            "original_accepted_english": item["original_accepted_english"],
            "next_context": v3_info["next_context"],
            "semantic_v3": {
                "rewrite_used": v3_info["semantic_v3"]["rewrite_used"],
                "selected_english": v3_info["semantic_v3"]["selected_english"],
            },
            "translategemma": {
                "translation": tg_info["translation"],
                "warnings": tg_info["warnings"],
                "requires_review": tg_info["requires_review"],
            },
            "qwen35": {
                "translation": qw_info["translation"],
                "warnings": qw_info["warnings"],
                "requires_review": qw_info["requires_review"],
            },
            "human_review": {
                "winner": None,
                "translategemma_score": None,
                "qwen_score": None,
                "notes": None,
            },
        }
        comparison_items.append(comp_rec)

        # Build comparison.txt block
        if item["series_name"] != current_series:
            current_series = item["series_name"]
            txt_blocks.append(f"\n{'='*70}\nSERIES: {current_series.upper()}\n{'='*70}\n")
        if c_name != current_chapter:
            current_chapter = c_name
            txt_blocks.append(f"\n--- {current_chapter.upper()} ---\n")

        txt_blocks.append(
            f"============================================================\n"
            f"[{item_id}]\n\n"
            f"PREVIOUS CONTEXT:\n"
            f"{' | '.join(v3_info['previous_context']) if v3_info['previous_context'] else '(none)'}\n\n"
            f"ORIGINAL ACCEPTED ENGLISH:\n"
            f"{item['original_accepted_english']}\n\n"
            f"V3 SELECTED ENGLISH:\n"
            f"{v3_info['semantic_v3']['selected_english']}\n\n"
            f"V3 REWRITE: {'yes' if v3_info['semantic_v3']['rewrite_used'] else 'no'}\n\n"
            f"TRANSLATEGEMMA:\n"
            f"{tg_info['translation']}\n\n"
            f"TG WARNINGS:\n"
            f"{', '.join(tg_info['warnings']) if tg_info['warnings'] else '(none)'}\n\n"
            f"QWEN3.5:\n"
            f"{qw_info['translation']}\n\n"
            f"QWEN WARNINGS:\n"
            f"{', '.join(qw_info['warnings']) if qw_info['warnings'] else '(none)'}\n\n"
            f"HUMAN:\n"
            f"winner: \n"
            f"notes: \n"
            f"============================================================\n"
        )

        # Compact review sheet line
        compact_review_lines.append(
            f"[{item_id}]\n"
            f"EN: {v3_info['semantic_v3']['selected_english']}\n"
            f"TG: {tg_info['translation']}\n"
            f"QW: {qw_info['translation']}\n"
            f"WINNER: [ ]\n"
            f"NOTE: [ ]\n"
        )

        # Critical disagreement check
        tg_tr = str(tg_info["translation"] or "").strip()
        qw_tr = str(qw_info["translation"] or "").strip()
        is_substantially_different = tg_tr != qw_tr
        has_warns = bool(tg_info["warnings"] or qw_info["warnings"])
        rewrite = v3_info["semantic_v3"]["rewrite_used"]

        if is_substantially_different or has_warns or rewrite:
            critical_disagreements.append(
                f"[{item_id}]\n"
                f"EN: {v3_info['semantic_v3']['selected_english']}\n"
                f"TG: {tg_tr} (warns={tg_info['warnings']})\n"
                f"QW: {qw_tr} (warns={qw_info['warnings']})\n"
                f"REWRITE: {'yes' if rewrite else 'no'}\n"
                f"------------------------------------------------------------\n"
            )

    (BENCHMARK_OUT_DIR / "comparison.json").write_text(
        json.dumps(comparison_items, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (BENCHMARK_OUT_DIR / "comparison.txt").write_text(
        "\n".join(txt_blocks), encoding="utf-8"
    )
    (BENCHMARK_OUT_DIR / "human_review_sheet.txt").write_text(
        "\n".join(compact_review_lines), encoding="utf-8"
    )
    (BENCHMARK_OUT_DIR / "critical_disagreements.txt").write_text(
        "\n".join(critical_disagreements), encoding="utf-8"
    )

    (BENCHMARK_OUT_DIR / "chapter_summaries.json").write_text(
        json.dumps(chapter_stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    dataset_hash = calculate_sha256(accepted_dataset)

    summary = {
        "dataset_hash": dataset_hash,
        "total_accepted_english_regions": len(accepted_dataset),
        "total_excluded_ocr_regions": len(excluded_regions),
        "total_chapters": len(chapter_manifest),
        "v3_rewrite_count": sum(1 for t in v3_targets if t["semantic_v3"]["rewrite_used"]),
        "translategemma_summary": {
            "total_calls": len(tg_results),
            "requires_review_count": sum(1 for r in tg_results if r["requires_review"]),
        },
        "qwen_summary": {
            "total_calls": len(qw_results),
            "requires_review_count": sum(1 for r in qw_results if r["requires_review"]),
        },
        "chapter_manifest": chapter_manifest,
    }

    (BENCHMARK_OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nAll benchmark artifacts written to {BENCHMARK_OUT_DIR}")


def main():
    BENCHMARK_OUT_DIR.mkdir(parents=True, exist_ok=True)

    accepted_file = BENCHMARK_OUT_DIR / "accepted_english_dataset.json"
    excluded_file = BENCHMARK_OUT_DIR / "excluded_regions.json"
    manifest_file = BENCHMARK_OUT_DIR / "chapter_manifest.json"
    v3_file = BENCHMARK_OUT_DIR / "v3_selected_targets.json"
    tg_file = BENCHMARK_OUT_DIR / "translategemma_results.json"
    qw_file = BENCHMARK_OUT_DIR / "qwen_results.json"

    # Resumable Phase 1
    if accepted_file.is_file() and excluded_file.is_file() and manifest_file.is_file():
        print("Reusing existing frozen English dataset...")
        accepted_dataset = json.loads(accepted_file.read_text(encoding="utf-8"))
        excluded_regions = json.loads(excluded_file.read_text(encoding="utf-8"))
        chapter_manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    else:
        accepted_dataset, excluded_regions, chapter_manifest = extract_real_chapter_dataset()
        accepted_file.write_text(json.dumps(accepted_dataset, indent=2, ensure_ascii=False), encoding="utf-8")
        excluded_file.write_text(json.dumps(excluded_regions, indent=2, ensure_ascii=False), encoding="utf-8")
        manifest_file.write_text(json.dumps(chapter_manifest, indent=2, ensure_ascii=False), encoding="utf-8")

        dataset_hash = calculate_sha256(accepted_dataset)
        hashes = {"accepted_english_dataset_hash": dataset_hash}
        (BENCHMARK_OUT_DIR / "input_hashes.json").write_text(
            json.dumps(hashes, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # Resumable Phase 2 (V3 Resolver)
    if v3_file.is_file():
        print("Reusing existing V3 selected targets...")
        v3_targets = json.loads(v3_file.read_text(encoding="utf-8"))
    else:
        v3_targets = run_phase_v3_resolver(accepted_dataset)
        v3_file.write_text(json.dumps(v3_targets, indent=2, ensure_ascii=False), encoding="utf-8")

    # Resumable Phase 3 (TranslateGemma)
    if tg_file.is_file():
        print("Reusing existing TranslateGemma results...")
        tg_results = json.loads(tg_file.read_text(encoding="utf-8"))
    else:
        tg_results = run_phase_translategemma(v3_targets)
        tg_file.write_text(json.dumps(tg_results, indent=2, ensure_ascii=False), encoding="utf-8")

    # Resumable Phase 4 (Qwen Translator)
    if qw_file.is_file():
        print("Reusing existing Qwen Translator results...")
        qw_results = json.loads(qw_file.read_text(encoding="utf-8"))
    else:
        qw_results = run_phase_qwen_translator(v3_targets)
        qw_file.write_text(json.dumps(qw_results, indent=2, ensure_ascii=False), encoding="utf-8")

    # Resumable Phase 5 (Artifact Generation)
    generate_gate_artifacts(
        accepted_dataset,
        excluded_regions,
        chapter_manifest,
        v3_targets,
        tg_results,
        qw_results,
    )


if __name__ == "__main__":
    main()
