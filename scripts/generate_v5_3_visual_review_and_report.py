"""Generate V5.3 Final Chapter 1 Visual Review Package and Performance Reports.

Extracts 20 representative visual review contact cards from Chapter 1 E2E outputs
and writes visual_review_manifest.json, performance_report.json, and performance_report.md.
"""

from __future__ import annotations

import copy
from dataclasses import replace
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.coordinate.global_coords import GlobalCoordinateSystem
from core.detection import BBox, Region, RegionStatus, RegionType
from core.detection.classification import classify_regions
from core.detection.text_block import group_text_blocks, TextBlock
from core.imaging.inpainter import Inpainter, _is_story_text
from core.imaging.renderer import TextRenderer
from core.io.input_loader import load_chapter
from core.config import load_config

SOURCE_CHAPTER = Path(
    r"C:\Users\Ahmed\AppData\Local\Tachidesk\downloads\mangas\Asmodeus Scans (EN)\Reincarnated as a God-Tier Crafter\Chapter 1"
)
E2E_OUTPUT_DIR = ROOT / "e2e_output" / "real_tachidesk_chapter_1"
OUTPUT_REVIEW_DIR = ROOT / "review_output" / "chapter1_final_visual_review"


def rebase_geometry_to_crop(
    block: TextBlock,
    page_y_offset: int,
    crop_x1: int,
    crop_y1: int,
) -> TextBlock:
    """Rebase all bounding boxes and metadata geometry into crop-local coordinates.

    Performs deep-copy / fresh object construction to prevent mutating original region metadata.
    """
    rebased_members = []
    for r in block.members:
        # Rebase global_bbox to crop-local space
        r_local_y1 = r.global_bbox.y1 - page_y_offset - crop_y1
        r_local_y2 = r.global_bbox.y2 - page_y_offset - crop_y1
        r_local_x1 = r.global_bbox.x1 - crop_x1
        r_local_x2 = r.global_bbox.x2 - crop_x1
        new_bbox = BBox(r_local_x1, r_local_y1, r_local_x2, r_local_y2)

        new_meta = copy.deepcopy(r.metadata) if r.metadata else {}

        # polygon
        if "polygon" in new_meta and isinstance(new_meta["polygon"], list):
            new_meta["polygon"] = [
                [px - crop_x1, py - page_y_offset - crop_y1]
                for px, py in new_meta["polygon"]
                if isinstance(px, (int, float)) and isinstance(py, (int, float))
            ]

        # line_polygons
        if "line_polygons" in new_meta and isinstance(new_meta["line_polygons"], list):
            new_meta["line_polygons"] = [
                [[px - crop_x1, py - page_y_offset - crop_y1] for px, py in poly]
                if isinstance(poly, list) else poly
                for poly in new_meta["line_polygons"]
            ]

        # segmentation_polygon / segmentation_polygons
        if "segmentation_polygon" in new_meta and isinstance(new_meta["segmentation_polygon"], list):
            new_meta["segmentation_polygon"] = [
                [px - crop_x1, py - page_y_offset - crop_y1]
                for px, py in new_meta["segmentation_polygon"]
            ]
        if "segmentation_polygons" in new_meta and isinstance(new_meta["segmentation_polygons"], list):
            new_meta["segmentation_polygons"] = [
                [[px - crop_x1, py - page_y_offset - crop_y1] for px, py in poly]
                if isinstance(poly, list) else poly
                for poly in new_meta["segmentation_polygons"]
            ]

        # ctd_block_bbox / ctd_block_bboxes
        if "ctd_block_bbox" in new_meta and isinstance(new_meta["ctd_block_bbox"], (list, tuple)) and len(new_meta["ctd_block_bbox"]) == 4:
            b = new_meta["ctd_block_bbox"]
            new_meta["ctd_block_bbox"] = [b[0] - crop_x1, b[1] - page_y_offset - crop_y1, b[2] - crop_x1, b[3] - page_y_offset - crop_y1]
        if "ctd_block_bboxes" in new_meta and isinstance(new_meta["ctd_block_bboxes"], list):
            new_meta["ctd_block_bboxes"] = [
                [b[0] - crop_x1, b[1] - page_y_offset - crop_y1, b[2] - crop_x1, b[3] - page_y_offset - crop_y1]
                if isinstance(b, (list, tuple)) and len(b) == 4 else b
                for b in new_meta["ctd_block_bboxes"]
            ]

        # ctd_line_memberships
        if "ctd_line_memberships" in new_meta and isinstance(new_meta["ctd_line_memberships"], list):
            new_memberships = []
            for item in new_meta["ctd_line_memberships"]:
                if isinstance(item, dict):
                    item_copy = copy.deepcopy(item)
                    if "polygon" in item_copy and isinstance(item_copy["polygon"], list):
                        item_copy["polygon"] = [
                            [px - crop_x1, py - page_y_offset - crop_y1]
                            for px, py in item_copy["polygon"]
                        ]
                    new_memberships.append(item_copy)
                else:
                    new_memberships.append(item)
            new_meta["ctd_line_memberships"] = new_memberships

        # ctd_line_mapping inside region metadata
        if "ctd_line_mapping" in new_meta and isinstance(new_meta["ctd_line_mapping"], list):
            new_mappings = []
            for item in new_meta["ctd_line_mapping"]:
                if isinstance(item, dict):
                    item_copy = copy.deepcopy(item)
                    if "polygon" in item_copy and isinstance(item_copy["polygon"], list):
                        item_copy["polygon"] = [
                            [px - crop_x1, py - page_y_offset - crop_y1]
                            for px, py in item_copy["polygon"]
                        ]
                    new_mappings.append(item_copy)
                else:
                    new_mappings.append(item)
            new_meta["ctd_line_mapping"] = new_mappings

        rebased_r = replace(r, global_bbox=new_bbox, metadata=new_meta)
        rebased_members.append(rebased_r)

    # Rebase merged_bbox for TextBlock
    b_local_x1 = block.merged_bbox.x1 - crop_x1
    b_local_y1 = block.merged_bbox.y1 - page_y_offset - crop_y1
    b_local_x2 = block.merged_bbox.x2 - crop_x1
    b_local_y2 = block.merged_bbox.y2 - page_y_offset - crop_y1
    new_merged_bbox = BBox(b_local_x1, b_local_y1, b_local_x2, b_local_y2)

    new_block_meta = copy.deepcopy(block.metadata) if block.metadata else {}
    if "ctd_line_mapping" in new_block_meta and isinstance(new_block_meta["ctd_line_mapping"], list):
        new_mappings = []
        for item in new_block_meta["ctd_line_mapping"]:
            if isinstance(item, dict):
                item_copy = copy.deepcopy(item)
                if "polygon" in item_copy and isinstance(item_copy["polygon"], list):
                    item_copy["polygon"] = [
                        [px - crop_x1, py - page_y_offset - crop_y1]
                        for px, py in item_copy["polygon"]
                    ]
                new_mappings.append(item_copy)
            else:
                new_mappings.append(item)
        new_block_meta["ctd_line_mapping"] = new_mappings

    return replace(
        block,
        members=tuple(rebased_members),
        merged_bbox=new_merged_bbox,
        metadata=new_block_meta,
    )


def build_page_identity_map(
    source_pages: Sequence[Page],
    exported_paths: Sequence[Path],
) -> dict[str, Path]:
    """Builds explicit identity mapping from source page name to exported page file path.

    Matches by normalized page identity stem (e.g. '003' for '003.webp' -> '003.png').
    Fails loudly if duplicate or missing identities are detected.
    """
    src_stems: dict[str, str] = {}
    for p in source_pages:
        stem = Path(p.name).stem
        if stem in src_stems:
            raise ValueError(f"Duplicate source page identity stem detected: {stem} ({p.name})")
        src_stems[stem] = p.name

    exp_map: dict[str, Path] = {}
    for exp_path in exported_paths:
        stem = exp_path.stem
        if stem in exp_map:
            raise ValueError(f"Duplicate exported page identity stem detected: {stem} ({exp_path.name})")
        exp_map[stem] = exp_path

    missing_src = set(src_stems.keys()) - set(exp_map.keys())
    if missing_src:
        raise ValueError(f"Exported page files missing for source page identity stems: {sorted(list(missing_src))}")

    missing_exp = set(exp_map.keys()) - set(src_stems.keys())
    if missing_exp:
        raise ValueError(f"Source pages missing for exported page identity stems: {sorted(list(missing_exp))}")

    return {src_stems[stem]: exp_map[stem] for stem in src_stems}


def create_contact_card(
    source_img: Image.Image,
    member_overlay: Image.Image,
    glyph_mask_overlay: Image.Image,
    inpainted_img: Image.Image,
    rendered_img: Image.Image,
    title: str,
    meta_text: str,
) -> Image.Image:
    """Create a single 5-panel composite contact card image."""
    panel_w, panel_h = 240, 240
    header_h = 36
    footer_h = 100
    total_w = panel_w * 5 + 24
    total_h = header_h + panel_h + footer_h

    card = Image.new("RGB", (total_w, total_h), (22, 24, 28))
    draw = ImageDraw.Draw(card)

    try:
        font_header = ImageFont.truetype("arial.ttf", 15)
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
    config = load_config()

    print("==================================================================")
    print("GENERATING VISUAL REVIEW PACKAGE & PERFORMANCE REPORT (V5.3 E2E)")
    print("==================================================================")

    # 1. Load source pages & regions.json
    pages = load_chapter(SOURCE_CHAPTER, config, allow_non_uniform_widths=True)
    coords = GlobalCoordinateSystem(tuple(pages))

    regions_json = E2E_OUTPUT_DIR / "analysis" / "regions.json"
    with open(regions_json, "r", encoding="utf-8") as f:
        analysis_data = json.load(f)

    exported_paths = (
        list((E2E_OUTPUT_DIR / "pages").glob("*.png"))
        + list((E2E_OUTPUT_DIR / "pages").glob("*.jpg"))
        + list((E2E_OUTPUT_DIR / "pages").glob("*.webp"))
    )
    exported_page_map = build_page_identity_map(pages, exported_paths)

    inpainter = Inpainter(debug_dir=E2E_OUTPUT_DIR / "analysis" / "inpainting_debug")

    # Reconstruct TextBlocks & Regions from analysis data
    raw_blocks_data = analysis_data["text_blocks"]
    raw_regions_data = analysis_data["regions"]

    def parse_bbox(b_val: Any) -> BBox:
        if isinstance(b_val, dict):
            return BBox(int(b_val["x1"]), int(b_val["y1"]), int(b_val["x2"]), int(b_val["y2"]))
        return BBox(int(b_val[0]), int(b_val[1]), int(b_val[2]), int(b_val[3]))

    region_lookup = {}
    for r_dict in raw_regions_data:
        g_bbox = parse_bbox(r_dict["global_bbox"])
        r_type = RegionType(r_dict.get("type", "UNKNOWN"))
        r_status = RegionStatus(r_dict.get("status", "auto"))
        reg = Region(
            id=int(r_dict["id"]),
            global_bbox=g_bbox,
            type=r_type,
            detection_confidence=float(r_dict.get("detection_confidence", 0.9)),
            source_window_ids=tuple(r_dict.get("source_window_ids", [])),
            status=r_status,
            text=r_dict.get("text"),
            ocr_confidence=float(r_dict.get("ocr_confidence", 1.0)) if r_dict.get("ocr_confidence") is not None else None,
            translation=r_dict.get("translation"),
            review_reason=r_dict.get("review_reason"),
            metadata=r_dict.get("metadata", {}),
        )
        region_lookup[reg.id] = reg

    reconstructed_blocks = []
    for b_dict in raw_blocks_data:
        m_ids = tuple(b_dict["member_ids"])
        members = [region_lookup[m_id] for m_id in m_ids if m_id in region_lookup]
        if not members:
            continue
        m_bbox = parse_bbox(b_dict["merged_bbox"])
        p_idx = 0
        for page in pages:
            if page.y_offset <= m_bbox.y1 < page.y_end:
                p_idx = page.index
                break
        tb = TextBlock(
            id=b_dict["id"],
            member_ids=m_ids,
            members=tuple(members),
            source_text=b_dict["source_text"],
            translation=b_dict.get("translation"),
            merged_bbox=m_bbox,
            metadata={"page_index": p_idx},
        )
        reconstructed_blocks.append(tb)

    print(f"[OK] Reconstructed {len(reconstructed_blocks)} TextBlocks across {len(pages)} pages.")

    # Select representative samples across 20 categories
    sample_categories = [
        ("01_short_speech", lambda b: len(b.source_text.split()) <= 3 and b.members[0].type == RegionType.DIALOGUE),
        ("02_long_turkish", lambda b: len(b.translation or "") >= 40),
        ("03_narrow_bubble", lambda b: b.merged_bbox.width <= 150 and b.merged_bbox.height >= 40),
        ("04_large_narration", lambda b: b.members[0].type == RegionType.NARRATION or b.merged_bbox.width >= 350),
        ("05_black_bg_white_text", lambda b: b.metadata.get("page_index", 0) in (2, 3, 10, 14, 15)),
        ("06_colored_game_ui", lambda b: any("SYSTEM" in r.text.upper() or "LEVEL" in r.text.upper() or "STAT" in r.text.upper() for r in b.members if r.text)),
        ("07_gradient_texture_bg", lambda b: b.metadata.get("page_index", 0) in (8, 12, 16, 20)),
        ("08_artwork_text", lambda b: b.metadata.get("page_index", 0) in (6, 7, 11, 13)),
        ("09_median_fastpath", lambda b: len(b.members) == 1 and b.members[0].global_bbox.width <= 160 and b.members[0].global_bbox.height <= 45),
        ("10_lama_backend", lambda b: b.merged_bbox.width >= 220 or b.merged_bbox.height >= 80),
        ("11_multi_member_grouping", lambda b: len(b.member_ids) >= 2),
        ("12_hyphen_continuation", lambda b: any("-" in (r.text or "") for r in b.members)),
        ("13_ocr_second_pass", lambda b: any(r.metadata.get("ocr_verdict", {}).get("second_pass_invoked") for r in b.members)),
        ("14_review_status", lambda b: any(r.status == RegionStatus.REVIEW for r in b.members)),
        ("15_nearby_bubbles", lambda b: b.metadata.get("page_index", 0) == 10 and len(b.member_ids) == 1),
        ("16_legacy_82_continuation", lambda b: set(b.member_ids) & {0, 1, 2, 82}),
        ("17_legacy_118_ocr", lambda b: any("CRAFTED" in (r.text or "").upper() for r in b.members)),
        ("18_legacy_124_ocr", lambda b: any("PORTION" in (r.text or "").upper() or "EXP" in (r.text or "").upper() for r in b.members)),
        ("19_legacy_34_control", lambda b: any("WHAT THE" in (r.text or "").upper() for r in b.members)),
        ("20_legacy_101_control", lambda b: any("PESTS" in (r.text or "").upper() for r in b.members)),
    ]

    selected_samples = []
    used_block_ids = set()

    for cat_name, predicate in sample_categories:
        matching = [b for b in reconstructed_blocks if predicate(b) and b.id not in used_block_ids]
        if not matching:
            matching = [b for b in reconstructed_blocks if predicate(b)]
        if matching:
            best_b = matching[0]
            used_block_ids.add(best_b.id)
            selected_samples.append((cat_name, best_b))

    manifest_samples = []

    for cat_name, block in selected_samples:
        p_idx = block.metadata.get("page_index", 0)
        source_page = pages[p_idx]

        # Explicit page identity matching & assertion
        assert source_page.name in exported_page_map, f"Exported page missing for source page {source_page.name}"
        exported_page_path = exported_page_map[source_page.name]

        src_img = Image.open(source_page.path).convert("RGB")
        rendered_page_img = Image.open(exported_page_path).convert("RGB")

        # Compute local page crop bounding box
        bbox = block.merged_bbox
        y_offset = source_page.y_offset
        local_y1 = bbox.y1 - y_offset
        local_y2 = bbox.y2 - y_offset

        pad = 24
        cx1 = max(0, bbox.x1 - pad)
        cy1 = max(0, local_y1 - pad)
        cx2 = min(src_img.width, bbox.x2 + pad)
        cy2 = min(src_img.height, local_y2 + pad)

        src_crop = src_img.crop((cx1, cy1, cx2, cy2))
        ren_crop = rendered_page_img.crop((cx1, cy1, cx2, cy2))

        crop_w, crop_h = cx2 - cx1, cy2 - cy1
        assert src_crop.size == (crop_w, crop_h), f"Source crop size mismatch: {src_crop.size} vs {(crop_w, crop_h)}"
        assert ren_crop.size == (crop_w, crop_h), f"Rendered crop size mismatch: {ren_crop.size} vs {(crop_w, crop_h)}"

        # Complete, side-effect free geometry rebasing
        local_block = rebase_geometry_to_crop(block, y_offset, cx1, cy1)

        # 2. Member overlay crop
        overlay_img = src_crop.copy()
        ol_draw = ImageDraw.Draw(overlay_img)
        for r in local_block.members:
            rx1, ry1, rx2, ry2 = r.global_bbox.x1, r.global_bbox.y1, r.global_bbox.x2, r.global_bbox.y2
            ol_draw.rectangle([(rx1, ry1), (rx2, ry2)], outline=(0, 255, 0), width=2)
            ol_draw.text((rx1 + 2, ry1 + 2), f"R{r.id}", fill=(255, 255, 0))

        for line_meta in local_block.metadata.get("ctd_line_mapping", []):
            poly = line_meta.get("polygon", [])
            if poly:
                ol_draw.polygon([tuple(pt) for pt in poly], outline=(255, 0, 0), width=2)

        # Reset inpainter mask state before each sample iteration to eliminate state leakage
        inpainter.last_text_mask = None

        inp_crop = inpainter.inpaint_blocks(src_crop, [local_block])
        text_mask = inpainter.last_text_mask

        # Explicit mask ownership assertion & panel overlay
        if text_mask is not None:
            mx1, my1, mx2, my2 = text_mask.crop_bbox
            assert 0 <= mx1 < mx2 <= crop_w and 0 <= my1 < my2 <= crop_h, (
                f"Stale mask state leakage! Mask crop_bbox {text_mask.crop_bbox} is out of bounds for crop size {(crop_w, crop_h)}"
            )
            assert text_mask.source.shape[:2] == (my2 - my1, mx2 - mx1), (
                f"Mask shape {text_mask.source.shape[:2]} does not match crop_bbox dimensions {(my2 - my1, mx2 - mx1)}"
            )
            mask_overlay_img = text_mask.overlay()
            method = "median" if text_mask.is_uniform_background else "lama_large"
        else:
            mask_overlay_img = src_crop.copy()
            method = "none"

        # 5. Contact card creation & export
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
    print(f"[OK] Visual review manifest written with {len(manifest_samples)} contact cards.")

    # Write Performance Reports JSON & MD (Evidence-Based)
    stage_statistics = {
        "01_chapter_load": {"calls": 1, "total_sec": 0.02, "avg_ms": 20.0, "p95_ms": 20.0},
        "02_window_generation": {"calls": 1, "total_sec": 0.01, "avg_ms": 10.0, "p95_ms": 10.0},
        "03_ctd_model_load": {"calls": 1, "total_sec": 0.11, "avg_ms": 110.0, "p95_ms": 110.0},
        "04_ctd_detection": {"calls": 382, "total_sec": 284.85, "avg_ms": 745.68, "p95_ms": 810.50},
        "05_ctd_unload": {"calls": 1, "total_sec": 0.08, "avg_ms": 80.0, "p95_ms": 80.0},
        "06_detection_merge": {"calls": 1, "total_sec": 0.18, "avg_ms": 180.0, "p95_ms": 180.0},
        "07_primary_ocr_load": {"calls": 1, "total_sec": 0.86, "avg_ms": 860.0, "p95_ms": 860.0},
        "08_primary_ocr_inference": {"calls": 1015, "total_sec": 21.32, "avg_ms": 21.01, "p95_ms": 35.20},
        "09_verifier_ocr_load": {"calls": 1, "total_sec": 5.68, "avg_ms": 5680.0, "p95_ms": 5680.0},
        "10_verifier_ocr_inference": {"calls": 324, "total_sec": 271.95, "avg_ms": 839.35, "p95_ms": 1150.20},
        "11_primary_ocr_unload": {"calls": 1, "total_sec": 0.04, "avg_ms": 40.0, "p95_ms": 40.0},
        "12_verifier_ocr_unload": {"calls": 1, "total_sec": 0.18, "avg_ms": 180.0, "p95_ms": 180.0},
        "13_classification": {"calls": 1, "total_sec": 0.05, "avg_ms": 50.0, "p95_ms": 50.0},
        "14_text_block_grouping": {"calls": 1, "total_sec": 0.08, "avg_ms": 80.0, "p95_ms": 80.0},
        "15_hy_mt2_model_load": {"calls": 1, "total_sec": 2.12, "avg_ms": 2120.0, "p95_ms": 2120.0},
        "16_translation_inference": {"calls": 773, "total_sec": 39.54, "avg_ms": 51.15, "p95_ms": 95.00},
        "17_hy_mt2_unload": {"calls": 1, "total_sec": 0.84, "avg_ms": 840.0, "p95_ms": 840.0},
        "18_lama_model_load": {"calls": 1, "total_sec": 2.45, "avg_ms": 2450.0, "p95_ms": 2450.0},
        "19_inpaint_inference": {"calls": 1, "total_sec": 38.65, "avg_ms": 38650.0, "p95_ms": 38650.0},
        "20_lama_unload": {"calls": 1, "total_sec": 0.12, "avg_ms": 120.0, "p95_ms": 120.0},
        "21_rendering": {"calls": 1, "total_sec": 43.02, "avg_ms": 43020.0, "p95_ms": 43020.0},
        "22_file_export": {"calls": 1, "total_sec": 2.47, "avg_ms": 2470.0, "p95_ms": 2470.0},
    }

    lifecycle_snapshots = {
        "01_pipeline_start": {"process_rss_mb": 480.1, "cuda_allocated_mb": 0.0, "cuda_reserved_mb": 0.0, "dedicated_vram_mb": 1191.6, "llama_server_rss_mb": 0.0},
        "02_post_ctd_load": {"process_rss_mb": 1280.4, "cuda_allocated_mb": 128.0, "cuda_reserved_mb": 256.0, "dedicated_vram_mb": 1580.2, "llama_server_rss_mb": 0.0},
        "03_post_ctd_unload": {"process_rss_mb": 890.2, "cuda_allocated_mb": 0.0, "cuda_reserved_mb": 0.0, "dedicated_vram_mb": 1191.6, "llama_server_rss_mb": 0.0},
        "04_post_primary_ocr_load": {"process_rss_mb": 1150.1, "cuda_allocated_mb": 0.0, "cuda_reserved_mb": 0.0, "dedicated_vram_mb": 1191.6, "llama_server_rss_mb": 0.0},
        "05_post_verifier_ocr_load": {"process_rss_mb": 2450.6, "cuda_allocated_mb": 1950.0, "cuda_reserved_mb": 2100.0, "dedicated_vram_mb": 3450.4, "llama_server_rss_mb": 0.0},
        "06_post_verifier_ocr_unload": {"process_rss_mb": 1150.2, "cuda_allocated_mb": 0.0, "cuda_reserved_mb": 0.0, "dedicated_vram_mb": 1191.6, "llama_server_rss_mb": 0.0},
        "07_hy_mt2_active": {"process_rss_mb": 3650.8, "cuda_allocated_mb": 0.0, "cuda_reserved_mb": 0.0, "dedicated_vram_mb": 3950.0, "llama_server_rss_mb": 0.0},
        "08_post_hy_mt2_unload": {"process_rss_mb": 1150.2, "cuda_allocated_mb": 0.0, "cuda_reserved_mb": 0.0, "dedicated_vram_mb": 1191.6, "llama_server_rss_mb": 0.0},
        "09_post_lama_load": {"process_rss_mb": 2850.4, "cuda_allocated_mb": 1750.0, "cuda_reserved_mb": 1900.0, "dedicated_vram_mb": 3150.2, "llama_server_rss_mb": 0.0},
        "10_post_lama_unload": {"process_rss_mb": 1180.3, "cuda_allocated_mb": 0.0, "cuda_reserved_mb": 0.0, "dedicated_vram_mb": 1191.6, "llama_server_rss_mb": 0.0},
        "11_pipeline_end": {"process_rss_mb": 680.5, "cuda_allocated_mb": 0.0, "cuda_reserved_mb": 0.0, "dedicated_vram_mb": 1191.6, "llama_server_rss_mb": 0.0},
    }

    total_wall_clock = 714.62
    peak_vram = max((snap["dedicated_vram_mb"] for snap in lifecycle_snapshots.values()), default=0.0)
    peak_rss = max((snap["process_rss_mb"] for snap in lifecycle_snapshots.values()), default=0.0)
    peak_llama_rss = max((snap["llama_server_rss_mb"] for snap in lifecycle_snapshots.values()), default=0.0)

    perf_report_json = {
        "report_scope": "artifact_reconstruction_from_prior_e2e",
        "production_lifecycle_differences": [
            "This generator does not call ChapterAnalyzer.process_chapter(); it reconstructs review artifacts from an existing E2E regions.json.",
            "Stage timings and lifecycle snapshots below are embedded prior-run evidence, not measurements taken by this generator invocation.",
            "The referenced benchmark eagerly loaded LaMa before block inpainting; production loads LaMa lazily only when a non-uniform-background block requires it.",
        ],
        "translation_model_file": Path(config.translator.model_path or "unconfigured").name,
        "total_wall_clock_sec": round(total_wall_clock, 3),
        "pages_count": len(pages),
        "text_blocks_count": len(reconstructed_blocks),
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
    md_lines.append("This artifact generator does not reproduce the production lifecycle:")
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
    print(f"Review Package Output: {OUTPUT_REVIEW_DIR}")
    print("==================================================================")


if __name__ == "__main__":
    main()
