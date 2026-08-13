"""Generate a bounded 13-block CTD-mask/LaMa/render validation set (never full-chapter E2E)."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.detection import BBox, Region, RegionStatus, RegionType
from core.detection.text_block import TextBlock
from core.imaging.inpainter import Inpainter
from core.imaging.renderer import TextRenderer
from core.io.input_loader import load_chapter
from providers.detector.ctd import ComicTextDetector


SOURCE = Path(r"C:\Users\Ahmed\AppData\Local\Tachidesk\downloads\mangas\Asmodeus Scans (EN)\Reincarnated as a God-Tier Crafter\Chapter 1")
ANALYSIS = ROOT / "e2e_output" / "real_tachidesk_chapter_1" / "analysis" / "regions.json"
OUTPUT = ROOT / "review_output" / "text_mask_validation_v5"
SELECTED_BLOCK_IDS = (7, 13, 18, 34, 61, 80, 82, 101, 118, 124, 132, 152, 181)


def _bbox(data) -> BBox:
    if isinstance(data, dict):
        return BBox(int(data["x1"]), int(data["y1"]), int(data["x2"]), int(data["y2"]))
    return BBox(*(int(value) for value in data))


def _intersection_area(a: BBox, b: BBox) -> int:
    intersection = a.intersection(b)
    return intersection.area if intersection is not None else 0


def _page_index(global_y: int, page_ranges: list[tuple[int, int]]) -> int:
    for index, (start, end) in enumerate(page_ranges):
        if start <= global_y < end:
            return index
    raise ValueError(f"No page for global y={global_y}")


def _font(size: int = 18):
    path = Path(r"C:\Windows\Fonts\arial.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def _contact_sheet(images: list[tuple[str, Image.Image]]) -> Image.Image:
    thumb_w, thumb_h = 360, 360
    sheet = Image.new("RGB", (thumb_w * len(images), thumb_h + 34), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (label, image) in enumerate(images):
        preview = image.copy().convert("RGB")
        preview.thumbnail((thumb_w - 8, thumb_h - 8), Image.Resampling.LANCZOS)
        x = index * thumb_w + (thumb_w - preview.width) // 2
        y = 34 + (thumb_h - preview.height) // 2
        sheet.paste(preview, (x, y))
        draw.text((index * thumb_w + 8, 7), label, fill="black", font=_font(17))
    return sheet


def _ctd_geometry(detector: ComicTextDetector, page: Image.Image, block_bbox: BBox, sample_id: int) -> tuple[list, list, dict]:
    pad_x = max(24, round(block_bbox.width * 0.25))
    pad_y = max(24, round(block_bbox.height * 0.25))
    x1, y1 = max(0, block_bbox.x1 - pad_x), max(0, block_bbox.y1 - pad_y)
    x2, y2 = min(page.width, block_bbox.x2 + pad_x), min(page.height, block_bbox.y2 + pad_y)
    crop = page.crop((x1, y1, x2, y2))
    target_local = BBox(block_bbox.x1 - x1, block_bbox.y1 - y1, block_bbox.x2 - x1, block_bbox.y2 - y1)
    target_expanded = BBox(
        max(0, target_local.x1 - max(8, round(block_bbox.width * 0.08))),
        max(0, target_local.y1 - pad_y),
        min(crop.width, target_local.x2 + max(8, round(block_bbox.width * 0.08))),
        min(crop.height, target_local.y2 + pad_y),
    )
    detections = list(detector.detect(crop, sample_id))
    candidates = []
    for detection in detections:
        overlap = _intersection_area(detection.bbox, target_expanded)
        if overlap <= 0:
            continue
        containment = overlap / max(1, min(detection.bbox.area, target_expanded.area))
        candidates.append((containment, overlap, detection))
    if not candidates:
        return [], [], dict(detector.last_output_metadata)
    # Keep line and segmentation geometry paired to the same canonical CTD block.
    _, _, selected = max(candidates, key=lambda item: (item[0], item[1], item[2].confidence))
    lines: list = []
    segmentation: list = []
    for detection in (selected,):
        for key, target in (("line_polygons", lines), ("segmentation_polygons", segmentation)):
            polygons = detection.metadata.get(key, [])
            for polygon in polygons:
                shifted = [[float(point[0]) + x1, float(point[1]) + y1] for point in polygon]
                poly_bbox = BBox(
                    int(min(point[0] for point in shifted)), int(min(point[1] for point in shifted)),
                    int(max(point[0] for point in shifted)) + 1, int(max(point[1] for point in shifted)) + 1,
                )
                if _intersection_area(poly_bbox, BBox(x1 + target_expanded.x1, y1 + target_expanded.y1,
                                                       x1 + target_expanded.x2, y1 + target_expanded.y2)) > 0:
                    target.append(shifted)
    return lines, segmentation, dict(detector.last_output_metadata)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = json.loads(ANALYSIS.read_text(encoding="utf-8"))
    raw_regions = {int(region["id"]): region for region in data["regions"]}
    raw_blocks = {int(block["id"]): block for block in data["text_blocks"]}
    pages = load_chapter(SOURCE, allow_non_uniform_widths=True)
    page_ranges = [(page.y_offset, page.y_offset + page.height) for page in pages]

    detector = ComicTextDetector()
    detector.load()
    inpainter = Inpainter(debug_dir=OUTPUT / "debug")
    renderer = TextRenderer()
    manifest: list[dict] = []
    try:
        for ordinal, block_id in enumerate(SELECTED_BLOCK_IDS, start=1):
            raw_block = raw_blocks[block_id]
            global_box = _bbox(raw_block["merged_bbox"])
            page_index = _page_index((global_box.y1 + global_box.y2) // 2, page_ranges)
            page_info = pages[page_index]
            page_start = page_info.y_offset
            with Image.open(page_info.path) as opened:
                page = opened.convert("RGB")
                if page.size != (page_info.width, page_info.height):
                    page = page.resize((page_info.width, page_info.height), Image.Resampling.LANCZOS)
            page_box = BBox(global_box.x1, global_box.y1 - page_start, global_box.x2, global_box.y2 - page_start)
            line_polygons, seg_polygons, onnx_metadata = _ctd_geometry(detector, page, page_box, ordinal)

            members: list[Region] = []
            for member_index, member_id in enumerate(raw_block["member_ids"]):
                raw_region = raw_regions[int(member_id)]
                global_member = _bbox(raw_region["global_bbox"])
                member_box = BBox(global_member.x1, global_member.y1 - page_start, global_member.x2, global_member.y2 - page_start)
                metadata = dict(raw_region.get("metadata") or {})
                # A fresh CTD canonical Region owns the complete block geometry.
                # Put it on one validation member to mimic that contract instead of
                # fragmenting it according to legacy V4 line bboxes.
                if member_index == 0:
                    if line_polygons:
                        metadata["line_polygons"] = line_polygons
                    if seg_polygons:
                        metadata["segmentation_polygons"] = seg_polygons
                members.append(Region(
                    id=int(member_id), global_bbox=member_box,
                    type=RegionType(raw_region.get("type", "unknown")),
                    detection_confidence=float(raw_region.get("detection_confidence", 1.0)),
                    source_window_ids=tuple(raw_region.get("source_window_ids", ())),
                    status=RegionStatus(raw_region.get("status", "auto")),
                    text=raw_region.get("text"), metadata=metadata,
                ))
            block = TextBlock(
                id=block_id,
                member_ids=tuple(region.id for region in members),
                members=tuple(members), merged_bbox=page_box,
                source_text=raw_block["source_text"], translation=raw_block.get("translation"),
                metadata={"validation": True, "page_index": page_index},
            )

            page_np = np.asarray(page, dtype=np.uint8)
            text_mask = inpainter.mask_builder.build_for_block(page_np, block)
            inpainted_np = inpainter._apply_mask(page_np, text_mask, f"sample_{ordinal:02d}_block_{block_id}")
            text_mask = inpainter.last_text_mask or text_mask
            if np.any(text_mask.refined):
                rendered, *_, _ = renderer.render_blocks(Image.fromarray(inpainted_np, "RGB"), [(block, raw_block["translation"])])
            else:
                rendered = Image.fromarray(inpainted_np, "RGB")
            x1, y1, x2, y2 = text_mask.crop_bbox
            source_crop = page.crop((x1, y1, x2, y2))
            inpainted_crop = Image.fromarray(inpainted_np[y1:y2, x1:x2], "RGB")
            rendered_crop = rendered.crop((x1, y1, x2, y2))
            sample_dir = OUTPUT / f"{ordinal:02d}_block_{block_id}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            source_crop.save(sample_dir / "1_source.png")
            predicted_debug = text_mask.predicted_segmentation if text_mask.predicted_segmentation is not None else np.zeros_like(text_mask.raw)
            glyph_debug = text_mask.glyph_refined if text_mask.glyph_refined is not None else np.zeros_like(text_mask.raw)
            Image.fromarray(predicted_debug, "L").save(sample_dir / "2_raw_ctd_segmentation.png")
            line_vis = source_crop.copy()
            line_draw = ImageDraw.Draw(line_vis)
            for polygon in line_polygons:
                line_draw.polygon([(point[0] - x1, point[1] - y1) for point in polygon], outline="red", width=2)
            line_vis.save(sample_dir / "3_dbnet_line_polygons.png")
            Image.fromarray(glyph_debug, "L").save(sample_dir / "4_upstream_glyph_mask.png")
            Image.fromarray(text_mask.refined, "L").save(sample_dir / "5_final_protected_mask.png")
            text_mask.overlay().save(sample_dir / "6_mask_overlay.png")
            inpainted_crop.save(sample_dir / "7_inpainted.png")
            rendered_crop.save(sample_dir / "8_turkish_rendered.png")
            _contact_sheet([
                ("1 Source", source_crop),
                ("2 Raw CTD seg", Image.fromarray(predicted_debug, "L").convert("RGB")),
                ("3 DBNet lines", line_vis),
                ("4 Glyph mask", Image.fromarray(glyph_debug, "L").convert("RGB")),
                ("5 Final mask", Image.fromarray(text_mask.refined, "L").convert("RGB")),
                ("6 Overlay", text_mask.overlay()),
                ("7 Inpainted", inpainted_crop),
                ("8 Turkish rendered", rendered_crop),
            ]).save(sample_dir / "contact_sheet.png")

            full_mask = np.zeros(page_np.shape[:2], dtype=bool)
            full_mask[y1:y2, x1:x2] = text_mask.refined > 0
            record = inpainter.debug_records[-1]
            line_region_mapping = []
            mapped_member_ids: set[int] = set()
            for line_index, polygon in enumerate(line_polygons):
                line_box = BBox(
                    int(min(point[0] for point in polygon)), int(min(point[1] for point in polygon)),
                    int(max(point[0] for point in polygon)) + 1, int(max(point[1] for point in polygon)) + 1,
                )
                candidates = []
                for member in members:
                    overlap = _intersection_area(line_box, member.global_bbox)
                    if overlap > 0:
                        candidates.append((overlap / max(1, min(line_box.area, member.global_bbox.area)), member.id))
                member_ids_for_line = [member_id for ratio, member_id in candidates if ratio >= .08]
                mapped_member_ids.update(member_ids_for_line)
                line_region_mapping.append({"line_index": line_index, "region_ids": member_ids_for_line})
            review_reasons = []
            if any(not item["region_ids"] for item in line_region_mapping):
                review_reasons.append("ctd_line_without_region_member")
            unmapped_members = sorted(set(block.member_ids) - mapped_member_ids)
            if line_polygons and unmapped_members:
                review_reasons.append(f"source_member_without_ctd_line:{unmapped_members}")
            if record["review"]:
                review_reasons.append("residual_boundary_after_second_pass")
            manifest.append({
                "ordinal": ordinal, "block_id": block_id, "page": page_info.path.name,
                "source_text": raw_block["source_text"], "translation": raw_block["translation"],
                "member_ids": raw_block["member_ids"], "ctd_line_polygon_count": len(line_polygons),
                "onnx_outputs": onnx_metadata,
                "line_region_mapping": line_region_mapping,
                "review_reason": review_reasons or None,
                "method": record["method"], "mask_pixels": record["mask_pixels"],
                "bubble_found": record["bubble_found"],
                "adaptive_dilation": record["adaptive_dilation"],
                "protected_pixels": record["protected_pixels"],
                "second_pass": record["second_pass"],
                "review": bool(record["review"] or review_reasons),
                "outside_mask_pixel_identical": bool(np.array_equal(inpainted_np[~full_mask], page_np[~full_mask])),
                "contact_sheet": str((sample_dir / "contact_sheet.png").relative_to(OUTPUT)),
            })
    finally:
        detector.unload()
        inpainter.unload()

    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "sample_count": len(manifest),
        "all_outside_mask_pixel_identical": all(item["outside_mask_pixel_identical"] for item in manifest),
        "median_fast_path_count": sum(item["method"] == "median" for item in manifest),
        "lama_large_count": sum(item["method"] == "lama_large" for item in manifest),
        "selected_block_ids": list(SELECTED_BLOCK_IDS),
        "full_chapter_e2e_run": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(OUTPUT.resolve())


if __name__ == "__main__":
    main()
