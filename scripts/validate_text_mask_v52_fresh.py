"""Fresh, cache-free validation of selected Chapter 1 source-page crops."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.coordinate.global_coords import GlobalCoordinateSystem
from core.detection import BBox, RegionStatus, RegionType
from core.detection.classification import classify_regions
from core.detection.merge import merge_duplicates
from core.detection.text_block import group_text_blocks
from core.imaging.inpainter import Inpainter
from core.imaging.renderer import TextRenderer
from core.models import Page
from providers.detector.ctd import ComicTextDetector
from providers.ocr.agreement import decide_ocr_agreement
from providers.ocr.paddleocr import PaddleOCRProvider
from providers.ocr.paddleocr_vl import PaddleOCRVLOcrProvider
from providers.ocr.qwen_repair import QwenRepairProvider
from providers.ocr.repair import OCRRepairInput
from providers.translation.base import TranslationInput, TranslationItem
from providers.translation.hy_mt2_gguf_translation import HyMT2GGUFTranslationProvider

SOURCE = Path(r"C:\Users\Ahmed\AppData\Local\Tachidesk\downloads\mangas\Asmodeus Scans (EN)\Reincarnated as a God-Tier Crafter\Chapter 1")
V52 = ROOT / "review_output" / "text_mask_validation_v5_1"
OUTPUT = ROOT / "review_output" / "text_mask_validation_v5_2_fresh"
LEGACY_IDS = (13, 34, 61, 80, 82, 101, 118, 124, 152)
PAGE_BY_ID = {
    13: "003.png", 34: "006.png", 61: "009.png",
    80: "011.png", 82: "011.png", 101: "013.png",
    118: "015.png", 124: "015.png", 152: "020.png",
}


def _anchor(legacy_id: int, page: Image.Image) -> tuple[int, int, int, int]:
    sample = next(V52.glob(f"*_block_{legacy_id}/1_source.png"))
    template = cv2.cvtColor(np.asarray(Image.open(sample).convert("RGB")), cv2.COLOR_RGB2GRAY)
    gray = cv2.cvtColor(np.asarray(page), cv2.COLOR_RGB2GRAY)
    result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    _, score, _, point = cv2.minMaxLoc(result)
    if score < .88:
        raise RuntimeError(f"anchor {legacy_id} template score too low: {score:.3f}")
    return point[0], point[1], point[0] + template.shape[1], point[1] + template.shape[0]


def _offset_metadata(metadata: dict, dx: int, dy: int) -> dict:
    result = dict(metadata)
    for key in ("line_polygons", "segmentation_polygons"):
        if isinstance(result.get(key), list):
            result[key] = [[[float(x) + dx, float(y) + dy] for x, y in polygon] for polygon in result[key]]
    for key in ("ctd_block_bbox",):
        value = result.get(key)
        if isinstance(value, list) and len(value) == 4:
            result[key] = [value[0] + dx, value[1] + dy, value[2] + dx, value[3] + dy]
    return result


def _intersects(a: BBox, b: BBox) -> bool:
    return a.intersection(b) is not None


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    detector, primary, verifier = ComicTextDetector(), PaddleOCRProvider("PP-OCRv6_medium_rec"), PaddleOCRVLOcrProvider()
    repair, translator = QwenRepairProvider(), HyMT2GGUFTranslationProvider()
    inpainter = Inpainter(debug_dir=OUTPUT / "debug")
    renderer = TextRenderer()
    manifest = []
    detector.load(); primary.load()
    verifier_loaded = repair_loaded = translator_loaded = False
    try:
        for fresh_index, legacy_id in enumerate(LEGACY_IDS, 1):
            page_path = SOURCE / PAGE_BY_ID[legacy_id]
            page = Image.open(page_path).convert("RGB")
            anchor = _anchor(legacy_id, page)
            ax1, ay1, ax2, ay2 = anchor
            pad_x, pad_y = max(80, (ax2 - ax1) // 2), max(80, (ay2 - ay1) // 2)
            rx1, ry1 = max(0, ax1 - pad_x), max(0, ay1 - pad_y)
            rx2, ry2 = min(page.width, ax2 + pad_x), min(page.height, ay2 + pad_y)
            roi = page.crop((rx1, ry1, rx2, ry2))
            target = BBox(ax1 - rx1, ay1 - ry1, ax2 - rx1, ay2 - ry1)

            detections = []
            for det in detector.detect(roi, fresh_index):
                if not _intersects(det.bbox, target):
                    continue
                detections.append(replace(det, metadata=_offset_metadata(det.metadata, 0, 0)))
            regions = merge_duplicates(detections, min_confidence=.4)
            fresh_regions = []
            ocr_outputs = []
            repair_queue = []
            for region in regions:
                crop_box = region.global_bbox
                crop = roi.crop((max(0, crop_box.x1 - 16), max(0, crop_box.y1 - 16),
                                 min(roi.width, crop_box.x2 + 16), min(roi.height, crop_box.y2 + 16)))
                p = primary.recognize(crop, region_bbox=region.global_bbox)
                verdict = decide_ocr_agreement(p, None)
                v = None
                if verdict.requires_review:
                    if not verifier_loaded:
                        verifier.load(); verifier_loaded = True
                    v = verifier.recognize(crop, region_bbox=region.global_bbox)
                    verdict = decide_ocr_agreement(p, v)
                text = verdict.accepted_text or verdict.provisional_text or p.text or ""
                status = RegionStatus.REVIEW if verdict.requires_review else RegionStatus.AUTO
                updated = replace(region, type=RegionType.DIALOGUE, text=text, ocr_confidence=p.confidence,
                                  status=status, review_reason=verdict.reason,
                                  metadata={**region.metadata, "fresh_ocr": {"primary": p.text, "verifier": v.text if v else None}})
                fresh_regions.append(updated)
                ocr_outputs.append({"region_id": updated.id, "primary": p.text, "verifier": v.text if v else None,
                                    "accepted": text, "status": status.value, "reason": verdict.reason})
                if verdict.needs_repair and text:
                    repair_queue.append((updated.id, OCRRepairInput(verdict.primary_raw, verdict.primary_normalized,
                                                                    verdict.verifier_raw, verdict.verifier_normalized,
                                                                    verdict.reason or "disagreement"), crop))
            if repair_queue:
                if not repair_loaded:
                    repair.load(); repair_loaded = True
                repaired = {r.id: r for r in fresh_regions}
                for region_id, repair_input, crop in repair_queue:
                    result = repair.repair(repair_input, crop)
                    if result.repaired_text and not result.unresolved:
                        repaired[region_id] = replace(repaired[region_id], text=result.repaired_text,
                                                      status=RegionStatus.AUTO, review_reason=None)
                fresh_regions = [repaired[r.id] for r in fresh_regions]

            local_page = Page(0, page_path, roi.width, roi.height, 0)
            coords = GlobalCoordinateSystem((local_page,))
            classified = classify_regions(fresh_regions, coords)
            blocks = group_text_blocks(classified, coords)
            candidates = [b for b in blocks if _intersects(b.merged_bbox, target)]
            block = max(candidates, key=lambda b: b.merged_bbox.intersection(target).area) if candidates else None
            review_reasons = []
            if block is None:
                review_reasons.append("no_fresh_textblock_for_anchor")
            source_text = block.source_text if block else ""
            translation = None
            if block and source_text:
                if not translator_loaded:
                    if repair_loaded:
                        repair.unload(); repair_loaded = False
                    translator.load(); translator_loaded = True
                output = translator.translate(TranslationInput([TranslationItem(block.id, source_text)]))
                translation = output.results[0].translation if output.results else None
                if output.results and output.results[0].requires_review:
                    review_reasons.append("translation_review")
            if translator_loaded:
                translator.unload(); translator_loaded = False

            source_np = np.asarray(roi, np.uint8)
            if block and translation:
                text_mask = inpainter.mask_builder.build_for_block(source_np, block)
                inpainted_np = inpainter._apply_mask(source_np, text_mask, f"fresh_{legacy_id}")
                text_mask = inpainter.last_text_mask or text_mask
                rendered, *_, _ = renderer.render_blocks(Image.fromarray(inpainted_np), [(block, translation)])
                record = inpainter.debug_records[-1]
                if record["review"]:
                    review_reasons.append("residual_boundary_after_second_pass")
            else:
                text_mask = None; inpainted_np = source_np; rendered = roi; record = {"method": "none"}

            sample = OUTPUT / f"{fresh_index:02d}_legacy_{legacy_id}"
            sample.mkdir(parents=True, exist_ok=True)
            roi.save(sample / "1_source.png")
            line_overlay = roi.copy(); draw = ImageDraw.Draw(line_overlay)
            region_overlay = roi.copy(); rdraw = ImageDraw.Draw(region_overlay)
            line_ids = []
            for region in classified:
                for line in region.metadata.get("ctd_line_memberships", []):
                    polygon = line.get("polygon", [])
                    if polygon:
                        draw.polygon([(x, y) for x, y in polygon], outline="red", width=2)
                        line_ids.append(line.get("line_id"))
                box = region.global_bbox
                rdraw.rectangle(box.to_tuple(), outline="lime", width=2)
                rdraw.text((box.x1, box.y1), f"R{region.id}: {region.text}", fill="yellow")
            line_overlay.save(sample / "2_detected_ctd_lines.png")
            region_overlay.save(sample / "3_region_member_overlay.png")
            (sample / "4_source_text.txt").write_text(source_text, encoding="utf-8")
            (sample / "5_ocr_members.json").write_text(json.dumps(ocr_outputs, ensure_ascii=False, indent=2), encoding="utf-8")
            (sample / "6_textblock_members.json").write_text(json.dumps({"member_ids": list(block.member_ids) if block else [],
                                                                          "source_text": source_text}, ensure_ascii=False, indent=2), encoding="utf-8")
            if text_mask:
                Image.fromarray(text_mask.glyph_refined, "L").save(sample / "7_refined_glyph_mask.png")
                text_mask.overlay().save(sample / "8_final_mask_overlay.png")
            Image.fromarray(inpainted_np).save(sample / "9_inpainted.png")
            rendered.save(sample / "10_rendered.png")
            full_mask = np.zeros(source_np.shape[:2], bool)
            if text_mask:
                x1, y1, x2, y2 = text_mask.crop_bbox; full_mask[y1:y2, x1:x2] = text_mask.refined > 0
            manifest.append({"legacy_block": legacy_id, "page": page_path.name, "fresh_region_ids": [r.id for r in classified],
                             "fresh_block_id": block.id if block else None, "member_ids": list(block.member_ids) if block else [],
                             "ctd_line_ids": line_ids, "ocr_outputs": ocr_outputs, "source_text": source_text,
                             "status": "review" if review_reasons else "auto", "review_reason": review_reasons or None,
                             "backend": record["method"], "outside_mask_pixel_identical": bool(np.array_equal(inpainted_np[~full_mask], source_np[~full_mask]))})
    finally:
        detector.unload(); primary.unload()
        if verifier_loaded: verifier.unload()
        if repair_loaded: repair.unload()
        if translator_loaded: translator.unload()
        inpainter.unload()
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"samples": len(manifest), "reviews": sum(x["status"] == "review" for x in manifest),
                      "outside_identical": all(x["outside_mask_pixel_identical"] for x in manifest),
                      "full_chapter_e2e_run": False, "cache_used": False}, indent=2))


if __name__ == "__main__":
    main()
