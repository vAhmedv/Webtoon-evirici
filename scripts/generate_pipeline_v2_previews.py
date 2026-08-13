"""Script to generate crop + classification previews for 15 representative problematic regions in Pipeline V2."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.coordinate.global_coords import GlobalCoordinateSystem
from core.detection.bbox import BBox
from core.detection.detection import Region, RegionStatus, RegionType
from core.imaging.region_cropper import RegionCropper
from core.io.input_loader import load_chapter


def main() -> None:
    chapter_dir = Path(
        r"C:\Users\Ahmed\AppData\Local\Tachidesk\downloads\mangas\Asmodeus Scans (EN)\Reincarnated as a God-Tier Crafter\Chapter 1"
    )
    output_dir = Path("review_output/pipeline_v2_previews")
    output_dir.mkdir(parents=True, exist_ok=True)

    pages = load_chapter(chapter_dir, allow_non_uniform_widths=True)
    coords = GlobalCoordinateSystem(tuple(pages))
    cropper = RegionCropper(pages, coords, padding=20)

    regions_json = Path("e2e_output/real_tachidesk_chapter_1/analysis/regions.json")
    if not regions_json.exists():
        print(f"Error: {regions_json} does not exist.")
        return

    with open(regions_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_regions = data.get("regions", [])

    items = []
    for r_dict in raw_regions:
        gb = r_dict["global_bbox"]
        if isinstance(gb, dict):
            bbox = BBox(x1=int(gb["x1"]), y1=int(gb["y1"]), x2=int(gb["x2"]), y2=int(gb["y2"]))
        else:
            bbox = BBox(x1=int(gb[0]), y1=int(gb[1]), x2=int(gb[2]), y2=int(gb[3]))

        region = Region(
            id=r_dict["id"],
            global_bbox=bbox,
            type=RegionType(r_dict.get("type", "unknown")),
            detection_confidence=float(r_dict.get("detection_confidence", 1.0)),
            source_window_ids=tuple(int(x) for x in r_dict.get("source_window_ids", (0,))),
            text=r_dict.get("text", ""),
            ocr_confidence=r_dict.get("ocr_confidence", 0.0),
            status=RegionStatus(r_dict.get("status", "auto")),
            metadata=r_dict.get("metadata", {}),
        )

        crop = cropper.crop_region(region, adaptive_padding=True)
        center_y = (bbox.y1 + bbox.y2) // 2
        page_index, _ = coords.global_to_page(center_y)

        items.append({
            "region": region,
            "crop": crop,
            "page_num": page_index + 1,
            "dict": r_dict,
        })

    # Pick 15 representative regions (mix of STORY_TEXT, SFX, SKIP, and Qwen repaired/review)
    random.seed(42)
    sample_items = random.sample(items, min(15, len(items)))

    for idx, item in enumerate(sample_items, start=1):
        reg = item["region"]
        crop_img = item["crop"].image
        r_dict = item["dict"]
        meta = r_dict.get("metadata", {})
        verdict = meta.get("ocr_verdict", {})

        # Upscale small crop
        w, h = crop_img.size
        scale = 3 if h < 100 else 2
        scaled_w = w * scale
        scaled_h = h * scale
        crop_scaled = crop_img.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)

        panel_w = max(scaled_w + 40, 680)
        panel_h = 170
        total_w = panel_w
        total_h = scaled_h + panel_h + 30

        card = Image.new("RGB", (total_w, total_h), color=(18, 20, 26))
        crop_x = (total_w - scaled_w) // 2
        crop_y = 15
        card.paste(crop_scaled, (crop_x, crop_y))

        draw = ImageDraw.Draw(card)
        try:
            font_header = ImageFont.truetype("arial.ttf", 18)
            font_body = ImageFont.truetype("arial.ttf", 15)
            font_bold = ImageFont.truetype("arialbd.ttf", 15)
        except Exception:
            font_header = font_body = font_bold = ImageFont.load_default()

        draw.rectangle([crop_x - 1, crop_y - 1, crop_x + scaled_w, crop_y + scaled_h], outline=(70, 80, 100), width=2)
        box_y1 = crop_y + scaled_h + 15
        box_y2 = total_h - 10
        draw.rectangle([15, box_y1, total_w - 15, box_y2], fill=(28, 32, 42), outline=(50, 60, 80), width=1)

        y = box_y1 + 10
        status_color = (100, 220, 120) if reg.status == RegionStatus.AUTO else ((240, 180, 80) if reg.status == RegionStatus.REVIEW else (180, 180, 180))
        draw.text((25, y), f"Region #{reg.id}  |  Page {item['page_num']}  |  Type: {reg.type.value.upper()}  |  Status: {reg.status.value.upper()}", fill=status_color, font=font_header)
        y += 26

        text_val = reg.text or "(empty/skipped)"
        draw.text((25, y), "Canonical Text:", fill=(160, 175, 200), font=font_bold)
        draw.text((180, y), f"\"{text_val[:50]}\"", fill=(255, 255, 255), font=font_body)
        y += 24

        second_pass = verdict.get("second_pass_invoked", False)
        draw.text((25, y), f"Second Pass (VL):", fill=(160, 175, 200), font=font_bold)
        draw.text((180, y), f"{'YES' if second_pass else 'NO (PP-OCRv6 Primary Accepted)'}", fill=(200, 220, 255), font=font_body)
        y += 24

        repaired = meta.get("repaired", False)
        reason = verdict.get("reason") or reg.review_reason or "N/A"
        draw.text((25, y), "Qwen Repaired:", fill=(160, 175, 200), font=font_bold)
        draw.text((180, y), f"{'YES' if repaired else 'NO'}  |  Reason: {reason}", fill=(220, 200, 250), font=font_body)

        out_path = output_dir / f"preview_region_{reg.id:03d}_page_{item['page_num']:02d}.png"
        card.save(out_path, format="PNG")

    print(f"[OK] Generated {len(sample_items)} visual classification preview cards in {output_dir.resolve()}")


if __name__ == "__main__":
    main()
