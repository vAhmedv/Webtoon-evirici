"""Generate visual preview cards for Text-Block Grouping and Refined Classification."""

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


def _safe_str(s: str) -> str:
    return (s or "").encode("ascii", errors="backslashreplace").decode("ascii")


def main() -> None:
    chapter_dir = Path(
        r"C:\Users\Ahmed\AppData\Local\Tachidesk\downloads\mangas\Asmodeus Scans (EN)\Reincarnated as a God-Tier Crafter\Chapter 1"
    )
    grouping_out_dir = Path("review_output/text_block_grouping_previews")
    class_out_dir = Path("review_output/classification_previews")

    grouping_out_dir.mkdir(parents=True, exist_ok=True)
    class_out_dir.mkdir(parents=True, exist_ok=True)

    pages = load_chapter(chapter_dir, allow_non_uniform_widths=True)
    coords = GlobalCoordinateSystem(tuple(pages))
    cropper = RegionCropper(pages, coords, padding=25)

    regions_json = Path("e2e_output/real_tachidesk_chapter_1/analysis/regions.json")
    if not regions_json.exists():
        print(f"Error: {regions_json} does not exist.")
        return

    with open(regions_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_regions = data.get("regions", [])
    raw_blocks = data.get("text_blocks", [])

    region_map = {r["id"]: r for r in raw_regions}

    # 1. Generate 10 Text-Block Grouping Previews
    multi_line_blocks = [b for b in raw_blocks if len(b.get("member_ids", [])) > 1]
    single_line_blocks = [b for b in raw_blocks if len(b.get("member_ids", [])) == 1]

    random.seed(42)
    selected_grouping_blocks = random.sample(multi_line_blocks, min(7, len(multi_line_blocks))) + random.sample(single_line_blocks, min(3, len(single_line_blocks)))

    for idx, block in enumerate(selected_grouping_blocks, start=1):
        m_ids = block.get("member_ids", [])
        m_objs = [region_map[m_id] for m_id in m_ids if m_id in region_map]
        if not m_objs:
            continue

        mb_boxes = []
        for r_dict in m_objs:
            gb = r_dict["global_bbox"]
            if isinstance(gb, dict):
                bbox = BBox(x1=int(gb["x1"]), y1=int(gb["y1"]), x2=int(gb["x2"]), y2=int(gb["y2"]))
            else:
                bbox = BBox(x1=int(gb[0]), y1=int(gb[1]), x2=int(gb[2]), y2=int(gb[3]))
            mb_boxes.append(bbox)

        # Build combined bounding box
        x1 = min(b.x1 for b in mb_boxes)
        y1 = min(b.y1 for b in mb_boxes)
        x2 = max(b.x2 for b in mb_boxes)
        y2 = max(b.y2 for b in mb_boxes)
        block_bbox = BBox(x1=x1, y1=y1, x2=x2, y2=y2)

        dummy_region = Region(
            id=block["id"],
            global_bbox=block_bbox,
            type=RegionType.DIALOGUE,
            detection_confidence=1.0,
            source_window_ids=(0,),
        )

        crop = cropper.crop_region(dummy_region, adaptive_padding=True)
        crop_img = crop.image

        # Upscale crop for high legibility
        w, h = crop_img.size
        scale = 3 if h < 120 else 2
        scaled_w = w * scale
        scaled_h = h * scale
        crop_scaled = crop_img.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)

        panel_w = max(scaled_w + 40, 720)
        panel_h = 180 + (len(m_objs) * 20)
        total_w = panel_w
        total_h = scaled_h + panel_h + 30

        card = Image.new("RGB", (total_w, total_h), color=(18, 22, 28))
        crop_x = (total_w - scaled_w) // 2
        crop_y = 15
        card.paste(crop_scaled, (crop_x, crop_y))

        draw = ImageDraw.Draw(card)
        try:
            font_header = ImageFont.truetype("arial.ttf", 18)
            font_body = ImageFont.truetype("arial.ttf", 14)
            font_bold = ImageFont.truetype("arialbd.ttf", 14)
        except Exception:
            font_header = font_body = font_bold = ImageFont.load_default()

        draw.rectangle([crop_x - 1, crop_y - 1, crop_x + scaled_w, crop_y + scaled_h], outline=(80, 160, 240), width=2)
        box_y1 = crop_y + scaled_h + 15
        box_y2 = total_h - 10
        draw.rectangle([15, box_y1, total_w - 15, box_y2], fill=(28, 34, 46), outline=(60, 80, 110), width=1)

        y = box_y1 + 10
        draw.text((25, y), f"TextBlock #{block['id']}  |  Members: {len(m_ids)} lines  |  IDs: {m_ids}", fill=(100, 220, 250), font=font_header)
        y += 26

        draw.text((25, y), f"Merged Source Text:", fill=(240, 200, 100), font=font_bold)
        draw.text((180, y), f"\"{_safe_str(block.get('source_text'))}\"", fill=(255, 255, 255), font=font_body)
        y += 22

        draw.text((25, y), f"Hy-MT2 Translation:", fill=(120, 220, 150), font=font_bold)
        draw.text((180, y), f"\"{_safe_str(block.get('translation') or '(not translated)')}\"", fill=(230, 255, 230), font=font_body)
        y += 26

        draw.text((25, y), "Member Region Lines:", fill=(180, 190, 210), font=font_bold)
        y += 20
        for r_obj in m_objs:
            txt = _safe_str(r_obj.get("text", ""))
            draw.text((35, y), f"• Region #{r_obj['id']} [{r_obj.get('type','').upper()}]: \"{txt}\"", fill=(210, 220, 240), font=font_body)
            y += 18

        out_path = grouping_out_dir / f"grouping_block_{block['id']:03d}.png"
        card.save(out_path, format="PNG")

    print(f"[OK] Generated {len(selected_grouping_blocks)} grouping preview cards in {grouping_out_dir.resolve()}")

    # 2. Generate 10 Classification Previews
    category_samples = []
    
    # Categorize raw regions
    dialogue_regs = [r for r in raw_regions if r.get("status") == "auto" and r.get("type") in ("dialogue", "narration", "unknown")]
    sfx_regs = [r for r in raw_regions if r.get("type") == "sfx" or "sfx" in str(r.get("review_reason", ""))]
    watermark_regs = [r for r in raw_regions if r.get("type") == "watermark" or "watermark" in str(r.get("review_reason", ""))]
    noise_regs = [r for r in raw_regions if r.get("status") == "skip" and r not in sfx_regs and r not in watermark_regs]
    review_regs = [r for r in raw_regions if r.get("status") == "review"]

    random.seed(42)
    selected_class_regs = (
        random.sample(dialogue_regs, min(3, len(dialogue_regs))) +
        random.sample(sfx_regs, min(2, len(sfx_regs))) +
        random.sample(watermark_regs, min(2, len(watermark_regs))) +
        random.sample(noise_regs, min(2, len(noise_regs))) +
        random.sample(review_regs, min(1, len(review_regs)))
    )
    if len(selected_class_regs) < 10:
        remaining_pool = [r for r in raw_regions if r not in selected_class_regs]
        selected_class_regs += random.sample(remaining_pool, 10 - len(selected_class_regs))

    for idx, r_dict in enumerate(selected_class_regs, start=1):
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
        crop_img = crop.image

        center_y = (bbox.y1 + bbox.y2) // 2
        page_index, _ = coords.global_to_page(center_y)

        w, h = crop_img.size
        scale = 3 if h < 100 else 2
        scaled_w = w * scale
        scaled_h = h * scale
        crop_scaled = crop_img.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)

        panel_w = max(scaled_w + 40, 700)
        panel_h = 160
        total_w = panel_w
        total_h = scaled_h + panel_h + 30

        card = Image.new("RGB", (total_w, total_h), color=(18, 20, 26))
        crop_x = (total_w - scaled_w) // 2
        crop_y = 15
        card.paste(crop_scaled, (crop_x, crop_y))

        draw = ImageDraw.Draw(card)
        try:
            font_header = ImageFont.truetype("arial.ttf", 18)
            font_body = ImageFont.truetype("arial.ttf", 14)
            font_bold = ImageFont.truetype("arialbd.ttf", 14)
        except Exception:
            font_header = font_body = font_bold = ImageFont.load_default()

        draw.rectangle([crop_x - 1, crop_y - 1, crop_x + scaled_w, crop_y + scaled_h], outline=(70, 80, 100), width=2)
        box_y1 = crop_y + scaled_h + 15
        box_y2 = total_h - 10
        draw.rectangle([15, box_y1, total_w - 15, box_y2], fill=(28, 32, 42), outline=(50, 60, 80), width=1)

        y = box_y1 + 10
        status_color = (100, 220, 120) if region.status == RegionStatus.AUTO else ((240, 180, 80) if region.status == RegionStatus.REVIEW else (180, 180, 180))
        draw.text((25, y), f"Region #{region.id}  |  Page {page_index + 1}  |  Class: {region.type.value.upper()}  |  Status: {region.status.value.upper()}", fill=status_color, font=font_header)
        y += 26

        txt_str = _safe_str(region.text) or "(empty)"
        draw.text((25, y), "OCR Text:", fill=(160, 175, 200), font=font_bold)
        draw.text((160, y), f"\"{txt_str[:55]}\"", fill=(255, 255, 255), font=font_body)
        y += 24

        reason = r_dict.get("review_reason") or "N/A"
        draw.text((25, y), "Classification Reason:", fill=(160, 175, 200), font=font_bold)
        draw.text((160, y), f"{reason}", fill=(220, 200, 250), font=font_body)

        out_path = class_out_dir / f"classification_region_{region.id:03d}_page_{page_index + 1:02d}.png"
        card.save(out_path, format="PNG")

    print(f"[OK] Generated {len(selected_class_regs)} classification preview cards in {class_out_dir.resolve()}")


if __name__ == "__main__":
    main()
