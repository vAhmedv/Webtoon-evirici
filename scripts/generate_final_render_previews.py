"""Generate final render visual preview cards comparing Source, Inpainted, and Turkish Rendered crops."""

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
from core.imaging.inpainter import Inpainter
from core.imaging.region_cropper import RegionCropper
from core.imaging.renderer import TextRenderer
from core.io.input_loader import load_chapter


def _safe_str(s: str) -> str:
    return (s or "").encode("ascii", errors="backslashreplace").decode("ascii")


def main() -> None:
    chapter_dir = Path(
        r"C:\Users\Ahmed\AppData\Local\Tachidesk\downloads\mangas\Asmodeus Scans (EN)\Reincarnated as a God-Tier Crafter\Chapter 1"
    )
    out_dir = Path("review_output/final_render_previews")
    out_dir.mkdir(parents=True, exist_ok=True)

    pages = load_chapter(chapter_dir, allow_non_uniform_widths=True)
    coords = GlobalCoordinateSystem(tuple(pages))

    # Build full global canvas
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

    regions_json = Path("e2e_output/real_tachidesk_chapter_1/analysis/regions.json")
    if not regions_json.exists():
        print(f"Error: {regions_json} does not exist.")
        return

    with open(regions_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_regions = data.get("regions", [])
    raw_blocks = data.get("text_blocks", [])
    region_map = {r["id"]: r for r in raw_regions}

    # Reconstruct TextBlock objects
    from core.detection.text_block import TextBlock

    blocks: list[tuple[TextBlock, str]] = []
    for b_dict in raw_blocks:
        b_id = b_dict["id"]
        m_ids = tuple(b_dict.get("member_ids", []))
        tr_text = b_dict.get("translation") or ""
        mb_bbox_list = b_dict["merged_bbox"]
        merged_bbox = BBox(x1=mb_bbox_list[0], y1=mb_bbox_list[1], x2=mb_bbox_list[2], y2=mb_bbox_list[3])
        
        member_regions = []
        for m_id in m_ids:
            if m_id in region_map:
                r_dict = region_map[m_id]
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
                    status=RegionStatus(r_dict.get("status", "auto")),
                    metadata=r_dict.get("metadata", {}),
                )
                member_regions.append(region)

        tb = TextBlock(
            id=b_id,
            member_ids=m_ids,
            members=tuple(member_regions),
            merged_bbox=merged_bbox,
            source_text=b_dict.get("source_text", ""),
            translation=tr_text,
        )
        if tr_text.strip():
            blocks.append((tb, tr_text))

    # Inpaint canvas & Render canvas
    inpainter = Inpainter()
    cleaned_canvas = inpainter.inpaint_blocks(global_canvas, [tb for tb, _ in blocks])

    renderer = TextRenderer()
    rendered_canvas, _ = renderer.render_blocks(cleaned_canvas, blocks)

    # Sample representative blocks across categories
    # 1. Multi-line "I WAS ON THE VERGE OF..." block
    level_cap_blocks = [b for b in blocks if "LEVEL CAP" in b[0].source_text.upper() or "VERGE OF" in b[0].source_text.upper()]
    # 2. 1-line short blocks
    short_blocks = [b for b in blocks if len(b[0].members) == 1 and len(b[0].source_text) < 25]
    # 3. 3-6 line normal blocks
    normal_blocks = [b for b in blocks if 3 <= len(b[0].members) <= 6]
    # 4. Long translation blocks
    long_tr_blocks = [b for b in blocks if len(b[1]) > 60]
    # 5. Narrow / Wide blocks
    narrow_blocks = [b for b in blocks if b[0].merged_bbox.width < 140]
    wide_blocks = [b for b in blocks if b[0].merged_bbox.width > 300]

    random.seed(42)
    selected_sample = []
    if level_cap_blocks:
        selected_sample.append(level_cap_blocks[0])

    selected_sample += random.sample(short_blocks, min(3, len(short_blocks)))
    selected_sample += random.sample(normal_blocks, min(4, len(normal_blocks)))
    selected_sample += random.sample(long_tr_blocks, min(3, len(long_tr_blocks)))
    selected_sample += random.sample(narrow_blocks, min(2, len(narrow_blocks)))
    selected_sample += random.sample(wide_blocks, min(2, len(wide_blocks)))

    # Deduplicate while preserving order
    seen_ids = set()
    unique_selected = []
    for pair in selected_sample:
        if pair[0].id not in seen_ids:
            seen_ids.add(pair[0].id)
            unique_selected.append(pair)

    sample_final = unique_selected[:14]

    padding = 30
    for idx, (block, tr_text) in enumerate(sample_final, start=1):
        mb = block.merged_bbox
        x1 = max(0, mb.x1 - padding)
        y1 = max(0, mb.y1 - padding)
        x2 = min(global_canvas.width, mb.x2 + padding)
        y2 = min(global_canvas.height, mb.y2 + padding)

        crop_source = global_canvas.crop((x1, y1, x2, y2))
        crop_inpaint = cleaned_canvas.crop((x1, y1, x2, y2))
        crop_render = rendered_canvas.crop((x1, y1, x2, y2))

        # Scale crops for clarity
        h = y2 - y1
        scale = 3 if h < 100 else 2
        sw = (x2 - x1) * scale
        sh = h * scale

        src_s = crop_source.resize((sw, sh), Image.Resampling.LANCZOS)
        inp_s = crop_inpaint.resize((sw, sh), Image.Resampling.LANCZOS)
        ren_s = crop_render.resize((sw, sh), Image.Resampling.LANCZOS)

        # Composite side-by-side card
        card_w = max(sw * 3 + 60, 780)
        panel_info_h = 160
        card_h = sh + panel_info_h + 40

        card = Image.new("RGB", (card_w, card_h), (18, 20, 26))

        # Paste 3 crops
        gap = (card_w - (sw * 3)) // 4
        card.paste(src_s, (gap, 35))
        card.paste(inp_s, (gap * 2 + sw, 35))
        card.paste(ren_s, (gap * 3 + sw * 2, 35))

        draw = ImageDraw.Draw(card)
        try:
            font_header = ImageFont.truetype("arial.ttf", 16)
            font_body = ImageFont.truetype("arial.ttf", 14)
            font_bold = ImageFont.truetype("arialbd.ttf", 14)
        except Exception:
            font_header = font_body = font_bold = ImageFont.load_default()

        # Crop labels
        draw.text((gap, 10), "1. Source Crop (Original)", fill=(240, 180, 100), font=font_bold)
        draw.text((gap * 2 + sw, 10), "2. Inpainted (Cleaned)", fill=(100, 220, 250), font=font_bold)
        draw.text((gap * 3 + sw * 2, 10), "3. Turkish Rendered", fill=(120, 240, 140), font=font_bold)

        # Draw crop outlines
        for px in [gap, gap * 2 + sw, gap * 3 + sw * 2]:
            draw.rectangle([px - 1, 34, px + sw, 35 + sh], outline=(60, 80, 110), width=1)

        # Info Box at bottom
        box_y1 = 45 + sh
        box_y2 = card_h - 10
        draw.rectangle([15, box_y1, card_w - 15, box_y2], fill=(28, 32, 44), outline=(50, 65, 90), width=1)

        y = box_y1 + 10
        draw.text((25, y), f"TextBlock #{block.id}  |  Members: {len(block.member_ids)} lines  |  IDs: {list(block.member_ids)}", fill=(100, 220, 250), font=font_header)
        y += 24

        draw.text((25, y), "Source English:", fill=(240, 200, 100), font=font_bold)
        draw.text((170, y), f"\"{_safe_str(block.source_text)[:65]}\"", fill=(255, 255, 255), font=font_body)
        y += 22

        draw.text((25, y), "Turkish Translation:", fill=(120, 240, 140), font=font_bold)
        draw.text((170, y), f"\"{_safe_str(tr_text)[:65]}\"", fill=(230, 255, 230), font=font_body)

        out_path = out_dir / f"final_render_block_{block.id:03d}.png"
        card.save(out_path, format="PNG")

    print(f"[OK] Generated {len(sample_final)} final render preview cards in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
