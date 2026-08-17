"""Turkish text rendering engine for speech bubbles and text regions.

Provides auto-wrapped, dynamically-scaled, centered text rendering with
high-readability stroke outlines tailored for webtoon comics.
"""

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
from typing import Any, Sequence
from PIL import Image, ImageDraw, ImageFont

from core.detection import Region, RegionStatus, RegionType


FONT_CANDIDATES = [
    r"C:\Windows\Fonts\comicbd.ttf",
    r"C:\Windows\Fonts\comic.ttf",
    r"C:\Windows\Fonts\segoeuib.ttf",
    r"C:\Windows\Fonts\arialbd.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\tahoma.ttf",
]


@lru_cache(maxsize=128)
def _get_font(font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load system font with requested size with in-memory LRU caching."""
    for font_path in FONT_CANDIDATES:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, size=font_size)
            except Exception:
                continue
    return ImageFont.load_default()



class TextRenderer:
    """Renders translated Turkish text into speech bubbles on a canvas."""

    def render_blocks(
        self,
        canvas: Image.Image,
        block_translations: Sequence[tuple[Any, str]],
    ) -> tuple[Image.Image, int, int]:
        """Renders TextBlock translations onto the canvas.

        Args:
            canvas: The full image canvas (PIL Image RGB).
            block_translations: Pairs of (TextBlock, translated_turkish_text).

        Returns:
            Tuple of (new PIL Image canvas with Turkish text rendered, rendered_count, overflow_count).
        """
        result = canvas.copy().convert("RGB")
        draw = ImageDraw.Draw(result)
        rendered_count = 0
        overflow_count = 0

        for block, turkish_text in block_translations:
            if not turkish_text or not turkish_text.strip():
                continue
            members = tuple(getattr(block, "members", ()))
            if not members or any(
                member.status != RegionStatus.AUTO
                or member.type in (RegionType.SFX, RegionType.WATERMARK)
                for member in members
            ):
                continue

            bbox = block.merged_bbox
            x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2
            box_w = max(1, x2 - x1)
            box_h = max(1, y2 - y1)

            # Responsive padding
            pad_x = max(2, int(box_w * 0.04))
            pad_y = max(2, int(box_h * 0.04))
            avail_w = max(12, box_w - 2 * pad_x)
            avail_h = max(12, box_h - 2 * pad_y)

            # Dynamic font fitting for block
            member_cnt = len(getattr(block, "members", [1]))
            font, lines, line_height, is_overflow = self._fit_block_text(
                turkish_text, avail_w, avail_h, member_cnt
            )

            if is_overflow:
                overflow_count += 1
                continue

            total_text_h = len(lines) * line_height
            start_y = y1 + pad_y + max(0, (avail_h - total_text_h) // 2)

            for i, line in enumerate(lines):
                line_y = start_y + i * line_height
                bbox_line = font.getbbox(line) if hasattr(font, "getbbox") else (0, 0, font.getsize(line)[0], font.getsize(line)[1])
                lw = bbox_line[2] - bbox_line[0]
                line_x = x1 + pad_x + max(0, (avail_w - lw) // 2)

                draw.text(
                    (line_x, line_y),
                    line,
                    font=font,
                    fill=(0, 0, 0),
                    stroke_width=2,
                    stroke_fill=(255, 255, 255),
                )

            rendered_count += 1

        return result, rendered_count, overflow_count

    def render_regions(
        self,
        canvas: Image.Image,
        region_translations: Sequence[tuple[Region, str]],
    ) -> Image.Image:
        """Renders translations onto the canvas.

        Args:
            canvas: The full image canvas (PIL Image RGB).
            region_translations: Pairs of (Region, translated_turkish_text).

        Returns:
            A new PIL Image canvas with Turkish text rendered.
        """
        result = canvas.copy().convert("RGB")
        draw = ImageDraw.Draw(result)

        for region, turkish_text in region_translations:
            if not turkish_text or not turkish_text.strip():
                continue

            if region.status != RegionStatus.AUTO or region.type in (RegionType.SFX, RegionType.WATERMARK):
                continue

            bbox = region.global_bbox
            x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2
            box_w = max(1, x2 - x1)
            box_h = max(1, y2 - y1)

            # Padding
            pad_x = max(2, int(box_w * 0.08))
            pad_y = max(2, int(box_h * 0.08))
            avail_w = max(10, box_w - 2 * pad_x)
            avail_h = max(10, box_h - 2 * pad_y)

            # Dynamic font fitting
            font, lines, line_height = self._fit_text(turkish_text, avail_w, avail_h)

            total_text_h = len(lines) * line_height
            start_y = y1 + pad_y + max(0, (avail_h - total_text_h) // 2)

            for i, line in enumerate(lines):
                line_y = start_y + i * line_height
                # Get text width
                bbox_line = font.getbbox(line) if hasattr(font, "getbbox") else (0, 0, font.getsize(line)[0], font.getsize(line)[1])
                lw = bbox_line[2] - bbox_line[0]
                line_x = x1 + pad_x + max(0, (avail_w - lw) // 2)

                draw.text(
                    (line_x, line_y),
                    line,
                    font=font,
                    fill=(0, 0, 0),
                    stroke_width=2,
                    stroke_fill=(255, 255, 255),
                )

        return result

    def _fit_block_text(
        self, text: str, max_w: int, max_h: int, member_count: int = 1
    ) -> tuple[ImageFont.FreeTypeFont | ImageFont.ImageFont, list[str], int, bool]:
        """Find optimal font size and word wrapping for a TextBlock using binary search.
        
        Dynamically scales font size to fill 65% - 85% of bubble dimensions without overflow.
        """
        clean_text = text.strip()
        words = clean_text.split()
        if not words:
            font = _get_font(12)
            return font, [""], 15, False

        # Maximum and minimum font size constraints
        min_size = 11
        max_size = min(72, max(min_size, int(max_h * 0.85), int(max_w * 0.90)))

        # Binary search for optimal font size that fits comfortably
        low = min_size
        high = max_size
        best_font = _get_font(min_size)
        best_lines = self._wrap_words(words, best_font, max_w, break_long_words=False)
        dummy_bbox = best_font.getbbox("Aygjpq") if hasattr(best_font, "getbbox") else (0, 0, 10, min_size)
        best_line_h = max(12, int((dummy_bbox[3] - dummy_bbox[1]) * 1.20))
        found = False

        while low <= high:
            mid = (low + high) // 2
            f = _get_font(mid)
            lines = self._wrap_words(words, f, max_w, break_long_words=False)
            d_box = f.getbbox("Aygjpq") if hasattr(f, "getbbox") else (0, 0, 10, mid)
            lh = max(12, int((d_box[3] - d_box[1]) * 1.20))
            th = len(lines) * lh
            max_lw = max((self._line_width(line, f) for line in lines), default=0)

            if th <= max_h and max_lw <= max_w:
                best_font = f
                best_lines = lines
                best_line_h = lh
                found = True
                low = mid + 1  # Try larger
            else:
                high = mid - 1  # Try smaller

        if found:
            return best_font, best_lines, best_line_h, False

        # Fallback with word breaking down to minimum size 10
        for size in range(min_size, 9, -1):
            f = _get_font(size)
            lines = self._wrap_words(words, f, max_w, break_long_words=True)
            d_box = f.getbbox("Aygjpq") if hasattr(f, "getbbox") else (0, 0, 10, size)
            lh = max(10, int((d_box[3] - d_box[1]) * 1.20))
            th = len(lines) * lh
            max_lw = max((self._line_width(line, f) for line in lines), default=0)

            if th <= max_h and max_lw <= max_w:
                return f, lines, lh, False

        # Severe overflow fallback
        f_min = _get_font(10)
        lines_min = self._wrap_words(words, f_min, max_w, break_long_words=True)
        d_box = f_min.getbbox("Aygjpq") if hasattr(f_min, "getbbox") else (0, 0, 10, 10)
        lh_min = max(10, int((d_box[3] - d_box[1]) * 1.20))
        th_min = len(lines_min) * lh_min
        is_overflow = th_min > max_h or any(self._line_width(l, f_min) > max_w for l in lines_min)

        return f_min, lines_min, lh_min, is_overflow

    def _fit_text(
        self, text: str, max_w: int, max_h: int
    ) -> tuple[ImageFont.FreeTypeFont | ImageFont.ImageFont, list[str], int]:
        """Find optimal font size and word wrapping that fits max_w and max_h using binary search."""
        clean_text = text.strip()
        words = clean_text.split()
        if not words:
            font = _get_font(12)
            return font, [""], 15

        min_size = 11
        max_size = min(64, max(min_size, int(max_h * 0.85), int(max_w * 0.90)))

        low = min_size
        high = max_size
        best_font = _get_font(min_size)
        best_lines = self._wrap_words(words, best_font, max_w, break_long_words=False)
        dummy_bbox = best_font.getbbox("Aygjpq") if hasattr(best_font, "getbbox") else (0, 0, 10, min_size)
        best_line_h = max(12, int((dummy_bbox[3] - dummy_bbox[1]) * 1.20))

        while low <= high:
            mid = (low + high) // 2
            f = _get_font(mid)
            lines = self._wrap_words(words, f, max_w, break_long_words=False)
            d_box = f.getbbox("Aygjpq") if hasattr(f, "getbbox") else (0, 0, 10, mid)
            lh = max(12, int((d_box[3] - d_box[1]) * 1.20))
            th = len(lines) * lh
            max_lw = max((self._line_width(line, f) for line in lines), default=0)

            if th <= max_h and max_lw <= max_w:
                best_font = f
                best_lines = lines
                best_line_h = lh
                low = mid + 1
            else:
                high = mid - 1

        return best_font, best_lines, best_line_h

    def _wrap_words(
        self,
        words: list[str],
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
        max_w: int,
        *,
        break_long_words: bool = True,
    ) -> list[str]:
        """Wrap words into lines that do not exceed max_w."""
        if not words:
            return [""]

        lines: list[str] = []
        current_line: list[str] = []

        expanded_words: list[str] = []
        for word in words:
            if break_long_words and self._line_width(word, font) > max_w:
                chunk = ""
                for char in word:
                    candidate = chunk + char
                    if chunk and self._line_width(candidate, font) > max_w:
                        expanded_words.append(chunk)
                        chunk = char
                    else:
                        chunk = candidate
                if chunk:
                    expanded_words.append(chunk)
            else:
                expanded_words.append(word)

        for word in expanded_words:
            candidate = " ".join(current_line + [word])
            w = self._line_width(candidate, font)
            if w <= max_w or not current_line:
                current_line.append(word)
            else:
                lines.append(" ".join(current_line))
                current_line = [word]

        if current_line:
            lines.append(" ".join(current_line))

        return lines

    @staticmethod
    def _line_width(
        text: str, font: ImageFont.FreeTypeFont | ImageFont.ImageFont
    ) -> int:
        bbox = font.getbbox(text) if hasattr(font, "getbbox") else (0, 0, font.getsize(text)[0], 10)
        return int(bbox[2] - bbox[0])
