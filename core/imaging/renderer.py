"""Turkish text rendering engine for speech bubbles and text regions.

Provides auto-wrapped, dynamically-scaled, centered text rendering with
high-readability stroke outlines tailored for webtoon comics.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence
from PIL import Image, ImageDraw, ImageFont

from core.detection import Region, RegionStatus, RegionType


FONT_CANDIDATES = [
    r"C:\Windows\Fonts\arialbd.ttf",
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\tahoma.ttf",
]


def _get_font(font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load system font with requested size, falling back to default."""
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
    ) -> tuple[Image.Image, int]:
        """Renders TextBlock translations onto the canvas.

        Args:
            canvas: The full image canvas (PIL Image RGB).
            block_translations: Pairs of (TextBlock, translated_turkish_text).

        Returns:
            Tuple of (new PIL Image canvas with Turkish text rendered, overflow_count).
        """
        result = canvas.copy().convert("RGB")
        draw = ImageDraw.Draw(result)
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
            pad_x = max(3, int(box_w * 0.05))
            pad_y = max(3, int(box_h * 0.05))
            avail_w = max(12, box_w - 2 * pad_x)
            avail_h = max(12, box_h - 2 * pad_y)

            # Dynamic font fitting for block
            member_cnt = len(getattr(block, "members", [1]))
            font, lines, line_height, is_overflow = self._fit_block_text(
                turkish_text, avail_w, avail_h, member_cnt
            )

            if is_overflow:
                overflow_count += 1

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

        return result, overflow_count

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
        """Find optimal font size and word wrapping for a TextBlock."""
        clean_text = text.strip()
        words = clean_text.split()

        # Target font size based on box dimensions and member line count
        est_size = max(13, int(max_h / max(1.8, member_count * 0.95)))
        start_size = min(48, est_size)

        for size in range(start_size, 9, -1):
            font = _get_font(size)
            lines = self._wrap_words(words, font, max_w)

            dummy_bbox = font.getbbox("Ayg") if hasattr(font, "getbbox") else (0, 0, 10, size)
            line_height = int((dummy_bbox[3] - dummy_bbox[1]) * 1.25)
            total_h = len(lines) * line_height

            if total_h <= max_h:
                return font, lines, line_height, False

        # Fallback to 10pt minimum font size
        font = _get_font(10)
        lines = self._wrap_words(words, font, max_w)
        dummy_bbox = font.getbbox("Ayg") if hasattr(font, "getbbox") else (0, 0, 10, 10)
        line_height = int((dummy_bbox[3] - dummy_bbox[1]) * 1.25)
        total_h = len(lines) * line_height
        is_overflow = total_h > max_h

        return font, lines, line_height, is_overflow

    def _fit_text(
        self, text: str, max_w: int, max_h: int
    ) -> tuple[ImageFont.FreeTypeFont | ImageFont.ImageFont, list[str], int]:
        """Find optimal font size and word wrapping that fits max_w and max_h."""
        clean_text = text.strip()
        words = clean_text.split()

        # Start from estimated font size based on box dimensions
        start_size = min(40, max(12, int(max_h / 2.5)))

        for size in range(start_size, 8, -1):
            font = _get_font(size)
            lines = self._wrap_words(words, font, max_w)

            # Estimate line height
            dummy_bbox = font.getbbox("Ayg") if hasattr(font, "getbbox") else (0, 0, 10, size)
            line_height = int((dummy_bbox[3] - dummy_bbox[1]) * 1.25)
            total_h = len(lines) * line_height

            if total_h <= max_h:
                return font, lines, line_height

        # Fallback to minimum size
        font = _get_font(9)
        lines = self._wrap_words(words, font, max_w)
        dummy_bbox = font.getbbox("Ayg") if hasattr(font, "getbbox") else (0, 0, 10, 9)
        line_height = int((dummy_bbox[3] - dummy_bbox[1]) * 1.25)
        return font, lines, line_height

    def _wrap_words(
        self, words: list[str], font: ImageFont.FreeTypeFont | ImageFont.ImageFont, max_w: int
    ) -> list[str]:
        """Wrap words into lines that do not exceed max_w."""
        if not words:
            return [""]

        lines: list[str] = []
        current_line: list[str] = []

        for word in words:
            candidate = " ".join(current_line + [word])
            bbox = font.getbbox(candidate) if hasattr(font, "getbbox") else (0, 0, font.getsize(candidate)[0], 10)
            w = bbox[2] - bbox[0]
            if w <= max_w or not current_line:
                current_line.append(word)
            else:
                lines.append(" ".join(current_line))
                current_line = [word]

        if current_line:
            lines.append(" ".join(current_line))

        return lines
