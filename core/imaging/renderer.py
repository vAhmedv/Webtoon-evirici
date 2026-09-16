"""Turkish text rendering engine for speech bubbles and text regions.

Provides auto-wrapped, dynamically-scaled, centered text rendering with
high-readability stroke outlines tailored for webtoon comics.
"""

from __future__ import annotations

from functools import lru_cache
import math
import os
import re
from pathlib import Path
from typing import Any, Sequence
from PIL import Image, ImageDraw, ImageFont, ImageStat
from loguru import logger

from core.detection import BBox, Region, RegionStatus, RegionType


# Faz 2 render guard eşikleri (genel; hiçbir sayfa/bölüme özel değer yok).
# OVERLAP_IOU_THRESHOLD: aynı yere çift basımı yakalar (kopya tespit kaçakları).
# SUBSTRING kuralı: biri diğerinin alt-kümesi olan kaynak metin + kısmi
# çakışma = aynı cümlenin iki tespiti (P009: IoU 0.47 eşiği aşamamıştı).
OVERLAP_IOU_THRESHOLD = 0.4
SUBSTRING_IOU_THRESHOLD = 0.2
# Çift-tırnak benzeri karakterler: tek sayıda kaldıklarında sahipsiz artıktır.
# ASCII kesme işareti (') HARİÇ — Türkçe tamlamalarda meşrudur ("VRMMO'su").
_DOUBLE_QUOTE_CHARS = ('"', '"', '"', ''', ''')


def _has_word_content(text: str) -> bool:
    """Metinde en az bir kelime karakteri var mı (Unicode word class)?"""
    return re.search(r"\w", text, re.UNICODE) is not None


def _clean_orphan_quotes(text: str) -> str:
    """Dengesiz (tek sayılı) çift-tırnak artıklarını temizler.

    OCR/çeviri kırıntıları sahipsiz `"` bırakabilir (`iz"bırakmak`,
    `"NE ...?!` + artı). Her tırnak tipinden tek sayıda varsa sonuncusu
    düşürülür; çiftliler ve ASCII kesme işareti korunur.
    """
    for q in _DOUBLE_QUOTE_CHARS:
        if text.count(q) % 2 == 1:
            idx = text.rfind(q)
            text = text[:idx] + text[idx + 1 :]
            logger.info(f"Renderer: sahipsiz tırnak temizlendi ({q!r}): {text[:60]!r}")
    return text


def _bond_terminal_punct(text: str) -> str:
    """Sondaki sahipsiz noktalamayı kelimeye bağlar, art arda bitiş
    işaretlerini tekler (S6).

    "AMA.. ." → "AMA…", "olmalısın!." → "olmalısın!",
    "..." → "…" (tek-glife iner). "?!" / "!?" bileşimleri korunur.
    """
    text = re.sub(r"\s+([.!?])$", r"\1", text)
    text = text.replace("...", "…")
    text = re.sub(r"!\.+", "!", text)
    text = re.sub(r"\?\.+", "?", text)
    text = re.sub(r"\.{2,}", "…", text)
    text = re.sub(r",{2,}", ",", text)
    return text


def _normalize_render_text(text: str) -> str:
    """Kopya-karşılaştırma için normalize et (küçük harf + tek boşluk)."""
    return re.sub(r"\s+", " ", text.strip().casefold())


def _group_overlapping(
    entries: list[tuple[Any, str, BBox]],
    iou_threshold: float = OVERLAP_IOU_THRESHOLD,
) -> list[list[tuple[Any, str, BBox]]]:
    """Çakışan kutuları aynı gruba alır (açgözlü, geçişli).

    Grup ölçütü: IoU eşiği VEYA (kaynak-altküme + düşük IoU). İkincisi
    kopya-tespit kaçağını yakalar: aynı cümlenin iki kutusu.
    Her gruptan yalnız en uzun metinli girdi render edilir; üst üste binmiş
    balonlar hiçbir dilde aynı anda okunamaz, çift basım her zaman kusurdur.
    """
    sources = [
        _normalize_render_text(getattr(b, "source_text", "") or "")
        for b, _, _ in entries
    ]

    def _linked(i: int, j: int, iou: float) -> bool:
        if iou > iou_threshold:
            return True
        if iou > SUBSTRING_IOU_THRESHOLD and sources[i] and sources[j]:
            if sources[i] in sources[j] or sources[j] in sources[i]:
                return True
        return False

    groups: list[list[tuple[Any, str, BBox]]] = []
    group_members: list[list[int]] = []
    for idx, entry in enumerate(entries):
        _, _, bbox = entry
        placed = False
        for group, members in zip(groups, group_members):
            if any(_linked(idx, m, bbox.iou(entries[m][2])) for m in members):
                group.append(entry)
                members.append(idx)
                placed = True
                break
        if not placed:
            groups.append([entry])
            group_members.append([idx])
    return groups


FONTS_DIR = Path(__file__).resolve().parents[2] / "assets" / "fonts"

FONT_CANDIDATES = [
    # Windows
    r"C:\Windows\Fonts\comicbd.ttf",
    r"C:\Windows\Fonts\comic.ttf",
    r"C:\Windows\Fonts\segoeuib.ttf",
    r"C:\Windows\Fonts\arialbd.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\tahoma.ttf",
    # Linux (çoğu distroda DejaVu/Liberation varsayılan gelir)
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]


@lru_cache(maxsize=128)
def _get_font(font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load comic/system font with requested size with in-memory LRU caching."""
    # 1. First priority: Check local custom comic fonts in assets/fonts/
    if FONTS_DIR.exists():
        for custom_font in sorted(FONTS_DIR.glob("*.ttf")):
            try:
                return ImageFont.truetype(str(custom_font), size=font_size)
            except Exception:
                continue

    # 2. Second priority: Check system comic and display fonts
    for font_path in FONT_CANDIDATES:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, size=font_size)
            except Exception:
                continue

    # 3. PIL'in font arama yolundaki DejaVu denemesi (özellikle Linux).
    for fallback_name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(fallback_name, size=font_size)
        except Exception:
            continue
    logger.warning(
        "No TTF font found (assets/fonts boş ve sistem fontu yok); "
        "bitmap fallback Türkçe glifleri (ğşçıİ) bozabilir. "
        "assets/fonts/ altına bir .ttf ekleyin."
    )
    return ImageFont.load_default()


def _render_target_bbox(block: Any) -> Any:
    """Blok hedef kutusu: kısmi blokta uygun üyelerin birleşimi, yoksa merged."""
    members: tuple[Any, ...] = tuple(getattr(block, "members", ()) or ())
    eligible: tuple[Any, ...] = tuple(
        m for m in members
        if getattr(m, "status", None) == RegionStatus.AUTO
        and getattr(m, "type", None) not in (RegionType.SFX, RegionType.WATERMARK)
    )
    if eligible and len(eligible) < len(members):
        x1 = min(m.global_bbox.x1 for m in eligible)
        y1 = min(m.global_bbox.y1 for m in eligible)
        x2 = max(m.global_bbox.x2 for m in eligible)
        y2 = max(m.global_bbox.y2 for m in eligible)
        return BBox(x1=x1, y1=y1, x2=x2, y2=y2)
    return block.merged_bbox


def _boxes_covered_by(
    residual_boxes: Sequence[Sequence[int]],
    line_rects: Sequence[Sequence[int]],
) -> bool:
    """Artık kutuların TAMAMI satır-dikdörtgenlerinin içinde mi?"""
    if not residual_boxes or not line_rects:
        return False
    for rx1, ry1, rx2, ry2 in residual_boxes:
        if not any(
            rx1 >= lx1 and ry1 >= ly1 and rx2 <= lx2 and ry2 <= ly2
            for lx1, ly1, lx2, ly2 in line_rects
        ):
            return False
    return True



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

        # Faz 2 ön-geçiş: hedef kutuları hesapla, sonra çakışma gruplarında
        # tekilleştir (aynı yere çift basım engeli). Sıralı tek geçişte geri
        # alınamayacağı için karar render'dan önce verilir.
        planned: list[tuple[Any, str, BBox]] = []
        seen_block_ids: set[Any] = set()
        for block, turkish_text in block_translations:
            if not turkish_text or not turkish_text.strip():
                continue
            block_id = getattr(block, "id", None)
            if block_id is not None:
                if block_id in seen_block_ids:
                    logger.warning(
                        f"Renderer: blok {block_id} iki kez listelenmiş; "
                        "ikinci basım atlandı."
                    )
                    continue
                seen_block_ids.add(block_id)
            members: tuple[Any, ...] = tuple(getattr(block, "members", ()) or ())
            # Üye-bazlı filtre: SFX/REVIEW üyeler atlanır, uygun üyeler render edilir.
            # (Eski davranış tüm bloğu atlıyordu; Faz 1a ile tutarlı kısmi render.)
            eligible: tuple[Any, ...] = tuple(
                m for m in members
                if m.status == RegionStatus.AUTO
                and m.type not in (RegionType.SFX, RegionType.WATERMARK)
            )
            if members and not eligible:
                continue

            cleaned = _bond_terminal_punct(_clean_orphan_quotes(turkish_text.strip()))
            if not _has_word_content(cleaned):
                logger.debug(
                    f"Renderer: blok {block_id} kelime içeriği yok "
                    f"({turkish_text[:30]!r}); atlandı."
                )
                continue

            # Hedef kutu tek kaynaktan (_render_target_bbox): kısmi blokta
            # uygun üyelerin birleşimi, yoksa bloğun merged kutusu.
            bbox = _render_target_bbox(block)
            planned.append((block, cleaned, bbox))

        winners: list[tuple[Any, str, BBox]] = []
        for group in _group_overlapping(planned):
            if len(group) == 1:
                winners.append(group[0])
                continue
            # Aynı metin iki kez tespit edilmiş ya da blok bölünmüş:
            # en uzun metinliyi bas, gerisini REVIEW izi bırakarak atla.
            group_sorted = sorted(
                group, key=lambda e: len(e[1].split()), reverse=True
            )
            keep = group_sorted[0]
            dropped = [getattr(b, "id", "?") for b, _, _ in group_sorted[1:]]
            logger.warning(
                f"Renderer: çakışan {len(group)} bloktan en uzun metinli "
                f"blok {getattr(keep[0], 'id', '?')} basıldı; atlananlar: "
                f"{dropped} (metin: {keep[1][:60]!r})"
            )
            winners.append(keep)

        for block, turkish_text, bbox in winners:
            # bbox: ön-geçişte hesaplanmış hedef kutu (kısmi bloklarda uygun
            # üyelerin birleşimi). Burada yeniden hesaplanmaz.
            source_text = getattr(block, "source_text", "") or ""
            if source_text.strip():
                ratio = len(turkish_text) / max(1, len(source_text))
                if ratio > 2.5:
                    logger.warning(
                        f"Block {getattr(block, 'id', '?')}: TR/EN oran {ratio:.2f} "
                        f"(src {len(source_text)} → tr {len(turkish_text)}); taşma riski."
                    )

            x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2
            box_w = max(1, x2 - x1)
            box_h = max(1, y2 - y1)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Generous bubble expansion: CTD text boxes are tightly cropped around English words.
            # Expanding the available area outwards allows Turkish text to fill speech bubbles comfortably.
            exp_x = max(6, int(box_w * 0.18))
            exp_y = max(6, int(box_h * 0.18))
            avail_w = max(28, box_w + 2 * exp_x)
            avail_h = max(28, box_h + 2 * exp_y)

            # Dynamic font fitting for block
            member_cnt = len(getattr(block, "members", [1]))
            font, lines, line_height, is_overflow = self._fit_block_text(
                turkish_text, avail_w, avail_h, member_cnt
            )
            # Kanıt izi (#3 odasınd şüphesi): kesik iddiası piksel-kanıtla
            # kapanır — planlanan kutu, sığan satırlar, taşma bayrağı (debug).
            logger.debug(
                f"Renderer: blok {getattr(block, 'id', '?')} kutu=({box_w}x{box_h}) "
                f"satir={len(lines)} taslak={turkish_text[:40]!r} tasma={is_overflow}"
            )

            if is_overflow:
                overflow_count += 1
                continue

            total_text_h = len(lines) * line_height
            start_y = cy - (total_text_h // 2)

            font_size = getattr(font, "size", 16)
            stroke_w = max(2, min(4, int(font_size * 0.08)))

            # Sample background luminance at center of bubble to guarantee crisp contrast
            cx_clamped = max(0, min(result.width - 1, cx))
            cy_clamped = max(0, min(result.height - 1, cy))
            x_s1 = max(0, cx_clamped - 8)
            y_s1 = max(0, cy_clamped - 8)
            x_s2 = min(result.width, cx_clamped + 8)
            y_s2 = min(result.height, cy_clamped + 8)
            crop = result.crop((x_s1, y_s1, x_s2, y_s2))
            stat = ImageStat.Stat(crop)
            avg_lum = float(sum(stat.mean[:3]) / max(1, len(stat.mean[:3])))

            if avg_lum < 115:  # Dark / Black speech bubble or narration box
                text_color = (255, 255, 255)  # Crisp white text
                stroke_color = (0, 0, 0)      # Black outline
            else:              # Light / White speech bubble
                text_color = (0, 0, 0)        # Crisp black text
                stroke_color = (255, 255, 255)  # White outline

            for i, line in enumerate(lines):
                line_y = start_y + i * line_height
                bbox_line = font.getbbox(line) if hasattr(font, "getbbox") else (0, 0, font.getsize(line)[0], font.getsize(line)[1])
                lw = bbox_line[2] - bbox_line[0]
                line_x = cx - (lw // 2)

                draw.text(
                    (line_x, line_y),
                    line,
                    font=font,
                    fill=text_color,
                    stroke_width=stroke_w,
                    stroke_fill=stroke_color,
                )

            rendered_count += 1

        return result, rendered_count, overflow_count

    def plan_text_rects(
        self,
        block_translations: Sequence[tuple[Any, str]],
    ) -> dict[int, list[list[int]]]:
        """Render planı: blok-id → satır-dikdörtgenleri (global koordinat).

        Çizim YAPMAZ; render_blocks ile aynı kutu/genişletme/sığdırma
        matematiğini kullanır. Taşan/sığmayan blok sözlükte YOKTUR.
        Artık-kurtarma (rescue) kapsama hesabı içindir.
        """
        rects: dict[int, list[list[int]]] = {}
        for block, turkish_text in block_translations:
            block_id = getattr(block, "id", None)
            if block_id is None:
                continue
            cleaned = _bond_terminal_punct(_clean_orphan_quotes((turkish_text or "").strip()))
            if not _has_word_content(cleaned):
                continue
            bbox = _render_target_bbox(block)
            x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2
            box_w = max(1, x2 - x1)
            box_h = max(1, y2 - y1)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            exp_x = max(6, int(box_w * 0.18))
            exp_y = max(6, int(box_h * 0.18))
            avail_w = max(28, box_w + 2 * exp_x)
            avail_h = max(28, box_h + 2 * exp_y)
            member_cnt = len(getattr(block, "members", [1]))
            font, lines, line_height, is_overflow = self._fit_block_text(
                cleaned, avail_w, avail_h, member_cnt
            )
            if is_overflow:
                continue
            total_text_h = len(lines) * line_height
            start_y = cy - (total_text_h // 2)
            block_rects: list[list[int]] = []
            for i, line in enumerate(lines):
                line_y = start_y + i * line_height
                bbox_line = font.getbbox(line) if hasattr(font, "getbbox") else (0, 0, 10, 12)
                lw = bbox_line[2] - bbox_line[0]
                line_x = cx - (lw // 2)
                block_rects.append([line_x, line_y, line_x + lw, line_y + line_height])
            rects[int(block_id)] = block_rects
        return rects

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
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Padding
            pad_x = max(2, int(box_w * 0.08))
            pad_y = max(2, int(box_h * 0.08))
            avail_w = max(10, box_w - 2 * pad_x)
            avail_h = max(10, box_h - 2 * pad_y)

            # Dynamic font fitting
            font, lines, line_height = self._fit_text(turkish_text, avail_w, avail_h)

            total_text_h = len(lines) * line_height
            start_y = y1 + pad_y + max(0, (avail_h - total_text_h) // 2)

            # Sample background luminance for contrast
            cx_clamped = max(0, min(result.width - 1, cx))
            cy_clamped = max(0, min(result.height - 1, cy))
            x_s1 = max(0, cx_clamped - 6)
            y_s1 = max(0, cy_clamped - 6)
            x_s2 = min(result.width, cx_clamped + 6)
            y_s2 = min(result.height, cy_clamped + 6)
            crop = result.crop((x_s1, y_s1, x_s2, y_s2))
            stat = ImageStat.Stat(crop)
            avg_lum = float(sum(stat.mean[:3]) / max(1, len(stat.mean[:3])))

            if avg_lum < 115:
                text_color = (255, 255, 255)
                stroke_color = (0, 0, 0)
            else:
                text_color = (0, 0, 0)
                stroke_color = (255, 255, 255)

            for i, line in enumerate(lines):
                line_y = start_y + i * line_height
                bbox_line = font.getbbox(line) if hasattr(font, "getbbox") else (0, 0, font.getsize(line)[0], font.getsize(line)[1])
                lw = bbox_line[2] - bbox_line[0]
                line_x = x1 + pad_x + max(0, (avail_w - lw) // 2)

                draw.text(
                    (line_x, line_y),
                    line,
                    font=font,
                    fill=text_color,
                    stroke_width=2,
                    stroke_fill=stroke_color,
                )

        return result

    @staticmethod
    def _clean_words(text: str) -> list[str]:
        """Splits text into words ensuring trailing punctuation is bonded to previous word."""
        raw_tokens = text.strip().split()
        if not raw_tokens:
            return []
        cleaned: list[str] = []
        for token in raw_tokens:
            if cleaned and all(c in ".?!,:;…-—~" for c in token):
                cleaned[-1] = cleaned[-1] + token
            else:
                cleaned.append(token)
        return cleaned

    def _break_long_word(
        self,
        word: str,
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
        max_w: int,
    ) -> list[str]:
        """Splits a single long word with hyphenation (-) if it exceeds available width."""
        if self._line_width(word, font) <= max_w:
            return [word]

        core_word = word
        suffix = ""
        while core_word and core_word[-1] in ".?!,:;…-—~":
            suffix = core_word[-1] + suffix
            core_word = core_word[:-1]

        if not core_word:
            return [word]

        chunks: list[str] = []
        current = ""
        for char in core_word:
            cand = current + char
            if current and self._line_width(cand + "-", font) > max_w:
                chunks.append(current + "-")
                current = char
            else:
                current = cand
        if current:
            chunks.append(current + suffix)
        elif suffix and chunks:
            chunks[-1] = chunks[-1] + suffix
        return chunks or [word]

    def _fit_block_text(
        self, text: str, max_w: int, max_h: int, member_count: int = 1
    ) -> tuple[ImageFont.FreeTypeFont | ImageFont.ImageFont, list[str], int, bool]:
        """Find optimal font size and word wrapping for a TextBlock using binary search.
        
        Dynamically scales font size and applies Elliptical Diamond Lettering to fill 70% - 90%
        of speech bubble dimensions comfortably without overflow.
        """
        words = self._clean_words(text)
        if not words:
            font = _get_font(14)
            return font, [""], 16, False

        # Maximum and minimum font size constraints (TR %30 uzun: taban düşük)
        word_count = len(words)
        if word_count <= 4:
            min_size = 12
            max_size = min(48, max(min_size, int(max_h * 0.75), int(max_w * 0.75)))
        elif word_count <= 10:
            min_size = 11
            max_size = min(36, max(min_size, int(max_h * 0.80), int(max_w * 0.80)))
        else:
            min_size = 9
            max_size = min(28, max(min_size, int(max_h * 0.85), int(max_w * 0.85)))

        # Binary search for optimal font size that fits comfortably
        low = min_size
        high = max_size
        best_font = _get_font(min_size)
        best_lines = self._wrap_words(words, best_font, max_w, break_long_words=False)
        dummy_bbox = best_font.getbbox("Aygjpq") if hasattr(best_font, "getbbox") else (0, 0, 10, min_size)
        best_line_h = max(14, int((dummy_bbox[3] - dummy_bbox[1]) * 1.12))
        found = False

        while low <= high:
            mid = (low + high) // 2
            f = _get_font(mid)
            d_box = f.getbbox("Aygjpq") if hasattr(f, "getbbox") else (0, 0, 10, mid)
            lh = max(12, int((d_box[3] - d_box[1]) * 1.12))

            # 1. Try Diamond / Elliptical wrapping first (standard comic speech bubble aesthetic)
            lines = self._wrap_words_elliptical(words, f, max_w, max_h, lh, break_long_words=False)
            if lines is None:
                # 2. Fallback to standard rectangular wrapping
                lines = self._wrap_words(words, f, max_w, break_long_words=False)

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

        # Fallback with word breaking down to minimum size 9 (TR sığması için)
        for size in range(min_size, 8, -1):
            f = _get_font(size)
            d_box = f.getbbox("Aygjpq") if hasattr(f, "getbbox") else (0, 0, 10, size)
            lh = max(9, int((d_box[3] - d_box[1]) * 1.12))
            lines = self._wrap_words_elliptical(words, f, max_w, max_h, lh, break_long_words=True)
            if lines is None:
                lines = self._wrap_words(words, f, max_w, break_long_words=True)
            th = len(lines) * lh
            max_lw = max((self._line_width(line, f) for line in lines), default=0)

            if th <= max_h and max_lw <= max_w:
                return f, lines, lh, False

        # Severe overflow fallback
        f_min = _get_font(9)
        lines_min = self._wrap_words(words, f_min, max_w, break_long_words=True)
        d_box = f_min.getbbox("Aygjpq") if hasattr(f_min, "getbbox") else (0, 0, 10, 9)
        lh_min = max(9, int((d_box[3] - d_box[1]) * 1.12))
        th_min = len(lines_min) * lh_min
        is_overflow = th_min > max_h or any(self._line_width(l, f_min) > max_w for l in lines_min)

        return f_min, lines_min, lh_min, is_overflow

    def _wrap_words_elliptical(
        self,
        words: list[str],
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
        max_w: int,
        max_h: int,
        line_h: int,
        *,
        break_long_words: bool = True,
    ) -> list[str] | None:
        """Wraps words using an elliptical constraint (Diamond Comic Lettering).
        
        Calculates allowed width for each line i based on ellipse geometry:
        w(y) = 2a * sqrt(1 - (y/b)^2)
        Returns the wrapped lines list if they fit inside the ellipse, or None.
        """
        if not words:
            return [""]

        # Pre-break any individual word that is wider than 85% of max_w with hyphenation
        expanded_words: list[str] = []
        for word in words:
            if break_long_words and self._line_width(word, font) > int(max_w * 0.85):
                expanded_words.extend(self._break_long_word(word, font, int(max_w * 0.85)))
            else:
                expanded_words.append(word)

        # Estimate possible line counts N from 1 to max allowed by height
        max_lines_possible = max(1, max_h // line_h)
        total_chars = sum(len(w) for w in expanded_words)
        est_lines = max(1, min(max_lines_possible, int(math.ceil(math.sqrt(total_chars / 3.5)))))
        candidate_line_counts = sorted(
            range(1, max_lines_possible + 1),
            key=lambda n: abs(n - est_lines)
        )

        b = max_h / 2.0
        a = max_w / 2.0

        for n_lines in candidate_line_counts:
            total_block_h = n_lines * line_h
            if total_block_h > max_h:
                continue

            allowed_widths: list[int] = []
            for i in range(n_lines):
                # Center of line i relative to vertical midpoint
                y_i = (i + 0.5) * line_h - (total_block_h / 2.0)
                norm_y = abs(y_i) / max(1.0, b)
                if norm_y >= 0.98:
                    w_i = int(max_w * 0.35)
                else:
                    w_i = int(2.0 * a * math.sqrt(max(0.05, 1.0 - norm_y * norm_y)) * 0.90)
                allowed_widths.append(max(15, w_i))

            # Attempt to pack words into these N lines
            lines: list[str] = []
            word_idx = 0
            num_words = len(expanded_words)
            fits = True

            for line_idx, max_line_w in enumerate(allowed_widths):
                cur_line_words: list[str] = []
                while word_idx < num_words:
                    cand = " ".join(cur_line_words + [expanded_words[word_idx]])
                    if self._line_width(cand, font) <= max_line_w or not cur_line_words:
                        cur_line_words.append(expanded_words[word_idx])
                        word_idx += 1
                    else:
                        break
                if cur_line_words:
                    lines.append(" ".join(cur_line_words))
                elif word_idx < num_words:
                    fits = False
                    break

            if fits and word_idx == num_words:
                return lines

        return None

    def _fit_text(
        self, text: str, max_w: int, max_h: int
    ) -> tuple[ImageFont.FreeTypeFont | ImageFont.ImageFont, list[str], int]:
        """Find optimal font size and word wrapping that fits max_w and max_h using binary search."""
        words = self._clean_words(text)
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
                expanded_words.extend(self._break_long_word(word, font, max_w))
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
