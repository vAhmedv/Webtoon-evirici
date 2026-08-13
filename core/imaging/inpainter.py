"""Mask-only speech text removal with a uniform fast path and Big-LaMa fallback."""

from __future__ import annotations

from pathlib import Path
from dataclasses import replace
from typing import Any, Sequence

import numpy as np
from PIL import Image

from core.detection import Region, RegionStatus, RegionType
from core.imaging.lama import LaMaLargeInpainter
from core.imaging.text_mask import TextMask, TextMaskBuilder


DEFAULT_LAMA_CHECKPOINT = Path(r"C:\AI\Models\LaMa\lama_large_512px.ckpt")


def _is_story_text(region: Region) -> bool:
    return (
        region.status == RegionStatus.AUTO
        and region.type not in (RegionType.SFX, RegionType.WATERMARK)
        and bool(region.text and region.text.strip())
    )


class Inpainter:
    """Removes source glyphs while preserving every pixel outside the refined mask."""

    def __init__(
        self,
        lama_checkpoint: str | Path = DEFAULT_LAMA_CHECKPOINT,
        debug_dir: str | Path | None = None,
        context_scale: float = 1.7,
    ) -> None:
        self.mask_builder = TextMaskBuilder(context_scale=context_scale)
        self.lama = LaMaLargeInpainter(lama_checkpoint)
        self.debug_dir = Path(debug_dir) if debug_dir is not None else None
        self.debug_records: list[dict[str, Any]] = []
        self.processed_block_ids: set[int] = set()
        self.review_block_ids: set[int] = set()
        self.last_text_mask: TextMask | None = None

    def unload(self) -> None:
        self.lama.unload()

    def inpaint_blocks(self, canvas: Image.Image, text_blocks: Sequence[Any]) -> Image.Image:
        result = np.ascontiguousarray(np.asarray(canvas.convert("RGB"), dtype=np.uint8))
        for block in text_blocks:
            eligible = tuple(region for region in getattr(block, "members", ()) if _is_story_text(region))
            if not eligible:
                continue
            mask = self.mask_builder._build(result, block.merged_bbox, eligible)
            if np.any(mask.refined):
                self.processed_block_ids.add(int(getattr(block, "id", -1)))
            result = self._apply_mask(result, mask, f"block_{getattr(block, 'id', len(self.debug_records) + 1):04d}")
        return Image.fromarray(result, "RGB")

    def inpaint_regions(self, canvas: Image.Image, regions: Sequence[Region]) -> Image.Image:
        result = np.ascontiguousarray(np.asarray(canvas.convert("RGB"), dtype=np.uint8))
        for region in regions:
            if not _is_story_text(region):
                continue
            mask = self.mask_builder.build_for_region(result, region)
            result = self._apply_mask(result, mask, f"region_{region.id:04d}")
        return Image.fromarray(result, "RGB")

    def _apply_mask(self, full_source: np.ndarray, text_mask: TextMask, debug_name: str) -> np.ndarray:
        x1, y1, x2, y2 = text_mask.crop_bbox
        refined = text_mask.refined > 0
        if not np.any(refined):
            self.last_text_mask = text_mask
            self._save_debug(debug_name, text_mask, text_mask.source, "empty")
            return full_source

        if text_mask.is_uniform_background:
            inpainted_crop = text_mask.source.copy()
            inpainted_crop[refined] = np.asarray(text_mask.background_color, dtype=np.uint8)
            method = "median"
        else:
            inpainted_crop = self.lama.inpaint(text_mask.source, text_mask.refined)
            method = "lama_large"

        second_pass = False
        review = False
        expanded = self._residual_expansion(text_mask, inpainted_crop)
        if np.any(expanded > text_mask.refined):
            second_pass = True
            text_mask = replace(text_mask, refined=expanded)
            refined = expanded > 0
            if text_mask.is_uniform_background:
                inpainted_crop = text_mask.source.copy()
                inpainted_crop[refined] = np.asarray(text_mask.background_color, dtype=np.uint8)
            else:
                inpainted_crop = self.lama.inpaint(text_mask.source, expanded)
        review = self._has_boundary_residual(text_mask, inpainted_crop)
        self.last_text_mask = text_mask
        if review and debug_name.startswith("block_"):
            try:
                self.review_block_ids.add(int(debug_name.removeprefix("block_")))
            except ValueError:
                pass

        # The final write is deliberately mask-only. The crop context, borders and
        # artwork remain byte-for-byte identical to the incoming source.
        result = np.array(full_source, copy=True)
        destination = result[y1:y2, x1:x2]
        destination[refined] = inpainted_crop[refined]
        self._save_debug(debug_name, text_mask, inpainted_crop, method, second_pass, review)
        return result

    @staticmethod
    def _residual_candidates(text_mask: TextMask, result: np.ndarray) -> np.ndarray:
        """Find high-contrast source glyph remnants in the one-pixel mask boundary."""
        import cv2

        mask = text_mask.refined
        if text_mask.dilation_radius <= 0 or not np.any(mask):
            return np.zeros_like(mask)
        ring = (cv2.dilate(mask, np.ones((3, 3), np.uint8)) > 0) & (mask == 0)
        allowed = cv2.dilate(text_mask.raw, np.ones((5, 5), np.uint8)) > 0
        gray = cv2.cvtColor(text_mask.source, cv2.COLOR_RGB2GRAY).astype(np.float32)
        result_gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY).astype(np.float32)
        bg = float(np.dot(np.asarray(text_mask.background_color), [0.299, 0.587, 0.114]))
        source_contrast = np.abs(gray - bg)
        result_contrast = np.abs(result_gray - bg)
        candidate = ring & allowed & (source_contrast >= 24) & (result_contrast >= 18)
        if text_mask.bubble_interior is not None:
            candidate &= text_mask.bubble_interior > 0
        if text_mask.protected is not None:
            candidate &= text_mask.protected == 0
        return candidate.astype(np.uint8) * 255

    @classmethod
    def _residual_expansion(cls, text_mask: TextMask, result: np.ndarray) -> np.ndarray:
        import cv2

        candidates = cls._residual_candidates(text_mask, result)
        if np.count_nonzero(candidates) < 2:
            return text_mask.refined
        tiny = cv2.dilate(candidates, np.ones((3, 3), np.uint8))
        return cv2.bitwise_or(text_mask.refined, tiny)

    @classmethod
    def _has_boundary_residual(cls, text_mask: TextMask, result: np.ndarray) -> bool:
        return int(np.count_nonzero(cls._residual_candidates(text_mask, result))) >= 2

    def _save_debug(self, name: str, text_mask: TextMask, inpainted: np.ndarray, method: str,
                    second_pass: bool = False, review: bool = False) -> None:
        record = {
            "name": name,
            "crop_bbox": list(text_mask.crop_bbox),
            "method": method,
            "mask_pixels": int(np.count_nonzero(text_mask.refined)),
            "crop_pixels": int(text_mask.refined.size),
            "bubble_found": text_mask.bubble_found,
            "adaptive_dilation": text_mask.dilation_radius,
            "protected_pixels": text_mask.protected_pixels,
            "second_pass": second_pass,
            "review": review,
        }
        self.debug_records.append(record)
        if self.debug_dir is None:
            return
        target = self.debug_dir / name
        target.mkdir(parents=True, exist_ok=True)
        Image.fromarray(text_mask.source, "RGB").save(target / "source.png")
        Image.fromarray(text_mask.raw, "L").save(target / "raw_text_mask.png")
        if text_mask.predicted_segmentation is not None:
            Image.fromarray(text_mask.predicted_segmentation, "L").save(target / "raw_ctd_segmentation.png")
        if text_mask.glyph_refined is not None:
            Image.fromarray(text_mask.glyph_refined, "L").save(target / "upstream_glyph_mask.png")
        Image.fromarray(text_mask.refined, "L").save(target / "refined_text_mask.png")
        if text_mask.bubble_interior is not None:
            Image.fromarray(text_mask.bubble_interior, "L").save(target / "bubble_interior.png")
        if text_mask.protected is not None:
            Image.fromarray(text_mask.protected, "L").save(target / "protected_structures.png")
        text_mask.overlay().save(target / "mask_overlay.png")
        Image.fromarray(inpainted, "RGB").save(target / "inpainted.png")
