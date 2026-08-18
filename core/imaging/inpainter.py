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

    def inpaint_batch(
        self,
        images: Sequence[np.ndarray],
        masks: Sequence[np.ndarray],
        batch_size: int = 24,
    ) -> list[np.ndarray]:
        """Inpaint a list of image crops with corresponding masks using GPU batching."""
        return self.lama.inpaint_batch(images, masks, batch_size=batch_size)

    def inpaint_blocks(self, canvas: Image.Image, text_blocks: Sequence[Any]) -> Image.Image:
        self.last_text_mask = None
        result = np.array(canvas.convert("RGB"), dtype=np.uint8, copy=True)
        
        # 1. Build text masks for all eligible blocks
        prepared: list[tuple[Any, TextMask, str]] = []
        for block in text_blocks:
            members = tuple(getattr(block, "members", ()))
            if not members or any(not _is_story_text(r) for r in members):
                continue
            eligible = members
            mask = self.mask_builder._build(result, block.merged_bbox, eligible)
            block_id = int(getattr(block, "id", -1))
            if np.any(mask.refined):
                self.processed_block_ids.add(block_id)
            else:
                # No approved text mask means the translated block cannot be
                # safely rendered. Count it as inpaint REVIEW so lifecycle
                # totals remain explicit and the original pixels stay intact.
                self.review_block_ids.add(block_id)
            debug_name = f"block_{getattr(block, 'id', len(self.debug_records) + len(prepared) + 1):04d}"
            prepared.append((block, mask, debug_name))

        # 2. Batch GPU LaMa inference for all blocks requiring full neural inpainting
        lama_jobs: list[tuple[int, np.ndarray, np.ndarray]] = []
        for idx, (block, mask, _) in enumerate(prepared):
            if not np.any(mask.refined):
                continue
            can_flat, _ = self._can_use_flat_fill(mask.source, mask.refined, mask.bubble_interior)
            if not can_flat and not mask.is_uniform_background:
                import cv2
                lama_mask = cv2.dilate(mask.refined, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
                lama_jobs.append((idx, mask.source, lama_mask))

        precomputed_crops: dict[int, np.ndarray] = {}
        if lama_jobs:
            lama_sources = [job[1] for job in lama_jobs]
            lama_masks = [job[2] for job in lama_jobs]
            batch_results = self.lama.inpaint_batch(lama_sources, lama_masks, batch_size=24)
            for (prep_idx, _, _), res_crop in zip(lama_jobs, batch_results):
                precomputed_crops[prep_idx] = res_crop

        # 3. Apply masks in-place directly on the single canvas buffer
        for idx, (block, mask, debug_name) in enumerate(prepared):
            pre_crop = precomputed_crops.get(idx)
            self._apply_mask(result, mask, debug_name, in_place=True, precomputed_crop=pre_crop)

        return Image.fromarray(result, "RGB")

    def inpaint_regions(self, canvas: Image.Image, regions: Sequence[Region]) -> Image.Image:
        result = np.array(canvas.convert("RGB"), dtype=np.uint8, copy=True)
        for region in regions:
            if not _is_story_text(region):
                continue
            mask = self.mask_builder.build_for_region(result, region)
            self._apply_mask(result, mask, f"region_{region.id:04d}", in_place=True)
        return Image.fromarray(result, "RGB")

    def _apply_mask(
        self,
        full_source: np.ndarray,
        text_mask: TextMask,
        debug_name: str,
        in_place: bool = False,
        precomputed_crop: np.ndarray | None = None,
    ) -> np.ndarray:
        x1, y1, x2, y2 = text_mask.crop_bbox
        refined = text_mask.refined > 0
        if not np.any(refined):
            self.last_text_mask = text_mask
            self._save_debug(debug_name, text_mask, text_mask.source, "empty")
            return full_source if in_place else np.array(full_source, copy=True)

        can_flat, flat_color = self._can_use_flat_fill(
            text_mask.source, text_mask.refined, text_mask.bubble_interior
        )

        if can_flat:
            inpainted_crop = self._apply_flat_fill_with_soft_blend(
                text_mask.source, text_mask.refined, flat_color
            )
            method = "flat_fill_fast"
        elif text_mask.is_uniform_background:
            inpainted_crop = text_mask.source.copy()
            inpainted_crop[refined] = np.asarray(text_mask.background_color, dtype=np.uint8)
            method = "median"
        elif precomputed_crop is not None:
            inpainted_crop = precomputed_crop
            method = "lama_large"
        else:
            import cv2
            lama_mask = cv2.dilate(text_mask.refined, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
            inpainted_crop = self.lama.inpaint(text_mask.source, lama_mask)
            method = "lama_large"

        residual_expansion_passes = 0
        review = False
        # Outlined/anti-aliased glyphs can extend several pixels beyond the
        # first color-selected component.  Continue only while the existing
        # boundary detector proves text-like source pixels immediately adjacent
        # to the approved mask.  The raw CTD envelope, bubble interior and
        # protected structures still bound every pass.
        max_expansion_passes = max(2, min(4, text_mask.dilation_radius + 1))
        for _ in range(max_expansion_passes):
            expanded = self._residual_expansion(text_mask, inpainted_crop)
            if not np.any(expanded > text_mask.refined):
                break
            residual_expansion_passes += 1
            text_mask = replace(text_mask, refined=expanded)
            refined = expanded > 0
            if can_flat:
                inpainted_crop = self._apply_flat_fill_with_soft_blend(
                    text_mask.source, expanded, flat_color
                )
            elif text_mask.is_uniform_background:
                inpainted_crop = text_mask.source.copy()
                inpainted_crop[refined] = np.asarray(text_mask.background_color, dtype=np.uint8)
            else:
                import cv2
                lama_expanded = cv2.dilate(expanded, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
                inpainted_crop = self.lama.inpaint(text_mask.source, lama_expanded)
        review = self._has_boundary_residual(text_mask, inpainted_crop)
        self.last_text_mask = text_mask
        if review and debug_name.startswith("block_"):
            try:
                self.review_block_ids.add(int(debug_name.removeprefix("block_")))
            except ValueError:
                pass

        target_canvas = full_source if in_place else np.array(full_source, copy=True)
        destination = target_canvas[y1:y2, x1:x2]
        destination[refined] = inpainted_crop[refined]
        self._save_debug(
            debug_name,
            text_mask,
            inpainted_crop,
            method,
            residual_expansion_passes=residual_expansion_passes,
            review=review,
        )
        return target_canvas



    @staticmethod
    def _can_use_flat_fill(
        image_crop: np.ndarray,
        mask_crop: np.ndarray,
        bubble_interior: np.ndarray | None = None,
        max_std_threshold: float = 14.0,
        overall_std_threshold: float = 12.0,
    ) -> tuple[bool, tuple[int, int, int]]:
        """Maskenin etrafındaki pikselleri analiz ederek düz renkli konuşma balonu kontrolü yapar."""
        import cv2

        if not np.any(mask_crop):
            return False, (255, 255, 255)

        mask = (mask_crop > 0).astype(np.uint8)

        # Maske çevresindeki 2-8px halka piksellerini belirle
        d_outer = cv2.dilate(mask, np.ones((9, 9), np.uint8))
        d_inner = cv2.dilate(mask, np.ones((2, 2), np.uint8))
        ring = (d_outer > 0) & (d_inner == 0)

        if bubble_interior is not None and np.any(bubble_interior):
            ring &= (bubble_interior > 0)

        ring_pixels = image_crop[ring]
        if len(ring_pixels) < 16:
            ring_pixels = image_crop[mask == 0]

        if len(ring_pixels) < 8:
            return False, (255, 255, 255)

        std_rgb = np.std(ring_pixels, axis=0)
        max_std = float(np.max(std_rgb))
        overall_std = float(np.std(ring_pixels))

        if overall_std <= overall_std_threshold or max_std <= max_std_threshold:
            median_color = tuple(int(round(c)) for c in np.median(ring_pixels, axis=0))
            return True, median_color

        return False, (255, 255, 255)

    @staticmethod
    def _apply_flat_fill_with_soft_blend(
        source: np.ndarray,
        mask: np.ndarray,
        fill_color: tuple[int, int, int],
    ) -> np.ndarray:
        """Düz renk dolgusunu yumuşak kenar harmanlama (soft-blend) ile uygular."""
        import cv2

        refined = mask > 0
        if not np.any(refined):
            return source.copy()

        # Maske kenarlarına 1-2px Gaussian Blur ile yumuşak geçiş
        mask_float = refined.astype(np.float32)
        blurred_mask = cv2.GaussianBlur(mask_float, (3, 3), 0.8)
        alpha = np.expand_dims(blurred_mask, axis=-1)

        fill_arr = np.full_like(source, fill_color, dtype=np.float32)
        src_float = source.astype(np.float32)

        blended = np.clip(fill_arr * alpha + src_float * (1.0 - alpha), 0, 255).astype(np.uint8)
        return blended

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
        import cv2

        candidates = cls._residual_candidates(text_mask, result)
        total_residual = int(np.count_nonzero(candidates))
        if total_residual < 2:
            return False
        outside_raw = candidates & (text_mask.raw == 0)
        outside_count = int(np.count_nonzero(outside_raw))
        if outside_count < 2:
            return False
        count, labels, stats, _ = cv2.connectedComponentsWithStats(outside_raw, 8)
        if count <= 1:
            return False
        areas = stats[1:, cv2.CC_STAT_AREA]
        large_components = int(np.sum(areas >= 10))
        total_components = count - 1
        return large_components >= 1 and total_components <= 10

    def _save_debug(
        self,
        name: str,
        text_mask: TextMask,
        inpainted: np.ndarray,
        method: str,
        second_pass: bool = False,
        review: bool = False,
        residual_expansion_passes: int | None = None,
    ) -> None:
        import cv2

        expansion_passes = (
            int(residual_expansion_passes)
            if residual_expansion_passes is not None
            else int(bool(second_pass))
        )
        candidates = self._residual_candidates(text_mask, inpainted)
        total_residual = int(np.count_nonzero(candidates))
        if total_residual < 2:
            remaining_residual_pixels = 0
        else:
            outside_raw = candidates & (text_mask.raw == 0)
            outside_count = int(np.count_nonzero(outside_raw))
            if outside_count < 2:
                remaining_residual_pixels = 0
            else:
                count, _, stats, _ = cv2.connectedComponentsWithStats(outside_raw, 8)
                if count <= 1:
                    remaining_residual_pixels = 0
                else:
                    areas = stats[1:, cv2.CC_STAT_AREA]
                    large_components = int(np.sum(areas >= 10))
                    total_components = count - 1
                    if large_components >= 1 and total_components <= 10:
                        remaining_residual_pixels = outside_count
                    else:
                        remaining_residual_pixels = 0
        record = {
            "name": name,
            "crop_bbox": list(text_mask.crop_bbox),
            "method": method,
            "mask_pixels": int(np.count_nonzero(text_mask.refined)),
            "crop_pixels": int(text_mask.refined.size),
            "bubble_found": text_mask.bubble_found,
            "adaptive_dilation": text_mask.dilation_radius,
            "protected_pixels": text_mask.protected_pixels,
            "second_pass": expansion_passes > 0,
            "residual_expansion_passes": expansion_passes,
            "remaining_boundary_residual_pixels": remaining_residual_pixels,
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
