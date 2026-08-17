"""CTD geometry guided, component-matched text-mask refinement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from PIL import Image

from core.detection import BBox, Region


def self_or_inverse(candidate: np.ndarray, predicted: np.ndarray) -> tuple[np.ndarray, int]:
    """Choose the polarity with the smaller XOR against CTD's predicted mask."""
    import cv2

    direct = int(cv2.bitwise_xor(candidate, predicted).sum())
    inverse_mask = 255 - candidate
    inverse = int(cv2.bitwise_xor(inverse_mask, predicted).sum())
    return (inverse_mask, inverse) if inverse < direct else (candidate, direct)


def merge_xor_components(candidates: list[tuple[np.ndarray, int]], predicted: np.ndarray) -> np.ndarray:
    """Greedily admit connected components only when they improve predicted-mask XOR."""
    import cv2

    _, target = cv2.threshold(cv2.erode(predicted, np.ones((3, 3), np.uint8)), 60, 255, cv2.THRESH_BINARY)
    if not np.any(target):
        _, target = cv2.threshold(predicted, 60, 255, cv2.THRESH_BINARY)
    merged = np.zeros_like(target)
    for candidate, _ in sorted(candidates, key=lambda item: item[1]):
        count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
        for label in range(1, count):
            x, y, w, h, _ = (int(value) for value in stats[label])
            if w * h < 3:
                continue
            component = (labels[y:y + h, x:x + w] == label).astype(np.uint8) * 255
            current = merged[y:y + h, x:x + w]
            proposed = cv2.bitwise_or(current, component)
            target_crop = target[y:y + h, x:x + w]
            if cv2.bitwise_xor(proposed, target_crop).sum() < cv2.bitwise_xor(current, target_crop).sum():
                current[:] = proposed
    # Upstream hole handling: add enclosed background CCs only if XOR improves.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(255 - merged, 8)
    areas = np.sort(stats[:, cv2.CC_STAT_AREA])
    hole_limit = int(areas[-2] if len(areas) > 1 else areas[-1])
    for label in range(1, count):
        x, y, w, h, area = (int(value) for value in stats[label])
        if area >= hole_limit:
            continue
        component = (labels[y:y + h, x:x + w] == label).astype(np.uint8) * 255
        current = merged[y:y + h, x:x + w]
        proposed = cv2.bitwise_or(current, component)
        target_crop = target[y:y + h, x:x + w]
        if cv2.bitwise_xor(proposed, target_crop).sum() < cv2.bitwise_xor(current, target_crop).sum():
            current[:] = proposed
    return merged


@dataclass(frozen=True)
class TextMask:
    crop_bbox: tuple[int, int, int, int]
    source: np.ndarray
    raw: np.ndarray
    refined: np.ndarray
    background_color: tuple[int, int, int]
    is_uniform_background: bool
    bubble_interior: np.ndarray | None = None
    protected: np.ndarray | None = None
    dilation_radius: int = 0
    predicted_segmentation: np.ndarray | None = None
    glyph_refined: np.ndarray | None = None

    @property
    def bubble_found(self) -> bool:
        return self.bubble_interior is not None and bool(np.any(self.bubble_interior))

    @property
    def protected_pixels(self) -> int:
        return int(np.count_nonzero(self.protected)) if self.protected is not None else 0

    def overlay(self) -> Image.Image:
        overlay = self.source.astype(np.float32).copy()
        selected = self.refined > 0
        overlay[selected] = overlay[selected] * 0.35 + np.array([255, 30, 30], np.float32) * 0.65
        return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), "RGB")


class TextMaskBuilder:
    """Match CTD mask components to DBNet lines and refine only those components."""

    def __init__(self, context_scale: float = 1.7) -> None:
        self.context_scale = max(1.2, float(context_scale))

    def build_for_region(self, image: Image.Image | np.ndarray, region: Region) -> TextMask:
        return self._build(image, region.global_bbox, [region])

    def build_for_block(self, image: Image.Image | np.ndarray, block: Any) -> TextMask:
        members = list(getattr(block, "members", ()))
        bbox = getattr(block, "merged_bbox", None)
        if bbox is None or not members:
            raise ValueError("TextBlock must contain members and merged_bbox")
        return self._build(image, bbox, members)

    def _build(self, image: Image.Image | np.ndarray, bbox: BBox, regions: Sequence[Region]) -> TextMask:
        import cv2

        full = np.asarray(image.convert("RGB") if isinstance(image, Image.Image) else image, dtype=np.uint8)
        height, width = full.shape[:2]
        cx, cy = bbox.center
        crop_w = max(bbox.width + 16, int(round(bbox.width * self.context_scale)))
        crop_h = max(bbox.height + 16, int(round(bbox.height * self.context_scale)))
        x1, y1 = max(0, int(round(cx - crop_w / 2))), max(0, int(round(cy - crop_h / 2)))
        x2, y2 = min(width, int(round(cx + crop_w / 2))), min(height, int(round(cy + crop_h / 2)))
        if x2 <= x1 or y2 <= y1:
            dummy_src = np.zeros((1, 1, 3), dtype=np.uint8)
            dummy_mask = np.zeros((1, 1), dtype=np.uint8)
            return TextMask(BBox(x1, y1, x2, y2), dummy_src, dummy_mask, dummy_mask, (255, 255, 255), True, 0.0)

        source = np.ascontiguousarray(full[y1:y2, x1:x2])
        raw = np.zeros(source.shape[:2], dtype=np.uint8, order='C')
        segmentation = np.zeros(source.shape[:2], dtype=np.uint8, order='C')
        if raw.shape[0] == 0 or raw.shape[1] == 0:
            return TextMask(BBox(x1, y1, x2, y2), source, raw, raw, (255, 255, 255), True, 0.0)

        for region in regions:
            lines = self._valid_polygons(region.metadata.get("line_polygons"))
            segments = self._valid_polygons(region.metadata.get("segmentation_polygons"))
            for polygon in lines or segments:
                if not polygon or len(polygon) < 3:
                    continue
                pts = np.array([[round(px - x1), round(py - y1)] for px, py in polygon], dtype=np.int32).reshape((-1, 1, 2))
                if pts.shape[0] < 3:
                    continue
                pts = np.ascontiguousarray(pts)
                cv2.fillPoly(raw, [pts], 255)
            for polygon in segments:
                if not polygon or len(polygon) < 3:
                    continue
                pts = np.array([[round(px - x1), round(py - y1)] for px, py in polygon], dtype=np.int32).reshape((-1, 1, 2))
                if pts.shape[0] < 3:
                    continue
                pts = np.ascontiguousarray(pts)
                cv2.fillPoly(segmentation, [pts], 255)

        bubble = self._extract_bubble(source, raw)
        protected = self._protected_structures(source, raw, bubble)
        glyph, radius = self._upstream_refine(source, raw, segmentation, bubble)
        refined = glyph.copy()
        refined[protected > 0] = 0
        if bubble is not None:
            refined[bubble == 0] = 0
        background, uniform = self._background_statistics(source, raw, refined, bubble)
        return TextMask((x1, y1, x2, y2), source, raw, refined, background, uniform,
                        bubble, protected, radius, segmentation, glyph)

    @staticmethod
    def _valid_polygons(value: object) -> list[list[list[float]]]:
        if not isinstance(value, list):
            return []
        result = []
        for polygon in value:
            if not isinstance(polygon, list) or len(polygon) < 3:
                continue
            try:
                result.append([[float(point[0]), float(point[1])] for point in polygon])
            except (TypeError, ValueError, IndexError):
                pass
        return result

    @staticmethod
    def _extract_bubble(source: np.ndarray, raw: np.ndarray) -> np.ndarray | None:
        """Return the smallest closed Canny contour which contains all CTD lines."""
        import cv2

        if not np.any(raw):
            return None
        blurred = cv2.GaussianBlur(source, (3, 3), 0)
        edges = cv2.Canny(blurred, 70, 140, L2gradient=True)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
        edges[raw > 0] = 0
        h, w = raw.shape
        cv2.rectangle(edges, (0, 0), (w - 1, h - 1), 255, 1)
        contours, _ = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        text_points = cv2.findNonZero(raw)
        if text_points is None:
            return None
        tx, ty, tw, th = cv2.boundingRect(text_points)
        best, best_area = None, float(h * w)
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            if x > tx or y > ty or x + cw < tx + tw or y + ch < ty + th:
                continue
            area = float(cv2.contourArea(contour))
            if area <= np.count_nonzero(raw) * 1.15 or area >= h * w * 0.92 or area >= best_area:
                continue
            candidate = np.zeros_like(raw)
            cv2.drawContours(candidate, [contour], -1, 255, -1)
            if np.count_nonzero((raw > 0) & (candidate == 0)):
                continue
            if x <= 1 or y <= 1 or x + cw >= w - 1 or y + ch >= h - 1:
                continue
            best, best_area = candidate, area
        if best is None:
            return None
        return cv2.erode(best, np.ones((3, 3), np.uint8))

    @staticmethod
    def _protected_structures(source: np.ndarray, raw: np.ndarray, bubble: np.ndarray | None) -> np.ndarray:
        import cv2

        gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 60, 150)
        h, w = raw.shape
        protected = np.zeros_like(raw)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=35,
                                minLineLength=max(24, int(max(h, w) * .32)), maxLineGap=3)
        if lines is not None:
            for x1, y1, x2, y2 in lines[:, 0]:
                length = float(np.hypot(x2 - x1, y2 - y1))
                axis_aligned = abs(x2 - x1) <= length * .12 or abs(y2 - y1) <= length * .12
                if axis_aligned:
                    line_mask = np.zeros_like(raw)
                    cv2.line(line_mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, 3)
                    overlap = np.count_nonzero((line_mask > 0) & (raw > 0)) / max(1, np.count_nonzero(line_mask))
                    if overlap <= .08:
                        protected = cv2.bitwise_or(protected, line_mask)
        if bubble is not None:
            boundary = cv2.subtract(cv2.dilate(bubble, np.ones((5, 5), np.uint8)),
                                    cv2.erode(bubble, np.ones((5, 5), np.uint8)))
            protected = cv2.bitwise_or(protected, boundary)
        # Crop-edge geometry is context and must never become a glyph component.
        protected[[0, -1], :] = 255
        protected[:, [0, -1]] = 255
        return protected

    @staticmethod
    def _upstream_refine(source: np.ndarray, raw: np.ndarray,
                         segmentation: np.ndarray,
                         bubble: np.ndarray | None = None) -> tuple[np.ndarray, int]:
        """Adapt CTD textmask.py: line-local color/Otsu candidates + XOR CC merge."""
        import cv2

        if not np.any(raw):
            return raw.copy(), 0
        if np.any(segmentation):
            # Preserve CTD segmentation, but supplement line-local glyph evidence
            # from the sparsest Otsu polarity. This follows CTD textmask's channel
            # candidate model and recovers glyphs when segmentation is partial.
            predicted = segmentation.copy()
            for channel in cv2.split(source):
                _, thresholded = cv2.threshold(channel, 1, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
                direct = cv2.bitwise_and(thresholded, raw)
                inverse = cv2.bitwise_and(255 - thresholded, raw)
                options = [item for item in (direct, inverse) if np.any(item)]
                if options:
                    sparse = min(options, key=np.count_nonzero)
                    predicted = cv2.bitwise_or(predicted, sparse)
        else:
            gray_full = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
            _, otsu = cv2.threshold(gray_full, 1, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
            envelope = cv2.dilate(raw, np.ones((3, 3), np.uint8)) > 0
            direct = np.where(envelope, otsu, 0).astype(np.uint8)
            inverse = np.where(envelope, 255 - otsu, 0).astype(np.uint8)
            direct_count, inverse_count = np.count_nonzero(direct), np.count_nonzero(inverse)
            predicted = direct if 0 < direct_count < inverse_count else inverse
        final = np.zeros_like(raw)
        contours, _ = cv2.findContours(raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        heights: list[int] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w <= 0 or h <= 0:
                continue
            heights.append(h)
            pad = max(2, int(round(h * .15)))
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(raw.shape[1], x + w + pad), min(raw.shape[0], y + h + pad)
            image_crop = np.ascontiguousarray(source[y1:y2, x1:x2])
            pred_crop = np.ascontiguousarray(predicted[y1:y2, x1:x2])
            if not np.any(pred_crop):
                continue
            candidates: list[tuple[np.ndarray, int]] = []
            for channel in cv2.split(image_crop):
                _, thresholded = cv2.threshold(channel, 1, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
                candidates.append(self_or_inverse(thresholded, pred_crop))
            gray = cv2.cvtColor(image_crop, cv2.COLOR_RGB2GRAY)
            eroded = cv2.erode(pred_crop, np.ones((3, 3), np.uint8)) > 127
            values = gray[eroded]
            if values.size:
                hist, edges = np.histogram(values, bins=255, range=(0, 255))
                for index in np.argsort(hist)[::-1][:3]:
                    color = (edges[index] + edges[index + 1]) / 2
                    candidate = cv2.inRange(gray, max(0, color - 30), min(255, color + 30))
                    candidates.append(self_or_inverse(candidate, pred_crop))
            merged = merge_xor_components(candidates, pred_crop)
            final[y1:y2, x1:x2] = cv2.bitwise_or(final[y1:y2, x1:x2], merged)
        text_size = float(np.median(heights)) if heights else 12.0
        radius = int(np.clip(round(text_size * .08), 2, 4))
        # Bounded recovery: the XOR-merge target is seeded from the (often
        # partial) segmentation, so valid glyph pixels that lie inside the raw
        # CTD envelope but outside the partial segmentation can be rejected.
        # Recover text-like pixels within the raw envelope that are connected to
        # the detected glyphs (anti-aliased edges, detached strokes, punctuation,
        # partially clipped letters). Recovery is bounded to:
        #   - the raw CTD envelope (line_polygons ∪ segmentation_polygons)
        #   - the bubble interior when present, so background outside the bubble
        #     (speech bubble borders, faces, artwork, panel lines) can never be
        #     recovered; this also keeps the per-contour background estimate
        #     stable (interior pixels), preventing tiny labels from ballooning
        #     when the raw envelope is large.
        #   - pixels with strong contrast (>= 24 gray units) against the
        #     per-contour interior background, matching the residual-review gate
        #   - a line-height-adaptive dilation of the glyph mask, so isolated
        #     unrelated strokes are never bridged
        # _build() subsequently clips Hough-frame protected structures and the
        # bubble boundary, so any recovered frame/border pixels are discarded.
        recovered = final.copy()
        recovery_radius = int(np.clip(round(text_size * .25), 3, 8))
        recovery_kernel = np.ones((2 * recovery_radius + 1, 2 * recovery_radius + 1), np.uint8)
        interior = bubble if bubble is not None else None
        contour_evidence: list[tuple[tuple[int, int, int, int], np.ndarray, np.ndarray]] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w <= 0 or h <= 0:
                continue
            pad = max(2, int(round(h * .15)))
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(raw.shape[1], x + w + pad), min(raw.shape[0], y + h + pad)
            image_crop = np.ascontiguousarray(source[y1:y2, x1:x2])
            final_crop = final[y1:y2, x1:x2]
            if not np.any(final_crop):
                continue
            glyph_pixels = final_crop > 0
            if interior is not None:
                interior_crop = interior[y1:y2, x1:x2] > 0
                bg_pixels = image_crop[(~glyph_pixels) & interior_crop]
            else:
                bg_pixels = image_crop[~glyph_pixels]
            if len(bg_pixels) < 8:
                continue
            bg = np.median(bg_pixels, axis=0)
            bg_gray = float(np.dot(bg, [0.299, 0.587, 0.114]))
            gray = cv2.cvtColor(image_crop, cv2.COLOR_RGB2GRAY).astype(np.float32)
            contrast = np.abs(gray - bg_gray) >= 24
            if interior is not None:
                region_mask = (raw[y1:y2, x1:x2] > 0) & (interior[y1:y2, x1:x2] > 0)
            else:
                region_mask = raw[y1:y2, x1:x2] > 0
            contour_evidence.append(((x1, y1, x2, y2), contrast, region_mask))
        for _ in range(20):
            expanded = recovered.copy()
            changed = False
            for (x1, y1, x2, y2), contrast, region_mask in contour_evidence:
                recovered_crop = recovered[y1:y2, x1:x2]
                if not np.any(recovered_crop):
                    continue
                dilated = cv2.dilate(recovered_crop, recovery_kernel) > 0
                text_like = contrast & dilated & region_mask
                new_pixels = text_like & (recovered_crop == 0)
                if np.any(new_pixels):
                    expanded[y1:y2, x1:x2] = cv2.bitwise_or(
                        expanded[y1:y2, x1:x2], text_like.astype(np.uint8) * 255)
                    changed = True
            if not changed:
                break
            recovered = expanded
        final = recovered
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        return cv2.dilate(final, kernel), radius

    @staticmethod
    def _background_statistics(source: np.ndarray, raw: np.ndarray, refined: np.ndarray,
                               bubble: np.ndarray | None = None) -> tuple[tuple[int, int, int], bool]:
        import cv2

        area = (cv2.dilate(refined, np.ones((7, 7), np.uint8)) == 0)
        area &= (bubble > 0) if bubble is not None else (raw > 0)
        pixels = source[area]
        if len(pixels) < 24:
            pixels = source[refined == 0]
        if not len(pixels):
            return (255, 255, 255), False
        median = np.median(pixels, axis=0)
        distances = np.linalg.norm(pixels.astype(np.float32) - median, axis=1)
        lum = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2GRAY).reshape(-1)
        uniform = float(np.std(lum)) <= 9 and float(np.percentile(distances, 90)) <= 20
        return tuple(int(round(v)) for v in median), uniform
