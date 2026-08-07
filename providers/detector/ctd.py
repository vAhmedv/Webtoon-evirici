"""ComicTextDetector (CTD) provider.

dmMaze/comic-text-detector tabanlı manga/comic text detector.
ONNX model ile çalışır (OpenCV DNN backend).

Pipeline:
1. Preprocessing (letterbox, normalize)
2. ONNX inference (3 outputs: YOLO blocks, DBNet lines, segmentation mask)
3. YOLO block postprocessing (NMS)
4. DBNet text-line extraction
5. Segmentation mask processing
6. Grouping
7. Canonical Detection output

Lisans: GPL-3.0
Model kaynağı: https://github.com/zyddnys/manga-image-translator/releases
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from loguru import logger

from core.detection import BBox, Detection, RegionType
from providers.detector.base import DetectorProvider

MODEL_FILENAME = "comictextdetector.pt.onnx"
MODEL_URL = "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/comictextdetector.pt.onnx"
MODEL_SHA256 = "1a86ace74961413cbd650002e7bb4dcec4980ffa21b2f19b86933372071d718f"


class ComicTextDetector(DetectorProvider):
    """ComicTextDetector provider (ONNX/OpenCV backend)."""

    def __init__(self, model_dir: str | Path | None = None, input_size: int = 1024) -> None:
        logger.debug(f"[THREAD] ComicTextDetector.__init__ thread id: {threading.get_ident()}")
        self._model_dir = Path(model_dir) if model_dir else Path(__file__).resolve().parent.parent.parent / "models" / "detectors" / "ctd"
        self._model_path = self._model_dir / MODEL_FILENAME
        self._loaded = False
        self._device = "cpu"
        self._net = None
        self._input_size = input_size
        self._conf_thresh = 0.4
        self._nms_thresh = 0.35
        self._seg_thresh = 0.3
        self._box_thresh = 0.6
        self._debug = False
        self._debug_dir = None

    @property
    def name(self) -> str:
        return "ComicTextDetector"

    @property
    def version(self) -> str:
        return "beta-0.3"

    @property
    def device(self) -> str:
        return self._device

    def set_debug(self, enabled: bool, output_dir: str | Path | None = None) -> None:
        """Debug modunu ayarlar."""
        self._debug = enabled
        if enabled and output_dir is not None:
            self._debug_dir = Path(output_dir)

    def load(self) -> None:
        logger.debug(f"[THREAD] ComicTextDetector.load thread id: {threading.get_ident()}")
        if self._loaded:
            return

        if not self._model_path.exists():
            raise FileNotFoundError(
                f"CTD ONNX model not found: {self._model_path}\n"
                f"Download from: {MODEL_URL}\n"
                f"Or run: python scripts/download_detector_models.py --detector ctd"
            )

        logger.info(f"Loading CTD ONNX model: {self._model_path}")
        self._net = cv2.dnn.readNetFromONNX(str(self._model_path))
        self._uoln = self._net.getUnconnectedOutLayersNames()
        self._loaded = True
        logger.info("CTD ONNX model loaded successfully")

    def unload(self) -> None:
        if not self._loaded:
            return
        self._net = None
        self._loaded = False

    def detect(self, image, window_id: int) -> Sequence[Detection]:
        logger.debug(f"[THREAD] ComicTextDetector.detect thread id: {threading.get_ident()}")
        if not self._loaded or self._net is None:
            raise RuntimeError("CTD model not loaded; call load() first")

        img = np.array(image)
        if img is None or img.size == 0:
            return []

        im_h, im_w = img.shape[:2]

        # Preprocess
        img_in, ratio, dw, dh = self._preprocess(img)
        scale_x = im_w / (self._input_size - dw)
        scale_y = im_h / (self._input_size - dh)

        if self._debug:
            self._save_debug("01_input.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            lb_vis = (img_in[0].transpose((1, 2, 0)) * 255).astype(np.uint8)
            self._save_debug("02_letterboxed.png", cv2.cvtColor(lb_vis, cv2.COLOR_RGB2BGR))

        # Inference
        self._net.setInput(img_in)
        outputs = self._net.forward(self._uoln)

        # Map outputs by name
        output_names = self._net.getUnconnectedOutLayersNames()
        output_dict = dict(zip(output_names, outputs))

        blk_output = output_dict.get("blk")
        det_output = output_dict.get("det")
        seg_output = output_dict.get("seg")

        if self._debug:
            self._log_output_info("blk", blk_output)
            self._log_output_info("det", det_output)
            self._log_output_info("seg", seg_output)

        # YOLO block postprocessing
        blocks = []
        if blk_output is not None:
            blocks = self._postprocess_yolo_blocks(blk_output, scale_x, scale_y, im_w, im_h)

        # DBNet text-line extraction
        lines = []
        if det_output is not None:
            lines = self._postprocess_dbnet_lines(det_output, scale_x, scale_y, im_w, im_h)

        # Segmentation mask
        seg_mask = None
        if seg_output is not None:
            seg_mask = self._postprocess_segmentation_mask(seg_output, im_w, im_h, dh)
            if self._debug:
                mask_vis = (seg_mask * 255).astype(np.uint8) if seg_mask.max() <= 1.0 else seg_mask.astype(np.uint8)
                cv2.imwrite(str(self._debug_dir / "05_segmentation_mask.png"), mask_vis)

        # Grouping
        grouped_blocks = self._group_blocks_and_lines(blocks, lines, seg_mask, im_w, im_h)

        if self._debug:
            self._save_grouped_visualization(img, grouped_blocks, im_w, im_h)

        # Convert to Detection
        detections: list[Detection] = []
        for block in grouped_blocks:
            x1, y1, x2, y2 = block["bbox"]
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(
                Detection(
                    bbox=BBox(x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2)),
                    confidence=float(block.get("confidence", 0.5)),
                    type=RegionType.UNKNOWN,
                    source_window_id=window_id,
                )
            )

        return detections

    def _preprocess(self, img: np.ndarray) -> tuple[np.ndarray, tuple[float, float], int, int]:
        """CTD-style letterbox preprocessing."""
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_in, ratio, (dw, dh) = self._letterbox(img_bgr, new_shape=(self._input_size, self._input_size), auto=False, stride=64)
        img_in = img_in.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img_in = np.ascontiguousarray(img_in)
        img_in = img_in.astype(np.float32) / 255.0
        img_in = np.expand_dims(img_in, axis=0)
        return img_in, ratio, dw, dh

    def _letterbox(self, im, new_shape=(640, 640), color=(0, 0, 0), auto=False, scaleFill=False, scaleup=True, stride=128):
        """Letterbox resize with aspect ratio preservation."""
        shape = im.shape[:2]
        if not isinstance(new_shape, tuple):
            new_shape = (new_shape, new_shape)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:
            r = min(r, 1.0)
        ratio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        if auto:
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)
        dh, dw = int(dh), int(dw)
        if shape[::-1] != new_unpad:
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        im = cv2.copyMakeBorder(im, 0, dh, 0, dw, cv2.BORDER_CONSTANT, value=color)
        return im, ratio, (dw, dh)

    def _postprocess_yolo_blocks(self, blk_output, scale_x, scale_y, im_w, im_h):
        """YOLO block output'tan NMS ile filtrelenmiş bbox'lar çıkarır."""
        blk_output = np.array(blk_output).reshape(-1, 7)
        if blk_output.size == 0:
            return []

        # Filter by confidence
        confs = blk_output[:, 4]
        mask = confs > self._conf_thresh
        blk_output = blk_output[mask]

        if blk_output.shape[0] == 0:
            return []

        boxes = blk_output[:, :4].copy()
        scores = blk_output[:, 4].copy()

        # Convert from [x1, y1, x2, y2] to [x1, y1, w, h] for NMSBoxes
        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), self._conf_thresh, self._nms_thresh)
        if len(indices) == 0:
            return []

        indices = indices.flatten()
        boxes = boxes[indices]
        scores = scores[indices]

        # Convert back to [x1, y1, x2, y2] and scale to original image
        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 1]
        boxes[:, 0] *= scale_x
        boxes[:, 1] *= scale_y
        boxes[:, 2] *= scale_x
        boxes[:, 3] *= scale_y

        result = []
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box
            result.append({
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "confidence": float(score),
                "type": "block",
            })
        return result

    def _postprocess_dbnet_lines(self, det_output, scale_x, scale_y, im_w, im_h):
        """DBNet text-line probability map'ten line polygon'ları çıkarır."""
        # det_output shape: (1, 2, H, W) or similar
        det_output = np.array(det_output)
        if det_output.ndim != 4:
            return []

        # Assume first channel is shrink map, second is threshold map
        # Or if single channel, use directly
        if det_output.shape[1] >= 2:
            shrink_map = det_output[0, 0, :, :]
            thresh_map = det_output[0, 1, :, :]
        else:
            shrink_map = det_output[0, 0, :, :]
            thresh_map = shrink_map

        # Binarize
        binary_map = (shrink_map > self._seg_thresh).astype(np.uint8) * 255

        # Find contours
        contours, _ = cv2.findContours(binary_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        lines = []
        for contour in contours:
            if len(contour) < 3:
                continue

            # Approximate polygon
            epsilon = 0.005 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) < 4:
                continue

            # Score based on shrink/threshold map
            score = self._compute_polygon_score(approx, shrink_map)
            if score < self._box_thresh:
                continue

            # Unclip/expand polygon
            expanded = self._unclip_polygon(approx, unclip_ratio=1.5)
            if expanded is None or len(expanded) < 4:
                continue

            # Scale to original image coordinates
            expanded = expanded.astype(np.float32)
            expanded[:, 0] *= scale_x
            expanded[:, 1] *= scale_y

            # Clip to image bounds
            expanded[:, 0] = np.clip(expanded[:, 0], 0, im_w - 1)
            expanded[:, 1] = np.clip(expanded[:, 1], 0, im_h - 1)

            lines.append({
                "polygon": expanded.astype(int),
                "score": float(score),
            })

        return lines

    def _compute_polygon_score(self, polygon, score_map):
        """Polygon içindeki ortalama skoru hesaplar."""
        mask = np.zeros(score_map.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 1)
        if mask.sum() == 0:
            return 0.0
        return float(score_map[mask > 0].mean())

    def _unclip_polygon(self, polygon, unclip_ratio=1.5):
        """Polygon'u genişletir (unclip)."""
        polygon = polygon.reshape(-1, 2)
        if len(polygon) < 3:
            return None

        # Simple scaling approach instead of pyclipper
        center = polygon.mean(axis=0)
        expanded = center + (polygon - center) * unclip_ratio
        return expanded.astype(np.int32)

    def _postprocess_segmentation_mask(self, seg_output, im_w, im_h, dh):
        """Segmentation mask'ı original image boyutuna getirir."""
        mask = seg_output[0, 0, :, :]
        # Remove padding
        if dh > 0:
            mask = mask[:-dh, :]
        # Threshold
        mask = (mask > self._seg_thresh).astype(np.uint8) * 255
        # Resize to original image size
        mask = cv2.resize(mask, (im_w, im_h), interpolation=cv2.INTER_LINEAR)
        return mask

    def _group_blocks_and_lines(self, blocks, lines, seg_mask, im_w, im_h):
        """YOLO block'ları ve DBNet line'ları gruplayarak canonical block'lar üretir."""
        if not blocks and not lines:
            return []

        if not blocks:
            # Create blocks from lines if no YOLO blocks
            for line in lines:
                x1, y1 = line["polygon"].min(axis=0)
                x2, y2 = line["polygon"].max(axis=0)
                blocks.append({
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": line["score"],
                    "type": "block",
                    "lines": [line],
                })
            return blocks

        # Assign lines to blocks
        for block in blocks:
            block["lines"] = []

        bbox_score_thresh = 0.4
        mask_score_thresh = 0.1

        for line in lines:
            bx1, by1, bx2, by2 = line["polygon"].min(axis=0)[0], line["polygon"].min(axis=0)[1], \
                                  line["polygon"].max(axis=0)[0], line["polygon"].max(axis=0)[1]
            line_area = (by2 - by1) * (bx2 - bx1)
            bbox_score = -1
            bbox_idx = -1

            for jj, blk in enumerate(blocks):
                blk_bbox = blk["bbox"]
                score = self._union_area(blk_bbox, [bx1, by1, bx2, by2]) / (line_area + 1e-6)
                if score > bbox_score:
                    bbox_score = score
                    bbox_idx = jj

            if bbox_score > bbox_score_thresh:
                blocks[bbox_idx]["lines"].append(line)
            elif seg_mask is not None:
                # Check mask score
                mask_score = seg_mask[int(by1):int(by2), int(bx1):int(bx2)].mean() / 255.0
                if mask_score >= mask_score_thresh:
                    blocks.append({
                        "bbox": [float(bx1), float(by1), float(bx2), float(by2)],
                        "confidence": line["score"],
                        "type": "block",
                        "lines": [line],
                    })

        # Filter blocks with no lines (use mask if available)
        filtered_blocks = []
        for blk in blocks:
            if len(blk.get("lines", [])) == 0:
                bx1, by1, bx2, by2 = blk["bbox"]
                if seg_mask is not None:
                    mask_score = seg_mask[int(by1):int(by2), int(bx1):int(bx2)].mean() / 255.0
                    if mask_score < mask_score_thresh:
                        continue
                # Create synthetic line from bbox
                blk["lines"] = [{
                    "polygon": np.array([[bx1, by1], [bx2, by1], [bx2, by2], [bx1, by2]]),
                    "score": blk["confidence"],
                }]
            filtered_blocks.append(blk)

        return filtered_blocks

    def _union_area(self, bbox_a, bbox_b):
        """İki bbox'ın kesişim alanını hesaplar."""
        x1 = max(bbox_a[0], bbox_b[0])
        y1 = max(bbox_a[1], bbox_b[1])
        x2 = min(bbox_a[2], bbox_b[2])
        y2 = min(bbox_a[3], bbox_b[3])
        if y2 < y1 or x2 < x1:
            return 0
        return (y2 - y1) * (x2 - x1)

    def _save_debug(self, filename: str, image: np.ndarray) -> None:
        """Debug görseli kaydeder."""
        if self._debug_dir is None:
            return
        path = self._debug_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), image)

    def _log_output_info(self, name: str, tensor) -> None:
        """ONNX output bilgisini loglar."""
        if tensor is None:
            logger.warning(f"Output '{name}' is None")
            return
        logger.info(
            f"Output '{name}': shape={tensor.shape}, dtype={tensor.dtype}, "
            f"min={tensor.min():.4f}, max={tensor.max():.4f}, mean={tensor.mean():.4f}"
        )

    def _save_grouped_visualization(self, img: np.ndarray, blocks: list[dict], im_w: int, im_h: int) -> None:
        """Gruplanmış block'ları görselleştirir."""
        vis = img.copy()
        for block in blocks:
            x1, y1, x2, y2 = map(int, block["bbox"])
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            for line in block.get("lines", []):
                poly = line["polygon"].astype(int)
                cv2.polylines(vis, [poly], True, (0, 0, 255), 1)
        self._save_debug("08_final.png", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
