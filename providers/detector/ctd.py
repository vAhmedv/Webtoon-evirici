"""ComicTextDetector (CTD) provider.

dmMaze/comic-text-detector tabanlı manga/comic text detector.
ONNX model ile çalışır (OpenCV DNN backend).

Lisans: GPL-3.0
Model kaynağı: https://github.com/zyddnys/manga-image-translator/releases
"""

from __future__ import annotations

import os
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

    def __init__(self, model_dir: str | Path | None = None) -> None:
        self._model_dir = Path(model_dir) if model_dir else Path(__file__).resolve().parent.parent.parent / "models" / "detectors" / "ctd"
        self._model_path = self._model_dir / MODEL_FILENAME
        self._loaded = False
        self._device = "cpu"
        self._net = None
        self._input_size = 1024
        self._conf_thresh = 0.4
        self._nms_thresh = 0.35

    @property
    def name(self) -> str:
        return "ComicTextDetector"

    @property
    def version(self) -> str:
        return "beta-0.3"

    @property
    def device(self) -> str:
        return self._device

    def load(self) -> None:
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
        if not self._loaded or self._net is None:
            raise RuntimeError("CTD model not loaded; call load() first")

        img = np.array(image)
        if img is None or img.size == 0:
            return []

        im_h, im_w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(img, scalefactor=1 / 255.0, size=(self._input_size, self._input_size))
        self._net.setInput(blob)
        blks, seg, det = self._net.forward(self._uoln)

        # Postprocess YOLO output
        blks = self._postprocess_yolo(blks, im_w, im_h)

        detections: list[Detection] = []
        for bbox in blks:
            x1, y1, x2, y2 = bbox
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(
                Detection(
                    bbox=BBox(x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2)),
                    confidence=0.9,
                    type=RegionType.UNKNOWN,
                    source_window_id=window_id,
                )
            )

        return detections

    def _postprocess_yolo(self, det, im_w, im_h):
        """YOLO output'tan NMS ile filtrelenmiş bbox'lar çıkarır."""
        det = np.array(det).reshape(-1, 7)
        if det.size == 0:
            return []

        confs = det[:, 4]
        mask = confs > self._conf_thresh
        det = det[mask]

        if det.shape[0] == 0:
            return []

        boxes = det[:, :4].copy()
        scores = det[:, 4].copy()

        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), self._conf_thresh, self._nms_thresh)
        if len(indices) == 0:
            return []

        indices = indices.flatten()
        boxes = boxes[indices]
        scores = scores[indices]

        scale_x = im_w / self._input_size
        scale_y = im_h / self._input_size
        boxes[:, 0] *= scale_x
        boxes[:, 1] *= scale_y
        boxes[:, 2] *= scale_x
        boxes[:, 3] *= scale_y

        return boxes
