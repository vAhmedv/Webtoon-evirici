"""YOLOv8 comic text segmenter provider.

ogkalu/comic-text-segmenter-yolov8m modelini kullanır.
Manga, webtoon, manhua için eğitilmiştir.

Not: ultralytics paketi gerekir (opsiyonel dependency).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

from loguru import logger

from core.detection import BBox, Detection, RegionType
from providers.detector.base import DetectorProvider

MODEL_FILENAME = "comic-text-segmenter.pt"
MODEL_URL = "https://huggingface.co/ogkalu/comic-text-segmenter-yolov8m/resolve/main/comic-text-segmenter.pt"


class Yolo8ComicTextDetector(DetectorProvider):
    """YOLOv8-based comic text segmenter provider."""

    def __init__(self, model_path: str | Path | None = None, confidence_threshold: float = 0.25) -> None:
        self._model_path = Path(model_path) if model_path else Path(__file__).resolve().parent.parent.parent / "models" / "detectors" / "yolo8_comic" / MODEL_FILENAME
        self._loaded = False
        self._device = "cpu"
        self._model = None
        self._confidence_threshold = confidence_threshold

    @property
    def name(self) -> str:
        return "YOLOv8 Comic Text Segmenter"

    @property
    def version(self) -> str:
        return "yolov8m-seg"

    @property
    def device(self) -> str:
        return self._device

    @property
    def confidence_threshold(self) -> float:
        return self._confidence_threshold

    @confidence_threshold.setter
    def confidence_threshold(self, value: float) -> None:
        self._confidence_threshold = float(value)

    def load(self) -> None:
        if self._loaded:
            return

        if not self._model_path.exists():
            raise FileNotFoundError(
                f"YOLOv8 comic text model not found: {self._model_path}\n"
                f"Download from: {MODEL_URL}\n"
                f"Or run: python scripts/download_detector_models.py --detector yolo8_comic"
            )

        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise RuntimeError(
                "ultralytics package required for YOLOv8 detector.\n"
                "Install with: pip install ultralytics"
            ) from e

        import torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading YOLOv8 comic text model on {self._device}: {self._model_path}")

        self._model = YOLO(str(self._model_path))
        self._model.to(self._device)
        self._loaded = True
        logger.info("YOLOv8 comic text model loaded successfully")

    def unload(self) -> None:
        if not self._loaded:
            return
        del self._model
        self._model = None
        self._loaded = False
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def detect(self, image, window_id: int) -> Sequence[Detection]:
        if not self._loaded or self._model is None:
            raise RuntimeError("YOLOv8 model not loaded; call load() first")

        import numpy as np
        from PIL import Image

        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image

        results = self._model.predict(
            img_array,
            conf=self._confidence_threshold,
            iou=0.45,
            imgsz=1024,
            device=self._device,
            verbose=False,
        )

        detections: list[Detection] = []
        if not results or len(results) == 0:
            return detections

        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return detections

        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()

        masks_xy = None
        if r.masks is not None and hasattr(r.masks, "xy") and len(r.masks.xy) == len(boxes):
            masks_xy = [poly.cpu().numpy() if hasattr(poly, "cpu") else np.asarray(poly) for poly in r.masks.xy]

        for idx, (bbox, conf, cls) in enumerate(zip(boxes, confs, classes)):
            x1, y1, x2, y2 = bbox.astype(int).tolist()
            if x2 <= x1 or y2 <= y1:
                continue

            metadata: dict[str, object] = {}
            if masks_xy is not None and idx < len(masks_xy):
                poly = masks_xy[idx]
                if len(poly) >= 3:
                    metadata["polygon"] = poly.tolist()

            detections.append(
                Detection(
                    bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    confidence=float(conf),
                    type=RegionType.UNKNOWN,
                    source_window_id=window_id,
                    mask=None,
                    metadata=metadata,
                )
            )

        return detections
