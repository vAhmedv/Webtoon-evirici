"""YOLOv8 konuşma-balonu detektörü (A2: sivri balon gözü).

ogkalu/comic-speech-bubble-detector-yolov8m (Apache-2.0; 8k manga/webtoon/
manhua/western, imgsz 1024, aşırı-en-boy oranlarına dayanıklı).
Canny'nin kapatamadığı sivri/patlak balonları kutu olarak verir; KESİN
ŞEKİL için değil, balon-İÇİ maske işlerine sınır olarak kullanılır.
Model diskte YOKSA load() FileNotFoundError verir — ASLA indirme denenmez.
Ağırlık models/ altında, repoya girmez (.gitignore).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

from loguru import logger

from core.detection import BBox, Detection, RegionType
from providers.detector.base import DetectorProvider

MODEL_FILENAME = "comic-speech-bubble-detector.pt"
MODEL_URL = "https://huggingface.co/ogkalu/comic-speech-bubble-detector-yolov8m"


class YoloBubbleDetector(DetectorProvider):
    """Sayfa-düzeyi YOLOv8m balon kutuları (metin değil, BALON)."""

    def __init__(self, model_path: str | Path | None = None, confidence_threshold: float = 0.35) -> None:
        default = (
            Path(__file__).resolve().parent.parent.parent
            / "models" / "detectors" / "yolo8_bubble" / MODEL_FILENAME
        )
        self._model_path = Path(model_path) if model_path else default
        self._loaded = False
        self._device = "cpu"
        self._model = None
        self._confidence_threshold = confidence_threshold

    @property
    def name(self) -> str:
        return "YOLOv8 Comic Speech-Bubble Detector"

    @property
    def version(self) -> str:
        return "yolov8m-bubble"

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
                f"YOLOv8 bubble model not found: {self._model_path}\n"
                f"Download from: {MODEL_URL}\n"
                "Pipeline continues WITHOUT bubble model (fail-open)."
            )
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise RuntimeError("ultralytics package required for YOLO bubble detector.") from e
        import torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading YOLO bubble model on {self._device}: {self._model_path}")
        self._model = YOLO(str(self._model_path))
        self._model.to(self._device)
        self._loaded = True

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
            raise RuntimeError("YOLO bubble model not loaded; call load() first")
        import numpy as np
        from PIL import Image

        img_array = np.array(image) if isinstance(image, Image.Image) else image
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
        for bbox, conf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = bbox.astype(int).tolist()
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(
                Detection(
                    bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    confidence=float(conf),
                    type=RegionType.UNKNOWN,
                    source_window_id=window_id,
                    mask=None,
                    metadata={"yolo_bubble": True},
                )
            )
        return detections
