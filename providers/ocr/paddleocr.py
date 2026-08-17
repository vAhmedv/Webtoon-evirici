"""PaddleOCR provider (candidate — PP-OCRv6 / en_PP-OCRv5).

PP-OCRv6 ve ``en_PP-OCRv5_mobile_rec`` adaylarını değerlendirmek için
isteğe bağlı bir sağlayıcı. ``paddleocr`` paketi kurulu değilse kayıtlı
değildir (registry try/except ile çevre). Kurulum yapılmaz: mevcut
PyTorch 2.11 / CUDA 12.8 ortamının bozulmaması prensibine uyulur.

ONNX Runtime backend'ini (``engine="onnxruntime"``) kullanır; bu da spec'in
tercih sırasındaki "ONNX Runtime backend" seçeneğine denk gelir.

Kullanım (manuel, isteğe bağlı):
    pip install paddleocr
    python scripts/download_ocr_models.py paddleocr_v6        # PP-OCRv6_medium_rec
    python scripts/download_ocr_models.py paddleocr_en_v5     # en_PP-OCRv5_mobile_rec

Model kaynakları / lisans: Apache-2.0 (PaddlePaddle/PaddleOCR).
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from core.detection import BBox
from providers.ocr.base import OCRLine, OCRProvider, OCRResult


class PaddleOCRProvider(OCRProvider):
    """PaddleOCR (ONNX Runtime) candidate provider.

    Args:
        model_name: Tanıma modeli. ``PP-OCRv6_medium_rec`` (varsayılan, 50 dil)
            veya ``en_PP-OCRv5_mobile_rec`` (İngilizce hafif).
    """

    def __init__(self, model_name: str = "PP-OCRv6_medium_rec") -> None:
        self._model_name = model_name
        self._loaded = False
        self._engine = None
        self._device = "cpu"

    @property
    def name(self) -> str:
        return f"PaddleOCR-{self._model_name}"

    @property
    def version(self) -> str:
        return self._model_name

    @property
    def device(self) -> str:
        return self._device

    @property
    def language(self) -> str:
        if self._model_name.startswith("en_"):
            return "en"
        return "multi"

    @property
    def status(self) -> str:
        return "candidate"

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        if self._loaded:
            return
        try:
            from paddleocr import PaddleOCR  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "PaddleOCR candidate için 'paddleocr' paketi gerekir.\n"
                "Kurulum (isteğe bağlı, env'ı korur): pip install paddleocr\n"
                "Not: Bu paketin kurulması PyTorch ortamını bozmayacaktır."
            ) from e
        # Güvenli default: CPU. PaddleOCR API "cpu" / "gpu:0" bekler.
        self._device = "cpu"

        logger.info(
            f"Loading PaddleOCR candidate: rec={self._model_name} "
            f"engine=onnxruntime device={self._device}"
        )
        self._engine = PaddleOCR(
            text_recognition_model_name=self._model_name,
            engine="onnxruntime",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device=self._device,
        )
        self._loaded = True
        logger.info(f"PaddleOCR candidate loaded: {self._model_name}")

    def unload(self) -> None:
        if not self._loaded:
            return
        self._engine = None
        self._loaded = False
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info("PaddleOCR candidate unloaded")

    def recognize_batch(
        self,
        images: Sequence[Any],
        region_bboxes: Sequence[BBox | None] | None = None,
        max_workers: int = 10,
    ) -> Sequence[OCRResult]:
        if not self._loaded:
            raise RuntimeError("PaddleOCR not loaded; call load() first")
        if not images:
            return []
        bboxes = region_bboxes if region_bboxes is not None else [None] * len(images)
        if len(images) == 1:
            return [self.recognize(images[0], bboxes[0])]

        import concurrent.futures
        results: list[OCRResult] = [None] * len(images)  # type: ignore
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(self.recognize, img, bbox): idx
                for idx, (img, bbox) in enumerate(zip(images, bboxes))
            }
            for fut in concurrent.futures.as_completed(futures):
                idx = futures[fut]
                results[idx] = fut.result()
        return results

    def recognize(
        self,
        image,
        region_bbox: BBox | None = None,
    ) -> OCRResult:
        if not self._loaded or self._engine is None:
            raise RuntimeError("PaddleOCR not loaded; call load() first")

        import numpy as np
        from PIL import Image

        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image

        results = self._engine.predict(img_array)
        lines = []
        texts: list[str] = []
        total_conf = 0.0
        count = 0

        # PaddleOCR 3.x: predict() sonuç listesi; içerikte rec_texts/rec_scores
        # veya eski rec_res/rec bulunabilir. Ayrıca bazı sürümler sonuçları
        # nested `res` içinde döndürür.
        for res in results:
            data = res.to_dict() if hasattr(res, "to_dict") else (res if isinstance(res, dict) else {})
            inner = data.get("res", data)
            if not isinstance(inner, dict):
                inner = {}

            rec_texts = inner.get("rec_texts") or inner.get("rec_text") or []
            rec_scores = inner.get("rec_scores") or inner.get("rec_score") or []

            if rec_texts and rec_scores and len(rec_texts) == len(rec_scores):
                for txt, conf in zip(rec_texts, rec_scores):
                    txt = str(txt)
                    try:
                        conf = float(conf)
                    except (TypeError, ValueError):
                        conf = 0.0
                    if not txt:
                        continue
                    lines.append(OCRLine(text=txt, confidence=conf))
                    texts.append(txt)
                    total_conf += conf
                    count += 1
            else:
                # Geri dönüşüm: eski rec_res / rec beklenen form.
                rec_res = inner.get("rec_res") or inner.get("rec") or []
                for item in rec_res:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        txt = str(item[0])
                        conf = float(item[1])
                    elif isinstance(item, dict):
                        txt = str(item.get("text", ""))
                        conf = float(item.get("score", item.get("confidence", 0.0)))
                    else:
                        txt = str(item)
                        conf = 0.0
                    if not txt:
                        continue
                    lines.append(OCRLine(text=txt, confidence=conf))
                    texts.append(txt)
                    total_conf += conf
                    count += 1

        raw_text = "\n".join(texts)
        normalized = " ".join(texts).strip()
        avg_conf = total_conf / count if count else 0.0

        warnings: list[str] = []
        if avg_conf < 0.5:
            warnings.append("low_ocr_confidence")
        if not normalized:
            warnings.append("empty_ocr_result")

        return OCRResult(
            text=normalized,
            confidence=avg_conf,
            raw_text=raw_text,
            lines=lines,
            warnings=warnings,
            metadata={
                "provider": self.name,
                "version": self._model_name,
                "language": self.language,
                "device": self._device,
                "region_bbox": _bbox_to_dict(region_bbox) if region_bbox else None,
            },
        )


def _bbox_to_dict(bbox: BBox) -> dict[str, int]:
    return {"x1": bbox.x1, "y1": bbox.y1, "x2": bbox.x2, "y2": bbox.y2}
