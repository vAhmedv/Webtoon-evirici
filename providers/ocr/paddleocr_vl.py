"""PaddleOCR-VL-1.6 provider (primary OCR).

PaddlePaddle/PaddleOCR-VL-1.6 modelini native Transformers ile kullanır.
Resmî PaddleOCR-VL-1.6 Transformers element-OCR yöntemini kaynak alır;
``OCR:`` görevini kullanır.

Model:
- BF16
- cuda:0
- quantization yok
- CPU offload yok
- device_map="auto" yok

Mevcut PyTorch/Transformers ortamını bozmaz (dependency downgrade yapılmaz).
"""

from __future__ import annotations

import time
from pathlib import Path

from loguru import logger

from core.detection import BBox
from core.ocr_normalizer import normalize_ocr_text
from providers.ocr.base import OCRLine, OCRProvider, OCRResult

MODEL_ID = "PaddlePaddle/PaddleOCR-VL-1.6"
TASK_PROMPT = "OCR:"
MAX_NEW_TOKENS = 128


class PaddleOCRVLOcrProvider(OCRProvider):
    """PaddleOCR-VL-1.6 primary OCR provider (Transformers native)."""

    def __init__(self, model_id: str = MODEL_ID) -> None:
        self._model_id = model_id
        self._loaded = False
        self._device = "cpu"
        self._processor = None
        self._model = None

    @property
    def name(self) -> str:
        return "PaddleOCR-VL-1.6"

    @property
    def version(self) -> str:
        return "1.6"

    @property
    def device(self) -> str:
        return self._device

    @property
    def language(self) -> str:
        return "en"

    @property
    def status(self) -> str:
        return "candidate/default"

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        if self._loaded:
            return
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as e:
            raise RuntimeError(
                "PaddleOCR-VL-1.6 için 'transformers' ve 'torch' gerekir."
            ) from e

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(
            f"Loading PaddleOCR-VL-1.6 on {self._device}: {self._model_id}"
        )

        self._processor = AutoProcessor.from_pretrained(self._model_id)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self._model_id,
            torch_dtype=torch.bfloat16,
        )
        self._model = self._model.to("cuda:0" if self._device == "cuda" else self._device)
        self._model.eval()
        self._loaded = True
        logger.info("PaddleOCR-VL-1.6 loaded successfully")

    def unload(self) -> None:
        if not self._loaded:
            return
        self._model = None
        self._processor = None
        self._loaded = False
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info("PaddleOCR-VL-1.6 unloaded")

    def recognize_batch(
        self,
        images: Sequence[Any],
        region_bboxes: Sequence[BBox | None] | None = None,
        batch_size: int = 32,
    ) -> Sequence[OCRResult]:
        if not self._loaded or self._model is None or self._processor is None:
            raise RuntimeError("PaddleOCR-VL-1.6 not loaded; call load() first")
        if not images:
            return []

        import torch
        from PIL import Image

        bboxes = region_bboxes if region_bboxes is not None else [None] * len(images)
        if len(images) == 1:
            return [self.recognize(images[0], bboxes[0])]

        self._processor.tokenizer.padding_side = "left"
        pairs = list(zip(images, bboxes))

        def _forward_chunk(chunk_pairs: Sequence[tuple[Any, BBox | None]]) -> list[OCRResult]:
            chunk_imgs = [p[0] for p in chunk_pairs]
            chunk_bboxes = [p[1] for p in chunk_pairs]
            chunk_results: list[OCRResult] = []

            pil_chunk = []
            for img in chunk_imgs:
                if hasattr(img, "to_pil"):
                    pil_chunk.append(img.to_pil().convert("RGB"))
                elif hasattr(img, "detach") and hasattr(img, "permute"):
                    arr = img.detach().cpu().permute(1, 2, 0).numpy()
                    pil_chunk.append(Image.fromarray(arr).convert("RGB"))
                elif isinstance(img, Image.Image):
                    pil_chunk.append(img.convert("RGB"))
                else:
                    pil_chunk.append(Image.fromarray(img).convert("RGB"))

            messages_batch = [
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": TASK_PROMPT},
                        ],
                    }
                ]
                for img in pil_chunk
            ]

            texts_input = [
                self._processor.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
                for msg in messages_batch
            ]
            images_input = [[img] for img in pil_chunk]

            ip = self._processor.image_processor
            size = dict(ip.size) if hasattr(ip.size, "items") else {}
            size["shortest_edge"] = size.get("shortest_edge", 112896)
            size["longest_edge"] = 1280 * 28 * 28

            inputs = self._processor(
                text=texts_input,
                images=images_input,
                padding=True,
                return_tensors="pt",
            )
            if hasattr(inputs, "to"):
                inputs = inputs.to(self._model.device)
            elif isinstance(inputs, dict):
                inputs = {k: v.to(self._model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

            start = time.perf_counter()
            with torch.inference_mode():
                outputs = self._model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, use_cache=True)
            elapsed = (time.perf_counter() - start) / len(chunk_imgs)

            input_len = inputs["input_ids"].shape[1]
            for i, out in enumerate(outputs):
                gen_tokens = out[input_len:]
                raw_text = self._processor.decode(gen_tokens, skip_special_tokens=True)
                canonical = normalize_ocr_text(raw_text)
                warnings: list[str] = []
                if not canonical:
                    warnings.append("empty_ocr_result")

                reg_bbox = chunk_bboxes[i]
                chunk_results.append(
                    OCRResult(
                        text=canonical,
                        confidence=None,
                        raw_text=raw_text,
                        lines=[],
                        warnings=warnings,
                        metadata={
                            "provider": self.name,
                            "version": self.version,
                            "language": self.language,
                            "device": self._device,
                            "model_id": self._model_id,
                            "inference_seconds": round(elapsed, 3),
                            "region_bbox": _bbox_to_dict(reg_bbox) if reg_bbox else None,
                        },
                    )
                )
            return chunk_results

        if not hasattr(self, "_batcher") or self._batcher is None:
            from core.system.adaptive_batcher import ElasticAdaptiveBatcher
            self._batcher = ElasticAdaptiveBatcher(default_batch_size=batch_size, min_batch_size=1, vram_ceiling=0.95)

        return self._batcher.execute(pairs, _forward_chunk, batch_size=batch_size)

    def recognize(
        self,
        image,
        region_bbox: BBox | None = None,
    ) -> OCRResult:
        if not self._loaded or self._model is None or self._processor is None:
            raise RuntimeError("PaddleOCR-VL-1.6 not loaded; call load() first")

        import torch
        from PIL import Image

        if hasattr(image, "to_pil"):
            pil_image = image.to_pil().convert("RGB")
        elif hasattr(image, "detach") and hasattr(image, "permute"):
            arr = image.detach().cpu().permute(1, 2, 0).numpy()
            pil_image = Image.fromarray(arr).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        else:
            pil_image = Image.fromarray(image).convert("RGB")

        start = time.perf_counter()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": TASK_PROMPT},
                ],
            }
        ]
        # PaddleOCR-VL-1.6 image processor uses `size` dict (longest_edge/shortest_edge),
        # not min_pixels/max_pixels. OCR task: longest_edge = 1280 * 28 * 28.
        ip = self._processor.image_processor
        size = dict(ip.size) if hasattr(ip.size, "items") else {}
        size["shortest_edge"] = size.get("shortest_edge", 112896)
        size["longest_edge"] = 1280 * 28 * 28
        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"images_kwargs": {"size": size}},
        )
        if hasattr(inputs, "to"):
            inputs = inputs.to(self._model.device)
        elif isinstance(inputs, dict):
            inputs = {k: v.to(self._model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self._model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, use_cache=True)
        raw_text = self._processor.decode(
            outputs[0][inputs["input_ids"].shape[-1]:-1]
        )
        elapsed = time.perf_counter() - start

        canonical = normalize_ocr_text(raw_text)
        warnings: list[str] = []
        if not canonical:
            warnings.append("empty_ocr_result")

        return OCRResult(
            text=canonical,
            confidence=None,  # VL-1.6 confidence vermez; uydurma yapılmaz
            raw_text=raw_text,
            lines=[],
            warnings=warnings,
            metadata={
                "provider": self.name,
                "version": self.version,
                "language": self.language,
                "device": self._device,
                "model_id": self._model_id,
                "inference_seconds": round(elapsed, 3),
                "region_bbox": _bbox_to_dict(region_bbox) if region_bbox else None,
            },
        )


def _bbox_to_dict(bbox: BBox) -> dict[str, int]:
    return {"x1": bbox.x1, "y1": bbox.y1, "x2": bbox.x2, "y2": bbox.y2}