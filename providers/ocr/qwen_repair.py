"""Qwen visual OCR repair / adjudication provider.

Qwen3.5-9B (8-bit) modeli bubble crop görüntüsünü okur ve
OCR disagreement'ı çözer.

Gorevi: crop'daki gerçek İngilizce metni belirlemek.
Translator değildir. OCR adayları sadece ipucdur.

Model yalnızca ``verdict.needs_repair == True`` olduğunda çalışır.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from PIL import Image

from providers.ocr.repair import OCRRepairInput, OCRRepairProvider, OCRRepairResult

DEFAULT_MODEL_PATH = r"C:\AI\Models\Qwen3.5-9B"
FALLBACK_MODEL_PATH = r"C:\AI\Models\Qwen3.5-4B"
MAX_NEW_TOKENS = 512
_TORCH_DTYPE = torch.bfloat16

# Kısa, odaklı prompt. JSON talep eder ama model doğal dil de çıkarabilir;
# parser buna göre esnek çalışır.
_USER_PROMPT = (
    "Read the English text in the provided bubble crop image.\n"
    "Two OCR candidates disagree:\n"
    "A: {primary}\n"
    "B: {verifier}\n"
    "Disagreement: {reason}\n"
    "{glossary}{context}\n"
    "Output JSON: {{\"status\": \"resolved\", \"text\": \"<text or null>\", \"reason\": \"<why>\"}}"
)


@dataclass(frozen=True)
class QwenRepairConfig:
    model_path: str = DEFAULT_MODEL_PATH
    fallback_model_path: str = FALLBACK_MODEL_PATH
    max_new_tokens: int = MAX_NEW_TOKENS
    torch_dtype: Any = _TORCH_DTYPE
    max_memory_gb: int = 12


@dataclass(frozen=True)
class QwenRepairInput:
    image: Image.Image
    primary_raw: str
    primary_normalized: str
    verifier_raw: str
    verifier_normalized: str
    reason: str
    known_names: list[str] = field(default_factory=list)
    nearby_context: str | None = None

    @classmethod
    def from_repair_input(cls, ri: OCRRepairInput, image: Image.Image) -> "QwenRepairInput":
        return cls(
            image=image,
            primary_raw=ri.primary_raw,
            primary_normalized=ri.primary_normalized,
            verifier_raw=ri.verifier_raw,
            verifier_normalized=ri.verifier_normalized,
            reason=ri.reason,
            known_names=list(ri.known_names),
            nearby_context=ri.nearby_context,
        )


@dataclass
class QwenRepairMetrics:
    model_load_vram_gb: float = 0.0
    peak_vram_gb: float = 0.0
    repair_model: str = ""


@dataclass
class OCRAdjudicatedResult:
    verdict: Any
    clean_source_text: str | None
    repair_result: OCRRepairResult | None = None
    repair_model: str | None = None
    requires_review: bool = False


class QwenRepairProvider(OCRRepairProvider):
    """Qwen3.5-9B visual OCR repair.

    Load stratejisi: 8-bit (bitsandbytes) -> 4-bit -> Qwen3.5-4B BF16 fallback.
    Model yalnızca bir kez load edilir; 3 case çalıştırıldıktan sonra unload edilir.
    """

    def __init__(self, config: QwenRepairConfig | None = None) -> None:
        self._config = config or QwenRepairConfig()
        self._loaded = False
        self._model = None
        self._processor = None
        self._device = "cpu"
        self.metrics = QwenRepairMetrics()

    @property
    def name(self) -> str:
        return "Qwen3.5-9B-OCR-Repair"

    @property
    def version(self) -> str:
        return "3.5-9B"

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def device(self) -> str:
        return self._device

    def _vram_gb(self) -> float:
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.memory_allocated() / (1024**3)

    def _peak_vram_gb(self) -> float:
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.max_memory_allocated() / (1024**3)

    def _reset_peak_vram(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def load(self) -> None:
        if self._loaded:
            return

        from transformers import AutoModelForImageTextToText, AutoProcessor

        if not torch.cuda.is_available():
            logger.info("No CUDA — loading Qwen3.5-9B on CPU")
            self._processor = AutoProcessor.from_pretrained(self._config.model_path, local_files_only=True)
            self._model = AutoModelForImageTextToText.from_pretrained(
                self._config.model_path, local_files_only=True, torch_dtype=self._config.torch_dtype
            )
            self._model.eval()
            self._device = "cpu"
            self.metrics.repair_model = "Qwen3.5-9B-bf16-cpu"
            self._loaded = True
            return

        errors: list[str] = []
        for quant in ("8bit", "4bit"):
            try:
                torch.cuda.empty_cache()
                label = self._load_quant(quant)
                self._loaded = True
                self.metrics.model_load_vram_gb = self._vram_gb()
                self.metrics.repair_model = f"Qwen3.5-9B-{label}"
                logger.info(f"Model-load VRAM: {self.metrics.model_load_vram_gb:.2f} GB ({label})")
                return
            except (RuntimeError, ValueError, OSError) as e:
                errors.append(f"{quant}: {e}")
                logger.warning(f"Qwen {quant} load failed: {e}")
                self._model = None
                self._processor = None
                torch.cuda.empty_cache()

        # Fallback: 4B BF16
        fb = self._load_fallback()
        if fb:
            self._loaded = True
            self.metrics.model_load_vram_gb = self._vram_gb()
            self.metrics.repair_model = fb
            logger.info(f"Fallback loaded: {fb}")
            return

        raise RuntimeError(f"Qwen load failed: {errors}")

    def _load_quant(self, quant: str) -> str:
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

        cfg = self._config
        bnb = None
        label = "bf16"
        if quant == "8bit":
            bnb = BitsAndBytesConfig(load_in_8bit=True)
            label = "8bit"
        elif quant == "4bit":
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=_TORCH_DTYPE,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            label = "4bit-nf4"

        self._processor = AutoProcessor.from_pretrained(self._config.model_path, local_files_only=True)
        kw: dict[str, Any] = {
            "local_files_only": True,
            "torch_dtype": cfg.torch_dtype,
            "device_map": "auto",
            "max_memory": {0: f"{cfg.max_memory_gb}GiB"},
        }
        if bnb:
            kw["quantization_config"] = bnb
        self._model = AutoModelForImageTextToText.from_pretrained(self._config.model_path, **kw)
        self._model.eval()
        self._device = "cuda"
        if torch.cuda.is_available():
            self._model = self._model.to("cuda:0")
        return label

    def _load_fallback(self) -> str | None:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        p = Path(self._config.fallback_model_path)
        if not p.exists():
            logger.warning(f"Fallback path not found: {p}")
            return None
        self._processor = AutoProcessor.from_pretrained(str(p), local_files_only=True)
        self._model = AutoModelForImageTextToText.from_pretrained(
            str(p), local_files_only=True, torch_dtype=self._config.torch_dtype,
            device_map="auto", max_memory={0: f"{self._config.max_memory_gb}GiB"},
        )
        self._model.eval()
        self._device = "cuda"
        if torch.cuda.is_available():
            self._model = self._model.to("cuda:0")
        return "Qwen3.5-4B-bf16"

    def unload(self) -> None:
        self._model = None
        self._processor = None
        self._loaded = False
        self._device = "cpu"
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Qwen repair model unloaded")

    def _build_prompt(self, inp: QwenRepairInput) -> str:
        primary = inp.primary_normalized or inp.primary_raw
        verifier = inp.verifier_normalized or inp.verifier_raw
        return _USER_PROMPT.format(
            primary=primary,
            verifier=verifier,
            reason=inp.reason,
            glossary=(f"Glossary: {', '.join(inp.known_names)}\n" if inp.known_names else ""),
            context=(f"Context: {inp.nearby_context}\n" if inp.nearby_context else ""),
        )

    def repair(self, repair_input: OCRRepairInput, image: Image.Image) -> OCRRepairResult:
        if not self._loaded or self._model is None or self._processor is None:
            raise RuntimeError("QwenRepairProvider not loaded; call load() first")

        inp = QwenRepairInput.from_repair_input(repair_input, image)
        prompt = self._build_prompt(inp)

        pil_image = image.convert("RGB") if isinstance(image, Image.Image) else Image.fromarray(image).convert("RGB")

        messages = [
            {"role": "system", "content": "You are a JSON-only OCR adjudicator. Output ONLY a JSON object."},
            {"role": "user", "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": prompt},
            ]},
        ]

        inputs = self._processor.apply_chat_template(
            messages, add_generation_prompt=True, return_dict=True, return_tensors="pt", tokenize=True
        )
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda:0") for k, v in inputs.items() if isinstance(v, torch.Tensor)}

        self._reset_peak_vram()
        with torch.no_grad():
            gen = self._model.generate(
                **inputs, do_sample=False, max_new_tokens=self._config.max_new_tokens,
            )

        ilen = inputs["input_ids"].shape[-1]
        raw = self._processor.decode(gen[0, ilen:], skip_special_tokens=True).strip()
        self.metrics.peak_vram_gb = max(self.metrics.peak_vram_gb, self._peak_vram_gb())

        result = self._parse_output(raw)
        return result

    # --- Parser: handles both JSON and natural language output ---
    def _parse_output(self, raw: str) -> OCRRepairResult:
        if not raw or len(raw) < 2:
            return self._unresolved("empty_output")

        # 1. Try JSON: find first { ... last }
        data = self._try_json(raw)
        if data is not None:
            status = str(data.get("status", "")).strip().lower()
            text = data.get("text")
            reason = str(data.get("reason", "")).strip()

            if status == "unresolved":
                return OCRRepairResult(
                    repaired_text=None, changed=False, unresolved=True,
                    metadata={"repair_model": self.metrics.repair_model,
                              "repair_reason": reason or "insufficient_visual_evidence",
                              "raw_output": raw[:500]},
                )

            if status == "resolved" and text is not None:
                repaired = str(text).strip()
                if not repaired or repaired == "null":
                    return self._unresolved("empty_text")
                if self._looks_like_placeholder(repaired):
                    return self._unresolved("placeholder_in_text")
                return OCRRepairResult(
                    repaired_text=repaired, changed=True, unresolved=False,
                    metadata={"repair_model": self.metrics.repair_model,
                              "repair_reason": reason or "visual_evidence",
                              "raw_output": raw[:500]},
                )

        # 2. Fallback: no valid JSON. Check for explicit "unresolved" verdict.
        if re.search(r'\bunresolved\b', raw, re.IGNORECASE):
            return self._unresolved("insufficient_visual_evidence")

        # 3. Fallback: extract text from model analysis output.
        # Model outputs analysis in natural language with quoted text lines.
        extracted = self._extract_text_from_natural_language(raw)
        if extracted:
            return OCRRepairResult(
                repaired_text=extracted, changed=True, unresolved=False,
                metadata={"repair_model": self.metrics.repair_model,
                          "repair_reason": "visual_evidence",
                          "raw_output": raw[:500]},
            )

        return self._unresolved("no_structured_output")

    @staticmethod
    def _try_json(text: str) -> dict[str, Any] | None:
        first = text.find("{")
        last = text.rfind("}")
        if first < 0 or last <= first:
            return None
        for candidate in [text[first : last + 1], text.strip()]:
            try:
                obj = json.loads(candidate)
                if isinstance(obj, dict) and "status" in obj:
                    return obj
            except json.JSONDecodeError:
                pass
        return None

    @staticmethod
    def _looks_like_placeholder(text: str) -> bool:
        markers = ["<exact text>", "<why>", "<text>", "<reason>"]
        return any(m in text for m in markers)

    @staticmethod
    def _extract_text_from_natural_language(text: str) -> str | None:
        """Modelin doğal dil çıktısından metni çıkarır.

        Model genellikle analizinde metni tek tırnak içinde veya
        "Line N: \"...\" " formatında çıkarır.
        """
        # Strateji 1: "Line N: \"TEXT\"" desenleri topla
        line_matches = re.findall(r'[Ll]ine\s*\d*:\s*"([^"]+)"', text)
        if line_matches:
            # Tüm satırları birleştir
            full = " ".join(m.strip() for m in line_matches)
            if len(full) >= 2:
                return full

        # Strateji 2: "Candidate B: \"TEXT\"" veya "Candidate A: \"TEXT\""
        cand_matches = re.findall(r'Candidate\s+[AB]:\s*"([^"]+)"', text)
        if cand_matches:
            # Son candidate match (modelin nihai kararı)
            return cand_matches[-1].strip()

        # Strateji 3: "The text reads:" veya "reads:" sonrasındaki tüm satır
        reads_match = re.search(r'(?:text\s+)?reads:\s*["\']?(.+?)(?:["\']|$)', text, re.IGNORECASE | re.DOTALL)
        if reads_match:
            val = reads_match.group(1).strip().strip('"').strip("'")
            if val and len(val) >= 2:
                return val

        # Strateji 4: "The correct.*: \"TEXT\"" 
        correct_match = re.search(r'(?:correct|right|should be):\s*["\']?([^"\']+)["\']?', text, re.IGNORECASE)
        if correct_match:
            val = correct_match.group(1).strip()
            if val and len(val) >= 2:
                return val

        # Strateji 5: Uzun quoted text blokları
        quoted = re.findall(r'"([^"]{3,})"', text)
        # En uzun quote'ı al (genellikle tam metin)
        if quoted:
            longest = max(quoted, key=len).strip()
            if len(longest) >= 2:
                return longest

        return None

    def _unresolved(self, reason: str) -> OCRRepairResult:
        return OCRRepairResult(
            repaired_text=None, changed=False, unresolved=True,
            metadata={"repair_model": self.metrics.repair_model,
                      "repair_reason": reason, "raw_output": ""},
        )

    def _parse_repair_output(self, raw_output: str) -> OCRRepairResult:
        """Public wrapper for _parse_output (used by tests)."""
        return self._parse_output(raw_output)


def adjudicate_ocr(
    verdict: "OCRVerdict",
    crop_image: Image.Image | None,
    repair_provider: QwenRepairProvider,
) -> OCRAdjudicatedResult:
    """OCR verdict'ı Qwen repair ile cozer.

    - needs_repair=False -> Qwen cagrilmaz, accepted_text = clean_source_text.
    - needs_repair=True -> Qwen repair calistirilir.
    """
    from providers.ocr.agreement import OCRVerdict

    if not verdict.needs_repair:
        logger.debug(
            f"Qwen skip: safe agreement (source={verdict.source})"
        )
        return OCRAdjudicatedResult(
            verdict=verdict,
            clean_source_text=verdict.accepted_text,
            repair_result=None,
            repair_model=None,
            requires_review=False,
        )

    if crop_image is None or not repair_provider.is_loaded:
        logger.warning("needs_repair=True ama Qwen skip (no crop/provider not loaded)")
        return OCRAdjudicatedResult(
            verdict=verdict,
            clean_source_text=None,
            repair_result=None,
            repair_model=None,
            requires_review=True,
        )

    ri = OCRRepairInput(
        primary_raw=verdict.primary_raw,
        primary_normalized=verdict.primary_normalized,
        verifier_raw=verdict.verifier_raw,
        verifier_normalized=verdict.verifier_normalized,
        reason=verdict.reason or "word_difference",
    )

    logger.info(
        f"Qwen repair: {verdict.reason}, "
        f"primary='{verdict.primary_normalized[:40]}', "
        f"verifier='{verdict.verifier_normalized[:40]}'"
    )

    result = repair_provider.repair(ri, crop_image)
    updated = replace(verdict, repaired_text=result.repaired_text)

    if result.unresolved:
        return OCRAdjudicatedResult(
            verdict=updated, clean_source_text=None,
            repair_result=result,
            repair_model=result.metadata.get("repair_model"),
            requires_review=True,
        )
    return OCRAdjudicatedResult(
        verdict=updated,
        clean_source_text=result.repaired_text,
        repair_result=result,
        repair_model=result.metadata.get("repair_model"),
        requires_review=False,
    )
