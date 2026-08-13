"""Qwen3.5-9B Turkish translation provider.

Reuses the 8-bit loading pattern from QwenRepairProvider (same model path,
same quantization, same local_files_only). Translation and OCR repair are
strictly separate: this provider never looks at OCR candidates.

Output is structured JSON so region IDs are preserved and traceable.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from loguru import logger
from typing import Any

from providers.translation.base import (
    TranslationInput,
    TranslationItem,
    TranslationOutput,
    TranslationOutputItem,
    TranslationProvider,
)

DEFAULT_MODEL_PATH = r"C:\AI\Models\Qwen3.5-9B"
MAX_NEW_TOKENS = 1024


def _get_torch():
    """Import torch only when the Transformers backend performs runtime work."""
    import torch

    return torch


@dataclass
class QwenTranslationMetrics:
    model_load_vram_gb: float = 0.0
    peak_vram_gb: float = 0.0
    translation_model: str = ""
    input_token_count: int = 0
    generated_token_count: int = 0
    max_new_tokens: int = MAX_NEW_TOKENS
    generation_seconds: float = 0.0
    tokens_per_sec: float = 0.0
    generation_call_count: int = 0
    json_retry_happened: bool = False


from providers.translation.qwen_prompt import (
    _SYSTEM_PROMPT,
    build_qwen_translation_prompt,
)


# Characters / markers that must never reach the translation result.
_THINK_MARKER = re.compile(r"<thinking\b", re.IGNORECASE)


def _vram_gb() -> float:
    torch = _get_torch()
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / (1024 ** 3)


def _peak_vram_gb() -> float:
    torch = _get_torch()
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


def _reset_peak_vram() -> None:
    torch = _get_torch()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _strip_thinking(text: str) -> str:
    """Remove Qwen3 reasoning/thinking blocks so they never leak into JSON output."""
    if not _THINK_MARKER.search(text):
        return text
    cleaned = re.sub(r"<thinking\b.*?</thinking\s*>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<reasoning\b.*?", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


class QwenTranslationProvider(TranslationProvider):
    """Qwen3.5-9B translation provider (8-bit quantization, same as OCR repair)."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH) -> None:
        self._model_path = model_path
        self._loaded = False
        self._model = None
        self._processor = None
        self._device = "cpu"
        self.metrics = QwenTranslationMetrics()
        self._item_term_maps: dict[int, dict[str, str]] = {}

    def _build_prompt(self, inp: TranslationInput) -> str:
        prompt_str, item_term_maps = build_qwen_translation_prompt(inp)
        self._item_term_maps = item_term_maps
        return prompt_str


    @property
    def name(self) -> str:
        return "Qwen3.5-9B-Translation"

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def device(self) -> str:
        return self._device

    def load(self) -> None:
        if self._loaded:
            return

        torch = _get_torch()
        torch.cuda.empty_cache()
        from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

        self._processor = AutoProcessor.from_pretrained(
            self._model_path, local_files_only=True
        )
        bnb = BitsAndBytesConfig(load_in_8bit=True)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self._model_path,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_memory={0: "12GiB"},
            quantization_config=bnb,
        )
        self._model.eval()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            self._model = self._model.to("cuda:0")
        self._loaded = True
        self.metrics.model_load_vram_gb = _vram_gb()
        self.metrics.translation_model = "Qwen3.5-9B-8bit"
        logger.info(
            f"Qwen translation model loaded, VRAM: {self.metrics.model_load_vram_gb:.2f} GB"
        )

    def unload(self) -> None:
        self._model = None
        self._processor = None
        self._loaded = False
        self._device = "cpu"
        torch = _get_torch()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Qwen translation model unloaded")


    def translate(self, inp: TranslationInput) -> TranslationOutput:
        if not self._loaded or self._model is None or self._processor is None:
            raise RuntimeError("QwenTranslationProvider not loaded; call load() first")

        from core.translation.batcher import TranslationBatcher
        batcher = TranslationBatcher()
        sub_inputs = batcher.create_batches(
            inp,
            tokenizer=self._processor.tokenizer if hasattr(self._processor, "tokenizer") else None,
        )

        if len(sub_inputs) == 1:
            return self._translate_single_batch(sub_inputs[0])

        logger.info(f"Executing translation across {len(sub_inputs)} sub-batches")
        sub_outputs: list[TranslationOutput] = []
        for sub_inp in sub_inputs:
            sub_outputs.append(self._translate_single_batch(sub_inp))

        return batcher.merge_outputs(inp, sub_outputs)

    def _translate_single_batch(
        self, inp: TranslationInput, retry_count: int = 0
    ) -> TranslationOutput:
        import time
        torch = _get_torch()

        prompt = self._build_prompt(inp)
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            tokenize=True,
            enable_thinking=False,
        )
        if torch.cuda.is_available():
            inputs = {
                k: v.to("cuda:0") for k, v in inputs.items() if isinstance(v, torch.Tensor)
            }

        _reset_peak_vram()
        t0 = time.perf_counter()
        with torch.no_grad():
            gen = self._model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=MAX_NEW_TOKENS,
            )
        gen_time = time.perf_counter() - t0

        ilen = inputs["input_ids"].shape[-1]
        gen_len = gen.shape[-1] - ilen
        raw_output = self._processor.decode(
            gen[0, ilen:], skip_special_tokens=True
        ).strip()
        raw_output = _strip_thinking(raw_output)

        self.metrics.peak_vram_gb = max(self.metrics.peak_vram_gb, _peak_vram_gb())
        self.metrics.input_token_count += ilen
        self.metrics.generated_token_count += gen_len
        self.metrics.max_new_tokens = MAX_NEW_TOKENS
        self.metrics.generation_seconds += gen_time
        self.metrics.tokens_per_sec = (
            self.metrics.generated_token_count / self.metrics.generation_seconds
            if self.metrics.generation_seconds > 0
            else 0.0
        )
        self.metrics.generation_call_count += 1

        output = self._parse_output(inp, raw_output)

        # Truncation recovery: if generated tokens hit max limit and some IDs are missing
        missing_ids = [r.region_id for r in output.results if r.translation is None]
        if missing_ids and gen_len >= MAX_NEW_TOKENS and retry_count < 2 and len(inp.items) > 1:
            logger.warning(
                f"Batch truncated at {gen_len} tokens with missing IDs {missing_ids}. Splitting batch to recover missing items."
            )
            self.metrics.json_retry_happened = True
            half = len(inp.items) // 2
            from dataclasses import replace
            sub_a = replace(inp, items=inp.items[:half])
            sub_b = replace(inp, items=inp.items[half:], context_items=inp.items[:half][-2:])
            out_a = self._translate_single_batch(sub_a, retry_count + 1)
            out_b = self._translate_single_batch(sub_b, retry_count + 1)
            from core.translation.batcher import TranslationBatcher
            return TranslationBatcher().merge_outputs(inp, [out_a, out_b])

        logger.info(
            f"Qwen batch complete: {ilen} in, {gen_len} gen, {gen_time:.2f}s ({gen_len / gen_time:.2f} tok/s if gen_time > 0 else 0)"
        )
        return output

    def _parse_output(self, inp: TranslationInput, raw: str) -> TranslationOutput:
        json_obj = _try_extract_json(raw)
        results: list[TranslationOutputItem] = []

        if json_obj is None:
            # Fallback: try to extract translations from natural language output
            translations = _extract_translations_from_natural_language(raw, inp.items)
            if translations:
                logger.warning("Translation JSON parse failed; using NL fallback extraction")
                json_obj = {"translations": translations}
            else:
                logger.warning("Translation output JSON parse failed")
                for item in inp.items:
                    results.append(
                        TranslationOutputItem(
                            region_id=item.region_id,
                            source=item.source,
                            translation=None,
                            raw_model_response=raw[:500],
                            validation_warnings=["json_parse_failure"],
                            requires_review=True,
                        )
                    )
                return TranslationOutput(
                    inputs=inp,
                    results=results,
                    raw_response=raw[:2000],
                    repair_model=self.metrics.translation_model,
                )

        translations: list[dict[str, Any]] = json_obj.get("translations", [])
        warnings_by_id: dict[int, list[str]] = _validate_output(
            inp.items, translations, json_obj, profile=inp.profile
        )

        for item in inp.items:
            match = next((t for t in translations if t.get("id") == item.region_id), None)
            if match is None:
                results.append(
                    TranslationOutputItem(
                        region_id=item.region_id,
                        source=item.source,
                        translation=None,
                        raw_model_response=raw[:500],
                        validation_warnings=warnings_by_id.get(item.region_id, ["missing_id"]),
                        requires_review=True,
                    )
                )
            else:
                translation = str(match.get("translation", "")).strip()
                warns = warnings_by_id.get(item.region_id, [])
                resolved = translation if translation and translation.lower() != "null" else None
                term_usages = match.get("term_usages", [])
                if not isinstance(term_usages, list):
                    term_usages = []
                fidelity_flags = match.get("fidelity_flags", [])
                if not isinstance(fidelity_flags, list):
                    fidelity_flags = []
                item_term_map = self._item_term_maps.get(item.region_id, {})
                results.append(
                    TranslationOutputItem(
                        region_id=item.region_id,
                        source=item.source,
                        translation=resolved,
                        raw_model_response=raw[:500],
                        validation_warnings=warns,
                        requires_review=bool(warns),
                        term_usages=term_usages,
                        fidelity_flags=fidelity_flags,
                        term_id_map=item_term_map,
                    )
                )

        return TranslationOutput(
            inputs=inp,
            results=results,
            raw_response=raw[:2000],
            repair_model=self.metrics.translation_model,
        )


def _try_extract_json(text: str) -> dict[str, Any] | None:
    # 1. Fenced code block
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        try:
            obj = json.loads(fence.group(1))
            if isinstance(obj, dict) and "translations" in obj:
                return obj
        except json.JSONDecodeError:
            pass

    # 2. First { to last }
    first = text.find("{")
    last = text.rfind("}")
    if first >= 0 and last > first:
        candidate = text[first : last + 1]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict) and "translations" in obj:
                return obj
        except json.JSONDecodeError:
            pass

    # 3. Try whole stripped text
    try:
        obj = json.loads(text.strip())
        if isinstance(obj, dict) and "translations" in obj:
            return obj
    except (json.JSONDecodeError, ValueError):
        pass

    return None


def _contains_word(name: str, text: str) -> bool:
    """Check if name exists in text as whole word/token (case-insensitive)."""
    if not name or not text:
        return False
    escaped_name = r"\s+".join(re.escape(part) for part in name.strip().split())
    pattern = rf"\b{escaped_name}\b"
    return bool(re.search(pattern, text, re.IGNORECASE))


def _validate_output(
    items: list[TranslationItem],
    translations: list[dict[str, Any]],
    raw_obj: dict[str, Any],
    profile: Any | None = None,
) -> dict[int, list[str]]:
    """Lightweight validation. Returns warnings keyed by region_id."""
    warnings: dict[int, list[str]] = {}

    # Check for duplicate IDs in model output
    ids_in_output: list[int] = []
    for t in translations:
        try:
            ids_in_output.append(int(t.get("id")))
        except (ValueError, TypeError):
            pass

    input_ids = {item.region_id for item in items}

    profile_names = profile.get_known_names_list() if profile and hasattr(profile, "get_known_names_list") else []

    for item in items:
        warns: list[str] = []

        # Missing from output
        if item.region_id not in ids_in_output:
            warns.append("missing_id")

        # Combine known_names from item and active profile
        combined_names = list(set(item.known_names + profile_names))

        # Check name preservation (token/word boundary aware, only for names present in item source)
        for name in combined_names:
            match = next((t for t in translations if t.get("id") == item.region_id), None)
            if match and name:
                # Check if name is present in source as a whole word/token
                if _contains_word(name, item.source):
                    translation = str(match.get("translation", ""))
                    if not _contains_word(name, translation):
                        if re.search(r"[\u3040-\u30ff\u4e00-\u9fff]", translation):
                            warns.append("cjk_hallucination")
                        warns.append("name_modified")

        # Per-item source-based checks
        out_item = next((t for t in translations if t.get("id") == item.region_id), None)
        if out_item:
            src = item.source
            translated = str(out_item.get("translation", "")).strip()

            if not translated or translated.lower() == "null":
                warns.append("empty_output")
            else:
                # Still entirely English or invalid characters?
                eng_words = re.findall(r"[A-Za-z]+", translated)
                non_eng = re.sub(r"[A-Za-z0-9\s.,!?:;'\"()\-–—…ğçşıöüĞÇŞİÖÜâîûÂÎÛ]", "", translated)
                if non_eng.strip():
                    warns.append("contains_cjk_or_unk_chars")
                elif len(eng_words) >= 3 and not re.search(r"[ğçşıöüĞÇŞİÖÜâîûÂÎÛ]", translated, re.IGNORECASE):
                    if len(re.sub(r"[\W]", "", translated)) > 10 and len(eng_words) / max(len(translated.split()), 1) > 0.8:
                        warns.append("still_english")

                # Abnormal length ratio (source much shorter or much longer)
                src_len = len(re.sub(r"\s+", "", src))
                out_len = len(re.sub(r"\s+", "", translated))
                if src_len > 0 and out_len / src_len > 5.0:
                    warns.append("suspicious_length_ratio")

                # Repetition
                words = translated.lower().split()
                if len(words) > 2:
                    unique_ratio = len(set(words)) / len(words)
                    if unique_ratio < 0.3:
                        warns.append("excessive_repetition")

        if warns:
            warnings[item.region_id] = warns

    # Duplicate ID warnings
    seen: dict[int, int] = {}
    for tid in ids_in_output:
        if tid in seen:
            warnings.setdefault(tid, []).append("duplicate_id")
        seen[tid] = seen.get(tid, 0) + 1

    # Extra IDs not in input
    for tid in ids_in_output:
        if tid not in input_ids:
            warnings.setdefault(tid, []).append("extra_id_not_in_input")

    return warnings


def _extract_translations_from_natural_language(
    raw: str, items: list[TranslationItem]
) -> list[dict[str, Any]]:
    """Fallback: extract translations from natural language model output.

    When the model doesn't output JSON but still produces translations,
    try to match them to the source items by order.
    """
    translations: list[dict[str, Any]] = []

    # Strategy 1: Look for "id=N" or "[N]" patterns followed by translation text
    id_pattern = re.compile(r'(?:id[=:\s]+|region_id[=:\s]+|\[(\d+)\])(\d+)')
    matches = id_pattern.findall(raw)

    # Strategy 2: Look for lines that look like translations (non-English, no English words)
    lines = [l.strip() for l in raw.split('\n') if l.strip()]
    translation_candidates = []
    for line in lines:
        # Skip headers, analysis text, English lines
        if any(skip in line.lower() for skip in [
            'thinking', 'analyze', 'bubble', 'source', 'output', 'translation',
            'constraint', 'id=', 'role:', 'task:', 'preamble', 'json',
            'mark', 'format', 'key', 'array', 'object', 'input:',
            'no ', 'do not', 'preserve', 'keep', 'natural', 'word',
        ]):
            continue
        if not line:
            continue
        # Check if line has Turkish characters (likely a translation)
        if re.search(r'[ğçşıöüĞÇŞİÖÜ]', line) or re.search(r'[^a-z]', line.lower()) and any(
            c in line for c in 'ğçşıöüĞÇŞİÖÜ'
        ):
            translation_candidates.append(line)

    # Strategy 3: Look for quoted text that's Turkish
    quoted = re.findall(r'"([^"]+)"', raw)
    for q in quoted:
        if re.search(r'[ğçşıöüĞÇŞİÖÜ]', q) and len(q) > 3:
            translation_candidates.append(q)

    translation_candidates = translation_candidates[:len(items)]

    if translation_candidates:
        for i, (item, trans) in enumerate(zip(items, translation_candidates)):
            translations.append({
                "id": item.region_id,
                "source": item.source,
                "translation": trans.strip(),
            })

    return translations
