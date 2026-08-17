"""Qwen3.5-9B GGUF single-item translation provider (V2 Shootout Provider).

Backed by llama.cpp /v1/chat/completions with native Qwen embedded chat template,
--reasoning off, temperature 0.0, and application-level terminology sentinels.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from typing import Any

from loguru import logger

from core.translation.profile_discovery import (
    contains_candidate_phrase,
    get_relevant_terms_for_item,
)
from core.translation.protection import (
    contains_unrestored_protected_term,
    detect_named_terms_in_items,
    has_untranslated_source_prose,
    is_term_only_source,
    protect_source_text,
    restore_protected_translation,
    validate_protected_terms,
)
from core.translation.source_normalization import normalize_translation_source_case
from core.translation.system_text import is_system_ui_line, translate_system_ui_line
from providers.translation.base import (
    TranslationInput,
    TranslationItem,
    TranslationOutput,
    TranslationOutputItem,
    TranslationProvider,
)
from providers.translation.translategemma_gguf_translation import (
    is_explanation_like_output,
)


DEFAULT_QWEN_MODEL_PATH = r"C:\AI\Models\Qwen3.5-9B-GGUF\Qwen3.5-9B-Q5_K_M.gguf"
DEFAULT_LLAMA_EXE_PATH = r"C:\AI\llama-cpp-cuda\llama.exe"
DEFAULT_QWEN_SERVER_URL = "http://127.0.0.1:8083"

QWEN_TRANSLATOR_SYSTEM_PROMPT = (
    "You are a precise English to Turkish translator.\n"
    "Translate the supplied English text into natural Turkish while preserving its exact meaning.\n"
    "Do not add, omit, explain, summarize, or invent information.\n"
    "Preserve proper names and named abilities.\n"
    "Output only the Turkish translation."
)


_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "am", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "just", "because", "but", "and", "or", "if", "while", "although",
    "i", "me", "my", "we", "our", "you", "your", "he", "him", "his",
    "she", "her", "it", "its", "they", "them", "their", "what", "which",
    "who", "whom", "this", "that", "these", "those",
})


_REPEATED_PATTERN_RE = re.compile(r"(.{2,})\1+", re.IGNORECASE)
_REPEATED_LETTERS_RE = re.compile(r"([a-zA-Z])\1{2,}", re.IGNORECASE)
_REPEATED_DIGITS_RE = re.compile(r"(\d)\1+", re.IGNORECASE)
_GAME_STAT_ENTITY_RE = re.compile(r"\b(?:Lv\.|Level|HP|MP|EXP|STR|AGI|INT)\s*\d+", re.IGNORECASE)


def _is_likely_sfx_output(text: str) -> bool:
    """Heuristic: detect onomatopoeia / SFX-like text that may legitimately echo."""
    words = text.split()
    if len(words) > 6 or len(words) == 0:
        return False

    for word in words:
        core = word.strip(".,!?~'-").lower()
        if not core or core in _STOP_WORDS:
            return False

    cores = [word.strip(".,!?~'-") for word in words if word.strip(".,!?~'-")]
    if not cores:
        return False

    if any(len(core) > 8 for core in cores):
        return False

    has_repeated_pattern = any(
        _REPEATED_PATTERN_RE.search(core) or _REPEATED_LETTERS_RE.search(core) or _REPEATED_DIGITS_RE.search(core)
        for core in cores
    )
    if has_repeated_pattern:
        return True

    all_short = all(len(core) <= 5 for core in cores)
    if not all_short:
        return False

    unique_cores = set(c.lower() for c in cores)
    has_repeated_tokens = len(unique_cores) < len(cores)

    return has_repeated_tokens


def _is_legitimate_entity_stat_echo(text: str) -> bool:
    """Heuristic: detect game entity titles / monster names with level/stat tags."""
    return bool(_GAME_STAT_ENTITY_RE.search(text))


@dataclass
class QwenGGUFMetricsV2:
    model_load_seconds: float = 0.0
    peak_vram_gb: float | None = None
    translation_model: str = "Qwen3.5-9B-Q5_K_M-GGUF-Translator"
    input_token_count: int = 0
    generated_token_count: int = 0
    generation_seconds: float = 0.0
    tokens_per_sec: float = 0.0
    generation_call_count: int = 0
    system_ui_bypass_count: int = 0
    term_only_bypass_count: int = 0
    retries: int = 0
    reasoning_contamination_count: int = 0
    micro_batch_requests: int = 0


@dataclass(frozen=True)
class _PreparedTranslationItemV2:
    item: TranslationItem
    normalized_source: str
    prepared_text: str
    placeholder_map: dict[str, Any]
    detected_named_terms: tuple[str, ...] = ()
    system_ui_translation: str | None = None
    term_only_translation: str | None = None


def _clean_qwen_output(raw_text: str, prepared_text: str) -> str:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned).strip()
    if (cleaned.startswith('"') and cleaned.endswith('"')) or (
        cleaned.startswith("“") and cleaned.endswith("”")
    ):
        if not (prepared_text.startswith('"') or prepared_text.startswith("“")):
            cleaned = cleaned[1:-1].strip()
    return cleaned


class QwenGGUFTranslationProviderV2(TranslationProvider):
    """Direct Qwen3.5-9B GGUF text translator for Shootout V1."""

    def __init__(
        self,
        model_path: str = DEFAULT_QWEN_MODEL_PATH,
        executable_path: str = DEFAULT_LLAMA_EXE_PATH,
        server_url: str = DEFAULT_QWEN_SERVER_URL,
        max_context_length: int = 2048,
        gpu_layers: int = 99,
        system_prompt: str = QWEN_TRANSLATOR_SYSTEM_PROMPT,
    ) -> None:
        self.model_path = model_path
        self.executable_path = executable_path
        self.server_url = server_url.rstrip("/")
        self.max_context_length = max_context_length
        self.gpu_layers = gpu_layers
        self.system_prompt = system_prompt

        self._process: subprocess.Popen | None = None
        self._owned_process: bool = False
        self._loaded: bool = False
        self.metrics = QwenGGUFMetricsV2()

    @property
    def name(self) -> str:
        return "Qwen3.5-9B-GGUF-Translator-V2"

    @property
    def is_loaded(self) -> bool:
        return self._loaded and self._check_health()

    def _check_health(self) -> bool:
        endpoint = f"{self.server_url}/health"
        try:
            req = urllib.request.Request(endpoint, method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        return False

    def load(self) -> None:
        if self.is_loaded:
            logger.info(f"QwenGGUFTranslationProviderV2: Server at {self.server_url} is ready.")
            return

        t0 = time.perf_counter()
        if self._check_health():
            self._loaded = True
            self._owned_process = False
            self.metrics.model_load_seconds = round(time.perf_counter() - t0, 2)
            logger.info(f"Connected to pre-existing Qwen llama-server at {self.server_url}")
            return

        if not os.path.exists(self.model_path):
            raise RuntimeError(f"GGUF model file not found: {self.model_path}")
        if not os.path.exists(self.executable_path):
            raise RuntimeError(f"llama executable not found: {self.executable_path}")

        port = "8083"
        host = "127.0.0.1"
        url_part = self.server_url.split("//")[-1]
        if ":" in url_part:
            host, port = url_part.split(":", 1)

        cmd = [self.executable_path]
        if self.executable_path.lower().endswith("llama.exe"):
            cmd.append("serve")

        cmd.extend([
            "-m", self.model_path,
            "-ngl", str(self.gpu_layers),
            "--host", host,
            "--port", port,
            "--alias", "qwen3.5-9b-translator",
            "--reasoning", "off",
            "-c", str(self.max_context_length),
        ])

        logger.info(f"Starting managed Qwen translator llama-server process: {' '.join(cmd)}")
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        self._owned_process = True

        ready = False
        start_t = time.perf_counter()
        while time.perf_counter() - start_t < 45.0:
            if self._process.poll() is not None:
                raise RuntimeError(
                    f"Qwen llama-server exited prematurely with code {self._process.returncode}"
                )
            if self._check_health():
                ready = True
                break
            time.sleep(0.5)

        if not ready:
            self.unload()
            raise RuntimeError("Timeout waiting for Qwen llama-server to initialize (45s)")

        self._loaded = True
        self.metrics.model_load_seconds = round(time.perf_counter() - t0, 2)
        logger.info(
            f"QwenGGUFTranslationProviderV2 loaded successfully, load time: {self.metrics.model_load_seconds}s"
        )

    def unload(self) -> None:
        if self._owned_process and self._process is not None:
            logger.info("Terminating managed Qwen llama-server process...")
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass
            self._process = None
            self._owned_process = False

        self._loaded = False
        logger.info("QwenGGUFTranslationProviderV2 unloaded.")

    def _query_chat_completion(self, prepared_text: str) -> tuple[str, int, int, float]:
        endpoint = f"{self.server_url}/v1/chat/completions"
        payload = {
            "model": "qwen3.5-9b-translator",
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prepared_text},
            ],
            "temperature": 0.0,
            "max_tokens": 256,
            "stream": False,
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=60.0) as resp:
            res_json = json.loads(resp.read().decode("utf-8"))
        gen_time = time.perf_counter() - t0

        choices = res_json.get("choices", [])
        if not choices:
            raise RuntimeError("Qwen llama-server returned empty choices")

        msg = choices[0].get("message", {})
        reasoning = msg.get("reasoning_content")
        raw_text = (msg.get("content") or "").strip()

        if reasoning or "<think" in raw_text.casefold():
            self.metrics.reasoning_contamination_count += 1
            raise RuntimeError("Qwen reasoning content contaminated response")

        usage = res_json.get("usage", {})
        in_toks = usage.get("prompt_tokens", 0)
        gen_toks = usage.get("completion_tokens", 0)

        return raw_text, in_toks, gen_toks, gen_time

    def _request_translation(self, prepared_text: str, label: str) -> tuple[str, str, bool]:
        raw_text = ""
        for attempt in range(2):
            if attempt:
                self.metrics.retries += 1
            self.metrics.generation_call_count += 1
            try:
                raw_text, in_toks, gen_toks, gen_sec = self._query_chat_completion(prepared_text)
            except Exception as exc:
                if attempt == 0:
                    logger.warning(f"Translation request failed for {label}: {exc}. Retrying...")
                    continue
                return "", "", True

            cleaned = _clean_qwen_output(raw_text, prepared_text)
            self.metrics.input_token_count += in_toks
            self.metrics.generated_token_count += gen_toks
            self.metrics.generation_seconds += gen_sec
            if self.metrics.generation_seconds > 0:
                self.metrics.tokens_per_sec = (
                    self.metrics.generated_token_count / self.metrics.generation_seconds
                )
            if cleaned:
                return raw_text, cleaned, False
            if attempt == 0:
                logger.warning(f"Empty translation for {label}. Retrying once...")

        return raw_text, "", False

    def _prepare_item(self, item: TranslationItem, inp: TranslationInput) -> _PreparedTranslationItemV2:
        original_source = item.source.strip()
        explicit_glossary: dict[str, str] = {}
        for entry in inp.glossary or []:
            if "->" in entry:
                source_term, target_term = entry.split("->", 1)
                explicit_glossary[source_term.strip()] = target_term.strip()
        source_text = normalize_translation_source_case(
            original_source,
            profile=inp.profile,
            approved_terms=explicit_glossary,
        )
        approved_terms, _ = get_relevant_terms_for_item(
            source_text,
            inp.profile,
            inp.candidate_store,
        )
        normalized_item = replace(item, source=source_text)
        detected_named_terms = detect_named_terms_in_items(
            [normalized_item], candidate_store=inp.candidate_store
        )

        proper_name_terms: set[str] = set()
        if inp.profile:
            proper_name_terms = {
                key.strip().upper()
                for key in inp.profile.known_names
                if contains_candidate_phrase(key, source_text)
            }

        if explicit_glossary:
            for source_term, target_term in explicit_glossary.items():
                if contains_candidate_phrase(source_term, source_text):
                    approved_terms[source_term.upper()] = target_term

        if is_system_ui_line(source_text):
            translated_ui = translate_system_ui_line(source_text, approved_terms=approved_terms)
            if translated_ui:
                return _PreparedTranslationItemV2(
                    item=item,
                    normalized_source=source_text,
                    prepared_text=source_text,
                    placeholder_map={},
                    detected_named_terms=tuple(sorted(detected_named_terms)),
                    system_ui_translation=translated_ui,
                )

        is_bypass, term_trans = is_term_only_source(
            source_text,
            approved_terms,
            detected_named_terms,
        )
        if is_bypass and term_trans:
            return _PreparedTranslationItemV2(
                item=item,
                normalized_source=source_text,
                prepared_text=source_text,
                placeholder_map={},
                detected_named_terms=tuple(sorted(detected_named_terms)),
                term_only_translation=term_trans,
            )

        prepared_text, placeholder_map = protect_source_text(
            source_text,
            approved_terms=approved_terms,
            detected_named_terms=detected_named_terms,
            proper_name_terms=proper_name_terms,
        )
        return _PreparedTranslationItemV2(
            item=item,
            normalized_source=source_text,
            prepared_text=prepared_text,
            placeholder_map=placeholder_map,
            detected_named_terms=tuple(sorted(detected_named_terms)),
        )

    def _finalize_prepared_item(
        self,
        prepared: _PreparedTranslationItemV2,
        cleaned: str,
        raw_text: str,
    ) -> TranslationOutputItem:
        item = prepared.item

        if cleaned == prepared.prepared_text and len(prepared.prepared_text.split()) > 2:
            if _is_likely_sfx_output(prepared.prepared_text) or _is_legitimate_entity_stat_echo(prepared.prepared_text):
                logger.info(
                    "Item %s output matches prepared text but is likely SFX or game entity stat title; accepting echo.",
                    item.region_id,
                )
            else:
                logger.warning(f"Item {item.region_id} output matches source text. Flagging wrapper/copy error.")
                return TranslationOutputItem(
                    region_id=item.region_id,
                    source=item.source,
                    translation=None,
                    raw_model_response=raw_text[:500],
                    validation_warnings=["source_translation_wrapper"],
                    requires_review=True,
                )

        if is_explanation_like_output(cleaned, item.source.strip()):
            logger.warning(f"Item {item.region_id} output identified as chatbot/explanation. Flagging for review.")
            return TranslationOutputItem(
                region_id=item.region_id,
                source=item.source,
                translation=None,
                raw_model_response=cleaned[:500],
                validation_warnings=["chatbot_or_explanation_output"],
                requires_review=True,
            )

        restored = restore_protected_translation(cleaned, prepared.placeholder_map)
        if contains_unrestored_protected_term(restored):
            return TranslationOutputItem(
                region_id=item.region_id,
                source=item.source,
                translation=None,
                raw_model_response=raw_text[:500],
                validation_warnings=["unrestored_protected_term"],
                requires_review=True,
            )

        warnings = validate_protected_terms(restored, prepared.placeholder_map)
        if not _is_likely_sfx_output(prepared.prepared_text) and not _is_legitimate_entity_stat_echo(prepared.prepared_text):
            if has_untranslated_source_prose(
                prepared.prepared_text,
                restored,
                prepared.placeholder_map,
            ) and "untranslated_source_prose" not in warnings:
                warnings.append("untranslated_source_prose")

        return TranslationOutputItem(
            region_id=item.region_id,
            source=item.source,
            translation=restored,
            raw_model_response=raw_text[:500],
            validation_warnings=warnings,
            requires_review=bool(warnings),
        )

    def translate(self, inp: TranslationInput) -> TranslationOutput:
        if not self.is_loaded:
            raise RuntimeError("QwenGGUFTranslationProviderV2 not loaded; call load() first")

        results: list[TranslationOutputItem] = []
        raw_responses: list[str] = []

        for item in inp.items:
            prepared = self._prepare_item(item, inp)
            if prepared.system_ui_translation is not None:
                self.metrics.system_ui_bypass_count += 1
                results.append(
                    TranslationOutputItem(
                        region_id=item.region_id,
                        source=item.source,
                        translation=prepared.system_ui_translation,
                        raw_model_response="[SYSTEM_UI_BYPASS]",
                        validation_warnings=[],
                        requires_review=False,
                    )
                )
                raw_responses.append("[SYSTEM_UI_BYPASS]")
                continue

            if prepared.term_only_translation is not None:
                self.metrics.term_only_bypass_count += 1
                results.append(
                    TranslationOutputItem(
                        region_id=item.region_id,
                        source=item.source,
                        translation=prepared.term_only_translation,
                        raw_model_response="[TERM_ONLY_BYPASS]",
                        validation_warnings=[],
                        requires_review=False,
                    )
                )
                raw_responses.append("[TERM_ONLY_BYPASS]")
                continue

            raw_text, cleaned, error_occurred = self._request_translation(
                prepared.prepared_text, label=f"item {item.region_id}"
            )
            raw_responses.append(raw_text)

            if error_occurred or not cleaned:
                warning = "translation_server_error" if error_occurred else "empty_translation"
                results.append(
                    TranslationOutputItem(
                        region_id=item.region_id,
                        source=item.source,
                        translation=None,
                        raw_model_response=raw_text[:500] if raw_text else "[Server Error]",
                        validation_warnings=[warning],
                        requires_review=True,
                    )
                )
            else:
                out_item = self._finalize_prepared_item(prepared, cleaned, raw_text)
                results.append(out_item)

        return TranslationOutput(
            inputs=inp,
            results=results,
            raw_response="\n---\n".join(raw_responses)[:2000],
            repair_model=self.name,
        )
