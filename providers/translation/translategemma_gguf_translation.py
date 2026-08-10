"""TranslateGemma 12B GGUF llama-server translation provider.

Uses TranslateGemma 12B GGUF running via llama-server (llama.exe) raw /completion endpoint.

Key design principles:
- Raw prompt rendering via core/translation/translategemma_template.py:
  <bos><start_of_turn>user\nTranslate from English to Turkish:\n{prepared_text}<end_of_turn>\n<start_of_turn>model\n
- Raw completion endpoint (/completion) to prevent chat template interpretation distortion.
- Application-level terminology protection using opaque sentinels (__WTTERM0001__).
- Term-only bypass for isolated protected named terms (0 model calls).
- System UI lexicon handling respecting approved glossary overrides.
- Chatbot / explanation output guard (detects assistant chat phrases & dictionary outputs).
- Exact server identity verification via GET /props model_path check.
- Per-item retries and error isolation (one failing item does not abort chapter).
- Truthful metrics (generation calls count actual HTTP requests; bypasses tracked separately).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from loguru import logger

from core.translation.protection import (
    OPAQUE_SENTINEL_PATTERN,
    ProtectedTermMeta,
    contains_unrestored_protected_term,
    detect_named_terms_in_items,
    has_untranslated_source_prose,
    is_term_only_source,
    protect_source_text,
    restore_protected_translation,
    validate_protected_terms,
)

from core.translation.system_text import is_system_ui_line, translate_system_ui_line
from core.translation.translategemma_template import render_translategemma_prompt
from providers.translation.base import (
    TranslationInput,
    TranslationItem,
    TranslationOutput,
    TranslationOutputItem,
    TranslationProvider,
)

DEFAULT_GEMMA_MODEL_PATH = r"C:\AI\Models\translategemma-12b-it-q5_k_m.gguf"
DEFAULT_LLAMA_EXE_PATH = r"C:\AI\llama-cpp-cuda\llama.exe"
DEFAULT_TRANSLATEGEMMA_SERVER_URL = "http://127.0.0.1:8081"
DEFAULT_MICRO_BATCH_SIZE = 4
MIN_MICRO_BATCH_SIZE = 3

_SEGMENT_MARKER_PATTERN = re.compile(r"__WTSEG\d{4}__")
_SEGMENT_MARKER_LIKE_PATTERN = re.compile(r"__WTSEG[A-Z0-9_]*", re.IGNORECASE)
_SOURCE_TRANSLATION_WRAPPER_PATTERN = re.compile(
    r"(?:\*{0,2}|_{0,2})\s*"
    r"(?:türkçe\s+çeviri(?:si)?|çeviri(?:si)?|turkish\s+translation|translation)"
    r"\s*:\s*(?:\*{0,2}|_{0,2})",
    re.IGNORECASE,
)


@dataclass
class TranslateGemmaGGUFMetrics:
    model_load_seconds: float = 0.0
    peak_vram_gb: float | None = None
    translation_model: str = "TranslateGemma-12B-IT-Q5_K_M-GGUF"
    input_token_count: int = 0
    generated_token_count: int = 0
    generation_seconds: float = 0.0
    tokens_per_sec: float = 0.0
    generation_call_count: int = 0
    micro_batch_requests: int = 0
    micro_batch_successes: int = 0
    micro_batch_fallbacks: int = 0
    single_item_fallback_calls: int = 0
    system_ui_bypass_count: int = 0
    term_only_bypass_count: int = 0
    retries: int = 0
    cuda_active: bool | None = None
    gpu_offload: str = "configured: ngl=99"


@dataclass
class _PreparedTranslationItem:
    item: TranslationItem
    prepared_text: str
    placeholder_map: dict[str, ProtectedTermMeta]


def contains_segment_marker(text: str) -> bool:
    """Return whether any opaque segment marker survived into item text."""
    return bool(_SEGMENT_MARKER_LIKE_PATTERN.search(text or ""))


def split_segmented_translation(
    translated_block: str,
    expected_markers: list[str],
) -> list[str] | None:
    """Strictly validate and split one translated micro-batch.

    Marker identity, count, order, and non-empty segment bodies must all match.
    No reconstruction is attempted when any structural property is invalid.
    """
    marker_like = _SEGMENT_MARKER_LIKE_PATTERN.findall(translated_block)
    if marker_like != expected_markers:
        return None

    matches = list(_SEGMENT_MARKER_PATTERN.finditer(translated_block))
    if [match.group(0) for match in matches] != expected_markers:
        return None

    segments: list[str] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(translated_block)
        segment = translated_block[match.end():end].strip()
        if not segment or contains_segment_marker(segment):
            return None
        segments.append(segment)
    return segments


def is_source_translation_wrapper_output(output_text: str, prepared_source: str) -> bool:
    """Detect source prose repeated before a translation-wrapper label."""
    label_match = _SOURCE_TRANSLATION_WRAPPER_PATTERN.search(output_text)
    if not label_match:
        return False

    # Preserve sentinel positions as an abstract token. This lets a short line
    # such as "Activate <protected ability>" count as substantial source prose
    # without treating the protected English name itself as leakage.
    source_normalized = OPAQUE_SENTINEL_PATTERN.sub(" protectedterm ", prepared_source)
    prefix_normalized = OPAQUE_SENTINEL_PATTERN.sub(
        " protectedterm ",
        output_text[:label_match.start()],
    )
    source_tokens = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", source_normalized.casefold())
    prefix_tokens = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", prefix_normalized.casefold())
    if not source_tokens or not prefix_tokens:
        return False

    matching_tokens = sum(
        block.size for block in SequenceMatcher(None, source_tokens, prefix_tokens).get_matching_blocks()
    )
    minimum_match = 1 if len(source_tokens) == 1 else 2
    return matching_tokens >= minimum_match and matching_tokens / len(source_tokens) >= 0.6


def _clean_translategemma_output(raw: str) -> str:
    """Strip chat/control delimiter artifacts and leading translation labels."""
    text = raw.strip()

    # Known Gemma/llama-server chat delimiter artifacts & alternative option splits
    delimiters = [
        "<|file_separator|>",
        "<|im_start|>",
        "<|im_end|>",
        "<|end|>",
        "<start_of_turn>",
        "<end_of_turn>",
        "</s>",
        "<bos>",
        "<eos>",
        "(Alternatively",
        "\n(Or,",
    ]

    for delim in delimiters:
        if delim in text:
            text = text.split(delim)[0].strip()

    # Strip exact leading translation prefixes e.g. "Türkçe çeviri:", "Çeviri:"
    prefix_pattern = re.compile(
        r"^(türkçe\s+çeviri|çeviri|turkish\s+translation|translation):\s*",
        re.IGNORECASE,
    )
    text = prefix_pattern.sub("", text).strip()

    # If wrapped in exact quotes, strip outer quotes
    if len(text) >= 2 and (
        (text.startswith('"') and text.endswith('"'))
        or (text.startswith("'") and text.endswith("'"))
    ):
        text = text[1:-1].strip()

    return text


def is_explanation_like_output(raw_text: str, source_text: str) -> bool:
    """Detect if model output is chatbot chatter or an explanatory dictionary entry."""
    clean = raw_text.strip()

    # Multi-bullet points indicate dictionary options
    bullet_count = len(re.findall(r"^\s*[\*\-\u2022\d+\.]\s+", clean, re.MULTILINE))
    if bullet_count >= 2:
        return True

    # Chatbot conversational phrases & explanation keywords
    chatbot_and_explanation_keywords = [
        "as a large language model",
        "can you tell me more",
        "is there anything else i can help",
        "i exist only as computer code",
        "i don't have a physical body",
        "here are a few options",
        "the most accurate translation",
        "depends slightly on the context",
        "if you need any help",
        "let me know if you need",
        "what are you good at",
        "knowing more about your",
        "ifadesinin anlamı",
        "anlamına gelebilir",
        "bağlama göre",
        "olası anlamları",
        "bağlamı bilmek önemlidir",
        "örnekler:",
        "filminin türkçe başlığı",
        "birebir çevirisi",
        "orijinal ingilizce",
        "anlamına gelir",
        "şu şekilde çevrilebilir",
    ]
    lower_clean = clean.lower()
    if any(kw in lower_clean for kw in chatbot_and_explanation_keywords):
        return True

    # Tiered length-ratio checks
    src_len = len(source_text)
    clean_len = len(clean)

    # Very short source (<= 20 chars): output > 120 chars is suspicious
    if src_len <= 20 and clean_len > 120:
        return True

    # Medium/long source: output > 4x source length and > 300 chars is suspicious
    if src_len > 20 and clean_len > max(300, src_len * 4):
        return True

    return False


class TranslateGemmaGGUFTranslationProvider(TranslationProvider):
    """Production TranslationProvider using TranslateGemma 12B GGUF llama-server backend via /completion."""

    def __init__(
        self,
        model_path: str = DEFAULT_GEMMA_MODEL_PATH,
        executable_path: str = DEFAULT_LLAMA_EXE_PATH,
        server_url: str = DEFAULT_TRANSLATEGEMMA_SERVER_URL,
        managed: bool = True,
        max_context_length: int = 2048,
        gpu_layers: int = 99,
        micro_batch_size: int = DEFAULT_MICRO_BATCH_SIZE,
        **kwargs,
    ) -> None:
        if not MIN_MICRO_BATCH_SIZE <= micro_batch_size <= DEFAULT_MICRO_BATCH_SIZE:
            raise ValueError("micro_batch_size must be 3 or 4")
        self.model_path = model_path
        self.executable_path = executable_path
        self.server_url = server_url.rstrip("/")
        self.managed = managed
        self.max_context_length = max_context_length
        self.gpu_layers = gpu_layers
        self.micro_batch_size = micro_batch_size

        self._process: subprocess.Popen | None = None
        self._owned_process: bool = False
        self.metrics = TranslateGemmaGGUFMetrics(gpu_offload=f"configured: ngl={gpu_layers}")
        self.micro_batch_history: list[dict[str, Any]] = []
        self._micro_batch_sequence = 0

    @property
    def name(self) -> str:
        return "TranslateGemma-12B-GGUF-Translation"

    @property
    def is_loaded(self) -> bool:
        return self._check_health()

    def _check_health(self) -> bool:
        """Verify llama-server health AND check model identity from /props."""
        try:
            req = urllib.request.Request(f"{self.server_url}/health", method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status != 200:
                    return False

            props_req = urllib.request.Request(f"{self.server_url}/props", method="GET")
            with urllib.request.urlopen(props_req, timeout=3) as props_resp:
                props_data = json.loads(props_resp.read().decode("utf-8"))
                loaded_model = str(props_data.get("model_path") or props_data.get("model_alias") or "")
                expected_basename = os.path.basename(self.model_path).lower()
                if loaded_model and expected_basename not in loaded_model.lower():
                    logger.warning(
                        f"Server on {self.server_url} has model '{loaded_model}', expected '{expected_basename}'. Identity check failed."
                    )
                    return False
            return True
        except Exception:
            return False

    def load(self) -> None:
        """Start managed llama-server process if not already running."""
        if self._check_health():
            logger.info(f"Connected to existing TranslateGemma llama-server at {self.server_url}")
            return

        if not os.path.exists(self.executable_path):
            raise FileNotFoundError(f"llama.exe not found at {self.executable_path}")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"TranslateGemma model file not found at {self.model_path}")

        t0 = time.perf_counter()
        port = self.server_url.split(":")[-1].split("/")[0]

        cmd = [
            self.executable_path,
            "serve",
            "-m",
            self.model_path,
            "-ngl",
            str(self.gpu_layers),
            "--host",
            "127.0.0.1",
            "--port",
            port,
            "--reasoning",
            "off",
            "-c",
            str(self.max_context_length),
        ]

        logger.info(f"Starting managed TranslateGemma llama-server: {' '.join(cmd)}")

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
                    f"TranslateGemma llama-server exited prematurely with code {self._process.returncode}"
                )
            if self._check_health():
                ready = True
                break
            time.sleep(0.5)

        if not ready:
            self.unload()
            raise RuntimeError("Timeout waiting for TranslateGemma llama-server to initialize (45s)")

        self.metrics.model_load_seconds = round(time.perf_counter() - t0, 2)
        logger.info(
            f"TranslateGemmaGGUFTranslationProvider loaded successfully, load time: {self.metrics.model_load_seconds}s"
        )

    def unload(self) -> None:
        """Terminate managed server process."""
        if self._owned_process and self._process is not None:
            logger.info("Terminating managed TranslateGemma llama-server process...")
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
            logger.info("TranslateGemmaGGUFTranslationProvider unloaded.")

    def _query_official_translation(self, prepared_text: str) -> tuple[str, int, int, float]:
        """Send raw completion request to TranslateGemma llama-server using exact rendered prompt."""
        endpoint = f"{self.server_url}/completion"
        prompt = render_translategemma_prompt(prepared_text, source_lang_code="en", target_lang_code="tr")
        payload = {
            "prompt": prompt,
            "temperature": 0.0,
            "n_predict": 256,
            "stop": ["<end_of_turn>", "<eos>", "<bos>", "<start_of_turn>"],
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

        raw_text = res_json.get("content", "").strip()
        in_toks = res_json.get("tokens_evaluated", 0)
        gen_toks = res_json.get("tokens_predicted", 0)

        return raw_text, in_toks, gen_toks, gen_time

    def _request_translation(
        self,
        prepared_text: str,
        *,
        label: str,
        retry_empty: bool,
    ) -> tuple[str, str, bool]:
        """Execute one bounded raw-completion request and record actual calls."""
        raw_text = ""
        for attempt in range(2):
            if attempt:
                self.metrics.retries += 1
            self.metrics.generation_call_count += 1
            try:
                raw_text, in_toks, gen_toks, gen_sec = self._query_official_translation(prepared_text)
            except Exception as exc:
                if attempt == 0:
                    logger.warning(f"Translation request failed for {label}: {exc}. Retrying once...")
                    continue
                logger.error(f"Translation request failed for {label} after retry: {exc}")
                return raw_text, "", True

            self.metrics.input_token_count += in_toks
            self.metrics.generated_token_count += gen_toks
            self.metrics.generation_seconds += gen_sec
            cleaned = _clean_translategemma_output(raw_text) if raw_text else ""
            if cleaned or not retry_empty:
                return raw_text, cleaned, False
            if attempt == 0:
                logger.warning(f"Translation request for {label} returned empty output. Retrying once...")

        return raw_text, "", False

    @staticmethod
    def _result_metadata(
        micro_batch_id: str | None,
        micro_batch_region_ids: list[int] | None,
    ) -> dict[str, Any]:
        return {
            "micro_batch_id": micro_batch_id,
            "micro_batch_region_ids": list(micro_batch_region_ids or []),
        }

    def _finalize_prepared_item(
        self,
        prepared: _PreparedTranslationItem,
        cleaned: str,
        raw_text: str,
        *,
        micro_batch_id: str | None = None,
        micro_batch_region_ids: list[int] | None = None,
    ) -> TranslationOutputItem:
        """Apply deterministic per-item guards and term restoration."""
        item = prepared.item
        metadata = self._result_metadata(micro_batch_id, micro_batch_region_ids)

        if contains_segment_marker(cleaned):
            return TranslationOutputItem(
                region_id=item.region_id,
                source=item.source,
                translation=None,
                raw_model_response=raw_text[:500],
                validation_warnings=["segment_marker_leak"],
                requires_review=True,
                **metadata,
            )

        if is_source_translation_wrapper_output(cleaned, prepared.prepared_text):
            return TranslationOutputItem(
                region_id=item.region_id,
                source=item.source,
                translation=None,
                raw_model_response=raw_text[:500],
                validation_warnings=["source_translation_wrapper"],
                requires_review=True,
                **metadata,
            )

        if is_explanation_like_output(cleaned, item.source.strip()):
            logger.warning(
                f"Item {item.region_id} output identified as chatbot or explanation-like. Flagging for review."
            )
            return TranslationOutputItem(
                region_id=item.region_id,
                source=item.source,
                translation=None,
                raw_model_response=cleaned[:500],
                validation_warnings=["chatbot_or_explanation_output"],
                requires_review=True,
                **metadata,
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
                **metadata,
            )

        warnings = validate_protected_terms(restored, prepared.placeholder_map)
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
            **metadata,
        )

    def _translate_single_prepared(
        self,
        prepared: _PreparedTranslationItem,
        *,
        micro_batch_id: str | None = None,
        micro_batch_region_ids: list[int] | None = None,
        fallback: bool = False,
    ) -> tuple[TranslationOutputItem, str]:
        item = prepared.item
        if fallback:
            self.metrics.single_item_fallback_calls += 1
        raw_text, cleaned, error_occurred = self._request_translation(
            prepared.prepared_text,
            label=f"item {item.region_id}",
            retry_empty=True,
        )
        metadata = self._result_metadata(micro_batch_id, micro_batch_region_ids)
        if error_occurred or not cleaned:
            warning = "translation_server_error" if error_occurred else "empty_translation"
            return (
                TranslationOutputItem(
                    region_id=item.region_id,
                    source=item.source,
                    translation=None,
                    raw_model_response=raw_text[:500] if raw_text else "[Server Error]",
                    validation_warnings=[warning],
                    requires_review=True,
                    **metadata,
                ),
                raw_text,
            )
        return (
            self._finalize_prepared_item(
                prepared,
                cleaned,
                raw_text,
                micro_batch_id=micro_batch_id,
                micro_batch_region_ids=micro_batch_region_ids,
            ),
            raw_text,
        )

    def _translate_micro_batch(
        self,
        prepared_items: list[_PreparedTranslationItem],
    ) -> tuple[list[TranslationOutputItem], list[str]]:
        """Translate one 3-4 item contextual block or safely fall it back."""
        self._micro_batch_sequence += 1
        micro_batch_id = f"micro_batch_{self._micro_batch_sequence:04d}"
        region_ids = [prepared.item.region_id for prepared in prepared_items]
        markers = [f"__WTSEG{index:04d}__" for index in range(1, len(prepared_items) + 1)]
        block = "\n".join(
            f"{marker} {prepared.prepared_text}"
            for marker, prepared in zip(markers, prepared_items)
        )

        self.metrics.micro_batch_requests += 1
        raw_text, cleaned, error_occurred = self._request_translation(
            block,
            label=micro_batch_id,
            retry_empty=False,
        )
        segments = (
            split_segmented_translation(cleaned, markers)
            if cleaned and not error_occurred
            else None
        )

        if segments is not None:
            self.metrics.micro_batch_successes += 1
            self.micro_batch_history.append(
                {"micro_batch_id": micro_batch_id, "region_ids": region_ids, "status": "SUCCESS"}
            )
            results = [
                self._finalize_prepared_item(
                    prepared,
                    segment,
                    segment,
                    micro_batch_id=micro_batch_id,
                    micro_batch_region_ids=region_ids,
                )
                for prepared, segment in zip(prepared_items, segments)
            ]
            return results, [raw_text]

        self.metrics.micro_batch_fallbacks += 1
        self.micro_batch_history.append(
            {"micro_batch_id": micro_batch_id, "region_ids": region_ids, "status": "FALLBACK"}
        )
        logger.warning(
            f"{micro_batch_id} failed strict segment validation; falling back only regions {region_ids}."
        )
        results: list[TranslationOutputItem] = []
        raw_responses = [raw_text]
        for prepared in prepared_items:
            result, single_raw = self._translate_single_prepared(
                prepared,
                micro_batch_id=micro_batch_id,
                micro_batch_region_ids=region_ids,
                fallback=True,
            )
            results.append(result)
            raw_responses.append(single_raw)
        return results, raw_responses

    def translate(self, inp: TranslationInput) -> TranslationOutput:
        """Translate adjacent ordinary items in strict contextual micro-batches."""
        if not self.is_loaded:
            self.load()

        if not inp.items:
            return TranslationOutput(inputs=inp, results=[], raw_response="", repair_model=self.name)

        from core.translation.profile_discovery import (
            contains_candidate_phrase,
            get_relevant_terms_for_item,
        )

        detected_named_terms = detect_named_terms_in_items(inp.items, inp.candidate_store)
        results: list[TranslationOutputItem] = []
        raw_responses: list[str] = []
        pending: list[_PreparedTranslationItem] = []

        def flush_pending() -> None:
            nonlocal pending
            if not pending:
                return
            if len(pending) >= MIN_MICRO_BATCH_SIZE:
                batch_results, batch_raw = self._translate_micro_batch(pending)
                results.extend(batch_results)
                raw_responses.extend(batch_raw)
            else:
                for prepared in pending:
                    result, raw_text = self._translate_single_prepared(prepared)
                    results.append(result)
                    raw_responses.append(raw_text)
            pending = []

        for item in inp.items:
            source_text = item.source.strip()
            approved_terms, _ = get_relevant_terms_for_item(
                source_text,
                inp.profile,
                inp.candidate_store,
            )
            proper_name_terms: set[str] = set()
            if inp.profile:
                proper_name_terms = {
                    key.strip().upper()
                    for key in inp.profile.known_names
                    if contains_candidate_phrase(key, source_text)
                }

            for entry in inp.glossary:
                if "->" not in entry:
                    continue
                source_term, target_term = entry.split("->", 1)
                source_term = source_term.strip()
                if contains_candidate_phrase(source_term, source_text):
                    approved_terms[source_term.upper()] = target_term.strip()

            if is_system_ui_line(source_text):
                system_translation = translate_system_ui_line(source_text, approved_terms)
                if system_translation:
                    flush_pending()
                    self.metrics.system_ui_bypass_count += 1
                    results.append(
                        TranslationOutputItem(
                            region_id=item.region_id,
                            source=item.source,
                            translation=system_translation,
                            raw_model_response="[System UI Lexicon]",
                        )
                    )
                    continue

            is_bypass, bypass_translation = is_term_only_source(
                source_text,
                approved_terms,
                detected_named_terms,
            )
            if is_bypass and bypass_translation:
                flush_pending()
                self.metrics.term_only_bypass_count += 1
                results.append(
                    TranslationOutputItem(
                        region_id=item.region_id,
                        source=item.source,
                        translation=bypass_translation,
                        raw_model_response="[Term-Only Bypass]",
                    )
                )
                continue

            prepared_text, placeholder_map = protect_source_text(
                source_text,
                approved_terms,
                detected_named_terms,
                proper_name_terms=proper_name_terms,
            )
            pending.append(
                _PreparedTranslationItem(
                    item=item,
                    prepared_text=prepared_text,
                    placeholder_map=placeholder_map,
                )
            )
            if len(pending) == self.micro_batch_size:
                flush_pending()

        flush_pending()

        if self.metrics.generation_seconds > 0:
            self.metrics.tokens_per_sec = round(
                self.metrics.generated_token_count / self.metrics.generation_seconds,
                2,
            )

        return TranslationOutput(
            inputs=inp,
            results=results,
            raw_response="\n---\n".join(raw_responses)[:2000],
            repair_model=self.name,
        )
