"""Google AI (Gemini) Free API Translation Provider for Webtoon Localizations.

Leverages Gemini Flash (gemini-2.5-flash / gemini-2.0-flash / gemini-1.5-flash)
for high-fidelity, context-aware, idiomatic Turkish comic translation.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Sequence

from providers.translation.base import (
    TranslationInput,
    TranslationItem,
    TranslationOutput,
    TranslationOutputItem,
    TranslationProvider,
)

logger = logging.getLogger(__name__)

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_ENDPOINT_TEMPLATE = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
)

WEBTOON_SYSTEM_INSTRUCTION = (
    "You are an elite Turkish Webtoon/Manga localization specialist and comic letterer. "
    "Your goal is to translate English webtoon dialogue into natural, punchy, fluent, "
    "and emotionally expressive Turkish that sounds like a published comic book.\n\n"
    "Guidelines:\n"
    "1. Translate dialogue with appropriate comic tone, maintaining character emotion, energy, and speech rhythm.\n"
    "2. Preserve slang, exclamations, insults, honorifics, and comic jargon idiomatically in Turkish (e.g. 'Damn it!' -> 'Kahretsin!', 'What the...?!' -> 'Ne...?!', 'Hey, brat!' -> 'Hey, velet!').\n"
    "3. Keep translations concise so they fit inside speech bubbles without unnecessary verbosity.\n"
    "4. Return STRICT JSON conforming to the requested schema with all dialogue IDs preserved exactly."
)


@dataclass
class GeminiMetrics:
    total_calls: int = 0
    total_items_translated: int = 0
    total_tokens_estimated: int = 0
    total_latency_sec: float = 0.0
    failed_calls: int = 0


class GeminiTranslationProvider(TranslationProvider):
    """Google Gemini AI API translation provider using direct lightweight HTTP REST."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = DEFAULT_GEMINI_MODEL,
        temperature: float = 0.3,
        max_output_tokens: int = 4096,
        timeout_sec: float = 30.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.timeout_sec = timeout_sec
        self.metrics = GeminiMetrics()
        self._is_loaded = bool(self.api_key)

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def name(self) -> str:
        return f"Gemini ({self.model_name})"

    @property
    def version(self) -> str:
        return "2.5" if "2.5" in self.model_name else "2.0"

    def load(self) -> None:
        if not self.api_key:
            self.api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
        if not self.api_key:
            raise ValueError(
                "Gemini API key is required. Set the GEMINI_API_KEY environment variable "
                "or configure it in config.yaml."
            )
        self._is_loaded = True
        logger.info("GeminiTranslationProvider loaded with model %s", self.model_name)

    def unload(self) -> None:
        self._is_loaded = False
        logger.debug("GeminiTranslationProvider unloaded")

    def translate_batch(
        self,
        texts: list[str],
        context: dict[str, Any] | None = None,
    ) -> list[str]:
        """Convenience method to translate a list of strings."""
        if not texts:
            return []
        items = [TranslationItem(region_id=idx + 1, source=t) for idx, t in enumerate(texts)]
        inp = TranslationInput(items=items)
        out = self.translate(inp)
        out_map = {item.region_id: item.translation for item in out.results}
        return [out_map.get(idx + 1, "") or "" for idx in range(len(texts))]

    def translate(self, inp: TranslationInput, chunk_size: int = 40) -> TranslationOutput:
        """Translates a batch of dialogue items using Google Gemini API."""
        if not self.is_loaded:
            self.load()

        if not inp.items:
            return TranslationOutput(inputs=inp, results=[], raw_response="", repair_model=self.name)

        results: list[TranslationOutputItem] = []
        raw_responses: list[str] = []

        # Process in manageable chunks (e.g. 40 dialogues per request)
        for i in range(0, len(inp.items), chunk_size):
            chunk = inp.items[i : i + chunk_size]
            chunk_results, raw_resp = self._translate_chunk(chunk, inp)
            results.extend(chunk_results)
            raw_responses.append(raw_resp)

        return TranslationOutput(
            inputs=inp,
            results=results,
            raw_response="\n---\n".join(raw_responses),
            repair_model=self.name,
        )

    def _translate_chunk(
        self,
        chunk: list[TranslationItem],
        parent_input: TranslationInput,
    ) -> tuple[list[TranslationOutputItem], str]:
        """Translates a single chunk of dialogues with structured JSON parsing."""
        dialogues_payload = [
            {"id": item.region_id, "english": item.source.strip()}
            for item in chunk
            if item.source and item.source.strip()
        ]

        if not dialogues_payload:
            empty_results = [
                TranslationOutputItem(
                    region_id=item.region_id,
                    source=item.source,
                    translation=item.source,
                    raw_model_response="",
                )
                for item in chunk
            ]
            return empty_results, ""

        user_prompt = (
            "Translate the following webtoon dialogue list into natural Turkish comic dialogue.\n"
            "Return a JSON object with a 'translations' array containing objects with 'id' (integer) and 'turkish' (string).\n\n"
            f"INPUT DIALOGUES:\n{json.dumps(dialogues_payload, ensure_ascii=False, indent=2)}"
        )

        request_body = {
            "system_instruction": {
                "parts": [{"text": WEBTOON_SYSTEM_INSTRUCTION}]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_prompt}],
                }
            ],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_output_tokens,
                "responseMimeType": "application/json",
            },
        }

        url = GEMINI_ENDPOINT_TEMPLATE.format(model=self.model_name, api_key=self.api_key)
        req = urllib.request.Request(
            url,
            data=json.dumps(request_body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        t_start = time.perf_counter()
        raw_response_text = ""
        parsed_translations: dict[int, str] = {}
        error_msg: str | None = None

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                response_data = json.loads(resp.read().decode("utf-8"))
                self.metrics.total_calls += 1
                self.metrics.total_latency_sec += time.perf_counter() - t_start

                candidates = response_data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        raw_response_text = parts[0].get("text", "")
                        parsed_translations = self._parse_json_translations(raw_response_text)

        except urllib.error.HTTPError as http_err:
            self.metrics.failed_calls += 1
            error_body = http_err.read().decode("utf-8", errors="replace")
            logger.error("Gemini API HTTP Error %s: %s", http_err.code, error_body)
            error_msg = f"HTTP {http_err.code}: {error_body}"
            # Fallback to secondary model if 2.5 is unavailable or quota limit
            if self.model_name != "gemini-1.5-flash" and http_err.code in (404, 400):
                logger.info("Falling back to gemini-1.5-flash...")
                self.model_name = "gemini-1.5-flash"
                return self._translate_chunk(chunk, parent_input)

        except Exception as exc:
            self.metrics.failed_calls += 1
            logger.error("Gemini API request failed: %s", exc)
            error_msg = str(exc)

        # Build output items
        output_items: list[TranslationOutputItem] = []
        for item in chunk:
            tr = parsed_translations.get(item.region_id)
            if tr:
                self.metrics.total_items_translated += 1
                output_items.append(
                    TranslationOutputItem(
                        region_id=item.region_id,
                        source=item.source,
                        translation=tr.strip(),
                        raw_model_response=raw_response_text[:300],
                        requires_review=False,
                    )
                )
            else:
                # Fallback: keep source or mark review
                output_items.append(
                    TranslationOutputItem(
                        region_id=item.region_id,
                        source=item.source,
                        translation=None,
                        raw_model_response=raw_response_text[:300] if raw_response_text else (error_msg or "Failed"),
                        validation_warnings=["gemini_translation_missing"] if not error_msg else [error_msg],
                        requires_review=True,
                    )
                )

        return output_items, raw_response_text

    @staticmethod
    def _parse_json_translations(text: str) -> dict[int, str]:
        """Extracts {id: turkish_translation} from model JSON response."""
        result: dict[int, str] = {}
        if not text or not text.strip():
            return result

        # Direct JSON load
        try:
            data = json.loads(text)
            items = []
            if isinstance(data, dict):
                items = data.get("translations", data.get("results", data.get("items", [])))
                if not items and "id" in data and "turkish" in data:
                    items = [data]
            elif isinstance(data, list):
                items = data

            for elem in items:
                if isinstance(elem, dict) and "id" in elem:
                    item_id = int(elem["id"])
                    tr_text = elem.get("turkish", elem.get("translation", elem.get("target", "")))
                    if tr_text:
                        result[item_id] = str(tr_text).strip()
        except Exception:
            # Fallback regex extraction in case of markdown code fences or malformed JSON
            json_match = re.search(r"\{.*\}|\[.*\]", text, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group(0))
                    items = data.get("translations", data) if isinstance(data, dict) else data
                    for elem in items:
                        if isinstance(elem, dict) and "id" in elem:
                            result[int(elem["id"])] = str(elem.get("turkish", "")).strip()
                except Exception:
                    pass

        return result
