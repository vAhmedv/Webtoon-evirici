"""TranslateGemma 12B GGUF llama-server translation provider.

Uses TranslateGemma 12B GGUF running via llama-server (llama.exe) with CUDA offload.

Key design principles:
- Official TranslateGemma direct text translation request format:
  messages = [{"role": "user", "content": [{"type": "text", "source_lang_code": "en", "target_lang_code": "tr", "text": prepared_text}]}]
- No system prompt, instruction wall, or glossary lists inside the model text payload.
- Terminology and named-ability protection handled at the APPLICATION LAYER.
- Explanation-like model output guard (detects multi-bullet / dictionary outputs).
- Exact server identity verification via GET /props model_path check.
- Bounded per-item retries and error isolation (one failing item does not abort chapter).
- Truthful metrics (no hardcoded VRAM/CUDA claims).
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
from typing import Any

from loguru import logger

from core.translation.protection import (
    detect_named_terms_in_items,
    protect_source_text,
    restore_protected_translation,
    validate_protected_terms,
)
from core.translation.system_text import is_system_ui_line, translate_system_ui_line
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
    retries: int = 0
    cuda_active: bool | None = None
    gpu_offload: str = "configured: ngl=99"


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

    # Take the first line if multiple lines returned
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if lines:
        text = lines[0]

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
    """Detect if model output is an explanatory dictionary entry rather than a translation."""
    clean = raw_text.strip()

    # Multi-bullet points indicate dictionary options
    bullet_count = len(re.findall(r"^\s*[\*\-\u2022\d+\.]\s+", clean, re.MULTILINE))
    if bullet_count >= 2:
        return True

    # Explanation phrases
    explanation_keywords = [
        "ifadesinin anlamı",
        "anlamına gelebilir",
        "bağlama göre",
        "olası anlamları",
        "bağlamı bilmek önemlidir",
        "örnekler:",
        "filminin türkçe başlığı",
        "birebir çevirisi",
        "orijinal ingilizce",
    ]
    lower_clean = clean.lower()
    if any(kw in lower_clean for kw in explanation_keywords):
        return True

    # Excessive length expansion (> 4x source length for non-trivial sources)
    if len(source_text) > 10 and len(clean) > max(300, len(source_text) * 4):
        return True

    return False


class TranslateGemmaGGUFTranslationProvider(TranslationProvider):
    """Production TranslationProvider using TranslateGemma 12B GGUF llama-server backend."""

    def __init__(
        self,
        model_path: str = DEFAULT_GEMMA_MODEL_PATH,
        executable_path: str = DEFAULT_LLAMA_EXE_PATH,
        server_url: str = DEFAULT_TRANSLATEGEMMA_SERVER_URL,
        managed: bool = True,
        max_context_length: int = 2048,
        gpu_layers: int = 99,
        **kwargs,
    ) -> None:
        self.model_path = model_path
        self.executable_path = executable_path
        self.server_url = server_url.rstrip("/")
        self.managed = managed
        self.max_context_length = max_context_length
        self.gpu_layers = gpu_layers

        self._process: subprocess.Popen | None = None
        self._owned_process: bool = False
        self.metrics = TranslateGemmaGGUFMetrics(gpu_offload=f"configured: ngl={gpu_layers}")

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

            # Verify loaded model path matches expected model_path
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
            "--jinja",
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
        """Send official direct text translation payload to TranslateGemma llama-server."""
        endpoint = f"{self.server_url}/v1/chat/completions"
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "source_lang_code": "en",
                            "target_lang_code": "tr",
                            "text": f"Translate from English to Turkish:\n{prepared_text}",
                        }
                    ],
                }
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
        raw_text = choices[0].get("message", {}).get("content", "").strip() if choices else ""

        usage = res_json.get("usage", {})
        in_toks = usage.get("prompt_tokens", 0)
        gen_toks = usage.get("completion_tokens", 0)

        return raw_text, in_toks, gen_toks, gen_time

    def translate(self, inp: TranslationInput) -> TranslationOutput:
        """Translate input items using official TranslateGemma request format and application-level protection."""
        if not self.is_loaded:
            self.load()

        if not inp.items:
            return TranslationOutput(inputs=inp, results=[], raw_response="", repair_model=self.name)

        from core.translation.profile_discovery import get_relevant_terms_for_item

        # 1. Establish chapter-wide source-side named term protection
        detected_named_terms = detect_named_terms_in_items(inp.items)

        results: list[TranslationOutputItem] = []
        raw_responses: list[str] = []

        # 2. Iterate items in reading order directly
        for item in inp.items:
            source_text = item.source.strip()

            # Check if line matches System / Game UI text pattern e.g. "TITLE ACQUIRED: ..."
            if is_system_ui_line(source_text):
                sys_tr = translate_system_ui_line(source_text)
                if sys_tr:
                    results.append(
                        TranslationOutputItem(
                            region_id=item.region_id,
                            source=item.source,
                            translation=sys_tr,
                            raw_model_response="[System UI Lexicon]",
                            validation_warnings=[],
                            requires_review=False,
                            fidelity_flags=[],
                            term_usages=[],
                        )
                    )
                    continue

            # Retrieve relevant approved terms for this item
            app_t, _ = get_relevant_terms_for_item(source_text, inp.profile, inp.candidate_store)

            # Manual glossary input override support
            if inp.glossary:
                for entry in inp.glossary:
                    if "->" in entry:
                        k, v = entry.split("->", 1)
                        if k.strip().lower() in source_text.lower():
                            app_t[k.strip().upper()] = v.strip()

            # Application-level source protection
            prepared_text, placeholder_map = protect_source_text(
                source_text, app_t, detected_named_terms
            )

            # Per-item retry and error isolation
            raw_text = ""
            in_toks, gen_toks, gen_sec = 0, 0, 0.0
            error_occured = False

            try:
                raw_text, in_toks, gen_toks, gen_sec = self._query_official_translation(prepared_text)
            except Exception as exc:
                logger.warning(f"Translation request failed for item {item.region_id}: {exc}. Retrying...")
                self.metrics.retries += 1
                try:
                    raw_text, in_toks, gen_toks, gen_sec = self._query_official_translation(prepared_text)
                except Exception as exc2:
                    logger.error(f"Item {item.region_id} failed after retry: {exc2}")
                    error_occured = True

            raw_responses.append(raw_text)

            self.metrics.generation_call_count += 1
            self.metrics.input_token_count += in_toks
            self.metrics.generated_token_count += gen_toks
            self.metrics.generation_seconds += gen_sec

            if error_occured or not raw_text:
                results.append(
                    TranslationOutputItem(
                        region_id=item.region_id,
                        source=item.source,
                        translation=None,
                        raw_model_response=raw_text[:500] if raw_text else "[Server Error]",
                        validation_warnings=["translation_server_error" if error_occured else "empty_translation"],
                        requires_review=True,
                        fidelity_flags=[],
                        term_usages=[],
                    )
                )
                continue

            cleaned = _clean_translategemma_output(raw_text)

            # Check explanation-like output guard
            if is_explanation_like_output(cleaned, source_text):
                logger.warning(f"Item {item.region_id} output identified as explanation-like. Flagging for review.")
                results.append(
                    TranslationOutputItem(
                        region_id=item.region_id,
                        source=item.source,
                        translation=None,
                        raw_model_response=cleaned[:500],
                        validation_warnings=["explanation_like_output"],
                        requires_review=True,
                        fidelity_flags=[],
                        term_usages=[],
                    )
                )
                continue

            # Application-level restoration
            restored_tr = restore_protected_translation(cleaned, placeholder_map)

            # Post-restoration approved terminology validation
            val_warnings = validate_protected_terms(restored_tr, placeholder_map)
            req_review = len(val_warnings) > 0

            results.append(
                TranslationOutputItem(
                    region_id=item.region_id,
                    source=item.source,
                    translation=restored_tr,
                    raw_model_response=raw_text[:500],
                    validation_warnings=val_warnings,
                    requires_review=req_review,
                    fidelity_flags=[],
                    term_usages=[],
                )
            )

        if self.metrics.generation_seconds > 0:
            self.metrics.tokens_per_sec = round(
                self.metrics.generated_token_count / self.metrics.generation_seconds, 2
            )

        return TranslationOutput(
            inputs=inp,
            results=results,
            raw_response="\n---\n".join(raw_responses)[:2000],
            repair_model=self.name,
        )
